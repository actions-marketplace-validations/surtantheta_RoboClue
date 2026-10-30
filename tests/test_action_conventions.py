"""I12 action-convention fingerprint: synthetic datasets with known conventions and declared consumers."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from policy_doctor.action_conventions import _declared, _lerobot_mask, action_conventions  # noqa: E402
from policy_doctor.artifacts import InputError  # noqa: E402
from policy_doctor.openpi_interface import MANIFEST  # noqa: E402

FPS = 30
OPENPI_FIXTURES = Path(__file__).parent / "fixtures/openpi_source_profile_215abfb"


def states(episodes=6, length=160, dims=3, seed=0, offset=0.0):
    """Smooth, aperiodic joint trajectories (one array per episode, two extra frames for lookahead)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        steps = rng.normal(0, 0.02, (length + 6, dims))
        steps = np.stack([np.convolve(steps[:, i], np.ones(5) / 5, mode="valid") for i in range(dims)], 1)
        out.append(offset + rng.uniform(-1, 1, dims) + np.cumsum(steps, axis=0)[:length + 2])
    return out


def convention(kind, s, rng):
    if kind == "absolute":  # target = state two frames ahead (leader-follower lag), plus small noise
        return s[2:] + rng.normal(0, 0.01, s[2:].shape)
    if kind == "delta":  # scaled per-step increment
        return (s[1:-1] - s[:-2]) * 20
    if kind == "velocity":
        return (s[1:-1] - s[:-2]) * FPS
    if kind == "copy":
        return s[:-2].copy()
    raise ValueError(kind)


def write_dataset(root, kind="absolute", names=None, state_names=None, dims=3, gripper=False, extra_state=0,
                  action_width=None, state_scale=1.0, groups=None, modality=None, offset=0.0, trajectories=None,
                  pairs=None):
    rng = np.random.default_rng(1)
    actions, observed, episode, frame = [], [], [], []
    for e, s in enumerate(pairs or (trajectories if trajectories is not None else states(dims=dims, offset=offset))):
        if pairs:  # (action, state) per episode, already generated
            a, st = s
        else:
            a = convention(kind, s, rng) if isinstance(kind, str) else np.stack(
                [convention(k, s[:, [i]], rng)[:, 0] for i, k in enumerate(kind)], 1)
            st = s[:-2] * state_scale
        if gripper:
            command = np.where(np.sin(np.arange(len(st)) / 15.0) > 0, 1.0, -1.0)
            width = np.convolve(-command, np.ones(5) / 5, mode="same") * 0.02 + 0.04
            a, st = np.column_stack([a, command]), np.column_stack([st, width])
        if extra_state:
            st = np.column_stack([st, rng.normal(0, 1, (len(st), extra_state))])
        if action_width is not None:
            a = a[:, :action_width]
        actions.append(a)
        observed.append(st)
        episode += [e] * len(a)
        frame += list(range(len(a)))
    action, state = np.concatenate(actions), np.concatenate(observed)
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {"action": {"dtype": "float32", "shape": [action.shape[1]], "names": names},
                "observation.state": {"dtype": "float32", "shape": [state.shape[1]], "names": state_names},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None}}
    columns = {"action": pa.array(action.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
               "observation.state": pa.array(state.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
               "timestamp": pa.array(np.array(frame, dtype=np.float32) / FPS),
               "frame_index": pa.array(frame, type=pa.int64()), "episode_index": pa.array(episode, type=pa.int64())}
    for key, (values, group_names) in (groups or {}).items():
        features[key] = {"dtype": "float32", "shape": [values.shape[1]], "names": group_names}
        columns[key] = pa.array(values.astype(np.float32).tolist(), type=pa.list_(pa.float32()))
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": FPS, "robot_type": "test_arm",
                                                     "features": features}))
    if modality is not None:
        (root / "meta/modality.json").write_text(json.dumps(modality))
    pq.write_table(pa.table(columns), root / "data/chunk-000/file-000.parquet")
    return root, action, state, np.array(episode)


TICK = 2 * np.pi / 4096  # a 4096-count encoder


def operator(kind, n, rng, amp=1.0):
    """I12 validator motions (synth.py): 'low' = a joint the operator barely moves, 'reach' = minimum-jerk reaches."""
    if kind == "low":
        return 0.3 + amp * np.cumsum(np.convolve(rng.normal(0, 0.6 / FPS, n + 4), np.ones(5) / 5, mode="valid"))[:n]
    out, x = [], rng.uniform(-1, 1)
    while len(out) < n:
        y, u = rng.uniform(-1, 1), np.linspace(0, 1, int(rng.uniform(1.0, 3.0) * FPS))
        out += list(x + (y - x) * (10 * u**3 - 15 * u**4 + 6 * u**5)) + [y] * int(rng.uniform(0.3, 1.0) * FPS)
        x = y
    return np.array(out[:n])


def follower_pairs(columns, episodes=8, length=300, seed=0):
    """Absolute targets (leader + jitter, optionally encoder-quantised) tracked by a first-order follower with time
    constant tau (I12 validator synth.py/synth2.py). columns: (motion, amp, tau, jitter, quant, deadband)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        a, s = [], []
        for kind, amp, tau, jitter, quant, deadband in columns:
            target = operator(kind, length, rng, amp) + rng.normal(0, jitter, length)
            target = np.round(target / quant) * quant if quant else target
            gain, follow = 1 - np.exp(-1 / (FPS * tau)), np.empty(length)
            follow[0] = target[0]
            for i in range(length - 1):
                error = target[i] - follow[i]
                follow[i + 1] = follow[i] + gain * (0.0 if abs(error) <= deadband else error)
            a.append(target)
            s.append(np.round(follow / quant) * quant if quant else follow)
        out.append((np.column_stack(a), np.column_stack(s)))
    return out


def coords(report, key="action"):
    group = next(g for g in report["fingerprint"]["groups"] if g["action_key"] == key)
    return group["coordinates"]


def codes(report):
    return {(f["code"], f["status"]) for f in report["findings"]}


def write_checkpoint(root, names, *, relative=True, exclude=("gripper",), stats=None, chunk=10, state_dim=None):
    root.mkdir()
    dim = len(names)
    features = {"observation.state": {"type": "STATE", "shape": [state_dim or dim]},
                "action": {"type": "ACTION", "shape": [dim]}}
    mapping = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    (root / "config.json").write_text(json.dumps({
        "type": "pi05", "input_features": {"observation.state": features["observation.state"]},
        "output_features": {"action": features["action"]}, "normalization_mapping": mapping, "chunk_size": chunk,
        "n_action_steps": chunk, "use_relative_actions": relative}))
    config = {"features": features, "norm_map": mapping, "eps": 1e-8}
    pre = [{"registry_name": "normalizer_processor", "config": config, "state_file": "pre.safetensors"}]
    post = [{"registry_name": "unnormalizer_processor", "config": config, "state_file": "post.safetensors"}]
    if relative:
        pre.insert(0, {"registry_name": "relative_actions_processor",
                       "config": {"enabled": True, "exclude_joints": list(exclude), "action_names": names}})
        post.append({"registry_name": "absolute_actions_processor", "config": {"enabled": True}})
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": pre}))
    (root / "policy_postprocessor.json").write_text(json.dumps({"steps": post}))
    stats = stats or {"action.mean": np.zeros(dim), "action.std": np.ones(dim)}
    stats = {**stats, "observation.state.mean": np.zeros(state_dim or dim), "observation.state.std": np.ones(state_dim or dim)}
    save_file({k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}, root / "pre.safetensors")
    save_file({k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}, root / "post.safetensors")
    return root


# ------------------------------------------------------------------------------------------- fingerprint


def test_absolute_targets_are_recognised(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute", names=["j0.pos", "j1.pos", "j2.pos"])
    report = action_conventions(root)
    assert [c["convention"] for c in coords(report)] == ["absolute"] * 3
    assert all(c["statistics"]["level_lag"] == 2 for c in coords(report))
    assert ("convention.fingerprint", "pass") in codes(report)
    assert report["status"] == "pass"
    assert report["schema_version"] == 1 and report["kind"] == "action_conventions"
    limits = " ".join(report["limitations"])
    assert "NaN" in limits and "+-15% scale tolerance" in limits and "escapes the follower check" in limits
    assert set(report) >= {"status", "counts", "coverage", "fingerprint", "findings", "scope"}
    for finding in report["findings"]:
        assert set(finding) == {"code", "status", "message", "remedy", "evidence"}
        assert all(set(e) >= {"path", "sha256", "pointer", "value"} for e in finding["evidence"])


@pytest.mark.parametrize("kind", ["delta", "velocity"])
def test_incremental_commands_are_recognised(tmp_path, kind):
    root, *_ = write_dataset(tmp_path / "ds", kind)
    report = action_conventions(root)
    found = coords(report)
    assert [c["convention"] for c in found] == ["incremental"] * 3
    scale = 20 if kind == "delta" else FPS
    assert all(abs(c["statistics"]["action_per_state_step"] - scale) / scale < 0.05 for c in found)
    assert all(abs(c["statistics"].get("follower_ratio", 0.0)) < 0.5 for c in found)  # the state does not pull back


def test_periodic_increment_matching_a_lagged_level_is_not_called_absolute(tmp_path):
    # 0.6 Hz sinusoid: a delta scaled by fps/omega equals the state a quarter period (12.5 frames) later.
    t = np.arange(162) / FPS
    omega = 2 * np.pi * 0.6
    trajectories = [np.column_stack([np.sin(omega * t + p) for p in (0.0, 1.0, 2.0)]) for _ in range(6)]
    rng = np.random.default_rng(0)
    root, *_ = write_dataset(tmp_path / "ds", "delta", trajectories=[s + rng.normal(0, 1e-4, s.shape) for s in trajectories])
    info = json.loads((root / "meta/info.json").read_text())
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    table["action"] = [[x / 20 * FPS / omega for x in row] for row in table["action"]]
    pq.write_table(pa.table({k: pa.array(v) for k, v in table.items()}), root / "data/chunk-000/file-000.parquet")
    (root / "meta/info.json").write_text(json.dumps(info))
    found = coords(action_conventions(root))
    assert all(c["statistics"]["level_rho"] < 0.5 for c in found)
    assert [c["convention"] for c in found] == ["unknown"] * 3


@pytest.mark.parametrize("columns", [
    [("low", 0.005, 0.2, 0.005, 0.0, 0.0)] * 3,  # validator V1 A_abs_P_fps30_tau0.2_low0.005_ln0.005
    [("low", 0.005, 0.05, 4 * TICK, TICK, 0.0)] * 3,  # V3: 4096-tick encoder, 4 ticks of leader jitter
    [("low", 0.005, 0.2, 4 * TICK, TICK, TICK)] * 3,  # V3: slower follower, one-tick deadband
    [("low", 0.005, 0.1, 2 * TICK, TICK, 0.0)] * 3,  # V3: 2 ticks of jitter
], ids=["jitter", "quantised_fast", "quantised_slow_deadband", "quantised_2ticks"])
def test_held_joint_with_leader_jitter_and_lagging_follower_is_not_incremental(tmp_path, columns):
    # Mode A (I12 validator): absolute targets on a barely moving joint look incremental to rho and the rate fit alone.
    root, *_ = write_dataset(tmp_path / "ds", pairs=follower_pairs(columns))
    report = action_conventions(root, openpi_config={"delta_action_mask": [True] * 3})
    found = coords(report)
    assert "incremental" not in {c["convention"] for c in found}
    pulled = [c for c in found if "pulls the state toward the action" in c["reason"]]
    assert pulled and all(abs(c["statistics"]["follower_ratio"]) >= 0.5 for c in pulled)
    assert not any(f["code"] == "convention.relative_on_incremental" for f in report["findings"])


@pytest.mark.parametrize("scale", [-1.0, 2.0, 3.0, 57.29578])
def test_scaled_or_sign_flipped_jittery_held_target_is_not_incremental(tmp_path, scale):
    # Recheck R-A (adv.py scaled*_tau0.1): the follower ratio equals the recording scale, outside a [0.5, 2] band.
    pairs = [(scale * a, s) for a, s in follower_pairs([("low", 0.005, 0.1, 3 * TICK, TICK, 0.0)] * 3)]
    names, state_names = ["delta_a", "delta_b", "delta_c"], ["a.pos", "b.pos", "c.pos"]
    root, *_ = write_dataset(tmp_path / "ds", names=names, state_names=state_names, pairs=pairs)
    found = coords(action_conventions(root))
    assert "incremental" not in {c["convention"] for c in found}
    assert any(abs(c["statistics"]["follower_ratio"] - scale) < 0.25 * abs(scale) for c in found)
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names))
    assert report["status"] != "error" and not any(f["code"] == "convention.relative_on_incremental"
                                                   for f in report["findings"])


def test_consumer_relative_on_absolute_with_a_static_noisy_joint(tmp_path):
    # Validator V3 consumer/lerobot_relative_on_absolute_with_static_joint: SO-like arm whose wrist_roll is held still.
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    columns = [("reach", 1.0, 0.1, 2 * TICK, 0.0, 0.0)] * 4 + [("low", 0.005, 0.1, 3 * TICK, 0.0, 0.0),
                                                                ("reach", 1.0, 0.1, 0.0, 0.0, 0.0)]
    root, *_ = write_dataset(tmp_path / "ds", names=names, state_names=names, pairs=follower_pairs(columns))
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names))
    found = coords(report)
    assert [c["convention"] for c in found[:4]] == ["absolute"] * 4 and found[4]["convention"] == "unknown"
    assert report["status"] != "error"
    assert not any(f["code"] in ("convention.relative_on_incremental", "convention.declared_conflict")
                   for f in report["findings"])


def test_static_or_partnerless_coordinates_stay_unknown(tmp_path):
    root, action, state, episode = write_dataset(tmp_path / "ds", "absolute", extra_state=0)
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    table["action"] = [row + [0.5] for row in table["action"]]  # a fourth, constant coordinate without partner
    info = json.loads((root / "meta/info.json").read_text())
    info["features"]["action"]["shape"] = [4]
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({k: pa.array(v) for k, v in table.items()}), root / "data/chunk-000/file-000.parquet")
    found = coords(action_conventions(root))
    assert found[3]["convention"] == "unknown" and "partner" in found[3]["reason"]


@pytest.mark.parametrize("scale", [57.29578, -1.0, 0.8, 1.2])
def test_scaled_or_sign_flipped_targets_are_not_called_absolute(tmp_path, scale):
    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    table["action"] = [[x * scale for x in row] for row in table["action"]]
    pq.write_table(pa.table({k: pa.array(v) for k, v in table.items()}), root / "data/chunk-000/file-000.parquet")
    found = coords(action_conventions(root))
    assert [c["convention"] for c in found] == ["unknown"] * 3
    assert all("scale" in c["reason"] for c in found)


def test_action_beyond_the_state_range_is_reported_as_saturation_not_scale(tmp_path):
    # A leader that goes past the follower's joint limit: inside the range the action tracks the state (I12 validator, 014).
    trajectories = [3 * s for s in states()]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", trajectories=trajectories)
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    limit = np.quantile([row[0] for row in table["observation.state"]], 0.4)
    table["observation.state"] = [[min(row[0], limit), *row[1:]] for row in table["observation.state"]]
    pq.write_table(pa.table({k: pa.array(v) for k, v in table.items()}), root / "data/chunk-000/file-000.parquet")
    found = coords(action_conventions(root))[0]
    assert found["convention"] == "unknown"
    assert "outside the state's observed range" in found["reason"] and "reflects the saturation" in found["reason"]
    assert found["statistics"]["action_beyond_state_range"] > 0.3
    assert 0.85 <= found["statistics"]["inside_range_affine_slope"] <= 1.15


def test_rotation_vector_names_are_not_velocity_declarations(tmp_path):
    # Recheck R-B: LeRobot's ee.wx/wy/wz are an absolute rotation vector (so_follower/robot_kinematic_processor.py:150-156).
    assert _declared("ee.wx", "") is None and _declared("wz", "") is None
    assert _declared("ang_vel.wx", "") == "velocity" and _declared("base_vx", "") == "velocity"
    names, state_names = ["ee.wx", "ee.wy", "ee.wz"], ["j4.pos", "j5.pos", "j6.pos"]
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=names, state_names=state_names)
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names))
    assert [f["status"] for f in report["findings"] if f["code"] == "convention.relative_on_incremental"] == ["warning"]
    assert report["status"] == "warning"


def test_names_that_contradict_the_numbers_warn(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=["a.pos", "b.pos", "c.pos"])
    report = action_conventions(root)
    assert ("convention.declared_conflict", "warning") in codes(report)


def test_velocity_command_tracking_measured_velocity_is_not_a_conflict(tmp_path):
    rng = np.random.default_rng(3)
    root, action, state, episode = write_dataset(tmp_path / "ds", "absolute")
    velocity = np.gradient(state, axis=0)
    groups = {"action.joint_velocity": (velocity + rng.normal(0, 1e-4, velocity.shape), ["j0", "j1", "j2"]),
              "observation.state.joint_velocity": (velocity, ["j0", "j1", "j2"])}
    root2, *_ = write_dataset(tmp_path / "ds2", "absolute", groups=groups)
    report = action_conventions(root2)
    assert {c["convention"] for c in coords(report, "action.joint_velocity")} == {"absolute"}
    assert ("convention.declared_conflict", "warning") not in codes(report)


def test_named_quaternion_that_is_not_unit_norm_warns(tmp_path):
    names = ["x", "y", "z", "rx", "ry", "rz", "rw"]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", dims=7, names=names, state_names=names)
    report = action_conventions(root)
    assert ("convention.rotation_declared_conflict", "warning") in codes(report)


def test_unit_norm_quaternion_is_reported_and_consistent(tmp_path):
    rng = np.random.default_rng(4)
    root, action, state, episode = write_dataset(tmp_path / "ds", "absolute")
    quat = rng.normal(size=(len(state), 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    groups = {"action.base_rot": (quat, ["qx", "qy", "qz", "qw"]),
              "observation.state.base_rot": (quat, ["qx", "qy", "qz", "qw"])}
    root2, *_ = write_dataset(tmp_path / "ds2", "absolute", groups=groups)
    report = action_conventions(root2)
    rotation = next(g for g in report["fingerprint"]["groups"] if g["action_key"] == "action.base_rot")["rotation"]
    assert rotation[0]["declared"] == "quaternion" and rotation[0]["values"].startswith("unit-norm")
    assert ("convention.rotation_declared_conflict", "warning") not in codes(report)


def test_position_and_width_named_x_y_z_w_are_not_a_quaternion(tmp_path):
    names = ["x", "y", "z", "w"]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", gripper=True, names=names, state_names=names)
    report = action_conventions(root)
    assert next(g for g in report["fingerprint"]["groups"])["rotation"] == []
    assert not any(f["code"] == "convention.rotation_declared_conflict" for f in report["findings"])
    quat = ["quat_x", "quat_y", "quat_z", "quat_w"]
    root2, *_ = write_dataset(tmp_path / "ds2", "absolute", gripper=True, names=quat, state_names=quat)
    assert ("convention.rotation_declared_conflict", "warning") in codes(action_conventions(root2))


def test_exact_copy_of_the_state_is_reported(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "copy")
    report = action_conventions(root)
    assert all(c["statistics"].get("exact_copy_lag") == 0 for c in coords(report))
    assert ("convention.action_copies_state", "pass") in codes(report)


def test_gripper_range_and_polarity_relative_to_state(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute", gripper=True, names=["a", "b", "c", "gripper"],
                             state_names=["a", "b", "c", "gripper"])
    report = action_conventions(root)
    gripper = next(g for g in report["fingerprint"]["groups"])["gripper"][0]
    assert gripper["levels"] == [-1.0, 1.0] and gripper["polarity_vs_state"] == "inverted"


def test_bounds_and_coverage(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    report = action_conventions(root, max_rows=500)
    assert report["coverage"]["rows_read"] == 500 and report["coverage"]["complete"] is False
    assert report["coverage"]["files"][0]["sha256"] == hashlib.sha256(
        (root / "data/chunk-000/file-000.parquet").read_bytes()).hexdigest()
    with pytest.raises(InputError):
        action_conventions(root, max_rows=0)
    with pytest.raises(InputError):
        action_conventions(root, max_files=True)


def test_zero_rows_with_a_gripper_name_is_unknown_not_a_crash(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute", gripper=True, names=["a", "b", "c", "gripper"])
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    pq.write_table(table.slice(0, 0), root / "data/chunk-000/file-000.parquet")
    report = action_conventions(root)
    assert report["status"] == "unknown" and report["coverage"]["reason"] == "the data files contain no rows"


def test_non_numeric_vector_elements_are_an_input_error(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    table["action"] = [[f"v{x}" for x in row] for row in table["action"]]
    pq.write_table(pa.table({k: pa.array(v) for k, v in table.items()}), root / "data/chunk-000/file-000.parquet")
    with pytest.raises(InputError, match="element type"):
        action_conventions(root)


def test_data_symlink_escaping_the_dataset_is_refused(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    outside = tmp_path / "outside"
    (root / "data").rename(outside)
    os.symlink(outside, root / "data")
    with pytest.raises(InputError):
        action_conventions(root)


def test_missing_pyarrow_reports_unknown(tmp_path, monkeypatch):
    import builtins

    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    report = action_conventions(root)
    assert report["status"] == "unknown" and "pyarrow" in report["coverage"]["reason"]


# ------------------------------------------------------------------------------------------- LeRobot


def test_lerobot_mask_replicates_native_build_mask():
    assert _lerobot_mask({"exclude_joints": ["gripper"], "action_names": None}, 3) == [True] * 3
    assert _lerobot_mask({"exclude_joints": [], "action_names": ["a", "b"]}, 2) == [True, True]
    assert _lerobot_mask({"exclude_joints": ["GRIP"], "action_names": ["shoulder", "left_gripper.pos"]}, 3) == [True, False, True]
    assert _lerobot_mask({"exclude_joints": [""], "action_names": ["a"]}, 1) == [True]


@pytest.mark.parametrize("names,status", [
    (["a", "b", "c"], "warning"),  # inferred convention + configured transform only
    (["delta_a", "delta_b", "delta_c"], "error"),  # the dataset declares deltas: a supported contradiction
    (["a.pos", "b.pos", "c.pos"], "unknown"),  # the names declare positions, which the inference does not overrule
])
def test_lerobot_relative_on_incremental_severity_follows_declared_intent(tmp_path, names, status):
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=names)
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names))
    found = [f for f in report["findings"] if f["code"] == "convention.relative_on_incremental"]
    assert [f["status"] for f in found] == [status]
    if status != "unknown":
        assert "reparameterised, harder target" in found[0]["message"]
        assert "adds the state back at inference" in found[0]["message"] and "195-237" in found[0]["message"]
    if status == "warning":
        assert found[0]["remedy"].startswith("If the collection code confirms incremental commands")
    assert report["status"] == {"warning": "warning", "error": "error", "unknown": "warning"}[status]


def test_lerobot_relative_without_saved_inverse_says_so(tmp_path):
    names = ["a", "b", "c"]
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=names)
    cp = write_checkpoint(tmp_path / "cp", names)
    (cp / "policy_postprocessor.json").write_text(json.dumps({"steps": json.loads(
        (cp / "policy_postprocessor.json").read_text())["steps"][:1]}))
    found = next(f for f in action_conventions(root, checkpoint=cp)["findings"]
                 if f["code"] == "convention.relative_on_incremental")
    assert "saves no enabled AbsoluteActionsProcessorStep" in found["message"]


def test_lerobot_relative_on_absolute_data_passes_and_excluded_gripper_is_not_compared(tmp_path):
    names = ["a", "b", "c", "gripper"]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", names=names, gripper=True)
    cp = write_checkpoint(tmp_path / "cp", names)
    report = action_conventions(root, checkpoint=cp)
    finding = next(f for f in report["findings"] if f["code"] == "convention.relative_on_absolute")
    assert finding["status"] == "pass" and len(finding["evidence"][1]["value"]) == 3
    assert not any(f["code"] == "convention.relative_undecided" for f in report["findings"])


def test_lerobot_gripper_included_in_mask_is_undecided(tmp_path):
    names = ["a", "b", "c", "gripper"]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", names=names, gripper=True)
    cp = write_checkpoint(tmp_path / "cp", names, exclude=())
    report = action_conventions(root, checkpoint=cp)
    assert ("convention.relative_undecided", "unknown") in codes(report)


def test_lerobot_action_wider_than_state_is_an_error(tmp_path):
    names = ["a", "b", "c"]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", names=names)
    info = json.loads((root / "meta/info.json").read_text())
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    table["observation.state"] = [row[:2] for row in table["observation.state"]]
    info["features"]["observation.state"]["shape"] = [2]
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({k: pa.array(v) for k, v in table.items()}), root / "data/chunk-000/file-000.parquet")
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names, state_dim=2))
    assert ("convention.relative_width", "error") in codes(report)


def test_lerobot_checkpoint_of_other_width_is_unknown(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=["a", "b", "c"])
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", ["a", "b"]))
    assert ("convention.relative_undecided", "unknown") in codes(report)
    assert not any(f["status"] == "error" for f in report["findings"])


def test_lerobot_relative_statistics_absolute_vs_relative(tmp_path):
    names = ["a", "b", "c"]
    root, action, state, episode = write_dataset(tmp_path / "ds", "absolute", names=names, offset=5.0)
    absolute = {"action.mean": action.mean(0), "action.std": action.std(0)}
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp1", names, stats=absolute))
    assert ("convention.relative_statistics", "warning") in codes(report)
    chunk, rel = 10, []
    for t in range(len(action) - chunk + 1):
        if episode[t] == episode[t + chunk - 1]:
            rel.append(action[t:t + chunk] - state[t])
    rel = np.concatenate(rel)
    relative = {"action.mean": rel.mean(0), "action.std": rel.std(0)}
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp2", names, stats=relative))
    assert ("convention.relative_statistics", "pass") in codes(report)


def test_lerobot_per_horizon_statistics_are_unknown_not_silent(tmp_path):
    names = ["a", "b", "c"]
    root, *_ = write_dataset(tmp_path / "ds", "absolute", names=names, offset=5.0)
    stats = {"action.mean": np.zeros((10, 3)), "action.std": np.ones((10, 3))}
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names, stats=stats))
    found = next(f for f in report["findings"] if f["code"] == "convention.relative_statistics")
    assert found["status"] == "unknown" and "[10, 3]" in found["message"]


def test_lerobot_unmodeled_relative_step_is_unknown_not_passthrough(tmp_path):
    names = ["a", "b", "c"]
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=names)
    cp = write_checkpoint(tmp_path / "cp", names, relative=False)
    config = json.loads((cp / "config.json").read_text())
    (cp / "config.json").write_text(json.dumps(dict(config, use_relative_actions=True)))
    steps = json.loads((cp / "policy_preprocessor.json").read_text())["steps"]
    steps.insert(0, {"registry_name": "delta_actions_processor", "config": {"enabled": True, "exclude_joints": ["gripper"]}})
    (cp / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))
    report = action_conventions(root, checkpoint=cp)
    assert ("convention.relative_unmodeled", "unknown") in codes(report)
    assert not any(f["code"] == "convention.lerobot_passthrough" for f in report["findings"])


def test_lerobot_passthrough_reports_dataset_convention(tmp_path):
    names = ["a", "b", "c"]
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=names)
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names, relative=False))
    finding = next(f for f in report["findings"] if f["code"] == "convention.lerobot_passthrough")
    assert finding["status"] == "pass" and "incremental" in finding["message"]


# ------------------------------------------------------------------------------------------- GR00T


MODALITY = {"state": {"arm": {"start": 0, "end": 3}, "gripper": {"start": 3, "end": 4}},
            "action": {"arm": {"start": 0, "end": 3}, "gripper": {"start": 3, "end": 4}}}


def groot_config(path, reps, flag=True, tags=("new_embodiment",)):
    configs = {tag: {"state": {"delta_indices": [0], "modality_keys": ["arm", "gripper"]},
                     "action": {"delta_indices": list(range(16)), "modality_keys": ["arm", "gripper"],
                                "action_configs": [{"rep": r, "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
                                                   for r in reps]}} for tag in tags}
    path.write_text(json.dumps({"processor_class": "Gr00tN1d6Processor",
                                "processor_kwargs": {"modality_configs": configs, "use_relative_action": flag}}))
    return path


@pytest.mark.parametrize("kind,expected", [("delta", ("convention.relative_on_incremental", "warning")),
                                           ("absolute", ("convention.relative_on_absolute", "pass"))])
def test_groot_relative_group(tmp_path, kind, expected):
    root, *_ = write_dataset(tmp_path / "ds", kind, gripper=True, names=["a", "b", "c", "gripper"], modality=MODALITY)
    report = action_conventions(root, groot_config=groot_config(tmp_path / "p.json", ["RELATIVE", "ABSOLUTE"]))
    assert expected in codes(report)
    assert ("convention.groot_passthrough", "pass") in codes(report)


def test_groot_delta_is_passthrough_and_warns_on_absolute_data(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute", gripper=True, modality=MODALITY)
    report = action_conventions(root, groot_config=groot_config(tmp_path / "p.json", ["DELTA", "ABSOLUTE"]))
    assert ("convention.groot_delta_passthrough", "warning") in codes(report)


def test_groot_relative_without_flag_is_inactive(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "delta", gripper=True, modality=MODALITY)
    report = action_conventions(root, groot_config=groot_config(tmp_path / "p.json", ["RELATIVE", "ABSOLUTE"], flag=False))
    assert ("convention.groot_relative_inactive", "warning") in codes(report)
    assert not any(f["code"] == "convention.relative_on_incremental" for f in report["findings"])


def test_groot_width_mismatch_and_missing_modality(tmp_path):
    modality = json.loads(json.dumps(MODALITY))
    modality["state"]["arm"] = {"start": 0, "end": 2}
    root, *_ = write_dataset(tmp_path / "ds", "absolute", gripper=True, modality=modality)
    report = action_conventions(root, groot_config=groot_config(tmp_path / "p.json", ["RELATIVE", "ABSOLUTE"]))
    assert ("convention.relative_width", "error") in codes(report)
    root2, *_ = write_dataset(tmp_path / "ds2", "absolute", gripper=True)
    report = action_conventions(root2, groot_config=groot_config(tmp_path / "p2.json", ["RELATIVE", "ABSOLUTE"]))
    assert ("convention.groot_modality", "unknown") in codes(report)


def test_groot_ambiguous_embodiment_is_unknown(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "absolute", gripper=True, modality=MODALITY)
    cfg = groot_config(tmp_path / "p.json", ["RELATIVE", "ABSOLUTE"], tags=("a", "b"))
    assert ("convention.groot_embodiment", "unknown") in codes(action_conventions(root, groot_config=cfg))
    report = action_conventions(root, groot_config=cfg, groot_embodiment="a")
    assert ("convention.relative_on_absolute", "pass") in codes(report)


# ------------------------------------------------------------------------------------------- openpi


def test_openpi_declared_mask(tmp_path):
    root, *_ = write_dataset(tmp_path / "ds", "delta", gripper=True)
    report = action_conventions(root, openpi_config={"delta_action_mask": [True, True, True, False]})
    assert ("convention.relative_on_incremental", "warning") in codes(report)
    found = next(f for f in report["findings"] if f["code"] == "convention.relative_on_incremental")
    assert "AbsoluteActions" in found["message"] and "is not declared here" in found["message"]
    wide = action_conventions(root, openpi_config={"delta_action_mask": [True] * 5})
    assert ("convention.relative_width", "error") in codes(wide)
    with pytest.raises(InputError):
        action_conventions(root, openpi_config={"delta_action_mask": [1, 0]})


@pytest.fixture
def openpi_source(tmp_path):
    source = tmp_path / "openpi"
    for relative, identity in MANIFEST["files"].items():
        data = (OPENPI_FIXTURES / (relative + ".txt")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == identity["sha256"]
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_bytes(data)
    return source


@pytest.mark.parametrize("name,mask", [
    ("pi0_libero", [True] * 6 + [False]),
    ("pi05_libero", None),
    ("pi0_aloha", [True] * 6 + [False] + [True] * 6 + [False]),
    ("pi0_droid", None),
    ("pi05_full_droid_finetune", [True] * 7 + [False]),
])
def test_openpi_builtin_masks_resolve_from_pinned_source(tmp_path, openpi_source, name, mask):
    root, *_ = write_dataset(tmp_path / "ds", "absolute", dims=14, gripper=True)
    report = action_conventions(root, openpi_config={"config_name": name, "source_root": str(openpi_source)})
    if mask is None:
        assert ("convention.openpi_passthrough", "pass") in codes(report)
    else:
        finding = next(f for f in report["findings"] if f["code"].startswith("convention.relative_"))
        resolved = finding["evidence"][0]["value"]["make_bool_mask"]
        assert [d > 0 for d in resolved for _ in range(abs(d))] == mask


@pytest.mark.parametrize("name", ["pi0_libero", "pi0_libero_low_mem_finetune", "pi0_fast_libero",
                                  "pi0_fast_libero_low_mem_finetune"])
def test_openpi_libero_extra_delta_is_documented_and_passes(tmp_path, openpi_source, name):
    # openpi 215abfb config.py:336-337 documents DeltaActions on LIBERO's delta actions as deliberate (old Pi0 checkpoints).
    root, *_ = write_dataset(tmp_path / "ds", "delta", dims=6, gripper=True)
    report = action_conventions(root, openpi_config={"config_name": name, "source_root": str(openpi_source)})
    found = [f for f in report["findings"] if f["code"] == "convention.relative_on_incremental"]
    assert [f["status"] for f in found] == ["pass"] and found[0]["remedy"] == ""
    message = found[0]["message"]
    assert "config.py:336-337" in message and "old Pi0 checkpoints" in message
    assert "config.py:332-333" in message and "already delta actions" in message
    assert "AbsoluteActions (config.py:342)" in message
    assert "pi05_libero sets extra_delta_transform=False (config.py:749)" in message
    assert not any(f["status"] in ("error", "warning") for f in report["findings"])


@pytest.mark.parametrize("incremental", [True, False])
def test_openpi_documented_transform_covers_undecided_coordinates(tmp_path, openpi_source, incremental):
    # Recheck must-fix: LIBERO's roll/pitch/yaw (undecided by the data) must not be asked to be confirmed absolute.
    rng = np.random.default_rng(6)
    pairs = []
    for s in states(dims=6):
        a = np.column_stack([(s[1:-1, :3] - s[:-2, :3]) * 20 if incremental else rng.normal(0, 1, (len(s) - 2, 3)),
                             rng.normal(0, 1, (len(s) - 2, 3))])
        pairs.append((a, s[:-2]))
    root, *_ = write_dataset(tmp_path / "ds", gripper=True, pairs=pairs)
    report = action_conventions(root, openpi_config={"config_name": "pi0_libero", "source_root": str(openpi_source)})
    consumer = [f for f in report["findings"] if f["code"].startswith("convention.relative_")]
    assert all("Confirm from the collection code" not in f["remedy"] for f in report["findings"])
    assert all("config.py:332-333" in f["message"] and "config.py:336-337" in f["message"] for f in consumer)
    if incremental:
        assert [(f["code"], f["status"]) for f in consumer] == [("convention.relative_on_incremental", "pass")]
        assert "3 (None), 4 (None), 5 (None), whose convention the data cannot decide" in consumer[0]["message"]
    else:
        assert [(f["code"], f["status"]) for f in consumer] == [("convention.relative_undecided", "unknown")]
        assert consumer[0]["remedy"] == ""


def test_openpi_config_outside_the_manifest_is_unknown_with_its_source(tmp_path, openpi_source):
    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    report = action_conventions(root, openpi_config={"config_name": "not_a_builtin", "source_root": str(openpi_source)})
    message = next(f["message"] for f in report["findings"] if f["code"] == "convention.openpi_config")
    assert "not written out in _CONFIGS" in message
    assert "roboarena_config.get_roboarena_configs() (config.py:969)" in message
    assert "polaris_config.get_polaris_configs() (config.py:970)" in message


def test_openpi_unverifiable_source_is_unknown(tmp_path, openpi_source):
    root, *_ = write_dataset(tmp_path / "ds", "absolute")
    (openpi_source / "src/openpi/training/config.py").write_text("# drifted\n")
    report = action_conventions(root, openpi_config={"config_name": "pi0_libero", "source_root": str(openpi_source)})
    assert ("convention.openpi_config", "unknown") in codes(report)


def test_status_precedence_error_over_warning(tmp_path):
    names = ["delta_a", "b", "c"]  # one declared delta (error), two undeclared (warning)
    root, *_ = write_dataset(tmp_path / "ds", "delta", names=names)
    report = action_conventions(root, checkpoint=write_checkpoint(tmp_path / "cp", names))
    assert report["counts"]["error"] and report["counts"]["warning"] and report["status"] == "error"
