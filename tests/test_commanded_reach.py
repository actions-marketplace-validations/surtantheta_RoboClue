"""Recorded command against reached position. The saturation case reproduces BlackWhite123/fold_cloth's wrist_roll
(.validation/i7-calibration: the follower stopped at 154.8 while the leader commanded up to 176.1)."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.commanded_reach import check_commanded_reach

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
RNG = np.random.default_rng(7)


def _targets(episodes=12, holds=6):
    """Per episode, a sequence of held joint targets; wrist_roll often goes to 170 (beyond a 150 stop)."""
    out = []
    for _ in range(episodes):
        seq = RNG.uniform(-40, 140, size=(holds, len(JOINTS)))
        seq[::2, 4] = 170.0
        seq[1::3, 4] = -60.0
        out.append(seq)
    return out


def _dataset(root, reach=lambda a: a, names=JOINTS, move=8, hold=25):
    rows = []
    for e, seq in enumerate(_targets()):
        command = seq[0]
        for target in seq:
            for k in range(move + hold):
                command = command + (target - command) * min(1.0, (k + 1) / move) if k < move else target
                rows.append((e, command.copy()))
    action = np.array([r[1] for r in rows])
    state = reach(action.copy()) + RNG.normal(0, 0.02, size=action.shape)
    episode = np.array([r[0] for r in rows])
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [len(names)], "names": names}
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30, "robot_type": "so_follower",
        "features": {"action": feature, "observation.state": feature}}))
    pq.write_table(pa.table({"action": action.tolist(), "observation.state": state.tolist(),
                             "episode_index": episode.tolist()}), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def _codes(report, status):
    return [(f["code"], f["evidence"][0]["value"]) for f in report["findings"] if f["status"] == status]


def _clip(column, low=None, high=None):
    def reach(a):
        a[:, column] = np.clip(a[:, column], low, high)
        return a
    return reach


def test_a_robot_that_reaches_its_commands_passes(tmp_path):
    report = check_commanded_reach(_dataset(tmp_path / "ds"))
    assert report["status"] == "pass", _codes(report, "warning")
    assert report["joints"]["gripper.pos"]["judged"] is False
    assert all(report["joints"][j]["still_frames"] > 1000 for j in JOINTS[:5])


def test_follower_stopping_at_a_limit_is_reported(tmp_path):
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=_clip(4, high=150.0)))
    finding = next(f for f in report["findings"] if f["code"] == "reach.held_at_an_end")
    assert finding["evidence"][0]["value"] == "wrist_roll.pos" and finding["status"] == "warning"
    # The held end carries the simulated sensor noise (state + N(0, 0.02)), so it reads about 150.0x.
    assert "sat at its highest values (about 150.0" in finding["message"] and "170.00 at the 99th percentile" in finding["message"]
    assert "lerobot_record.py:325-327" in finding["message"] and "never executed" not in finding["message"]
    assert report["status"] == "warning"
    assert [v for c, v in _codes(report, "warning")] == ["wrist_roll.pos"]


def test_lower_limit_saturation(tmp_path):
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=_clip(4, low=-45.0)))
    finding = next(f for f in report["findings"] if f["code"] == "reach.held_at_an_end")
    assert "sat at its lowest values" in finding["message"] and "-60.00 at the 99th percentile" in finding["message"]


def test_steady_offset_across_the_range(tmp_path):
    def offset(a):
        a[:, 2] -= 12.0
        return a
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=offset))
    assert ("reach.steady_offset", "elbow_flex.pos") in _codes(report, "warning")
    assert "reach.held_at_an_end" not in {c for c, _ in _codes(report, "warning")}


def test_a_blocked_gripper_is_not_judged(tmp_path):
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=_clip(5, low=20.0)))
    assert report["status"] == "pass" and report["joints"]["gripper.pos"]["judged"] is False


def test_joint_resting_at_its_extreme_without_a_gap_passes(tmp_path):
    # Commands and positions both stop at 150: the joint often sits at its highest reading, but reaches it.
    ds = _dataset(tmp_path / "ds")
    path = ds / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    columns = {}
    for name in ("action", "observation.state"):
        values = np.asarray(table.column(name).to_pylist())
        values[:, 4] = np.clip(values[:, 4], None, 150.0)
        columns[name] = values.tolist()
    pq.write_table(pa.table({**columns, "episode_index": table.column("episode_index")}), path)
    assert check_commanded_reach(ds)["status"] == "pass"


def test_other_quantities_are_not_compared(tmp_path):
    report = check_commanded_reach(_dataset(tmp_path / "ds", names=[f"motor_{i}" for i in range(6)]))
    assert report["status"] == "unknown" and _codes(report, "unknown")[0][0] == "reach.not_applicable"


def test_without_still_frames_nothing_is_decided(tmp_path):
    ds = _dataset(tmp_path / "ds")
    t = np.arange(4000)
    moving = np.stack([60 * np.sin(t / (9 + j)) + 40 for j in range(len(JOINTS))], axis=1)
    pq.write_table(pa.table({"action": moving.tolist(), "observation.state": moving.tolist(),
                             "episode_index": (t // 400).tolist()}), ds / "data" / "chunk-000" / "file-000.parquet")
    report = check_commanded_reach(ds)
    assert report["status"] == "unknown" and not _codes(report, "pass")
    assert {c for c, _ in _codes(report, "unknown")} == {"reach.joint_undecidable"}


def test_bounds_are_explicit(tmp_path):
    ds = _dataset(tmp_path / "ds")
    with pytest.raises(InputError, match="max_rows"):
        check_commanded_reach(ds, max_rows=10)
    with pytest.raises(InputError, match="max_files"):
        check_commanded_reach(ds, max_files=0)


def _rewrite(ds, edit):
    path = ds / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    a, s = np.asarray(table.column("action").to_pylist()), np.asarray(table.column("observation.state").to_pylist())
    edit(a, s)
    pq.write_table(pa.table({"action": a.tolist(), "observation.state": s.tolist(),
                             "episode_index": table.column("episode_index")}), path)


def test_held_end_is_found_despite_a_mild_offset_elsewhere(tmp_path):
    # Validator 2a: a 3% tracking offset used to suppress the held-end finding.
    def reach(a):
        a = _clip(4, high=150.0)(a)
        a[:, 4] -= 6.0
        return a
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=reach))
    assert ("reach.held_at_an_end", "wrist_roll.pos") in _codes(report, "warning")


def test_one_jittery_joint_does_not_hide_the_others(tmp_path):
    # Validator: an idle joint with 1-tick jitter used to remove every still frame for every joint.
    ds = _dataset(tmp_path / "ds", reach=_clip(4, high=150.0))
    _rewrite(ds, lambda a, s: s.__setitem__((slice(None), 0), s[:, 0] + np.where(np.arange(len(s)) % 2, 0.6, -0.6)))
    assert ("reach.held_at_an_end", "wrist_roll.pos") in _codes(check_commanded_reach(ds), "warning")


def test_nan_and_dead_joints_are_unknown_not_pass(tmp_path):
    ds = _dataset(tmp_path / "ds")
    def edit(a, s):
        a[10, 1] = np.nan
        s[:, 3] = 7.0
    _rewrite(ds, edit)
    report = check_commanded_reach(ds)
    assert ("reach.nonfinite_values", "shoulder_lift.pos") in _codes(report, "unknown")
    assert ("reach.joint_undecidable", "wrist_flex.pos") in _codes(report, "unknown")
    assert report["status"] == "unknown" and not _codes(report, "warning")


def test_relative_actions_under_position_names_are_not_compared(tmp_path):
    ds = _dataset(tmp_path / "ds")
    _rewrite(ds, lambda a, s: a.__setitem__((slice(None), 2), np.diff(s[:, 2], prepend=s[0, 2])))
    assert ("reach.not_comparable", "elbow_flex.pos") in _codes(check_commanded_reach(ds), "unknown")


def test_hand_joints_are_not_judged(tmp_path):
    names = JOINTS[:5] + ["thumb_cmc.pos"]
    report = check_commanded_reach(_dataset(tmp_path / "ds", names=names, reach=_clip(5, low=20.0)))
    assert report["joints"]["thumb_cmc.pos"]["judged"] is False and not _codes(report, "warning")


def test_rows_beyond_the_declared_frames_are_flagged_unknown(tmp_path):
    ds = _dataset(tmp_path / "ds")
    info = json.loads((ds / "meta" / "info.json").read_text())
    info["total_frames"] = 10
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    assert "reach.rows_disagree_with_metadata" in {c for c, _ in _codes(check_commanded_reach(ds), "unknown")}


def test_one_outlier_command_does_not_change_the_verdict(tmp_path):
    ds = _dataset(tmp_path / "ds")
    _rewrite(ds, lambda a, s: a.__setitem__((5, 2), -212.0))
    assert check_commanded_reach(ds)["status"] == "pass"


def test_missing_columns_raise_input_error(tmp_path):
    ds = _dataset(tmp_path / "ds")
    path = ds / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.drop(["episode_index"]), path)
    with pytest.raises(InputError, match="episode_index"):
        check_commanded_reach(ds)


def test_low_frame_rate_keeps_the_same_verdict(tmp_path):
    ds = _dataset(tmp_path / "ds", reach=_clip(4, high=150.0))
    info = json.loads((ds / "meta" / "info.json").read_text())
    info["fps"] = 10
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    assert ("reach.held_at_an_end", "wrist_roll.pos") in _codes(check_commanded_reach(ds), "warning")


def test_evidence_hashes_the_data_files(tmp_path):
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=_clip(4, high=150.0)))
    finding = next(f for f in report["findings"] if f["code"] == "reach.held_at_an_end")
    assert any(e["path"].startswith("data/") and len(e["sha256"]) == 64 for e in finding["evidence"])


def test_held_end_with_an_offset_the_other_way_is_found(tmp_path):
    # Validator recheck 2a': the follower reading slightly above its command elsewhere must not hide a held end.
    def reach(a):
        a = _clip(4, high=150.0)(a)
        a[:, 4] += 6.0
        return a
    assert ("reach.held_at_an_end", "wrist_roll.pos") in _codes(check_commanded_reach(_dataset(tmp_path / "ds", reach=reach)), "warning")


def test_hope_jr_finger_joints_are_not_judged(tmp_path):
    names = JOINTS[:3] + ["index_mcp.pos", "middle_pip.pos", "ring_dip.pos"]
    report = check_commanded_reach(_dataset(tmp_path / "ds", names=names, reach=_clip(4, high=150.0)))
    assert all(report["joints"][n]["judged"] is False for n in names[3:]) and not _codes(report, "warning")


def test_joint_held_for_most_of_its_range_is_held_not_incomparable(tmp_path):
    # Validator recheck R1: a heavily capped joint has low correlation but is a held end, not a delta series.
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=_clip(4, high=-30.0)))
    assert ("reach.held_at_an_end", "wrist_roll.pos") in _codes(report, "warning")


def test_mid_range_stall_is_not_reported_as_agreement(tmp_path):
    # Validator recheck R2: a joint frozen mid-range in half of each episode must not pass.
    ds = _dataset(tmp_path / "ds")
    def freeze(a, s):
        e = np.repeat(np.arange(12), len(s) // 12 + 1)[:len(s)]
        for k in range(12):
            idx = np.flatnonzero(e == k)
            s[idx[: len(idx) // 2], 2] = 50.0
    _rewrite(ds, freeze)
    report = check_commanded_reach(ds)
    flagged = {c for c, v in _codes(report, "warning") + _codes(report, "unknown") if v == "elbow_flex.pos"}
    assert flagged & {"reach.mixed_residual", "reach.held_at_an_end", "reach.steady_offset"}
    assert ("reach.joint", "elbow_flex.pos") not in _codes(report, "pass")


def test_empty_data_raises_input_error(tmp_path):
    ds = _dataset(tmp_path / "ds")
    path = ds / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(path).slice(0, 0), path)
    with pytest.raises(InputError, match="no rows"):
        check_commanded_reach(ds)


def _radian_dataset(root, reach):
    """A yam-like arm recorded in radians (span about 0.8): the degree-unit floors must not apply."""
    ds = _dataset(root, reach=reach)
    _rewrite(ds, lambda a, s: (a.__setitem__(slice(None), np.radians(a) * 0.25), s.__setitem__(slice(None), np.radians(s) * 0.25)))
    return ds


def test_radian_units_do_not_inherit_degree_floors(tmp_path):
    # Final recheck N1: an 8% steady offset on a radian arm was called a held end at the median of all frames.
    def offset(a):
        a[:, 2] -= 0.08 * 180.0
        return a
    report = check_commanded_reach(_radian_dataset(tmp_path / "ds", offset))
    assert ("reach.steady_offset", "elbow_flex.pos") in _codes(report, "warning")
    assert ("reach.held_at_an_end", "elbow_flex.pos") not in _codes(report, "warning")
    assert check_commanded_reach(_radian_dataset(tmp_path / "ok", lambda a: a))["status"] == "pass"


def test_names_containing_gripper_are_not_judged(tmp_path):
    names = JOINTS[:5] + ["rightGripper.pos"]
    report = check_commanded_reach(_dataset(tmp_path / "ds", names=names, reach=_clip(5, low=20.0)))
    assert report["joints"]["rightGripper.pos"]["judged"] is False


def test_rest_pose_agreement_with_an_offset_elsewhere_is_a_steady_offset(tmp_path):
    # Fix recheck: arms agree at the bottom rest pose, the follower reads 8% lower elsewhere. Not a held end.
    def reach(a):
        a = a.copy()
        a[:, 4] = np.where(a[:, 4] <= -59.0, a[:, 4], a[:, 4] - 0.08 * 230.0)
        return a
    report = check_commanded_reach(_dataset(tmp_path / "ds", reach=reach))
    assert ("reach.held_at_an_end", "wrist_roll.pos") not in _codes(report, "warning")
    assert ("reach.steady_offset", "wrist_roll.pos") in _codes(report, "warning")


def test_degree_step_signature_keeps_motor_floors_on_small_ranges(tmp_path):
    ds = _dataset(tmp_path / "ds")
    def on_steps(a, s):
        for arr in (a, s):
            arr[:, 1] = np.round((arr[:, 1] * 0.04) * 4095 / 360 * 2) / 2 * 360 / 4095  # a 7-degree joint on half steps
    _rewrite(ds, on_steps)
    report = check_commanded_reach(ds)
    assert report["joints"]["shoulder_lift.pos"]["units_basis"] == "Feetech degree steps"
