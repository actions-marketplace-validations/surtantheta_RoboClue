"""Units, frames and timebase manifest.

Values are generated with the formulas of LeRobot 146d847 src/lerobot/motors/motors_bus.py:915-942 from integer
encoder ticks; the native motor-bus ground truth runs separately against pinned LeRobot.
"""
import json
import sys

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.units_manifest import units_manifest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SO = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = {"shoulder_pan": (760, 3330), "shoulder_lift": (820, 3220), "elbow_flex": (860, 3080),
       "wrist_flex": (840, 3240), "wrist_roll": (0, 4095), "gripper": (2010, 3480)}
COINCIDENT = {m: (900, 3175) for m in SO}  # a 2275-tick range with a half-tick middle: both grids coincide


def m100(raw, lo, hi):
    return (np.clip(raw, lo, hi) - lo) / (hi - lo) * 200 - 100


def degrees(raw, lo, hi):
    return (raw - (lo + hi) / 2) * 360 / 4095


def r0100(raw, lo, hi):
    return (np.clip(raw, lo, hi) - lo) / (hi - lo) * 100


def old_degrees(raw, offset):
    return (raw + offset) / 2048 * 180


def arm(n, mode, prefix="", seed=0, cal=CAL):
    rng = np.random.default_rng(seed)
    cols = []
    for motor in SO:
        lo, hi = cal[motor]
        raw = rng.integers(lo, hi + 1, n).astype(np.float64)
        if motor == "gripper":
            cols.append(r0100(raw, lo, hi))
        elif mode == "deg":
            cols.append(degrees(raw, lo, hi))
        elif mode == "m100":
            cols.append(m100(raw, lo, hi))
        elif mode == "old":
            cols.append(old_degrees(raw - 2048, -100))
        else:
            raise ValueError(mode)
    names = [f"{prefix}{m}.pos" for m in SO] if mode != "old" else [f"main_{m}" for m in SO]
    return names, np.stack(cols, axis=1)


def write(tmp_path, features, *, fps=30, robot_type="so101_follower", data=True, stats=None, name="ds",
          period=None, episodes=None, files=1, fixed=False, stamps=None):
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    n = next(iter(features.values()))[1].shape[0]
    info_features = {key: {"dtype": "float32", "shape": [len(names)], "names": names}
                     for key, (names, _) in features.items()}
    for column, dtype in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
                          ("index", "int64"), ("task_index", "int64")):
        info_features[column] = {"dtype": dtype, "shape": [1], "names": None}
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "robot_type": robot_type,
                                                    "fps": fps, "features": info_features}))
    if stats is not None:
        (root / "meta/stats.json").write_text(json.dumps(stats))
    if not data:
        return root
    episode = np.zeros(n, dtype=np.int64) if episodes is None else episodes
    frame = np.concatenate([np.arange(np.sum(episode == e)) for e in np.unique(episode)])
    stamp = frame * (period if period is not None else 1 / fps) if stamps is None else stamps
    columns = {}
    for key, (names, values) in features.items():
        flat = pa.array(values.astype(np.float32).ravel())
        if fixed:
            columns[key] = pa.FixedSizeListArray.from_arrays(flat, len(names))
        else:
            offsets = pa.array(np.arange(0, n * len(names) + 1, len(names), dtype=np.int32))
            columns[key] = pa.ListArray.from_arrays(offsets, flat)
    columns.update(timestamp=pa.array(stamp.astype(np.float32)), frame_index=pa.array(frame),
                   episode_index=pa.array(episode), index=pa.array(np.arange(n)), task_index=pa.array(np.zeros(n, np.int64)))
    table = pa.table(columns)
    (root / "data/chunk-000").mkdir(parents=True)
    bounds = np.linspace(0, n, files + 1).astype(int)
    for i in range(files):
        pq.write_table(table.slice(bounds[i], bounds[i + 1] - bounds[i]), root / f"data/chunk-000/file-{i:03d}.parquet")
    return root


def so_dataset(tmp_path, state_mode, action_mode=None, **kw):
    state = arm(400, state_mode, seed=1)
    action = arm(400, action_mode or state_mode, seed=2)
    return write(tmp_path, {"observation.state": state, "action": action}, **kw)


def command(*flags, robot="so101_follower"):
    return " ".join(["lerobot-record", f"--robot.type={robot}", "--robot.port=/dev/ttyACM0", *flags,
                     "--dataset.repo_id=me/x", "--dataset.single_task=pick"])


def codes(report, status=None):
    return [f["code"] for f in report["findings"] if status is None or f["status"] == status]


def arm_mode(report, key, arm=""):
    return report["manifest"]["features"][key]["arms"][arm or "(single arm)"]


def test_headline_pm100_data_on_default_degrees_robot_is_an_error(tmp_path):
    root = so_dataset(tmp_path, "m100")
    report = units_manifest(root, robot_command=command())
    assert report["status"] == "error"
    mismatch = [f for f in report["findings"] if f["code"] == "units.joint_mode_mismatch"]
    assert len(mismatch) == 1 and mismatch[0]["message"].startswith("observation.state")  # the action is not compared
    assert "RANGE_M100_100" in mismatch[0]["message"] and "DEGREES" in mismatch[0]["message"]
    assert "config_so_follower.py:42" in mismatch[0]["message"]  # the default, not a flag, decided the mode
    assert mismatch[0]["remedy"].startswith("Run with --robot.use_degrees=false")
    assert "wrist_roll up to" in mismatch[0]["message"]  # the effect on commands is quantified
    assert {e["pointer"] for e in mismatch[0]["evidence"]} >= {"/robot/use_degrees"}
    assert arm_mode(report, "observation.state")["mode"] == "RANGE_M100_100"
    roll = report["manifest"]["features"]["observation.state"]["coordinates"][4]
    assert roll["evidence"]["range_m100_100_ticks"] == 4095 and roll["status"] == "decided"


def test_pm100_data_on_explicit_pm100_robot_passes_its_joints(tmp_path):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command("--robot.use_degrees=false"))
    assert codes(report, "error") == []
    assert codes(report, "pass").count("units.joint_mode_match") == 1
    assert report["status"] == "pass"
    gripper = report["manifest"]["features"]["action"]["coordinates"][5]
    assert gripper["unit"] == "RANGE_0_100"


def test_degrees_data_on_pm100_robot_is_an_error_and_on_default_passes(tmp_path):
    root = so_dataset(tmp_path, "deg")
    bad = units_manifest(root, robot_command=command("--robot.use_degrees", "false"))
    assert codes(bad, "error") == ["units.joint_mode_mismatch"]
    assert "--robot.use_degrees=true" in bad["findings"][0]["remedy"]
    good = units_manifest(root, robot_command=command())
    assert good["status"] == "pass" and arm_mode(good, "action")["mode"] == "DEGREES"


def test_bimanual_arms_are_compared_per_arm(tmp_path):
    left, right = arm(400, "deg", "left_", 1), arm(400, "m100", "right_", 2)
    state = (left[0] + right[0], np.hstack([left[1], right[1]]))
    root = write(tmp_path, {"observation.state": state}, robot_type="bi_so_follower")
    report = units_manifest(root, robot_command=command("--robot.right_arm_config.use_degrees=false",
                                                        robot="bi_so_follower"))
    assert report["status"] == "pass", report["findings"]
    report = units_manifest(root, robot_command=command(robot="bi_so_follower"))
    mismatch = [f for f in report["findings"] if f["code"] == "units.joint_mode_mismatch"]
    assert len(mismatch) == 1 and "right arm" in mismatch[0]["message"]
    assert "--robot.right_arm_config.use_degrees=false" in mismatch[0]["remedy"]


def test_coincident_grids_stay_unknown_never_wrong(tmp_path):
    for mode in ("deg", "m100"):
        state = arm(400, mode, seed=3, cal=COINCIDENT)
        root = write(tmp_path, {"observation.state": state}, name=mode)
        report = units_manifest(root, robot_command=command())
        assert codes(report, "error") == [] and "units.joint_mode_match" not in codes(report)
        assert "units.joint_mode_undecided" in codes(report, "unknown")
        assert arm_mode(report, "observation.state")["status"] == "undecided"


def test_policy_outputs_and_static_values_are_unknown(tmp_path):
    state = arm(400, "m100", seed=1)
    noisy = (state[0], arm(400, "m100", seed=2)[1] * 0.9 + np.random.default_rng(5).normal(0, 0.01, (400, 6)))
    root = write(tmp_path, {"observation.state": state, "action": noisy})
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    assert arm_mode(report, "observation.state")["mode"] == "RANGE_M100_100"
    assert arm_mode(report, "action")["status"] == "undecided"
    assert "no motor-bus grid" in arm_mode(report, "action")["reason"]
    assert report["status"] == "unknown" and codes(report, "error") == []
    static = (state[0], np.repeat(state[1][:1], 400, axis=0))
    report = units_manifest(write(tmp_path, {"observation.state": static}, name="static"), robot_command=command())
    assert "only 1 distinct value" in arm_mode(report, "observation.state")["reason"]
    assert codes(report, "error") == []


def test_rounded_values_are_not_taken_as_a_grid(tmp_path):
    names, values = arm(400, "deg", seed=1)
    root = write(tmp_path, {"observation.state": (names, np.round(values, 1))})
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    shoulder = report["manifest"]["features"]["observation.state"]["coordinates"][0]
    assert shoulder["evidence"]["rounded_to"] == 0.1 and shoulder["unit"] is None
    # wrist_roll spans ±180: rounded values beyond ±100 still rule RANGE_M100_100 out, and nothing more
    assert arm_mode(report, "observation.state")["status"] == "partial"
    assert "units.joint_mode_mismatch" in codes(report, "error")
    assert units_manifest(root, robot_command=command())["status"] == "unknown"


def test_beyond_100_without_grid_rules_out_pm100_only(tmp_path):
    names, values = arm(400, "deg", seed=1)
    values = values + np.random.default_rng(1).normal(0, 0.01, values.shape)  # smoothed: no grid
    values[:, 4] = np.linspace(-150, 150, 400)
    root = write(tmp_path, {"observation.state": (names, values)})
    assert arm_mode(units_manifest(root), "observation.state")["mode"] == "NOT_RANGE_M100_100"
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    assert "units.joint_mode_mismatch" in codes(report, "error")
    report = units_manifest(root, robot_command=command())
    assert codes(report, "error") == [] and "units.joint_mode_undecided" in codes(report, "unknown")


def test_pre_pr777_recordings(tmp_path):
    root = write(tmp_path, {"observation.state": arm(400, "old", seed=1), "action": arm(400, "old", seed=2)},
                 robot_type="so100")
    alone = units_manifest(root)
    assert codes(alone, "warning") == ["units.pre_pr777_calibration"]
    assert arm_mode(alone, "observation.state", "main_")["mode"] == "PRE_PR777_DEGREES"
    deployed = units_manifest(root, robot_command=command())
    assert codes(deployed, "error") == ["units.pre_pr777_calibration"]
    assert "backwardcomp.mdx" in deployed["findings"][0]["message"]


def test_state_and_action_recorded_in_different_modes(tmp_path):
    report = units_manifest(so_dataset(tmp_path, "deg", "m100"))
    assert codes(report, "warning") == ["units.state_action_mode_disagree"] and codes(report, "error") == []


def test_one_arm_with_two_modes_is_a_conflict_not_a_unit(tmp_path):
    deg, m100_ = arm(400, "deg", seed=1)[1], arm(400, "m100", seed=2)[1]
    mixed = np.hstack([deg[:, :2], m100_[:, 2:]])
    report = units_manifest(write(tmp_path, {"observation.state": ([f"{m}.pos" for m in SO], mixed)}),
                            robot_command=command())
    assert arm_mode(report, "observation.state")["status"] == "conflict"
    assert "units.arm_mode_conflict" in codes(report, "warning") and codes(report, "error") == []


def test_duplicate_names_map_no_motor(tmp_path):
    names, values = arm(400, "m100", seed=1)
    names = names[:5] + [names[0]]
    report = units_manifest(write(tmp_path, {"observation.state": (names, values)}), robot_command=command())
    assert report["manifest"]["features"]["observation.state"]["arms"] == {}
    assert codes(report, "error") == []


def test_yaml_config_per_arm(tmp_path):
    pytest.importorskip("yaml")
    left, right = arm(400, "m100", "left_", 1), arm(400, "m100", "right_", 2)
    root = write(tmp_path, {"observation.state": (left[0] + right[0], np.hstack([left[1], right[1]]))},
                 robot_type="bi_so_follower")
    config = tmp_path / "record.yaml"
    config.write_text("robot:\n  type: bi_so_follower\n  left_arm_config: {port: /dev/a, use_degrees: false}\n"
                      "  right_arm_config: {port: /dev/b}\ndataset: {repo_id: me/x, fps: 30}\n")
    report = units_manifest(root, robot_config=config)
    mismatch = [f for f in report["findings"] if f["code"] == "units.joint_mode_mismatch"]
    assert len(mismatch) == 1 and "(right arm)" in mismatch[0]["message"]
    assert "units.joint_mode_match" in codes(report, "pass")


def test_quaternions_named_in_the_data(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(300, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    pose = np.hstack([rng.normal(size=(300, 3)), q])
    report = units_manifest(write(tmp_path, {"observation.state.base_pose": (names, pose)}, robot_type="g1"))
    rotation = report["manifest"]["rotations"][0]
    assert rotation["order"] == "xyzw" and rotation["scalar"] == "last" and rotation["unit_norm"] is True
    assert "units.quaternion_unit" in codes(report, "pass")
    pose[:50, 3:] *= 2
    report = units_manifest(write(tmp_path, {"observation.state.base_pose": (names, pose)}, name="bad"))
    assert codes(report, "warning") == ["units.quaternion_not_unit"]
    wxyz = units_manifest(write(tmp_path, {"observation.state.orientation": (["w", "x", "y", "z"], q)}, name="w"))
    assert wxyz["manifest"]["rotations"][0]["order"] == "wxyz"
    plain = units_manifest(write(tmp_path, {"observation.state.target": (["w", "x", "y", "z"], q)}, name="p"))
    assert plain["manifest"]["rotations"] == []  # w/x/y/z outside a rotation-named feature is not a claim


def test_timebase(tmp_path):
    state = arm(400, "m100", seed=1)
    episodes = np.repeat([0, 1], 200)
    report = units_manifest(write(tmp_path, {"observation.state": state}, episodes=episodes))
    assert report["manifest"]["timebase"]["status"] == "decided"
    assert "units.timebase_consistent" in codes(report, "pass")
    report = units_manifest(write(tmp_path, {"observation.state": state}, period=1 / 15, name="slow"))
    assert codes(report, "warning") == ["units.timebase_contradicts_fps"]
    assert abs(report["manifest"]["timebase"]["timestamp_fps"] - 15) < 0.01


@pytest.mark.parametrize("robot, flag, expected", [
    ("koch_follower", None, "RANGE_M100_100"), ("koch_follower", "--robot.use_degrees=true", "DEGREES"),
    ("omx_follower", None, "RANGE_M100_100"), ("so100_follower", None, "DEGREES"),
])
def test_deployment_modes_per_robot_type(tmp_path, robot, flag, expected):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command(*([flag] if flag else []), robot=robot))
    assert report["manifest"]["deployment"]["arms"][""]["mode"] == expected
    assert ("units.joint_mode_match" in codes(report)) == (expected == "RANGE_M100_100")


def test_lekiwi_arm_prefix_and_default(tmp_path):
    names, values = arm(400, "m100", "arm_", 1)
    root = write(tmp_path, {"observation.state": (names, values)}, robot_type="lekiwi")
    report = units_manifest(root, robot_command=command("--robot.remote_ip=x", robot="lekiwi"))
    assert report["manifest"]["deployment"]["arms"]["arm_"]["use_degrees"] is True
    assert "units.joint_mode_mismatch" in codes(report, "error")
    client = units_manifest(root, robot_command=command("--robot.remote_ip=x", robot="lekiwi_client"))
    assert "units.deployment_mode_unknown" in codes(client, "unknown") and codes(client, "error") == []


@pytest.mark.parametrize("flags", [("--robot.use_degrees=$MODE",), ("--config_path=cfg.json",),
                                   ("--robot.use_degrees=maybe",)])
def test_unresolved_deployment_mode_is_unknown(tmp_path, flags):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command(*flags))
    assert codes(report, "error") == []
    assert "units.deployment_mode_unknown" in codes(report, "unknown")


def test_unmodeled_robot_and_unmatched_names_are_unknown(tmp_path):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command(robot="hope_jr"))
    assert codes(report, "unknown")[0] == "units.deployment_mode_unknown"
    joints = (["left_waist", "left_shoulder"], np.random.default_rng(0).uniform(-1, 1, (300, 2)))
    aloha = write(tmp_path, {"observation.state": joints}, robot_type="aloha", name="aloha")
    report = units_manifest(aloha, robot_command=command())
    assert "units.deployment_mode_unknown" in codes(report, "unknown") and codes(report, "error") == []
    for coordinate in report["manifest"]["features"]["observation.state"]["coordinates"]:
        assert coordinate["unit"] is None and coordinate["status"] == "unknown"  # radians are never asserted


def test_config_file_deployment(tmp_path):
    config = tmp_path / "rollout.json"
    config.write_text(json.dumps({"robot": {"type": "so101_follower", "port": "/dev/x", "use_degrees": False},
                                  "fps": 30, "policy": {"path": "x"}}))
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_config=config)
    assert report["status"] == "pass"
    assert report["manifest"]["deployment"]["source"]["kind"] == "config"
    with pytest.raises(InputError):
        units_manifest(so_dataset(tmp_path, "m100", name="b"), robot_command=command(), robot_config=config)


def test_without_values_stats_can_only_rule_out_pm100(tmp_path, monkeypatch):
    names = [f"{m}.pos" for m in SO]
    stats = {"observation.state": {"min": [-150, -90, -80, -70, -60, 0], "max": [150, 90, 80, 70, 60, 50]}}
    root = write(tmp_path, {"observation.state": (names, np.zeros((2, 6)))}, data=False, stats=stats)
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    assert report["coverage"]["values_read"] is False
    assert "units.joint_mode_mismatch" in codes(report, "error")
    assert units_manifest(root, robot_command=command())["status"] == "unknown"
    with_data = so_dataset(tmp_path, "m100", name="m")
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    report = units_manifest(with_data, robot_command=command())
    assert report["coverage"]["values_read"] is False and "pyarrow" in report["coverage"]["reason"]
    assert report["status"] == "unknown"


def test_bounds_and_containment(tmp_path):
    root = so_dataset(tmp_path, "m100")
    partial = units_manifest(root, max_rows=50)
    assert partial["coverage"]["complete"] is False and partial["coverage"]["rows_read"] == 50
    with pytest.raises(InputError):
        units_manifest(root, max_rows=0)
    outside = tmp_path / "outside.parquet"
    (root / "data/chunk-000/file-000.parquet").rename(outside)
    (root / "data/chunk-000/file-000.parquet").symlink_to(outside)
    with pytest.raises(InputError, match="escapes"):
        units_manifest(root)


def test_output_contract(tmp_path):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command())
    assert set(report) == {"schema_version", "kind", "status", "counts", "coverage", "manifest", "findings", "scope"}
    assert report["kind"] == "units_manifest" and report["schema_version"] == 1
    assert set(report["counts"]) == {"error", "warning", "unknown", "pass"}
    for finding in report["findings"]:
        assert set(finding) == {"code", "status", "message", "remedy", "evidence"}
        for item in finding["evidence"]:
            assert set(item) == {"path", "sha256", "pointer", "value"}
    json.dumps(report, allow_nan=False)  # strict JSON: no NaN or infinity
    files = report["coverage"]["files"]
    assert len(files) == 1 and len(files[0]["sha256"]) == 64


def test_policy_actions_beyond_100_rule_nothing_out(tmp_path):
    state = arm(400, "m100", seed=1)
    overshoot = state[1] + np.random.default_rng(7).normal(0, 0.5, state[1].shape)
    overshoot[:, 1] = np.linspace(-100.4, 100.4, 400)  # a rollout records the raw policy action, before the bus clamps
    root = write(tmp_path, {"observation.state": state, "action": (state[0], overshoot)})
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    assert codes(report, "error") == []
    assert arm_mode(report, "action")["status"] == "undecided"
    assert "policy outputs" in arm_mode(report, "action")["reason"]
    assert "units.joint_mode_match" in codes(report, "pass")  # observation.state still decides


def test_current_names_on_the_pre_pr777_grid(tmp_path):
    names, values = arm(400, "old", seed=1)
    root = write(tmp_path, {"observation.state": ([n.removeprefix("main_") + ".pos" for n in names], values)})
    migrated = units_manifest(root, robot_command=command("--robot.use_degrees=true"))
    assert codes(migrated, "error") == [] and "units.joint_mode_undecided" in codes(migrated, "unknown")
    wrong = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    mismatch = [f for f in wrong["findings"] if f["code"] == "units.joint_mode_mismatch"]
    assert len(mismatch) == 1 and "--robot.use_degrees=true" in mismatch[0]["remedy"]


def test_inverted_drive_mode_and_fixed_size_lists_and_several_files(tmp_path):
    names, values = arm(600, "m100", seed=4)
    values[:, :5] = -values[:, :5]  # drive_mode 1 on a Feetech arm: -norm (motors_bus.py:931)
    root = write(tmp_path, {"observation.state": (names, values)}, fixed=True, files=3)
    report = units_manifest(root)
    assert arm_mode(report, "observation.state")["mode"] == "RANGE_M100_100"
    assert report["coverage"]["data_files_read"] == 3 and report["coverage"]["complete"] is True
    partial = units_manifest(root, max_files=1)
    assert partial["coverage"]["complete"] is False and partial["coverage"]["stopped"] == "max_files=1"


def test_bare_flag_and_nested_arm_mapping(tmp_path):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command("--robot.use_degrees", robot="koch_follower"))
    assert report["manifest"]["deployment"]["arms"][""]["use_degrees"] is True
    assert "units.joint_mode_mismatch" in codes(report, "error")
    left, right = arm(400, "m100", "left_", 1), arm(400, "m100", "right_", 2)
    root = write(tmp_path, {"observation.state": (left[0] + right[0], np.hstack([left[1], right[1]]))}, name="bi")
    nested = command("--robot.left_arm_config='{port: /dev/a, use_degrees: false}'",
                     "--robot.right_arm_config.use_degrees=false", robot="bi_so_follower")
    report = units_manifest(root, robot_command=nested)
    left_arm = report["manifest"]["deployment"]["arms"]["left_"]
    assert left_arm["use_degrees"] is False and left_arm["origin"] == "set in the command"
    assert report["status"] == "pass"


def test_other_coincident_range(tmp_path):
    cal = {m: (1000, 1910) for m in SO}  # 910 ticks divides 4550: the grids coincide
    report = units_manifest(write(tmp_path, {"observation.state": arm(400, "m100", seed=5, cal=cal)}),
                            robot_command=command())
    assert arm_mode(report, "observation.state")["status"] == "undecided"
    assert codes(report, "error") == []


def test_nan_timestamps_and_non_numeric_columns(tmp_path):
    state = arm(400, "m100", seed=1)
    stamps = np.arange(400) / 30
    stamps[10] = np.nan
    report = units_manifest(write(tmp_path, {"observation.state": state}, stamps=stamps), robot_command=command())
    json.dumps(report, allow_nan=False)
    assert report["manifest"]["timebase"]["status"] == "decided"
    root = write(tmp_path, {"observation.state": state}, name="text")
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    words = pa.array([[str(v) for v in row] for row in state[1].tolist()], type=pa.list_(pa.string()))
    pq.write_table(table.set_column(table.schema.get_field_index("observation.state"), "observation.state", words), path)
    report = units_manifest(root, robot_command=command())
    assert arm_mode(report, "observation.state")["status"] == "undecided" and codes(report, "error") == []


def test_stats_evidence_and_dangling_data_link(tmp_path):
    names = [f"{m}.pos" for m in SO]
    stats = {"observation.state": {"min": [-150, -90, -80, -70, -60, 0], "max": [150, 90, 80, 70, 60, 50]},
             "action": {"min": [-150, -90, -80, -70, -60, 0], "max": [150, 90, 80, 70, 60, 50]}}
    root = write(tmp_path, {"observation.state": (names, np.zeros((2, 6))), "action": (names, np.zeros((2, 6)))},
                 data=False, stats=stats)
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    mismatch = [f for f in report["findings"] if f["code"] == "units.joint_mode_mismatch"]
    assert len(mismatch) == 1 and mismatch[0]["message"].startswith("observation.state")  # actions may overshoot
    assert mismatch[0]["evidence"][0]["path"].endswith("meta/stats.json") and len(mismatch[0]["evidence"][0]["sha256"]) == 64
    root = so_dataset(tmp_path, "m100", name="dangling")
    target = root / "data/chunk-000/file-000.parquet"
    target.rename(root / "data/moved.bin")
    target.symlink_to(root / "data/missing.parquet")
    with pytest.raises(InputError):
        units_manifest(root)


# Regressions from the independent validation of 9daa6525 (D1-D5, scope, robustness).

LEADER = {"shoulder_pan": (700, 3400), "shoulder_lift": (900, 3150), "elbow_flex": (800, 3000),
          "wrist_flex": (850, 3300), "wrist_roll": (0, 4095), "gripper": (2000, 3300)}


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32).astype(np.float64)


def degrees_working_span(n, seed):
    """DEGREES readings over working spans away from 0 (bfloat16 keeps 1/16 steps there)."""
    rng = np.random.default_rng(seed)
    cols = []
    spans = [(1300, 2700), (900, 1900), (2250, 3050), (2200, 2900), (1000, 3100), (2100, 3400)]
    for motor, (a, b) in zip(SO, spans, strict=True):
        lo, hi = CAL[motor]
        raw = rng.integers(a, b + 1, n).astype(np.float64)
        cols.append(r0100(raw, lo, hi) if motor == "gripper" else degrees(raw, lo, hi))
    return [f"{m}.pos" for m in SO], np.stack(cols, axis=1)


def test_requantized_values_are_unknown_not_pm100(tmp_path):
    names, values = degrees_working_span(3000, 11)
    for label, requantize in (("q025", lambda v: np.round(v * 4) / 4), ("q005", lambda v: np.round(v * 20) / 20),
                              ("q0125", lambda v: np.round(v * 8) / 8), ("bf16", bf16)):
        quantized = requantize(values)
        root = write(tmp_path, {"observation.state": (names, quantized), "action": (names, quantized)}, name=label)
        report = units_manifest(root, robot_command=command())
        assert codes(report, "error") == [], (label, report["findings"])  # no --robot.use_degrees=false remedy
        assert arm_mode(report, "observation.state")["mode"] is None
        assert "requantized" in arm_mode(report, "observation.state")["reason"], label
    lift = [c for c in report["manifest"]["features"]["observation.state"]["coordinates"] if c["motor"] == "shoulder_lift"]
    assert lift[0]["unit"] is None and lift[0]["evidence"]["requantized_step"] in (1 / 16, 1 / 8)
    # true RANGE_M100_100 readings, whose ranges are not short-decimal steps, are still decided
    report = units_manifest(so_dataset(tmp_path, "m100", name="true"), robot_command=command())
    assert arm_mode(report, "observation.state")["mode"] == "RANGE_M100_100"


def test_teleop_leader_mode_is_not_compared_with_the_robot(tmp_path):
    state, action = arm(3000, "deg", seed=3), arm(3000, "m100", seed=4, cal=LEADER)
    root = write(tmp_path, {"observation.state": state, "action": action})
    for flags, expected_errors in (((), []), (("--robot.use_degrees=false",), ["units.joint_mode_mismatch"])):
        report = units_manifest(root, robot_command=command(*flags))
        assert codes(report, "error") == expected_errors, report["findings"]
        assert "units.state_action_mode_disagree" in codes(report, "warning")
        assert not any(f["message"].startswith("action") for f in report["findings"]
                       if f["code"] in ("units.joint_mode_mismatch", "units.joint_mode_match"))
        remedies = [f["remedy"] for f in report["findings"] if f["remedy"]]
        assert all("use_degrees=false" not in r for r in remedies), remedies  # one remedy: the follower's mode
        assert any("--robot.use_degrees=true" in r for r in remedies)
    alone = units_manifest(root)
    assert "--robot.use_degrees=true" in alone["findings"][0]["remedy"]
    assert "lerobot_record.py:320-327" in alone["findings"][0]["message"]


def test_joint_off_the_arm_grid_is_flagged_not_inherited(tmp_path):
    names, state = arm(3000, "deg", seed=1)
    lo, hi = CAL["elbow_flex"]
    fifteen = state.copy()
    fifteen[:, 2] = m100(np.repeat(np.arange(2000, 2015), 200).astype(float), lo, hi)  # off the DEGREES grid
    smooth = state.copy()
    smooth[:, 2] = np.random.default_rng(2).normal(30, 20, 3000)  # on no grid
    for label, values in (("fifteen", fifteen), ("smooth", smooth)):
        report = units_manifest(write(tmp_path, {"observation.state": (names, values)}, name=label),
                                robot_command=command())
        elbow = report["manifest"]["features"]["observation.state"]["coordinates"][2]
        assert elbow["status"] == "off_arm_grid" and elbow["unit"] is None, (label, elbow)
        assert arm_mode(report, "observation.state")["off_grid"] == ["elbow_flex"]
        assert "units.joint_off_arm_grid" in codes(report, "warning") and report["status"] != "pass"
    few = state.copy()
    few[:, 2] = degrees(np.repeat(np.arange(2000, 2005), 600).astype(float), lo, hi)  # low motion, on the grid
    report = units_manifest(write(tmp_path, {"observation.state": (names, few)}, name="few"), robot_command=command())
    elbow = report["manifest"]["features"]["observation.state"]["coordinates"][2]
    assert elbow["status"] == "decided_by_arm" and elbow["unit"] == "DEGREES"
    assert report["status"] == "pass"


def test_action_is_compared_only_when_its_arm_has_no_state(tmp_path):
    state, action = arm(400, "m100", seed=1), arm(400, "m100", seed=2, cal=LEADER)
    both = units_manifest(write(tmp_path, {"observation.state": state, "action": action}), robot_command=command())
    assert [f["message"].split(" ")[0] for f in both["findings"] if f["code"] == "units.joint_mode_mismatch"] == [
        "observation.state"]
    alone = units_manifest(write(tmp_path, {"action": action}, name="action"), robot_command=command())
    message = next(f["message"] for f in alone["findings"] if f["code"] == "units.joint_mode_mismatch")
    pan = action[1][:, 0].astype(np.float32).astype(np.float64)
    R = LEADER["shoulder_pan"][1] - LEADER["shoulder_pan"][0]  # without a state, the action's own grid is all
    error = max(abs(v * (4095 / 360 - R / 200)) for v in (pan.min(), pan.max())) * 360 / 4096
    assert f"shoulder_pan up to {error:.1f}° (range {R} ticks)" in message


def test_quaternion_wording_and_rotation_tokens(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    zero = np.hstack([rng.normal(size=(3000, 3)), q])
    zero[0, 3:] = 0
    report = units_manifest(write(tmp_path, {"observation.state.base_pose": (names, zero)}, robot_type="g1"))
    warning = next(f for f in report["findings"] if f["code"] == "units.quaternion_not_unit")
    assert "1 of 3000 rows" in warning["message"] and "100.0%" not in warning["message"]
    assert "not checked" in warning["message"] and "component order" not in warning["remedy"]
    good = units_manifest(write(tmp_path, {"observation.state.base_pose": (names, np.hstack([zero[:, :3], q]))},
                                name="good"))
    passed = next(f for f in good["findings"] if f["code"] == "units.quaternion_unit")
    assert "cannot confirm the order" in passed["message"]
    protocol = units_manifest(write(tmp_path, {"observation.protocol": (["w", "x", "y", "z"], q)}, name="protocol"))
    assert protocol["manifest"]["rotations"] == []  # "rot" inside a word is not a rotation name
    ee = units_manifest(write(tmp_path, {"observation.ee_quat": (["w", "x", "y", "z"], q)}, name="ee"))
    assert ee["manifest"]["rotations"][0]["order"] == "wxyz"


def test_scope_states_the_default_timebase_and_policy_assumptions(tmp_path):
    report = units_manifest(so_dataset(tmp_path, "m100"), robot_command=command())
    assert "146d847" in report["scope"] and "--policy.path" in report["scope"]
    assert "median" in report["scope"] and "dataset_writer.py:223-227" in report["scope"]
    mismatch = next(f for f in report["findings"] if f["code"] == "units.joint_mode_mismatch")
    assert "other LeRobot versions may default differently" in mismatch["message"]
    assert "the policy is not read" in mismatch["remedy"]
    timebase = next(f for f in report["findings"] if f["code"] == "units.timebase_consistent")
    assert "by construction" in timebase["message"] and "399 of 399 steps" in timebase["message"]


def test_null_rows_unread_columns_and_nan_stats(tmp_path):
    root = so_dataset(tmp_path, "m100")
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    rows = table.column("observation.state").to_pylist()
    rows[3] = None
    index = table.schema.get_field_index("observation.state")
    pq.write_table(table.set_column(index, "observation.state", pa.array(rows, type=pa.list_(pa.float32()))), path)
    report = units_manifest(root, robot_command=command("--robot.use_degrees=false"))
    assert report["coverage"]["null_rows"] == {"observation.state": 1}  # only the null row is dropped
    assert arm_mode(report, "observation.state")["mode"] == "RANGE_M100_100"
    assert report["coverage"]["complete"] is True and report["status"] == "pass"
    text = so_dataset(tmp_path, "m100", name="text")
    path = text / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    words = pa.array([[str(v) for v in row] for row in table.column("action").to_pylist()], type=pa.list_(pa.string()))
    pq.write_table(table.set_column(table.schema.get_field_index("action"), "action", words), path)
    report = units_manifest(text)
    assert report["coverage"]["unread_rows"] == {"action": 400} and report["coverage"]["complete"] is False
    flags = so_dataset(tmp_path, "m100", name="flags")  # a boolean column (next.done) is read, not skipped
    info = json.loads((flags / "meta/info.json").read_text())
    info["features"]["next.done"] = {"dtype": "bool", "shape": [1], "names": None}
    (flags / "meta/info.json").write_text(json.dumps(info))
    path = flags / "data/chunk-000/file-000.parquet"
    pq.write_table(pq.read_table(path).append_column("next.done", pa.array(np.zeros(400, dtype=bool))), path)
    report = units_manifest(flags)
    assert report["coverage"]["unread_rows"] == {} and report["coverage"]["complete"] is True
    nan = so_dataset(tmp_path, "m100", name="nan")
    (nan / "meta/stats.json").write_text('{"observation.state": {"min": [NaN, 0, 0, 0, 0, 0]}}')
    report = units_manifest(nan, robot_command=command("--robot.use_degrees=false"))
    assert "stats.json is not read" in report["coverage"]["stats_not_used"] and report["status"] == "pass"


# Regressions from the recheck of e23834aa (X2a/X2f, X2c, 027 wording, SCOPE).


def test_state_action_disagreement_is_a_warning_with_a_koch_remedy(tmp_path):
    state, action = arm(3000, "deg", seed=13), arm(3000, "m100", seed=14, cal=LEADER)
    root = write(tmp_path, {"observation.state": state, "action": action})
    for flags in ((), ("--robot.use_degrees=true",)):  # the deployment reproduces the follower's recorded mode
        report = units_manifest(root, robot_command=command(*flags))
        assert report["status"] == "warning" and codes(report, "error") == [], report["findings"]
    report = units_manifest(root, robot_command=command("--robot.use_degrees=true", robot="koch_follower"))
    disagree = next(f for f in report["findings"] if f["code"] == "units.state_action_mode_disagree")
    assert disagree["status"] == "warning" and "koch_leader.py:45-50" in disagree["message"]
    assert "record with --robot.use_degrees=false" in disagree["remedy"] and "--teleop" not in disagree["remedy"]
    assert codes(units_manifest(root, robot_command=command("--robot.use_degrees=false")), "error") == [
        "units.joint_mode_mismatch"]  # the state's own mismatch still errors


def test_undecided_state_keeps_the_leader_action_from_deciding(tmp_path):
    rng = np.random.default_rng(5)
    lo, hi = 900, 3175  # 2275 ticks, half-tick middle: the DEGREES and RANGE_M100_100 grids coincide
    raw = rng.integers(1300, 2801, (3000, 6)).astype(np.float64)
    state = degrees(raw, lo, hi)
    state[:, 5] = r0100(raw[:, 5], lo, hi)
    names = [f"{m}.pos" for m in SO]
    action = arm(3000, "m100", seed=6, cal=LEADER)
    root = write(tmp_path, {"observation.state": (names, state), "action": action})
    report = units_manifest(root, robot_command=command())
    assert arm_mode(report, "observation.state")["status"] == "undecided"
    assert codes(report, "error") == [], report["findings"]  # no "Run with --robot.use_degrees=false" from the leader
    unknown = [f["message"] for f in report["findings"] if f["message"].startswith("action")]
    assert len(unknown) == 1 and "the action shows the teleoperator's mode" in unknown[0]


def test_recalibrated_range_joints_are_named_per_episode(tmp_path):
    second = dict(CAL, shoulder_pan=(700, 3300), shoulder_lift=(850, 3190), elbow_flex=(900, 3050),
                  wrist_flex=(800, 3100))
    names, first = arm(1500, "m100", seed=9)
    state = np.vstack([first, arm(1500, "m100", seed=10, cal=second)[1]])
    episodes = np.repeat(np.arange(10), 300)
    report = units_manifest(write(tmp_path, {"observation.state": (names, state)}, episodes=episodes),
                            robot_command=command())
    pan = report["manifest"]["features"]["observation.state"]["coordinates"][0]
    assert pan["unit"] == "RANGE_M100_100" and pan["status"] == "decided_by_arm"
    assert "calibrations that change between sessions" in pan["basis"]
    warning = next(f for f in report["findings"] if f["code"] == "units.joint_off_arm_grid")
    assert "shoulder_pan is RANGE_M100_100 under calibrations that change between sessions" in warning["message"]
    assert "2570, 2600" in warning["message"]  # the two calibrated ranges, one per session
    processed = np.clip(state + np.random.default_rng(3).normal(0, 0.01, state.shape), -100, 100)
    processed[:, 4] = state[:, 4]  # wrist_roll alone still decides RANGE_M100_100
    report = units_manifest(write(tmp_path, {"observation.state": (names, processed)}, episodes=episodes, name="p"))
    warning = next(f for f in report["findings"] if f["code"] == "units.joint_off_arm_grid")
    assert "may be RANGE_M100_100 readings under changing calibrations" in warning["message"]
    assert "under calibrations that change between sessions" not in warning["message"]


def test_scope_states_requantization_and_degree_scale_limits(tmp_path):
    scope = units_manifest(so_dataset(tmp_path, "m100"))["scope"]
    assert "1/3" in scope and "cannot be told from a genuine calibration" in scope
    assert "360/4096" in scope and "pre-PR #777" in scope
