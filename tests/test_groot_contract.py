"""GR00T dataset contract. Fixtures copy NVIDIA's cube_to_bowl_5 demo metadata and SO100 config (Isaac-GR00T 944fe70).
Fault cases are the ones GR00T's own code was run on (.validation/k4-groot/data/native_probe.json, the validator's
cases C1-C19 / R1-R16 in .validation/k4-groot/validator/, and .validation/k4-groot/rework/)."""
import hashlib
import json
import os

import pytest

from policy_doctor import groot_contract
from policy_doctor.artifacts import InputError
from policy_doctor.groot_contract import check_groot_dataset, read_modality_config, stats_fingerprint

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "video": {"front": {"original_key": "observation.images.front"}, "wrist": {"original_key": "observation.images.wrist"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
# examples/SO100/so100_config.py at 944fe70, comments removed.
CONFIG = '''
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
so100_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''
RELATIVE_ARM = "ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)"


def _stats(width):
    return {name: [0.5] * width for name in ("mean", "std", "min", "max", "q01", "q99")}


def _groot_fingerprint(name, feature):
    """gr00t/data/stats.py:192-198, copied: the reference the module's reimplementation is held to."""
    payload = {"feature": name, "dtype": feature.get("dtype"), "shape": feature.get("shape")}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _dataset(root, modality=MODALITY, version="v2.1", stats_width=6, cameras=("wrist", "front"), info_edit=None):
    meta = root / "meta"
    meta.mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [6], "names": NAMES}
    camera = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]}
    info = {"codebase_version": version, "fps": 30, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"action": vector, "observation.state": vector, "task_index": {"dtype": "int64", "shape": [1]},
                         **{f"observation.images.{c}": dict(camera) for c in cameras}}}
    if info_edit:
        info_edit(info)
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "modality.json").write_text(json.dumps(modality))
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["put the cube in the bowl"], "length": 3}) + "\n")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "put the cube in the bowl"}) + "\n")
    (meta / "stats.json").write_text(json.dumps({"action": _stats(stats_width), "observation.state": _stats(stats_width)}))
    (meta / "relative_stats.json").write_text(json.dumps({"single_arm": _stats(5)}))
    return root


def _config(tmp_path, text=CONFIG, name="so100_config.py"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _codes(report, status=None):
    return {f["code"] for f in report["findings"] if status is None or f["status"] == status}


def _finding(report, code, status=None):
    return next(f for f in report["findings"] if f["code"] == code and (status is None or f["status"] == status))


def _modality(**changes):
    value = json.loads(json.dumps(MODALITY))
    for path, entry in changes.items():
        side, _, group = path.partition("__")
        if entry is None and group:
            del value[side][group]
        elif entry is None:
            del value[side]
        elif group:
            value[side][group] = entry
        else:
            value[side] = entry
    return value


def _check(tmp_path, modality=MODALITY, config=CONFIG, **kwargs):
    return check_groot_dataset(_dataset(tmp_path / "ds", modality, **kwargs), _config(tmp_path, config))


# ---------------------------------------------------------------- healthy layout and the two silent headline faults

def test_nvidia_demo_layout_and_config_pass(tmp_path):
    report = _check(tmp_path)
    assert report["status"] == "pass", [f for f in report["findings"] if f["status"] != "pass"]
    reads = [f["message"] for f in report["findings"] if f["code"] == "groot.state_slice"]
    assert "state.gripper reads observation.state[5:6] = ['gripper.pos']." in reads
    assert report["modality_config"]["action"]["relative"] == {"single_arm": "single_arm"}
    assert report["modality_config_name"] == "so100_config"
    assert report["modality_config"]["action"]["delta_indices"] == list(range(16))
    assert all(f["applies_to"] for f in report["findings"])
    # The profile is named in every report; the field sits after consumer and is the only one added for N1.7.
    assert report["groot_version"] == "n1.7"
    assert list(report) == ["schema_version", "kind", "status", "counts", "dataset", "consumer", "groot_version",
                            "modality_config", "modality_config_name", "files", "findings", "paths", "scope"]


def test_template_from_a_seven_dof_arm_is_caught(tmp_path):
    # GR00T gave single_arm 6 values (including gripper.pos) and gripper 0, with no error, on both paths.
    seven = _modality(state__single_arm={"start": 0, "end": 6}, state__gripper={"start": 6, "end": 7},
                      action__single_arm={"start": 0, "end": 6}, action__gripper={"start": 6, "end": 7})
    report = _check(tmp_path, seven)
    errors = [f for f in report["findings"] if f["status"] == "error"]
    assert {f["code"] for f in errors} >= {"groot.state_slice", "groot.action_slice"}
    assert any("cut to 0 value(s), with no error or warning" in f["message"] for f in errors)
    assert "groot.state_names" in _codes(report, "warning")  # single_arm now reads gripper.pos


def test_cameras_renamed_and_reordered_are_reported_as_swapped(tmp_path):
    # GR00T paired config front -> wrist video and wrist -> front video, logging only a warning.
    renamed = _modality(video={"cam_wrist": {"original_key": "observation.images.wrist"},
                               "cam_front": {"original_key": "observation.images.front"}})
    mapping = _finding(_check(tmp_path, renamed), "groot.video_mapping")
    assert mapping["status"] == "error"
    assert "front reads observation.images.wrist" in mapping["message"] and "Auto-mapping by position" in mapping["message"]


def test_renamed_cameras_in_the_right_order_are_a_warning(tmp_path):
    renamed = _modality(video={"top": {"original_key": "observation.images.front"},
                               "hand": {"original_key": "observation.images.wrist"}})
    mapping = _finding(_check(tmp_path, renamed), "groot.video_mapping")
    assert mapping["status"] == "warning" and "front reads observation.images.front" in mapping["message"]


# ---------------------------------------------------------------- F5 cameras

def test_swapped_original_keys_under_matching_names_are_caught(tmp_path):
    # Validator C5: names match the config, so GR00T reads by name with no warning, and front reads the wrist video.
    swapped = _modality(video={"front": {"original_key": "observation.images.wrist"},
                               "wrist": {"original_key": "observation.images.front"}})
    mapping = _finding(_check(tmp_path, swapped), "groot.video_mapping")
    assert mapping["status"] == "error" and "front reads observation.images.wrist" in mapping["message"]
    assert "by name" in mapping["message"] and "Swapped" in mapping["message"]


@pytest.mark.parametrize("names", [("cam_front", "cam_wrist"), ("cam_0", "cam_1")])
def test_swapped_original_keys_under_other_names_are_caught(tmp_path, names):
    # Validator C6 (prefixed names that agree with the config) and C6b (neutral names); GR00T pairs by position.
    swapped = _modality(video={names[0]: {"original_key": "observation.images.wrist"},
                               names[1]: {"original_key": "observation.images.front"}})
    assert _finding(_check(tmp_path, swapped), "groot.video_mapping")["status"] == "error"


def test_entry_names_are_used_when_the_columns_say_nothing(tmp_path):
    neutral = _modality(video={"cam_wrist": {"original_key": "observation.images.cam_0"},
                               "cam_front": {"original_key": "observation.images.cam_1"}})
    mapping = _finding(_check(tmp_path, neutral, cameras=("cam_0", "cam_1")), "groot.video_mapping")
    assert mapping["status"] == "error" and "entry cam_front is named for it" in mapping["message"]


def test_correct_positional_pairing_is_not_called_swapped(tmp_path):
    # Validator C4 (NVIDIA's libero_sim config): "image" is a substring of "wrist_image" but not a whole-token match
    # of the agentview column; GR00T maps image -> agentview and wrist_image -> wrist_image, which is right.
    config = CONFIG.replace('modality_keys=["front", "wrist"]', 'modality_keys=["image", "wrist_image"]')
    libero = _modality(video={"agentview": {"original_key": "observation.images.agentview"},
                              "wrist_image": {"original_key": "observation.images.wrist_image"}})
    mapping = _finding(_check(tmp_path, libero, config, cameras=("agentview", "wrist_image")), "groot.video_mapping")
    assert mapping["status"] == "warning", mapping["message"]
    assert "image reads observation.images.agentview" in mapping["message"]


def test_config_cameras_need_a_video_section(tmp_path):
    # Validator C18: GR00T raised KeyError: 'video' at loader:426.
    finding = _finding(_check(tmp_path, _modality(video=None)), "groot.video")
    assert finding["status"] == "error" and "lerobot_episode_loader.py:282-284, 426" in finding["message"]
    no_path = check_groot_dataset(_dataset(tmp_path / "ds2", _modality(video=None),
                                           info_edit=lambda i: i.pop("video_path")), _config(tmp_path))
    assert _finding(no_path, "groot.video")["status"] == "warning"


def test_camera_count_mismatch_is_an_error(tmp_path):
    one = _modality(video={"cam": {"original_key": "observation.images.front"}})
    finding = _finding(_check(tmp_path, one), "groot.video_mapping")
    assert finding["status"] == "error" and ":285-291" in finding["message"]


# ---------------------------------------------------------------- F2 video column

def test_video_feature_needs_only_to_be_declared(tmp_path):
    # jungwook2358/robocasa_mg_gr00t_100 declares cameras with dtype "image" and GR00T loads them (validator N22).
    def as_image(info):
        for camera in ("front", "wrist"):
            info["features"][f"observation.images.{camera}"]["dtype"] = "image"
    assert _check(tmp_path, info_edit=as_image)["status"] == "pass"
    undeclared = _modality(video__front={"original_key": "observation.images.top"})
    finding = _finding(check_groot_dataset(_dataset(tmp_path / "ds2", undeclared), _config(tmp_path)), "groot.video_column")
    assert finding["status"] == "error" and ":429-431" in finding["message"]
    # Validator C17: declared, but not a camera feature -> only a warning (GR00T checks presence, not dtype).
    non_video = _modality(video__front={"original_key": "task_index"})
    finding = _finding(check_groot_dataset(_dataset(tmp_path / "ds3", non_video), _config(tmp_path)), "groot.video_column")
    assert finding["status"] == "warning" and "'int64'" in finding["message"]


# ---------------------------------------------------------------- F1 statistics

def test_missing_statistics_are_generated_at_fine_tuning(tmp_path):
    # Validator C2 / IPEC-COMMUNITY LIBERO: factory.py:57-60 generates stats.json; only direct loaders assert.
    ds = _dataset(tmp_path / "ds")
    (ds / "meta" / "stats.json").unlink()
    report = check_groot_dataset(ds, _config(tmp_path))
    finding = _finding(report, "groot.statistics")
    assert report["status"] == "pass" and finding["status"] == "pass"
    assert "factory.py:57-59" in finding["message"] and "lerobot_episode_loader.py:180-182" in finding["message"]
    assert "open_loop_eval.py:327" in finding["message"] and "gr00t/data/stats.py" in finding["remedy"]


def test_stale_statistics_of_another_width_are_regenerated(tmp_path):
    # Validator C3 / N3: a LeRobot stats.json of the wrong width with no __fingerprints__ is recomputed.
    report = _check(tmp_path, stats_width=7)
    assert report["status"] == "pass"
    finding = _finding(report, "groot.statistics")
    assert finding["status"] == "pass" and "recomputes it" in finding["message"] and "do not match" in finding["message"]


def test_fingerprinted_statistics_of_another_width_are_an_error(tmp_path):
    # Validator N3b: GR00T keeps a fingerprint-matching entry, so a wrong width reaches training silently.
    ds = _dataset(tmp_path / "ds")
    info = json.loads((ds / "meta" / "info.json").read_text())
    stats = {"action": _stats(5), "observation.state": _stats(5),
             "__fingerprints__": {c: _groot_fingerprint(c, info["features"][c]) for c in ("action", "observation.state")}}
    (ds / "meta" / "stats.json").write_text(json.dumps(stats))
    report = check_groot_dataset(ds, _config(tmp_path))
    errors = [f for f in report["findings"] if f["code"] == "groot.statistics" and f["status"] == "error"]
    assert len(errors) == 2 and all(f["applies_to"] == "fine-tuning" for f in errors)
    assert "keeps it" in errors[0]["message"] and "527-530" in errors[0]["message"]
    # A fingerprint of another schema (shape [5]) is stale: GR00T recomputes, so it is only a note.
    stats["__fingerprints__"] = {c: _groot_fingerprint(c, {"dtype": "float32", "shape": [5]}) for c in ("action", "observation.state")}
    (ds / "meta" / "stats.json").write_text(json.dumps(stats))
    assert "groot.statistics" not in _codes(check_groot_dataset(ds, _config(tmp_path)), "error")


def test_fingerprint_matches_values_written_by_groot():
    # __fingerprints__ written by GR00T's own generate_stats into demo_data/droid_sample and libero_demo (944fe70).
    assert stats_fingerprint("observation.state", {"dtype": "float32", "shape": [17], "names": ["x"] * 17}) == \
        "sha256:edda7d765942f8b5f187f6c395dfc6981e7aa38ac9e97503a7ffd53e0b71ac7d"
    assert stats_fingerprint("action", {"dtype": "float32", "shape": [17]}) == \
        "sha256:99173041873a763043fa70da1891b7ebdf30e154106979be7b5748a063d28980"
    assert stats_fingerprint("observation.state", {"dtype": "float32", "shape": [8]}) == \
        "sha256:cdd88a979a931bb5fa6173e45c09227ad312fdce23b82d46bc5b8804625e20f2"
    for name, feature in (("action", {"dtype": "float64", "shape": [7]}), ("x.y", {"shape": [2, 3]}), ("ü", {})):
        assert stats_fingerprint(name, feature) == _groot_fingerprint(name, feature)


def test_statistics_are_read_like_groot(tmp_path):
    ds = _dataset(tmp_path / "ds")
    # Validator R5: NaN is valid for GR00T's json.load (and generate_stats can write it).
    (ds / "meta" / "stats.json").write_text('{"action": %s, "observation.state": %s, "timestamp": {"std": [NaN]}}'
                                            % (json.dumps(_stats(6)), json.dumps(_stats(6))))
    assert check_groot_dataset(ds, _config(tmp_path))["status"] == "pass"
    for broken in ("{not json", ""):  # fine-tuning regenerates both (stats.py:79-86); json.load in the loader fails
        (ds / "meta" / "stats.json").write_text(broken)
        finding = _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.statistics")
        assert finding["status"] == "warning" and finding["applies_to"] == "direct loaders"
    (ds / "meta" / "stats.json").write_text("[]")
    assert _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.statistics")["status"] == "error"


def test_used_columns_must_be_float_columns_declared_in_info_json(tmp_path):
    # generate_stats keeps statistics only for declared float columns (stats.py:256, 274-276) -> KeyError loader:527.
    # GR00T-Dateset/gr1_arms_waist-* declare both columns with dtype "object" (native: rework F1_*_object case).
    def object_columns(info):
        info["features"]["observation.state"]["dtype"] = "object"
        info["features"]["action"]["dtype"] = "object"
    report = _check(tmp_path, info_edit=object_columns)
    errors = [f for f in report["findings"] if f["status"] == "error"]
    assert [f["code"] for f in errors] == ["groot.state_column", "groot.action_column"]  # one per column, not per group
    assert all(f["applies_to"] == "fine-tuning" and "stats.py:256" in f["message"] and ":527" in f["message"]
               for f in errors)
    assert "['single_arm', 'gripper']" in errors[0]["message"]

    def undeclared(info):
        del info["features"]["observation.state"]
    finding = _finding(check_groot_dataset(_dataset(tmp_path / "ds2", info_edit=undeclared), _config(tmp_path)),
                       "groot.state_column")
    assert finding["status"] == "error" and "does not declare" in finding["message"]


def test_missing_relative_statistics_are_generated_at_fine_tuning(tmp_path):
    # Validator C1 / NVIDIA's SO100 workflow (convert_v3_to_v2.py copies only stats.json).
    ds = _dataset(tmp_path / "ds")
    (ds / "meta" / "relative_stats.json").unlink()
    report = check_groot_dataset(ds, _config(tmp_path))
    finding = _finding(report, "groot.relative_statistics")
    assert report["status"] == "pass" and finding["status"] == "pass" and "stats.py:415-435" in finding["message"]
    (ds / "meta" / "relative_stats.json").write_text("[]")  # the loader calls .pop on it (loader:193)
    assert _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.relative_statistics")["status"] == "error"
    # Unparseable: with a relative action generate_rel_stats opens the loader (stats.py:321) before rewriting it.
    (ds / "meta" / "relative_stats.json").write_text("{not json")
    assert _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.relative_statistics")["status"] == "error"
    absolute = CONFIG.replace("ActionRepresentation.RELATIVE", "ActionRepresentation.ABSOLUTE")
    finding = _finding(check_groot_dataset(ds, _config(tmp_path, absolute, "absolute.py")), "groot.relative_statistics")
    assert finding["status"] == "warning" and finding["applies_to"] == "direct loaders"


# ---------------------------------------------------------------- F3 relative actions

def test_relative_state_must_be_a_config_state_key(tmp_path):
    # Validator C7: modality.json has state single_arm, but the config's state keys are arm/gripper (stats.py:311).
    both = _modality(state={"single_arm": {"start": 0, "end": 5}, "arm": {"start": 0, "end": 5},
                            "gripper": {"start": 5, "end": 6}})
    config = CONFIG.replace('modality_keys=["single_arm", "gripper"]),', 'modality_keys=["arm", "gripper"]),', 1)
    finding = _finding(_check(tmp_path, both, config), "groot.relative_state")
    assert finding["status"] == "error" and "stats.py:311" in finding["message"]
    assert "data_config.md:218-220" in finding["message"] and "Not in GR00T's docs" not in finding["message"]


def test_explicit_state_key_none_means_the_action_key(tmp_path):
    # Validator C8: GR00T uses `state_key or action_key` (stats.py:310).
    config = CONFIG.replace("format=ActionFormat.DEFAULT),", "format=ActionFormat.DEFAULT, state_key=None),", 1)
    report = _check(tmp_path, config=config)
    assert report["status"] == "pass" and report["modality_config"]["action"]["relative"] == {"single_arm": "single_arm"}


def test_relative_state_and_action_delta_indices_must_align(tmp_path):
    # Validator C13: action list(range(1, 17)) vs state [0]; stats.py:317-320 asserts.
    finding = _finding(_check(tmp_path, config=CONFIG.replace("list(range(0, 16))", "list(range(1, 17))")),
                       "groot.relative_delta")
    assert finding["status"] == "error" and "stats.py:316-320" in finding["message"]
    unread = _check(tmp_path / "b", config=CONFIG.replace("list(range(0, 16))", "np.arange(16).tolist()"))
    assert _finding(unread, "groot.relative_delta")["status"] == "unknown"


def test_relative_width_applies_to_joint_space_only(tmp_path):
    narrow = _modality(state={"single_arm": {"start": 0, "end": 4}, "gripper": {"start": 5, "end": 6}})
    finding = _finding(_check(tmp_path, narrow), "groot.relative_width")
    assert finding["status"] == "error" and "pose.py:270-274" in finding["message"]
    eef = CONFIG.replace(RELATIVE_ARM, RELATIVE_ARM.replace("NON_EEF", "EEF").replace("DEFAULT", "XYZ_ROT6D"))
    assert "groot.relative_width" not in _codes(_check(tmp_path / "b", narrow, eef))


def test_action_configs_that_are_not_a_literal_list_are_unknown(tmp_path):
    # Validator R12: relative actions used to be dropped silently.
    config = CONFIG.replace(f"action_configs=[\n            {RELATIVE_ARM},",
                            f"action_configs=[{RELATIVE_ARM}] * 1 + [", 1)
    assert config != CONFIG
    finding = _finding(_check(tmp_path, config=config), "groot.relative_config")
    assert finding["status"] == "unknown" and "not a literal list" in finding["message"]


def test_action_configs_as_dicts_and_positional_arguments_are_read(tmp_path):
    config = CONFIG.replace(RELATIVE_ARM, '{"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "single_arm"}')
    assert read_modality_config(_config(tmp_path, config))["action"]["relative"] == {"single_arm": "single_arm"}
    positional = CONFIG.replace(RELATIVE_ARM, "ActionConfig(ActionRepresentation.RELATIVE, ActionType.NON_EEF, "
                                              "ActionFormat.DEFAULT, 'gripper')")
    assert read_modality_config(_config(tmp_path, positional))["action"]["relative"] == {"single_arm": "gripper"}


# ---------------------------------------------------------------- F4 entries the config does not read

def test_faults_in_entries_the_config_does_not_read_are_notes(tmp_path):
    # Validator C9: an out-of-range state group and an undeclared camera the config never names; GR00T trained (N9).
    extra = _modality(state__extra={"start": 6, "end": 9}, video__side={"original_key": "observation.images.side"})
    report = _check(tmp_path, extra)
    assert report["counts"]["error"] == 0, [f for f in report["findings"] if f["status"] == "error"]
    notes = [f for f in report["findings"] if f["status"] == "pass" and "noted only" in f["message"]]
    assert {f["code"] for f in notes} == {"groot.state_slice", "groot.video_column"}
    without_config = check_groot_dataset(tmp_path / "ds")  # without a config every entry counts as read
    assert {"groot.state_slice", "groot.video_column"} <= _codes(without_config, "error")


# ---------------------------------------------------------------- language and annotations

def test_language_delta_indices_must_be_zero(tmp_path):
    # Validator C10: loader:274 asserts.
    config = CONFIG.replace('delta_indices=[0], modality_keys=["annotation', 'delta_indices=[0, 1], modality_keys=["annotation')
    finding = _finding(_check(tmp_path, config=config), "groot.language_delta")
    assert finding["status"] == "error" and ":274" in finding["message"]


def test_sub_task_language_needs_sub_tasks_in_episodes(tmp_path):
    # Validator C11: KeyError: 'sub_tasks' at loader:547.
    config = CONFIG.replace('modality_keys=["annotation.human.task_description"]', 'modality_keys=["sub_task"]')
    finding = _finding(_check(tmp_path, config=config), "groot.episodes")
    assert finding["status"] == "error" and "sub_tasks" in finding["message"] and ":543-553" in finding["message"]
    task = CONFIG.replace('modality_keys=["annotation.human.task_description"]', 'modality_keys=["task"]')
    assert _check(tmp_path / "b", config=task)["status"] == "pass"  # episodes rows carry tasks


def test_annotation_column_must_hold_task_indices(tmp_path):
    # Validator C12: TypeError: unhashable type: 'numpy.ndarray' at loader:383.
    vector = _modality(annotation={"human.task_description": {"original_key": "observation.state"}})
    finding = _finding(_check(tmp_path, vector), "groot.annotation_values")
    assert finding["status"] == "error" and ":382-384" in finding["message"]


def test_annotation_section_that_is_not_an_object(tmp_path):
    # Validator R2 used to raise AttributeError.
    finding = _finding(_check(tmp_path, _modality(annotation=["human.task_description"])), "groot.language_key")
    assert finding["status"] == "error"


def test_annotation_column_known_only_from_statistics_is_accepted(tmp_path):
    # kimtaey/robocasa_mg_gr00t_100: info.json omits the annotation column, stats.json (from the parquet) has it.
    entry = _modality(annotation={"human.task_description": {}})
    ds = _dataset(tmp_path / "ds", entry)
    stats = json.loads((ds / "meta" / "stats.json").read_text())
    (ds / "meta" / "stats.json").write_text(json.dumps({**stats, "annotation.human.task_description": _stats(1)}))
    assert _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.annotation_column")["status"] == "pass"
    (ds / "meta" / "stats.json").write_text(json.dumps(stats))  # metadata alone cannot decide (K4 recheck ii)
    assert _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.annotation_column")["status"] == "warning"


def test_task_lines_and_annotation_columns(tmp_path):
    ds = _dataset(tmp_path / "ds", _modality(annotation={"human.task_description": {"original_key": "task"}}))
    (ds / "meta" / "tasks.jsonl").write_text(json.dumps({"index": 0, "task": "x"}) + "\n")
    finding = _finding(check_groot_dataset(ds), "groot.tasks")
    assert finding["status"] == "error" and ":171" in finding["message"]
    assert "groot.annotation_column" in _codes(check_groot_dataset(ds), "unknown")  # used or not depends on the config
    assert "groot.annotation_column" in _codes(check_groot_dataset(ds, _config(tmp_path)), "warning")


def test_unused_annotation_without_a_column_is_not_a_fault(tmp_path):
    # NVIDIA's simplerenv_bridge_sample declares "human.validity": {} with no such column; GR00T never reads it.
    ds = _dataset(tmp_path / "ds", _modality(annotation={"human.task_description": {"original_key": "task_index"},
                                                         "human.validity": {}}))
    assert check_groot_dataset(ds, _config(tmp_path))["status"] == "pass"


# ---------------------------------------------------------------- F6 metadata lines and robustness

@pytest.mark.parametrize("text, problem", [("\n" + json.dumps({"task_index": 0, "task": "x"}) + "\n", "blank line"),
                                           ('{"task_index": 0, "task": "x"\n', "not JSON")])
def test_tasks_lines_groot_cannot_parse(tmp_path, text, problem):
    # Validator C14 (blank line) and R1 (malformed line): JSONDecodeError at loader:170.
    ds = _dataset(tmp_path / "ds")
    (ds / "meta" / "tasks.jsonl").write_text(text)
    finding = _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.tasks")
    assert finding["status"] == "error" and problem in finding["message"] and ":168-170" in finding["message"]


def test_episodes_need_a_length(tmp_path):
    # Validator C15: KeyError: 'length' at loader:213, when GR00T opens the dataset.
    ds = _dataset(tmp_path / "ds")
    (ds / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["x"]}) + "\n")
    finding = _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.episodes")
    assert finding["status"] == "error" and "no length" in finding["message"] and ":211-213" in finding["message"]


def test_very_long_jsonl_is_checked_up_to_a_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(groot_contract, "MAX_LINES", 2)
    ds = _dataset(tmp_path / "ds")
    (ds / "meta" / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": "t"}) + "\n" for i in range(3)))
    assert _finding(check_groot_dataset(ds, _config(tmp_path)), "groot.tasks")["status"] == "unknown"


def test_malformed_entries_become_findings(tmp_path):
    # Validator R3 (original_key list, TypeError) and R4 (names shorter than the shape, IndexError).
    listed = _modality(state__single_arm={"start": 0, "end": 5, "original_key": ["observation.state"]})
    assert _finding(_check(tmp_path, listed), "groot.state_column")["status"] == "error"

    def short_names(info):
        info["features"]["observation.state"]["names"] = ["a", "b", "c"]
    report = check_groot_dataset(_dataset(tmp_path / "ds2", _modality(state__gripper=None), info_edit=short_names))
    assert "groot.state_unused" in _codes(report, "warning")


def test_info_json_fields_groot_reads(tmp_path):
    def broken(info):
        info.pop("chunks_size")
        info["features"]["timestamp"] = {"shape": [1]}
    report = _check(tmp_path, info_edit=broken)
    messages = [f["message"] for f in report["findings"] if f["code"] == "groot.metadata" and f["status"] == "error"]
    assert any(":198, 201" in m for m in messages) and any("stats.py:256" in m for m in messages)


def test_lerobot_v3_dataset_needs_conversion(tmp_path):
    report = check_groot_dataset(_dataset(tmp_path / "ds", version="v3.0"))
    finding = _finding(report, "groot.format")
    assert finding["status"] == "error" and "convert_v3_to_v2.py" in finding["remedy"]


def test_config_group_missing_from_modality_json(tmp_path):
    report = _check(tmp_path, _modality(action__gripper=None))
    finding = _finding(report, "groot.action_config")
    assert finding["status"] == "error" and "['gripper']" in finding["message"] and ":517" in finding["message"]


def test_overlapping_and_unused_values(tmp_path):
    overlap = _modality(state__gripper={"start": 4, "end": 6}, action__single_arm={"start": 0, "end": 4})
    warnings = _codes(check_groot_dataset(_dataset(tmp_path / "ds", overlap)), "warning")
    assert {"groot.state_overlap", "groot.action_unused"} <= warnings


def test_without_a_config_the_config_checks_are_unknown(tmp_path):
    report = check_groot_dataset(_dataset(tmp_path / "ds"))
    assert report["status"] == "unknown" and "groot.config" in _codes(report, "unknown")


# ---------------------------------------------------------------- containment

def test_metadata_symlinked_outside_the_dataset_is_refused(tmp_path):
    ds = _dataset(tmp_path / "ds")
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps(MODALITY))
    (ds / "meta" / "modality.json").unlink()
    (ds / "meta" / "modality.json").symlink_to(outside)
    with pytest.raises(InputError, match="escapes"):
        check_groot_dataset(ds)


def test_hugging_face_cache_snapshot_is_read(tmp_path):
    # Validator R15: snapshots/<rev>/meta/* are symlinks into the same repo cache's blobs/.
    source = _dataset(tmp_path / "plain")
    cache = tmp_path / "datasets--org--name"
    (cache / "blobs").mkdir(parents=True)
    snapshot = cache / "snapshots" / "abc123"
    (snapshot / "meta").mkdir(parents=True)
    for path in (source / "meta").iterdir():
        blob = cache / "blobs" / hashlib.sha256(path.read_bytes()).hexdigest()
        blob.write_bytes(path.read_bytes())
        os.symlink(os.path.relpath(blob, snapshot / "meta"), snapshot / "meta" / path.name)
    report = check_groot_dataset(snapshot, _config(tmp_path))
    assert report["status"] == "pass" and len(report["files"]) == 6
    # Any other escape is still refused, including a symlink into some other blobs/ directory.
    other = tmp_path / "other" / "blobs"
    other.mkdir(parents=True)
    (other / "x").write_text(json.dumps(MODALITY))
    (snapshot / "meta" / "modality.json").unlink()
    (snapshot / "meta" / "modality.json").symlink_to(other / "x")
    with pytest.raises(InputError, match="escapes"):
        check_groot_dataset(snapshot)


# ---------------------------------------------------------------- config parsing

def test_config_file_is_parsed_never_executed(tmp_path):
    marker = tmp_path / "executed"
    hostile = f"open({str(marker)!r}, 'w').write('x')\n" + CONFIG
    config = read_modality_config(_config(tmp_path, hostile))
    assert config["video"]["keys"] == ["front", "wrist"] and not marker.exists()
    with pytest.raises(InputError, match="literal list"):
        read_modality_config(_config(tmp_path, CONFIG.replace('modality_keys=["front", "wrist"]', "modality_keys=KEYS")))
    constant = "CAMERAS = ['front', 'wrist']\n" + CONFIG.replace('modality_keys=["front", "wrist"]', "modality_keys=CAMERAS")
    assert read_modality_config(_config(tmp_path, constant))["video"]["keys"] == ["front", "wrist"]  # validator R11
    with pytest.raises(InputError, match="literal list"):  # a constant changed later is not a constant
        read_modality_config(_config(tmp_path, constant + "CAMERAS.append('top')\n"))


def test_unparseable_config_files_are_refused(tmp_path):
    # Validator R7b (syntax error) and R8 (not UTF-8): InputError, not SyntaxError / UnicodeDecodeError.
    with pytest.raises(InputError, match="Cannot parse"):
        read_modality_config(_config(tmp_path, CONFIG + "\ndef (:\n"))
    path = tmp_path / "latin1.py"
    path.write_bytes(CONFIG.encode() + b'\nNOTE = "\xff\xfe"\n')
    with pytest.raises(InputError, match="Cannot parse"):
        read_modality_config(path)
    # Validator R8 put the bytes in a comment; Python's own compile() (how GR00T imports a config) accepts that,
    # and so does the ast parse of the same bytes.
    path.write_bytes(CONFIG.encode() + b"\n# \xff\xfe\n")
    compile(path.read_bytes(), str(path), "exec")
    assert read_modality_config(path)["video"]["keys"] == ["front", "wrist"]


def test_config_name_selects_one_of_several_dicts(tmp_path):
    # gr00t/configs/data/embodiment_configs.py keeps every built-in config in MODALITY_CONFIGS (validator: "found 8").
    body = CONFIG.split("so100_config = ", 1)[1].rsplit("\nregister_modality_config", 1)[0]
    text = ("from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig\n"
            f"MODALITY_CONFIGS = {{\n\"so100\": {body},\n\"other\": {body.replace('front', 'top')},\n}}\n")
    path = _config(tmp_path, text, "embodiment_configs.py")
    with pytest.raises(InputError, match=r"2 modality config dicts \['other', 'so100'\].*config_name"):
        read_modality_config(path)
    with pytest.raises(InputError, match="named 'missing'"):
        read_modality_config(path, config_name="missing")
    assert read_modality_config(path, config_name="other")["video"]["keys"] == ["top", "wrist"]
    report = check_groot_dataset(_dataset(tmp_path / "ds"), path, config_name="so100")
    assert report["status"] == "pass" and report["modality_config_name"] == "so100"
    unregistered = CONFIG.replace("so100_config", "so100_config_b").rsplit("\nregister_modality_config", 1)[0]
    two = _config(tmp_path, CONFIG + unregistered, "two.py")
    assert read_modality_config(two)["video"]["keys"] == ["front", "wrist"]  # the registered one


def test_oversized_config_file_is_refused(tmp_path):
    with pytest.raises(InputError, match="no larger than"):
        read_modality_config(_config(tmp_path, CONFIG + "#" * (1024 * 1024)))


def test_snapshot_subfolder_is_read_but_a_symlinked_blobs_directory_is_not(tmp_path):
    # K4 recheck N1/N2: a dataset in a subfolder of a cache snapshot is read; a blobs/ that is itself a symlink is not.
    source = _dataset(tmp_path / "plain")
    cache = tmp_path / "datasets--org--name"
    (cache / "blobs").mkdir(parents=True)
    snapshot = cache / "snapshots" / "abc123" / "sub"
    (snapshot / "meta").mkdir(parents=True)
    for path in (source / "meta").iterdir():
        blob = cache / "blobs" / hashlib.sha256(path.read_bytes()).hexdigest()
        blob.write_bytes(path.read_bytes())
        os.symlink(os.path.relpath(blob, snapshot / "meta"), snapshot / "meta" / path.name)
    assert check_groot_dataset(snapshot, _config(tmp_path))["status"] == "pass"
    outside = tmp_path / "outside"
    (cache / "blobs").rename(outside)
    (cache / "blobs").symlink_to(outside)
    with pytest.raises(InputError, match="escapes"):
        check_groot_dataset(snapshot)


def test_keyword_registration_selects_the_registered_dict(tmp_path):
    unregistered = CONFIG.replace("so100_config", "so100_config_b").rsplit("\nregister_modality_config", 1)[0]
    text = (CONFIG + unregistered).replace("register_modality_config(so100_config,", "register_modality_config(config=so100_config,")
    assert "config=so100_config" in text
    assert read_modality_config(_config(tmp_path, text, "kw.py"))["video"]["keys"] == ["front", "wrist"]


def test_annotation_column_missing_from_metadata_is_a_warning(tmp_path):
    # K4 recheck (ii): GR00T reads the parquet directly; metadata alone cannot decide.
    ds = _dataset(tmp_path / "ds", _modality(annotation={"human.task_description": {"original_key": "lang_idx"}}))
    finding = next(f for f in check_groot_dataset(ds, _config(tmp_path))["findings"] if f["code"] == "groot.annotation_column")
    assert finding["status"] == "warning"
