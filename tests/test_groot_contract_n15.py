"""GR00T dataset contract, Isaac-GR00T N1.5 profile (groot_version="n1.5", pinned 4af2b62).

Every fault case below was run through N1.5's own LeRobotSingleDataset, on both the direct-loader path and the
fine-tuning path (the data config's transforms), on a copy of NVIDIA's cube_to_bowl_5 demo or the real
GR00T-Dateset gr1_arms_waist-PlateToBowl episode: .validation/k4-groot-n15/out/n15_native.json (case name in each
comment). Fixtures copy cube_to_bowl_5's metadata, which N1.5 reads with its built-in so100_dualcam config."""
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.groot_contract import COMBINED_MAX_BYTES, check_groot_dataset, check_groot_dataset_versions
from policy_doctor.groot_contract_n15 import read_data_config

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "video": {"front": {"original_key": "observation.images.front"}, "wrist": {"original_key": "observation.images.wrist"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
SIX = ("mean", "std", "min", "max", "q01", "q99")
# A user config the N1.5 way: a subclass of a built-in data config (examples/SimplerEnv/custom_data_config.py style).
CUSTOM = '''
from gr00t.experiment.data_config import So100DataConfig


class MyArmConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    language_keys = ["annotation.human.task_description"]
'''


def _stats(width, offset=0.0):
    return {name: [offset + i for i in range(width)] for name in SIX}


def _dataset(root, modality=MODALITY, info_edit=None, stats="default", episodes=None, tasks=None):
    meta = root / "meta"
    meta.mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [6], "names": NAMES}
    info = {"codebase_version": "v2.1", "fps": 30, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"action": dict(vector), "observation.state": dict(vector),
                         "task_index": {"dtype": "int64", "shape": [1]},
                         **{f"observation.images.{c}": {"dtype": "video", "shape": [480, 640, 3],
                                                        "names": ["height", "width", "channels"],
                                                        "info": {"video.channels": 3, "video.fps": 30}}
                            for c in ("front", "wrist")}}}
    if info_edit:
        info_edit(info)
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "modality.json").write_text(json.dumps(modality))
    (meta / "episodes.jsonl").write_text(episodes if episodes is not None else
                                         json.dumps({"episode_index": 0, "tasks": ["cube into yellow bowl"], "length": 3}) + "\n")
    (meta / "tasks.jsonl").write_text(tasks if tasks is not None else json.dumps({"task_index": 0, "task": "cube into yellow bowl"}) + "\n")
    if stats == "default":
        stats = {"action": _stats(6), "observation.state": _stats(6), "timestamp": _stats(1)}
    if isinstance(stats, str):
        (meta / "stats.json").write_text(stats)
    elif stats is not None:
        (meta / "stats.json").write_text(json.dumps(stats))
    return root


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


def _check(tmp_path, config_name="so100_dualcam", config=None, **kwargs):
    return check_groot_dataset(_dataset(tmp_path / "ds", **kwargs), config, config_name=config_name, groot_version="n1.5")


def _file(tmp_path, text, name="cfg.py"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _codes(report, status=None):
    return {f["code"] for f in report["findings"] if status is None or f["status"] == status}


def _finding(report, code, status=None):
    return next(f for f in report["findings"] if f["code"] == code and (status is None or f["status"] == status))


# ---------------------------------------------------------------- healthy layouts and the version switch

def test_demo_layout_with_the_builtin_config_passes(tmp_path):
    # Native real_cube_to_bowl_5: both paths ran; state mask 6, action mask 6.
    report = _check(tmp_path)
    assert report["status"] == "pass", [f for f in report["findings"] if f["status"] != "pass"]
    assert report["consumer"] == "Isaac-GR00T N1.5 4af2b62" and report["groot_version"] == "n1.5"
    assert report["modality_config_name"] == "so100_dualcam"
    assert report["modality_config"]["video"]["keys"] == ["video.front", "video.wrist"]
    assert report["modality_config"]["transform"]["max_state_dim"] == 64


def test_default_version_is_n17_and_unchanged(tmp_path):
    ds = _dataset(tmp_path / "ds")
    report = check_groot_dataset(ds)
    assert report["consumer"] == "Isaac-GR00T 944fe70" and report["groot_version"] == "n1.7"
    assert report == check_groot_dataset(ds, groot_version="n1.7")
    with pytest.raises(InputError, match="groot_version"):
        check_groot_dataset(ds, groot_version="n1.6")


def test_auto_version_is_decided_only_by_an_unambiguous_config(tmp_path):
    ds = _dataset(tmp_path / "ds")
    n15 = check_groot_dataset(ds, _file(tmp_path, CUSTOM), groot_version="auto")
    assert n15["groot_version"] == "n1.5" and n15["modality_config_name"] == "MyArmConfig"
    n17_text = ("from gr00t.data.types import ModalityConfig\nfrom gr00t.configs.data.embodiment_configs import "
                "register_modality_config\ncfg = {'state': ModalityConfig(delta_indices=[0], modality_keys=['single_arm']),"
                " 'action': ModalityConfig(delta_indices=[0], modality_keys=['single_arm'])}\nregister_modality_config(cfg)\n")
    n17 = _file(tmp_path, n17_text, "n17.py")
    assert check_groot_dataset(ds, n17, groot_version="auto") == check_groot_dataset(ds, n17)
    for unknown in (check_groot_dataset(ds, groot_version="auto"),
                    check_groot_dataset(ds, _file(tmp_path, CUSTOM + n17_text, "both.py"), groot_version="auto")):
        assert unknown["status"] == "unknown" and _codes(unknown) == {"groot.version"} and unknown["groot_version"] is None


def test_gr1_arms_waist_object_dtype_is_fine_on_n15(tmp_path):
    # Native real_gr1_platetobowl_as_published: N1.5 never reads info.json dtypes and trained (state mask 58);
    # the N1.7 profile reports these columns (its generate_stats skips non-float columns).
    def object_columns(info):
        info["features"]["observation.state"]["dtype"] = "object"
        info["features"]["action"]["dtype"] = "object"
    assert _check(tmp_path, info_edit=object_columns)["status"] == "pass"
    n17 = check_groot_dataset(tmp_path / "ds")
    assert "groot.state_column" in _codes(n17, "error")


# ---------------------------------------------------------------- slices

@pytest.mark.parametrize("modality", [_modality(state__gripper={"start": 5, "end": 7}),
                                      _modality(state__extra={"start": 6, "end": 9})])
def test_out_of_range_slices_are_loud_even_when_unused(tmp_path, modality):
    # Native used_slice_out_of_range and unused_slice_out_of_range: IndexError at dataset.py:388 when opened.
    finding = _finding(_check(tmp_path, modality=modality), "groot.state_slice", "error")
    assert "IndexError" in finding["message"] and "383-388" in finding["message"]
    assert finding["applies_to"] == "fine-tuning and direct loaders"


def test_negative_start_and_empty_slices_are_silent_errors(tmp_path):
    # Native negative_start: state.single_arm read [gripper, 5 joints] with no error; empty_slice: shape [1, 0].
    wrapped = _finding(_check(tmp_path, modality=_modality(state__single_arm={"start": -1, "end": 5})),
                       "groot.state_slice", "error")
    assert "wraps around" in wrapped["message"]
    empty = _finding(_check(tmp_path / "b", modality=_modality(state__gripper={"start": 6, "end": 6})),
                     "groot.state_slice", "error")
    assert "empty" in empty["message"]


def test_modality_json_is_validated_like_pydantic(tmp_path):
    # Native state_section_missing, rotation_type_invalid, start_as_string (accepted), schema probe.
    for bad in (_modality(state=None), _modality(state__single_arm={"start": 0, "end": 5, "rotation_type": "quat"}),
                _modality(video=None), _modality(state__gripper={"start": 5.5, "end": 6})):
        report = check_groot_dataset(_dataset(tmp_path / str(len(list(tmp_path.iterdir()))), modality=bad),
                                     config_name="so100_dualcam", groot_version="n1.5")
        assert _codes(report, "error") == {"groot.modality_schema"}
    assert _check(tmp_path / "ok", modality=_modality(state__gripper={"start": "5", "end": 6.0}))["status"] == "pass"


def test_null_original_key_and_unused_group_without_statistics(tmp_path):
    # Native original_key_null (KeyError: None at dataset.py:382) and unused_group_other_column (KeyError there).
    null = _check(tmp_path, modality=_modality(state__gripper={"start": 5, "end": 6, "original_key": None}))
    assert "groot.state_column" in _codes(null, "error")
    other = _check(tmp_path / "b", modality=_modality(state__extra={"start": 0, "end": 2, "original_key": "observation.velocity"}))
    assert any("observation.velocity" in f["message"] for f in other["findings"] if f["code"] == "groot.statistics"
               and f["status"] == "error")


# ---------------------------------------------------------------- cameras

def test_renamed_cameras_are_not_paired_by_position(tmp_path):
    # Native renamed_cameras: ValueError at dataset.py:503 (N1.7 would pair them by position).
    renamed = _modality(video={"cam_front": {"original_key": "observation.images.front"},
                               "cam_wrist": {"original_key": "observation.images.wrist"}})
    finding = _finding(_check(tmp_path, modality=renamed), "groot.video_mapping", "error")
    assert "by name only" in finding["message"]


def test_swapped_original_keys_are_silent_and_caught(tmp_path):
    # Native swapped_original_keys: video.front decoded observation.images.wrist, no error on either path.
    swapped = _modality(video={"front": {"original_key": "observation.images.wrist"},
                               "wrist": {"original_key": "observation.images.front"}})
    finding = _finding(_check(tmp_path, modality=swapped), "groot.video_mapping", "error")
    assert "Swapped" in finding["message"] and "video.front reads observation.images.wrist" in finding["message"]


def test_every_video_entry_is_read_when_the_dataset_opens(tmp_path):
    # Native video_without_original_key (KeyError 'front'), unused_video_undeclared, video_feature_without_info.
    no_key = _check(tmp_path, modality=_modality(video__front={}))
    assert "own name" in _finding(no_key, "groot.video_column", "error")["message"]
    unused = _check(tmp_path / "b", modality=_modality(video__side={"original_key": "observation.images.side"}))
    assert "whether or not the config reads" in _finding(unused, "groot.video_column", "error")["message"]

    def no_info(info):
        del info["features"]["observation.images.front"]["info"]
    assert "groot.video_column" in _codes(_check(tmp_path / "c", info_edit=no_info), "error")

    def lerobot_v20(info):  # names "channel" + video_info, as GR00T-Dateset gr1_arms_waist writes it
        feature = info["features"]["observation.images.front"]
        feature.update(names=["height", "width", "channel"], video_info={"video.fps": 20.0})
        del feature["info"]
    assert _check(tmp_path / "d", info_edit=lerobot_v20)["status"] == "pass"


def test_video_dtype_is_not_read(tmp_path):
    # Native video_dtype_image: both paths ran.
    def as_image(info):
        for camera in ("front", "wrist"):
            info["features"][f"observation.images.{camera}"]["dtype"] = "image"
    assert _check(tmp_path, info_edit=as_image)["status"] == "pass"


# ---------------------------------------------------------------- statistics

@pytest.mark.parametrize("stats, status", [
    ("{", "error"),                                                     # native stats_invalid_json
    ("", "error"),                                                      # native stats_empty
    ("[]", "error"),                                                    # native stats_json_list
    (None, "pass"),                                                     # native stats_missing: regenerated
    ({"action": {n: [0.0] * 6 for n in ("mean", "std", "min", "max")},  # native stats_lerobot_v20_style: regenerated
      "observation.state": {n: [0.0] * 6 for n in ("mean", "std", "min", "max")}}, "pass"),
    ({"action": _stats(6), "observation.state": _stats(6),              # N1.7 fingerprints fail validation
      "__fingerprints__": {"action": "sha256:00"}}, "pass"),
])
def test_statistics_file_states(tmp_path, stats, status):
    finding = _finding(_check(tmp_path, stats=stats), "groot.statistics")
    assert finding["status"] == status


def test_recomputing_statistics_over_image_columns_is_a_warning(tmp_path):
    # Native stats_regenerated_over_image_column: TypeError at dataset.py:84 when no stats.json is kept and the parquet
    # files hold a LeRobot image column. Metadata cannot tell whether the column is in the parquet files.
    def image(info):
        info["features"]["observation.images.top"] = {"dtype": "image", "shape": [8, 8, 3]}
    report = _check(tmp_path, stats=None, info_edit=image)
    assert "TypeError" in _finding(report, "groot.statistics", "warning")["message"]
    assert "groot.statistics" not in _codes(_check(tmp_path / "b", info_edit=image), "warning")  # stats.json kept


def test_kept_statistics_that_do_not_fit_are_errors(tmp_path):
    # Native stats_narrow_width5 / stats_with_count_field (IndexError, dataset.py:388), stats_without_action_entry
    # (KeyError, :382), stats_wide_width7_other_robot (kept; gripper normalized with another robot's values).
    narrow = _check(tmp_path / "a", stats={"action": _stats(5), "observation.state": _stats(5)})
    assert "IndexError" in _finding(narrow, "groot.statistics", "error")["message"]
    count = _check(tmp_path / "b", stats={c: {**_stats(6), "count": [4148]} for c in ("action", "observation.state")})
    assert "count" in _finding(count, "groot.statistics", "error")["message"]
    no_action = _check(tmp_path / "c", stats={"observation.state": _stats(6)})
    assert "KeyError" in _finding(no_action, "groot.statistics", "error")["message"]
    wide = _finding(_check(tmp_path / "d", stats={"action": _stats(7, 100.0), "observation.state": _stats(7, 100.0)}),
                    "groot.statistics", "error")
    assert wide["applies_to"] == "fine-tuning" and "no error" in wide["message"]


# ---------------------------------------------------------------- language, jsonl, info.json

def test_language_rules(tmp_path):
    # Native two_language_keys (fine-tuning only), task_language_key, language_key_not_in_modality,
    # annotation_vector_column, annotation_column_absent.
    two = CUSTOM.replace('language_keys = ["annotation.human.task_description"]',
                         'language_keys = ["annotation.human.task_description", "annotation.human.alias"]')
    report = _check(tmp_path / "a", config_name=None, config=_file(tmp_path, two),
                    modality=_modality(annotation={"human.task_description": {"original_key": "task_index"},
                                                   "human.alias": {"original_key": "task_index"}}))
    assert _finding(report, "groot.language_key", "error")["applies_to"] == "fine-tuning"
    task = _check(tmp_path / "b", config_name=None,
                  config=_file(tmp_path, CUSTOM.replace('"annotation.human.task_description"]', '"task"]'), "t.py"))
    assert "annotation." in _finding(task, "groot.language_key", "error")["message"]
    missing = _check(tmp_path / "c", modality=_modality(annotation={"human.other": {"original_key": "task_index"}}))
    assert "groot.language_key" in _codes(missing, "error")
    vector = _check(tmp_path / "d", modality=_modality(annotation={"human.task_description": {"original_key": "observation.state"}}))
    assert ".item()" in _finding(vector, "groot.annotation_values", "error")["message"]
    absent = _check(tmp_path / "e", modality=_modality(annotation={"human.task_description": {"original_key": "lang_idx"}}))
    assert _finding(absent, "groot.annotation_column")["status"] == "warning"


def test_jsonl_and_info_fields(tmp_path):
    # Native episodes_length_string (TypeError at :429), tasks_without_task_index (:486), info_without_video_path.
    line = json.dumps({"episode_index": 0, "length": "3"}) + "\n"
    assert "length is not an integer" in _finding(_check(tmp_path / "a", episodes=line), "groot.episodes")["message"]
    tasks = json.dumps({"index": 0, "task": "x"}) + "\n"
    assert _finding(_check(tmp_path / "b", tasks=tasks), "groot.tasks")["status"] == "error"

    def no_video_path(info):
        del info["video_path"]
    assert ":474" in _finding(_check(tmp_path / "c", info_edit=no_video_path), "groot.metadata")["message"]


# ---------------------------------------------------------------- data config parsing and fine-tuning transforms

def test_builtin_custom_and_map_configs_are_read_without_executing(tmp_path):
    marker = tmp_path / "executed"
    hostile = f"open({str(marker)!r}, 'w').write('x')\n" + CUSTOM
    config = read_data_config(_file(tmp_path, hostile))
    assert not marker.exists() and config["class"] == "MyArmConfig"
    assert config["state"]["keys"] == ["state.single_arm", "state.gripper"]  # inherited from So100DataConfig
    assert config["action"]["delta_indices"] == list(range(16))
    assert config["transform"]["normalization"]["state.gripper"] == "min_max"  # So100's transform, self.state_keys
    assert read_data_config(None, "fourier_gr1_arms_waist")["transform"]["sincos"] == [
        "state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"]  # via super()
    mapped = _file(tmp_path, CUSTOM + "\nDATA_CONFIG_MAP = {'mine': MyArmConfig()}\n", "map.py")
    assert read_data_config(mapped, "mine")["class"] == "MyArmConfig"
    assert read_data_config(mapped, "pkg.mod:MyArmConfig")["data_config"] == "mine"


def test_config_forms_that_cannot_be_read(tmp_path):
    two = _file(tmp_path, CUSTOM + CUSTOM.replace("MyArmConfig", "Other").split("\n", 2)[2], "two.py")
    with pytest.raises(InputError, match="pass config_name"):
        read_data_config(two)
    with pytest.raises(InputError, match="no data config 'nope'"):
        read_data_config(two, "nope")
    dynamic = CUSTOM.replace('video_keys = ["video.front", "video.wrist"]', 'video_keys = [f"video.{c}" for c in "ab"]')
    report = _check(tmp_path, config_name=None, config=_file(tmp_path, dynamic, "d.py"))
    assert "not a literal list" in _finding(report, "groot.config", "unknown")["message"]
    override = CUSTOM + "\n    def modality_config(self):\n        return {}\n"
    assert "overrides modality_config" in _finding(
        _check(tmp_path / "b", config_name=None, config=_file(tmp_path, override, "o.py")), "groot.config", "unknown")["message"]
    unknown_base = "from somewhere import Base\n\nclass X(Base):\n    video_keys = []\n"
    with pytest.raises(InputError, match="0 N1.5 data config classes"):
        read_data_config(_file(tmp_path, unknown_base, "u.py"))
    bare = "from gr00t.experiment.data_config import BaseDataConfig\n\nclass X(BaseDataConfig):\n    video_keys = []\n"
    assert "AttributeError" in _finding(_check(tmp_path / "c", config_name=None, config=_file(tmp_path, bare, "b.py")),
                                        "groot.config", "error")["message"]


def test_fine_tuning_transform_rules(tmp_path):
    # Native gripper_dtype_int_min_max, rotation_without_rotation_type, mixed_camera_resolution, dotted_state_key
    # and unprefixed_state_keys; all failed on the fine-tuning path only (the last also when the dataset opened).
    def check(name, config=None, **kwargs):
        return check_groot_dataset(_dataset(tmp_path / name, **kwargs), config, config_name=None if config else "so100_dualcam",
                                   groot_version="n1.5")
    assert _finding(check("int", modality=_modality(state__gripper={"start": 5, "end": 6, "dtype": "int64"})),
                    "groot.normalization", "error")["applies_to"] == "fine-tuning"
    rotation = CUSTOM.replace("class MyArmConfig(So100DataConfig):", "class MyArmConfig(So100DataConfig):\n"
                              "    def transform(self):\n        return [StateActionTransform(apply_to=self.state_keys, "
                              "target_rotations={'state.single_arm': 'rotation_6d'}), ConcatTransform(), "
                              "GR00TTransform(max_state_dim=64, max_action_dim=32)]\n")
    assert "groot.rotation" in _codes(check("rot", _file(tmp_path, rotation, "r.py")), "error")

    def wrist_smaller(info):
        info["features"]["observation.images.wrist"]["shape"] = [240, 320, 3]
    assert "groot.video_resolution" in _codes(check("res", info_edit=wrist_smaller), "error")
    dotted = CUSTOM.replace('language_keys', 'state_keys = ["state.single.arm", "state.gripper"]\n    language_keys')
    report = check("dot", _file(tmp_path, dotted, "dot.py"),
                   modality=_modality(state={"single.arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}}))
    assert _finding(report, "groot.config_key", "error")["applies_to"] == "fine-tuning"
    bare = CUSTOM.replace('language_keys', 'state_keys = ["single_arm", "gripper"]\n    language_keys')
    assert "groot.config_key" in _codes(check("bare", _file(tmp_path, bare, "bare.py")), "error")


def test_model_width_limits(tmp_path):
    # Native gr1_all_state_sincos_88_dims: state mask 64 of 88 (silent); gr1_all_actions_44_dims: AssertionError.
    names = [f"j{i}" for i in range(44)]
    groups = {"a": (0, 14), "b": (14, 28), "c": (28, 44)}
    modality = {side: {g: {"start": s, "end": e} for g, (s, e) in groups.items()} for side in ("state", "action")}
    modality.update(video={"ego_view": {"original_key": "observation.images.front"}},
                    annotation={"human.task_description": {"original_key": "task_index"}})

    def wide(info):
        for column in ("observation.state", "action"):
            info["features"][column].update(shape=[44], names=names)
    sincos = ('from gr00t.experiment.data_config import FourierGr1ArmsOnlyDataConfig\n\n'
              'class Wide(FourierGr1ArmsOnlyDataConfig):\n'
              '    state_keys = ["state.a", "state.b", "state.c"]\n    action_keys = ["action.a"]\n'
              '    language_keys = ["annotation.human.task_description"]\n')
    stats = {"action": _stats(44), "observation.state": _stats(44)}
    report = check_groot_dataset(_dataset(tmp_path / "s", modality=modality, info_edit=wide, stats=stats),
                                 _file(tmp_path, sincos), groot_version="n1.5")
    finding = _finding(report, "groot.state_width", "error")
    assert "88" in finding["message"] and "first 64" in finding["message"]
    actions = sincos.replace('action_keys = ["action.a"]', 'action_keys = ["action.a", "action.b", "action.c"]')
    report = check_groot_dataset(_dataset(tmp_path / "a", modality=modality, info_edit=wide, stats=stats),
                                 _file(tmp_path, actions, "a.py"), groot_version="n1.5")
    assert "44" in _finding(report, "groot.action_width", "error")["message"]


def test_builtin_source_matches_nvidia_classes_verbatim(tmp_path):
    # So100DataConfig / So100DualCamDataConfig copied from gr00t/experiment/data_config.py:211-271 (4af2b62), with the
    # imports; the validation harness compares all twelve DATA_CONFIG_MAP entries against the pinned file.
    verbatim = '''
from gr00t.experiment.data_config import BaseDataConfig


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
'''
    parsed = read_data_config(_file(tmp_path, verbatim), "So100DualCamDataConfig")
    builtin = read_data_config(None, "so100_dualcam")
    assert builtin.pop("data_config") == "so100_dualcam" and parsed.pop("data_config") is None
    assert parsed == builtin


# ---------------------------------------------------------------- validator round 2 (k4-groot-n15/validator VERDICT.md)
# Each repro below is the validator's own config or plant (validator/configs/v15_configs.py, v15_forms.py,
# v15_plain.py, v15_notebook.py; plants in validator/tools/v15_native.py), run natively in
# validator/out/j2/native_plants.json; case names in the comments.

HEAD = '''
from dataclasses import dataclass, field
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import BaseDataConfig, BimanualPandaGripperDataConfig, So100DualCamDataConfig
from gr00t.model.transforms import GR00TTransform
'''
STATE_TF = "StateActionTransform(apply_to=self.state_keys, normalization_modes={key: 'min_max' for key in self.state_keys})"
ACTION_TF = "StateActionTransform(apply_to=self.action_keys, normalization_modes={key: 'min_max' for key in self.action_keys})"
CONCAT_TF = ("ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys, "
             "action_concat_order=self.action_keys)")
GROOT_TF = "GR00TTransform(state_horizon=1, action_horizon=16, max_state_dim=64, max_action_dim=32)"
SO100_ATTRS = '''    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
'''


def _transform(state=STATE_TF, action=ACTION_TF, concat=CONCAT_TF, groot=GROOT_TF, indent="    "):
    """The validator's so100 transform() (v15_configs.py), with one call replaced."""
    body = ["VideoToTensor(apply_to=self.video_keys)", "VideoCrop(apply_to=self.video_keys, scale=0.95)",
            "VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear')",
            "VideoToNumpy(apply_to=self.video_keys)", "StateActionToTensor(apply_to=self.state_keys)", state,
            "StateActionToTensor(apply_to=self.action_keys)", action, concat, groot]
    calls = "".join(f"{indent}        {call},\n" for call in body)
    return (f"{indent}def transform(self):\n{indent}    transforms = [\n{calls}{indent}    ]\n"
            f"{indent}    return ComposedModalityTransform(transforms=transforms)\n")


def _config(tmp_path, body, name="v.py"):
    return _file(tmp_path, HEAD + body, name)


def _run(tmp_path, body, name="Cfg", **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = _config(tmp_path, body, f"{name}.py")
    return check_groot_dataset(_dataset(tmp_path / f"ds_{name}", **kwargs), path, config_name=name,
                               groot_version="n1.5")


def test_attributes_set_on_the_instance_are_unknown_not_errors(tmp_path):
    # Native c2_init_attrs, c2_init_attrs_plain, c2_property_attrs, c2_property_override, form_dataclass_fields:
    # all open and train on N1.5; the checker must not report AttributeError for them.
    init = "class Cfg(BaseDataConfig):\n    def __init__(self):\n" + "".join(
        f"        self.{line.strip()}\n" for line in SO100_ATTRS.strip().splitlines()) + "\n" + _transform()
    prop = "class Cfg(So100DualCamDataConfig):\n" + "".join(
        f"    @property\n    def {line.split(' = ')[0].strip()}(self):\n        return {line.split(' = ')[1]}\n"
        for line in SO100_ATTRS.strip().splitlines())
    override = "class Cfg(So100DualCamDataConfig):\n    @property\n    def video_keys(self):\n        return ['video.front']\n"
    dataclass = "@dataclass\nclass Cfg(BaseDataConfig):\n" + "".join(
        f"    {line.split(' = ')[0].strip()}: list = field(default_factory=lambda: {line.split(' = ')[1]})\n"
        for line in SO100_ATTRS.strip().splitlines()) + _transform()
    hooked = "class Cfg(BaseDataConfig):\n    def __init__(self):\n        self._setup()\n\n" + _transform()
    for label, body in (("init", init), ("prop", prop), ("override", override), ("dataclass", dataclass),
                        ("hooked", hooked)):
        report = _run(tmp_path / label, body)
        assert report["status"] == "unknown", (label, [f for f in report["findings"] if f["status"] == "error"])
        assert _codes(report, "error") == set() and "groot.config" in _codes(report, "unknown"), label
    # with no __init__ (or other hook) anywhere in the chain a missing attribute is still the AttributeError
    bare = _run(tmp_path / "bare", "class Cfg(BaseDataConfig):\n    video_keys = []\n" + _transform())
    assert "AttributeError" in _finding(bare, "groot.config", "error")["message"]


def test_a_class_attribute_assigned_twice_is_read_from_the_last_assignment(tmp_path):
    # Native c7_dup_attr_good_last (opens) and c7_dup_attr_bad_last (ValueError, video.webcam, dataset.py:503).
    good = "class Cfg(So100DualCamDataConfig):\n    video_keys = ['video.webcam']\n    video_keys = ['video.front', 'video.wrist']\n"
    assert _run(tmp_path / "good", good)["status"] == "pass"
    bad = "class Cfg(So100DualCamDataConfig):\n    video_keys = ['video.front', 'video.wrist']\n    video_keys = ['video.webcam']\n"
    assert "video.webcam" in _finding(_run(tmp_path / "bad", bad), "groot.video_mapping", "error")["message"]
    for label, body in (("tuple", "    video_keys, extra = ['video.front', 'video.wrist'], 1\n"),
                        ("aug", "    video_keys = ['video.front']\n    video_keys += ['video.wrist']\n"),
                        ("branch", "    if True:\n        video_keys = ['video.webcam']\n")):
        report = _run(tmp_path / label, "class Cfg(So100DualCamDataConfig):\n" + body)
        assert report["status"] == "unknown" and not _codes(report, "error"), label


ROT_MODALITY = _modality(state={"rot": {"start": 0, "end": 3, "rotation_type": "axis_angle", "absolute": False},
                                "rest": {"start": 3, "end": 6}})
POS_ROT = _modality(state={"pos": {"start": 0, "end": 3}, "rot": {"start": 3, "end": 6, "rotation_type": "axis_angle"}})


def test_rotation_rules_follow_set_metadata(tmp_path):
    rot_attrs = SO100_ATTRS.replace('["state.single_arm", "state.gripper"]', '["state.rot", "state.rest"]')

    def rel(target):
        state = ("StateActionTransform(apply_to=self.state_keys, normalization_modes={'state.rot': 'min_max', "
                 f"'state.rest': 'min_max'}}, target_rotations={{'state.rot': '{target}'}})")
        return "class Cfg(BaseDataConfig):\n" + rot_attrs + _transform(state=state)
    # Native c13_rel_rot_same_rep: target == rotation_type builds no rotation transformer (state_action.py:428).
    assert _run(tmp_path / "same", rel("axis_angle"), modality=ROT_MODALITY)["status"] == "pass"
    relative = _finding(_run(tmp_path / "conv", rel("rotation_6d"), modality=ROT_MODALITY), "groot.rotation", "error")
    assert "relative" in relative["message"] and "440-457" in relative["message"]

    pos_attrs = SO100_ATTRS.replace('["state.single_arm", "state.gripper"]', '["state.pos", "state.rot"]')

    def absolute(mode):
        state = ("StateActionTransform(apply_to=self.state_keys, normalization_modes={'state.pos': 'min_max', "
                 f"'state.rot': '{mode}'}}, target_rotations={{'state.rot': 'rotation_6d'}})")
        return "class Cfg(BaseDataConfig):\n" + pos_attrs + _transform(state=state)
    # Native c4_abs_rot_mean_std: AssertionError at state_action.py:445; c4_control_abs_rot_min_max opens.
    assert "min_max" in _finding(_run(tmp_path / "ms", absolute("mean_std"), modality=POS_ROT), "groot.rotation",
                                 "error")["message"]
    assert _run(tmp_path / "mm", absolute("min_max"), modality=POS_ROT)["status"] == "pass"


def test_auto_needs_n17_style_keys_and_every_report_names_its_version(tmp_path):
    # validator v15_plain.py (accepted by N1.5's import_external_data_config) and v15_notebook.py (N1.5 notebook
    # form): both have N1.5's prefixed keys, so "auto" must not choose N1.7 (validator D4: 2 false errors).
    keys = ('{"video": ModalityConfig(delta_indices=[0], modality_keys=["video.front", "video.wrist"]),\n'
            ' "state": ModalityConfig(delta_indices=[0], modality_keys=["state.single_arm", "state.gripper"]),\n'
            ' "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=["action.single_arm", '
            '"action.gripper"]),\n "language": ModalityConfig(delta_indices=[0], '
            'modality_keys=["annotation.human.task_description"])}')
    plain = (f"from gr00t.data.dataset import ModalityConfig\n\nclass Plain:\n    def modality_config(self):\n"
             f"        return {keys}\n\n    def transform(self):\n        return None\n")
    notebook = f"from gr00t.data.dataset import ModalityConfig\nmodality_configs = {keys}\n"
    ds = _dataset(tmp_path / "ds")
    for name, text in (("plain.py", plain), ("notebook.py", notebook)):
        report = check_groot_dataset(ds, _file(tmp_path, text, name), groot_version="auto")
        assert report["status"] == "unknown" and report["groot_version"] is None and _codes(report) == {"groot.version"}
    bare = notebook.replace('"video.', '"').replace('"state.', '"').replace('"action.', '"')
    assert check_groot_dataset(ds, _file(tmp_path, bare, "n17.py"), groot_version="auto")["groot_version"] == "n1.7"
    mixed = notebook.replace('"video.front"', '"front"')
    assert check_groot_dataset(ds, _file(tmp_path, mixed, "m.py"), groot_version="auto")["groot_version"] is None
    assert check_groot_dataset(ds)["groot_version"] == "n1.7" and _check(tmp_path / "b")["groot_version"] == "n1.5"


def test_module_class_without_its_file_is_refused(tmp_path):
    # Validator D5 (configs/v15_shadow.py): N1.5 imports the user's module, whose class may shadow a built-in name.
    with pytest.raises(InputError, match="pass that module's .py file"):
        read_data_config(None, "my_configs:So100DualCamDataConfig")
    builtin = read_data_config(None, "gr00t.experiment.data_config:So100DualCamDataConfig")
    assert builtin["video"]["keys"] == ["video.front", "video.wrist"]
    shadow = _file(tmp_path, "from gr00t.experiment.data_config import So100DataConfig\n\n"
                   "class So100DualCamDataConfig(So100DataConfig):\n    video_keys = ['video.webcam', 'video.wrist']\n")
    assert read_data_config(shadow, "my_configs:So100DualCamDataConfig")["video"]["keys"] == ["video.webcam", "video.wrist"]


def test_transform_delegating_to_a_helper_is_unknown(tmp_path):
    # Native demo_gr1_all_state_via_helper: the helper's 88 sin-cos values were truncated to 64 (state mask 64).
    body = ("def build(config):\n    return ComposedModalityTransform(transforms=[GR00TTransform(state_horizon=1, "
            "action_horizon=16, max_state_dim=4, max_action_dim=32)])\n\n\nclass Cfg(So100DualCamDataConfig):\n"
            "    def transform(self):\n        return build(self)\n")
    report = _run(tmp_path, body)
    assert "helper" in _finding(report, "groot.config", "unknown")["message"] and not _codes(report, "error")
    assert report["modality_config"]["transform"]["max_state_dim"] is None


@pytest.mark.parametrize("entry, used, error", [
    ({"start": -1, "end": 7}, False, True),   # native c1_unused_negstart_past_width: IndexError at dataset.py:388
    ({"start": -1, "end": 3}, False, False),  # native c1_control_unused_negstart_in_range: opens
    ({"start": -1, "end": 7}, True, True),    # native c1_used_negstart_past_width: IndexError at dataset.py:388
    ({"start": -7, "end": 2}, False, True),   # np.arange(-7, 2) on 6 values: index -7 is out of bounds
])
def test_negative_start_past_the_width_is_loud_whether_or_not_read(tmp_path, entry, used, error):
    report = _check(tmp_path, modality=_modality(**{"state__gripper" if used else "state__extra": entry}))
    slices = [f for f in report["findings"] if f["code"] == "groot.state_slice" and f["status"] == "error"]
    assert bool(slices) == error
    for finding in slices:
        assert "IndexError" in finding["message"] and "with no error" not in finding["message"]
        assert finding["applies_to"] == "fine-tuning and direct loaders"
    if error:
        assert "groot.statistics" in _codes(report, "error")  # the kept stats.json is indexed the same way


def test_loud_faults_of_the_transform_keys(tmp_path):
    # Native c3_inherited_norm_dict (AssertionError 'right_arm_eef_pos statistics not found', state_action.py:408).
    inherited = "class Cfg(BimanualPandaGripperDataConfig):\n" + SO100_ATTRS
    finding = _finding(_run(tmp_path / "a", inherited), "groot.transform_keys", "error")
    assert "state.right_arm_eef_pos" in finding["message"] and finding["applies_to"] == "fine-tuning"
    # Native c3_rotation_key_not_applied (KeyError 'state.extra_rot', :421).
    state = (STATE_TF[:-1] + ", target_rotations={'state.extra_rot': 'rotation_6d'})")
    extra = _modality(state__extra_rot={"start": 0, "end": 3, "rotation_type": "axis_angle"})
    report = _run(tmp_path / "b", "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state), modality=extra)
    assert "419-421" in _finding(report, "groot.transform_keys", "error")["message"]
    # Native c3c_norm_key_not_applied (KeyError 'state.gripper', :460).
    state = ("StateActionTransform(apply_to=['state.single_arm'], normalization_modes={'state.single_arm': 'min_max', "
             "'state.gripper': 'min_max'})")
    report = _run(tmp_path / "c", "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state))
    assert "458-460" in _finding(report, "groot.transform_keys", "error")["message"]
    # An invalid target representation: RotationType raises ValueError (:425), after the source rotation_type assert
    # (:422; native fx_bad_rotation_target, a slice without rotation_type, fails there).
    state = (STATE_TF[:-1] + ", target_rotations={'state.single_arm': 'rot6d'})")
    typed = _modality(state__single_arm={"start": 0, "end": 5, "rotation_type": "axis_angle"})
    report = _run(tmp_path / "d", "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state), modality=typed)
    assert "'rot6d'" in _finding(report, "groot.transform_keys", "error")["message"]
    report = _run(tmp_path / "e", "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state))
    assert "422" in _finding(report, "groot.rotation", "error")["message"] and "groot.transform_keys" not in _codes(report)
    # A config key missing from modality.json is reported once, as groot.state_config, not again per transform.
    report = _check(tmp_path / "f", modality=_modality(state__gripper=None))
    assert "groot.state_config" in _codes(report, "error") and "groot.transform_keys" not in _codes(report)


def test_bad_normalization_mode_and_horizons(tmp_path):
    # Native c6_bad_norm_mode: ValueError 'Invalid normalization mode: minmax' at the first sample.
    state = "StateActionTransform(apply_to=self.state_keys, normalization_modes={'state.single_arm': 'minmax', 'state.gripper': 'min_max'})"
    report = _run(tmp_path / "a", "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state))
    assert "'minmax'" in _finding(report, "groot.normalization", "error")["message"]
    # Native c14_horizon_mismatch: action_horizon=16 with 8 action_indices asserts (model/transforms.py:283).
    report = _run(tmp_path / "b", "class Cfg(So100DualCamDataConfig):\n    action_indices = list(range(8))\n\n"
                  + _transform())
    assert "action_horizon=16" in _finding(report, "groot.horizon", "error")["message"]
    # Native form_half_horizon: the built-in transform uses len(self.action_indices).
    assert _run(tmp_path / "c", "class Cfg(So100DualCamDataConfig):\n    action_indices = list(range(8))\n")["status"] == "pass"
    # A fault in a branch that may not run (Libero-style action_norm) is unknown, not an error.
    branchy = ("class Cfg(So100DualCamDataConfig):\n    def transform(self, norm='min_max'):\n"
               "        if norm == 'min_max':\n            action = StateActionTransform(apply_to=self.action_keys, "
               "normalization_modes={k: 'min_max' for k in self.action_keys})\n        else:\n"
               "            action = StateActionTransform(apply_to=self.action_keys, normalization_modes={k: 'minmax' "
               "for k in self.action_keys})\n" + _transform(action="action", indent="    ").split("\n", 1)[1])
    report = _run(tmp_path / "d", branchy)
    assert not _codes(report, "error") and "groot.normalization" in _codes(report, "unknown")


def test_null_video_info_and_repeated_episode_index(tmp_path):
    # Native c9_video_info_null: TypeError at dataset.py:348 on both paths.
    def null(info):
        info["features"]["observation.images.front"].update(names=["height", "width", "channel"], video_info=None)
    assert "TypeError" in _finding(_check(tmp_path / "a", info_edit=null), "groot.video_column", "error")["message"]
    # Native p_duplicate_episode_line: ValueError at dataset.py:602 when a sample of episode 0 is read.
    line = json.dumps({"episode_index": 0, "tasks": ["x"], "length": 3}) + "\n"
    finding = _finding(_check(tmp_path / "b", episodes=line + line), "groot.episodes", "error")
    assert "earlier line" in finding["message"] and finding["applies_to"] == "fine-tuning and direct loaders"
    # Native fx_episode_index_float_duplicate: a float id breaks data_path formatting ({episode_index:06d},
    # ValueError at dataset.py:584) when a sample is read; with a data_path that formats it otherwise it is not flagged.
    floats = line + json.dumps({"episode_index": 1.0, "tasks": ["x"], "length": 3}) + "\n"
    assert "not an integer" in _finding(_check(tmp_path / "c", episodes=floats), "groot.episodes", "error")["message"]

    def plain_path(info):
        info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index}.parquet"
    assert "groot.episodes" not in _codes(_check(tmp_path / "d", episodes=floats, info_edit=plain_path))


def test_concat_order_that_leaves_a_group_or_camera_out(tmp_path):
    # Native c12_concat_drops_gripper (state mask 5 of 6, no error) and c12_concat_drops_wrist (wrist decoded,
    # 1 image to the model, no error).
    def run(name, concat, **kwargs):
        return _run(tmp_path / name, "class Cfg(So100DualCamDataConfig):\n" + _transform(concat=concat), **kwargs)
    gripper = _finding(run("g", CONCAT_TF.replace("state_concat_order=self.state_keys",
                                                  "state_concat_order=['state.single_arm']")), "groot.concat", "error")
    assert "state.gripper" in gripper["message"] and "with no error" in gripper["message"]
    wrist = _finding(run("w", CONCAT_TF.replace("video_concat_order=self.video_keys",
                                                "video_concat_order=['video.front']")), "groot.concat", "error")
    assert "video.wrist" in wrist["message"] and "never reach the model" in wrist["message"]
    action = _finding(run("a", CONCAT_TF.replace("action_concat_order=self.action_keys",
                                                 "action_concat_order=['action.single_arm']")), "groot.concat", "error")
    assert "139-143" in action["message"]
    missing = _finding(run("m", "ConcatTransform(video_concat_order=self.video_keys, action_concat_order=self.action_keys)"),
                       "groot.concat", "error")
    assert "no state_concat_order" in missing["message"]
    assert run("ok", CONCAT_TF)["status"] == "pass"
    foreign = HEAD + "from mine import Rename\n\nclass Cfg(So100DualCamDataConfig):\n" + _transform(
        concat=CONCAT_TF.replace("state_concat_order=self.state_keys", "state_concat_order=['state.single_arm']")
    ).replace("        VideoToTensor", "        Rename(),\n        VideoToTensor", 1)
    report = check_groot_dataset(_dataset(tmp_path / "f"), _file(tmp_path, foreign, "f.py"), config_name="Cfg",
                                 groot_version="n1.5")
    assert not _codes(report, "error") and "groot.concat" in _codes(report, "unknown")


def test_reports_list_what_n15_does_not_check(tmp_path):
    with_config = _check(tmp_path / "a")["coverage"]["not_checked"]
    without = check_groot_dataset(_dataset(tmp_path / "b"), groot_version="n1.5")["coverage"]["not_checked"]
    assert any("rotation_type against the slice width" in item for item in with_config)
    assert "fine-tuning transforms" in without and "fine-tuning transforms" not in with_config


# ---------------------------------------------------------------- both versions, per dataset


def _object_dtype(info):  # GR00T-Dateset gr1_arms_waist: N1.5 trains on it, N1.7 fine-tuning raises KeyError
    info["features"]["observation.state"]["dtype"] = "object"
    info["features"]["action"]["dtype"] = "object"


def test_combined_names_the_one_failing_version(tmp_path):
    n17_only = check_groot_dataset_versions(_dataset(tmp_path / "a", info_edit=_object_dtype))
    assert n17_only["status"] == "error" and n17_only["fails_on"] == ["n1.7"] and n17_only["mode"] == "both"
    assert n17_only["headline"].startswith("Fails on Isaac-GR00T N1.7 fine-tuning (crash, if the config reads")
    assert "no error on Isaac-GR00T N1.5" in n17_only["headline"]
    assert "--groot-version n1.5" in n17_only["remedy"] and "--groot-version n1.7" in n17_only["remedy"]
    assert n17_only["versions"]["n1.5"]["fails"] is False and n17_only["versions"]["n1.7"]["errors_if_read"] == 2
    assert {f["groot_version"] for f in n17_only["findings"]} == {"n1.5", "n1.7"}
    # 120ft-style LeRobot count statistics: IndexError when N1.5 opens the dataset; N1.7 recomputes them.
    count = check_groot_dataset_versions(_dataset(tmp_path / "b", stats={c: {**_stats(6), "count": [4]}
                                                                         for c in ("action", "observation.state")}))
    assert count["fails_on"] == ["n1.5"] and count["headline"].startswith("Fails on Isaac-GR00T N1.5 (crash at open)")
    assert count["versions"]["n1.5"]["errors_certain"] >= 1 and count["status"] == "error"
    # Another robot's wider stats.json: silent on N1.5 fine-tuning, recomputed by N1.7.
    wide = check_groot_dataset_versions(_dataset(tmp_path / "c", stats={c: _stats(7, 100.0)
                                                                        for c in ("action", "observation.state")}))
    assert wide["headline"].startswith("Fails on Isaac-GR00T N1.5 fine-tuning (silently wrong, if the config reads")


def test_combined_both_none_explicit_and_detected(tmp_path):
    gone = _dataset(tmp_path / "b")
    (gone / "meta/modality.json").unlink()
    both = check_groot_dataset_versions(gone)
    assert both["status"] == "error" and both["fails_on"] == ["n1.5", "n1.7"]
    assert both["headline"].startswith("Fails on Isaac-GR00T N1.5 (crash at open) and on Isaac-GR00T N1.7")
    healthy = check_groot_dataset_versions(_dataset(tmp_path / "c"))
    assert healthy["fails_on"] == [] and healthy["status"] in ("unknown", "pass")
    assert healthy["headline"].startswith("No error on Isaac-GR00T N1.5 or Isaac-GR00T N1.7")
    ds = _dataset(tmp_path / "d", info_edit=_object_dtype)
    explicit = check_groot_dataset_versions(ds, groot_version="n1.5")
    assert explicit["versions_run"] == ["n1.5"] and explicit["status"] != "error" and explicit["groot_version"] == "n1.5"
    assert explicit["total_findings"] == len(check_groot_dataset(ds, groot_version="n1.5")["findings"])
    custom = check_groot_dataset_versions(ds, _file(tmp_path, CUSTOM))
    assert custom["mode"] == "detected" and custom["versions_run"] == ["n1.5"]
    assert check_groot_dataset_versions(ds, config_name="so100_dualcam")["versions_run"] == ["n1.5"]
    n17 = _file(tmp_path, "from gr00t.data.types import ModalityConfig\ncfg = {'state': ModalityConfig(delta_indices=[0], "
                "modality_keys=['single_arm', 'gripper']), 'action': ModalityConfig(delta_indices=[0], "
                "modality_keys=['single_arm', 'gripper'])}\n", "n17.py")
    assert check_groot_dataset_versions(ds, n17)["versions_run"] == ["n1.7"]
    for kwargs in ({"modality_config": _file(tmp_path, CUSTOM + "\nfrom gr00t.data.types import ModalityConfig\n"
                                             "c = {'state': ModalityConfig(modality_keys=['a'], delta_indices=[0])}\n",
                                             "both.py")},
                   {"config_name": "my_configs:So100DualCamDataConfig"}):
        undetermined = check_groot_dataset_versions(ds, **kwargs)
        assert undetermined["mode"] == "undetermined" and undetermined["status"] == "unknown"
        assert undetermined["versions_run"] == [] and "--groot-version" in undetermined["remedy"]
    with pytest.raises(InputError, match="groot_version"):
        check_groot_dataset_versions(ds, groot_version="auto")


def test_combined_findings_cap_keeps_every_version_and_code(tmp_path):
    # 150 unused groups past the width (an error per group on both versions), N1.7-only dtype errors, N1.5-only
    # count statistics: the cap must still keep each version's every error code.
    modality = _modality(state={**MODALITY["state"], **{f"g{i}": {"start": 6, "end": 9} for i in range(150)}})
    stats = {c: {**_stats(6), "count": [4]} for c in ("action", "observation.state")}
    report = check_groot_dataset_versions(_dataset(tmp_path / "ds", modality=modality, info_edit=_object_dtype,
                                                   stats=stats))
    every = {(v, f["code"]) for v in ("n1.5", "n1.7")
             for f in check_groot_dataset(tmp_path / "ds", groot_version=v)["findings"] if f["status"] == "error"}
    kept = {(f["groot_version"], f["code"]) for f in report["findings"] if f["status"] == "error"}
    assert every == kept and {v for v, _ in every} == {"n1.5", "n1.7"}
    assert report["findings_truncated"] and report["total_findings"] > len(report["findings"])
    assert len(json.dumps(report["findings"])) <= COMBINED_MAX_BYTES
    assert report["counts"]["error"] == sum(1 for v in ("n1.5", "n1.7") for f in check_groot_dataset(
        tmp_path / "ds", groot_version=v)["findings"] if f["status"] == "error")


def test_combined_refuses_inputs_it_cannot_scope(tmp_path):
    ds = _dataset(tmp_path / "ds")
    with pytest.raises(InputError, match="N1.7 modality config"):
        check_groot_dataset_versions(ds, config_name="so100_config", groot_version="n1.7")
    with pytest.raises(InputError, match="Dataset directory not found"):
        check_groot_dataset_versions(tmp_path / "missing")
    # an N1.5 config that cannot be read statically leaves the N1.5 rules config-less, and the result says so
    unread = _file(tmp_path, HEAD + "@dataclass\nclass Cfg(BaseDataConfig):\n" + SO100_ATTRS + _transform(), "u.py")
    report = check_groot_dataset_versions(ds, unread)
    assert report["versions_run"] == ["n1.5"] and report["versions"]["n1.5"]["statement"].startswith(
        "No data config was given or read")
    assert "Isaac-GR00T N1.5: fine-tuning transforms" in report["coverage"]["not_checked"]


# ---------------------------------------------------------------- recheck of afc6810a (validator/recheck/NOTES.md)
# Repros from validator/recheck/configs/rc_configs.py, run natively in recheck/out/native.json (r2_135517).

INT_GRIPPER = _modality(state__gripper={"start": 5, "end": 6, "dtype": "int64"})


def test_an_edited_super_transform_is_unknown(tmp_path):
    # R9 SuperFiltered: the parent's state StateActionTransform is filtered out; native opens and trains (gripper left
    # unnormalized), so the parent's int64 + min_max fault is not certain (was a false error).
    filtered = ("class Cfg(So100DualCamDataConfig):\n    def transform(self):\n        base = super().transform()\n"
                "        kept = [t for t in base.transforms\n                if not (isinstance(t, StateActionTransform)"
                " and 'state.gripper' in t.apply_to)]\n        return ComposedModalityTransform(transforms=kept)\n")
    report = _run(tmp_path / "f", filtered, modality=INT_GRIPPER)
    assert not _codes(report, "error") and "groot.normalization" in _codes(report, "unknown")
    assert "super().transform() returns" in _finding(report, "groot.config", "unknown")["message"]
    # R17 SuperInsertJitter: mutated in place (native ok) -> no error either
    insert = ("class Cfg(So100DualCamDataConfig):\n    def transform(self):\n        composed = super().transform()\n"
              "        composed.transforms.insert(2, VideoCrop(apply_to=self.video_keys, scale=0.9))\n"
              "        return composed\n")
    assert not _codes(_run(tmp_path / "i", insert), "error")
    # returned as is, the parent's transforms are what N1.5 builds (R12: native ValueError, state_action.py:463)
    plain = "class Cfg(So100DualCamDataConfig):\n    def transform(self):\n        return super().transform()\n"
    assert "groot.normalization" in _codes(_run(tmp_path / "p", plain, modality=INT_GRIPPER), "error")


def test_methods_bound_by_assignment_are_unknown(tmp_path):
    # R10 TransformAssigned (transform = module function; native ok, was a false error from the parent's transform)
    # and R2 AssignedInitBad (__init__ = module function setting video.webcam; native ValueError, was a false pass).
    helper = "def _fn(self):\n    self.video_keys = ['video.webcam']\n\n\n"
    for label, body in (("transform", "    transform = _fn\n"), ("init", "    __init__ = _fn\n"),
                        ("modality_config", "    modality_config = _fn\n"),
                        ("conditional", "    if True:\n        def __init__(self):\n            pass\n")):
        report = _run(tmp_path / label, helper + "class Cfg(So100DualCamDataConfig):\n" + body, modality=INT_GRIPPER)
        assert report["status"] == "unknown" and not _codes(report, "error"), label
        assert "other than by a plain def" in _finding(report, "groot.config", "unknown")["message"], label
    # R11 control: the same transform written as a def is read (gripper not normalized: no error)
    state = "StateActionTransform(apply_to=['state.single_arm'], normalization_modes={'state.single_arm': 'min_max'})"
    assert _run(tmp_path / "m", "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state),
                modality=INT_GRIPPER)["status"] == "pass"
    # two defs of transform: Python keeps the last one
    bad = "StateActionTransform(apply_to=self.state_keys, normalization_modes={'state.single_arm': 'minmax'})"
    twice = "class Cfg(So100DualCamDataConfig):\n" + _transform(state=bad) + "\n" + _transform()
    assert _run(tmp_path / "t", twice)["status"] == "pass"


@pytest.mark.parametrize("missing", ["state.left_arm", "action.left_arm", "video.missing", "missing.key", "state",
                                     "state.too.many"])
def test_apply_to_key_without_a_modality_entry(tmp_path, missing):
    # R8 ApplyToUnknownKey: native fine-tuning AssertionError 'left_arm config not found' (state_action.py:397).
    state = (f"StateActionTransform(apply_to=['state.single_arm', 'state.gripper', {missing!r}], "
             "normalization_modes={'state.single_arm': 'min_max', 'state.gripper': 'min_max'})")
    finding = _finding(_run(tmp_path, "class Cfg(So100DualCamDataConfig):\n" + _transform(state=state)),
                       "groot.transform_keys", "error")
    assert missing in finding["message"] and "state_action.py" in finding["message"]
    assert finding["applies_to"] == "fine-tuning"
    assert "meta/modality.json" in finding["remedy"]


def test_version_selection_minors(tmp_path):
    n17 = _file(tmp_path, "from gr00t.data.types import ModalityConfig\ncfg = {'video': ModalityConfig("
                "delta_indices=[0], modality_keys=['front']), 'state': ModalityConfig(delta_indices=[0], "
                "modality_keys=[]), 'action': ModalityConfig(delta_indices=[0], modality_keys=['single_arm'])}\n",
                "n17.py")
    ds = _dataset(tmp_path / "ds")
    with pytest.raises(InputError, match="is an Isaac-GR00T N1.7 modality config"):
        check_groot_dataset(ds, n17, groot_version="n1.5")
    assert check_groot_dataset(ds, n17, groot_version="auto")["groot_version"] == "n1.7"  # recheck a17_empty_state
    assert check_groot_dataset_versions(ds, n17)["versions_run"] == ["n1.7"]
    combined = check_groot_dataset_versions(ds)
    assert combined["versions"]["n1.7"]["consumer"] == "Isaac-GR00T N1.7 944fe70"
    assert combined["consumer"] == "Isaac-GR00T N1.5 4af2b62; Isaac-GR00T N1.7 944fe70"


@pytest.mark.parametrize("key", ["annotation.task", "video.front"])
def test_state_action_runtime_metadata_and_statistics_modalities(tmp_path, key):
    # Independent native controls 135714: annotation metadata is not part of DatasetModalities;
    # video metadata exists, but DatasetStatistics has no video section.
    extra = f"StateActionTransform(apply_to=[{key!r}]" + (
        ", normalization_modes={'video.front': 'min_max'})" if key.startswith("video.") else ")")
    body = "class Cfg(So100DualCamDataConfig):\n" + _transform()
    body = body.replace("        VideoCrop", f"        {extra},\n        VideoCrop", 1)
    report = _run(tmp_path, body, modality=_modality(annotation={
        **MODALITY["annotation"], "task": {"original_key": "task_index"}}))
    finding = _finding(report, "groot.transform_keys", "error")
    assert "DatasetModalities" in finding["message"] if key.startswith("annotation.") else \
        "DatasetStatistics" in finding["message"]


@pytest.mark.parametrize("stage,expected", [("raw", "error"), ("tensor", "pass"), ("numpy", "error"),
                                           ("dynamic", "unknown")])
def test_video_state_action_requires_a_proved_tensor_input(tmp_path, stage, expected):
    # A no-op StateActionTransform still asserts a torch dtype. The native positive control is AFTER VideoToTensor.
    extra = "StateActionTransform(apply_to=['video.front'])"
    body = "class Cfg(So100DualCamDataConfig):\n" + _transform()
    marker = {"raw": "VideoToTensor", "tensor": "VideoCrop", "numpy": "StateActionToTensor",
              "dynamic": "VideoCrop"}[stage]
    body = body.replace("        " + marker, f"        {extra},\n        " + marker, 1)
    if stage == "dynamic":
        body = body.replace("        return ComposedModalityTransform", "        transforms = list(transforms)\n"
                            "        return ComposedModalityTransform")
    report = _run(tmp_path, body)
    assert report["status"] == expected
    if expected != "pass":
        assert "groot.transform_dtype" in _codes(report, expected)
