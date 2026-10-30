"""The Isaac-GR00T N1.5 profile of the GR00T dataset contract check (groot_contract.check_groot_dataset with
groot_version="n1.5").

Isaac-GR00T N1.5 (branch n1d5 = tag n1.5-release, 4af2b62) reads a LeRobot v2 dataset differently from N1.7, so the
N1.7 rules do not carry over one for one. Every rule below cites the pinned N1.5 line that makes it true:

- one loader on every path: LeRobotSingleDataset (gr00t/data/dataset.py) opens the dataset for fine-tuning
  (scripts/gr00t_finetune.py:200-211, with the data config's transforms) and for the direct loaders
  (scripts/eval_policy.py:129, scripts/load_dataset.py:240, without transforms). Faults found when the dataset is
  opened or read apply to both; faults in the transforms (normalization, rotations, padding to the model's state and
  action width, the language key) apply to fine-tuning;
- meta/modality.json is validated with pydantic (gr00t/data/schema.py), and every state, action and video entry is
  read when the dataset is opened (dataset.py:310-357, 374-388), not only the ones the data config names;
- state and action groups are taken with numpy index arrays (dataset.py:383-388, 775-779): an end past the vector's
  width raises IndexError (loud), while a negative start wraps around to the last values without an error;
- meta/stats.json is kept when every entry validates, and otherwise recomputed from the parquet files and written
  over the old file (dataset.py:359-373). There is no fingerprint, so a stats.json from another robot with a wider
  vector is kept and sliced silently;
- cameras are paired with modality.json by name only (dataset.py:488-505); there is no positional fallback, and an
  entry with no original_key reads the info.json feature named like the entry (dataset.py:339-342);
- the data config is a BaseDataConfig subclass (gr00t/experiment/data_config.py), not register_modality_config.
  It is parsed with ast, never imported or executed. Its six key/index attributes are read as class attributes
  (the last assignment in a class body, as Python keeps); attributes set on the instance (__init__, __post_init__,
  @dataclass fields, properties) are reported unknown, never guessed;
- fine-tuning builds the config's transform(): StateActionTransform.set_metadata checks its keys when the dataset is
  opened (gr00t/data/transform/state_action.py:386-472), ConcatTransform concatenates only the keys it lists
  (gr00t/data/transform/concat.py:74-155) and GR00TTransform checks the horizons (gr00t/model/transforms.py:252, 283).

Silent faults reproduced with N1.5's own code (.validation/k4-groot-n15/out/n15_native.json and the validator's
validator/out/j2/native_plants.json): camera entries whose original_keys are swapped under the config's names, a
negative slice start, a stats.json of another robot with a wider vector, a state wider than the model's
max_state_dim (truncated), and a ConcatTransform order that leaves a configured state group or camera out.

Every report carries coverage.not_checked: what this profile does not check (NOT_CHECKED below).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .artifacts import InputError, read_json
from .groot_contract import (
    DIRECT,
    FT,
    MAX_CONFIG_BYTES,
    MAX_LINES,
    _call_name,
    _constants,
    _designate,
    _distinct,
    _ints,
    _names,
    _read_cache,
    _Report,
    _scan_jsonl,
    _size,
    _string,
    _strings,
    _width,
)

GROOT_N15 = "Isaac-GR00T N1.5 4af2b62"
N15_COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
DATASET = "gr00t/data/dataset.py"
SCHEMA = "gr00t/data/schema.py"
DATA_CONFIG = "gr00t/experiment/data_config.py"
MODEL_TF = "gr00t/model/transforms.py"
STATE_TF = "gr00t/data/transform/state_action.py"
CONCAT_TF = "gr00t/data/transform/concat.py"
VIDEO_TF = "gr00t/data/transform/video.py"
FINETUNE = "scripts/gr00t_finetune.py"
DIRECT_USERS = "scripts/eval_policy.py:129, scripts/load_dataset.py:240"
DEFAULT_COLUMNS = {"state": "observation.state", "action": "action"}  # schema.py:83-98
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")  # schema.py:173-179
ROTATIONS = ("axis_angle", "quaternion", "rotation_6d", "matrix", "euler_angles_rpy", "euler_angles_ryp",
             "euler_angles_pry", "euler_angles_pyr", "euler_angles_yrp", "euler_angles_ypr")  # schema.py:27-39
ROTATION_WIDTH = {"axis_angle": 3, "quaternion": 4, "rotation_6d": 6, "matrix": 9}  # euler_angles_*: 3
NORMALIZATION_MODES = ("q99", "mean_std", "min_max", "scale", "binary")  # Normalizer.forward, state_action.py:113-189
ATTRS = ("video_keys", "state_keys", "action_keys", "language_keys", "observation_indices", "action_indices")
# Methods that run when the class is instantiated or an attribute is read; with one of them in the class chain, an
# attribute missing from the class bodies may still exist on the instance.
_HOOKS = ("__init__", "__new__", "__post_init__", "__getattr__", "__getattribute__", "__init_subclass__",
          "__setattr__")
# N1.5 transform classes (gr00t/data/transform/*.py, gr00t/model/transforms.py); none of them adds or removes keys
# before ConcatTransform. A call to anything else in transform() may, so concatenation findings become unknown.
_TRANSFORMS = {"VideoToTensor", "VideoCrop", "VideoResize", "VideoRandomRotation", "VideoHorizontalFlip",
               "VideoGrayscale", "VideoColorJitter", "VideoRandomGrayscale", "VideoRandomPosterize", "VideoToNumpy",
               "StateActionToTensor", "StateActionTransform", "StateActionPerturbation", "StateActionDropout",
               "StateActionSinCosTransform", "ConcatTransform", "GR00TTransform", "ComposedModalityTransform"}
_PLAIN_CALLS = {"len", "list", "range", "dict", "tuple", "set", "sorted", "super", "transform", "int", "float", "str",
                "zip", "enumerate", "print", "isinstance"}
_BRANCHES = tuple(getattr(ast, name) for name in ("If", "IfExp", "For", "AsyncFor", "While", "Try", "TryStar", "With",
                                                  "AsyncWith", "Match", "BoolOp", "Lambda", "FunctionDef",
                                                  "AsyncFunctionDef", "ListComp", "SetComp", "DictComp",
                                                  "GeneratorExp") if hasattr(ast, name))
_DYNAMIC = object()  # an attribute that is set in a way this check does not read statically
NOT_CHECKED = [
    "Parquet contents: column presence when meta/info.json omits a column, real widths and values",
    "video files: existence, codec support in the chosen backend, and decoded resolution against meta/info.json "
    "(VideoToTensor asserts it at fine-tuning, gr00t/data/transform/video.py:525-545)",
    "recomputing meta/stats.json over non-numeric parquet columns (warned only when meta/info.json declares image "
    "features), and a read-only dataset directory when meta/stats.json must be rewritten",
    "rotation_type against the slice width (a quaternion slice of 3 values fails when fine-tuning reads a sample)",
    "ConcatTransform's state width rule for rotated keys (gr00t/data/transform/concat.py:121-130)",
    "a transform() without a ConcatTransform, and transform classes other than N1.5's own",
    "LeRobotMixtureDataset: several datasets whose fps or modality.json entries differ (gr00t/data/dataset.py:1281)",
    "whether --data-config can load the class: N1.5 reads only its own DATA_CONFIG_MAP or an importable "
    "module:Class (gr00t/experiment/data_config.py:74-144)",
    "the checkpoint's metadata and embodiment tag, training and task success",
]
NOT_CHECKED_WITHOUT_CONFIG = ["config keys", "camera pairing", "language", "fine-tuning transforms"]
SIDES = (("video", "video_keys", "observation_indices"), ("state", "state_keys", "observation_indices"),
         ("action", "action_keys", "action_indices"), ("language", "language_keys", "observation_indices"))
_UNREAD = "<not statically readable>"

# numpy dtype names N1.5 passes to np.dtype (dataset.py:316); whether np.issubdtype(.., np.floating) (dataset.py:317).
# Names outside these two sets are reported as unknown rather than guessed (numpy is not imported here).
_FLOAT_DTYPES = {"float16", "float32", "float64", "float", "double", "half", "single", "longdouble", "float128",
                 "f2", "f4", "f8", "f16", "<f2", "<f4", "<f8", ">f4", ">f8", "=f4", "=f8", "d", "e", "f", "g"}
_OTHER_DTYPES = {"int8", "int16", "int32", "int64", "int", "uint8", "uint16", "uint32", "uint64", "uint", "bool",
                 "bool_", "?", "b", "B", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "<i4", "<i8", "object", "O",
                 "complex64", "complex128", "complex", "str", "U", "bytes", "S"}

# The built-in data configs of gr00t/experiment/data_config.py:150-788 at 4af2b62, reduced to what this check reads:
# class attributes, and in transform() the GR00TTransform widths, sin-cos, normalization, rotation, crop and concat
# calls. Video colour/resize steps are left out. Parsed with the same ast reader as a user's file, so a user config
# that subclasses a built-in resolves through it. The validation harness checks that every DATA_CONFIG_MAP entry
# parsed from the pinned file equals the same entry parsed from this text.
BUILTIN_SOURCE = '''
class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "min_max" for key in self.action_keys}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={key: "min_max" for key in self.state_keys}),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "min_max" for key in self.action_keys}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={key: "min_max" for key in self.state_keys}),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "min_max" for key in self.action_keys}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_leg", "state.right_leg", "state.waist", "state.left_arm", "state.right_arm",
                  "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


class FourierGr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist",
                  "state.neck"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist",
                   "action.neck"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={key: "min_max" for key in self.state_keys}),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "min_max" for key in self.action_keys}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"]
    state_keys = ["state.right_arm_eef_pos", "state.right_arm_eef_quat", "state.right_gripper_qpos",
                  "state.left_arm_eef_pos", "state.left_arm_eef_quat", "state.left_gripper_qpos"]
    action_keys = ["action.right_arm_eef_pos", "action.right_arm_eef_rot", "action.right_gripper_close",
                   "action.left_arm_eef_pos", "action.left_arm_eef_rot", "action.left_gripper_close"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    state_normalization_modes = {"state.right_arm_eef_pos": "min_max", "state.right_gripper_qpos": "min_max",
                                 "state.left_arm_eef_pos": "min_max", "state.left_gripper_qpos": "min_max"}
    state_target_rotations = {"state.right_arm_eef_quat": "rotation_6d", "state.left_arm_eef_quat": "rotation_6d"}
    action_normalization_modes = {"action.right_gripper_close": "binary", "action.left_gripper_close": "binary"}

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionTransform(apply_to=self.state_keys, normalization_modes=self.state_normalization_modes,
                                 target_rotations=self.state_target_rotations),
            StateActionTransform(apply_to=self.action_keys, normalization_modes=self.action_normalization_modes),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


class BimanualPandaHandDataConfig(BimanualPandaGripperDataConfig):
    video_keys = ["video.right_wrist_view", "video.left_wrist_view", "video.ego_view"]
    state_keys = ["state.right_arm_eef_pos", "state.right_arm_eef_quat", "state.right_hand",
                  "state.left_arm_eef_pos", "state.left_arm_eef_quat", "state.left_hand"]
    action_keys = ["action.right_arm_eef_pos", "action.right_arm_eef_rot", "action.right_hand",
                   "action.left_arm_eef_pos", "action.left_arm_eef_rot", "action.left_hand"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    state_normalization_modes = {"state.right_arm_eef_pos": "min_max", "state.right_hand": "min_max",
                                 "state.left_arm_eef_pos": "min_max", "state.left_hand": "min_max"}
    action_normalization_modes = {"action.right_hand": "min_max", "action.left_hand": "min_max"}
    state_target_rotations = {"state.right_arm_eef_quat": "rotation_6d", "state.left_arm_eef_quat": "rotation_6d"}


class SinglePandaGripperDataConfig(BimanualPandaGripperDataConfig):
    video_keys = ["video.left_view", "video.right_view", "video.wrist_view"]
    state_keys = ["state.end_effector_position_relative", "state.end_effector_rotation_relative",
                  "state.gripper_qpos", "state.base_position", "state.base_rotation"]
    action_keys = ["action.end_effector_position", "action.end_effector_rotation", "action.gripper_close",
                   "action.base_motion", "action.control_mode"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    state_normalization_modes = {"state.end_effector_position_relative": "min_max",
                                 "state.end_effector_rotation_relative": "min_max",
                                 "state.gripper_qpos": "min_max", "state.base_position": "min_max",
                                 "state.base_rotation": "min_max"}
    state_target_rotations = {"state.end_effector_rotation_relative": "rotation_6d",
                              "state.base_rotation": "rotation_6d"}
    action_normalization_modes = {"action.end_effector_position": "min_max",
                                  "action.end_effector_rotation": "min_max", "action.gripper_close": "binary",
                                  "action.base_motion": "min_max", "action.control_mode": "binary"}


class FourierGr1ArmsWaistDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist"]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        return super().transform()


class OxeDroidDataConfig(BaseDataConfig):
    video_keys = ["video.exterior_image_1", "video.exterior_image_2", "video.wrist_image"]
    state_keys = ["state.eef_position", "state.eef_rotation", "state.gripper_position"]
    action_keys = ["action.eef_position_delta", "action.eef_rotation_delta", "action.gripper_position"]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={"state.eef_position": "min_max",
                                                      "state.gripper_position": "min_max"},
                                 target_rotations={"state.eef_rotation": "rotation_6d"}),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={"action.gripper_position": "binary"},
                                 target_rotations={"action.eef_rotation_delta": "axis_angle"}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


class AgibotGenie1DataConfig(BaseDataConfig):
    video_keys = ["video.top_head", "video.hand_left", "video.hand_right"]
    state_keys = ["state.left_arm_joint_position", "state.right_arm_joint_position", "state.left_effector_position",
                  "state.right_effector_position", "state.head_position", "state.waist_position"]
    action_keys = ["action.left_arm_joint_position", "action.right_arm_joint_position",
                   "action.left_effector_position", "action.right_effector_position", "action.head_position",
                   "action.waist_position", "action.robot_velocity"]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={key: "min_max" for key in self.state_keys}),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "min_max" for key in self.action_keys}),
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices),
                           max_state_dim=64, max_action_dim=32),
        ]


DATA_CONFIG_MAP = {
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "agibot_genie1": AgibotGenie1DataConfig(),
}
'''
_BUILTIN_MODULE = "gr00t.experiment.data_config"


# ---------------------------------------------------------------- data config (.py, parsed with ast only)

class _Module:
    """One parsed config file: its config classes, string constants, import aliases and name -> class maps."""

    def __init__(self, tree: ast.Module, builtin: bool = False):
        self.builtin = builtin
        self.constants = _constants(tree)
        self.classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.aliases = {}  # local name -> built-in class name, from `from gr00t.experiment.data_config import X as Y`
        self.modules = {_BUILTIN_MODULE}  # names bound to the module itself (`import ... as dc`)
        self.star = False
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == _BUILTIN_MODULE and not node.level:
                for name in node.names:
                    if name.name == "*":
                        self.star = True
                    else:
                        self.aliases[name.asname or name.name] = name.name
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name == _BUILTIN_MODULE and name.asname:
                        self.modules.add(name.asname)
        self.maps = {}  # DATA_CONFIG_MAP-like dicts: "key" -> class name
        for node in tree.body:
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Dict) and value.keys:
                pairs = [(_string(k, self.constants), v) for k, v in zip(value.keys, value.values, strict=True)]
                if all(k is not None and isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and not v.args
                       and not v.keywords for k, v in pairs):
                    for key, call in pairs:
                        self.maps[key] = call.func.id


_BUILTIN = _Module(ast.parse(BUILTIN_SOURCE), builtin=True)


def _stores(node) -> set:
    """Names a statement binds in the enclosing scope: assignment, del, for/with/except targets, walrus, imports,
    nested defs."""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and not isinstance(sub.ctx, ast.Load):
            names.add(sub.id)
        elif isinstance(sub, ast.alias):
            names.add((sub.asname or sub.name).split(".")[0])
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
        elif type(sub).__name__ in ("MatchAs", "MatchStar") and getattr(sub, "name", None):
            names.add(sub.name)
        elif type(sub).__name__ == "MatchMapping" and getattr(sub, "rest", None):
            names.add(sub.rest)
    return names


def _class_body(node: ast.ClassDef) -> dict:
    """{name: value node} as the class body leaves each name: the LAST plain `name = value` (or `name: T = value`),
    as Python keeps it; _DYNAMIC for a name the body also binds any other way (a method or property, a tuple,
    augmented, conditional or nested assignment, del, an import)."""
    values, dynamic = {}, set()
    for item in node.body:
        if isinstance(item, ast.Assign) and all(isinstance(t, ast.Name) for t in item.targets):
            for target in item.targets:
                values[target.id] = item.value
            dynamic |= _stores(item.value)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.value is not None:  # a bare annotation binds nothing (a @dataclass is reported separately)
                values[item.target.id] = item.value
                dynamic |= _stores(item.value)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            dynamic.add(item.name)
        else:
            dynamic |= _stores(item)
    return {**values, **dict.fromkeys(dynamic, _DYNAMIC)}


def _instance_attrs(node: ast.ClassDef) -> set | None:
    """Attributes the class's methods set on the instance (self.x = ...); None when a method may set any attribute
    (setattr, __setattr__, vars, __dict__)."""
    names = set()
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = item.args.posonlyargs + item.args.args
        me = params[0].arg if params else None
        for sub in ast.walk(item):
            if isinstance(sub, ast.Attribute) and sub.attr == "__dict__":
                return None
            if isinstance(sub, ast.Call) and _call_name(sub) in ("setattr", "__setattr__", "vars"):
                return None
            if isinstance(sub, ast.Attribute) and not isinstance(sub.ctx, ast.Load) \
                    and isinstance(sub.value, ast.Name) and sub.value.id == me:
                names.add(sub.attr)
    return names


def _method(node: ast.ClassDef, name: str):
    """The def a class body leaves bound to name: the last one, as Python keeps it."""
    return next((item for item in reversed(node.body) if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and item.name == name), None)


def _bound_otherwise(node: ast.ClassDef) -> set:
    """Names the class body binds other than by a top-level def (transform = fn, __init__ = setup, an import, a
    conditional def): a method bound that way is not read."""
    names = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(item, ast.Assign):
            names |= {n.id for t in item.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
        names |= _stores(item)
    return names


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return None if inner is None else f"{inner}.{node.attr}"
    return None


def _base(where: _Module, node) -> tuple[_Module | None, str]:
    """Where a base class expression points: (module holding the class, class name), or (None, why not)."""
    dotted = _dotted(node)
    if dotted is None:
        return None, f"a base class written as {ast.unparse(node)} cannot be read statically"
    prefix, _, name = dotted.rpartition(".")
    if prefix:
        return (_BUILTIN, name) if prefix in where.modules else (None, f"base class {dotted} cannot be resolved")
    if where.builtin or name in where.classes:
        return where, name
    if name in where.aliases:
        return _BUILTIN, where.aliases[name]
    if where.star:
        return _BUILTIN, name
    return None, f"base class {name} is neither defined in the file nor imported from {_BUILTIN_MODULE}"


def _chain(module: _Module, name: str) -> tuple[list, str | None]:
    """[(class node, its module), ...] from the class up to (not including) BaseDataConfig, or (chain, problem)."""
    chain, seen = [], set()
    where, current = module, name
    while True:
        if (id(where), current) in seen:
            return chain, f"class {current} inherits from itself"
        seen.add((id(where), current))
        node = where.classes.get(current)
        if node is None:
            return chain, f"{current} is not a data config class of {_BUILTIN_MODULE} at 4af2b62"
        chain.append((node, where))
        bases = [b for b in node.bases if _dotted(b) not in ("ABC", "abc.ABC", "object")]
        if len(bases) != 1:
            return chain, f"class {node.name} must have exactly one base class to be read statically"
        where, current = _base(where, bases[0])
        if where is None:
            return chain, current
        if current == "BaseDataConfig":  # data_config.py:38-66
            return chain, None


def _is_config_class(module: _Module, name: str) -> bool:
    chain, problem = _chain(module, name)
    return problem is None and bool(chain)


def _attr_value(chain: list, name: str):
    """What self.<name> is on the instance N1.5 builds: (value node, its module) from the nearest class body that sets
    it; (_DYNAMIC, None) when a method sets it on the instance or a class body binds it in a way not read
    statically; (None, None) when no class in the chain sets it."""
    for node, _where in chain:
        attrs = _instance_attrs(node)
        if attrs is None or name in attrs:
            return _DYNAMIC, None
    for node, where in chain:
        value = _class_body(node).get(name)
        if value is not None:
            return value, where
    return None, None


def _keys_of(node, where: _Module, chain: list):
    """A key list: a literal list, a module constant, or self.<attribute> resolved on the selected class."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        value, owner = _attr_value(chain, node.attr)
        return None if value is None or value is _DYNAMIC else _strings(value, owner.constants)
    return _strings(node, where.constants)


def _length(node, where: _Module, chain: list) -> int | None:
    """A GR00TTransform horizon: an int literal, or len() of a literal list or of self.<attribute>."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"
            and len(node.args) == 1 and not node.keywords):
        return None
    inner = node.args[0]
    if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) and inner.value.id == "self":
        value, owner = _attr_value(chain, inner.attr)
        if value is None or value is _DYNAMIC:
            return None
        inner, where = _resolve_name(value, owner), owner
    items = _ints(inner)
    items = _strings(inner, where.constants) if items is None else items
    return None if items is None else len(items)


def _modes_of(node, where: _Module, chain: list):
    """{key: str} from a dict literal, {key: "mode" for key in self.x}, or self.<dict attribute>."""
    if node is None:
        return {}
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        value, owner = _attr_value(chain, node.attr)
        return None if value is None or value is _DYNAMIC else _modes_of(value, owner, chain)
    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        pairs = {_string(k, where.constants): _string(v, where.constants) for k, v in zip(node.keys, node.values,
                                                                                            strict=True)}
        return None if any(k is None or v is None for k, v in pairs.items()) else pairs
    if isinstance(node, ast.DictComp) and len(node.generators) == 1 and not node.generators[0].ifs:
        gen = node.generators[0]
        mode = _string(node.value, where.constants)
        if isinstance(node.key, ast.Name) and isinstance(gen.target, ast.Name) and node.key.id == gen.target.id \
                and mode is not None:
            keys = _keys_of(gen.iter, where, chain)
            return None if keys is None else {key: mode for key in keys}
    return None


def _video_input_kinds(method, where, chain):
    """Video dtypes before calls in a plainly returned transform list; other constructions stay unknown.

    Constructor visitation order is not execution order. Only a literal ComposedModalityTransform list (or a
    local list assigned before that return) proves the order in which N1.5 applies its transforms.
    """
    lists = {}
    body = list(method.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return):
        return {}
    for statement in body[:-1]:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and isinstance(statement.value, (ast.List, ast.Tuple))):
            return {}
        lists[statement.targets[0].id] = statement.value
    result = body[-1].value
    if not isinstance(result, ast.Call) or _call_name(result) != "ComposedModalityTransform" or result.args \
            or any(kw.arg is None for kw in result.keywords):
        return {}
    sequence = next((kw.value for kw in result.keywords if kw.arg == "transforms"), None)
    if isinstance(sequence, ast.Name):
        sequence = lists.get(sequence.id)
    if not isinstance(sequence, (ast.List, ast.Tuple)) or not all(isinstance(n, ast.Call) for n in sequence.elts):
        return {}
    # Unused constructor calls or arbitrary expressions can mutate a list/config; do not infer through them.
    if any(value is not sequence for value in lists.values()):
        return {}
    kinds, before, default = {}, {}, "numpy"
    for call in sequence.elts:
        name = _call_name(call)
        fields = {kw.arg: kw.value for kw in call.keywords}
        keys = None if call.args or None in fields else _keys_of(fields.get("apply_to"), where, chain)
        if name == "StateActionTransform":
            before[call] = {key: kinds.get(key, default) for key in keys or [] if key.startswith("video.")}
        if name in ("VideoToTensor", "VideoToNumpy"):
            if keys is None:
                kinds, default = {}, "unknown"
            else:
                kinds.update({key: "torch" if name == "VideoToTensor" else "numpy" for key in keys})
        elif name not in _TRANSFORMS or name in ("ConcatTransform", "GR00TTransform") \
                or name.startswith("StateAction") and name != "StateActionTransform" \
                and (keys is None or any(key.startswith("video.") for key in keys)):
            kinds, default = {}, "unknown"
    return before


def _transform_facts(chain: list, start: int = 0) -> dict | None:
    """What transform() builds, from the first class (at or after `start`) in the chain that defines it.

    "state_action" and "concat_orders" keep each StateActionTransform / ConcatTransform call on its own, with
    "conditional" set when the call sits in a branch, loop, boolean operator or nested function (it may not run)."""
    for index in range(start, len(chain)):
        node, where = chain[index]
        method = _method(node, "transform")
        if method is None:
            continue
        facts = {"max_state_dim": None, "max_action_dim": None, "state_horizon": None, "action_horizon": None,
                 "sincos": [], "normalization": {}, "target_rotations": {}, "state_action": [], "concat_orders": [],
                 "video_crop": False, "concat": False, "groot_transform": False, "other_calls": [], "unread": []}
        parents = {child: parent for parent in ast.walk(method) for child in ast.iter_child_nodes(parent)}
        video_inputs = _video_input_kinds(method, where, chain)

        def conditional(call, parents=parents, method=method):
            node = parents.get(call)
            while node is not None and node is not method:
                if isinstance(node, _BRANCHES):
                    return True
                node = parents.get(node)
            return False

        supered = False
        for call in (n for n in ast.walk(method) if isinstance(n, ast.Call)):
            name = _call_name(call)
            kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
            opaque = bool(call.args) or any(k.arg is None for k in call.keywords)  # positional or **kwargs
            if name == "transform" and isinstance(call.func, ast.Attribute) and _call_name(call.func.value) == "super":
                supered = True
                parent = _transform_facts(chain, index + 1)
                if parent is None:
                    facts["unread"].append("super().transform() has no parent transform")
                elif conditional(call) or not (isinstance(parents.get(call), ast.Return)
                                               and parents.get(parents[call]) is method):
                    # Only `return super().transform(...)` as a statement of transform() itself builds the parent
                    # transforms as they are; assigned, filtered, indexed or mutated, they may not be (R9).
                    for entry in parent["state_action"] + parent["concat_orders"]:
                        entry["conditional"] = True
                    for field in ("state_action", "concat_orders", "other_calls", "unread"):
                        facts[field].extend(parent[field])
                    facts["unread"].append("transform() changes or branches on what super().transform() returns")
                else:
                    _merge(facts, parent)
            elif name == "GR00TTransform":
                facts["groot_transform"] = True
                for field in ("max_state_dim", "max_action_dim"):
                    value = kw.get(field)
                    value = value.value if isinstance(value, ast.Constant) and type(value.value) is int else None
                    facts[field] = value
                    if value is None:
                        facts["unread"].append(f"GR00TTransform {field}")
                for field in ("state_horizon", "action_horizon"):
                    facts[field] = None if opaque else _length(kw.get(field), where, chain)
                    if facts[field] is None:
                        facts["unread"].append(f"GR00TTransform {field}")
            elif name == "StateActionSinCosTransform":
                keys = _keys_of(kw.get("apply_to"), where, chain)
                if keys is None:
                    facts["unread"].append("StateActionSinCosTransform apply_to")
                else:
                    facts["sincos"].extend(keys)
            elif name == "StateActionTransform":
                entry = {"apply_to": None if opaque else _keys_of(kw.get("apply_to"), where, chain),
                         "conditional": conditional(call), "video_inputs": video_inputs.get(call, {})}
                if entry["apply_to"] is None:
                    facts["unread"].append("StateActionTransform apply_to")
                for field, target in (("normalization_modes", "normalization"), ("target_rotations",
                                                                                  "target_rotations")):
                    modes = None if opaque else _modes_of(kw.get(field), where, chain)
                    entry[target] = modes
                    if modes is None:
                        facts["unread"].append(f"StateActionTransform {field}")
                        continue
                    for key, mode in modes.items():
                        if facts[target].get(key, mode) != mode:  # two branches disagree (Libero action_norm)
                            facts[target][key] = None
                        else:
                            facts[target][key] = mode
                facts["state_action"].append(entry)
            elif name == "VideoCrop":
                facts["video_crop"] = True
            elif name == "ConcatTransform":
                facts["concat"] = True
                orders = {"conditional": conditional(call)}
                for side in ("video", "state", "action"):
                    value = kw.get(f"{side}_concat_order")
                    if opaque:
                        orders[side] = _UNREAD
                    elif value is None or isinstance(value, ast.Constant) and value.value is None:
                        orders[side] = None  # video_concat_order is required; state/action default to None
                    else:
                        keys = _keys_of(value, where, chain)
                        orders[side] = _UNREAD if keys is None else keys
                    if orders[side] == _UNREAD:
                        facts["unread"].append(f"ConcatTransform {side}_concat_order")
                facts["concat_orders"].append(orders)
            elif name not in _TRANSFORMS and name not in _PLAIN_CALLS:
                facts["other_calls"].append(name or ast.unparse(call.func)[:80])
        if not facts["groot_transform"] and not supered:  # e.g. `return my_helper(self)`: what it builds is not read
            facts["unread"].append("transform() builds no GR00TTransform and calls no super().transform(); a "
                                   "helper it delegates to is not read")
        facts["other_calls"] = sorted(set(facts["other_calls"]))
        return facts
    return None


def _merge(facts: dict, parent: dict) -> None:
    for field in ("max_state_dim", "max_action_dim", "state_horizon", "action_horizon"):
        facts[field] = facts[field] if facts[field] is not None else parent[field]
    facts["sincos"].extend(parent["sincos"])
    for target in ("normalization", "target_rotations"):
        for key, mode in parent[target].items():
            facts[target][key] = None if facts[target].get(key, mode) != mode else mode
    for field in ("state_action", "concat_orders", "other_calls", "unread"):
        facts[field].extend(parent[field])
    for flag in ("video_crop", "concat", "groot_transform"):
        facts[flag] = facts[flag] or parent[flag]


def _resolve(module: _Module, name: str, map_key: str | None) -> dict:
    """The config N1.5 builds from one data config class: BaseDataConfig.modality_config() (data_config.py:40-62)."""
    chain, problem = _chain(module, name)
    config = {"class": name, "data_config": map_key, "unsupported": [], "missing": []}
    if problem is not None:
        config["unsupported"].append(problem)
        return config
    for node, _where in chain:
        if _method(node, "modality_config") is not None:
            config["unsupported"].append(f"class {node.name} overrides modality_config()")
            return config
        if node.decorator_list:  # @dataclass turns annotated attributes into instance fields; others may rewrite it
            config["unsupported"].append(f"class {node.name} is decorated (@{ast.unparse(node.decorator_list[0])}), "
                                         "which can set its attributes on the instance")
            return config
        if node.keywords:
            config["unsupported"].append(f"class {node.name} passes {[k.arg for k in node.keywords]} to its class "
                                         "statement (a metaclass can change its attributes)")
            return config
        bound = sorted(_bound_otherwise(node) & {"transform", "modality_config", *_HOOKS})
        if bound:
            config["unsupported"].append(f"class {node.name} binds {bound} other than by a plain def in its body (an "
                                         "assignment, import or conditional def), "
                                         "so what N1.5 runs there is not read")
            return config
    hooks = [f"{node.name}.{hook}" for node, _where in chain for hook in _HOOKS if _method(node, hook) is not None]
    values, dynamic, hooked = {}, [], []
    for attr in ATTRS:
        value, owner = _attr_value(chain, attr)
        if value is _DYNAMIC:
            dynamic.append(attr)
            continue
        if value is None:
            (hooked if hooks else config["missing"]).append(attr)
            continue
        parsed = _ints(_resolve_name(value, owner)) if attr.endswith("indices") else _strings(value, owner.constants)
        if parsed is None:
            config["unsupported"].append(f"{attr} is not a literal list")
        values[attr] = parsed
    if dynamic:
        config["unsupported"].append(f"{dynamic} are set on the instance (in a method such as __init__) or bound in a "
                                     "way this check does not read (a property, a conditional, augmented or tuple "
                                     "assignment)")
    if hooked:
        config["unsupported"].append(f"{hooked} are not class attributes, and {hooks} may set them on the instance")
    if config["unsupported"] or config["missing"]:
        return config
    for side, keys, indices in SIDES:
        config[side] = {"keys": values[keys], "delta_indices": values[indices]}
    config["transform"] = _transform_facts(chain)
    return config


def _resolve_name(node, owner: _Module):
    return owner.constants.get(node.id, node) if isinstance(node, ast.Name) else node


def read_data_config(path: str | Path | None, config_name: str | None = None) -> dict:
    """Parse an Isaac-GR00T N1.5 data config without executing it.

    `path` is a .py file defining BaseDataConfig subclasses (or the pinned gr00t/experiment/data_config.py), and
    `config_name` a class name or a key of a DATA_CONFIG_MAP-like dict in it; "module:Class" (the form N1.5's
    --data-config takes, data_config.py:74-123) is accepted with the module's file. With no path, config_name names a
    built-in DATA_CONFIG_MAP entry or class (data_config.py:775-788); "module:Class" without a file is refused unless
    the module is gr00t.experiment.data_config, because N1.5 would import that module's own class of the name.
    Returns {"class", "data_config", "video"/"state"/"action"/"language": {"keys", "delta_indices"},
    "transform": facts or None, "unsupported": [...], "missing": [...]}.
    """
    if config_name is not None and ":" in config_name:
        module_name, config_name = config_name.split(":", 1)
        if path is None and module_name != _BUILTIN_MODULE:
            raise InputError(f"config_name {module_name}:{config_name} names a class of module {module_name!r}, which "
                             f"N1.5 imports (data_config.py:101-118) and which may define its own {config_name}; pass "
                             f"that module's .py file as the data config. Without a file only {_BUILTIN_MODULE} "
                             "classes and DATA_CONFIG_MAP keys are known.")
    if path is None:
        if config_name is None:
            raise InputError("Pass a data config file, or config_name naming a built-in N1.5 data config.")
        module = _BUILTIN
    else:
        path = Path(path)
        if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
            raise InputError(f"Expected a data config .py file no larger than {MAX_CONFIG_BYTES} bytes: {path}")
        try:
            module = _Module(ast.parse(path.read_bytes(), filename=str(path)))
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            raise InputError(f"Cannot parse data config {path}: {exc}") from exc
    where = path if path is not None else f"the built-in {DATA_CONFIG}"
    if path is not None and not any(_is_config_class(module, name) for name in module.classes) \
            and detect_version(path) == "n1.7":
        raise InputError(f"{path} is an Isaac-GR00T N1.7 modality config (dicts of ModalityConfig with bare group "
                         "names), not an N1.5 data config; check it with groot_version='n1.7' (--groot-version n1.7).")
    if config_name is not None:
        if config_name in module.maps:
            return _resolve(module, module.maps[config_name], config_name)
        if config_name in module.classes:
            return _resolve(module, config_name, next((k for k, v in module.maps.items() if v == config_name), None))
        choices = sorted(module.maps) + sorted(module.classes)
        raise InputError(f"{where} has no data config {config_name!r}; it defines {choices}.")
    candidates = [name for name in module.classes if _is_config_class(module, name)]
    if len(candidates) != 1:
        raise InputError(f"{where} defines {len(candidates)} N1.5 data config classes {sorted(candidates)}; pass "
                         "config_name (a class name, or a DATA_CONFIG_MAP key such as so100).")
    return _resolve(module, candidates[0], next((k for k, v in module.maps.items() if v == candidates[0]), None))


def detect_version(path: str | Path) -> str | None:
    """"n1.5" or "n1.7" when the config file's form says so unambiguously, else None. Never guesses.

    "n1.5": BaseDataConfig subclasses and no N1.7-style ModalityConfig dict. "n1.7": dicts of ModalityConfig whose
    video/state/action keys are all bare group names, and no BaseDataConfig subclass. Anything else (both forms, a
    plain class or notebook dict with N1.5's prefixed keys, keys that cannot be read) is None."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
        return None
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    module = _Module(tree)
    constants = module.constants
    styles = set()  # how each dict of ModalityConfig writes its video/state/action keys
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys and all(k is not None and _string(k, constants) is not None
                                                            for k in node.keys) \
                and any(_call_name(v) == "ModalityConfig" for v in node.values):
            styles |= _key_styles(node, constants)
    n15 = any(_is_config_class(module, name) for name in module.classes)
    # N1.7 configs list bare group names (single_arm); N1.5 writes modality-prefixed keys (state.single_arm), also in
    # its notebook dicts and plain classes, which the N1.5 profile cannot read: those stay undetermined.
    if n15 and styles <= {"n1.5"}:
        return "n1.5"
    if not n15 and styles == {"n1.7"}:
        return "n1.7"
    return None


def _key_styles(node: ast.Dict, constants: dict) -> set:
    """{"n1.5" | "n1.7" | None} for the video/state/action ModalityConfig values of one dict: prefixed keys
    (video.front) are N1.5's form, bare group names N1.7's, anything else (unreadable, mixed) None."""
    styles = set()
    for key, value in zip(node.keys, node.values, strict=True):
        modality = _string(key, constants)
        if modality not in ("video", "state", "action") or _call_name(value) != "ModalityConfig":
            continue
        fields = dict(zip(("delta_indices", "modality_keys"), value.args, strict=False))
        fields.update({k.arg: k.value for k in value.keywords if k.arg is not None})
        keys = _strings(fields.get("modality_keys"), constants)
        if keys == []:  # an empty key list is written the same way in both versions (N1.7 state=[] is valid)
            continue
        prefixed = [k.startswith(modality + ".") for k in keys or []]
        styles.add(None if not prefixed else "n1.5" if all(prefixed) else "n1.7" if not any(prefixed) else None)
    return styles


# ---------------------------------------------------------------- dataset rules

def _schema(report: _Report, modality) -> dict:
    """Validate meta/modality.json as LeRobotModalityMetadata.model_validate does (dataset.py:309, schema.py:45-119).

    Returns the sections that validate (state/action/video/annotation), so later rules only read those."""
    problems = []
    if not isinstance(modality, dict):
        problems.append("it is not a JSON object")
        modality = {}
    for side in ("state", "action", "video"):
        if side not in modality:
            problems.append(f"it has no {side} section (required, {SCHEMA}:104-115)")
        elif not isinstance(modality[side], dict):
            problems.append(f"its {side} section is not an object")
    if modality.get("annotation") is not None and not isinstance(modality.get("annotation"), dict):
        problems.append("its annotation section is neither an object nor null")
    for side in ("state", "action"):
        for group, entry in (modality.get(side) or {}).items() if isinstance(modality.get(side), dict) else ():
            where = f"{side}.{group}"
            if not isinstance(entry, dict):
                problems.append(f"{where} is not an object")
                continue
            for field in ("start", "end"):
                if field not in entry:
                    problems.append(f"{where} has no {field} ({SCHEMA}:57-64)")
                elif not _pyd_int(entry[field]):
                    problems.append(f"{where}.{field} = {entry[field]!r} is not an integer ({SCHEMA}:57-64)")
            rotation = entry.get("rotation_type")
            if rotation is not None and rotation not in ROTATIONS:
                problems.append(f"{where}.rotation_type = {rotation!r} is not one of {list(ROTATIONS)} "
                                f"({SCHEMA}:27-39, 65-67)")
            if "absolute" in entry and not _pyd_bool(entry["absolute"]):
                problems.append(f"{where}.absolute = {entry['absolute']!r} is not a boolean ({SCHEMA}:68)")
            if "dtype" in entry and not isinstance(entry["dtype"], str):
                problems.append(f"{where}.dtype = {entry['dtype']!r} is not a string ({SCHEMA}:69-72)")
            value = entry.get("range")
            if value is not None and not (isinstance(value, list) and len(value) == 2
                                          and all(isinstance(v, (int, float)) for v in value)):
                problems.append(f"{where}.range = {value!r} is not a pair of numbers ({SCHEMA}:73-76)")
            if "original_key" in entry and entry["original_key"] is not None and not isinstance(entry["original_key"],
                                                                                                    str):
                problems.append(f"{where}.original_key = {entry['original_key']!r} is not a string ({SCHEMA}:77-98)")
    for side in ("video", "annotation"):
        section = modality.get(side)
        for key, entry in section.items() if isinstance(section, dict) else ():
            if not isinstance(entry, dict):
                problems.append(f"{side}.{key} is not an object")
            elif entry.get("original_key") is not None and not isinstance(entry["original_key"], str):
                problems.append(f"{side}.{key}.original_key = {entry['original_key']!r} is not a string "
                                f"({SCHEMA}:45-51)")
    if problems:
        report.add("groot.modality_schema", "error",
                   f"meta/modality.json does not validate as N1.5's LeRobotModalityMetadata: {'; '.join(problems[:6])}"
                   + (f" (and {len(problems) - 6} more)" if len(problems) > 6 else "")
                   + f". N1.5 raises pydantic ValidationError when it opens the dataset ({DATASET}:308-309).",
                   ("meta/modality.json", "", problems[:20]),
                   remedy="Follow getting_started/LeRobot_compatible_data_schema.md: state, action and video "
                          "objects of {start, end, original_key?} / {original_key} entries.")
        return {}
    return modality


def _pyd_int(value) -> bool:
    """What pydantic 2.10 accepts for an int field in lax mode: ints, booleans, integral floats and integer strings
    (native schema probe, .validation/k4-groot-n15/out/n15_native.json schema_probe.start)."""
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        return re.fullmatch(r"\s*[+-]?\d+\s*", value) is not None
    return False


def _pyd_bool(value) -> bool:
    if isinstance(value, bool) or value in (0, 1):
        return True
    return isinstance(value, str) and value.strip().lower() in ("true", "false", "t", "f", "yes", "no", "y", "n",
                                                                 "on", "off", "1", "0")


def _int(value) -> int:
    return int(float(value)) if not isinstance(value, str) else int(value.strip())


def _info(report: _Report, info: dict) -> dict:
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2"):
        missing = [n for n in ("meta/episodes.jsonl", "meta/tasks.jsonl") if not report.file(n).is_file()]
        report.add("groot.format", "error" if missing else "warning",
                   f"meta/info.json declares codebase_version {version or 'none'}. {GROOT_N15} never reads the "
                   f"version; it opens the LeRobot v2 files meta/episodes.jsonl and meta/tasks.jsonl ({DATASET}:"
                   "402-404, 482-484)" + (f", and {missing} are missing." if missing else ", which are present."),
                   ("meta/info.json", "/codebase_version", version),
                   remedy="Convert the dataset to LeRobot v2 (for example with Isaac-GR00T N1.7's "
                          "scripts/lerobot_conversion/convert_v3_to_v2.py).")
    lines = {"data_path": 470, "video_path": 474, "chunks_size": 478}
    missing = [key for key in lines if key not in info]
    if missing:
        report.add("groot.metadata", "error",
                   f"meta/info.json has no {missing}; N1.5 reads data_path, video_path and chunks_size whenever it "
                   f"opens the dataset and raises KeyError ({DATASET}:" + ", ".join(str(lines[k]) for k in missing)
                   + ").", ("meta/info.json", "", sorted(info)), remedy="Restore the LeRobot v2 info.json fields.")
    features = info.get("features")
    return {k: v for k, v in features.items() if isinstance(v, dict)} if isinstance(features, dict) else {}


def _dtype_kind(dtype: str) -> str | None:
    """"float" / "other" for dtype names np.dtype accepts, None when this check does not know the name."""
    name = dtype.strip()
    return "float" if name in _FLOAT_DTYPES else "other" if name in _OTHER_DTYPES else None


def _groups(report: _Report, modality: dict, features: dict, side: str, config: dict | None,
            kept: set | None) -> dict:
    """State/action groups. N1.5 reads every entry when it opens the dataset (metadata and statistics,
    dataset.py:310-328, 374-388) and the data only for the config's keys (dataset.py:572-575, 757-779).
    `kept` is the set of meta/stats.json keys when N1.5 keeps that file, else None (it recomputes it)."""
    entries = modality.get(side) or {}
    used = None if config is None else [k[len(side) + 1:] for k in config[side]["keys"] if k.startswith(side + ".")]
    valid, covered = {}, {}
    if not entries:
        if used is None or used:
            report.add(f"groot.{side}_groups", "error" if used else "warning",
                       f"modality.json has no {side} groups; N1.5 reads {side} only through them ({DATASET}:310-328, "
                       f"757-779).", ("meta/modality.json", f"/{side}", entries),
                       remedy=f"Declare each {side} group with start/end indices into its column.")
        return {}
    for group, entry in entries.items():
        pointer = f"/{side}/{group}"
        evidence = [("meta/modality.json", pointer, entry)]
        is_used = used is None or group in used
        start, end = _int(entry["start"]), _int(entry["end"])
        column = entry.get("original_key", DEFAULT_COLUMNS[side])
        dtype = entry.get("dtype", "float64")
        kind = _dtype_kind(dtype)
        if kind is None:
            report.add(f"groot.{side}_dtype", "unknown",
                       f"{side}.{group} has dtype {dtype!r}; N1.5 passes it to np.dtype when it opens the dataset "
                       f"({DATASET}:316), which raises TypeError for a name numpy does not know. This check does "
                       "not know it either.", *evidence, remedy="Use a numpy dtype name such as float32.")
        if column is None:
            report.add(f"groot.{side}_column", "error",
                       f"{side}.{group} sets original_key to null; N1.5 looks up its statistics with that key and "
                       f"raises KeyError: None when it opens the dataset ({DATASET}:379-382).", *evidence,
                       remedy=f"Remove original_key (it defaults to {DEFAULT_COLUMNS[side]}) or name the column.")
            continue
        feature = features.get(column)
        width = _width(feature) if feature is not None else None
        if width is not None and end > start and (end > width or start < -width):  # np.arange leaves [-width, width)
            names = _names(feature, width)
            outside = ([f"end {end} is past the last index {width - 1}"] if end > width else []) \
                + ([f"start {start} is below -{width}"] if start < -width else [])
            report.add(f"groot.{side}_slice", "error",
                       f"{side}.{group} slices [{start}:{end}] of {column}, which has {width} values ("
                       + " and ".join(outside) + "). N1.5 indexes with np.arange(start, end): the statistics raise "
                       f"IndexError when the dataset is opened ({DATASET}:383-388), for every modality.json entry "
                       f"whether or not the config reads it, and the data does when a sample is read "
                       f"({DATASET}:775-779)" + (", although a negative start alone would wrap around" if start < 0
                                                 else "") + ". A modality.json copied from another robot does this.",
                       *evidence, ("meta/info.json", f"/features/{column}/shape", feature.get("shape")),
                       remedy=f"Rewrite the {side} slices for this robot's {width}-value {column}"
                              + (f" ({', '.join(names)})." if names else "."))
            continue
        if start < 0 or end <= start:
            problem = (f"{side}.{group} has start={start}, end={end}: N1.5 takes the values at np.arange(start, end) "
                       f"({DATASET}:383-388, 775-779), so "
                       + (f"the negative start wraps around to the last value(s) of {column} with no error"
                          if start < 0 and end > start else "the group is empty, with no error"))
            report.add(f"groot.{side}_slice", "error" if is_used else "pass",
                       (problem + "." if is_used else
                        f"{side}.{group} is not one of the config's {side} keys, so N1.5 reads only its statistics "
                        f"({DATASET}:374-388); noted only: {problem}."), *evidence,
                       remedy="Use integers with 0 <= start < end." if is_used else "")
            continue
        if feature is None:
            if kept is None:  # a stats.json N1.5 keeps but lacks the column is an error in _statistics
                report.add(f"groot.{side}_column", "warning",
                           f"{side}.{group} reads column {column}, which meta/info.json does not declare. N1.5 never "
                           f"reads info.json for it: it recomputes meta/stats.json from the parquet files, looks up "
                           f"{column} there when it opens the dataset ({DATASET}:370-382), whether or not the config "
                           f"reads {side}.{group}, and raises KeyError unless the parquet files hold the column. The "
                           "slice was not checked against a width.", *evidence,
                           remedy="Declare the column in meta/info.json with its shape, or remove the entry.")
            elif column in kept:
                report.add(f"groot.{side}_column", "pass",
                           f"{side}.{group} reads column {column}; meta/info.json does not declare it, but the "
                           f"meta/stats.json N1.5 keeps has its statistics ({DATASET}:374-388), so the parquet files "
                           "most likely hold it.", *evidence)
        elif width is None:
            report.add(f"groot.{side}_slice", "unknown", f"{column} has no one-dimensional shape in meta/info.json, so "
                       f"the {side}.{group} slice cannot be checked.", *evidence,
                       ("meta/info.json", f"/features/{column}", feature))
        if is_used:
            valid[group] = (column, start, end)
            if width is not None:
                for index in range(start, end):
                    covered.setdefault((column, index), []).append(group)
                names = _names(feature, width)
                if names:
                    report.add(f"groot.{side}_slice", "pass",
                               f"{side}.{group} reads {column}[{start}:{end}] = {names[start:end]}.", *evidence)
        elif width is not None:
            report.add(f"groot.{side}_slice", "pass", f"{side}.{group} is not one of the config's {side} keys; N1.5 "
                       f"still reads its statistics when it opens the dataset ({DATASET}:374-388), and its slice "
                       f"[{start}:{end}] fits {column}.", *evidence)
    for (column, index), owners in sorted(covered.items()):
        if len(owners) > 1:
            report.add(f"groot.{side}_overlap", "warning",
                       f"{column}[{index}] belongs to {owners}; N1.5 feeds it to the model in each group "
                       f"({DATASET}:775-779).", ("meta/modality.json", f"/{side}", entries),
                       remedy="Make groups disjoint unless the duplication is intended.")
    for column in sorted({c for c, _, _ in valid.values() if _width(features.get(c) or {}) is not None}):
        width = _width(features[column])
        names = _names(features[column], width)
        unused = [i for i in range(width) if (column, i) not in covered]
        if unused:
            label = [names[i] for i in unused] if names else unused
            report.add(f"groot.{side}_unused", "warning",
                       f"{column} values {label} are in no {side} group"
                       + (" the config reads" if used is not None else "") + ", so N1.5 never reads them.",
                       ("meta/modality.json", f"/{side}", entries), ("meta/info.json", f"/features/{column}",
                                                                      features[column]),
                       remedy="Add them to a group, or confirm they are meant to be ignored.")
    for group, (column, start, end) in valid.items():
        feature = features.get(column)
        names = _names(feature, _width(feature)) if feature is not None and _width(feature) is not None else None
        if not names:
            continue
        span = [n.lower() for n in names[start:end]]
        if "gripper" in group.lower() and not any("gripper" in n for n in span) and any("gripper" in n.lower()
                                                                                        for n in names):
            report.add(f"groot.{side}_names", "warning", f"{side}.{group} reads {names[start:end]}, none of which "
                       "is a gripper value.", ("meta/modality.json", f"/{side}/{group}", entries[group]),
                       remedy="Check the slice indices against meta/info.json names.")
        elif "gripper" not in group.lower() and any("gripper" in n for n in span) and \
                any("gripper" in other.lower() for other in entries if other != group and (used is None
                                                                                          or other in used)):
            report.add(f"groot.{side}_names", "warning", f"{side}.{group} reads a gripper value ({names[start:end]}) "
                       "although a separate gripper group exists.",
                       ("meta/modality.json", f"/{side}/{group}", entries[group]),
                       remedy="Check the slice indices against meta/info.json names.")
    return valid


def _stat_widths(stats, column):
    entry = stats.get(column) if isinstance(stats, dict) else None
    if not isinstance(entry, dict):
        return None
    return {n: len(v) if isinstance(v, list) else type(v).__name__ for n, v in entry.items()}


def _stat_valid(value) -> bool:
    """DatasetStatisticalValues.model_validate (schema.py:173-179): all six names present, each something numpydantic
    turns into an array. The native schema probe found lists, nested lists, scalars, strings and NaN accepted, and
    null and objects rejected; extra names such as LeRobot's count are accepted (and sliced later)."""
    return isinstance(value, dict) and all(name in value and value[name] is not None
                                           and not isinstance(value[name], dict) for name in STAT_NAMES)


def stats_kept(state: str, stats) -> bool:
    """Whether N1.5 keeps meta/stats.json as it is (dataset.py:359-366) rather than recomputing it."""
    return state == "ok" and isinstance(stats, dict) and all(_stat_valid(v) for v in stats.values())


def _statistics(report: _Report, state: str, stats, modality: dict, features: dict) -> None:
    """meta/stats.json as N1.5 reads it when the dataset is opened (dataset.py:359-388)."""
    uses = {}  # column -> [(group, start, end)] over every state/action entry, read or not by the config
    for side in ("state", "action"):
        for group, entry in (modality.get(side) or {}).items():
            column = entry.get("original_key", DEFAULT_COLUMNS[side])
            if isinstance(column, str):
                uses.setdefault(column, []).append((f"{side}.{group}", _int(entry["start"]), _int(entry["end"])))
    if state == "too large":
        report.add("groot.statistics", "unknown", f"meta/stats.json was not read ({stats}).", ("meta/stats.json", "", None))
        return
    if state == "unreadable":
        report.add("groot.statistics", "error",
                   f"meta/stats.json cannot be parsed ({stats}). N1.5 opens it with json.load and recomputes the "
                   f"statistics only on FileNotFoundError or a pydantic ValidationError ({DATASET}:361-366), so the "
                   "parse error stops every path when the dataset is opened.", ("meta/stats.json", "", None),
                   remedy="Delete meta/stats.json; N1.5 regenerates it from the parquet files.")
        return
    if state == "ok" and not isinstance(stats, dict):
        report.add("groot.statistics", "error",
                   f"meta/stats.json is a JSON {type(stats).__name__}, not an object; N1.5 calls .values() on it and "
                   f"fails when the dataset is opened ({DATASET}:364).", ("meta/stats.json", "", None),
                   remedy="Delete meta/stats.json; N1.5 regenerates it from the parquet files.")
        return
    invalid = sorted(k for k, v in stats.items() if not _stat_valid(v)) if state == "ok" else []
    if state == "missing" or invalid:
        reason = ("meta/stats.json is missing" if state == "missing" else
                  f"meta/stats.json entries {invalid[:5]} are not {list(STAT_NAMES)} arrays"
                  + (" (Isaac-GR00T N1.7 writes a __fingerprints__ entry, which is one)"
                     if "__fingerprints__" in invalid else "") + ", so N1.5's pydantic check fails")
        report.add("groot.statistics", "pass",
                   f"{reason}. N1.5 recomputes statistics for every parquet column and writes them to meta/stats.json "
                   f"when the dataset is first opened, on every path ({DATASET}:359-373); the dataset directory must "
                   "be writable.", ("meta/stats.json", "", invalid[:20] or None))
        images = sorted(k for k, v in features.items() if v.get("dtype") == "image")
        if images:
            report.add("groot.statistics", "warning",
                       f"meta/info.json declares {images} with dtype image, which LeRobot stores in the parquet files as "
                       f"{{bytes, path}} records. N1.5 recomputes statistics over every parquet column that is not a "
                       f"string with np.asarray(x, dtype=float32) and raises TypeError on such a column "
                       f"({DATASET}:75-85). If these cameras are stored only as videos, this does not apply.",
                       ("meta/info.json", "/features", images), remedy="Ship a meta/stats.json that N1.5 keeps (the six "
                       "statistics for every column, no other entries), or store the cameras as videos.")
        return
    for column, groups in uses.items():
        names = [g for g, _, _ in groups]
        if column not in stats:
            report.add("groot.statistics", "error",
                       f"meta/stats.json validates, so N1.5 keeps it, but it has no entry for {column}, which {names} "
                       f"read; N1.5 raises KeyError when it opens the dataset ({DATASET}:379-382), whether or not the "
                       "config reads these groups.", ("meta/stats.json", "", sorted(stats)[:20]),
                       remedy="Delete meta/stats.json so N1.5 regenerates it, or add the column's statistics.")
            continue
        widths = _stat_widths(stats, column)
        def outside(width, start, end):  # np.arange(start, end) indexes outside [-width, width)
            return end > start and (not isinstance(width, int) or width < end or start < -width)

        short = sorted({n for n, w in widths.items() for _, start, end in groups if outside(w, start, end)})
        if short:
            over = [f"{g} [{s}:{e}]" for g, s, e in groups if any(outside(widths[n], s, e) for n in short)]
            report.add("groot.statistics", "error",
                       f"meta/stats.json {column} statistics { {n: widths[n] for n in short} } are narrower than "
                       f"{over}. N1.5 keeps the file because it validates, and indexes every statistic in the entry "
                       f"with np.arange(start, end), raising IndexError when it opens the dataset ({DATASET}:382-388)"
                       + (" (LeRobot's count field is indexed too)" if "count" in short else "") + ".",
                       ("meta/stats.json", f"/{column}", widths),
                       remedy="Delete meta/stats.json so N1.5 regenerates it from the parquet files.")
            continue
        feature = features.get(column)
        width = _width(feature) if feature is not None else None
        wide = {n: w for n, w in widths.items() if n in STAT_NAMES and isinstance(w, int) and width is not None
                and w != width}
        if wide:
            report.add("groot.statistics", "error",
                       f"meta/stats.json {column} statistics have widths {wide}, but meta/info.json gives {column} "
                       f"{width} values. N1.5 keeps a stats.json that validates (there is no fingerprint, "
                       f"{DATASET}:359-366) and slices it with the groups' indices ({DATASET}:383-388), so {names} are "
                       "normalized with another vector's statistics, with no error.",
                       ("meta/stats.json", f"/{column}", widths), ("meta/info.json", f"/features/{column}", feature),
                       applies_to=FT, remedy="Delete meta/stats.json so N1.5 regenerates it from the parquet files.")



def _cameras(report: _Report, modality: dict, features: dict, config: dict | None) -> dict:
    """Video entries: N1.5 reads every entry's info.json feature when it opens the dataset (dataset.py:337-357) and
    pairs config cameras with entries by name (dataset.py:488-505, 658-666)."""
    entries = modality.get("video") or {}
    wanted = [k[len("video."):] for k in config["video"]["keys"] if k.startswith("video.")] if config else None
    resolutions = {}
    for key, entry in entries.items():
        column = key if entry.get("original_key") is None else entry["original_key"]  # dataset.py:339-341
        pointer = ("meta/modality.json", f"/video/{key}", entry)
        feature = features.get(column)
        if feature is None:
            report.add("groot.video_column", "error",
                       f"video.{key} reads {column}"
                       + ("" if entry.get("original_key") else " (with no original_key N1.5 uses the entry's own name)")
                       + f", which meta/info.json does not declare; N1.5 raises KeyError when it opens the dataset "
                       f"({DATASET}:339-342), whether or not the config reads this camera.", pointer,
                       ("meta/info.json", "/features", sorted(features)),
                       remedy="Point original_key at a declared video feature, e.g. observation.images." + key + ".")
            continue
        problem, resolution = _video_feature(feature)
        if problem:
            report.add("groot.video_column", "error",
                       f"video.{key} reads {column}, whose meta/info.json feature {problem}; N1.5 fails when it opens "
                       f"the dataset ({DATASET}:343-357, {SCHEMA}:191-196), whether or not the config reads this camera.",
                       pointer, ("meta/info.json", f"/features/{column}", feature),
                       remedy="Give the feature LeRobot v2 video metadata: shape with names height/width, and "
                              "info.video.channels / info.video.fps (or a 'channel' name and video_info.video.fps).")
            continue
        resolutions[key] = (column, resolution)
    if wanted is None:
        return resolutions
    missing = [k for k in wanted if k not in entries]
    if missing:
        report.add("groot.video_mapping", "error",
                   f"Config cameras {['video.' + k for k in missing]} are not modality.json video entries "
                   f"{sorted(entries)}; N1.5 pairs cameras by name only and raises ValueError when it opens the "
                   f"dataset ({DATASET}:488-505, {SCHEMA}:151-156). (Isaac-GR00T N1.7 would pair them by position.)",
                   ("meta/modality.json", "/video", sorted(entries)),
                   remedy="First choose or review a training config matching this dataset's actual cameras. Do not rename a public dataset to silence the check. Only change an owned separate candidate's mapping when the independently reviewed recipe and camera sources require it.")
    present = [k for k in wanted if k in entries]
    reads = {k: resolutions[k][0] for k in present if k in resolutions}
    if len(present) < 1 or not reads:
        return resolutions
    columns = [k if e.get("original_key") is None else e["original_key"] for k, e in entries.items()]
    by_column = _designate(present, _distinct(list(dict.fromkeys(columns))))
    owner = {column: camera for camera, column in by_column.items()}
    swapped = {}
    for camera in present:
        column = reads.get(camera)
        if column is None:
            continue
        if owner.get(column, camera) != camera:
            swapped[camera] = f"{column} is named for config camera {owner[column]}"
        elif camera in by_column and by_column[camera] != column:
            swapped[camera] = f"the video named for it is {by_column[camera]}"
    pairs = ", ".join(f"video.{k} reads {reads[k]}" for k in present if k in reads)
    message = (f"Config cameras are paired with modality.json video entries by name, and each reads the video of its "
               f"entry's original_key ({DATASET}:658-666), with no warning: {pairs}")
    if swapped:
        message += ". Swapped: " + "; ".join(f"video.{k} ({why})" for k, why in swapped.items()) + "."
    report.add("groot.video_mapping", "error" if swapped else "pass", message if swapped else message + ".",
               ("meta/modality.json", "/video", entries),
               remedy="Point each camera entry's original_key at its own video." if swapped else "")
    return resolutions


def _video_feature(feature: dict):
    """Why N1.5 cannot build VideoMetadata from this feature (dataset.py:343-357, schema.py:191-196), and (w, h)."""
    shape, names = feature.get("shape"), feature.get("names")
    if not isinstance(shape, list) or not isinstance(names, list) or "height" not in names or "width" not in names:
        return "has no shape with names 'height' and 'width'", None
    try:
        height, width = shape[names.index("height")], shape[names.index("width")]
    except IndexError:
        return "has fewer shape entries than names", None
    channels = fps = None
    video_info = feature.get("video_info")
    if "channel" in names and "video_info" in feature and not isinstance(video_info, dict):
        # dataset.py:346-350 catches only ValueError and KeyError: None["video.fps"] raises TypeError
        return ("has a 'channel' name and a video_info that is " + ("null" if video_info is None else
                                                                   f"a {type(video_info).__name__}, not an object")
                + f", so N1.5 raises TypeError reading video_info['video.fps'] ({DATASET}:346-348)"), None
    if "channel" in names and isinstance(video_info, dict) and "video.fps" in video_info:
        index = names.index("channel")
        channels, fps = (shape[index] if index < len(shape) else None), video_info["video.fps"]
    else:
        extra = feature.get("info")
        if not isinstance(extra, dict) or "video.channels" not in extra or "video.fps" not in extra:
            return ("has neither a 'channel' name with video_info.video.fps nor info.video.channels and "
                    "info.video.fps"), None
        channels, fps = extra["video.channels"], extra["video.fps"]
    # pydantic lax mode (native schema probe): integral floats pass as ints, numeric strings as fps.
    if not (_pyd_int(channels) and _int(channels) > 0) or _positive(fps) is None:
        return f"gives channels={channels!r} and fps={fps!r} (VideoMetadata needs channels > 0 and fps > 0)", None
    if not (_pyd_int(width) and _pyd_int(height)):
        return f"gives a non-integer resolution {width!r}x{height!r}", None
    return None, (_int(width), _int(height))


def _positive(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _language(report: _Report, modality: dict, features: dict, stats_keys: set, config: dict | None) -> list:
    if config is None:
        return []
    keys = config["language"]["keys"]
    annotation = modality.get("annotation")
    if len(keys) > 1:
        report.add("groot.language_key", "error",
                   f"The config has {len(keys)} language keys {keys}; N1.5's GR00TTransform asserts there is exactly "
                   f"one when fine-tuning reads a sample ({MODEL_TF}:166-169). Direct loaders read them all "
                   f"({DATASET}:572-575).", ("meta/modality.json", "/annotation", None), applies_to=FT,
                   remedy="Keep one language key in the config.")
    for key in keys:
        if not key.startswith("annotation."):
            report.add("groot.language_key", "error",
                       f"Config language key {key} does not start with 'annotation.'; N1.5 raises ValueError when it "
                       f"opens the dataset ({DATASET}:488-505, {SCHEMA}:166-167) and asserts the prefix when it reads "
                       f"language ({DATASET}:820-822). (Isaac-GR00T N1.7's task/sub_task keys do not exist in N1.5.)",
                       ("meta/modality.json", "/annotation", None),
                       remedy="Use annotation.<modality.json annotation entry>.")
            continue
        subkey = key[len("annotation."):]
        if not isinstance(annotation, dict) or subkey not in annotation:
            report.add("groot.language_key", "error",
                       f"Config language key {key} has no modality.json annotation entry {subkey}; N1.5 "
                       + ("asserts the annotation section exists" if not isinstance(annotation, dict) else
                          "raises ValueError") + f" when it opens the dataset ({DATASET}:488-505, {SCHEMA}:157-165).",
                       ("meta/modality.json", "/annotation", sorted(annotation) if isinstance(annotation, dict)
                        else annotation), remedy="First select or review a training config matching the dataset's existing language annotation. "
                        "Do not edit a public dataset merely to silence this check. Only if the reviewed owned recipe "
                        f"requires annotation.{subkey}, add a source-backed mapping in a separate candidate and recheck it.")
            continue
        entry = annotation[subkey]
        column = key if entry.get("original_key") is None else entry["original_key"]  # dataset.py:830-832
        pointer = ("meta/modality.json", f"/annotation/{subkey}", entry)
        feature = features.get(column)
        if feature is None:
            if column in stats_keys:
                report.add("groot.annotation_column", "pass",
                           f"The config's language key {key} reads column {column}. meta/info.json does not declare "
                           "it, but meta/stats.json has statistics for it, so the parquet files most likely hold it "
                           f"({DATASET}:829-835).", pointer)
            else:
                report.add("groot.annotation_column", "warning",
                           f"The config's language key {key} reads column {column}, which neither meta/info.json nor "
                           f"meta/stats.json lists. N1.5 reads it from the parquet files and fails with KeyError only "
                           f"if they lack it ({DATASET}:834); a LeRobot v2 info.json normally lists every parquet "
                           "column.", pointer,
                           remedy="Confirm the parquet files hold this column, or point original_key at task_index.")
            continue
        size, dtype = _size(feature), feature.get("dtype")
        if size is not None and size > 1:
            report.add("groot.annotation_values", "error",
                       f"The config's language key {key} reads {column}, a {feature.get('shape')} {dtype} vector; "
                       f"N1.5 calls .item() on each value and fails when a sample is read ({DATASET}:834).",
                       pointer, ("meta/info.json", f"/features/{column}", feature),
                       remedy="Point original_key at task_index or another integer task-index column.")
        elif "int" not in str(dtype):
            report.add("groot.annotation_values", "warning",
                       f"The config's language key {key} reads {column} of dtype {dtype!r}; N1.5 looks each value up "
                       f"in meta/tasks.jsonl by task_index ({DATASET}:834-835).", pointer,
                       ("meta/info.json", f"/features/{column}", feature),
                       remedy="Point original_key at task_index or another integer task-index column.")
        else:
            report.add("groot.annotation_column", "pass", f"The config's language key {key} reads {column} "
                       f"({dtype}), looked up in meta/tasks.jsonl ({DATASET}:829-835).", pointer)
    return keys


def _metadata_lines(report: _Report, info: dict) -> None:
    seen = set()
    # trajectory ids are formatted into data_path, {episode_index:06d} in LeRobot v2 (dataset.py:584-586)
    formatted = re.search(r"\{episode_index:[^}]*d\}", str(info.get("data_path", ""))) is not None

    def episode(row):
        kinds = []
        index = row.get("episode_index")
        if type(index) is int:
            if index in seen:
                kinds.append("an episode_index of an earlier line")
            seen.add(index)
        elif "episode_index" in row and formatted and type(index) is not bool:  # numpy casts bools among ints
            kinds.append("an episode_index that is not an integer")
        if "episode_index" not in row:
            kinds.append("no episode_index")
        if "length" not in row:
            kinds.append("no length")
        elif type(row["length"]) is not int:
            kinds.append("length is not an integer")
        return kinds

    def task(row):
        return [] if "task_index" in row and "task" in row else ["no task_index or task"]

    consequences = {
        ("episodes", "blank line"): f"N1.5 parses every line as JSON and fails ({DATASET}:403-404).",
        ("episodes", "not JSON"): f"N1.5 parses every line as JSON and fails ({DATASET}:403-404).",
        ("episodes", "not a JSON object"): f"N1.5 indexes each row by key and fails ({DATASET}:408).",
        ("episodes", "not UTF-8 text"): f"N1.5 reads it as text and fails ({DATASET}:403-404).",
        ("episodes", "no episode_index"): f"N1.5 reads every row's episode_index when it opens the dataset and raises "
                                          f"KeyError ({DATASET}:408).",
        ("episodes", "no length"): f"N1.5 reads every row's length when it opens the dataset and raises KeyError "
                                   f"({DATASET}:409).",
        ("episodes", "length is not an integer"): f"N1.5 iterates range(length) when it opens the dataset and fails "
                                                  f"on anything but an integer ({DATASET}:427-430).",
        ("episodes", "an episode_index of an earlier line"): f"N1.5 then finds two trajectories with that index and "
                                                             f"raises ValueError whenever a sample of that episode is "
                                                             f"read, on every path ({DATASET}:590-604, 689).",
        ("episodes", "an episode_index that is not an integer"): f"N1.5 keeps the ids in one numpy array and formats "
                                                                f"them into meta/info.json data_path with an integer "
                                                                f"format, which raises ValueError when a sample is "
                                                                f"read, on every path ({DATASET}:405-411, 584-586).",
        ("tasks", "blank line"): f"N1.5 parses every line as JSON and fails ({DATASET}:483-484).",
        ("tasks", "not JSON"): f"N1.5 parses every line as JSON and fails ({DATASET}:483-484).",
        ("tasks", "not a JSON object"): f"N1.5 builds a task table indexed by task_index and fails ({DATASET}:485-486).",
        ("tasks", "not UTF-8 text"): f"N1.5 reads it as text and fails ({DATASET}:483-484).",
        ("tasks", "no task_index or task"): f"N1.5 indexes the task table by task_index when it opens the dataset "
                                            f"({DATASET}:486) and reads its task column for each instruction "
                                            f"({DATASET}:835); a missing field raises KeyError.",
    }
    for kind, check in (("episodes", episode), ("tasks", task)):
        name = f"meta/{kind}.jsonl"
        path = report.file(name)
        if not path.is_file():
            report.add("groot.metadata", "error", f"{name} is missing; N1.5 opens it whenever it opens the dataset "
                       f"({DATASET}:" + ("402-404" if kind == "episodes" else "482-484") + ").", (name, "", None),
                       remedy="Convert the dataset to LeRobot v2 metadata.")
            continue
        scan = _scan_jsonl(path, check)
        for problem, numbers in scan["problems"].items():
            report.add(f"groot.{kind}", "error", f"{name}: {len(numbers)} line(s) with {problem} (first {numbers[:5]}). "
                       + consequences[kind, problem], (name, "", numbers[:5]),
                       remedy="Write one JSON object per line, with no blank lines"
                              + (", each with its own integer episode_index and an integer length."
                                 if kind == "episodes" else
                                 ", each with task_index and task."))
        if scan["truncated"]:
            report.add(f"groot.{kind}", "unknown", f"{name} has more than {MAX_LINES} lines; only the first {MAX_LINES} "
                       "were checked.", (name, "", MAX_LINES))


def _config_keys(report: _Report, modality: dict, config: dict) -> None:
    """Config keys must be <modality>.<modality.json entry> (dataset.py:488-505, schema.py:121-167)."""
    for side in ("state", "action", "video"):
        entries = modality.get(side) or {}
        unprefixed = [k for k in config[side]["keys"] if not k.startswith(side + ".")]
        if unprefixed:
            report.add("groot.config_key", "error",
                       f"Config {side} keys {unprefixed} do not start with '{side}.'; N1.5 config keys name the "
                       f"modality ({DATASET}:694, 757) and the dataset fails to open ({DATASET}:488-505, "
                       f"{SCHEMA}:136-167). (Isaac-GR00T N1.7 configs list bare group names.)",
                       ("meta/modality.json", f"/{side}", sorted(entries)),
                       remedy=f"Write the keys as {side}.<modality.json {side} entry>.")
        missing = [k for k in config[side]["keys"] if k.startswith(side + ".") and k[len(side) + 1:] not in entries]
        if missing and side != "video":
            report.add(f"groot.{side}_config", "error",
                       f"Config {side} keys {missing} are not modality.json {side} entries; N1.5 raises ValueError when "
                       f"it opens the dataset ({DATASET}:488-505, {SCHEMA}:139-150).",
                       ("meta/modality.json", f"/{side}", sorted(entries)),
                       remedy=f"First select or review a training config matching the dataset's existing {side} entries. "
                              "Do not edit a public dataset merely to silence this check. Only when a reviewed owned "
                              f"recipe requires it, add source-backed {side} entries to a separate candidate or correct the config keys.")
        dotted = [k for k in config[side]["keys"] if k.startswith(side + ".") and k.count(".") > 1]
        transform = config.get("transform") or {}
        if dotted and (transform.get("concat") or transform.get("video_crop")):
            report.add("groot.config_key", "error",
                       f"Config {side} keys {dotted} contain more than one '.'; fine-tuning's transforms split keys "
                       f"into exactly two parts and fail when the dataset is opened ({CONCAT_TF}:76-88, "
                       f"{STATE_TF}:391-393, {VIDEO_TF}:130-132).", ("meta/modality.json", f"/{side}", sorted(entries)),
                       applies_to=FT, remedy=f"Rename the {side} entries without dots.")


def _transform_checks(report: _Report, modality: dict, config: dict, groups: dict, cameras: dict) -> None:
    """Fine-tuning transforms built by the config's transform() (scripts/gr00t_finetune.py:202)."""
    facts = config.get("transform")
    if facts is None:
        report.add("groot.config", "error",
                   f"Data config class {config['class']} defines no transform(); BaseDataConfig.transform is abstract, "
                   f"so N1.5 cannot instantiate the class ({DATA_CONFIG}:38-66, 118).", ("meta/modality.json", "", None),
                   applies_to=FT, remedy="Define transform() (copy one from gr00t/experiment/data_config.py).")
        return
    if facts["unread"]:
        report.add("groot.config", "unknown", f"Parts of {config['class']}.transform() are not literal: "
                   f"{sorted(set(facts['unread']))}; the checks that depend on them were skipped.",
                   ("meta/modality.json", "", None), applies_to=FT)
    entries = {side: modality.get(side) or {} for side in ("state", "action", "video")}
    for call in facts["state_action"]:
        _state_action_call(report, entries, config, call)
    _concat_checks(report, config, facts)
    for field, attr, side, line in (("state_horizon", "observation_indices", "state", 252),
                                    ("action_horizon", "action_indices", "action", 283)):
        horizon, steps = facts[field], config[side]["delta_indices"]
        if horizon is not None and steps is not None and config[side]["keys"] and horizon != len(steps):
            report.add("groot.horizon", "error",
                       f"The config's GR00TTransform has {field}={horizon}, but its {attr} has {len(steps)} steps, "
                       f"so each sample's {side} has {len(steps)} rows; GR00TTransform asserts they match when "
                       f"fine-tuning reads a sample ({MODEL_TF}:{line}).", ("meta/modality.json", f"/{side}", None),
                       applies_to=FT, remedy=f"Pass {field}=len(self.{attr}) to GR00TTransform.")
    widths = {}
    for side in ("state", "action"):
        total, unknown = 0, []
        for key in config[side]["keys"]:
            sub = key[len(side) + 1:] if key.startswith(side + ".") else None
            group = groups.get(side, {}).get(sub)
            if group is None:
                unknown.append(key)
                continue
            width = group[2] - group[1]
            rotation = facts["target_rotations"].get(key)
            if rotation is not None and rotation == (entries[side].get(sub) or {}).get("rotation_type"):
                rotation = None  # same representation: no rotation transformer (state_action.py:428), width kept
            if rotation is not None:
                width = 3 if rotation.startswith("euler_angles") else ROTATION_WIDTH.get(rotation)
                if width is None or key in facts["sincos"]:
                    unknown.append(key)
                    continue
            if key in facts["sincos"]:
                width *= 2
            total += width
        widths[side] = (total, unknown)
    state_total, state_unknown = widths["state"]
    limit = facts["max_state_dim"]
    if limit is not None and not state_unknown and state_total > limit:
        report.add("groot.state_width", "error",
                   f"The config's state keys give the model {state_total} values after its transforms"
                   + (" (sin-cos doubles " + ", ".join(k for k in config["state"]["keys"] if k in facts["sincos"]) + ")"
                      if any(k in facts["sincos"] for k in config["state"]["keys"]) else "")
                   + f", more than max_state_dim={limit}. GR00TTransform keeps the first {limit} and drops the rest, "
                   f"with no error ({MODEL_TF}:256-259).", ("meta/modality.json", "/state", sorted(entries["state"])),
                   applies_to=FT, remedy="Raise max_state_dim in the config's GR00TTransform, or read fewer state values.")
    action_total, action_unknown = widths["action"]
    limit = facts["max_action_dim"]
    if limit is not None and not action_unknown and action_total > limit \
            and not any(k in facts["target_rotations"] for k in config["action"]["keys"]):
        report.add("groot.action_width", "error",
                   f"The config's action keys give {action_total} values, more than max_action_dim={limit}; "
                   f"GR00TTransform asserts when fine-tuning reads a sample ({MODEL_TF}:288-290).",
                   ("meta/modality.json", "/action", sorted(entries["action"])), applies_to=FT,
                   remedy="Raise max_action_dim in the config's GR00TTransform, or read fewer action values.")
    if facts["video_crop"]:
        used = {k: cameras[k[len("video."):]] for k in config["video"]["keys"] if k[len("video."):] in cameras}
        sizes = {k: v[1] for k, v in used.items()}
        if len(set(sizes.values())) > 1:
            report.add("groot.video_resolution", "error",
                       f"The config's cameras have different resolutions in meta/info.json "
                       f"{ {k: f'{w}x{h}' for k, (w, h) in sizes.items()} }; fine-tuning's VideoCrop asserts all "
                       f"cameras share one resolution ({VIDEO_TF}:260-263).", ("meta/info.json", "/features",
                                                                               {k: v[0] for k, v in used.items()}),
                       applies_to=FT, remedy="Resize the videos to one resolution, or give the cameras separate crops.")


def _state_action_call(report: _Report, entries: dict, config: dict, call: dict) -> None:
    """One StateActionTransform as set_metadata reads it when fine-tuning opens the dataset (state_action.py:386-472),
    and its normalizers when a sample is read (:107-189). A call in a branch may not run: its faults are unknown."""
    apply_to, modes, rotations = call["apply_to"], call["normalization"], call["target_rotations"]
    if apply_to is None:
        return
    modes, rotations = modes or {}, rotations or {}
    status = "unknown" if call["conditional"] else "error"
    branch = "If this StateActionTransform is built (it sits in a branch, loop or helper): " \
        if call["conditional"] else ""
    loaded = set(config["state"]["keys"]) | set(config["action"]["keys"])

    def entry_of(key):
        side, _, sub = key.partition(".")
        return entries[side].get(sub) if side in entries else None

    def add(code, message, key, remedy):
        side, _, sub = key.partition(".")
        report.add(code, status, branch + message, ("meta/modality.json", f"/{side}/{sub}", entry_of(key)),
                   applies_to=FT, remedy=remedy)

    keys = []  # problems with which keys the transform names, in the order set_metadata meets them
    for key in apply_to:
        if key.count(".") != 1:
            keys.append(f"apply_to names {key}, which is not 'modality.key' (AssertionError, "
                        f"{STATE_TF}:391-394)")
        elif key.partition(".")[0] not in entries:
            keys.append(f"apply_to names {key}, but N1.5 DatasetModalities has only state, action and video, not "
                        f"{key.partition('.')[0]} (AssertionError '... config not found', {STATE_TF}:395-396)")
        elif entry_of(key) is None and key not in loaded:  # a config key is already reported by groot.<side>_config
            keys.append(f"apply_to names {key}, which has no modality.json entry (AssertionError '... config not "
                        f"found', {STATE_TF}:391-399)")
    for key in modes:
        if key.count(".") != 1:
            keys.append(f"normalization_modes names {key}, which is not 'modality.key' (AssertionError, "
                        f"{STATE_TF}:403-405)")
        elif key.partition(".")[0] not in ("state", "action"):
            keys.append(f"normalization_modes names {key}, but N1.5 DatasetStatistics has only state and action "
                        f"(AssertionError '... statistics not found', {STATE_TF}:403-407)")
        elif entry_of(key) is None and not (key in apply_to and key in loaded):  # else groot.<side>_config says it
            keys.append(f"normalization_modes names {key}, which has no modality.json entry and so no statistics "
                        f"(AssertionError '... statistics not found', {STATE_TF}:404-408)")
    for key, rep in rotations.items():
        if key not in apply_to:
            keys.append(f"target_rotations names {key}, which is not in its apply_to {apply_to} (KeyError, "
                        f"{STATE_TF}:419-421)")
        elif rep not in ROTATIONS and (entry_of(key) or {}).get("rotation_type") is not None:  # :422 comes first
            keys.append(f"target_rotations converts {key} to {rep!r}, which is not one of {list(ROTATIONS)} "
                        f"(ValueError, {STATE_TF}:423-425)")
    for key in modes:
        if entry_of(key) is not None and key not in apply_to:
            keys.append(f"normalization_modes names {key}, which is not in its apply_to {apply_to} (KeyError, "
                        f"{STATE_TF}:458-460)")
    if keys:
        report.add("groot.transform_keys", status,
                   branch + "A StateActionTransform in the config's transform() fails when fine-tuning opens the "
                   "dataset (set_transforms_metadata, gr00t/data/dataset.py:169): " + "; ".join(keys) + ".",
                   ("meta/modality.json", "", sorted(apply_to)), applies_to=FT,
                   remedy="Give apply_to only modality.key names present in meta/modality.json. Name only keys of "
                          "that transform's apply_to in its normalization_modes and target_rotations "
                          "(an inherited state_normalization_modes / state_target_rotations dict may name another "
                          "robot's keys). StateActionTransform metadata supports state/action/video only, and "
                          "normalization statistics support state/action only.")
    for key in apply_to:
        if not key.startswith("video.") or key not in config["video"]["keys"] or entry_of(key) is None:
            continue
        kind = call.get("video_inputs", {}).get(key, "unknown")
        if kind == "torch":
            continue
        report.add("groot.transform_dtype", status if kind == "numpy" else "unknown",
                   branch + f"StateActionTransform applies to {key}, which requires a torch dtype when a sample is "
                   f"read ({STATE_TF}:474-483). " + (
                       "The returned literal transform list supplies NumPy video here; fine-tuning raises an input "
                       "dtype assertion." if kind == "numpy" else
                       "Its input dtype is not established: this check cannot prove the returned transform order "
                       "and the preceding conversion for this video key."),
                   ("meta/modality.json", "/video", key), applies_to=FT,
                   remedy="Place StateActionTransform only after VideoToTensor for that key and before VideoToNumpy, "
                          "or remove the unnecessary transform. For a dynamic list, verify the actual returned order.")
    converted = set()
    for key, rep in rotations.items():
        entry = entry_of(key)
        if key not in apply_to or entry is None:
            continue
        side, _, sub = key.partition(".")
        source = entry.get("rotation_type")
        if source is None:
            add("groot.rotation", f"The config converts {key} to {rep}, but modality.json {side}.{sub} has no "
                f"rotation_type; fine-tuning asserts a source rotation type when the dataset is opened "
                f"({STATE_TF}:419-422).", key, f"Set {side}.{sub}.rotation_type to the stored representation (e.g. "
                "quaternion).")
            continue
        if rep not in ROTATIONS:  # reported above (ValueError, :423-425)
            continue
        if source == rep:  # no rotation transformer is built (state_action.py:427-431); it is normalized as is
            continue
        converted.add(key)
        if key not in modes:
            continue
        if not _pyd_truth(entry.get("absolute", True)):
            add("groot.rotation", f"The config converts {key} from {source} to {rep} and normalizes it, but "
                f"modality.json marks it relative (absolute: false); fine-tuning raises ValueError when the dataset is "
                f"opened ({STATE_TF}:440-457).", key, "Drop its normalization mode or mark it absolute.")
        elif modes[key] != "min_max":
            add("groot.rotation", f"The config converts absolute {key} from {source} to {rep} and normalizes it with "
                f"{modes[key]!r}; fine-tuning asserts converted absolute rotations use min_max when the dataset is "
                f"opened ({STATE_TF}:441-446).", key, f"Normalize {key} with min_max (its bounds come from {rep}).")
    for key, mode in modes.items():
        entry = entry_of(key)
        if entry is None or key not in apply_to or key in converted:
            continue
        side, _, sub = key.partition(".")
        if _dtype_kind(entry.get("dtype", "float64")) == "other" and mode != "binary":
            add("groot.normalization", f"The config normalizes {key} with {mode}, but modality.json declares "
                f"{side}.{sub} with dtype {entry.get('dtype')!r}, which N1.5 treats as not continuous "
                f"({DATASET}:316-320); fine-tuning raises ValueError when the dataset is opened ({STATE_TF}:458-465).",
                key, f"Declare a float dtype for {side}.{sub}, or normalize it with binary.")
        elif mode not in NORMALIZATION_MODES and key in loaded:
            add("groot.normalization", f"The config normalizes {key} with {mode!r}, which is not one of "
                f"{list(NORMALIZATION_MODES)}; N1.5 accepts the name when the dataset is opened and raises ValueError "
                f"('Invalid normalization mode') when fine-tuning reads a sample ({STATE_TF}:107-189).", key,
                f"Use one of {list(NORMALIZATION_MODES)}.")


def _concat_checks(report: _Report, config: dict, facts: dict) -> None:
    """ConcatTransform keeps only the keys it lists (concat.py:74-155): a loaded state group or camera it leaves out
    never reaches the model, with no error; a listed key that is not loaded asserts when a sample is read."""
    orders = facts["concat_orders"]
    for call in orders:
        unsure = call["conditional"] or len(orders) > 1 or facts["other_calls"]
        status = "unknown" if unsure else "error"
        why = ("" if not unsure else "If this ConcatTransform is the one that runs (it sits in a branch or helper, "
               "or transform() builds several, or calls " + (f"{facts['other_calls'][:5]}, " if facts["other_calls"]
                                                             else "") + "which may change the keys): ")
        for side in ("video", "state", "action"):
            order, wanted = call[side], config[side]["keys"]
            if order == _UNREAD or not wanted:
                continue
            field = f"{side}_concat_order"
            evidence = ("meta/modality.json", f"/{side}", {"config": wanted, field: order})
            if order is None:
                report.add("groot.concat", status, why + (
                    f"The config's ConcatTransform has no {field}, which is a required field, so building "
                    f"transform() raises a pydantic ValidationError ({CONCAT_TF}:36-40)." if side == "video" else
                    f"The config's ConcatTransform has no {field}, so it asserts {field} is not None when fine-tuning "
                    f"reads a sample ({CONCAT_TF}:" + ("117" if side == "state" else "139") + ")."), evidence,
                    applies_to=FT, remedy=f"Pass {field}=self.{side}_keys.")
                continue
            omitted = [k for k in wanted if k not in order]
            extra = [k for k in order if k not in wanted]
            if side == "action" and (omitted or extra):
                report.add("groot.concat", status, why + f"The config's ConcatTransform action_concat_order {order} is "
                           f"not the set of action keys it loads {wanted}; ConcatTransform asserts they are equal when "
                           f"fine-tuning reads a sample ({CONCAT_TF}:139-143).", evidence, applies_to=FT,
                           remedy="Pass action_concat_order=self.action_keys.")
                continue
            if extra:
                report.add("groot.concat", status, why + f"The config's ConcatTransform {field} lists {extra}, which "
                           f"the config does not load as {side} keys {wanted}; ConcatTransform asserts when "
                           f"fine-tuning reads a sample ({CONCAT_TF}:" + ("96-98" if side == "video" else "118-120")
                           + ").", evidence, applies_to=FT, remedy=f"Pass {field}=self.{side}_keys.")
            if omitted:
                report.add("groot.concat", status, why + (
                    f"The config loads cameras {omitted}, but its ConcatTransform video_concat_order {order} leaves them "
                    f"out: they are decoded for every sample and never reach the model, with no error "
                    f"({CONCAT_TF}:91-110)." if side == "video" else
                    f"The config loads and normalizes state keys {omitted}, but its ConcatTransform state_concat_order "
                    f"{order} leaves them out of the state vector the model receives, with no error "
                    f"({CONCAT_TF}:113-133)."), evidence, applies_to=FT,
                    remedy=f"Pass {field}=self.{side}_keys, or drop {omitted} from the config's {side}_keys.")


def _pyd_truth(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "t", "yes", "y", "on", "1")
    return bool(value)


def check(dataset: str | Path, modality_config: str | Path | None = None, config_name: str | None = None) -> dict:
    """The N1.5 profile of check_groot_dataset."""
    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Dataset directory not found: {root}")
    report = _Report(root)
    config = None
    if modality_config is not None or config_name is not None:
        config = read_data_config(modality_config, config_name)
    info = read_json(report.file("meta/info.json"))
    features = _info(report, info)
    modality_path = report.file("meta/modality.json")
    if not modality_path.is_file():
        report.add("groot.modality", "error", f"meta/modality.json is missing; N1.5 asserts it exists when it opens "
                   f"the dataset ({DATASET}:301-304).", ("meta/modality.json", "", None),
                   remedy="Write one following getting_started/LeRobot_compatible_data_schema.md.")
        return _finish(report, root, config)
    modality = _schema(report, read_json(modality_path))
    stats_state, stats = _read_cache(report.file("meta/stats.json"))
    usable = config is not None and not config["unsupported"] and not config["missing"]
    if config is not None and config["missing"]:
        report.add("groot.config", "error",
                   f"Data config class {config['class']} has no {config['missing']}; BaseDataConfig.modality_config() "
                   f"reads all six and raises AttributeError ({DATA_CONFIG}:40-62).", ("meta/modality.json", "", None),
                   remedy="Set video_keys, state_keys, action_keys, language_keys, observation_indices and "
                          "action_indices on the class.")
    if config is not None and config["unsupported"]:
        report.add("groot.config", "unknown",
                   f"Data config class {config['class']} cannot be read statically: {config['unsupported']}. "
                   "Config keys, cameras, language and the fine-tuning transforms were not checked.",
                   ("meta/modality.json", "", None),
                   remedy="Write the six key/index attributes as literal lists on the class or its built-in base.")
    if not modality:
        return _finish(report, root, config)
    run_config = config if usable else None
    if run_config is not None:
        _config_keys(report, modality, run_config)
    kept = set(stats) if stats_kept(stats_state, stats) else None
    groups = {side: _groups(report, modality, features, side, run_config, kept)
              for side in ("state", "action")}
    cameras = _cameras(report, modality, features, run_config)
    _language(report, modality, features, set(stats) if isinstance(stats, dict) else set(), run_config)
    _metadata_lines(report, info)
    _statistics(report, stats_state, stats, modality, features)
    if run_config is not None:
        _transform_checks(report, modality, run_config, groups, cameras)
    if config is None:
        report.add("groot.config", "unknown", "No data config given, so config keys, camera pairing, language and the "
                   "fine-tuning transforms were not checked, and every modality.json entry was treated as read.",
                   ("meta/modality.json", "", None),
                   remedy="Pass the data config: config_name for a built-in (e.g. so100), or the .py file and class.")
    return _finish(report, root, config)


def _finish(report: _Report, root: Path, config: dict | None) -> dict:
    counts = {s: sum(f["status"] == s for f in report.findings) for s in ("error", "warning", "unknown", "pass")}
    status = next((s for s in ("error", "warning", "unknown") if counts[s]), "pass")
    shown = None
    if config is not None:
        shown = {k: v for k, v in config.items() if k in ("video", "state", "action", "language")}
        shown["transform"] = config.get("transform")
    name = None if config is None else (config.get("data_config") or config.get("class"))
    usable = config is not None and not config["unsupported"] and not config["missing"]
    return {"schema_version": 2, "kind": "groot_dataset_contract", "status": status, "counts": counts,
            "dataset": str(root), "consumer": GROOT_N15, "groot_version": "n1.5", "modality_config": shown,
            "modality_config_name": name, "files": report.files, "findings": report.findings,
            "paths": {FT: f"{FINETUNE}:200-211: LeRobotSingleDataset with the data config's transforms",
                      DIRECT: f"LeRobotSingleDataset without transforms ({DIRECT_USERS})"},
            "coverage": {"not_checked": ([] if usable else NOT_CHECKED_WITHOUT_CONFIG) + NOT_CHECKED},
            "scope": "Static metadata check of meta/modality.json against meta/info.json, the jsonl metadata, "
                     "meta/stats.json and an optional N1.5 data config. Parquet contents, video frames and codecs, "
                     "the checkpoint's metadata, training and task success are not checked."}


def unknown_version(dataset: str | Path, reason: str) -> dict:
    """A report that runs no version-specific rule because the GR00T version could not be determined."""
    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Dataset directory not found: {root}")
    report = _Report(root)
    report.add("groot.version", "unknown", f"The Isaac-GR00T version could not be determined: {reason}. N1.5 and N1.7 "
               "read modality.json, statistics and cameras differently, so no rule was applied.",
               ("meta/modality.json", "", None), remedy="Pass groot_version='n1.5' or 'n1.7'.")
    counts = {s: sum(f["status"] == s for f in report.findings) for s in ("error", "warning", "unknown", "pass")}
    return {"schema_version": 2, "kind": "groot_dataset_contract", "status": "unknown", "counts": counts,
            "dataset": str(root), "consumer": None, "groot_version": None, "modality_config": None,
            "modality_config_name": None, "files": report.files, "findings": report.findings, "paths": {},
            "coverage": {"not_checked": ["every dataset rule: the Isaac-GR00T version is unknown"]},
            "scope": "No rule applied: the GR00T version is unknown."}


__all__ = ["BUILTIN_SOURCE", "GROOT_N15", "N15_COMMIT", "NOT_CHECKED", "check", "detect_version", "read_data_config",
           "unknown_version"]
