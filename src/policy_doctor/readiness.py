"""Will this checkpoint read this robot the way it read its training data? A static, per-axis answer.

Compares a checkpoint (and, when given, its training dataset) with the robot interface read from the command or
config that will run it (robot_interface). Every rule cites what LeRobot 146d847 does at deployment. Axes are
reported as cleared, not cleared or unknown. A cleared axis means that declared check completed; it certifies nothing
about task success, achieved loop timing, latency, dynamics, calibration or physical units, which are not in these
files.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from .artifacts import LEROBOT_REVISION, load_checkpoint, load_dataset, step_kind
from .diagnostics import (
    LEROBOT_POLICY_TYPES,
    MODES,
    REBUILT_STATISTICS,
    Finding,
    _mode_vectors,
    _vector,
    _visual_shape,
    audit,
    check_stat,
    ev,
    kind,
    normal_mode,
    normalizer,
)

LEROBOT = f"LeRobot {LEROBOT_REVISION[:7]}"
AXES = ("loading", "cameras", "image_size", "state", "action", "joint_order", "declared_rate", "normalization",
        "relative_actions", "units")
# What each policy type does when a camera it was trained on is absent (verified at 146d847).
MISSING_CAMERA = {
    "pi0": ("appends a blank (-1) image for each missing camera after the present ones",
            "src/lerobot/policies/pi0/modeling_pi0.py:1014-1063"),
    "pi05": ("appends a blank (-1) image for each missing camera after the present ones",
             "src/lerobot/policies/pi05/modeling_pi05.py:1376-1436"),
    "smolvla": ("keeps the present cameras in order and pads missing ones only up to empty_cameras, dropping the rest",
                "src/lerobot/policies/smolvla/modeling_smolvla.py:379-418"),
    "act": ("indexes every trained camera, so a missing one raises KeyError at the first inference",
            "src/lerobot/policies/act/modeling_act.py:133-136"),
    "diffusion": ("stacks every trained camera, so a missing one raises KeyError at the first inference",
                  "src/lerobot/policies/diffusion/modeling_diffusion.py:150-152"),
}
# Policies that raise when none of their trained cameras is present.
ALL_MISSING_RAISES = {"pi0": "src/lerobot/policies/pi0/modeling_pi0.py:1017-1021",
                      "pi05": "src/lerobot/policies/pi05/modeling_pi05.py:1379-1383",
                      "smolvla": "src/lerobot/policies/smolvla/modeling_smolvla.py:382-385"}
CRASHES_ON_MISSING_CAMERA = {"act", "diffusion"}
# Policies that resize any input resolution with padding to the model's size.
RESIZES = {"smolvla": "src/lerobot/policies/smolvla/modeling_smolvla.py:389-395",
           "pi0": "src/lerobot/policies/pi0/modeling_pi0.py:1041-1043",
           "pi05": "src/lerobot/policies/pi05/modeling_pi05.py:1376-1436"}
# Policies that zero-pad a state narrower than max_state_dim, when the normalizer has let it through
# (src/lerobot/policies/common/vla_utils.py:116-131). pi05 turns state into tokens instead.
PADS_STATE = {"pi0": "src/lerobot/policies/pi0/modeling_pi0.py:1069",
              "smolvla": "src/lerobot/policies/smolvla/modeling_smolvla.py:451"}
# Steps lerobot-rollout always overrides; a saved preprocessor without them fails to load.
ROLLOUT_OVERRIDES = (("device_processor", "device"), ("rename_observations_processor", "rename"))


def _effective_mode(step: dict | None, key: str) -> str | None:
    feature = ((step or {}).get("config", {}).get("features") or {}).get(key)
    if not isinstance(feature, dict):
        return None
    selected = step["config"].get("normalize_observation_keys")
    if key != "action" and isinstance(selected, list) and key not in selected:
        return "IDENTITY"
    return normal_mode(step, key, feature)


def _trained_width(step: dict | None, key: str) -> int | None:
    """Width of usable vectors selected by the saved mode; not authenticated training history."""
    vectors = _active_vectors(step, key)
    return max(map(len, vectors.values())) if vectors else None


def _active_vectors(step: dict | None, key: str) -> dict | None:
    """Finite active vectors with native one-dimensional scalar broadcasting preserved."""
    mode = _effective_mode(step, key)
    stats = (step or {}).get("stats", {}).get(key, {})
    vectors = _mode_vectors(stats, mode) if mode in MODES and MODES[mode] else None
    if isinstance(vectors, dict):
        return vectors
    if mode not in MODES or not MODES[mode]:
        return None
    vectors = {name: _vector([stats[name]] if type(stats.get(name)) in (int, float) else stats.get(name))
               for name in MODES[mode]}
    if any(value is None for value in vectors.values()):
        return None
    if len({len(value) for value in vectors.values()} - {1}) > 1:
        return None
    if mode == "MEAN_STD":
        if (vectors["std"] < 0).any():
            return None
    elif (vectors[MODES[mode][0]] > vectors[MODES[mode][1]]).any():
        return None
    return vectors


def _runtime_unknown(cp, side: str) -> bool:
    return (_factory_unknown(cp)
            or any(step_kind(step["name"]) == "unsupported" for step in getattr(cp, side)))


def _factory_unknown(cp) -> bool:
    return cp.config.get("type") not in LEROBOT_POLICY_TYPES or cp.config.get("type") in REBUILT_STATISTICS


def _unmodeled_policy_overrides(robot) -> set:
    return set(robot.get("unsupported_policy_overrides", [])) | (set(robot.get("policy_overrides", {})) - {"n_action_steps"})


def _numerical_interfaces(cp, robot) -> list[tuple]:
    names = robot["features"].get("observation.state", {}).get("names")
    result = [("pre", "observation.state", len(names), "declared robot input")] if names is not None else []
    # Robot action_names is a desired output interface, never the input to an unnormalizer. These two native
    # policies derive their model output width from config.action_feature (ACT:379, SmolVLA:219-222).
    shape = cp.features.get("action", {}).get("shape")
    if cp.config.get("type") in ("act", "smolvla") and isinstance(shape, list) and len(shape) == 1 \
            and type(shape[0]) is int and shape[0] > 0:
        result.append(("post", "action", shape[0], "declared model output"))
    return result


def _chw(shape) -> list | None:
    """A 3-axis image shape as (channels, height, width), or None when the layout cannot be told."""
    if not isinstance(shape, list) or len(shape) != 3 or not all(type(v) is int for v in shape):
        return None
    if shape[0] in (1, 3):
        return shape
    if shape[-1] in (1, 3):
        return [shape[2], shape[0], shape[1]]
    return None


class _Report:
    def __init__(self, robot: dict, omitted: dict | None = None):
        self.findings: list[Finding] = []
        self.axes: dict = {}
        self.robot = robot
        # Processor JSON file name -> step pointers whose tensor state an intake deliberately did not fetch.
        self.omitted = {name: set(pointers) for name, pointers in (omitted or {}).items()}

    def withheld(self, path, pointer) -> bool:
        return bool(path) and pointer in self.omitted.get(Path(path).name, set())

    def add(self, axis, code, status, message, *evidence, remedy=""):
        self.findings.append(Finding(f"readiness.{code}", status, message, list(evidence), remedy))
        self.axes.setdefault(axis, []).append(status)

    def note(self, axis, reason):
        """An axis that could not be decided, without inventing a finding."""
        self.axes.setdefault(axis, []).append(("unknown", reason))

    def robot_ev(self, pointer, value):
        return {"source": self.robot.get("source"), "kind": "robot_interface", "pointer": pointer, "value": value}


def _axis_status(entries) -> dict:
    statuses = [e if isinstance(e, str) else e[0] for e in entries]
    reasons = [e[1] for e in entries if not isinstance(e, str)]
    status = ("not_cleared" if any(s in ("error", "warning") for s in statuses) else
              "unknown" if any(s == "unknown" for s in statuses) else "cleared")
    return {"status": status, "reasons": reasons} if reasons else {"status": status}


def _loading(cp, report: _Report) -> None:
    """Whether lerobot-rollout can load the saved processors with the overrides it always passes."""
    if _factory_unknown(cp) or "type" in _unmodeled_policy_overrides(report.robot):
        report.note("loading", "This policy may rebuild its processors or use a plugin factory; whether the saved native "
                               "processor restoration path is used is unknown")
        for side in ("pre", "post"):
            for step in getattr(cp, side):
                if step.get("missing_state"):
                    report.add("loading", "processor_copy_incomplete", "unknown",
                               f"{side}processor references a state file absent from this copy; the selected factory's "
                               "restoration path is unmodeled, so a native saved-loader failure is not established.",
                               ev(cp, step["pointer"], {"state_file": step.get("state_file")}, step["source"]))
        return
    if not cp.pre and not cp.post:
        report.add("loading", "processors_missing", "error",
                   f"The checkpoint saves no policy processors, so lerobot-rollout stops with ProcessorMigrationError "
                   f"before running ({LEROBOT} src/lerobot/processor/pipeline.py:1639).",
                   ev(cp, "/type", cp.config.get("type")),
                   remedy="Migrate the checkpoint with LeRobot's migrate_policy_normalization and deploy the migrated copy.")
        return
    for side in ("pre", "post"):
        for step in getattr(cp, side):
            if not step.get("missing_state"):
                continue
            evidence = ev(cp, step["pointer"], {"name": step["name"], "state_file": step.get("state_file")}, step["source"])
            if report.withheld(step.get("source"), step.get("pointer")):
                report.add("loading", "processor_state_uninspected", "unknown",
                           f"{side}processor state is listed at the selected pin but was deliberately not fetched; "
                           "loading is unassessed.", evidence, remedy="Fetch the referenced processor state before checking loading.")
            elif step_kind(step["name"]) in ("normalize", "unnormalize"):
                report.add("loading", "processor_state_missing", "error",
                           f"{side}processor references a state file absent from this checkpoint copy. The native pipeline "
                           f"raises FileNotFoundError while loading it, even in IDENTITY mode ({LEROBOT} "
                           "src/lerobot/processor/pipeline.py:1319-1331).", evidence,
                           remedy="Restore the referenced state file from the exact checkpoint export.")
            else:
                report.note("loading", f"{side}processor references an absent state file, but this step's native state-loading "
                                       "behavior is not modeled")
    kinds = {step_kind(s["name"]) for s in cp.pre}
    absent = [name for name, step in ROLLOUT_OVERRIDES if step not in kinds]
    if absent:
        report.add("loading", "override_without_step", "error",
                   f"lerobot-rollout always passes overrides for device_processor and rename_observations_processor "
                   f"({LEROBOT} src/lerobot/rollout/context.py:576-579), but this preprocessor has no {absent}, so "
                   "loading raises 'Override keys ... do not match any step' (src/lerobot/processor/pipeline.py:1393-1397).",
                   ev(cp, "/steps", [s["name"] for s in cp.pre], cp.pre[0]["source"] if cp.pre else None),
                   remedy="Re-export the checkpoint's processors with the current LeRobot version.")
    else:
        report.add("loading", "overrides_accepted", "pass",
                   "The saved preprocessor has the device and rename steps lerobot-rollout overrides.",
                   ev(cp, "/steps", [s["name"] for s in cp.pre], cp.pre[0]["source"]))


def _saved_rename(cp, robot, report: _Report) -> None:
    """lerobot-rollout replaces the checkpoint's saved rename_map with --rename_map, even when that is empty."""
    steps = [s for s in cp.pre if step_kind(s["name"]) == "rename"]
    saved = (steps[0]["config"].get("rename_map") or {}) if steps else {}
    if not saved:
        return
    given = robot.get("rename_map") or {}
    provided = list(robot["features"])
    differs = [k for k in provided if saved.get(k, k) != given.get(k, k)]
    reference = ev(cp, steps[0]["pointer"] + "/config/rename_map", saved, steps[0]["source"])
    if not differs:
        report.add("cameras", "rename_as_trained", "pass",
                   "With this command's --rename_map, the robot's observations reach the policy under the keys the "
                   "training rename produced.", reference)
        return
    merged = {**given, **{k: saved[k] for k in provided if k in saved}}
    report.add("cameras", "rename_replaced", "error",
               f"This checkpoint was trained with rename_map {saved}, saved in its preprocessor. lerobot-rollout replaces "
               f"it with the command's --rename_map ({given or 'empty'}) ({LEROBOT} src/lerobot/rollout/context.py:578, "
               f"src/lerobot/processor/pipeline.py:1253), so {differs} reach the policy under other keys than in training.",
               reference, report.robot_ev("/rename_map", given),
               remedy=f"Add --rename_map='{json.dumps(merged)}' to the command.")


def _padded_cameras(cp) -> set:
    count = cp.config.get("empty_cameras")
    count = count if type(count) is int else 0
    return {k for k in cp.config.get("input_features") or {}
            if (m := re.fullmatch(r"observation\.images\.empty_camera_(\d+)", k)) and int(m.group(1)) < count}


def _cameras(cp, robot, report: _Report) -> list[str]:
    padded = _padded_cameras(cp)
    policy = [k for k in (cp.config.get("input_features") or {})
              if kind(cp.features.get(k, {}), k) == "VISUAL" and k not in padded]
    rename = robot.get("rename_map") or {}
    rollout = robot.get("program") == "rollout"
    if "robot.cameras" in robot.get("unresolved", []):
        report.note("cameras", "the robot's cameras are not resolved from the command")
        report.note("image_size", "the robot's cameras are not resolved from the command")
        return []
    provided = [k for k in robot["features"] if k.startswith("observation.images.")]
    targets: dict = {}
    for key in robot["features"]:
        targets.setdefault(rename.get(key, key), []).append(key)
    for target, sources in targets.items():
        if len(sources) > 1:
            report.add("cameras", "rename_collision", "error",
                       f"rename_map sends {sources} to the same key {target}; the later one silently replaces the "
                       f"earlier ({LEROBOT} src/lerobot/processor/rename_processor.py:43-47).",
                       report.robot_ev("/rename_map", rename), remedy="Give every robot observation its own policy key.")
    for source in rename:
        if source not in robot["features"]:
            report.add("cameras", "rename_unused", "warning",
                       f"rename_map renames {source}, which this robot does not provide; the entry is ignored.",
                       report.robot_ev("/rename_map", rename), remedy="Use the robot's observation key as the rename source.")
    if not policy:
        report.add("image_size", "no_images", "pass", "The checkpoint takes no camera input, so there is no image size to match.",
                   ev(cp, "/input_features", cp.config.get("input_features")))
        report.add("cameras", "no_camera_inputs", "pass",
                   "The checkpoint takes no camera input" + (f" besides {sorted(padded)}, which it pads itself." if padded else "."),
                   ev(cp, "/input_features", cp.config.get("input_features")))
        return []
    available = {rename.get(k, k) for k in provided}
    present = [k for k in policy if k in available]
    missing = [k for k in policy if k not in available]
    policy_type = cp.config.get("type")
    if not missing:
        report.add("cameras", "cameras_match", "pass",
                   f"Every camera the checkpoint was trained on is provided: {policy}"
                   + (f"; {sorted(padded)} are padded by the policy, as in training." if padded else "."),
                   ev(cp, "/input_features", cp.config.get("input_features")))
        return present
    evidence = (ev(cp, "/input_features", cp.config.get("input_features")), report.robot_ev("/cameras", robot["cameras"]))
    # lerobot-rollout refuses to start when neither camera set contains the other, unless a rename_map is given.
    if rollout and not rename and not (set(policy) <= set(provided) or set(provided) <= set(policy)):
        report.add("cameras", "rollout_refuses", "error",
                   f"The checkpoint expects {policy} but the robot provides {provided}. {LEROBOT} lerobot-rollout stops "
                   "before moving with 'Visual feature mismatch' (src/lerobot/rollout/context.py:539-559).", *evidence,
                   remedy="Rename the robot cameras to the trained keys with --rename_map, or record with those names.")
        return present
    if not present and policy_type in ALL_MISSING_RAISES:
        report.add("cameras", "cameras_missing", "error",
                   f"None of the trained cameras {policy} is provided, so {policy_type} raises ValueError at the first "
                   f"inference ({LEROBOT} {ALL_MISSING_RAISES[policy_type]}).", *evidence,
                   remedy="Provide the trained cameras under their trained keys (use --rename_map for different names).")
        return present
    behaviour = MISSING_CAMERA.get(policy_type)
    if behaviour is None:
        report.add("cameras", "cameras_missing", "error",
                   f"The robot does not provide {missing}, which this {policy_type} checkpoint was trained on; how "
                   "this policy type handles a missing camera is not modeled.", *evidence,
                   remedy="Provide the trained cameras under their trained keys.")
        return present
    description, citation = behaviour
    shift = (policy_type not in CRASHES_ON_MISSING_CAMERA and
             any(policy.index(m) < max(policy.index(k) for k in present) for m in missing))
    report.add("cameras", "cameras_missing", "error",
               f"The robot does not provide {missing}. {policy_type} {description} ({LEROBOT} {citation})."
               + (" Present cameras then move into earlier input slots than in training." if shift else ""), *evidence,
               remedy="Provide every trained camera under its trained key (use --rename_map for different names).")
    return present


def _training_size(cp, ds, key: str):
    """(channels, height, width) the policy saw for ``key`` in training, and where that comes from.

    The training dataset is authoritative when given: fine-tuning keeps a base checkpoint's input_features in
    config.json (src/lerobot/policies/factory.py:303-305), so its declared image shape can be stale.
    """
    if ds is not None:
        steps = [s for s in cp.pre if step_kind(s["name"]) == "rename"]
        saved = (steps[0]["config"].get("rename_map") or {}) if steps else {}
        source = next((k for k, v in saved.items() if v == key), key)
        feature = ds.features.get(source)
        if isinstance(feature, dict):
            shape = _visual_shape(feature) or _chw(feature.get("shape"))
            if shape and len(shape) == 3:
                return shape, f"training data {source}"
    if key in cp.features and _chw(cp.features[key].get("shape")):
        return _chw(cp.features[key]["shape"]), "config.json (may be stale after fine-tuning)"
    return None, None


def _image_size(cp, ds, robot, present: list[str], report: _Report) -> None:
    reverse = {}
    for key in robot["features"]:
        reverse.setdefault((robot.get("rename_map") or {}).get(key, key), key)
    policy_type = cp.config.get("type")
    for key in present:
        trained, basis = _training_size(cp, ds, key)
        provided = robot["features"].get(reverse.get(key, key), {}).get("shape")
        if trained is None or provided is None:
            report.note("image_size", f"{key}: resolution unknown on the {'checkpoint' if trained is None else 'robot'} side")
            continue
        height, width = provided[0], provided[1]
        if [height, width] == list(trained[1:3]):
            report.add("image_size", "image_size_match", "pass", f"{key}: robot images are {width}x{height}, as in {basis}.",
                       report.robot_ev(f"/features/{reverse.get(key, key)}", provided))
            continue
        same_aspect = abs(width / height - trained[2] / trained[1]) < 0.01
        if policy_type in RESIZES:
            report.add("image_size", "image_resized", "pass" if same_aspect else "warning",
                       f"{key}: the robot sends {width}x{height} where {basis} has {trained[2]}x{trained[1]}. "
                       f"{policy_type} resizes with padding ({LEROBOT} {RESIZES[policy_type]})"
                       + (", with the same aspect ratio." if same_aspect else
                          ", so the aspect ratio changes and padding changes the scene placement relative to that reference."
                          + (" Actual training geometry is unverified." if basis.startswith("config.json") else "")),
                       ev(cp, f"/features/{key}", cp.features.get(key)), report.robot_ev(f"/features/{reverse.get(key, key)}", provided),
                       remedy="" if same_aspect else "Set the camera to the training resolution or aspect ratio.")
        else:
            report.note("image_size", f"{key}: robot {width}x{height} vs {basis} {trained[2]}x{trained[1]}; how "
                                      f"{policy_type} handles another resolution is not modeled")


def _widths(cp, robot, report: _Report) -> None:
    rename = robot.get("rename_map") or {}
    policy_type = cp.config.get("type")
    state_names = robot["features"].get("observation.state", {}).get("names")
    for axis, key, robot_names in (("state", "observation.state", state_names), ("action", "action", robot.get("action_names"))):
        if key not in cp.features:
            continue
        if axis == "state" and rename.get(key, key) != key:
            report.add(axis, "state_renamed_away", "error",
                       f"rename_map renames {key} to {rename[key]}, so the policy never receives {key} "
                       f"({LEROBOT} src/lerobot/processor/rename_processor.py:43-47).",
                       report.robot_ev("/rename_map", rename), remedy=f"Remove {key} from --rename_map.")
            continue
        if robot_names is None:
            report.note(axis, f"the robot's {key} names are unknown for robot type {robot.get('robot_type')}")
            continue
        side = "pre" if axis == "state" else "post"
        step = normalizer(cp, side)
        mode = _effective_mode(step, key)
        trained = _trained_width(step, key)
        declared = (cp.features[key].get("shape") or [None])[-1]
        provided = len(robot_names)
        if _unmodeled_policy_overrides(robot) or _runtime_unknown(cp, side):
            report.note(axis, f"{key}: this policy or a custom processor may rebuild or alter saved statistics; "
                              "the effective numerical width is unknown")
            continue
        if step and mode != "IDENTITY" and trained is None:
            state = ("were not fetched" if report.withheld(step.get("source"), step.get("pointer"))
                     else "are missing, unusable, or use an unmodeled active mode")
            report.note(axis, f"{key}: the active saved statistics {state}, so the numerical width is unknown "
                              f"(config.json declares {declared}, which fine-tuning can leave stale)")
            continue
        width = trained if trained and trained != 1 else declared
        stale = f" (config.json declares {declared}; the active saved statistics hold {trained})" if trained and trained != 1 and declared != trained else ""
        if trained == 1:
            report.note(axis, f"{key}: one-value statistics broadcast and do not establish the model's input width or historical training width")
        if type(width) is not int or width <= 0:
            report.note(axis, f"{key}: no usable declared or active numerical width is available")
            continue
        if axis == "action" and policy_type in ("act", "smolvla") and trained and trained != 1 \
                and type(declared) is int and declared != 1 and trained != declared:
            source = ("src/lerobot/policies/act/modeling_act.py:379,519" if policy_type == "act"
                      else "src/lerobot/policies/smolvla/modeling_smolvla.py:219-222")
            report.add(axis, "action_model_output_width", "error",
                       f"The declared {policy_type} model output has {declared} coordinates before unnormalization, "
                       f"while the active postprocessor vectors have {trained}. The robot's desired {provided} action "
                       "coordinates do not establish the postprocessor's input width; applying this declared model "
                       f"output to those vectors raises an elementwise shape error ({LEROBOT} {source}).",
                       ev(cp, f"/features/{key}", cp.features[key]),
                       ev(cp, step["pointer"], step["config"], step["source"]),
                       remedy="Re-export a consistent model output declaration and paired postprocessor.")
        elif axis == "action" and trained and trained != 1 and declared != trained:
            report.note(axis, f"The active postprocessor vectors have {trained} values and config.json declares {declared}; "
                              "this policy's actual model-output width before the postprocessor is not modeled")
        if provided == width:
            basis = "active saved statistics" if trained and trained != 1 else "declared policy interface"
            report.add(axis, f"{axis}_width_match", "pass", f"{key}: the robot provides {provided} values, matching the {basis}{stale}.",
                       report.robot_ev(f"/features/{key}" if axis == "state" else "/action_names", robot_names))
            if axis == "state" and policy_type == "act" and provided != declared:
                report.add(axis, "state_projection_width", "error",
                           f"The active statistics accept {provided} values, but ACT constructs its state projection "
                           f"for the declared {declared} values ({LEROBOT} src/lerobot/policies/act/modeling_act.py:352-354).",
                           ev(cp, f"/features/{key}", cp.features[key]),
                           remedy="Re-export consistent policy dimensions and statistics for the intended robot.")
            elif axis == "state" and provided != declared:
                maximum = cp.config.get("max_state_dim")
                if policy_type not in PADS_STATE or type(maximum) is not int or provided > maximum:
                    report.note(axis, f"The active statistics accept {provided} values but config.json declares {declared}; "
                                      "this policy's handling of that stale state declaration is not modeled")
            continue
        if trained and trained != 1 and provided != 1 and mode != "IDENTITY":
            message = (f"{key}: the robot provides {provided} values against {width} active saved statistics{stale}. The "
                       f"saved normalizer applies {trained} statistics elementwise before the policy sees them, so this "
                       f"raises a shape error at the first inference ({LEROBOT} src/lerobot/processor/normalize_processor.py:352-424).")
        elif axis == "state" and policy_type in PADS_STATE and provided < (cp.config.get("max_state_dim") or 0):
            message = (f"{key}: the robot provides {provided} values against the declared {width}-value interface{stale}. "
                       f"Nothing normalizes it to a fixed width, and {policy_type} zero-pads the state to max_state_dim "
                       f"without an error ({LEROBOT} {PADS_STATE[policy_type]}, src/lerobot/policies/common/vla_utils.py:116-131), "
                       "so matching the historical state interface is not established by padding.")
        else:
            message = (f"{key}: the robot provides {provided} values against a {width}-value declared or active numerical "
                       f"interface{stale}; the two cannot be matched one to one. "
                       + ("A one-value input can broadcast, so a categorical normalization shape error is not inferred." if provided == 1 else ""))
        report.add(axis, f"{axis}_width_mismatch", "error", message,
                   ev(cp, f"/features/{key}", cp.features[key]),
                   report.robot_ev(f"/features/{key}" if axis == "state" else "/action_names", robot_names),
                   remedy="Run the checkpoint on the robot type and configuration it was trained with.")


def _joint_order(cp, ds, robot, report: _Report) -> None:
    if ds is None:
        report.note("joint_order", "no training dataset given, so the trained joint order is unknown")
        return
    checkpoint_names = cp.config.get("action_feature_names")
    rollout = robot.get("program") == "rollout"
    for key, robot_names in (("action", robot.get("action_names")),
                             ("observation.state", robot["features"].get("observation.state", {}).get("names"))):
        trained = ds.features.get(key, {}).get("names")
        if robot_names is None or trained is None:
            report.note("joint_order", f"{key}: names unknown on the {'robot' if robot_names is None else 'dataset'} side")
            continue
        if list(trained) == list(robot_names):
            report.add("joint_order", "order_match", "pass", f"{key}: the robot reports joints in the trained order.",
                       ev(ds, f"/features/{key}/names", trained))
        elif set(trained) == set(robot_names):
            reorders = rollout and bool(checkpoint_names) and set(checkpoint_names) == set(robot_names)
            report.add("joint_order", "order_permuted", "pass" if reorders else "error",
                       f"{key}: the robot reports the trained joints in a different order. " +
                       ("lerobot-rollout reorders them to the checkpoint's saved action_feature_names "
                        f"({LEROBOT} src/lerobot/rollout/context.py:130-162)." if reorders else
                        "Nothing reorders them here (lerobot-rollout does only for a checkpoint that saves "
                        f"action_feature_names, {LEROBOT} src/lerobot/rollout/context.py:130-162): each joint receives "
                        "another joint's value."),
                       ev(ds, f"/features/{key}/names", trained), report.robot_ev("/action_names", robot_names),
                       remedy="" if reorders else "Make the robot report joints in the trained order.")
        else:
            report.add("joint_order", "names_differ", "warning",
                       f"{key}: the training dataset names {list(trained)[:8]} and the robot names {list(robot_names)[:8]} "
                       "differ, so the order cannot be matched by name.",
                       ev(ds, f"/features/{key}/names", trained), report.robot_ev("/action_names", robot_names),
                       remedy="Confirm the robot is the one the data was recorded on, and map its joints explicitly.")


def _rate(cp, ds, robot, report: _Report) -> None:
    fps = robot["fps"]
    try:
        usable_rate = type(fps.get("value")) in (int, float) and math.isfinite(fps["value"]) and fps["value"] > 0
    except OverflowError:
        usable_rate = False
    if not usable_rate:
        report.note("declared_rate", "the declared loop rate is unresolved or not a positive finite number")
        return
    for name, camera in robot.get("cameras", {}).items():
        if isinstance(camera.get("fps"), (int, float)) and isinstance(fps.get("value"), (int, float)) and camera["fps"] < fps["value"]:
            report.add("declared_rate", "camera_slower_than_loop", "warning",
                       f"camera {name} is set to {camera['fps']} fps but the control loop to {fps['value']}; the policy "
                       "may see repeated frames if those configured rates are achieved.", report.robot_ev(f"/cameras/{name}/fps", camera["fps"]),
                       remedy="Match the camera rate to the loop rate.")
    if not isinstance(fps.get("value"), (int, float)):
        report.note("declared_rate", f"the loop rate is not resolved from the command ({fps.get('source')})")
        return
    if ds is None:
        report.note("declared_rate", "no training dataset given, so the trained rate is unknown (LeRobot does not store it in the checkpoint)")
        return
    trained = ds.config.get("fps")
    if not isinstance(trained, (int, float)) or isinstance(trained, bool):
        report.note("declared_rate", "the training dataset declares no usable fps")
        return
    origin = "set by the command" if fps["declared"] else f"the default, since the command sets none: {fps['source']}"
    steps = robot.get("policy_overrides", {}).get("n_action_steps", cp.config.get("n_action_steps"))
    if trained == fps["value"]:
        report.add("declared_rate", "rate_match", "pass",
                   f"The loop rate, {fps['value']} fps ({origin}), equals the training rate. The rate the loop actually "
                   "achieves on the robot is not in these files.", ev(ds, "/fps", trained))
        return
    chunk = (f" A chunk of {steps} actions then spans {steps / fps['value']:.2f} s instead of {steps / trained:.2f} s."
             if isinstance(steps, int) and trained else "")
    report.add("declared_rate", "rate_mismatch", "error",
               f"The command declares {fps['value']} fps ({origin}) but the training data was recorded at {trained} fps; "
               f"LeRobot does not compare the two ({LEROBOT} src/lerobot/rollout/configs.py:300).{chunk}",
               ev(ds, "/fps", trained), report.robot_ev("/fps", fps),
               remedy=f"Run the loop at --fps={trained}.")


MISSING_STATISTICS = re.compile(r": Missing (mean|std|min|max|q01|q99|q10|q90) statistics\.$")


def _normalization(cp, checkpoint, dataset, robot, report: _Report) -> None:
    found = [f for f in audit(checkpoint, dataset)["findings"] if f["code"].startswith("normalization.")]
    overrides = bool(_unmodeled_policy_overrides(robot))
    for finding in found:
        # A state an intake listed but deliberately did not fetch is uninspected, not missing; a state that is
        # genuinely absent stays an error (the same rule as hub._withheld_unknown).
        withheld = any(report.withheld(e.get("path"), e.get("pointer")) for e in finding.get("evidence", []))
        if withheld and finding["code"] == "normalization.statistics" and finding["status"] == "error" \
                and MISSING_STATISTICS.search(finding["message"]):
            finding.update(code="normalization.statistics_uninspected", status="unknown",
                           message="Referenced tensor state is listed at this commit and was not fetched by this intake.",
                           remedy="Fetch the paired processor state, or use a local checkpoint with its state files.")
        for side in ("pre", "post"):
            step = normalizer(cp, side)
            if not step or not (overrides or _runtime_unknown(cp, side)):
                continue
            same_step = any((e.get("path") == step.get("source") and e.get("pointer") == step.get("pointer"))
                            or (step.get("state_file") and e.get("path") == step["state_file"])
                            for e in finding.get("evidence", []))
            if not same_step or finding["code"] == "normalization.epsilon":
                continue
            eps = step["config"].get("eps", 1e-8)
            usable_eps = type(eps) in (int, float) and 0 < eps < 1e300
            shape_only = usable_eps and any(_active_vectors(step, key) and finding["message"] == f"{side}:{key}: {problem}."
                             and re.fullmatch(r"\w+ must contain \d+ finite values", problem or "")
                             for key, feature in (step["config"].get("features") or {}).items()
                             if (problem := check_stat(step["stats"].get(key, {}), _effective_mode(step, key),
                                                       (feature.get("shape") or [None])[-1])))
            if finding["code"] == "normalization.statistics" and finding["status"] == "error" and not shape_only:
                continue  # Missing, malformed or domain-invalid raw statistics remain independent evidence.
            finding.update(code="normalization.effective_context_unassessed", status="unknown",
                           message="The effective processor route or policy overrides are unmodeled. Conditional on the unchanged "
                                   "saved route and mode, the metadata check reports: " + finding["message"]
                                   + " Effective normalization and decoding consequences for this command remain unassessed.",
                           remedy="Inspect the effective processor configuration and loaded statistics for this command.")
            break
        if finding["code"] == "normalization.statistics" and finding["status"] == "error":
            for side in ("pre", "post"):
                step = normalizer(cp, side)
                if not step or overrides or _runtime_unknown(cp, side):
                    continue
                for key, feature in (step["config"].get("features") or {}).items():
                    same_step = any(e.get("path") == step.get("source") and e.get("pointer") == step.get("pointer")
                                    for e in finding.get("evidence", []))
                    unused = ((side == "post" and kind(feature, key) != "ACTION")
                              or (side == "pre" and kind(feature, key) == "ACTION")
                              or _effective_mode(step, key) == "IDENTITY")
                    if unused and same_step and finding["message"].startswith(f"{side}:{key}: "):
                        finding.update(code="normalization.inactive_statistics", status="unknown",
                                       message=f"{side}:{key}: this selected step does not apply these statistics to the rollout "
                                               "feature, so the saved metadata check does not establish a runtime normalization failure. "
                                               "Original metadata check: " + finding["message"],
                                       remedy="Inspect the saved metadata separately from the selected rollout's effective processor.")
        # The saved feature declaration can be stale while the actual input and active vectors agree. Keep that
        # metadata inconsistency, but do not allege a numerical failure solely from the declared shape.
        for side, key, provided, basis in _numerical_interfaces(cp, robot):
            step = normalizer(cp, side)
            if not step or overrides or _runtime_unknown(cp, side):
                continue
            mode = _effective_mode(step, key)
            width = _trained_width(step, key)
            feature = (step["config"].get("features") or {}).get(key)
            eps = step["config"].get("eps", 1e-8)
            if not width or not isinstance(feature, dict) or type(eps) not in (int, float) or not 0 < eps < 1e300:
                continue
            problem = check_stat(step["stats"].get(key, {}), mode, (feature.get("shape") or [None])[-1])
            if (finding["code"] == "normalization.statistics" and finding["status"] == "error"
                    and problem and re.fullmatch(r"\w+ must contain \d+ finite values", problem)
                    and finding["message"] == f"{side}:{key}: {problem}."
                    and any(e.get("path") == step.get("source") and e.get("pointer") == step.get("pointer")
                            for e in finding.get("evidence", [])) and (width == provided or width == 1)):
                lengths = [len(value) for value in _active_vectors(step, key).values()]
                broadcast = 1 in lengths
                finding.update(code="normalization.broadcast_statistics" if broadcast else "normalization.saved_shape_declaration",
                               status="unknown" if broadcast else "warning",
                               message=f"{side}:{key}: active saved vector lengths {lengths} {'broadcast over' if broadcast else 'match'} "
                                       f"the {basis}'s {provided} values while the processor declares {feature['shape']}. "
                                       + ("The strict declaration-width check does not model this native broadcasting; intended "
                                          "coordinate sharing remains unverified." if broadcast else
                                          "This is an inconsistent saved declaration, not a demonstrated numerical normalization failure."),
                               remedy="Verify and retain the intended broadcasting contract." if broadcast else
                                      "Re-export consistent feature declarations; retain the intended saved statistics.")
        report.findings.append(Finding(**finding))
        report.axes.setdefault("normalization", []).append(finding["status"])
    if not found:
        report.note("normalization", "no normalization statistics to compare")
    for side, key, provided, basis in _numerical_interfaces(cp, robot):
        step = normalizer(cp, side)
        width = _trained_width(step, key)
        if not width or overrides or _runtime_unknown(cp, side):
            continue
        if provided != width and provided != 1 and width != 1:
            report.add("normalization", "normalization_provided_width", "error",
                       f"{side}:{key}: the {basis} has {provided} coordinates while the active "
                       f"saved vectors have {width}. Applying this step to that width raises a native elementwise "
                       f"shape error ({LEROBOT} src/lerobot/processor/normalize_processor.py:352-424); this does not "
                       "establish the complete policy's actual runtime output.",
                       ev(cp, step["pointer"], step["config"], step["source"]),
                       report.robot_ev("/features/"+key, robot["features"].get(key)) if side == "pre"
                       else ev(cp, "/features/action", cp.features["action"]),
                       remedy="Use a consistent declared interface and active processor statistics.")


def _relative(cp, robot, report: _Report) -> None:
    steps = [s for s in cp.pre if step_kind(s["name"]) == "relative" and s["config"].get("enabled")]
    if not steps:
        report.add("relative_actions", "absolute_actions", "pass", "The checkpoint saves no enabled relative-action step.",
                   ev(cp, "/use_relative_actions", cp.config.get("use_relative_actions")))
        return
    config = steps[0]["config"]
    saved, exclude = config.get("action_names"), config.get("exclude_joints") or []
    reference = ev(cp, steps[0]["pointer"] + "/config", config, steps[0]["source"])
    if not saved or not exclude:
        report.add("relative_actions", "all_relative", "warning",
                   f"The relative-action step saves {'no action_names' if not saved else 'no exclude_joints'}, so every "
                   f"action dimension, the gripper included, is made relative to the current state "
                   f"({LEROBOT} src/lerobot/processor/relative_action_processor.py:125-142).", reference,
                   remedy="Confirm the gripper was trained relative too, or export the step with action_names.")
        return
    tokens = [str(t).lower() for t in exclude if t]
    # The mask comes from the saved names by position (relative_action_processor.py:125-142), not from the robot.
    absolute = [i for i, name in enumerate(saved) if any(t == str(name).lower() or t in str(name).lower() for t in tokens)]
    robot_names = robot.get("action_names")
    if robot_names is None:
        report.note("relative_actions", "the robot's action names are unknown")
        return
    wrong = [(i, saved[i], robot_names[i] if i < len(robot_names) else None) for i in absolute
             if i >= len(robot_names) or robot_names[i] != saved[i]]
    if wrong:
        report.add("relative_actions", "relative_mask_misplaced", "error",
                   "The relative-action step keeps action positions absolute by the checkpoint's saved names, but this "
                   "robot reports other joints there: " + ", ".join(f"position {i} is {s} in training and {r} here"
                                                                    for i, s, r in wrong[:4])
                   + f" ({LEROBOT} src/lerobot/processor/relative_action_processor.py:125-142).", reference,
                   report.robot_ev("/action_names", robot_names),
                   remedy="Make the robot report joints in the trained order.")
        return
    report.add("relative_actions", "relative_mask", "pass",
               f"Relative actions keep positions {absolute} ({[saved[i] for i in absolute]}) absolute, and this robot "
               f"reports the same joints there ({LEROBOT} src/lerobot/processor/relative_action_processor.py:125-142).",
               reference)


def _units(robot, report: _Report) -> None:
    flag = robot.get("flags", {}).get("use_degrees")
    if flag is None:
        report.note("units", "joint units and action conventions are not recorded in these files")
        return
    def describe(value):
        return "degrees" if value else "a -100..100 range" if value is False else "unknown units"
    reported = (", ".join(f"{side} arm joints in {describe(v)}" for side, v in flag.items()) if isinstance(flag, dict)
                else f"arm joints in {describe(flag)}")
    report.note("units", f"the training data's units are not recorded in these files; this robot reports {reported}, "
                         "and its gripper in a 0..100 range (src/lerobot/robots/so_follower/so_follower.py:68-77)")


def _route(reason: str) -> str:
    if "image size" in reason or "width/height" in reason:
        return "image_size"
    if reason.startswith(("camera", "robot.cameras", "--robot.cameras", "rename_map", "--rename_map")):
        return "cameras"
    if "fps" in reason:
        return "declared_rate"
    return "state"


def readiness(checkpoint: str, robot: dict, dataset: str | None = None, *,
              omitted_states: dict[str, list[str]] | None = None) -> dict:
    """Per-axis readiness of ``checkpoint`` on the robot described by ``robot`` (from robot_interface).

    ``omitted_states`` maps a processor JSON file name to the step pointers (``/steps/<i>``) whose tensor state an
    intake listed but deliberately did not fetch; their statistics are reported as uninspected, never as missing.
    """
    cp = load_checkpoint(checkpoint)
    ds = load_dataset(dataset) if dataset else None
    report = _Report(robot, omitted_states)
    for reason in robot.get("unknown", []):
        report.note(_route(reason), reason)
    unsupported = _unmodeled_policy_overrides(robot)
    if unsupported:
        for axis in AXES:
            report.note(axis, f"policy overrides {sorted(unsupported)} are not modeled; effective behavior on this axis is unknown")
    if "policy.n_action_steps" in robot.get("unresolved", []):
        for axis in ("action", "declared_rate"):
            report.note(axis, "policy.n_action_steps is unresolved; the saved default cannot establish this axis")
    if robot.get("program") == "rollout":
        _loading(cp, report)
        _saved_rename(cp, robot, report)
    else:
        report.note("loading", f"{LEROBOT} lerobot-record runs no policy (src/lerobot/scripts/lerobot_record.py:16-18); "
                               "use the lerobot-rollout command that will run this checkpoint")
    present = _cameras(cp, robot, report)
    _image_size(cp, ds, robot, present, report)
    _widths(cp, robot, report)
    _joint_order(cp, ds, robot, report)
    _rate(cp, ds, robot, report)
    _normalization(cp, checkpoint, dataset, robot, report)
    _relative(cp, robot, report)
    _units(robot, report)
    findings = [asdict(f) for f in report.findings]
    counts = {s: sum(f["status"] == s for f in findings) for s in ("error", "warning", "unknown", "pass")}
    axes = {axis: _axis_status(report.axes.get(axis, [("unknown", "not assessed")])) for axis in AXES}
    by_status = {state: [a for a, v in axes.items() if v["status"] == state] for state in ("not_cleared", "unknown", "cleared")}
    # Units are never recorded in these files, so they stay a stated caveat rather than holding every report at unknown.
    decidable_unknown = [a for a in by_status["unknown"] if a != "units"]
    status = ("error" if counts["error"] else "warning" if counts["warning"] else
              "unknown" if decidable_unknown or counts["unknown"] else "pass")
    summary = (f"{len(by_status['not_cleared'])} axis(es) not cleared ({', '.join(by_status['not_cleared']) or 'none'}), "
               f"{len(by_status['unknown'])} unknown ({', '.join(by_status['unknown']) or 'none'}), "
               f"{len(by_status['cleared'])} cleared.")
    return {"schema_version": 1, "kind": "robot_readiness", "status": status, "summary": summary, "axes": axes,
            "counts": counts, "findings": findings,
            "robot": {k: robot.get(k) for k in ("program", "robot_type", "robot_id", "fps", "source")},
            "checkpoint": cp.summary(), "dataset": ds.summary() if ds else None,
            "scope": ("Static comparison of the checkpoint, its training data and the robot command, as declared. A "
                      "cleared axis means that declared check completed; it does not certify task success, achieved "
                      "loop timing, latency, dynamics, calibration or physical units.")}
