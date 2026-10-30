"""Deterministic compatibility findings with explicit evidence and unknowns."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .artifacts import (
    LEROBOT_REVISION,
    InputError,
    Snapshot,
    contained,
    load_checkpoint,
    load_dataset,
    load_deployment,
    native_registry_name,
    neutral_source,
    read_json,
    step_kind,
)
from .mapping import feature_mapping, validate_rename_map

MODES = {"IDENTITY": (), "MEAN_STD": ("mean", "std"), "MIN_MAX": ("min", "max"),
         "QUANTILES": ("q01", "q99"), "QUANTILE10": ("q10", "q90")}
LEROBOT = f"LeRobot {LEROBOT_REVISION[:7]}"
# Policy types present in LeRobot 146d847 (src/lerobot/policies). Types outside this set are not modeled.
LEROBOT_POLICY_TYPES = {"act", "diffusion", "dm05", "eo1", "evo1", "fastwam", "fineart_vla", "flux3", "gaussian_actor",
                        "groot", "lawam", "lingbot_va", "molmoact2", "multi_task_dit", "pi0", "pi05", "pi0_fast",
                        "smolvla", "tdmpc", "vla_jepa", "vqbet", "wall_x", "xvla"}
# Native designs that unnormalize actions with a mode the policy config leaves as IDENTITY: (mode, citation).
ASYMMETRIC_ACTION_NORMALIZATION = {"lingbot_va": ("QUANTILES", "src/lerobot/policies/lingbot_va/processor_lingbot_va.py:17-19, 59-63")}
# Policies whose local inference accepts a state narrower than max_state_dim: (how, citation, chunk-runner citation).
# LeRobot's chunk runner still refuses any tensor input whose shape differs from config.json.
STATE_PADDING = {"smolvla": ("zero-pads state to max_state_dim", "src/lerobot/policies/smolvla/modeling_smolvla.py:448-451",
                             "src/lerobot/policies/smolvla/modeling_smolvla.py:370-371"),
                 "pi0": ("zero-pads state to max_state_dim", "src/lerobot/policies/pi0/modeling_pi0.py:1067-1070",
                         "src/lerobot/policies/pretrained.py:304-305"),
                 "pi05": ("turns state into prompt tokens of any width, and zero-pads it to max_state_dim only with "
                          "proprioceptive memory", "src/lerobot/policies/pi05/processor_pi05.py:74-78, "
                          "src/lerobot/policies/pi05/modeling_pi05.py:1443-1445", "src/lerobot/policies/pretrained.py:304-305")}
# Fields of LeRobot's generic relative step (src/lerobot/processor/relative_action_processor.py:119-120);
# action_names (line 121) is optional and absent from some public exports.
RELATIVE_STEP_FIELDS = ("enabled", "exclude_joints")
# State tokenizers that bin state assuming the normalizer already ran (they must follow it).
STATE_TOKENIZERS = {"pi05_prepare_state_tokenizer_processor_step": "src/lerobot/policies/pi05/processor_pi05.py:75",
                    "pi0_fast_prepare_state_tokenizer_processor_step": "src/lerobot/policies/pi0_fast/processor_pi0_fast.py:70"}
# Policies whose prepare_images requires one present camera and pads or drops the others.
CAMERA_TOLERANT = {"smolvla": "src/lerobot/policies/smolvla/modeling_smolvla.py:379-418",
                   "pi0": "src/lerobot/policies/pi0/modeling_pi0.py:1014-1059",
                   "pi05": "src/lerobot/policies/pi05/modeling_pi05.py:1376-1434"}
# Policy types whose make_<type>_pre_post_processors_from_pretrained rebuilds or pads saved statistics
# (src/lerobot/policies/factory.py:182-198 dispatches to them); every other type deploys the saved statistics as-is.
REBUILT_STATISTICS = {"dm05", "evo1", "flux3", "groot", "lawam", "molmoact2"}
# src/lerobot/processor/normalize_processor.py lines of each mode's forward and inverse transform.
FORWARD_LINES = {"MEAN_STD": "361-365", "MIN_MAX": "376-387", "QUANTILES": "397-404", "QUANTILE10": "414-421"}
INVERSE_LINES = {"MEAN_STD": "363-364", "MIN_MAX": "376-385", "QUANTILES": "397-403", "QUANTILE10": "414-420"}
RAISE_LINES = {"MEAN_STD": "352-358", "MIN_MAX": "367-373", "QUANTILES": "389-395", "QUANTILE10": "406-412"}
DEGENERATE_SPREAD = 1e-6
# How far a dataset's distribution may sit from the saved statistics, in their normalized units, before
# RoboClue warns. Narrower data inside the saved range is fine.
MAX_OFFSET, MAX_SCALE = 1.0, 2.0


@dataclass
class Finding:
    code: str
    status: str
    message: str
    evidence: list[dict]
    remedy: str = ""
    verification: str = ""


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    # Stale-statistics reconciliations, decided only after every comparison (a deployment can veto them).
    deferred: list[tuple] = field(default_factory=list)
    # Training-dataset features in the same audit; a deployment is excused only when they agree with the statistics.
    dataset_features: dict = field(default_factory=dict)

    def add(self, code, status, message, *evidence, remedy="", verification=""):
        self.findings.append(Finding(code, status, message, list(evidence), remedy, verification))

    def to_dict(self) -> dict:
        counts = {s: sum(f.status == s for f in self.findings) for s in ("error", "warning", "unknown", "pass")}
        status = next((s for s in ("error", "unknown", "warning") if counts[s]), "pass")
        return {"schema_version": 1, "status": status, "counts": counts,
                "scope": "Artifact compatibility; model inference and physical task success were not tested.",
                "limitations": ["Model weights and inference are not inspected.",
                                "Metadata does not establish calibration, controller correctness or task competence.",
                                "Dataset checks cover metadata, not video integrity or training membership.",
                                "Training semantics in deployment contracts are user declarations, not independent measurements."],
                "findings": [asdict(f) for f in self.findings], "artifacts": self.artifacts}


def ev(snapshot: Snapshot, pointer: str, value: Any, path: str | None = None) -> dict:
    if snapshot.kind == "checkpoint" and pointer.startswith("/features/"):
        key, _, suffix = pointer[len("/features/"):].partition("/")
        category = "output_features" if key in (snapshot.config.get("output_features") or {}) else "input_features"
        pointer = f"/{category}/{key.replace('~', '~0').replace('/', '~1')}" + ("/" + suffix if suffix else "")
    artifact = snapshot.artifacts[0]
    if path:
        artifact = next((a for a in snapshot.artifacts if a.path == path), artifact)
    reference = {"path": artifact.path, "sha256": artifact.sha256, "pointer": pointer, "value": value}
    try:
        source = snapshot.json_documents[artifact.path]
        for token in pointer.lstrip("/").split("/") if pointer else []:
            key = token.replace("~1", "/").replace("~0", "~")
            source = source[int(key)] if isinstance(source, list) else source[key]
        reference["value_kind"] = "source" if source == value else "derived"
        if reference["value_kind"] == "derived":
            reference["source_value"] = source
    except (KeyError, IndexError, ValueError, TypeError):
        reference["value_kind"] = "missing"
    return reference


def kind(feature: dict, key: str) -> str:
    if feature.get("type"):
        return feature["type"]
    if feature.get("dtype") in ("video", "image") or key.startswith("observation.image"):
        return "VISUAL"
    return "ACTION" if key == "action" else "STATE"


def normalizer(snapshot: Snapshot, side: str) -> dict | None:
    expected = "normalize" if side == "pre" else "unnormalize"
    steps = [s for s in getattr(snapshot, side) if step_kind(s["name"]) == expected]
    return steps[0] if len(steps) == 1 else None


def normal_mode(step: dict, feature_key: str, feature: dict) -> str:
    mapping = step["config"].get("norm_map", {})
    if not isinstance(mapping, dict):
        raise InputError("Processor norm_map must be an object")
    return mapping.get(kind(feature, feature_key), "IDENTITY")


def check_stat(stats: dict, mode: str, dimension: int) -> str | None:
    if mode not in MODES:
        return f"Unsupported normalization mode {mode}"
    for name in MODES[mode]:
        if name not in stats:
            return f"Missing {name} statistics"
        try:
            array = np.asarray(stats[name], dtype=float)
        except (TypeError, ValueError):
            return f"Invalid numeric {name} statistics"
        if array.shape != (dimension,) or not np.isfinite(array).all():
            return f"{name} must contain {dimension} finite values"
    if mode == "MEAN_STD" and np.any(np.asarray(stats["std"]) < 0):
        return "Negative standard deviation"
    if mode in ("MIN_MAX", "QUANTILES", "QUANTILE10"):
        lo, hi = MODES[mode]
        if np.any(np.asarray(stats[lo]) > np.asarray(stats[hi])):
            return f"{lo} exceeds {hi}"
    return None


def _asymmetry_verdict(cp: Snapshot, mode: str) -> tuple[str, str]:
    """Verdict for a one-sided, stats-backed action unnormalization: native design, unmodeled type, or fault."""
    policy = cp.config.get("type")
    if policy in ASYMMETRIC_ACTION_NORMALIZATION:
        native, source = ASYMMETRIC_ACTION_NORMALIZATION[policy]
        if mode == native:
            return "pass", (f" This is the documented {policy} design ({LEROBOT} {source}): outputs are unnormalized "
                            f"with {native} from the saved processor's statistics.")
        return "error", (f" The native {policy} design unnormalizes actions with {native} only ({LEROBOT} {source}); "
                         f"{mode} is not that design.")
    if policy not in LEROBOT_POLICY_TYPES:
        return "unknown", (f" Policy type {policy!r} is not in {LEROBOT}; its own training-target normalization "
                           "would decide whether this asymmetry is intended.")
    return "error", ""


def _state_tokenizer_order(cp: Snapshot, step: dict) -> str | None:
    """Why a native state tokenizer would bin unnormalized state, or None when its input is normalized (or unused).

    pi05 bins state only when include_state_in_prompt is true (src/lerobot/policies/pi05/processor_pi05.py:54, 74);
    the pi0_fast step has no such field and always bins state (src/lerobot/policies/pi0_fast/processor_pi0_fast.py:43-73).
    """
    registry = native_registry_name(step["name"])
    if registry not in STATE_TOKENIZERS or (registry == "pi05_prepare_state_tokenizer_processor_step"
                                            and step["config"].get("include_state_in_prompt") is False):
        return None
    before = [s for s in cp.pre[:cp.pre.index(step)] if step_kind(s["name"]) == "normalize"]
    if not before:
        return "no normalizer precedes it here"
    features = before[-1]["config"].get("features") or {}
    if features == {}:
        # lerobot/pi05_base and pi05_libero_base ship this way; fine-tuning overrides the normalizer's features
        # from the policy config, so this describes the export as saved.
        return ("the preceding normalizer declares no features, so as saved it leaves state unchanged (base "
                f"checkpoints ship this way; {LEROBOT} fine-tuning overrides the normalizer's features from the policy "
                "config, src/lerobot/scripts/lerobot_train.py:682-688)")
    state = features.get("observation.state") if isinstance(features, dict) else None
    if not isinstance(state, dict):
        return "the preceding normalizer does not declare observation.state, so it leaves state unchanged"
    if normal_mode(before[-1], "observation.state", state) == "IDENTITY":
        return "the preceding normalizer leaves observation.state at IDENTITY"
    return None


def _relative_stand_ins(cp: Snapshot, side: str) -> list[str]:
    """Unmodeled steps on one side whose own configuration declares enabled relative-action handling."""
    names = []
    for s in getattr(cp, side):
        config = s["config"]
        if step_kind(s["name"]) != "unsupported":
            continue
        flagged = config.get("use_relative_action") is True or config.get("use_relative_actions") is True
        shaped = side == "pre" and config.get("enabled") is True and all(k in config for k in RELATIVE_STEP_FIELDS)
        if flagged or shaped:
            names.append(s["name"])
    return names


def _processor_checks(cp: Snapshot, report: Report) -> None:
    if not cp.pre or not cp.post:
        report.add("processor.missing", "unknown", "Saved pre/postprocessors are absent; legacy embedded normalization is not inspected.",
                   ev(cp, "/type", cp.config["type"]), remedy="Save the processors used by this exact deployment.")
    for side in ("pre", "post"):
        categories = [step_kind(s["name"]) for s in getattr(cp, side)]
        for category in ("normalize", "unnormalize", "relative", "absolute"):
            if categories.count(category) > 1:
                report.add("processor.ambiguous", "error", f"{side}: multiple {category} processors; the simple decoding contract is ambiguous.",
                           ev(cp, "/steps", categories, getattr(cp, side)[0]["source"]))
        for step in getattr(cp, side):
            if step_kind(step["name"]) == "unsupported":
                report.add("processor.unsupported_" + side, "unknown", f"Custom {side}processor {step['name']} is not validated.",
                           ev(cp, step["pointer"], step["name"], step["source"]))
            elif side == "pre" and (problem := _state_tokenizer_order(cp, step)):
                registry = native_registry_name(step["name"])
                report.add("processor.native_order", "unknown",
                           f"preprocessor {step['name']} bins state into [-1, 1] assuming the normalizer already ran "
                           f"({LEROBOT} {STATE_TOKENIZERS[registry]}), but {problem}.",
                           ev(cp, step["pointer"], step["name"], step["source"]),
                           remedy="Place the state tokenizer after a normalizer that normalizes observation.state, as the native factory does.")
            elif step_kind(step["name"]) == "neutral":
                report.add("processor.native_neutral", "pass",
                           f"{side}processor {step['name']} leaves declared state, action and camera features unchanged "
                           f"({LEROBOT} {neutral_source(step['name'])}, transform_features).",
                           ev(cp, step["pointer"], step["name"], step["source"]))
        if side == "post":
            if cp.post and "unnormalize" not in categories:
                report.add("processor.unnormalizer_missing", "unknown", "No saved unnormalizer is available to establish action decoding.",
                           ev(cp, "/steps", categories, cp.post[0]["source"]))
        for step in getattr(cp, side):
            config = step["config"]
            if step_kind(step["name"]) == "rename":
                validate_rename_map(config.get("rename_map", {}))
            reference = ev(cp, step["pointer"], {"name": step["name"], "config": config}, step["source"])
            if step.get("state_file") and not step.get("missing_state"):
                artifact = next(a for a in cp.artifacts if a.path == step["state_file"])
                reference["state_artifact"] = {"path": artifact.path, "sha256": artifact.sha256,
                                               "format": "safetensors", "feature_keys": list(step["stats"])}
            if step.get("missing_state"):
                report.add("processor.state_missing", "error", f"{side}processor references a missing state file.", reference,
                           remedy="Restore the referenced processor state from the checkpoint export.")
            if step_kind(step["name"]) in ("normalize", "unnormalize"):
                eps = config.get("eps", 1e-8)
                if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
                    report.add("normalization.epsilon", "error", "Normalization epsilon must be positive.", reference)
                features = config.get("features", {})
                if not isinstance(features, dict):
                    raise InputError("Processor features must be an object")
                for key, ft in features.items():
                    if not isinstance(ft, dict) or not isinstance(ft.get("shape"), list):
                        raise InputError("Malformed processor feature")
                    mode = normal_mode(step, key, ft)
                    cfg_mode = (cp.config.get("normalization_mapping") or {}).get(kind(ft, key))
                    if cfg_mode is not None and mode != cfg_mode:
                        # A saved processor that adds stats-backed normalization the config leaves as IDENTITY is a
                        # documented native design (lingbot_va keeps its quantiles only in the saved processor).
                        one_sided = (cfg_mode == "IDENTITY" and mode != "IDENTITY" and bool(step.get("state_file"))
                                     and step_kind(step["name"]) == "unnormalize")
                        status, note = _asymmetry_verdict(cp, mode) if one_sided else ("error", "")
                        report.add("normalization.config_mismatch", status,
                                   f"{key}: saved processor uses {mode} where the policy config declares {cfg_mode}." + note,
                                   reference, ev(cp, "/normalization_mapping", cp.config["normalization_mapping"]),
                                   remedy="Use the checkpoint's paired processor/config export; verify any intentional override."
                                   if status == "error" else "")
                    if kind(ft, key) != "VISUAL":
                        if len(ft["shape"]) != 1 or not ft["shape"]:
                            report.add("processor.vector_shape", "unknown", f"Cannot validate non-vector stats for {key}.", reference)
                            continue
                        problem = check_stat(step["stats"].get(key, {}), mode, ft["shape"][-1])
                        external = bool(problem) and problem.startswith("Missing ") and not step.get("state_file")
                        # Base checkpoints (smolvla_base, for one) ship a state file without statistics for their features.
                        unfilled = bool(problem) and problem.startswith("Missing ") and step.get("state_file") and not step["stats"].get(key)
                        report.add("normalization.statistics", "error" if problem else "pass",
                                   f"{side}:{key}: {problem or 'required statistics are valid' }." +
                                   (f" No statistics file is referenced: {LEROBOT} passes values through unchanged when statistics "
                                    "are absent (src/lerobot/processor/normalize_processor.py:333), so this checkpoint cannot be "
                                    "deployed as saved. A base checkpoint receives statistics when it is fine-tuned." if external else ""),
                                   reference,
                                   # The message keeps its form: hub intake recognizes withheld state by it (hub.py).
                                   remedy=(f"The state file holds no statistics for {key}, as base checkpoints ship: {LEROBOT} passes "
                                           "the values through unchanged (src/lerobot/processor/normalize_processor.py:333), so it "
                                           "cannot be deployed as saved, but fine-tuning writes the dataset's statistics "
                                           "(src/lerobot/scripts/lerobot_train.py:677-679, 697-703). Expected if you are about to "
                                           "fine-tune it." if unfilled else
                                           "Export the effective processors with the target robot's correct statistics; a base "
                                           "checkpoint may require dataset-specific overrides." if problem else ""))
                    if key not in cp.features:
                        noop = (("its mode is IDENTITY", "src/lerobot/processor/normalize_processor.py:333") if mode == "IDENTITY" else
                                ("postprocessor transitions carry only the action",
                                 "src/lerobot/processor/factory.py:226-232, src/lerobot/processor/converters.py:343-349")
                                if side == "post" and kind(ft, key) != "ACTION" else None)
                        if noop:
                            report.add("processor.feature_unbound", "pass",
                                       f"Processor feature {key} is absent from policy config, but {noop[0]}, so this step leaves it "
                                       f"unchanged ({LEROBOT} {noop[1]}).",
                                       reference, ev(cp, "/input_features", cp.config.get("input_features")))
                        else:
                            report.add("processor.feature_unbound", "warning", f"Processor feature {key} is absent from policy config; inspect factory overrides and rename mapping.",
                                       reference, ev(cp, "/input_features", cp.config.get("input_features")),
                                       verification="Capture the effective processor pipeline after loading, including runtime overrides.")
                    elif ft["shape"] != cp.features[key]["shape"] and mode == "IDENTITY":
                        report.add("processor.feature_shape", "pass",
                                   f"Processor shape for {key} differs from policy config, but its mode is IDENTITY, so this "
                                   f"step leaves it unchanged ({LEROBOT} src/lerobot/processor/normalize_processor.py:333).",
                                   reference, ev(cp, "/input_features", cp.features[key]))
                    elif ft["shape"] != cp.features[key]["shape"]:
                        unmodeled = any(step_kind(s["name"]) == "unsupported" for s in getattr(cp, side))
                        report.add("processor.feature_shape", "unknown" if unmodeled else "error",
                                   f"Processor shape for {key} differs from policy config." +
                                   (" An unmodeled adapter may change intermediate dimensions; its effect is unverified." if unmodeled else ""), reference,
                                   ev(cp, "/input_features", cp.features[key]))
    pre, post = normalizer(cp, "pre"), normalizer(cp, "post")
    if pre and post and "action" in cp.features:
        for ft in ("action",):
            pm, qm = normal_mode(pre, ft, cp.features[ft]), normal_mode(post, ft, cp.features[ft])
            if pm != qm:
                one_sided = pm == "IDENTITY" and qm != "IDENTITY" and bool(post.get("state_file"))
                status, note = _asymmetry_verdict(cp, qm) if one_sided else ("error", "")
                report.add("normalization.pair_mode", status,
                           f"Action preprocessing uses {pm} and postprocessing uses {qm}." + note,
                           ev(cp, pre["pointer"], pm, pre["source"]), ev(cp, post["pointer"], qm, post["source"]),
                           remedy="Restore paired action normalization." if status == "error" else "")
            if pm == qm and pm in MODES:
                for key in MODES[pm]:
                    a, b = pre["stats"].get(ft, {}).get(key), post["stats"].get(ft, {}).get(key)
                    if a is not None and b is not None and not np.array_equal(a, b):
                        report.add("normalization.pair_statistics", "error", f"Action {key} differs between paired processors.",
                                   ev(cp, pre["pointer"], a, pre["source"]), ev(cp, post["pointer"], b, post["source"]),
                                   remedy="Restore paired normalization states; do not recompute only one side.")
    rel = [s for s in cp.pre if step_kind(s["name"]) == "relative" and s["config"].get("enabled")]
    absolute = [s for s in cp.post if step_kind(s["name"]) == "absolute" and s["config"].get("enabled")]
    # An unmodeled preprocessor step stands in for a missing relative step only if its own configuration declares
    # enabled relative-action handling. Nothing pinned shows an arbitrary flagged step reversing relative actions,
    # so a missing absolute step is never excused (only GR00T's native decode is verified, below).
    unmodeled = _relative_stand_ins(cp, "pre") if absolute and not rel else []
    if bool(rel) != bool(absolute) and unmodeled:
        report.add("action.relative_pair", "unknown",
                   f"The generic relative/absolute pair is incomplete, but unmodeled step(s) {unmodeled[:4]} declare enabled "
                   "relative-action handling and may implement it natively.",
                   ev(cp, "/use_relative_actions", cp.config.get("use_relative_actions")),
                   remedy="Confirm from the step's source how relative actions are applied and reversed.")
    elif bool(rel) != bool(absolute):
        report.add("action.relative_pair", "error", "Relative preprocessing and absolute postprocessing are not paired.",
                   ev(cp, "/use_relative_actions", cp.config.get("use_relative_actions")),
                   remedy="Export and reconnect the paired relative/absolute processors.")
    native_relative = [s for s in cp.pre + cp.post if s["config"].get("use_relative_action") is True
                       or s["config"].get("use_relative_actions") is True]
    if cp.config.get("use_relative_actions") is True and not rel and native_relative:
        verified = [s for s in native_relative if s["name"] == "groot_n1_7_action_decode_v1"]
        step = (verified or native_relative)[0]
        report.add("action.config_relative_native", "pass" if verified else "unknown",
                   f"use_relative_actions is realised inside native step {step['name']} rather than the generic relative/absolute pair." +
                   (f" {LEROBOT} adds the generic step only when native GR00T relative actions are absent "
                    "(src/lerobot/policies/groot/processor_groot.py:1232-1250)." if verified else
                    " That step is not modeled; its relative-action semantics are unverified."),
                   ev(cp, "/use_relative_actions", True), ev(cp, step["pointer"], step["config"].get("use_relative_action",
                      step["config"].get("use_relative_actions")), step["source"]))
    elif "use_relative_actions" in cp.config and cp.config["use_relative_actions"] is not bool(rel):
        unmodeled = _relative_stand_ins(cp, "pre") if cp.config["use_relative_actions"] is True else []
        report.add("action.config_relative_mismatch", "unknown" if unmodeled else "error",
                   "Policy relative-action flag disagrees with saved preprocessing." +
                   (f" Unmodeled step(s) {unmodeled[:4]} may implement relative actions natively." if unmodeled else ""),
                   ev(cp, "/use_relative_actions", cp.config["use_relative_actions"]))
    if rel:
        positions = [step_kind(s["name"]) for s in cp.pre]
        if "normalize" in positions and positions.index("relative") > positions.index("normalize"):
            report.add("action.processor_order", "error", "Relative conversion follows normalization; the anchor and actions may use different units.",
                       ev(cp, "/steps", positions, rel[0]["source"]))
        positions = [step_kind(s["name"]) for s in cp.post]
        if "unnormalize" in positions and "absolute" in positions and positions.index("absolute") < positions.index("unnormalize"):
            report.add("action.processor_order", "error", "Absolute conversion precedes unnormalization.",
                       ev(cp, "/steps", positions, absolute[0]["source"]))


def _visual_shape(feature: dict) -> list | None:
    dims = feature["shape"]
    axes = feature.get("names")
    if feature.get("layout") == "HWC" or axes in (["height", "width", "channel"], ["height", "width", "channels"]):
        return [dims[2], dims[0], dims[1]] if len(dims) == 3 else None
    if feature.get("layout") == "CHW" or feature.get("type") == "VISUAL":
        return dims
    return None


def _stale_state_declaration(cp: Snapshot, other: Snapshot, report: Report, key: str, original: str,
                             expected: dict, actual: dict) -> bool:
    """Report a declared state width that may be stale; False leaves the mismatch to the generic error.

    LeRobot keeps a pretrained config's input_features when fine-tuning (factory.py:303-305), so config.json can
    declare the base robot's state width. That is only survivable for policies that zero-pad state to max_state_dim;
    policies that size layers from the declared width (ACT, diffusion, VQ-BeT, TD-MPC) get the generic error.
    """
    a, b, policy, limit = expected["shape"], actual["shape"], cp.config.get("type"), cp.config.get("max_state_dim")
    if key != "observation.state" or policy not in STATE_PADDING or type(limit) is not int \
            or len(b) != 1 or not isinstance(b[0], int) or b[0] > limit:
        return False
    how, local, runner = STATE_PADDING[policy]
    pre = normalizer(cp, "pre")
    stats = (pre or {}).get("stats", {}).get(key) or {}
    processor_feature = ((pre or {}).get("config", {}).get("features") or {}).get(key)
    mode = normal_mode(pre, key, processor_feature if isinstance(processor_feature, dict) else expected) if pre else "IDENTITY"
    # The width that decides is the width of the statistics this mode actually uses (mean for IDENTITY, which uses none).
    names = MODES.get(mode) or ("mean",)
    if any(n not in stats for n in names):
        trained = None
    elif mode in MODES and MODES[mode]:
        trained = b if check_stat(stats, mode, b[0]) is None else list(np.asarray(stats[names[0]]).shape)
    else:
        trained = list(np.asarray(stats["mean"]).shape)
    factory = f"{LEROBOT} src/lerobot/policies/factory.py:303-305"
    consequence = (f"Local inference {how} ({LEROBOT} {local}; max_state_dim={limit}), but LeRobot's chunk runner "
                   f"refuses a state whose shape differs from config.json ({LEROBOT} {runner}, called by remote "
                   "inference at src/lerobot/inference/policy_runner.py:186).")
    remedy = "Re-declare input_features from the training dataset before serving or relying on config.json."
    keys = ((pre or {}).get("config") or {}).get("normalize_observation_keys")
    normalized = (isinstance(processor_feature, dict) and mode != "IDENTITY"
                  and (keys is None or (isinstance(keys, list) and key in keys)))
    runtime = ("runtime normalization uses the saved statistics" if normalized else
               f"the saved normalizer leaves this state unnormalized (its features, mode or normalize_observation_keys "
               f"omit it; {LEROBOT} src/lerobot/processor/normalize_processor.py:281-284, 333)")
    # A deployment is excused only by statistics the saved normalizer actually applies: an IDENTITY mode or an
    # excluded state leaves the saved vectors unused, and unused leftovers do not establish the working width.
    if other.kind != "dataset" and not normalized:
        return False
    # A training dataset in the same audit that disagrees with the deployment's width contradicts the stale-declaration
    # explanation: its statistics then did not come from this data, so the deployment keeps the error.
    trained_with = (report.dataset_features.get(key) or {}).get("shape")
    if other.kind != "dataset" and trained_with is not None and trained_with != b:
        return False
    if trained == b:
        report.add(f"{other.kind}.feature_shape", "warning",
                   f"{key}: the config declares {a}, but the checkpoint's own saved statistics and the {other.kind} both "
                   f"have {b}. LeRobot keeps a pretrained config's input_features when fine-tuning ({factory}), so the "
                   f"declaration is stale; {runtime}. {consequence}",
                   ev(cp, "/features/" + key, expected), ev(other, "/features/" + original, actual),
                   ev(cp, f"/{key}.{names[0]}", {"shape": trained, "mode": mode}, pre["state_file"]), remedy=remedy)
        report.deferred.append((key, b[0], other.kind))
        return True
    # Undecided statistics never excuse a deployment: an IDENTITY or unfetched normalizer lets a wrong width run
    # silently, so the generic error stays (validator 1a; 0.38 behaviour).
    if trained is None and other.kind == "dataset":
        report.add(f"{other.kind}.feature_shape", "warning",
                   f"{key}: the config declares {a} but the {other.kind} has {b}. This may be a stale declaration inherited "
                   f"from a pretrained base ({factory}); the saved statistics that would decide it are not available "
                   f"(a Hub check reads JSON metadata only, so it never decides this). {consequence}",
                   ev(cp, "/features/" + key, expected), ev(other, "/features/" + original, actual),
                   remedy="Audit a local download with the saved normalizer statistics to decide.")
        return True
    return False


def _reconcile_stale_statistics(cp: Snapshot, dep: Snapshot | None, report: Report, key: str, width: int,
                                source: str) -> None:
    """Statistics that are valid at the dataset width are not a fault of their own under a stale declaration.

    The normalizer applies saved statistics elementwise and never reads the declared feature shape
    (src/lerobot/processor/normalize_processor.py:305-424), so the width error is the stale declaration again,
    unless a deployment supplies a different width: then normalization itself cannot run, and the error stays.
    """
    if dep is not None:
        renames = feature_mapping(cp, dep)
        original = next((k for k in dep.features if renames.get(k, k) == key), None)
        supplied = dep.features[original]["shape"] if original is not None else None
        if supplied is not None and supplied != [width]:
            report.add("deployment.statistics_width", "error",
                       f"{key}: the deployment supplies {supplied}, but the checkpoint's saved normalizer statistics have "
                       f"{width} values (the training data's width). {LEROBOT} applies the saved statistics elementwise "
                       "(src/lerobot/processor/normalize_processor.py:352-424), so this state cannot be normalized" +
                       ("; the chunk runner does not catch it because config.json declares the same stale width."
                        if supplied == cp.features.get(key, {}).get("shape") else "."),
                       ev(dep, "/features/" + original, dep.features[original]),
                       remedy=f"Supply the {width}-value state the policy was trained on, and re-declare input_features.")
            return
    for side in ("pre", "post"):
        step = normalizer(cp, side)
        features = (step or {}).get("config", {}).get("features") or {}
        if not step or not isinstance(features.get(key), dict):
            continue
        if check_stat(step["stats"].get(key, {}), normal_mode(step, key, features[key]), width) is not None:
            continue
        for finding in report.findings:
            if finding.code == "normalization.statistics" and finding.status == "error" \
                    and finding.message.startswith(f"{side}:{key}: "):
                finding.status = "warning"
                finding.message = (f"{side}:{key}: saved statistics have {width} values where the processor declares "
                                   f"{features[key]['shape']}; they are valid at the {source} width, so this is the stale "
                                   f"declaration reported for {source}.feature_shape, not separate damage.")
                finding.remedy = "Re-declare the processor and policy features from the training dataset."


def _spread(stats: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(centre, spread) of each dimension as the normalizer uses them: mean/std, or the lower bound and range."""
    if mode == "MEAN_STD":
        return np.asarray(stats["mean"], dtype=float), np.asarray(stats["std"], dtype=float)
    lo, hi = (np.asarray(stats[name], dtype=float) for name in MODES[mode])
    return lo, hi - lo


def _vector(value) -> np.ndarray | None:
    """A finite 1-D float vector from JSON or tensor numbers; None for strings, nesting, booleans or overflow."""
    if not isinstance(value, list) or not value or any(type(v) not in (int, float) for v in value):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except OverflowError:
        return None
    return array if np.isfinite(array).all() else None


def _mode_vectors(entry, mode: str) -> dict | str:
    """The mode's statistics as equal-length vectors, or a short reason they are unusable."""
    vectors = {name: _vector(entry.get(name)) for name in MODES[mode]}
    bad = [name for name, vector in vectors.items() if vector is None]
    if bad:
        return f"has {'/'.join(bad)} values that are not lists of finite numbers"
    if len({v.shape for v in vectors.values()}) != 1:
        return f"has {'/'.join(MODES[mode])} of different lengths"
    if mode == "MEAN_STD" and (vectors["std"] < 0).any():
        return "has a negative std"
    if mode != "MEAN_STD" and (vectors[MODES[mode][0]] > vectors[MODES[mode][1]]).any():
        return f"has {MODES[mode][0]} above {MODES[mode][1]}"
    return vectors


def _labels(dims: list[int], names: list | None) -> str:
    shown = dims[:8]
    text = ", ".join(f"{i} ({names[i]})" if names else str(i) for i in shown)
    return text + (f", ... ({len(dims) - len(shown)} more; {len(dims)} total)" if len(dims) > len(shown) else "")


def _number(value: float) -> str:
    return f"{value:.4g}" if np.isfinite(value) else "beyond float range"


def dataset_statistics_findings(ds: Snapshot) -> list[dict]:
    """Screen saved dataset statistics without assuming a policy's active mode, epsilon or output range."""
    stats = ds.config.get("stats")
    path = next((a.path for a in ds.artifacts if a.role == "dataset_stats"), None)
    from .statistics_counts import statistics_count_findings

    info_artifact = next((asdict(a) for a in ds.artifacts if a.role == 'dataset_info'), None)
    stats_artifact = next((asdict(a) for a in ds.artifacts if a.role == 'dataset_stats'), None)
    findings = statistics_count_findings(ds.config, stats, info_artifact=info_artifact,
                                         stats_artifact=stats_artifact)
    if not isinstance(stats, dict):
        return [{"code": "dataset.statistics_unassessed", "status": "unknown",
                 "message": "Saved dataset statistics are absent or not an object; normalization consequences are unknown.",
                 "evidence": [], "remedy": "Supply the original native stats.json and the intended normalizer mode and epsilon."}]
    pairs = (("min", "max"), ("q01", "q99"), ("q10", "q90"))
    conditional = ("The active normalization mode and epsilon are not established by stats.json; IDENTITY leaves "
                   "values unchanged. ")
    scope = " These are conditional statistics findings, not observed row or robot behavior."
    for key, feature in ds.features.items():
        role = "ACTION" if key.startswith("action") else kind(feature, key)
        if role == "VISUAL" or not key.startswith(("observation", "action")):
            continue
        entry = stats.get(key)
        pointer = "/" + key.replace("~", "~0").replace("/", "~1")
        evidence = [ev(ds, pointer, entry, path)]
        remedy = "Supply valid native statistics for this feature and check the intended normalizer mode and epsilon."
        if not isinstance(entry, dict):
            findings.append({"code": "dataset.statistics_unassessed", "status": "unknown",
                             "message": f"{key}: missing or unsupported statistics entry.", "evidence": evidence,
                             "remedy": remedy})
            continue
        vectors = {name: _vector(entry[name]) for name in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90")
                   if name in entry}
        widths = {len(v) for v in vectors.values() if v is not None}
        expected = feature.get("shape")
        wrong_width = isinstance(expected, list) and len(expected) == 1 and widths != {expected[0]}
        usable_pair = any(lo in vectors and hi in vectors for lo, hi in (("mean", "std"), *pairs))
        if not vectors or not usable_pair or any(v is None for v in vectors.values()) or len(widths) != 1 or wrong_width:
            findings.append({"code": "dataset.statistics_unassessed", "status": "unknown",
                             "message": f"{key}: statistic vectors are malformed, incomplete or inconsistent with the feature width.",
                             "evidence": evidence, "remedy": remedy})
            continue
        width = next(iter(widths))
        names = feature.get("names") if isinstance(feature.get("names"), list) and len(feature["names"]) == width else None
        invalid = [i for i in range(width) if ("std" in vectors and vectors["std"][i] < 0)
                   or any(lo in vectors and hi in vectors and vectors[lo][i] > vectors[hi][i] for lo, hi in pairs)]
        if invalid:
            findings.append({"code": "dataset.statistics_invalid", "status": "warning", "indices": invalid[:16],
                             "indices_total": len(invalid), "indices_omitted": max(0, len(invalid) - 16),
                             "message": conditional + f"{key}: dimension(s) {_labels(invalid, names)} have a negative std or reversed "
                                        "saved range/quantile bounds." + scope,
                             "evidence": evidence, "remedy": "Recompute the dataset statistics before selecting them for normalization."})
        if "min" in vectors and "max" in vectors:
            # These are arithmetic constraints on the supplied summaries, not
            # a comparison with dataset rows. Allow serialized float32 rounding
            # and small absolute noise; do not diagnose tiny spans as defects.
            low, high = vectors["min"], vectors["max"]
            scale = np.maximum(1., np.maximum(np.abs(low), np.abs(high)))
            if "mean" in vectors:
                scale = np.maximum(scale, np.abs(vectors["mean"]))
            tolerance = 1e-8 + 1e-6 * scale
            with np.errstate(over="ignore", invalid="ignore"):
                outside = ([i for i in range(width) if i not in invalid and
                            (vectors["mean"][i] < low[i] - tolerance[i]
                             or vectors["mean"][i] > high[i] + tolerance[i])]
                           if "mean" in vectors else [])
                span = high - low
                zero_with_range = ([i for i in range(width) if i not in invalid and vectors["std"][i] == 0
                                    and np.isfinite(span[i]) and span[i] > tolerance[i]]
                                   if "std" in vectors else [])
            for constraint, indices, description in (
                ("mean_within_range", outside, "saved mean lies outside its saved min/max"),
                ("zero_std_with_nonzero_range", zero_with_range, "saved std is exactly zero while saved min/max span differs"),
            ):
                if indices:
                    findings.append({"code": "dataset.statistics_inconsistent", "status": "error",
                        "constraint": constraint, "indices": indices[:16], "indices_total": len(indices),
                        "indices_omitted": max(0, len(indices) - 16),
                        "message": f"{key}: {description} on dimension(s) {_labels(indices, names)}. "
                                   "These summaries cannot describe the same finite-value cohort within "
                                   "1e-6 relative plus 1e-8 absolute rounding tolerance (scale floor 1). Dataset rows were not compared.",
                        "evidence": evidence,
                        "remedy": "Confirm the summaries describe the same feature/cohort and recompute them from its "
                                  "intended source rows into a separate owned candidate; preserve the original statistics."})
        spreads = [vectors["std"]] if "std" in vectors else []
        with np.errstate(over="ignore", invalid="ignore"):
            if "min" in vectors and "max" in vectors:
                spreads.append(vectors["max"] - vectors["min"])
        if any(not np.isfinite(v).all() for v in spreads):
            findings.append({"code": "dataset.statistics_unassessed", "status": "unknown",
                             "message": f"{key}: derived statistic range exceeds finite arithmetic.",
                             "evidence": evidence, "remedy": remedy})
            continue
        constant = [i for i in range(width) if spreads and i not in invalid and all(s[i] == 0 for s in spreads)]
        small = [i for i in range(width) if spreads and i not in invalid and i not in constant
                 and all(0 <= s[i] <= DEGENERATE_SPREAD for s in spreads)]
        if constant:
            consequence = ("MEAN_STD action decoding with an exactly zero std returns the saved mean for any model output. "
                           "Range-based decoding instead substitutes its configured epsilon, so even a zero saved range "
                           "does not prove the output is constant." if role == "ACTION" else
                           "If used for input normalization, MEAN_STD divides by std + configured epsilon; range modes "
                           "substitute configured epsilon when the range is zero. No numeric gain is established here.")
            findings.append({"code": "dataset.constant_dimension", "status": "warning", "indices": constant[:16],
                             "indices_total": len(constant), "indices_omitted": max(0, len(constant) - 16),
                             "message": conditional + f"{key}: saved statistics declare zero spread in dimension(s) {_labels(constant, names)}. "
                                        + consequence + " This may be legitimate for an unused or padded feature." + scope,
                             "evidence": evidence, "remedy": "Check these dimensions are meant to stay fixed and verify the actual normalization configuration; otherwise improve the recording."})
        if small:
            findings.append({"code": "dataset.small_spread", "status": "warning", "indices": small[:16],
                             "indices_total": len(small), "indices_omitted": max(0, len(small) - 16),
                             "message": conditional + f"{key}: saved dimension(s) {_labels(small, names)} have small positive spread "
                                        f"(at most {DEGENERATE_SPREAD:g}); positive spread is not a constant dimension." + scope,
                             "evidence": evidence, "remedy": "Inspect the recorded scale and intended normalization; do not infer immobility from a small positive spread."})
        collapsed = [i for i in range(width) if i not in invalid and any(
            lo in vectors and hi in vectors and vectors[hi][i] == vectors[lo][i] for lo, hi in pairs[1:])]
        if collapsed:
            findings.append({"code": "dataset.collapsed_quantiles", "status": "warning", "indices": collapsed[:16],
                             "indices_total": len(collapsed), "indices_omitted": max(0, len(collapsed) - 16),
                             "message": conditional + f"{key}: saved quantile bounds are exactly equal on dimension(s) {_labels(collapsed, names)}. "
                                        "If the matching quantile normalization mode is selected, its denominator is the configured "
                                        f"epsilon ({LEROBOT} src/lerobot/processor/normalize_processor.py:397-421). "
                                        "Other modes are unaffected." + scope,
                             "evidence": evidence, "remedy": "Before training a quantile-normalized policy on this dataset, inspect its actual epsilon and recompute quantiles or exclude the intended fixed dimensions."})
    return findings

def _normalized(stats: dict, other: dict, mode: str, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Offset and scale of ``other``'s distribution in the units ``stats`` normalize to (training data: 0 and 1)."""
    centre, spread = _spread(stats, mode)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if mode == "MEAN_STD":
            mean, std = _spread(other, mode)
            return (mean - centre) / (spread + eps), std / (spread + eps)
        lo, width = _spread(other, mode)
        denom = np.where(spread == 0, eps, spread)
        low, high = 2 * (lo - centre) / denom - 1, 2 * (lo + width - centre) / denom - 1
        return (low + high) / 2, (high - low) / 2


def _training_source(cp: Snapshot) -> tuple[dict, str | None]:
    """Evidence and the dataset repo_id the checkpoint's train_config.json names, if it is present and says so."""
    path = cp.root / "train_config.json"
    if not path.is_file() or path.is_symlink():
        return {}, None
    try:
        path = contained(cp.root, "train_config.json")
        document = read_json(path)
    except InputError:
        return {}, None  # Optional context: an unreadable training config only removes the explanation.
    repo = document["dataset"].get("repo_id") if isinstance(document.get("dataset"), dict) else None
    cp.record(path, "training_config", document)
    return ev(cp, "/dataset/repo_id", repo, str(path)), repo if isinstance(repo, str) else None


def _statistics_checks(cp: Snapshot, ds: Snapshot | None, report: Report) -> None:
    """Saved normalizer statistics that are degenerate, or far from the dataset's, judged by what deployment does.

    Deployment normalizes with the checkpoint's saved statistics: lerobot-eval loads them unchanged
    (src/lerobot/scripts/lerobot_eval.py:808-818), and a pretrained processor ignores dataset statistics except for
    the policies in REBUILT_STATISTICS (src/lerobot/policies/factory.py:182-210). A fresh fine-tune writes the
    training dataset's statistics into the checkpoint (src/lerobot/scripts/lerobot_train.py:677-679, 697-703).
    """
    renames = feature_mapping(cp, ds) if ds else {}
    dataset_stats = (ds.config.get("stats") if ds else None) or {}
    stats_path = next((a.path for a in ds.artifacts if a.role == "dataset_stats"), None) if ds else None
    # Plugin policy types can rebuild statistics through their own factories (src/lerobot/policies/factory.py:446-463).
    unmodeled = cp.config.get("type") in REBUILT_STATISTICS or cp.config.get("type") not in LEROBOT_POLICY_TYPES
    training = None
    for side in ("pre", "post"):
        step = normalizer(cp, side)
        if not step or not step.get("state_file") or step.get("missing_state"):
            continue
        eps = step["config"].get("eps", 1e-8)
        if type(eps) not in (int, float) or not 0 < eps < 1e300:
            continue  # An unusable epsilon is reported by _processor_checks; its arithmetic is not modeled here.
        eps = float(eps)
        selected = step["config"].get("normalize_observation_keys")
        artifact = next(a for a in cp.artifacts if a.path == step["state_file"])
        for key, feature in (step["config"].get("features") or {}).items():
            role = kind(feature, key)
            # Inputs matter on the preprocessor; only the action is decoded by the postprocessor.
            if role == "VISUAL" or (side == "pre") == (role == "ACTION"):
                continue
            if isinstance(selected, list) and role != "ACTION" and key not in selected:
                continue  # Left unnormalized (src/lerobot/processor/normalize_processor.py:281-283).
            mode = normal_mode(step, key, feature)
            saved = _mode_vectors(step["stats"].get(key, {}), mode) if mode in FORWARD_LINES else None
            if not isinstance(saved, dict):
                continue  # IDENTITY leaves values unchanged; invalid statistics are reported by _processor_checks.
            width = len(next(iter(saved.values())))
            original = next((k for k in (ds.features if ds else {}) if renames.get(k, k) == key), None)
            names = next((n for n in ((ds.features[original].get("names") if original else None),
                                      cp.features.get(key, {}).get("names"))
                          if isinstance(n, list) and len(n) == width), None)
            evidence = {"path": artifact.path, "sha256": artifact.sha256, "format": "safetensors",
                        "tensors": [f"{key}.{name}" for name in MODES[mode]]}
            if unmodeled:
                report.add("normalization.dataset_statistics", "unknown",
                           f"{side}:{key}: {cp.config.get('type')} may rebuild or pad saved statistics when loading "
                           "(src/lerobot/policies/factory.py:182-198, 446-463), which RoboClue does not model. "
                           "Effective normalization and decoding consequences remain unknown, even if saved statistics are degenerate.",
                           evidence, remedy="Capture the effective native processor and its loaded statistics before interpreting runtime effects.")
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                centre, spread = _spread(saved, mode)
            if not np.isfinite(spread).all():
                report.add("normalization.dataset_statistics", "unknown",
                           f"{side}:{key}: derived saved range exceeds finite arithmetic; runtime effects are unassessed.",
                           evidence, remedy="Inspect the original statistics and effective native processor.")
                continue
            flat = [int(i) for i in np.flatnonzero(spread <= DEGENERATE_SPREAD)]
            if flat:
                values = [round(float(centre[i]), 6) for i in flat[:8]]
                constant = dict(evidence, indices=flat[:16], indices_total=len(flat),
                                indices_omitted=max(0, len(flat) - 16), saved_values=values,
                                saved_values_omitted=max(0, len(flat) - len(values)),
                                spreads=[float(spread[i]) for i in flat[:16]], mode=mode, configured_eps=eps)
                if role == "ACTION":
                    denom = spread if mode == "MEAN_STD" else np.where(spread == 0, eps, spread)
                    inverse_scales = denom if mode == "MEAN_STD" else denom / 2
                    constant["inverse_scales"] = [float(inverse_scales[i]) for i in flat[:16]]
                    formula = "output * std + mean" if mode == "MEAN_STD" else "(output + 1) * denominator / 2 + lower_bound"
                    report.add("normalization.constant_dimension", "pass",
                               f"{key}: dimension(s) {_labels(flat, names)} have saved spread at most {DEGENERATE_SPREAD:g} ({mode}). "
                               f"The native inverse is {formula}; a unit change in normalized output changes these decoded values by "
                               f"{constant['inverse_scales'][:8]} (src/lerobot/processor/normalize_processor.py:{INVERSE_LINES[mode]}). "
                               "Only MEAN_STD with exactly zero std is independent of model output. Range modes substitute configured "
                               "epsilon for an exactly zero range. No model-output bound, immobility or task consequence is inferred.",
                               constant, remedy="Check the intended scale and effective processor; a small positive spread is not a fixed actuator.")
                else:
                    with np.errstate(over="ignore", divide="ignore"):
                        gains = 1 / (spread + eps) if mode == "MEAN_STD" else 2 / np.where(spread == 0, eps, spread)
                    gain = float(gains[flat[0]])
                    constant["forward_gains"] = [float(gains[i]) if np.isfinite(gains[i]) else None for i in flat[:16]]
                    report.add("normalization.constant_input", "warning" if any(gains[i] > MAX_SCALE for i in flat) else "pass",
                               f"{key}: dimension(s) {_labels(flat, names)} have saved spread at most {DEGENERATE_SPREAD:g} ({mode}). "
                               f"Under this saved mode and configured eps={eps:g}, the first listed dimension multiplies a deviation by "
                               f"about {_number(gain)} (src/lerobot/processor/normalize_processor.py:{FORWARD_LINES[mode]}). "
                               "This is a scalar normalization consequence; the robot input distribution and physical effect are unknown.",
                               constant, remedy="Check the expected input variation and effective processor; retrain or change inputs only if this scale is unintended.")
            if not dataset_stats or original is None:
                continue
            other, pointer = dataset_stats.get(original), "/" + original.replace("~", "~0").replace("/", "~1")
            dataset_ref = ev(ds, pointer, other, stats_path)
            fine_tune = (f"fine-tuning this checkpoint on this dataset replaces the saved statistics with the dataset's "
                         f"({LEROBOT} src/lerobot/scripts/lerobot_train.py:677-679, 697-703)")
            if other is None or other == {}:
                if not unmodeled:
                    report.add("normalization.dataset_statistics", "warning",
                               f"{side}:{key}: the dataset's stats.json has no statistics for {original}. Deploying the "
                               f"checkpoint is unaffected (it keeps its saved statistics), but {fine_tune}, and then {LEROBOT} "
                               f"passes {key} through unnormalized (src/lerobot/processor/normalize_processor.py:332-334).",
                               evidence, dataset_ref,
                               remedy="Compute statistics for this feature before fine-tuning on the dataset.")
                continue
            problem = (f"is a {type(other).__name__}, not a statistics object" if not isinstance(other, dict) else
                       f"lacks {'/'.join(n for n in MODES[mode] if n not in other)}"
                       if any(n not in other for n in MODES[mode]) else None)
            if problem is None:
                parsed = _mode_vectors(other, mode)
                problem = None if isinstance(parsed, dict) else parsed
            if problem is not None:
                if not unmodeled:
                    raises = problem.startswith("lacks")
                    report.add("normalization.dataset_statistics", "warning",
                               f"{side}:{key}: the dataset's stats.json entry for {original} {problem}. Deploying the "
                               f"checkpoint is unaffected (it keeps its saved statistics), but {fine_tune}" +
                               (f", and then normalization raises ValueError (src/lerobot/processor/normalize_processor.py:"
                                f"{RAISE_LINES[mode]})." if raises else ", so fix this entry first."),
                               evidence, dataset_ref,
                               remedy="Add or recompute the dataset's statistics before fine-tuning on it (LeRobot's "
                                      "augment_dataset_quantile_stats.py adds quantiles).")
                continue
            if len(next(iter(parsed.values()))) != width:
                if not unmodeled:
                    report.add("normalization.dataset_statistics", "warning",
                               f"{side}:{key}: the dataset's stats.json has {len(next(iter(parsed.values())))} values per "
                               f"statistic for {original}, where the checkpoint's saved statistics have {width}. Deploying "
                               f"the checkpoint is unaffected (it keeps its saved statistics), but {fine_tune}, and those "
                               "cannot be applied elementwise to this feature.",
                               evidence, dataset_ref,
                               remedy="Make the dataset's statistics match its declared feature width before fine-tuning on it.")
                continue
            if all(np.array_equal(saved[n], parsed[n]) for n in MODES[mode]):
                report.add("normalization.dataset_statistics", "pass",
                           f"{side}:{key}: saved {mode} statistics equal the dataset's stats.json for {original}.",
                           evidence, dataset_ref)
                continue
            offset, scale = _normalized(saved, parsed, mode, eps)
            other_centre, other_spread = _spread(parsed, mode)
            live = spread > 0
            # Constant in the checkpoint but not in the dataset: the dataset varies, or sits at another value.
            stuck = [int(i) for i in np.flatnonzero(~live & ((other_spread > 0) | (other_centre != centre)))]
            far = [int(i) for i in np.flatnonzero(live & ~(np.isfinite(offset) & np.isfinite(scale)
                                                           & (np.abs(offset) <= MAX_OFFSET) & (scale <= MAX_SCALE)))]
            ranking = np.where(live, np.nan_to_num(np.abs(offset) + np.abs(np.log(np.clip(scale, 1e-12, 1e300))),
                                                   nan=np.inf, posinf=np.inf), -1)
            worst = far[int(np.argmax(ranking[far]))] if far else int(np.argmax(ranking))
            summary = (f"offset {_number(float(offset[worst]))} and scale x{_number(float(scale[worst]))} on dimension "
                       f"{_labels([worst], names)} in the saved normalizer's nominal coordinates")
            drift = dict(evidence, offsets=[round(float(v), 4) if np.isfinite(v) else None for v in offset[:16]],
                         scales=[round(float(v), 4) if np.isfinite(v) else None for v in scale[:16]],
                         constant_in_checkpoint_but_not_dataset=stuck[:16],
                         constant_in_checkpoint_but_not_dataset_total=len(stuck),
                         constant_in_checkpoint_but_not_dataset_omitted=max(0, len(stuck) - 16))
            if not far and not stuck:
                report.add("normalization.dataset_statistics", "pass",
                           f"{side}:{key}: saved {mode} statistics differ from the dataset's stats.json for {original}, but "
                           f"the dataset's data stays close to them: at worst {summary}, within RoboClue's threshold "
                           f"(|offset| <= {MAX_OFFSET:g} and normalized scale <= {MAX_SCALE:g})." if live.any() else
                           f"{side}:{key}: saved {mode} statistics and the dataset's stats.json for {original} agree on "
                           "every dimension the checkpoint keeps constant.",
                           drift, dataset_ref)
                continue
            if training is None:
                training = _training_source(cp)
                if training[0]:
                    report.artifacts.append(asdict(next(a for a in cp.artifacts if a.path == training[0]["path"])))
            source, repo = training
            context = (f" The checkpoint's train_config.json names {repo} as its training dataset. A fresh fine-tune writes "
                       f"that dataset's statistics into the checkpoint ({LEROBOT} src/lerobot/scripts/lerobot_train.py:"
                       f"677-679, 697-703), so if this dataset is {repo}, the dataset changed after training, the statistics "
                       "were replaced on purpose (hotswap_stats, src/lerobot/processor/normalize_processor.py:562-588), or "
                       "a resumed run kept adapted statistics; none of these by itself makes the checkpoint faulty."
                       if repo else " These files do not establish what data the checkpoint was trained on, so this is not "
                       "evidence of a faulty checkpoint.")
            parts = []
            if far:
                parts.append(f"data distributed like the dataset's stats.json for {original} lands at {summary}; "
                             f"{len(far)} dimension(s) {_labels(far, names)} are past RoboClue's threshold (|offset| > "
                             f"{MAX_OFFSET:g}, or normalized scale > {MAX_SCALE:g} under the saved mode and epsilon)")
            if stuck:
                detail = ("with exactly zero saved std, MEAN_STD inverse returns the saved mean regardless of model output"
                          if side == "post" and mode == "MEAN_STD" else
                          "the range inverse substitutes configured epsilon and remains dependent on model output; no output bound is established"
                          if side == "post" else
                          "input deviations follow the saved mode's denominator (std + eps, or range with epsilon substituted at zero)")
                parts.append(f"dimension(s) {_labels(stuck, names)} have exactly zero saved spread but move or sit "
                             f"elsewhere in the dataset; {detail} (src/lerobot/processor/normalize_processor.py:"
                             f"{INVERSE_LINES[mode] if side == 'post' else FORWARD_LINES[mode]})")
            subject = ("the robot's data looks like this dataset, the policy receives inputs outside the range its saved "
                       "statistics describe" if side == "pre" else
                       "the robot should act like this dataset, the policy's decoded commands follow the saved statistics "
                       "instead")
            report.add("normalization.dataset_statistics", "unknown" if unmodeled else "warning",
                       f"{side}:{key}: under the checkpoint's saved {mode} statistics, " + "; ".join(parts) + ". " +
                       (f"{cp.config.get('type')} may rebuild or pad its saved statistics when loading "
                        "(src/lerobot/policies/factory.py:182-198, 446-463), which RoboClue does not model, so the effect is "
                        "unknown." if unmodeled else
                        f"{LEROBOT} deploys the saved statistics unchanged (src/lerobot/scripts/lerobot_eval.py:808-818; "
                        f"src/lerobot/policies/factory.py:182-210), so if {subject}.") + context,
                       *[e for e in (drift, dataset_ref, source) if e],
                       remedy="If this checkpoint is meant for data like this dataset, fine-tune on it (which writes its "
                              "statistics) or confirm the robot's data matches the saved statistics before deploying.")


def _feature_checks(cp: Snapshot, other: Snapshot, report: Report, renames: dict) -> None:
    for key, expected in cp.features.items():
        original = next((k for k in other.features if renames.get(k, k) == key), None)
        prefix = f"{other.kind}.feature"
        padded = re.fullmatch(r"observation\.images\.empty_camera_(\d+)", key)
        if original is None and padded and type(cp.config.get("empty_cameras")) is int \
                and int(padded.group(1)) < cp.config["empty_cameras"]:
            report.add(prefix + "_missing", "pass",
                       f"{key} is padded by the policy itself (empty_cameras={cp.config['empty_cameras']}); a {other.kind} "
                       f"never supplies it ({LEROBOT} src/lerobot/policies/pi0/configuration_pi0.py:132-138, "
                       "src/lerobot/policies/pi05/configuration_pi05.py:161-167).",
                       ev(cp, "/empty_cameras", cp.config["empty_cameras"]))
            continue
        supplied_cameras = any(kind(f, k) == "VISUAL" and any(renames.get(o, o) == k for o in other.features)
                               for k, f in cp.features.items())
        if original is None and kind(expected, key) == "VISUAL" and cp.config.get("type") in CAMERA_TOLERANT \
                and supplied_cameras and other.kind == "dataset":
            report.add(prefix + "_missing", "warning",
                       f"{other.kind} never supplies {key}; {cp.config['type']} runs with the cameras present and pads or "
                       f"drops the rest ({LEROBOT} {CAMERA_TOLERANT[cp.config['type']]}), so this camera is never seen.",
                       ev(cp, "/features/" + key, expected), ev(other, "/features", list(other.features)),
                       remedy="Confirm the deployment supplies the same cameras the training data did.")
            continue
        if original is None:
            report.add(prefix + "_missing", "error", f"{other.kind} does not supply {key}.",
                       ev(cp, "/features/" + key, expected), ev(other, "/features", list(other.features)),
                       remedy="Provide the expected feature or declare an explicit camera rename.")
            continue
        actual = other.features[original]
        a, b = expected["shape"], actual["shape"]
        if kind(expected, key) == "VISUAL":
            a, b = _visual_shape(expected), _visual_shape(actual)
            if a is None or b is None:
                report.add(prefix + "_layout", "unknown", f"Image layout for {original} is not declared.", ev(other, "/features/" + original, actual))
                continue
        if a != b and kind(expected, key) == "VISUAL" and cp.config.get("type") == "smolvla" \
                and cp.config.get("resize_imgs_with_padding") and a[0] == b[0]:
            report.add(prefix + "_shape", "pass",
                       f"{key}: {other.kind} images {b} are resized natively to {cp.config['resize_imgs_with_padding']} "
                       f"({LEROBOT} src/lerobot/policies/smolvla/modeling_smolvla.py:389-391).",
                       ev(cp, "/resize_imgs_with_padding", cp.config["resize_imgs_with_padding"]),
                       ev(other, "/features/" + original, actual))
            continue
        # A deployment is downgraded only when the saved statistics show it supplies the trained width: the default
        # sync rollout engine pads it (src/lerobot/inference/sync.py:109); only the remote runner validates it.
        if a != b and kind(expected, key) != "VISUAL" and key in (cp.config.get("input_features") or {}) \
                and _stale_state_declaration(cp, other, report, key, original, expected, actual):
            continue
        if a != b:
            visual = kind(expected, key) == "VISUAL"
            report.add(prefix + "_shape", "warning" if visual else "error",
                       f"{key}: policy shape {a} differs from {other.kind} shape {b}.",
                       ev(cp, "/features/" + key, expected), ev(other, "/features/" + original, actual),
                       remedy="Verify the resize/preprocessing applied to this camera." if visual else
                       f"Supply {key} with the declared shape {a}; if the training data really had {b}, "
                       f"re-declare the checkpoint's {'output' if key in (cp.config.get('output_features') or {}) else 'input'}"
                       "_features instead.")
        else:
            report.add(prefix + "_shape", "pass", f"{key}: dimensions agree after declared layout conversion.",
                       ev(other, "/features/" + original, actual))


def _relative_order_checks(cp: Snapshot, others: list[Snapshot], report: Report) -> None:
    """The decoder adds the first action_dim raw-state coordinates positionally."""
    dim = cp.features.get("action", {}).get("shape", [0])[-1]
    for step in cp.pre:
        if step_kind(step["name"]) != "relative" or not step["config"].get("enabled"):
            continue
        expected = step["config"].get("action_names")
        reference = ev(cp, step["pointer"] + "/config/action_names", expected, step["source"])
        if expected is not None and (not isinstance(expected, list) or len(expected) != dim):
            report.add("action.processor_order_names", "error", "Relative processor action_names must cover every action dimension.", reference)
            continue
        for source in [cp, *others]:
            for key in ("action", "observation.state"):
                feature = source.features.get(key, {})
                actual = feature.get("names")
                if source is cp and actual is None:
                    continue  # Checkpoint feature names are commonly absent; compare supplied interfaces below.
                # State may contain additional coordinates; upstream uses its first action_dim values.
                compared = actual[:dim] if actual is not None and key == "observation.state" else actual
                status = "unknown" if expected is None or compared is None else "pass" if expected == compared else "error"
                report.add("action.processor_order_names", status,
                           f"{source.kind}:{key}: compare the relative processor order with the positional action/anchor order.",
                           reference, ev(source, "/features/" + key + "/names", actual),
                           remedy="Export the effective processor and named interfaces; verify any reorder on a saved chunk.")


def audit(checkpoint: str, dataset: str | None = None, deployment: str | None = None) -> dict:
    cp = load_checkpoint(checkpoint)
    ds = load_dataset(dataset) if dataset else None
    dep = load_deployment(deployment) if deployment else None
    report = Report()
    snapshots = [s for s in (cp, ds, dep) if s]
    report.artifacts = [asdict(a) for s in snapshots for a in s.artifacts]
    if ds:
        for key, feature in ds.features.items():
            if feature.get("coordinate_names_status") == "duplicate":
                pointer = "/features/" + key.replace("~", "~0").replace("/", "~1") + "/names"
                labels = feature["raw_names"]
                labels = next(iter(labels.values())) if isinstance(labels, dict) else labels
                repeated = sorted({n for n in labels if labels.count(n) > 1})
                report.add("dataset.coordinate_names_duplicate", "warning",
                           f"{key}: coordinate labels repeat {repeated}; name-based ordering or selection (for example "
                           "relative_exclude_joints) is ambiguous. Positions and dimensions remain checkable.",
                           ev(ds, pointer, feature["raw_names"]),
                           remedy="Give every coordinate a unique source-backed name in the dataset metadata.")
            if feature.get("coordinate_names_status") == "unresolved":
                pointer = "/features/" + key.replace("~", "~0").replace("/", "~1") + "/names"
                report.add("dataset.coordinate_names_unresolved", "unknown",
                           f"{key}: source labels do not establish one name per coordinate; dimensions remain separately checkable.",
                           ev(ds, pointer, feature["raw_names"]),
                           remedy="Supply a source-backed per-coordinate declaration before comparing joint order; do not expand group labels.")
    _processor_checks(cp, report)
    _relative_order_checks(cp, [s for s in (ds, dep) if s], report)
    # Keyed by the effective policy key after saved renames, as _feature_checks compares them (mapping.py:17-34);
    # a raw key renamed away no longer stands for the feature it used to name.
    report.dataset_features = {renamed: ds.features[raw] for raw, renamed in feature_mapping(cp, ds).items()} if ds else {}
    for other in (ds, dep):
        if other:
            _feature_checks(cp, other, report, feature_mapping(cp, other))
    for key, width, source in report.deferred:
        _reconcile_stale_statistics(cp, dep, report, key, width, source)
    _statistics_checks(cp, ds, report)
    if ds and dep:
        actual_fps = dep.config.get("fps")
        report.add("timing.fps", "unknown" if actual_fps is None else "pass" if actual_fps == ds.config["fps"] else "warning",
                   "Compare dataset sampling rate and declared deployment command rate; a difference requires timing validation.",
                   ev(ds, "/fps", ds.config["fps"]), ev(dep, "/fps", actual_fps))
        if ds.config.get("robot_type") not in (None, "unknown") and dep.config.get("robot_type") not in (None, "unknown"):
            if ds.config["robot_type"] != dep.config["robot_type"]:
                report.add("robot.type", "warning", "Dataset and deployment name different robot types; inspect calibration and action semantics.",
                           ev(ds, "/robot_type", ds.config["robot_type"]), ev(dep, "/robot_type", dep.config["robot_type"]))
        for key in ("action", "observation.state"):
            a, b = ds.features.get(key, {}).get("names"), dep.features.get(key, {}).get("names")
            if a is None or b is None:
                report.add("joints.order", "unknown", f"{key}: both named dimension orders are required to check ordering.",
                           ev(ds, "/features/" + key + "/names", a), ev(dep, "/features/" + key + "/names", b))
            elif a != b:
                report.add("joints.order", "error", f"{key}: dataset and deployment dimension order differs.",
                           ev(ds, "/features/" + key + "/names", a), ev(dep, "/features/" + key + "/names", b),
                           remedy="Apply an explicit tested reorder at the interface, or restore the recorded order.")
            else:
                report.add("joints.order", "pass", f"{key}: declared dimension order agrees.", ev(dep, "/features/" + key + "/names", b))
    if dep:
        history = cp.config.get("n_obs_steps")
        actual_history = dep.config.get("n_obs_steps")
        if history is not None:
            report.add("observation.history", "unknown" if actual_history is None else "pass" if history == actual_history else "error",
                       "Declared runtime observation history is compared to the checkpoint; timestamp selection and stacking order require saved observations.",
                       ev(cp, "/n_obs_steps", history), ev(dep, "/n_obs_steps", actual_history),
                       remedy="Match the checkpoint's observation history and verify timestamps/stacking on a saved input.")
        for key in ("units", "frame", "command_representation", "gripper_convention"):
            expected = dep.config.get("training_contract", {}).get(key)
            actual = dep.config.get(key)
            report.add("semantics." + key, "unknown" if expected is None or actual is None else "pass" if expected == actual else "error",
                       f"{key}: comparison uses explicit user-supplied training/deployment declarations; numeric metadata alone cannot establish semantics.",
                       ev(dep, "/training_contract/" + key, expected), ev(dep, "/" + key, actual),
                       remedy="Supply source-backed semantics; inspect the collection and command interfaces.")
        pre = [s for s in cp.pre if step_kind(s["name"]) == "relative"]
        expected = "relative" if any(s["config"].get("enabled") for s in pre) else "absolute" if pre else None
        # Absence of a relative step is not proof of absolute semantics for an arbitrary policy.
        actual = dep.config.get("model_output_representation")
        if expected is not None:
            report.add("action.model_representation", "unknown" if actual is None else "pass" if expected == actual else "error",
                       "Compare the saved relative processor with the declared interpretation of model outputs; an omitted declaration remains unknown.",
                       ev(cp, "/steps", pre, pre[0]["source"]), ev(dep, "/model_output_representation", actual))
        absolute_steps = [s for s in cp.post if step_kind(s["name"]) == "absolute" and s["config"].get("enabled")]
        if absolute_steps and all(step_kind(s["name"]) in ("absolute", "unnormalize", "device") for s in cp.post):
            command = dep.config.get("command_representation")
            report.add("action.command_representation", "unknown" if command is None else "pass" if command == "absolute" else "error",
                       "The saved decoder adds the raw state anchor and produces absolute commands; compare the actual command interface.",
                       ev(cp, absolute_steps[0]["pointer"], absolute_steps[0]["config"], absolute_steps[0]["source"]),
                       ev(dep, "/command_representation", command),
                       remedy="Match the driver's command interpretation to the saved decoder and verify saved commands before deployment.")
        runtime_mode = dep.config.get("normalization_mapping")
        if runtime_mode is not None:
            post = normalizer(cp, "post")
            if post and runtime_mode != post["config"].get("norm_map"):
                report.add("normalization.runtime_override", "error", "Declared runtime normalization differs from the saved postprocessor.",
                           ev(dep, "/normalization_mapping", runtime_mode), ev(cp, post["pointer"], post["config"].get("norm_map"), post["source"]))
        execute = dep.config.get("execute_steps")
        trained = cp.config.get("n_action_steps")
        chunk = cp.config.get("chunk_size")
        if execute is not None and chunk is not None and execute > chunk:
            report.add("timing.execution_exceeds_chunk", "error", "Deployment executes more actions than the predicted chunk contains.",
                       ev(dep, "/execute_steps", execute), ev(cp, "/chunk_size", chunk))
        elif execute is not None and trained is not None:
            report.add("timing.execution_horizon", "pass" if execute == trained else "warning",
                       "Declared execution horizon matches the checkpoint default." if execute == trained else "Execution horizon differs from the checkpoint default; evaluate its effect on feedback cadence.",
                       ev(dep, "/execute_steps", execute), ev(cp, "/n_action_steps", trained))
        else:
            report.add("timing.execution_horizon", "unknown", "Execution horizon cannot be compared from supplied metadata.", ev(dep, "/execute_steps", execute))
    else:
        report.add("deployment.missing", "unknown", "No deployment contract supplied; runtime interfaces were not compared.", remedy="Create and fill a deployment contract with policy-doctor template.")
    return report.to_dict()
