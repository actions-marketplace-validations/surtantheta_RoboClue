"""Bounded native artifact intake. No checkpoint code or model weights are executed."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_STATS_BYTES = 32 * 1024 * 1024


class InputError(ValueError):
    """An artifact cannot be safely or unambiguously interpreted."""


def finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise InputError("Non-finite numbers are not accepted")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


HUB_LOCAL_ONLY = ("hf:// sources are read only with --fetch by `check`, `preflight` and `readiness`; this command reads "
                  "local files. Use one of those, or download the repository and pass its directory.")


def _json_file_problem(path: Path) -> str:
    """Name the actual problem: a Hub URI, a missing directory or file, or a directory where a file was expected."""
    if "hf:" in path.parts:
        return HUB_LOCAL_ONLY
    if not path.parent.is_dir():
        return f"Directory not found: {path.parent} (looking for {path.name})"
    if not path.exists():
        return f"File not found: {path}"
    if path.is_dir():
        return f"Expected a JSON file, found a directory: {path}"
    return f"Expected a JSON file no larger than {MAX_JSON_BYTES} bytes: {path}"


def read_json_value(path: str | Path) -> Any:
    """One bounded read of any JSON value, with read_json's checks except the object requirement."""
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            raise InputError(_json_file_problem(path))
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        finite_tree(value)
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InputError(f"Cannot read JSON {path}: {exc}") from exc


def read_json(path: str | Path) -> dict:
    value = read_json_value(path)
    if not isinstance(value, dict):
        raise InputError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    from .fs_guard import open_regular

    digest = hashlib.sha256()
    with open_regular(path, 'artifact') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise InputError("Artifact references must be paths within the checkpoint")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InputError("Artifact reference escapes the checkpoint through a symlink")
    return path


def shape(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(x) is not int or x <= 0 for x in value):
        raise InputError(f"{label}: shape must be a nonempty list of positive integers")
    return value


def names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise InputError("Feature names must be a list of nonempty strings (or a single-axis mapping)")
    if len(set(value)) != len(value):
        raise InputError("Feature names contain duplicates")
    return value


@dataclass
class Artifact:
    path: str
    role: str
    sha256: str
    size_bytes: int


@dataclass
class Snapshot:
    kind: str
    root: Path
    config: dict
    features: dict = field(default_factory=dict)
    pre: list[dict] = field(default_factory=list)
    post: list[dict] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    json_documents: dict[str, dict] = field(default_factory=dict, repr=False)

    def record(self, path: Path, role: str, document: dict | None = None) -> None:
        if document is not None:
            self.json_documents[str(path.resolve())] = document
        if any(x.path == str(path.resolve()) for x in self.artifacts):
            return
        self.artifacts.append(Artifact(str(path.resolve()), role, sha256(path), path.stat().st_size))

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "root": str(self.root),
            "policy_type": self.config.get("type") if self.kind == "checkpoint" else None,
            "features": self.features,
            "fps": self.config.get("fps"),
            "processors": {
                "pre": [step["name"] for step in self.pre],
                "post": [step["name"] for step in self.post],
            },
            "artifacts": [asdict(a) for a in self.artifacts],
        }


def _features(raw: Any, *, dataset_names: bool = False) -> dict:
    if not isinstance(raw, dict):
        raise InputError("features must be an object")
    result = {}
    for key, feature in raw.items():
        if not isinstance(feature, dict):
            raise InputError(f"Feature {key} must be an object")
        item = dict(feature)
        if dataset_names:
            # These are derived reader annotations, never source authority.
            item.pop("coordinate_names_status", None)
            item.pop("raw_names", None)
        item["shape"] = shape(feature.get("shape"), key)
        if "names" in feature and key in ("action", "observation.state"):
            value = feature["names"]
            # Native dataset metadata may be readable without declaring one
            # name per vector coordinate. Retain the original field, never
            # expand a label or infer a joint order from these encodings.
            unresolved = (dataset_names and len(item["shape"]) == 1
                          and ((isinstance(value, list) and bool(value) and len(value) != item["shape"][0])
                               or value == {"axes": None}))
            # One label per coordinate, but repeated labels: positions stay checkable,
            # name-based order and selection do not. Checkpoint names remain strict.
            flat = next(iter(value.values())) if isinstance(value, dict) and len(value) == 1 else value
            duplicate = (dataset_names and len(item["shape"]) == 1 and isinstance(flat, list)
                         and len(flat) == item["shape"][0] and len(set(flat)) != len(flat)
                         and all(isinstance(x, str) and x for x in flat))
            if unresolved:
                if isinstance(value, list):
                    names(value)  # Duplicate / non-string labels remain invalid.
                item["names"] = None
                item["raw_names"] = list(value) if isinstance(value, list) else dict(value)
                item["coordinate_names_status"] = "unresolved"
            elif duplicate:
                item["names"] = None
                item["raw_names"] = list(value) if isinstance(value, list) else dict(value)
                item["coordinate_names_status"] = "duplicate"
            else:
                item["names"] = names(value)
            if item["names"] is not None and len(item["names"]) != item["shape"][-1]:
                raise InputError(f"{key}: names length does not match vector dimension")
        result[key] = item
    return result


def _pipeline(snapshot: Snapshot, filename: str, role: str) -> list[dict]:
    path = contained(snapshot.root, filename)
    if not path.exists():
        return []
    data = read_json(path)
    snapshot.record(path, role, data)
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) > 100:
        raise InputError(f"{filename}: steps must be a list with at most 100 entries")
    result = []
    for index, entry in enumerate(steps):
        if not isinstance(entry, dict) or not isinstance(entry.get("config", {}), dict):
            raise InputError(f"Invalid processor step {index} in {filename}")
        name = entry.get("registry_name") or entry.get("class")
        if not isinstance(name, str):
            raise InputError(f"Processor {index} has no registry_name or class")
        step = {"name": name, "config": entry.get("config", {}), "stats": {},
                "source": str(path.resolve()), "pointer": f"/steps/{index}"}
        config = step["config"]
        if step_kind(name) in ("relative", "absolute") and type(config.get("enabled", False)) is not bool:
            raise InputError("Relative/absolute processor enabled must be a boolean")
        if step_kind(name) == "relative":
            if config.get("action_names") is not None:
                names(config["action_names"])
            excluded = config.get("exclude_joints", [])
            if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
                raise InputError("exclude_joints must be a list of strings")
        if step_kind(name) in ("normalize", "unnormalize"):
            _features(config.get("features", {}))
            mapping = config.get("norm_map", {})
            if not isinstance(mapping, dict) or any(not isinstance(v, str) for v in mapping.values()):
                raise InputError("Processor norm_map must map feature types to mode strings")
        if entry.get("state_file"):
            state_path = contained(snapshot.root, entry["state_file"])
            step["state_file"] = str(state_path)
            if state_path.exists():
                if state_path.suffix != ".safetensors" or state_path.stat().st_size > MAX_STATS_BYTES:
                    raise InputError("Processor state must be bounded safetensors; model weights are not read")
                # Only normalization states are interpreted, and only numeric stats are retained.
                if step_kind(name) in ("normalize", "unnormalize"):
                    from safetensors import SafetensorError
                    from safetensors.numpy import load_file

                    try:
                        flat = load_file(str(state_path))
                        if sum(x.size for x in flat.values()) > 1_000_000:
                            raise InputError("Processor stats contain too many values")
                        for key, value in flat.items():
                            feature_key, _, stat = key.rpartition(".")
                            if stat in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90"):
                                step["stats"].setdefault(feature_key, {})[stat] = value.tolist()
                            elif stat in ("count", "q50"):
                                # Preserve native metadata without treating a scalar
                                # count or unused median as a decoding dimension.
                                step.setdefault("auxiliary_stats", {}).setdefault(feature_key, {})[stat] = value.tolist()
                        finite_tree(step["stats"])
                        finite_tree(step.get("auxiliary_stats", {}))
                    except (OSError, ValueError, TypeError, SafetensorError) as exc:
                        raise InputError(f"Invalid normalization state {state_path}: {exc}") from exc
                snapshot.record(state_path, f"{role}_state")
            else:
                step["missing_state"] = True
        result.append(step)
    return result


# Native steps whose transform_features leaves state, action and camera features
# unchanged (or only adds language-token features) at LeRobot 146d847.
LEROBOT_REVISION = "146d8474932c8b566717eb2738ba459337ab71c4"
NEUTRAL_STEPS = {
    "tokenizer_processor": ("lerobot.processor.tokenizer_processor", "TokenizerProcessorStep",
                            "src/lerobot/processor/tokenizer_processor.py:57"),
    "smolvla_new_line_processor": ("lerobot.processor.newline_task_processor", "NewLineTaskProcessorStep",
                                   "src/lerobot/processor/newline_task_processor.py:24"),
    "pi0_new_line_processor": ("lerobot.policies.pi0.processor_pi0", "Pi0NewLineProcessor",
                               "src/lerobot/policies/pi0/processor_pi0.py:38"),
    "pi05_prepare_state_tokenizer_processor_step": ("lerobot.policies.pi05.processor_pi05", "Pi05PrepareStateTokenizerProcessorStep",
                                                    "src/lerobot/policies/pi05/processor_pi05.py:42"),
    "pi0_fast_prepare_state_tokenizer_processor_step": ("lerobot.policies.pi0_fast.processor_pi0_fast",
                                                        "Pi0FastPrepareStateAndLanguageTokenizerProcessorStep",
                                                        "src/lerobot/policies/pi0_fast/processor_pi0_fast.py:43"),
    "xvla_add_domain_id": ("lerobot.policies.xvla.processor_xvla", "XVLAAddDomainIdProcessorStep",
                           "src/lerobot/policies/xvla/processor_xvla.py:396"),
    "xvla_image_to_float": ("lerobot.policies.xvla.processor_xvla", "XVLAImageToFloatProcessorStep",
                            "src/lerobot/policies/xvla/processor_xvla.py:245"),
    "xvla_imagenet_normalize": ("lerobot.policies.xvla.processor_xvla", "XVLAImageNetNormalizeProcessorStep",
                                "src/lerobot/policies/xvla/processor_xvla.py:319"),
}
_NEUTRAL_CLASSES = {cls: (module, registry) for registry, (module, cls, _) in NEUTRAL_STEPS.items()}


def native_registry_name(name: str) -> str:
    """Registry name for a verified native step saved under its class path; other names unchanged."""
    if "." in name:
        module, _, cls = name.rpartition(".")
        entry = _NEUTRAL_CLASSES.get(cls)
        if entry and entry[0] == module:
            return entry[1]
    return name


def neutral_source(name: str) -> str | None:
    """Pinned source location establishing a native step as interface-neutral."""
    if "." in name:
        module, _, cls = name.rpartition(".")
        entry = _NEUTRAL_CLASSES.get(cls)
        return NEUTRAL_STEPS[entry[1]][2] if entry and entry[0] == module else None
    return NEUTRAL_STEPS[name][2] if name in NEUTRAL_STEPS else None


def step_kind(name: str) -> str:
    # Exact registry/class names: never guess semantics from a substring.
    if "." in name:
        module, _, cls = name.rpartition(".")
        expected = {
            "NormalizerProcessorStep": "lerobot.processor.normalize_processor",
            "UnnormalizerProcessorStep": "lerobot.processor.normalize_processor",
            "RelativeActionsProcessorStep": "lerobot.processor.relative_action_processor",
            "AbsoluteActionsProcessorStep": "lerobot.processor.relative_action_processor",
            "DeviceProcessorStep": "lerobot.processor.device_processor",
            "AddBatchDimensionProcessorStep": "lerobot.processor.batch_processor",
            "RenameObservationsProcessorStep": "lerobot.processor.rename_processor",
        }
        if expected.get(cls) != module:
            return "neutral" if neutral_source(name) else "unsupported"
        name = cls
    return {
        "normalizer_processor": "normalize", "NormalizerProcessorStep": "normalize",
        "unnormalizer_processor": "unnormalize", "UnnormalizerProcessorStep": "unnormalize",
        "relative_actions_processor": "relative", "RelativeActionsProcessorStep": "relative",
        "absolute_actions_processor": "absolute", "AbsoluteActionsProcessorStep": "absolute",
        "device_processor": "device", "DeviceProcessorStep": "device",
        "to_batch_processor": "batch", "AddBatchDimensionProcessorStep": "batch",
        "rename_observations_processor": "rename", "RenameObservationsProcessorStep": "rename",
    }.get(name, "neutral" if neutral_source(name) else "unsupported")


def load_checkpoint(path: str | Path) -> Snapshot:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.name != "config.json":
            raise InputError("Checkpoint input must be a directory or config.json")
        root = root.parent
    config_path = contained(root, "config.json")
    if root.is_dir() and not config_path.exists():
        if (root / "meta" / "info.json").is_file():
            raise InputError(f"{root} is a LeRobot dataset (it has meta/info.json and no config.json), not a checkpoint. "
                             f"Check it with `policy-doctor preflight {root}` or `inspect --kind dataset`, or pass it "
                             "as --dataset beside a checkpoint.")
        raise InputError(f"No config.json in {root}. A LeRobot checkpoint directory holds config.json and its "
                         "processor files (in a training output, checkpoints/<step>/pretrained_model).")
    config = read_json(config_path)
    if not isinstance(config.get("type"), str):
        raise InputError("LeRobot checkpoint config.json needs a policy type")
    for key in ("chunk_size", "n_action_steps", "n_obs_steps"):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise InputError(f"Checkpoint {key} must be a positive integer")
    if config.get("normalization_mapping") is not None and not isinstance(config["normalization_mapping"], dict):
        raise InputError("normalization_mapping must be an object")
    features = {}
    for key in ("input_features", "output_features"):
        features.update(_features(config.get(key) or {}))
    snapshot = Snapshot("checkpoint", root, config, features)
    snapshot.record(config_path, "checkpoint_config", config)
    snapshot.pre = _pipeline(snapshot, "policy_preprocessor.json", "preprocessor")
    snapshot.post = _pipeline(snapshot, "policy_postprocessor.json", "postprocessor")
    return snapshot


def dataset_root(path: str | Path) -> Path:
    """Resolve accepted dataset aliases before applying a caller's read boundary."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent.parent if root.parent.name == "meta" else root.parent
    return root


def load_dataset(path: str | Path) -> Snapshot:
    root = dataset_root(path)
    info = contained(root, "meta/info.json")
    if root.is_dir() and not info.exists() and (root / "config.json").is_file():
        raise InputError(f"{root} is a policy checkpoint (it has config.json and no meta/info.json), not a dataset. "
                         "Pass it as the checkpoint, and give a LeRobot dataset directory here.")
    config = read_json(info)
    fps = config.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise InputError("Dataset fps must be a positive number")
    snapshot = Snapshot("dataset", root, config, _features(config.get("features"), dataset_names=True))
    snapshot.record(info, "dataset_info", config)
    stats = contained(root, "meta/stats.json")
    if stats.exists():
        snapshot.config = dict(config, stats=read_json(stats))
        snapshot.record(stats, "dataset_stats", snapshot.config["stats"])
    return snapshot


def load_deployment(path: str | Path) -> Snapshot:
    path = Path(path).expanduser().resolve()
    config = read_json(path)
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise InputError("Deployment contract requires schema_version: 1")
    features = _features(config.get("features", {}))
    snapshot = Snapshot("deployment", path.parent, config, features)
    snapshot.record(path, "deployment_contract", config)
    for key in ("fps", "execute_steps"):
        value = config.get(key)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
            raise InputError(f"Deployment {key} must be positive")
    if config.get("execute_steps") is not None and type(config["execute_steps"]) is not int:
        raise InputError("execute_steps must be an integer")
    if config.get("n_obs_steps") is not None and (type(config["n_obs_steps"]) is not int or config["n_obs_steps"] <= 0):
        raise InputError("n_obs_steps must be a positive integer")
    training = config.get("training_contract", {})
    if not isinstance(training, dict):
        raise InputError("training_contract must be an object")
    for key in ("normalization_mapping", "rename_map"):
        if key in config and not isinstance(config[key], dict):
            raise InputError(f"{key} must be an object")
    dim = features.get("action", {}).get("shape", [None])[-1]
    for declarations in (config, training):
        units = declarations.get("units")
        if units is not None and (not isinstance(units, list) or len(units) != dim or
                                  any(not isinstance(x, str) or not x.strip() for x in units)):
            raise InputError("units must contain one nonempty unit name per action dimension")
        for key in ("frame", "gripper_convention"):
            if declarations.get(key) is not None and (not isinstance(declarations[key], str) or not declarations[key].strip()):
                raise InputError(f"{key} must be a nonempty string or null")
        if declarations.get("command_representation") not in (None, "absolute", "relative", "delta"):
            raise InputError("command_representation must be absolute, relative, delta or null")
    if config.get("model_output_representation") not in (None, "absolute", "relative", "delta"):
        raise InputError("model_output_representation must be absolute, relative, delta or null")
    if config.get("limits") is not None:
        import numpy as np

        try:
            limits = np.asarray(config["limits"], dtype=float)
            if limits.shape != (dim, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
                raise ValueError("invalid limits")
        except (TypeError, ValueError) as exc:
            raise InputError("limits must be [action_dim, 2] ordered finite bounds") from exc
    return snapshot
