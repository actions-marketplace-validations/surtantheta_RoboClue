"""Bounded, data-only interpretation of saved GR00T N1.7 processor artifacts.

This is an internal selected-profile resolver, not an effective training-run or
full processor validator. No native module, model, tokenizer or factory is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

from .artifacts import InputError
from .native_intake import _directory_fd

REVISION = "944fe70c4fd54e3fcb80a7d7258a328f9c061176"
PROFILE = "unitree_g1_full_body_with_waist_height_nav_cmd"
NATIVE_SOURCES = {
    "processing_gr00t_n1d7.py": "77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f",
    "state_action_processor.py": "0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145",
    "utils.py": "fa765b240295d9884ab62326c8a21465ae21e6f10133a4b68ac321e8101e4ab7",
    "types.py": "329acd11ca7a92c3894bbe44fe3fed2677b9c9f83bee6adcb4aeab8b1e846a1a",
    "embodiment_configs.py": "4c7626426cfd184034df772bca9f1ab5c2014c139c388f833012ca98ac6f948a",
}
FILE_LIMITS = {
    "processor_config.json": 1024 * 1024,
    "statistics.json": 4 * 1024 * 1024,
    "embodiment_id.json": 64 * 1024,
}
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FINDINGS = 32
STATE_WIDTHS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
}
ACTION_WIDTHS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}
DEFAULTS = {
    "use_percentiles": False,
    "clip_outliers": True,
    "apply_sincos_state_encoding": False,
    "use_relative_action": False,
    "use_mean_std": False,
    "exclude_state": False,
    "state_dropout_prob": 0.0,
    "max_state_dim": 29,
    "max_action_dim": 29,
    "max_action_horizon": 50,
}
_KNOWN_KWARGS = set(DEFAULTS) | {
    "modality_configs",
    "statistics",
    "embodiment_id_mapping",
    "image_crop_size",
    "image_target_size",
    "shortest_image_edge",
    "crop_fraction",
    "random_rotation_angle",
    "color_jitter_params",
    "formalize_language",
    "model_name",
    "model_type",
    "use_albumentations",
    "extra_augmentation_config",
    "transformers_loading_kwargs",
    "letter_box_transform",
}
_UNASSESSED = {
    "selected_sample_comparison": "No selected sample was evaluated.",
    "source_association": "Processor artifacts do not establish recording/export association.",
    "installed_native_source": "Class spelling does not establish the user's installed native revision.",
    "historical_execution": "Caller-selected saved files are not authenticated run history.",
    "full_training_sample": "Images, language, augmentation, mode and default Torch dtype were not evaluated.",
    "model_compatibility": "No model metadata, weights, projector or forward was read or executed.",
    "physical_timing_units": "Numeric artifact declarations do not establish physical timing or units.",
    "full_native_constructor": "Other embodiments and nonnumeric constructor routes were not executed.",
}


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _text(value, *, limit=256):
    return (
        type(value) is str
        and 0 < len(value.encode("utf-8", "surrogatepass")) <= limit
        and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
    )


def _path(value):
    if not isinstance(value, (str, Path)):
        raise InputError("Expected a local path string.")
    value = str(value)
    if not _text(value, limit=4096) or value.startswith("~") or "://" in value:
        raise InputError("Use a literal local path without tilde, URI or control characters.")
    path = Path(value)
    if ".." in path.parts:
        raise InputError("Parent components are not supported; use an intended absolute path.")
    return path.absolute()


def _authorized_root(directory, read_roots):
    root = _path(directory)
    if read_roots is not None:
        if not isinstance(read_roots, (tuple, list)) or not read_roots or len(read_roots) > 32:
            raise InputError("read_roots must be a nonempty bounded list of local paths.")
        roots = [_path(item) for item in read_roots]
        if not any(root == item or root.is_relative_to(item) for item in roots):
            raise InputError("Checkpoint directory is outside authorized read roots.")
    return root


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError("Duplicate JSON key.")
            result[key] = value
        return result

    def bad(_):
        raise InputError("Nonfinite JSON number.")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad)
        queue = [(value, 0)]
        count = 0
        while queue:
            item, depth = queue.pop()
            count += 1
            if depth > 32 or count > 250000:
                raise InputError("JSON nesting or node limit exceeded.")
            if type(item) is dict:
                if len(item) > 4096 or any(not _text(k, limit=256) for k in item):
                    raise InputError("Unsupported JSON object keys.")
                queue.extend((v, depth + 1) for v in item.values())
            elif type(item) is list:
                if len(item) > 16384:
                    raise InputError("JSON collection limit exceeded.")
                queue.extend((v, depth + 1) for v in item)
            elif type(item) is str:
                if len(item.encode("utf-8", "surrogatepass")) > 4096 or any(
                    0xD800 <= ord(c) <= 0xDFFF for c in item
                ):
                    raise InputError("Unsupported JSON string.")
            elif type(item) in (int, float):
                if not math.isfinite(item):
                    raise InputError("Nonfinite JSON number.")
        if type(value) is not dict:
            raise InputError("Expected a native JSON object.")
        return value
    except (UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        raise InputError("Unsupported or malformed bounded JSON.") from exc


def _read(root_fd, name, remaining):
    limit = min(FILE_LIMITS[name], remaining)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise InputError("Artifact must be a bounded regular file.")
        chunks, size = [], 0
        while size <= limit:
            body = os.read(fd, min(65536, limit + 1 - size))
            if not body:
                break
            chunks.append(body)
            size += len(body)
        if size > limit or size != before.st_size or _stamp(before) != _stamp(os.fstat(fd)):
            raise InputError("Artifact size or identity changed during read.")
        raw = b"".join(chunks)
        return (
            _json(raw),
            {"filename": name, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": size},
            _stamp(before),
        )
    finally:
        os.close(fd)


def _finding(code, status, message, pointer=""):
    return {
        "code": code,
        "status": status,
        "message": message,
        "pointer": pointer,
        "next": "Inspect the actual saved native artifacts; preserve the originals and select a supported profile.",
    }


def _status(findings):
    return (
        "ERROR"
        if any(f["status"] == "ERROR" for f in findings)
        else ("UNKNOWN" if any(f["status"] == "UNKNOWN" for f in findings) else "PASS")
    )


def _modality(config):
    if type(config) is not dict or set(config) - {
        "delta_indices",
        "modality_keys",
        "sin_cos_embedding_keys",
        "mean_std_embedding_keys",
        "action_configs",
    }:
        raise InputError("Unsupported native modality declaration.")
    deltas, keys = config.get("delta_indices"), config.get("modality_keys")
    if (
        type(deltas) is not list
        or not 0 < len(deltas) <= 512
        or any(type(d) is not int or abs(d) > 4096 for d in deltas)
        or len(set(deltas)) != len(deltas)
        or type(keys) is not list
        or not 0 < len(keys) <= 64
        or any(not _text(k) for k in keys)
        or len(set(keys)) != len(keys)
    ):
        raise InputError("Unsupported offsets or modality keys.")
    result = {"delta_indices": deltas, "modality_keys": keys}
    for field in ("sin_cos_embedding_keys", "mean_std_embedding_keys"):
        values = config.get(field)
        if values is not None and (
            type(values) is not list
            or len(values) > len(keys)
            or any(v not in keys for v in values)
            or len(set(values)) != len(values)
        ):
            raise InputError("Unsupported per-key embedding declaration.")
        result[field] = values or []
    configs = config.get("action_configs")
    if configs is not None:
        if type(configs) is not list or len(configs) != len(keys):
            raise InputError("Action configurations must match native ordered keys.")
        result["action_configs"] = []
        for action in configs:
            if type(action) is not dict or set(action) - {"rep", "type", "format", "state_key"}:
                raise InputError("Unsupported action configuration.")
            if (
                action.get("rep") not in ("ABSOLUTE", "RELATIVE", "DELTA")
                or action.get("type") not in ("NON_EEF", "EEF")
                or action.get("format") not in ("DEFAULT", "XYZ_ROT6D", "XYZ_ROTVEC")
                or (action.get("state_key") is not None and not _text(action["state_key"]))
            ):
                raise InputError("Native JSON action enums require supported uppercase member names.")
            result["action_configs"].append({**action, "state_key": action.get("state_key")})
    else:
        result["action_configs"] = None
    return result


def _selected_profile(config):
    if type(config) is not dict or set(config) != {"video", "state", "action", "language"}:
        raise InputError("Only the selected pinned G1 modality structure is supported.")
    result = {key: _modality(value) for key, value in config.items()}
    expected = {
        "state": list(STATE_WIDTHS),
        "action": list(ACTION_WIDTHS),
        "video": ["ego_view"],
        "language": ["annotation.human.task_description"],
    }
    for modality, keys in expected.items():
        if result[modality]["modality_keys"] != keys:
            raise InputError("Saved modality order does not match the qualified G1 profile.")
        offsets = list(range(50)) if modality == "action" else [0]
        if result[modality]["delta_indices"] != offsets:
            raise InputError("Saved offsets do not match the qualified G1 profile.")
    actions = result["action"]["action_configs"]
    if actions is None:
        raise InputError("The qualified G1 action declarations are required.")
    for key, action in zip(ACTION_WIDTHS, actions, strict=True):
        rep = "RELATIVE" if key in ("left_arm", "right_arm") else "ABSOLUTE"
        if (
            action["rep"] != rep
            or action["type"] != "NON_EEF"
            or action["format"] != "DEFAULT"
            or action["state_key"] not in (None, key)
        ):
            raise InputError("Unsupported change to the qualified G1 action representation.")
    return result


def _array(value, width, *, relative=False):
    if type(value) is not list or not value:
        raise InputError("Statistics require finite, nonempty numeric arrays.")
    shape = [len(value)]
    rows = [value]
    if type(value[0]) is list:
        if not relative or len(value) != 50:
            raise InputError("Only the supported relative horizon statistics may be two-dimensional.")
        shape = [50, width]
        rows = value
    if any(type(row) is not list or len(row) != width for row in rows):
        raise InputError("Saved statistic dimensions do not match the qualified field width.")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in rows for v in row):
        raise InputError("Statistics require finite nonboolean numeric values.")
    return shape


def _parameters(group, width, bounds, *, relative=False):
    if type(group) is not dict:
        raise InputError("Saved statistics group is missing or malformed.")
    shapes = {}
    required = {*bounds, "mean", "std"}
    for name in required:
        shapes[name] = _array(group.get(name), width, relative=relative)
    if len({tuple(s) for s in shapes.values()}) != 1:
        raise InputError("Saved statistic shapes differ within a field.")

    def flat(value):
        return [v for row in value for v in row] if type(value[0]) is list else value

    if any(v < 0 for v in flat(group["std"])):
        raise InputError("Negative saved standard deviation is not a supported numeric context.")
    if any(lo > hi for lo, hi in zip(flat(group[bounds[0]]), flat(group[bounds[1]]), strict=True)):
        raise InputError("Reversed saved bounds are not a supported numeric context.")
    return shapes


def _interpret(config, statistics, embodiment, findings):
    kwargs = config.get("processor_kwargs")
    if config.get("processor_class") != "Gr00tN1d7Processor" or type(kwargs) is not dict:
        raise InputError("Expected an ordinary Gr00tN1d7Processor saved configuration.")
    if set(kwargs) - _KNOWN_KWARGS:
        raise InputError("Unsupported processor keyword names; values were not copied into the report.")
    flags, defaulted = {}, []
    for name, default in DEFAULTS.items():
        value = kwargs.get(name, default)
        if type(default) is bool:
            good = type(value) is bool
        elif type(default) is int:
            good = type(value) is int and 1 <= value <= 4096
        else:
            good = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        if not good:
            raise InputError("Unsupported typed numeric processor setting.")
        flags[name] = value
        if name not in kwargs:
            defaulted.append("/processor_kwargs/" + name)
    configs = kwargs.get("modality_configs")
    if type(configs) is not dict or not configs or len(configs) > 64:
        raise InputError("A bounded native modality_configs object is required.")
    if embodiment is None:
        if len(configs) != 1:
            raise InputError("Select an explicit native embodiment when multiple entries are present.")
        embodiment = next(iter(configs))
    if embodiment != PROFILE or embodiment not in configs:
        raise InputError("The selected embodiment is outside the supported pinned G1 profile.")
    # This native guard applies to all configured embodiments, independently of
    # whether the selected profile's statistics can be interpreted.
    for entry in configs.values():
        if type(entry) is dict and type(entry.get("action")) is dict:
            offsets = entry["action"].get("delta_indices")
            if type(offsets) is list and len(offsets) > flags["max_action_horizon"]:
                findings.append(
                    _finding(
                        "consumer.action_horizon_capacity",
                        "ERROR",
                        "A saved action offset count exceeds max_action_horizon; the pinned native constructor refuses this.",
                        "/processor_kwargs/max_action_horizon",
                    )
                )
                break
    modalities = _selected_profile(configs[embodiment])
    if type(statistics.get(embodiment)) is not dict:
        raise InputError("Selected embodiment statistics are absent from statistics.json.")
    stats = statistics[embodiment]
    fields, parameters = [], {"state": {}, "action": {}}
    incomplete = False
    for modality, widths in (("state", STATE_WIDTHS), ("action", ACTION_WIDTHS)):
        for index, (key, width) in enumerate(widths.items()):
            try:
                declaration = modalities[modality]
                relative = (
                    modality == "action"
                    and flags["use_relative_action"]
                    and declaration["action_configs"][index]["rep"] == "RELATIVE"
                )
                source = "relative_action" if relative else modality
                absolute = stats.get(modality, {}).get(key) if type(stats.get(modality)) is dict else None
                bounds = ("q01", "q99") if flags["use_percentiles"] else ("min", "max")
                _parameters(absolute, width, bounds)
                group = absolute
                if relative:
                    group = stats.get(source, {}).get(key) if type(stats.get(source)) is dict else None
                    if group is None:
                        findings.append(
                            _finding(
                                "consumer.relative_statistics_missing",
                                "ERROR",
                                "Selected relative action "
                                + key
                                + " lacks relative_action statistics; the pinned constructor requires them.",
                                "/" + embodiment + "/relative_action/" + key,
                            )
                        )
                    bounds = ("min", "max")
                shapes = _parameters(group, width, bounds, relative=relative)
                sincos = (
                    modality == "state"
                    and flags["apply_sincos_state_encoding"]
                    and key in declaration["sin_cos_embedding_keys"]
                )
                meanstd = key in declaration["mean_std_embedding_keys"]
                operation = "sincos" if sincos else ("meanstd" if meanstd else "minmax")
                names = [] if sincos else (["mean", "std"] if meanstd else list(bounds))
                active = (
                    {"mean": group["mean"], "std": group["std"]}
                    if meanstd and not sincos
                    else ({"min": group[bounds[0]], "max": group[bounds[1]]} if not sincos else {})
                )
                pointer = "/" + embodiment + "/" + source + "/" + key
                field = {
                    "modality": modality,
                    "key": key,
                    "input_dim": width,
                    "output_dim": width * (2 if sincos else 1),
                    "operation": operation,
                    "relative": relative,
                    "state_reference": key if relative else None,
                    "clip": flags["clip_outliers"] and (modality == "action" or operation == "minmax"),
                    "statistics_pointer": pointer,
                    "parameter_names": names,
                    "parameter_shapes": {name: shapes[name] for name in names},
                    "active_parameter_sha256": _hash(active),
                }
                fields.append(field)
                parameters[modality][key] = active
            except InputError:
                incomplete = True
                findings.append(
                    _finding(
                        "consumer.statistics_uninterpretable",
                        "UNKNOWN",
                        "Saved "
                        + modality
                        + "."
                        + key
                        + " statistics cannot establish the supported finite numeric context.",
                        "/" + embodiment + "/" + modality + "/" + key,
                    )
                )
    widths = {m: sum(f["output_dim"] for f in fields if f["modality"] == m) for m in ("state", "action")}
    declared_widths = {
        "state": sum(
            width
            * (
                2
                if flags["apply_sincos_state_encoding"]
                and key in modalities["state"]["sin_cos_embedding_keys"]
                else 1
            )
            for key, width in STATE_WIDTHS.items()
        ),
        "action": sum(ACTION_WIDTHS.values()),
    }
    state_lower = (
        sum(STATE_WIDTHS.values())
        if flags["exclude_state"] or flags["state_dropout_prob"] > 0
        else declared_widths["state"]
    )
    state_upper = state_lower if flags["exclude_state"] else declared_widths["state"]
    for modality in ("state", "action"):
        required = state_lower if modality == "state" else declared_widths[modality]
        if required > flags["max_" + modality + "_dim"]:
            findings.append(
                _finding(
                    "consumer." + modality + "_capacity",
                    "ERROR",
                    "Selected "
                    + modality
                    + " width at least "
                    + str(required)
                    + " exceeds the saved processor capacity "
                    + str(flags["max_" + modality + "_dim"])
                    + "; the pinned padding route cannot hold these coordinates.",
                    "/processor_kwargs/max_" + modality + "_dim",
                )
            )
    if state_lower <= flags["max_state_dim"] < state_upper:
        findings.append(
            _finding(
                "consumer.state_capacity_mode_dependent",
                "UNKNOWN",
                "State padding capacity depends on unrecorded training mode and stochastic dropout; encoded and dropped widths differ.",
                "/processor_kwargs/max_state_dim",
            )
        )
    active_fields = [
        {
            k: v
            for k, v in field.items()
            if k not in ("statistics_pointer", "parameter_shapes", "parameter_names")
        }
        for field in fields
    ]
    interpretation = {
        "schema_version": 1,
        "native_revision": REVISION,
        "profile": PROFILE,
        "modalities": {
            m: {"keys": d["modality_keys"], "offsets": d["delta_indices"]} for m, d in modalities.items()
        },
        "fields": active_fields,
        "wrapper_boundary": {
            k: flags[k]
            for k in (
                "exclude_state",
                "state_dropout_prob",
                "max_state_dim",
                "max_action_dim",
                "max_action_horizon",
            )
        },
        "normalizer_policy": "numpy-isclose-minmax; zero-std-raw-passthrough; input-dtype-preserved",
    }
    descriptor = {
        "schema_version": 1,
        "native_revision": REVISION,
        "profile": PROFILE,
        "selected_embodiment": embodiment,
        "flags": flags,
        "defaulted_pointers": defaulted,
        "default_source": "processing_gr00t_n1d7.py:Gr00tN1d7Processor.__init__/from_pretrained",
        "modalities": modalities,
        "fields": fields,
        "widths": widths,
        "action_offset_count": 50,
        "action_offset_span": 50,
        "wrapper_state_width_range": [state_lower, state_upper],
        "configured_embodiment_count": len(configs),
        "interpretation_sha256": _hash(interpretation) if not incomplete else None,
        "inert_serialized_flags": ["use_mean_std"],
        "numeric_context_complete": not incomplete,
    }
    return descriptor, parameters


def _resolve(checkpoint_directory, embodiment, read_roots):
    root = _authorized_root(checkpoint_directory, read_roots)
    if embodiment is not None and not _text(embodiment):
        raise InputError("embodiment must be a bounded native tag string.")
    findings, artifacts, docs, stamps = [], {}, {}, {}
    descriptor, parameters = None, None
    fd = None
    intake_ok = True
    try:
        fd = _directory_fd(root)
        root_stamp = _stamp(os.fstat(fd))
        remaining = MAX_TOTAL_BYTES
        for name in FILE_LIMITS:
            try:
                doc, artifact, stamp = _read(fd, name, remaining)
            except FileNotFoundError:
                if name == "embodiment_id.json":
                    stamps[name] = None
                    continue
                raise InputError("A required native processor artifact is absent.") from None
            docs[name], artifacts[name], stamps[name] = doc, artifact, stamp
            remaining -= artifact["size_bytes"]
        mapping = docs.get("embodiment_id.json")
        if mapping is not None and (
            len(mapping) > 256
            or any(not _text(k) or type(v) is not int or not 0 <= v <= 65535 for k, v in mapping.items())
        ):
            raise InputError("Unsupported embodiment ID mapping.")
        descriptor, parameters = _interpret(
            docs["processor_config.json"], docs["statistics.json"], embodiment, findings
        )
        # Check pathname membership as well as the retained descriptors: replacing
        # a file/root between reads must not leave a complete context claim.
        for name, expected in stamps.items():
            try:
                actual = _stamp(os.stat(name, dir_fd=fd, follow_symlinks=False))
            except FileNotFoundError:
                actual = None
            if actual != expected:
                raise InputError("Native artifact membership changed during intake.")
        current = _directory_fd(root)
        try:
            if _stamp(os.fstat(current)) != root_stamp:
                raise InputError("Checkpoint directory identity changed during intake.")
        finally:
            os.close(current)
    except (InputError, OSError, ValueError, TypeError, OverflowError, RecursionError):
        intake_ok = False
        descriptor, parameters = None, None
        findings.append(
            _finding(
                "consumer.artifact_intake_incomplete",
                "UNKNOWN",
                "The bounded authorized native artifacts could not establish a supported context; no native code was executed.",
            )
        )
    finally:
        if fd is not None:
            os.close(fd)
    numeric_complete = descriptor is not None and descriptor["numeric_context_complete"]
    axes = {key: {"status": "UNKNOWN", "reason": reason} for key, reason in _UNASSESSED.items()}
    axes["artifact_intake"] = {"status": "PASS" if intake_ok else "UNKNOWN"}
    axes["processor_numeric_context"] = {
        "status": "PASS" if numeric_complete else "UNKNOWN",
        "scope": "selected saved G1 numeric interpretation under the named pinned source profile",
    }
    capacity_findings = [f for f in findings if "capacity" in f["code"]]
    axes["declared_processor_capacity"] = {
        "status": _status(capacity_findings) if capacity_findings or descriptor is not None else "UNKNOWN"
    }
    # Severity retention is independent of the total bounded payload.
    ordered = sorted(
        enumerate(findings), key=lambda p: ({"ERROR": 0, "UNKNOWN": 1}.get(p[1]["status"], 2), p[0])
    )
    report = {
        "kind": "native_consumer_context",
        "schema_version": 1,
        "status": _status(findings),
        "requested_scope": "saved_processor_numeric_context",
        "checkpoint_directory": str(root),
        "requested_embodiment": embodiment,
        "descriptor": descriptor,
        "axes": axes,
        "artifacts": artifacts,
        "findings": [f for _, f in ordered[:MAX_FINDINGS]],
        "finding_count": len(findings),
        "findings_omitted": max(0, len(findings) - MAX_FINDINGS),
        "coverage": {
            "file_count": len(artifacts),
            "body_bytes": sum(a["size_bytes"] for a in artifacts.values()),
            "numeric_fields": len(descriptor["fields"]) if descriptor else 0,
            "native_code_executed": False,
            "selected_samples_evaluated": 0,
        },
        "evaluator": {
            "native_source_sha256": NATIVE_SOURCES,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "assurance": "Caller-selected artifact bytes; no authenticated historical run or installed native-source claim.",
    }
    return report, parameters


def resolve_native_processor_context(checkpoint_directory, *, embodiment=None, read_roots=None):
    """Read only the three fixed native JSON roles; no weights or network.

    Unauthorized or ambiguous caller paths raise InputError before filesystem
    probes. Missing, malformed, unsupported or unstable artifacts return UNKNOWN;
    independently established capacity/required-relative-stat contradictions remain ERROR.
    """
    return _resolve(checkpoint_directory, embodiment, read_roots)[0]


def load_consumer_numeric_parameters(context, *, read_roots=None):
    """Internal reread for a selected-sample worker, never a caller-authored map.

    Reauthorize the recorded root, reread the same bounded native roles and require
    exact file hashes and interpretation. Returned arrays are saved parameters,
    not source-membership evidence. The public context remains digest/pointer-only.
    """
    if (
        type(context) is not dict
        or context.get("kind") != "native_consumer_context"
        or type(context.get("schema_version")) is not int
        or context.get("schema_version") != 1
        or "descriptor" not in context
        or (context.get("descriptor") is not None and type(context.get("descriptor")) is not dict)
    ):
        raise InputError("A generated consumer context is required.")
    descriptor = context["descriptor"]
    try:
        expected_descriptor_hash = _hash(descriptor)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise InputError("Unsupported consumer context descriptor.") from exc
    embodiment = context.get("requested_embodiment")
    if embodiment is None and descriptor is not None:
        embodiment = descriptor.get("selected_embodiment")
    report, parameters = _resolve(context.get("checkpoint_directory"), embodiment, read_roots)
    if (
        report["artifacts"] != context.get("artifacts")
        or _hash(report["descriptor"]) != expected_descriptor_hash
        or report["evaluator"] != context.get("evaluator")
    ):
        raise InputError("Consumer artifact identity or interpretation changed; resolve a fresh context.")
    parameters = parameters or {"state": {}, "action": {}}
    return {
        "descriptor": report["descriptor"],
        "parameters": parameters,
        "fields": [
            dict(field, parameters=parameters[field["modality"]][field["key"]])
            for field in (report["descriptor"]["fields"] if report["descriptor"] else [])
        ],
        "interpretation_sha256": report["descriptor"]["interpretation_sha256"]
        if report["descriptor"]
        else None,
        "context_status": report["status"],
        "context_findings": report["findings"],
        "context_axes": report["axes"],
        "authority": "Revalidated saved artifact parameters; selected raw sample origin is not established.",
    }
