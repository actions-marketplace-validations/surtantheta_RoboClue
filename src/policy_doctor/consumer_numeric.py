"""Bounded numeric projection for the pinned G1 inner StateActionProcessor.

Inputs are internal selected-sample arrays and revalidated saved-artifact context,
not a user-written normalization contract. No native package is imported. The
equations follow GR00T 944fe70's state_action_processor.py, utils.py and joint
pose conversion; wrapper padding/default Torch dtype and native parity remain
separate boundaries. Reports contain hashes and bounded witnesses, never arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import InputError

REVISION = "944fe70c4fd54e3fcb80a7d7258a328f9c061176"
PROFILE = "unitree_g1_full_body_with_waist_height_nav_cmd"
MAX_ELEMENTS = 64_000
MAX_FIELDS = 14
MAX_WITNESSES = 16
STAGES = ("raw", "representation", "normalized", "clipped")
WIDTHS = {
    "state": {"left_leg": 6, "right_leg": 6, "waist": 3, "left_arm": 7,
              "right_arm": 7, "left_hand": 7, "right_hand": 7},
    "action": {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7,
               "waist": 3, "base_height_command": 1, "navigate_command": 3},
}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_hash(value):
    return _hash(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _summary(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": _hash(array.tobytes(order="C"))}


def _mask_summary(mask):
    return {"shape": list(mask.shape), "true_coordinates": int(np.count_nonzero(mask)),
            "sha256": _hash(mask.tobytes(order="C"))}


def _finding(code, status, message, field=None):
    result = {"code": code, "status": status, "message": message,
              "next": "Inspect the saved native context and selected raw sample; preserve all originals."}
    if field is not None:
        result["field"] = field
    return result


def _status(findings):
    return "ERROR" if any(f["status"] == "ERROR" for f in findings) else (
        "UNKNOWN" if any(f["status"] == "UNKNOWN" for f in findings) else "PASS")


def _selection(sample):
    """Keep a bounded JSON locator, without promoting it to source authority."""
    value = sample.get("selection", {})
    try:
        encoded = json.dumps(value, allow_nan=False)
        if len(encoded) > 16_384 or type(value) is not dict:
            return {}
        # This also rejects non-JSON objects before any array conversion hook.
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError):
        return {}


def _array(value, width, rows):
    # Reject object arrays, subclasses/memmaps and __array__ hooks before copying.
    if type(value) is not np.ndarray or value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Selected raw fields must be ordinary float32/float64 arrays.")
    if value.shape != (rows, width) or value.size > MAX_ELEMENTS:
        raise ValueError("Selected raw field dimensions differ from the pinned modality offsets.")
    before = _summary(value)
    copied = value.copy(order="C")
    if before != _summary(value) or before != _summary(copied):
        raise ValueError("Selected raw array changed while copied.")
    return copied


def _parameters(field, supplied, values):
    operation = field["operation"]
    names = field.get("parameter_names")
    if operation == "minmax":
        if names not in (["min", "max"], ["q01", "q99"]):
            raise ValueError("Unsupported minmax parameter selection.")
        # The context loader maps the source-selected q01/q99 pair onto the
        # native cache's min/max names, while keeping source pointers public.
        names = ["min", "max"]
    elif operation == "meanstd":
        if names != ["mean", "std"]:
            raise ValueError("Unsupported mean/std parameter selection.")
    else:
        raise ValueError("This numeric operation has no qualified implementation.")
    if type(supplied) is not dict or set(supplied) != set(names):
        raise ValueError("Active parameters are incomplete.")
    arrays = []
    for name in names:
        # The helper rereads strict JSON lists; preserve native inferred dtype.
        item = supplied[name]
        if type(item) is not list:
            raise ValueError("Active parameters must come from saved JSON arrays.")
        array = np.array(item)
        if (array.dtype.kind not in "if" or array.dtype.itemsize > 8
                or array.shape not in ((values.shape[1],), values.shape)
                or array.size > MAX_ELEMENTS or not np.isfinite(array).all()):
            raise ValueError("Unsupported active parameter type, extent or nonfinite value.")
        arrays.append(array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("Active parameter dimensions differ.")
    return arrays


def _normalize(values, left, right, operation):
    # Retain both native assignment casts: quotient -> zeros_like input, then
    # affine map -> that same input dtype. No invented range/epsilon floor.
    active = ~np.isclose(right, left) if operation == "minmax" else right != 0
    result = np.zeros_like(values)
    with np.errstate(all="ignore"):
        if operation == "minmax":
            result[..., active] = (values[..., active] - left[active]) / (right[active] - left[active])
            result[..., active] = 2 * result[..., active] - 1
        else:
            result[..., active] = (values[..., active] - left[active]) / right[active]
            result[..., ~active] = values[..., ~active]
    return result, np.broadcast_to(active, values.shape).copy()


def _load(context, read_roots):
    from .consumer_context import load_consumer_numeric_parameters

    return load_consumer_numeric_parameters(context, read_roots=read_roots)


def _trace(context, sample, read_roots):
    findings = []
    report = {"schema_version": 1, "kind": "consumer_numeric_trace", "status": "UNKNOWN",
              "scope": "Pure projection of the selected pinned inner SAP numeric route; not a native execution.",
              "fields": [], "findings": findings, "selection": {},
              "evaluator": {"implementation_sha256": _hash(Path(__file__).read_bytes()),
                            "numpy_version": np.__version__, "native_revision": REVISION},
              "boundaries": {"raw_sample_origin": "UNKNOWN", "native_parity": "UNKNOWN",
                             "wrapper_padding_masks_default_torch_dtype": "UNKNOWN",
                             "images_language_augmentation": "UNKNOWN", "full_training_model_physical": "UNKNOWN"}}
    private = {}
    try:
        loaded = _load(context, read_roots)
        descriptor = loaded["descriptor"]
        # Severity comes from the authorized reread, not editable saved-report
        # status/findings. It survives incomplete numeric context independently.
        fresh_findings = loaded["context_findings"]
        if type(fresh_findings) is not list:
            raise ValueError("Fresh saved-context findings are required.")
        findings.extend(dict(f) for f in fresh_findings[:32]
                        if type(f) is dict and f.get("status") == "ERROR")
        report["context_status"] = loaded["context_status"]
        if loaded["context_status"] == "ERROR" and not findings:
            findings.append(_finding("consumer.context_contradiction", "ERROR",
                "The revalidated saved native context contains an independent contradiction."))
        report["artifact_identity_sha256"] = _json_hash({
            "checkpoint_directory": context.get("checkpoint_directory"),
            "artifacts": context.get("artifacts", {}),
        })
        if type(descriptor) is not dict:
            raise ValueError("The revalidated saved context has no supported numeric descriptor.")
        report["interpretation_sha256"] = descriptor.get("interpretation_sha256")
        if (descriptor.get("native_revision") != REVISION or descriptor.get("profile") != PROFILE
                or descriptor.get("numeric_context_complete") is not True):
            raise ValueError("A complete pinned G1 numeric context is required.")
        if type(sample) is not dict or set(sample) - {"states", "actions", "selection"}:
            raise ValueError("An internal states/actions selected sample is required.")
        fields = descriptor["fields"]
        if type(fields) is not list or len(fields) != MAX_FIELDS:
            raise ValueError("The complete pinned G1 field set is required.")
        expected = {(m, k) for m, keys in WIDTHS.items() for k in keys}
        if {(f["modality"], f["key"]) for f in fields} != expected:
            raise ValueError("Selected field membership differs from the pinned G1 profile.")
        report["selection"] = _selection(sample)
        snapshot = {}
        total = 0
        for field in fields:
            modality, key = field["modality"], field["key"]
            rows = len(descriptor["modalities"][modality]["delta_indices"])
            group = sample.get("states" if modality == "state" else "actions")
            if type(group) is not dict or key not in group:
                findings.append(_finding("consumer.sample_missing_field", "ERROR",
                    "The selected raw sample lacks a required native field.", modality + "." + key))
                continue
            try:
                snapshot[(modality, key)] = _array(group[key], WIDTHS[modality][key], rows)
                total += snapshot[(modality, key)].size
                if total > MAX_ELEMENTS:
                    raise ValueError("Selected sample aggregate numeric extent exceeds the bound.")
            except ValueError as error:
                findings.append(_finding("consumer.sample_uninterpretable", "UNKNOWN", str(error), modality + "." + key))
        for field in fields:
            modality, key = field["modality"], field["key"]
            label = modality + "." + key
            if (modality, key) not in snapshot:
                continue
            raw = snapshot[(modality, key)]
            if not np.isfinite(raw).all():
                findings.append(_finding("consumer.sample_nonfinite", "ERROR",
                    "The selected raw numeric field contains a nonfinite value.", label))
                continue
            try:
                represented = raw.copy()
                if field["relative"]:
                    if modality != "action" or key not in {"left_arm", "right_arm"}:
                        raise ValueError("Only pinned NON_EEF DEFAULT arm-relative conversion is supported.")
                    reference = snapshot.get(("state", field["state_reference"]))
                    if reference is None or not np.isfinite(reference).all():
                        raise ValueError("A finite raw selected state reference is required.")
                    represented = raw.astype(np.float64) - reference[-1].astype(np.float64)
                parameters = _parameters(field, loaded["parameters"][modality][key], represented)
                normalized, active = _normalize(represented, *parameters, field["operation"])
                if type(field["clip"]) is not bool:
                    raise ValueError("Clipping must be a resolved native boolean.")
                clipped = np.clip(normalized, -1.0, 1.0) if field["clip"] else normalized.copy()
                if not np.isfinite(normalized).all() or not np.isfinite(clipped).all():
                    findings.append(_finding("consumer.projected_nonfinite", "ERROR",
                        "The supported numeric projection produced a nonfinite result.", label))
                    continue
                stages = dict(zip(STAGES, (raw, represented, normalized, clipped), strict=True))
                private[label] = stages
                report["fields"].append({"field": label, "operation": field["operation"],
                    "relative": field["relative"], "clip": field["clip"],
                    "active_parameter_sha256": field["active_parameter_sha256"],
                    "stages": {name: _summary(value) for name, value in stages.items()},
                    "active_normalization_coordinates": _mask_summary(active),
                    "inactive_normalization_coordinates": _mask_summary(~active),
                    "out_of_range_before_clipping": _mask_summary((normalized < -1) | (normalized > 1)),
                    "clipping_changed_coordinates": _mask_summary(normalized != clipped)})
            except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
                findings.append(_finding("consumer.operation_unassessed", "UNKNOWN", str(error)[:256], label))
        report["expected_fields"] = MAX_FIELDS
        report["computed_fields"] = len(private)
        if len(private) != MAX_FIELDS and not any(f["status"] == "UNKNOWN" for f in findings):
            findings.append(_finding("consumer.numeric_coverage_incomplete", "UNKNOWN",
                "Not all selected numeric fields were projected."))
    except (InputError, ValueError, KeyError, TypeError, OverflowError) as error:
        findings.append(_finding("consumer.numeric_context_unavailable", "UNKNOWN", str(error)[:256]))
    report["status"] = _status(findings)
    report["finding_count"] = len(findings)
    report["findings"] = sorted(findings, key=lambda f: f["status"] != "ERROR")[:32]
    report["findings_omitted"] = max(0, len(findings) - 32)
    return report, private


def trace_consumer_numeric(context: dict, raw_sample: dict, *, read_roots=None) -> dict[str, Any]:
    """Project one internally selected sample after exact artifact revalidation.

    The context loader reauthorizes all fixed artifact paths. Raw arrays are
    caller-held evidence with no inferred source membership or native execution.
    """
    return _trace(context, raw_sample, read_roots)[0]


def _locator(report, field, row):
    selection = report.get("selection", {})
    indices = selection.get("state_indices" if field.startswith("state.") else "action_indices")
    return {"sample_row": row, "caller_selected_index": indices[row]
            if type(indices) is list and row < len(indices) and type(indices[row]) is int else None,
            "source_membership": "UNKNOWN"}


def compare_consumer_numeric(reference_context: dict, reference_sample: dict,
                             candidate_context: dict, candidate_sample: dict, *, read_roots=None) -> dict[str, Any]:
    """Compare supported inner numeric projections without equating drift with error.

    Value equality is exact with no fitted tolerance. Dtype/byte/identity changes
    remain distinct; incomplete stages never establish equivalence.
    """
    before, left = _trace(reference_context, reference_sample, read_roots)
    after, right = _trace(candidate_context, candidate_sample, read_roots)
    complete = len(left) == len(right) == MAX_FIELDS and before["status"] != "UNKNOWN" and after["status"] != "UNKNOWN"
    first = None
    first_value = None
    differences = []
    final_fields = []
    parameter_changes = []
    candidate_fields = {f["field"]: f for f in after["fields"]}
    for field in before["fields"]:
        other = candidate_fields.get(field["field"])
        if other and field["active_parameter_sha256"] != other["active_parameter_sha256"]:
            parameter_changes.append({"field": field["field"],
                                      "reference_sha256": field["active_parameter_sha256"],
                                      "candidate_sha256": other["active_parameter_sha256"]})
    counts = {"value_coordinates": 0, "dtype_fields": 0, "shape_fields": 0, "byte_fields": 0}
    for stage in STAGES:
        for field in left:
            if field not in right:
                continue
            a, b = left[field][stage], right[field][stage]
            shape_changed = a.shape != b.shape
            dtype_changed = a.dtype != b.dtype
            byte_changed = shape_changed or dtype_changed or a.tobytes(order="C") != b.tobytes(order="C")
            changed = np.not_equal(a, b) if not shape_changed else None
            value_count = int(np.count_nonzero(changed)) if changed is not None else None
            if first is None and byte_changed:
                coordinate = np.argwhere(changed)[0].tolist() if value_count else None
                first = {"stage": stage, "field": field, "coordinate": coordinate,
                         "change": "shape" if shape_changed else ("value" if value_count else
                                    "dtype" if dtype_changed else "bytes"),
                         "before": _summary(a), "after": _summary(b),
                         "evidence": "Checker-computed projection; not an observed native execution."}
                if coordinate is not None:
                    first["row"] = _locator(before, field, coordinate[0])
                    first["reference_row"] = _locator(before, field, coordinate[0])
                    first["candidate_row"] = _locator(after, field, coordinate[0])
            if first_value is None and value_count:
                coordinate = np.argwhere(changed)[0].tolist()
                first_value = {"stage": stage, "field": field, "coordinate": coordinate,
                               "row": _locator(before, field, coordinate[0]),
                               "reference_row": _locator(before, field, coordinate[0]),
                               "candidate_row": _locator(after, field, coordinate[0]),
                               "evidence": "Checker-computed projection; native parity remains unassessed."}
            if stage == "clipped":
                counts["value_coordinates"] += value_count or 0
                counts["dtype_fields"] += int(dtype_changed)
                counts["shape_fields"] += int(shape_changed)
                counts["byte_fields"] += int(byte_changed)
                final_fields.append({"field": field, "shape_changed": shape_changed,
                    "dtype_changed": dtype_changed, "bytes_changed": byte_changed,
                    "value_coordinates": value_count,
                    "value_change_mask": _mask_summary(changed) if changed is not None else None})
                if changed is not None:
                    for coordinate in np.argwhere(changed)[:MAX_WITNESSES - len(differences)]:
                        index = tuple(coordinate)
                        differences.append({"field": field, "coordinate": coordinate.tolist(),
                            "before": float(a[index]), "after": float(b[index]),
                            "row": _locator(before, field, int(coordinate[0])),
                            "reference_row": _locator(before, field, int(coordinate[0])),
                            "candidate_row": _locator(after, field, int(coordinate[0]))})
    identity_changed = (before["artifact_identity_sha256"] != after["artifact_identity_sha256"]
                        if before.get("artifact_identity_sha256") and after.get("artifact_identity_sha256") else None)
    interpretation_changed = (before["interpretation_sha256"] != after["interpretation_sha256"]
                              if before.get("interpretation_sha256") and after.get("interpretation_sha256") else None)
    value_effect = ("CHANGED" if counts["value_coordinates"] else
                    "EQUAL" if complete and not counts["shape_fields"] else "UNKNOWN")
    return {"schema_version": 1, "kind": "consumer_numeric_comparison",
            "status": "ERROR" if "ERROR" in (before["status"], after["status"]) else ("PASS" if complete else "UNKNOWN"),
            "scope": "Selected supported inner SAP numeric projection only; a changed recipe is not itself wrong.",
            "reference": before, "candidate": after,
            "artifact_identity_changed": identity_changed, "active_interpretation_changed": interpretation_changed,
            "selection_changed": before["selection"] != after["selection"]
                                 if before["selection"] and after["selection"] else None,
            "first_observed_divergence": first, "numeric_projection_value_effect": value_effect,
            "first_value_divergence": first_value,
            "active_parameter_changes": parameter_changes,
            "final_inner_numeric_counts": counts, "first_value_differences": differences,
            "final_inner_fields": final_fields,
            "counting_scope": "Computed field coordinate occurrences; not unique original recording rows.",
            "first_value_differences_omitted": max(0, counts["value_coordinates"] - len(differences)),
            "complete_field_coverage": complete,
            "final_wrapper_model_target_effect": "UNKNOWN", "native_parity": "UNKNOWN",
            "assurance": "Saved-artifact identity and caller-supplied numeric evidence do not authenticate a run or sample origin."}
