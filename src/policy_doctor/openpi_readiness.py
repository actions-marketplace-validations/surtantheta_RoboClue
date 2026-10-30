"""Conditional OpenPI source-profile comparison against original selected statistics.

This pure proposal performs no transport, native imports, policy execution or mutation.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from .artifacts import InputError
from .openpi_intake import OpenPIStatisticsSnapshot
from .openpi_interface import resolve_serving_command

UNKNOWN_BOUNDARIES = {
    "actual_command_directory": "unbound by this pure engine; source_root is a declared profile, not proof of the pasted command's cwd",
    "actual_runtime": "user-installed source/dependencies and executing policy are uninspected",
    "checkpoint_weights_and_model_architecture": "not loaded or verified",
    "historical_training_mode": "not authenticated by current config selection or statistics shape",
    "physical_units_and_robot_suitability": "not established by these files",
    "final_native_DroidOutputs_composition": "unproved; the separate native import was dependency blocked, so the final8 slice remains source-derived",
}


def _unknown(message):
    return {"status": "unknown", "message": message}


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("ambiguous JSON")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _binding(snapshot, selected):
    if selected.get("status") != "resolved_supported_builtin":
        return _unknown("The command has no supported source-profile selection")
    if not isinstance(snapshot, OpenPIStatisticsSnapshot) or not isinstance(snapshot.document, dict) or not isinstance(snapshot.raw, bytes):
        return _unknown("Complete original statistics bytes and parsed JSON are unavailable")
    raw, provenance = snapshot.raw, snapshot.provenance
    if not 0 < len(raw) <= 1024 * 1024 or not isinstance(provenance, dict):
        return _unknown("Statistics representation is outside the bounded supported intake")
    try:
        if _parse(raw) != snapshot.document:
            return _unknown("Original statistics bytes and parsed document disagree")
        sha = hashlib.sha256(raw).hexdigest()
        if provenance.get("sha256") != sha or provenance.get("size_bytes") != len(raw):
            return _unknown("Statistics content identity does not match the retained bytes")
        source = selected["statistics_reference"]
        if provenance.get("source") != source:
            return _unknown("The original statistics source differs from this command's selected asset")
        parts = urlsplit(source)
        if not parts.scheme:
            if not Path(source).is_absolute() or provenance.get("kind") != "local_regular_file":
                return _unknown("Local statistics lack an absolute selected source and trusted regular-file intake observation")
            return {"status": "pass", "source": source, "sha256": sha, "size_bytes": len(raw),
                    "identity_kind": "local_content_snapshot",
                    "origin_scope": "Exact local bytes captured by the trusted bounded regular-file reader; the path can change later and origin, training history and cloud generation remain unassessed"}
        if parts.scheme != "gs" or parts.netloc != "openpi-assets":
            return _unknown("The selected statistics source has no supported content binding")
        metadata, metadata_raw = snapshot.metadata, snapshot.metadata_raw
        if not isinstance(metadata, dict) or not isinstance(metadata_raw, bytes) or len(metadata_raw) > 64 * 1024 or _parse(metadata_raw) != metadata:
            return _unknown("Original object metadata bytes are unavailable or inconsistent")
        generation, md5 = metadata.get("generation"), metadata.get("md5Hash")
        if (not isinstance(generation, str) or not generation.isascii() or not generation.isdigit()
                or not generation or len(generation) > 30 or int(generation) <= 0
                or metadata.get("bucket") != parts.netloc or metadata.get("name") != parts.path.removeprefix("/")
                or metadata.get("size") != str(len(raw)) or not isinstance(md5, str)):
            return _unknown("Original object metadata does not bind the selected generation, source and size")
        if (base64.b64decode(md5, validate=True) != hashlib.md5(raw).digest()
                or provenance.get("content_identity_verified") is not True
                or provenance.get("generation") != generation
                or provenance.get("bucket") != metadata["bucket"] or provenance.get("object") != metadata["name"]
                or provenance.get("declared_md5_base64") != md5
                or provenance.get("declared_size_bytes") != len(raw)
                or provenance.get("metadata_sha256") != hashlib.sha256(metadata_raw).hexdigest()):
            return _unknown("Intake generation/content observations are incomplete or inconsistent")
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return _unknown("Original statistics/provenance cannot be interpreted consistently")
    return {"status": "pass", "source": source, "generation": generation, "sha256": sha,
            "size_bytes": len(raw), "md5_base64": md5,
            "origin_scope": "Retained bytes are bound to the trusted intake's selected object metadata; this is not historical training authentication"}


def _vector(value):
    if not isinstance(value, list) or len(value) <= 1:
        return None
    try:
        return len(value) if all(type(item) in (int, float) and math.isfinite(item) for item in value) else None
    except OverflowError:
        return None


def _array_shape(value, depth=0):
    if depth > 32:
        return None
    if isinstance(value, list):
        if not value:
            return (0,)
        shapes = [_array_shape(item, depth + 1) for item in value]
        return (len(value), *shapes[0]) if shapes[0] is not None and all(shape == shapes[0] for shape in shapes) else None
    try:
        return () if type(value) in (int, float) and math.isfinite(value) else None
    except OverflowError:
        return None


def _normalization(snapshot, selected, binding):
    if binding["status"] != "pass":
        return _unknown("Selected statistics bytes are not bound to this source profile")
    roles = snapshot.document.get("norm_stats")
    if not isinstance(roles, dict):
        return _unknown("Native typed statistics mapping is unavailable; a runtime error is not inferred")
    if not roles:
        return _unknown("The empty selector can pass through native wrappers; normalization coverage is unassessed")
    use_quantiles = selected["normalization"]["use_quantile_norm"]
    observations, absent_bounds, numerical_unknown = {}, [], False
    for role, values in roles.items():
        if not isinstance(role, str) or not isinstance(values, dict):
            return _unknown("A native statistics leaf has an unmodeled representation")
        # The native loader accepts finite numeric vectors of length one and
        # unequal mean/std lengths. Their arithmetic is a separate question
        # from the constructors' global non-null-quantile requirement.
        shapes = [_array_shape(values.get(name)) for name in ("mean", "std")]
        if any(shape is None or len(shape) != 1 or shape[0] < 1 for shape in shapes):
            return _unknown("Scalar, non-vector or missing statistics have an unmodeled typed representation; no numerical failure is inferred")
        mean, std = shapes[0][0], shapes[1][0]
        numerical_unknown |= mean <= 1 or std <= 1 or mean != std
        if any(values.get(name) is not None and (not isinstance(values[name], list) or _array_shape(values[name]) is None)
               for name in ("q01", "q99")):
            return _unknown("A saved bound field has an unsupported native typed representation; its runtime effect is not inferred")
        observations[role] = {"mean_width": mean, "std_width": std,
                              "native_extra_fields": sorted(set(values) - {"mean", "std", "q01", "q99"})}
        if use_quantiles:
            if values.get("q01") is None or values.get("q99") is None:
                absent_bounds.append(role)
            else:
                q01, q99 = _vector(values["q01"]), _vector(values["q99"])
                if q01 is None or q99 is None or q01 != q99 or q01 != mean:
                    numerical_unknown = True
    common = {"effective_use_quantiles": use_quantiles, "roles": observations,
              "native_execution": "not performed by this pure engine",
              "scope": "Conditional on the pinned source/profile and supported parser semantics, not proof of the user's executing runtime"}
    if absent_bounds:
        return {**common, "status": "error", "code": "openpi.normalization.missing_quantile_bounds",
                "missing_bound_roles": sorted(absent_bounds),
                "message": "This source profile selects quantiles, while the selected NormStats leaves have null/absent bounds. The unchanged native Normalize and Unnormalize constructors reject that combination before arithmetic."}
    if numerical_unknown:
        return {**common, "status": "unknown", "message": "Constructor requirements are satisfied, but broadcast or unequal vector shapes are outside this numerical coverage; no numerical failure is inferred."}
    if set(roles) != {"state", "actions"}:
        return {**common, "status": "unknown", "message": "Native forward selection is non-strict and inverse selection is strict. Missing or extra selectors are not corruption; the effective input/output tree coverage is unassessed."}
    if observations["actions"]["mean_width"] != selected["model"]["action_dim"]:
        return {**common, "status": "unknown", "message": "Declared pre-inverse action width and saved vectors differ; native padding/broadcasting needs separate coverage, not a fabricated shape error."}
    return {**common, "status": "pass", "code": "openpi.normalization.source_profile_requirement_satisfied",
            "message": "The selected mode's constructor requirement is satisfied for these supported vectors. This does not certify the full model, training pairing or robot."}


def _assessment(command, source_root, snapshot):
    selected = resolve_serving_command(command, source_root)
    binding = _binding(snapshot, selected)
    requirement = _normalization(snapshot, selected, binding)
    return {"resolution": selected, "statistics_binding": binding, "normalization": requirement,
            "intake_notes": list(snapshot.reasons) if isinstance(snapshot, OpenPIStatisticsSnapshot) else [],
            "boundaries": {"input_prefix": "The qualified native component accepts a generated8-wide state prefix against original32-wide vectors; actual observations remain uninspected",
                           "pre_inverse": selected.get("action_boundaries"),
                           "final_DroidOutputs": "Source-derived8-coordinate slice follows inverse normalization; native composition remains unproved"}}


def _changes(reference, current):
    result = []
    for group, key in [("model", "model_type"), ("model", "action_horizon"), ("model", "action_dim"),
                       ("normalization", "use_quantile_norm")]:
        before, after = current[group][key], reference[group][key]
        if before != after:
            result.append({"field": f"{group}.{key}", "current": before, "restored_reference": after})
    return result


def compare_openpi_readiness(current_command, reference_command, *, source_root, current_statistics,
                             reference_statistics, restore_reference_intent=False):
    """Compare actual declared commands; propose only an explicitly intended reference restoration."""
    if type(restore_reference_intent) is not bool:
        raise InputError("restore_reference_intent must be an explicit strict boolean")
    current = _assessment(current_command, source_root, current_statistics)
    reference = _assessment(reference_command, source_root, reference_statistics)
    result = {"schema_version": 1, "kind": "openpi_readiness", "status": "unknown",
              "current": current, "reference": reference, "correction": None,
              "unknowns": dict(UNKNOWN_BOUNDARIES),
              "scope": "Conditional source-profile comparison of original selected statistics; no serving/native execution, model/hardware clearance or authenticated training contract"}
    if current["normalization"]["status"] == "error":
        result["status"] = "error"
    current_id, reference_id = current["statistics_binding"], reference["statistics_binding"]
    if current_id["status"] != "pass" or reference_id["status"] != "pass":
        result["correction_reason"] = "Both selected original assets must be consistently bound before a corrective comparison"
        return result
    if current_id.get("generation") is None or reference_id.get("generation") is None:
        result["correction_reason"] = "Local content supports conditional diagnosis, but this correction route requires the same versioned public object; local origin and history remain unassessed"
        return result
    if any(current_id[key] != reference_id[key] for key in ("source", "generation", "sha256", "size_bytes", "md5_base64")):
        result["correction_reason"] = "The initial corrective route requires the exact same selected immutable object, not a matching asset name or an alias"
        return result
    current_selection, reference_selection = current["resolution"], reference["resolution"]
    if (current_selection["source_revision"] != reference_selection["source_revision"]
            or current_selection["source_fingerprints"] != reference_selection["source_fingerprints"]):
        result["correction_reason"] = "Reference/current source profiles differ"
        return result
    result["consequential_changes"] = _changes(reference_selection, current_selection)
    if not restore_reference_intent:
        result["correction_reason"] = "No explicit instruction to restore the reference intent; a mode/config change cannot be recommended unconditionally"
        return result
    if current["normalization"]["status"] != "error" or reference["normalization"]["status"] != "pass":
        result["correction_reason"] = "No supported changed-profile contradiction with an assessed reference requirement is available"
        return result
    replay = _assessment(reference_command, source_root, current_statistics)
    if replay["normalization"]["status"] != "pass" or replay["statistics_binding"] != current_id:
        result["correction_reason"] = "The exact proposed reference selection did not replay at the same content pin"
        return result
    result["correction"] = {"command": reference_command,
                            "intent_basis": "Caller explicitly requested restoring this declared reference; no historical training or working-model authority is inferred",
                            "changes": result["consequential_changes"], "preserved_statistics": dict(current_id),
                            "target": "openpi.normalization.missing_quantile_bounds", "replay": replay,
                            "targeted_requirement_status": "pass", "overall_after_correction": "unknown",
                            "native_verification": "Not executed by this pure engine; the proposal qualification must bind original-body native constructor/wrapper results derived from these resolutions"}
    return result
