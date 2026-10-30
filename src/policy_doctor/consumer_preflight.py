"""Ordinary-folder consumer comparison through the existing preflight entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .artifacts import InputError
from .consumer_context import _authorized_root, resolve_native_processor_context


def _preview(text, budget):
    """Bound JSON-escaped content bytes, including supplementary Unicode text."""
    size, chars = 0, []
    for char in text:
        cost = len(json.dumps(char, ensure_ascii=True)) - 2
        if size + cost > budget:
            break
        size += cost
        chars.append(char)
    value = "".join(chars)
    return value, len(value) < len(text)


def _text_digest(text):
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def prepare_consumer_comparison(source, checkpoint, *, reference_dataset=None,
                                reference_checkpoint=None, samples=None,
                                require_equal=False, read_roots=None):
    """Authorize every local selector before any optional preflight network intake."""
    if type(require_equal) is not bool:
        raise InputError("consumer_require_equal must be a strict boolean.")
    requested = any(value is not None for value in (reference_dataset, reference_checkpoint, samples)) or require_equal
    if not requested:
        return None
    if checkpoint is None or reference_checkpoint is None:
        raise InputError("Consumer comparison requires --checkpoint and --consumer-reference-checkpoint.")
    if samples is not None:
        if type(samples) is not list or not 1 <= len(samples) <= 9:
            raise InputError("consumer_samples must contain one to nine selected windows.")
        for item in samples:
            if (type(item) is not dict or set(item) - {"episode_index", "anchor_index", "allow_padding"}
                    or not {"episode_index", "anchor_index"} <= set(item)):
                raise InputError("Each consumer sample needs episode_index, anchor_index and optional allow_padding.")
            for key, maximum in (("episode_index", 999999), ("anchor_index", 4095)):
                if type(item[key]) is not int or not 0 <= item[key] <= maximum:
                    raise InputError("Consumer " + key + " must be a bounded nonnegative integer.")
            if type(item.get("allow_padding", False)) is not bool:
                raise InputError("Consumer allow_padding must be a strict boolean.")
    roots = {name: str(_authorized_root(value, read_roots)) for name, value in (
        ("reference_dataset", source if reference_dataset is None else reference_dataset),
        ("reference_checkpoint", reference_checkpoint),
        ("candidate_dataset", source), ("candidate_checkpoint", checkpoint))}
    return {**roots, "samples": samples, "require_equal": require_equal}


def _canonical(finding, **extra):
    value = {**finding, **extra}
    value["status"] = str(value.get("status", "UNKNOWN")).lower()
    value["remedy"] = value.get("next", "Inspect the saved processor configuration and cited selected rows.")
    value.setdefault("evidence", [])
    return value


def _artifacts(contexts, roots, comparison=None, planning=None):
    result = []
    captures = [contexts]
    if comparison:
        captures.append(comparison.get("contexts", {}))
    for capture in captures:
        for side, context in capture.items():
            for item in context.get("artifacts", {}).values():
                result.append({**item, "path": str(Path(roots[side + "_checkpoint"]) / item["filename"]),
                               "role": "saved_native_processor"})
    if planning:
        for item in planning.get("readset", []):
            result.append({**item, "path": str(Path(roots["reference_dataset"]) / item["path"]),
                           "role": "consumer_sample_plan"})
    if comparison:
        for item in comparison.get("evidence", []):
            if item.get("kind") == "selected_file":
                value = item["value"]
                result.append({**value, "path": str(Path(roots[value["dataset"] + "_dataset"]) / value["path"]),
                               "role": "selected_consumer_source"})
    # Preserve differing identities at a repeated path for the parent stability check.
    unique = {}
    for item in result:
        unique[(item["path"], item["sha256"], item["size_bytes"])] = item
    return list(unique.values())


def check_consumer_comparison(plan, *, read_roots=None):
    """Compare supported inner numeric windows; broader native/model axes stay explicit."""
    contexts = {side: resolve_native_processor_context(plan[side + "_checkpoint"], read_roots=read_roots)
                for side in ("reference", "candidate")}
    findings = [_canonical(item, side=side) for side, context in contexts.items()
                for item in context.get("findings", []) if item.get("status") != "PASS"]
    planning = None
    comparison = None
    selected = plan["samples"]
    try:
        if selected is None:
            from .consumer_sample_plan import plan_consumer_samples

            planning = plan_consumer_samples(plan["reference_dataset"], contexts["reference"], read_roots=read_roots)
            findings.extend(_canonical(item, side="reference") for item in planning.get("findings", [])
                            if item.get("cause_scope") != "saved_native_context")
            if planning.get("planning_status") != "PASS":
                raise InputError("Automatic consumer window selection could not establish a supported plan.")
            if planning.get("context_artifacts") != contexts["reference"].get("artifacts"):
                raise InputError("Saved reference processor changed during automatic window selection.")
            selected = planning["samples"]
        from .consumer_workflow import compare_native_consumer_samples

        comparison = compare_native_consumer_samples(
            plan["reference_dataset"], plan["reference_checkpoint"],
            plan["candidate_dataset"], plan["candidate_checkpoint"],
            samples=selected, require_equal=plan["require_equal"], read_roots=read_roots)
        findings.extend(_canonical(item) for item in comparison.get("findings", []))
        if planning:
            expected = {item["path"]: (item["sha256"], item["size_bytes"]) for item in planning["readset"]}
            observed = {name: set() for name in expected}
            for item in comparison.get("evidence", []):
                if item.get("kind") != "selected_file":
                    continue
                value = item["value"]
                if value.get("dataset") == "reference" and value.get("path") in observed:
                    observed[value["path"]].add((value["sha256"], value["size_bytes"]))
            if any(observed[name] != {identity} for name, identity in expected.items()):
                findings.append(_canonical({"code": "consumer.plan_input_drift", "status": "UNKNOWN",
                    "message": "Reference metadata observed during comparison differs from automatic window planning."}))
        for side, context in contexts.items():
            if comparison.get("contexts", {}).get(side, {}).get("artifacts") != context.get("artifacts"):
                findings.append(_canonical({"code": "consumer.context_input_drift", "status": "UNKNOWN",
                    "message": "Saved processor identity changed between initial intake and comparison."}, side=side))
    except (InputError, ImportError, OSError, ValueError) as error:
        findings.append(_canonical({"code": "consumer.comparison_unavailable", "status": "UNKNOWN",
            "message": str(error)[:400]}))
    unique = {}
    for finding in findings:
        key = tuple(str(finding.get(name)) for name in ("status", "code", "message", "side", "pointer", "sample_index"))
        unique.setdefault(key, finding)
    findings = sorted(unique.values(), key=lambda f: {"error": 0, "unknown": 1, "warning": 2}.get(f["status"], 3))
    states = [f["status"] for f in findings]
    states.append(comparison["status"].lower() if comparison else "unknown")
    status = next(s for s in ("error", "unknown", "warning", "pass") if s in states)
    return {"kind": "consumer_sample_comparison", "status": status,
            "requested_scope": "selected supported inner numeric comparison",
            "scope": "Current selected numeric rows and supported saved-processor arithmetic; no full native processor, model or historical-run claim.",
            "numeric_projection_value_effect": comparison.get("numeric_projection_value_effect", "UNKNOWN") if comparison else "UNKNOWN",
            "require_equal": plan["require_equal"],
            "findings": findings[:64], "total_findings": len(findings), "findings_truncated": len(findings) > 64,
            "artifacts": _artifacts(contexts, plan, comparison, planning),
            "coverage": comparison.get("coverage", {"requested_windows": len(selected or []), "fully_projected_windows": 0}) if comparison else {
                "requested_windows": len(selected or []), "fully_projected_windows": 0},
            "planning": planning, "comparison": comparison,
            "boundaries": comparison.get("boundaries", {}) if comparison else {
                "native_processor_parity": "UNKNOWN", "final_wrapper_model_target_effect": "UNKNOWN"}}


def concise_consumer_comparison(component):
    """Keep the actual effect, error cause and first source witness in concise MCP."""
    comparison = component.get("comparison") or {}
    sample = next((item for item in comparison.get("samples", []) if item.get("first_observed_divergence")), None)
    divergence = sample.get("first_observed_divergence") if sample else None
    if divergence:
        divergence = {key: divergence.get(key) for key in (
            "stage", "field", "coordinate", "change", "row", "reference_row", "candidate_row", "evidence") if key in divergence}
    windows = comparison.get("samples", [])

    def changed(key):
        observations = [window.get(key) for window in windows]
        return True if True in observations else False if observations and all(v is False for v in observations) else None

    value_sample = next((item for item in windows if item.get("first_value_divergence")), None)
    value_divergence = value_sample.get("first_value_divergence") if value_sample else None
    def source_witness(point, selected_sample, *, include_files=True):
        if not point:
            return None
        source_witness = {key: point[key] for key in (
            "stage", "field", "coordinate", "reference_row", "candidate_row") if key in point}
        source_witness["requested_selection"] = selected_sample.get("requested_selection")
        source_witness["source_mapping"] = {}
        role, _, key = point["field"].partition(".")
        for side, rows in selected_sample.get("source_rows", {}).items():
            index = rows.get("field_mapping_ref")
            evidence = comparison.get("evidence", [])
            files = []
            for ref in rows.get("artifact_refs", []):
                if type(ref) is int and 0 <= ref < len(evidence) and evidence[ref].get("kind") == "selected_file":
                    artifact = evidence[ref]["value"]
                    if artifact["path"].endswith(".parquet") or artifact["path"] == "meta/modality.json":
                        files.append({k: artifact[k] for k in ("path", "sha256", "size_bytes")})
            directory = comparison.get("sources", {}).get(side + "_dataset", "")
            if include_files:
                preview, truncated = _preview(directory, 512)
                source_witness[side + "_source"] = {"dataset_directory": preview,
                    "directory_truncated": truncated, "files": files[:2]}
                if truncated:
                    source_witness[side + "_source"]["full_directory_sha256"] = _text_digest(directory)
            if type(index) is int and 0 <= index < len(evidence):
                entry = evidence[index]
                if entry.get("kind") == "declared_slices":
                    mapping = next((item for item in entry["value"] if
                                    item.get("modality") == role and item.get("key") == key), None)
                    if mapping:
                        coordinate = point["coordinate"][1]
                        source_witness["source_mapping"][side] = {
                            "column": _preview(mapping["column"], 128)[0],
                            "declared_slice": [mapping["start"], mapping["end"]],
                            "source_coordinate": mapping["start"] + coordinate}
                        if _preview(mapping["column"], 128)[1]:
                            source_witness["source_mapping"][side].update(
                                column_truncated=True, full_column_sha256=_text_digest(mapping["column"]))
        source_witness["scope"] = "Current selected physical rows and declared column slices; not physical or historical truth."
        if not include_files:
            source_witness["source_identity_from"] = "first_value_divergence"
        return source_witness

    first_source = source_witness(value_divergence, value_sample)
    critical = [item for item in component.get("findings", []) if item.get("status") == "error"]
    unresolved = [item for item in component.get("findings", []) if item.get("status") == "unknown"]
    witness_sample = next((item for item in windows if item.get("first_value_differences")), None)
    final_witness = witness_sample["first_value_differences"][0] if witness_sample else None
    final_source = source_witness({"stage": "clipped", **final_witness}, witness_sample,
                                 include_files=witness_sample is not value_sample) if final_witness else None

    def cause(items):
        if not items:
            return None
        value, clipped = {}, []
        for key, budget in (("code", 128), ("message", 512), ("side", 32), ("pointer", 256)):
            text = items[0].get(key)
            value[key] = _preview(text, budget)[0] if isinstance(text, str) else text
            if isinstance(text, str) and value[key] != text:
                clipped.append(key)
        if clipped:
            value["text_truncated"] = clipped
        return value
    return {"status": component["status"], "requested_scope": component["requested_scope"],
            "numeric_projection_value_effect": component.get("numeric_projection_value_effect", "UNKNOWN"),
            "artifact_identity_changed": changed("artifact_identity_changed"),
            "active_interpretation_changed": changed("active_interpretation_changed"),
            "require_equal": component.get("require_equal", False), "coverage": component.get("coverage", {}),
            "first_observed_divergence": divergence,
            "first_value_divergence": first_source,
            "first_final_value_difference": final_witness, "first_final_value_source": final_source,
            "first_error": cause(critical), "first_unresolved": cause(unresolved),
            "full_native_processor_and_model_target": "UNKNOWN"}
