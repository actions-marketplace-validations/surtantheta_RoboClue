"""Ordinary saved folders to a bounded selected inner numeric comparison.

No arrays or authored contracts cross this API. Pure arithmetic observations are
kept separate from selected current-file evidence and untested native targets.
"""

from __future__ import annotations

import hashlib
import json

from .artifacts import InputError
from .consumer_context import _authorized_root, resolve_native_processor_context
from .consumer_numeric import compare_consumer_numeric
from .consumer_samples import load_consumer_sample

MAX_SAMPLES = 9
MAX_REPORT_BYTES = 256 * 1024
# Each selector may read 3 MiB JSON + 8 MiB Parquet twice. Each side also
# revalidates <=8 MiB context in selector and arithmetic; 2 initial contexts.
MAX_BODY_ALLOWANCE = (16 + MAX_SAMPLES * 76) * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _selectors(samples):
    if type(samples) is not list or not 1 <= len(samples) <= MAX_SAMPLES:
        raise InputError("samples must contain one to nine explicit selected windows.")
    result = []
    for sample in samples:
        if (type(sample) is not dict or set(sample) - {"episode_index", "anchor_index", "allow_padding"}
                or not {"episode_index", "anchor_index"} <= set(sample)):
            raise InputError("Each sample requires only episode_index, anchor_index and optional allow_padding.")
        for key, maximum in (("episode_index", 999999), ("anchor_index", 4095)):
            if type(sample[key]) is not int or not 0 <= sample[key] <= maximum:
                raise InputError(key + " must be a bounded nonnegative integer.")
        if type(sample.get("allow_padding", False)) is not bool:
            raise InputError("allow_padding must be a strict boolean.")
        result.append({**sample, "allow_padding": sample.get("allow_padding", False)})
    return result


def _finding(code, status, message, **extra):
    return {"code": code, "status": status, "message": message, **extra,
            "next": "Inspect the selected saved artifacts and rows; preserve the reference independently."}


def _context_summary(context):
    descriptor = context.get("descriptor") or {}
    return {"status": context["status"], "findings": context["findings"],
            "findings_omitted": context.get("findings_omitted", 0), "axes": context["axes"],
            "artifacts": context["artifacts"], "evaluator": context["evaluator"],
            "interpretation_sha256": descriptor.get("interpretation_sha256"),
            "flags": descriptor.get("flags"),
            "numeric_context_complete": descriptor.get("numeric_context_complete", False)}


def compare_native_consumer_samples(reference_dataset, reference_checkpoint,
                                    candidate_dataset, candidate_checkpoint, *, samples,
                                    embodiment=None, require_equal=False, read_roots=None):
    """Compare at most nine stored-ID/physical-anchor windows from ordinary folders.

    CHANGED is an observation. require_equal declares only a selected supported
    inner-value invariant, not a complete learner target or historical baseline.
    Invalid request/authorization raises before reads. Per-side artifact/selection
    failures are reported without discarding other samples or known context errors.
    """
    selected = _selectors(samples)
    if type(require_equal) is not bool:
        raise InputError("require_equal must be a strict boolean.")
    # Lexical guards/authorization for ALL inputs precede every filesystem read.
    roots = {name: _authorized_root(path, read_roots) for name, path in (
        ("reference_dataset", reference_dataset), ("reference_checkpoint", reference_checkpoint),
        ("candidate_dataset", candidate_dataset), ("candidate_checkpoint", candidate_checkpoint))}
    if embodiment is not None and (type(embodiment) is not str or not 0 < len(embodiment) <= 256
                                   or any(ord(c) < 32 for c in embodiment)):
        raise InputError("embodiment must be a bounded tag string.")
    contexts = {side: resolve_native_processor_context(roots[side + "_checkpoint"],
                embodiment=embodiment, read_roots=read_roots) for side in ("reference", "candidate")}
    evidence, evidence_ids = [], {}
    observed_files, instability = {}, []

    def intern(kind, value):
        identity = _digest({"kind": kind, "value": value})
        if identity not in evidence_ids:
            evidence_ids[identity] = len(evidence)
            evidence.append({"kind": kind, "sha256": identity, "value": value})
        return evidence_ids[identity]

    def source(side, loaded, sample_index):
        selection = loaded["raw_sample"]["selection"]
        # A side label is not file identity: reference and candidate may share a
        # root. Compare observations across ALL windows/sides without reopening
        # or discarding either positive per-window capture.
        for artifact in loaded["source_evidence"]["artifacts"]:
            key = (str(roots[side + "_dataset"]), artifact["path"])
            observation = {"sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"],
                           "side": side, "sample_index": sample_index}
            previous = observed_files.setdefault(key, observation)
            if (previous["sha256"], previous["size_bytes"]) != (observation["sha256"], observation["size_bytes"]):
                instability.append({"dataset_directory": key[0], "relative_path": key[1],
                                    "first_observation": previous, "different_observation": observation})
        artifact_refs = [intern("selected_file", {"dataset": side, **artifact})
                         for artifact in loaded["source_evidence"]["artifacts"]]
        return {"scope": "Current selected physical rows and declared slices; not recording/native authority.",
                "status": loaded["source_evidence"]["status"], "artifact_refs": artifact_refs,
                "field_mapping_ref": intern("declared_slices", selection["fields"]),
                **{key: selection[key] for key in (
                    "stored_episode_index", "native_episode_position", "anchor_index", "episode_length",
                    "state_indices", "action_indices", "action_global_indices", "allow_padding",
                    "clamped_occurrences", "native_training_start_eligible")},
                "recording_association": "UNKNOWN", "native_parity": "UNKNOWN"}

    windows = []
    # Surface fresh saved-context contradictions even when selection stops before
    # arithmetic. Keep exact cause/pointer once, with side authority explicit.
    findings = [dict(finding, side=side, cause_scope="saved_native_context")
                for side, context in contexts.items() for finding in context["findings"]
                if finding["status"] == "ERROR"]

    def context_cause(side, finding):
        return (side, finding.get("code"), finding.get("message"),
                finding.get("pointer"), finding.get("field"))

    seeded_causes = {context_cause(f["side"], f) for f in findings}
    for index, selector in enumerate(selected):
        window = {"sample_index": index, "requested_selection": selector, "source_rows": {},
                  "status": "UNKNOWN", "numeric_projection_value_effect": "UNKNOWN", "findings": []}
        loaded = {}
        for side in ("reference", "candidate"):
            try:
                loaded[side] = load_consumer_sample(roots[side + "_dataset"], contexts[side],
                                                   **selector, read_roots=read_roots)
                window["source_rows"][side] = source(side, loaded[side], index)
            except (InputError, OSError, ValueError, ImportError) as error:
                window["findings"].append(_finding("consumer.selection_unavailable", "UNKNOWN",
                    str(error)[:256], side=side))
        if len(loaded) == 2:
            comparison = compare_consumer_numeric(contexts["reference"], loaded["reference"]["raw_sample"],
                contexts["candidate"], loaded["candidate"]["raw_sample"], read_roots=read_roots)
            for key in ("artifact_identity_changed", "active_interpretation_changed", "selection_changed",
                        "active_parameter_changes", "first_observed_divergence", "first_value_divergence",
                        "numeric_projection_value_effect", "final_inner_numeric_counts", "final_inner_fields",
                        "first_value_differences", "first_value_differences_omitted", "complete_field_coverage"):
                window[key] = comparison[key]
            # Full traces/readsets are intentionally not repeated for each window.
            window["numeric_evaluator"] = {side: comparison[side]["evaluator"]
                                           for side in ("reference", "candidate")}
            window["numeric_findings"] = {side: comparison[side]["findings"]
                                          for side in ("reference", "candidate")}
            for side in ("reference", "candidate"):
                findings.extend(dict(f, side=side, sample_index=index)
                    for f in comparison[side]["findings"]
                    if context_cause(side, f) not in seeded_causes)
            window["status"] = comparison["status"]
            if require_equal and window["numeric_projection_value_effect"] == "CHANGED":
                window["status"] = "ERROR"
                window["findings"].append(_finding("consumer.selected_value_invariant_changed", "ERROR",
                    "Proven selected inner numeric values changed under the requested equality invariant."))
        findings.extend(dict(f, sample_index=index) for f in window["findings"])
        windows.append(window)
    context_errors = any(context["status"] == "ERROR" for context in contexts.values())
    known_error = context_errors or any(w["status"] == "ERROR" for w in windows)
    effects = [w["numeric_projection_value_effect"] for w in windows]
    effect = "CHANGED" if "CHANGED" in effects else "EQUAL" if all(e == "EQUAL" for e in effects) else "UNKNOWN"
    findings.sort(key=lambda finding: {"ERROR": 0, "UNKNOWN": 1}.get(finding["status"], 2))
    stability = {"status": "UNKNOWN" if instability or any(len(w["source_rows"]) < 2 for w in windows) else "PASS",
                 "scope": "Equality of captured file hashes/sizes across observed selections only; not continuous immutability.",
                 "observed_unique_files": len(observed_files), "differing_observations": instability,
                 "differing_observation_count": len(instability)}
    if instability:
        findings.append(_finding("consumer.batch_input_instability", "UNKNOWN",
            "Different versions of the same input file were observed across selected windows or sides."))
    report = {"schema_version": 1, "kind": "native_consumer_sample_comparison",
              "status": "ERROR" if known_error else ("PASS" if all(
                  w.get("complete_field_coverage", False) for w in windows)
                  and stability["status"] == "PASS" else "UNKNOWN"),
              "requested_scope": "selected supported inner numeric comparison",
              "scope": "Selected supported inner numeric projections; no full native sample or model target claim.",
              "input_stability": stability,
              "require_equal": require_equal, "sources": {k: str(v) for k, v in roots.items()},
              "contexts": {side: _context_summary(context) for side, context in contexts.items()},
              "samples": windows, "evidence": evidence, "findings": findings[:64],
              "findings_total": len(findings), "findings_omitted": max(0, len(findings) - 64),
              "numeric_projection_value_effect": effect,
              "coverage": {"requested_windows": len(selected),
                  "fully_projected_windows": sum(w.get("complete_field_coverage", False) for w in windows),
                  "changed_windows": effects.count("CHANGED"), "equal_windows": effects.count("EQUAL"),
                  "unresolved_windows": effects.count("UNKNOWN")},
              "bounds": {"max_windows": MAX_SAMPLES, "max_report_bytes": MAX_REPORT_BYTES,
                  "conservative_body_read_allowance_bytes": MAX_BODY_ALLOWANCE,
                  "accounting": "Sum of fixed application-level reader allowances, not measured I/O or wire traffic."},
              "boundaries": {key: "UNKNOWN" for key in (
                  "native_assembler_parity", "native_processor_parity", "wrapper_padding_masks_torch_dtype",
                  "final_wrapper_model_target_effect", "images_language_augmentation",
                  "historical_working_reference", "original_recording_association", "physical_truth")}}
    # Keep counts, identities and primary diagnoses even if unusually verbose
    # evidence would exceed the response allowance. No descriptor is resealed.
    if len(json.dumps(report, allow_nan=False).encode()) > MAX_REPORT_BYTES:
        for window in windows:
            window["first_value_differences_omitted"] = (
                window.get("first_value_differences_omitted", 0) + len(window.get("first_value_differences", [])))
            window["first_value_differences"] = []
        report["evidence_detail_reduced"] = True
    if len(json.dumps(report, allow_nan=False).encode()) > MAX_REPORT_BYTES:
        raise InputError("Selected comparison exceeds its report bound; request fewer windows.")
    return report
