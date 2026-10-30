"""One existing readiness front door for a declared OpenPI serving job."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
from pathlib import Path

from .artifacts import InputError
from .openpi_intake import _object, resolve_openpi_statistics
from .openpi_interface import Unsupported, command_fields, resolve_serving_command
from .openpi_readiness import compare_openpi_readiness
from .readiness_intake import ReadinessIntakeBudget
from .readiness_workflow import _local, authorize_command_directory


def looks_openpi(command):
    return isinstance(command, str) and "serve_policy.py" in command


def _unknown(message, remedy):
    return {"schema_version": 1, "kind": "openpi_readiness", "status": "unknown",
            "summary": message, "current": {"normalization": {"status": "unknown", "message": message}},
            "reference": {"normalization": {"status": "unknown", "message": "Reference not assessed"}},
            "correction": None, "unknowns": {"command_context": message},
            "findings": [{"code": "openpi.command_context_unknown", "status": "unknown", "message": message, "remedy": remedy}],
            "next_actions": [remedy], "scope": "No supported command context was established; no public intake, code or serving execution occurred."}


def _generation(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{0,29}", value)):
        raise InputError("Statistics generations must be positive ASCII decimal object generations, not model revisions")


def _effective_command(command, base, roots):
    """Canonicalize only a supported literal checkpoint directory, with known cwd."""
    if command is None:
        return None
    # Do not let shlex erase control characters or turn unsupported syntax into support.
    if any(ord(character) < 32 for character in command):
        return command
    try:
        _, fields = command_fields(command)
    except (Unsupported, ValueError):
        return command
    checkpoint = fields.get("policy.dir")
    if checkpoint is None or "://" in checkpoint:
        return command
    if checkpoint.startswith("~"):
        raise InputError("Literal policy.dir beginning with ~ is ambiguous under native OpenPI path semantics; supply the intended absolute checkpoint path")
    selected = _local(checkpoint, roots, base)
    if Path(checkpoint).is_absolute():
        return command
    words = shlex.split(command)
    for index, word in enumerate(words):
        if word.startswith("--policy.dir="):
            words[index] = "--policy.dir=" + selected
            break
        if word == "--policy.dir":
            words[index + 1] = selected
            break
    return shlex.join(words)


def check_openpi_readiness(robot_command, *, reference_command=None, source_root=None, command_directory=None,
        statistics_generation=None, reference_statistics_generation=None, restore_reference_intent=False,
        fetch=False, max_bytes=8 * 1024 ** 2, max_files=16, read_roots=None, base_directory=None, _resolutions=None):
    """Resolve all selected inputs before public reads; never execute command text."""
    if type(fetch) is not bool or type(restore_reference_intent) is not bool:
        raise InputError("OpenPI fetch/restore_reference_intent must be strict booleans")
    for value in [robot_command, reference_command]:
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 32768):
            raise InputError("Serving commands must be bounded nonempty text")
    if not isinstance(robot_command, str):
        raise InputError("An actual current serving command is required")
    _generation(statistics_generation)
    _generation(reference_statistics_generation)
    if reference_command is None and (restore_reference_intent or reference_statistics_generation is not None):
        raise InputError("Reference generation/restoration requires an actual supplied reference_command")
    budget = ReadinessIntakeBudget(max_bytes=max_bytes, max_files=max_files)
    # A read root or a fingerprinted source tree never supplies a pasted command's cwd.
    if read_roots is not None and command_directory is None:
        return _unknown("OpenPI's relative serving entry point has no explicitly declared command directory",
                        "Supply an authorized absolute command_directory for the intended OpenPI checkout.")
    base = (authorize_command_directory(command_directory, read_roots=read_roots, resolutions=_resolutions)
            if command_directory is not None else str(Path(base_directory or Path.cwd())))
    base = authorize_command_directory(base, read_roots=read_roots)
    root = (_local(source_root, read_roots, base, explicit="OpenPI source root", resolutions=_resolutions)
            if source_root is not None else base)
    if root != base:
        raise InputError("source_root must match the known command directory for scripts/serve_policy.py; an unrelated source profile cannot attest it")
    effective_current = _effective_command(robot_command, base, read_roots)
    effective_reference = _effective_command(reference_command, base, read_roots)
    selections = {"current": resolve_serving_command(effective_current, root)}
    selections["reference"] = resolve_serving_command(effective_reference, root) if reference_command is not None else {"status": "unknown"}
    selected_sources = {}
    generations = {"current": statistics_generation, "reference": reference_statistics_generation}
    # BOTH current/reference local paths and public authorities are guarded before ANY HTTP.
    for role, selected in selections.items():
        if selected.get("status") != "resolved_supported_builtin":
            continue
        source = selected["statistics_reference"]
        if "://" in source:
            _object(source)
        else:
            source = _local(source, read_roots, base)
            if generations[role] is not None:
                raise InputError("Statistics-object generations cannot be applied to a selected local file")
        selected_sources[role] = source
    snapshots, cache = {}, {}
    # Unsupported current selection has no justified public asset to inspect.
    if selections["current"].get("status") == "resolved_supported_builtin":
        for role in ["current", "reference"]:
            if role not in selected_sources:
                continue
            source, generation = selected_sources[role], generations[role]
            key = (source, generation)
            if key not in cache:
                snapshot = resolve_openpi_statistics(source, fetch=fetch, budget=budget,
                    read_roots=read_roots, generation=generation)
                cache[key] = snapshot
                # A retained exact generation, even with unavailable content, is not silently retried.
                if snapshot.provenance.get("generation") is not None:
                    cache[(source, snapshot.provenance["generation"])] = snapshot
            snapshots[role] = cache[key]
    report = compare_openpi_readiness(effective_current, effective_reference, source_root=root,
        current_statistics=snapshots.get("current"), reference_statistics=snapshots.get("reference"),
        restore_reference_intent=restore_reference_intent)
    report["summary"] = ("The current source profile conflicts with the selected statistics; supported correction remains conditional."
        if report["status"] == "error" else "The selected serving job retains unassessed runtime, model and historical boundaries.")
    report["commands"] = {"current": robot_command, "reference": reference_command,
        "effective_current": effective_current, "effective_reference": effective_reference,
        "interpretation": "Only supported literal relative checkpoint directories are resolved against the known command directory; source-profile interpretation, not runtime inspection"}
    report["command_context"] = {"command_directory": base, "source_root": root,
        "basis": "Explicit intended command directory" if command_directory is not None else "Actual diagnostic CLI/Python working directory",
        "condition": "Relative script selection is bound to this directory/profile; no running serving process was inspected",
        "current_command_sha256": hashlib.sha256(robot_command.encode()).hexdigest(),
        "reference_intent": "Caller-declared comparison intent, not historical working proof" if reference_command is not None else "No reference supplied"}
    report["intake_budget"] = budget.summary()
    report["statistics_intake"] = {role: {"status": value.status, "provenance": copy.deepcopy(value.provenance),
        "reasons": list(value.reasons), "requested_generation": generations[role]} for role, value in snapshots.items()}
    report["snapshot_reuse"] = {"selected_inputs": len(snapshots), "distinct_attempt_keys": len({id(x) for x in snapshots.values()}),
        "scope": "Identical URI/requested generation reuses the retained snapshot, including unavailable evidence; no automatic retry"}
    report["unknowns"]["actual_command_directory"] = "The intended directory/source profile is declared above; no running native process was observed"
    current = report["current"]["normalization"]
    remedy = (report.get("correction") or {}).get("command")
    next_action = ("If restoring the supplied reference is intended, review its source-derived model/horizon/mode changes and use the exact correction below."
        if remedy else report.get("correction_reason", "Inspect the original selected statistics and intended policy/config pairing."))
    report["findings"] = [{"code": current.get("code", "openpi.normalization_unassessed"), "status": current["status"],
        "message": current.get("message", "Current normalization remains unassessed"), "remedy": next_action},
        {"code": "openpi.remaining_boundaries", "status": "unknown",
         "message": "Actual runtime, checkpoint/model pairing, historical training mode and physical behavior remain unassessed.",
         "remedy": "Validate the executing environment and intended model/statistics pairing; a constructor correction is not complete serving repair."}]
    if reference_command is None:
        report["next_actions"] = ["Inspect the intended config and selected statistics pairing; do not change the effective normalization mode without validating model compatibility.",
            "Supply an actual reference_command only if a comparison is wanted; opt in with restore_reference_intent to request a conditional restoration."]
    else:
        report["next_actions"] = [next_action]
    return report


def check_readiness(*, robot_command=None, robot_config=None, checkpoint=None, dataset=None,
        policy_revision="main", dataset_revision=None, fetch=False, processor_state=False,
        max_bytes=8 * 1024 ** 2, max_files=16, read_roots=None, base_directory=None,
        reference_command=None, source_root=None, command_directory=None, statistics_generation=None,
        reference_statistics_generation=None, restore_reference_intent=False, openpi_fetch=None):
    """Keep the existing readiness tool/command; permissions remain stack-specific."""
    if openpi_fetch is not None and type(openpi_fetch) is not bool:
        raise InputError("OpenPI intake authority must be a strict boolean")
    special = any(x is not None for x in [reference_command, source_root, statistics_generation, reference_statistics_generation]) or restore_reference_intent is not False
    if not looks_openpi(robot_command):
        if special:
            raise InputError("OpenPI-specific comparison/source/generation options require a serving command; unsupported combinations are not ignored")
        from .readiness_workflow import check_robot_readiness

        resolved = []  # an explicitly named command directory reached through a symlink: resolved once, reported
        base = authorize_command_directory(command_directory, read_roots=read_roots, resolutions=resolved) if command_directory is not None else base_directory
        result = check_robot_readiness(robot_command=robot_command, robot_config=robot_config, checkpoint=checkpoint,
            dataset=dataset, policy_revision=policy_revision, dataset_revision=dataset_revision, fetch=fetch,
            processor_state=processor_state, max_bytes=max_bytes, max_files=max_files, read_roots=read_roots,
            base_directory=base)
        return dict(result, resolved_paths=resolved + list(result.get("resolved_paths") or [])) if resolved else result
    if robot_config is not None or checkpoint is not None or dataset is not None or policy_revision != "main" or dataset_revision is not None:
        raise InputError("OpenPI serving readiness does not accept robot_config, dataset or Hub policy/revision overrides; select the checkpoint in the actual command")
    resolved = []  # filled when check_openpi_readiness authorizes the named directory and source root
    result = check_openpi_readiness(robot_command, reference_command=reference_command, source_root=source_root,
        command_directory=command_directory, statistics_generation=statistics_generation,
        reference_statistics_generation=reference_statistics_generation, restore_reference_intent=restore_reference_intent,
        fetch=fetch if openpi_fetch is None else openpi_fetch, max_bytes=max_bytes, max_files=max_files,
        read_roots=read_roots, base_directory=base_directory, _resolutions=resolved)
    return dict(result, resolved_paths=resolved + list(result.get("resolved_paths") or [])) if resolved else result


def concise_openpi_summary(value):
    """Keep distinct evidence once, including exact correction and remaining unknowns."""
    def subset(part, keys):
        return {key: copy.deepcopy(part[key]) for key in keys if key in part}
    def normalization(part):
        return subset(part, ["status", "code", "message", "effective_use_quantiles", "missing_bound_roles", "roles"])
    def binding(part):
        return subset(part, ["status", "reason", "message", "source", "generation", "sha256", "size_bytes", "md5_base64", "identity_kind", "origin_scope"])
    current_binding = binding(value.get("current", {}).get("statistics_binding", {}))
    def assessment(part):
        selected = part.get("resolution", {})
        pin = binding(part.get("statistics_binding", {}))
        if pin and pin == current_binding and pin.get("status") == "pass":
            pin = {"status": "pass", "same_pin_as": "current.statistics_binding"}
        return {"resolution": subset(selected, ["status", "reason", "config_id", "checkpoint_reference", "source_revision", "model"]),
                "statistics_binding": pin, "normalization": normalization(part.get("normalization", {}))}
    result = subset(value, ["schema_version", "kind", "status", "unknowns", "scope", "correction_reason",
                            "consequential_changes", "intake_budget", "next_actions"])
    result.update(summary=True, readiness_summary=value.get("summary"), current=assessment(value.get("current", {})),
        reference=assessment(value.get("reference", {})), correction=None,
        full_report_sha256=hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        omissions="Repeated pins, fingerprint maps, detailed native boundary prose and intake notes omitted. Conditions, changes, exact correction and unknowns retained.")
    result["current"]["statistics_binding"] = current_binding
    context = value.get("command_context", {})
    result["command_context"] = subset(context, ["command_directory", "basis", "reference_intent"])
    result["source_condition"] = ("Declared cwd equals source root; supported fingerprinted profile/locked Tyro semantics only. No running process, native constructor, model or robot was executed by this diagnostic."
        if context else "No supported command context was established; no public intake or execution occurred.")
    if value.get("correction"):
        correction = value["correction"]
        result["correction"] = subset(correction, ["command", "intent_basis", "changes", "target", "targeted_requirement_status", "overall_after_correction", "native_verification"])
        result["correction"]["preserved_statistics"] = {"same_pin_as": "current.statistics_binding"}
        replay = correction["replay"]
        result["correction"]["replay"] = {"statistics_binding": binding(replay.get("statistics_binding", {})),
            "normalization": normalization(replay.get("normalization", {}))}
        if result["correction"]["replay"]["statistics_binding"] == current_binding:
            result["correction"]["replay"]["statistics_binding"] = {"status": "pass", "same_pin_as": "current.statistics_binding"}
    if len(json.dumps(result, allow_nan=False).encode()) > 8192:
        raise InputError("OpenPI concise report exceeds 8 KiB; use bounded command/path text or request detailed")
    return result
