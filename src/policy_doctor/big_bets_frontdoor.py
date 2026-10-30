"""CLI, readiness guidance and read-only MCP interfaces for bounded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from . import first_use_views, flight_recorder, health_cards, training_guard_capture, upgrade_impact
from .artifacts import InputError, dataset_root
from .conversion_fidelity import _read_small, _reference_json

SUMMARY_LIMIT = 32 * 1024
RESULT_LIMIT = 192 * 1024
MAX_LOCAL_ENTRIES = 20000
MAX_JSON_BYTES = 4 * 1024 * 1024
LIMITS = (
    "Bounded artifact evidence only. Caller declarations, recorded simulation and saved captures "
    "do not authenticate a producer, establish motor delivery, certify model quality or show physical success. "
    "Missing or unsupported evidence remains UNKNOWN. MCP never launches training or writes reports."
)


def _typed(schema, description):
    try:
        from pydantic import Field
        from pydantic.json_schema import WithJsonSchema
    except ImportError:
        return Any
    return Annotated[Any, WithJsonSchema(schema), Field(description=description)]


Flag = _typed({"type": "boolean"}, "Strict Boolean; strings and numbers are refused.")
OptionalFlag = _typed({"anyOf": [{"type": "boolean"}, {"type": "null"}]}, "Strict Boolean or null.")
RowBound = _typed({"type": "integer", "minimum": 1, "maximum": 200000}, "Maximum numeric rows, 1..200000.")
FileBound = _typed({"type": "integer", "minimum": 1, "maximum": 1024}, "Maximum files, 1..1024.")
ByteBound = _typed({"type": "integer", "minimum": 1, "maximum": 64 * 1024 * 1024},
                   "Public intake byte budget, 1..64 MiB.")
Seconds = _typed({"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
                 "Explicit finite positive seconds, or null for unassessed.")
Tracking = _typed({"anyOf": [{"type": "object", "minProperties": 1, "maxProperties": 256,
    "additionalProperties": {"type": "object", "additionalProperties": False, "required": ["unit", "absolute"],
        "properties": {"unit": {"type": "string", "minLength": 1},
                       "absolute": {"type": "number", "minimum": 0}}}}, {"type": "null"}]},
    "Explicit joint name to {unit, absolute} residual tolerances; no unit inference.")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _strict_flag(value, name, *, optional=False):
    if type(value) is not bool and not (optional and value is None):
        raise InputError(f"{name} must be a strict Boolean" + (" or null" if optional else ""))


def _bound(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise InputError(f"{name} must be an integer in 1..{maximum}")


def _health_config(config):
    if type(config) is not dict or config.keys() - health_cards.CONFIG_KEYS:
        raise InputError("Unsupported health-card config options")
    for key in ("video_headers", "commanded_reach", "groot_contract", "units"):
        if key in config:
            _strict_flag(config[key], key, optional=key == "units")
    for key, maximum in (("max_rows", 200000), ("max_files", 1024), ("max_bytes", 64 * 1024 * 1024)):
        if key in config:
            _bound(config[key], key, maximum)
    return config


def _local(value, label):
    if type(value) is not str or not value or len(value) > 4096 or "\x00" in value or "://" in value:
        raise InputError(f"{label} requires a nonempty local path")
    return value


def _read_json(path):
    raw, _ = _read_small(_local(str(path), "JSON input"), limit=MAX_JSON_BYTES)
    return _reference_json(raw)


def _tracking_tolerance_arg(value):
    """Accept a bounded strict JSON object or the existing guarded JSON file."""
    try:
        if value.lstrip().startswith("{"):
            raw = value.encode("utf8")
            if len(raw) > MAX_JSON_BYTES:
                raise InputError("Inline JSON exceeds its byte bound")
            document = _reference_json(raw)
        else:
            document = _read_json(value)
        if type(document) is not dict:
            raise InputError("Tracking tolerance must be a JSON object")
        return document
    except (InputError, OSError, ValueError) as error:
        raise InputError(
            "--tracking-tolerance expects a JSON file path or inline JSON object: " + str(error)
        ) from error


def _check_training_capture(contract, criterion_sha256, capture):
    """Keep the exact verifier gate, with a useful matching-runtime diagnostic."""
    raw, _ = _read_small(_local(str(contract), "Contract"), limit=MAX_JSON_BYTES)
    if hashlib.sha256(raw).hexdigest() == criterion_sha256:
        document = _reference_json(raw)
        verifier = document.get("verifier") if isinstance(document, dict) else None
        if isinstance(verifier, dict) and isinstance(verifier.get("numpy"), str):
            import numpy as np

            recorded = verifier["numpy"]
            if recorded != np.__version__:
                raise InputError(
                    f"The retained target criterion requires NumPy {recorded}; this runtime uses {np.__version__}. "
                    "Run this check in the producer's matching NumPy environment. "
                    "Keep the retained contract and criterion_sha256 unchanged."
                )
    report = training_guard_capture.check_capture(contract, criterion_sha256, capture)
    # Saved target captures do not carry a live adapter's additional inputs.
    document = _reference_json(raw)
    intent = document.get("intent", {}) if isinstance(document, dict) else {}
    inputs = intent.get("roboclue_model_inputs") if isinstance(intent, dict) else None
    if isinstance(inputs, dict) and (inputs.get("fields") or inputs.get("declarations")):
        report["additional_model_input_coverage"] = {
            "status": "unknown", "declared_fields": inputs.get("fields", []),
            "declarations_assessed": False,
            "reason": "The saved target capture has no camera/state or live declaration observations.",
            "next": "Use the original live LeRobot adapter for those retained witnesses; this verdict covers targets and masks only.",
        }
    return report


def _short(value, *, depth=0):
    """Bound previews and mark omissions; never expose a truncated remedy as an instruction."""
    if type(value) is str:
        if len(value.encode()) <= 640:
            return value
        return {"text_omitted": True, "utf8_bytes": len(value.encode()), "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if type(value) in (int, float, bool) or value is None:
        return value
    if depth > 6:
        return {"content_omitted": True, "sha256": _sha(value)}
    if type(value) is list:
        shown = [_short(item, depth=depth + 1) for item in value[:4]]
        result = shown if len(value) <= 4 else {"shown": shown, "omitted": len(value) - 4, "sha256": _sha(value)}
    elif type(value) is dict:
        shown = {(str(key) if len(str(key).encode()) <= 256 else "key_sha256:" + _sha(str(key))):
                 _short(item, depth=depth + 1) for key, item in list(value.items())[:24]}
        if len(value) > 24:
            shown.update(fields_omitted=len(value) - 24, full_sha256=_sha(value))
        result = shown
    else:
        raise InputError("Report contains a non-JSON value")
    if len(_json(result).encode()) <= 4096:
        return result
    # Keep small first-level fields (including source keys, field names and
    # coverage flags) even when a provenance or config object is enormous.
    compact = {"content_omitted": True, "full_sha256": _sha(value)}
    if type(result) is dict:
        for key, item in result.items():
            if type(item) not in (dict, list):
                candidate = {**compact, key: item}
                if len(_json(candidate).encode()) <= 4096:
                    compact = candidate
    return compact


def _finding_status(kind, finding):
    state = finding.get("status")
    if state is not None:
        return state
    # The native guard emits only mismatch diagnostics, without a status per
    # finding. Do not guess the severity of an unfamiliar future diagnostic.
    if kind == "native_target_guard" and finding.get("code") in {
            "target_schema_mismatch", "target_values_differ"}:
        return "error"
    return None


def summarize(report, *, local_cli_argv=None):
    """Preserve verdict, coverage and first source witness in a bounded preview."""
    kind = report.get("kind")
    body = report.get("report", report) if kind == "dataset_health_card" else report
    findings = body.get("findings", [])
    rank = {"error": 0, "changed": 0, "warning": 1, "unknown": 2, "pass": 3}
    ordered = sorted((item for item in findings if isinstance(item, dict)),
                     key=lambda item: rank.get(_finding_status(kind, item), 2))
    total = body.get("finding_count", body.get("total_findings", len(findings)))
    states = [_finding_status(kind, item) if isinstance(item, dict) else None for item in findings]
    known = ("error", "warning", "unknown", "pass", "changed")
    unclassified = sum(state not in known for state in states)
    count_states = ("error", "warning", "unknown", "pass") + (("changed",) if "changed" in states else ())
    result = {"kind": kind, "status": report.get("status", "unknown"), "summary": True,
              "full_report_sha256": _sha(report), "limits": LIMITS,
              "scope": _short(report.get("scope", body.get("scope", "Unassessed"))),
              "findings_total": total,
              "findings_available": len(findings),
              "source_findings_truncated": bool(body.get("findings_truncated", False)),
              "findings_shown": [_short(item) for item in ordered[:4]],
              "findings_omitted": max(0, total - min(4, len(ordered))),
              "counts": report.get("counts", body.get("counts", {
                  state: states.count(state)
                  for state in count_states}))}
    derived_scope = "classified_available_findings" if unclassified else "available_findings" if total > len(findings) else "all_findings"
    result["counts_scope"] = report.get("counts_scope", body.get("counts_scope", derived_scope))
    result["unclassified_available_findings"] = unclassified
    result["counts_meaning"] = (
        "Original report counts; see the reported count scope and coverage."
        if "counts" in report or "counts" in body else
        "Diagnostic findings by status; report-level verdicts and checked samples/values are separate.")
    if local_cli_argv is not None:
        result["local_cli_argv"] = local_cli_argv
        result["local_cli_destinations"] = "Replace NEW.json and NEW_DIR with separate new paths; argv is data, not an executed command."
    if kind == "dataset_health_card":
        result.update(card_id=report.get("card_id"), source_revision=_short(report.get("source_revision")),
                      consumer=_short(report.get("request", {}).get("consumer")),
                      limitations=_short(report.get("limitations")),
                      components={name: {"status": part.get("status", "unknown"),
                                         "coverage": _short(part.get("coverage", {}))}
                                  for name, part in body.get("components", {}).items() if isinstance(part, dict)},
                      next="Run roboclue health-card SOURCE --output NEW.json --artifacts NEW_DIR for the bound report and badge.")
        try:
            result["user_view"] = first_use_views.health_view(report)
        except InputError:
            result["user_view"] = {"overall_status": report.get("status", "unknown"),
                                   "view_status": "unavailable",
                                   "interpretation": "A compact issue table requires a complete preflight report. Original findings and coverage remain authoritative."}
        numeric = body.get('components', {}).get('numeric_rows') or {}
        comparison = numeric.get('statistics_comparison')
        if isinstance(comparison, dict) and 'numeric_rows' in result['components']:
            result['components']['numeric_rows']['statistics_comparison'] = {
                key: _short(comparison.get(key)) for key in
                ('status', 'rows', 'complete_cohort', 'compared_features', 'compared_metrics',
                 'reason', 'mismatch_count', 'scope')}
        units = body.get('components', {}).get('units')
        if isinstance(units, dict) and 'units' in result['components']:
            from .product_questions import units_view

            result['components']['units'].update(units_view(units))
            result['next'] = ('For deployment use preflight_check with the actual robot_command or robot_config. '
                              'Dataset-only unit consistency does not compare the consuming robot; decoded video and policy processors remain unassessed.')
    elif kind == "upgrade_impact":
        result.update(criterion_sha256=report.get("criterion_sha256"),
                      affected_asset_ids=[_short(name) for name in report.get("affected_asset_ids", [])], assets=[])
        for key in ("criterion_intent", "comparison_basis"):
            if key in report:
                result[key] = _short(report[key])
        for asset in report.get("assets", []):
            candidate = asset.get("candidate_check", {})
            delta = asset.get("native_boundary_delta", {})
            witness = candidate.get("first_difference") or next(iter(delta.get("changes", [])), None)
            if witness is None:
                witness = next(iter(candidate.get("findings", [])), None)
            result["assets"].append({key: _short(asset.get(key)) for key in
                ("asset_id", "status", "code", "change_kind", "coverage")})
            result["assets"][-1].update(first_source_witness=_short(witness),
                associated_config_changes=_short(asset.get("associated_config_changes", [])),
                reference_status=asset.get("reference_check", {}).get("status", "unknown"),
                candidate_status=candidate.get("status", "unknown"),
                native_delta_status=delta.get("status", "unknown"))
            for key in ("criterion_intent", "comparison_basis"):
                if key in asset:
                    result["assets"][-1][key] = _short(asset[key])
        result["next"] = "Run roboclue upgrade-impact MANIFEST --output NEW.json --artifacts NEW_DIR for full evidence and alerts.md."
    elif kind == "native_target_guard":
        for key in ("code", "message", "criterion_sha256", "samples_checked", "target_values_checked", "mask_values_checked",
                    "differing_target_values", "differing_mask_values", "first_difference", "uncovered_source_keys"):
            if key in report:
                result[key] = _short(report[key])
        if "additional_model_input_coverage" in report:
            result["additional_model_input_coverage"] = _short(report["additional_model_input_coverage"])
        if report.get("status") == "pass":
            result["next"] = "Covered targets and masks match the retained criterion. Keep the producer, contract and pin unchanged. Live training and uncovered model inputs require their own adapter evidence."
        elif report.get("status") == "error":
            result["next"] = "Review the reported input error or cited target mismatch before changing data or configuration; recheck any confirmed correction against the same retained contract and pin."
        else:
            result["next"] = "Collect the missing supported capture or source identity identified by the report. Keep the retained contract and pin unchanged; missing intent cannot be inferred from a suspect batch."
    elif kind == "flight_recording_triage":
        result.update(recording=_short(report.get("recording")), criteria=_short(report.get("criteria")),
                      next="Inspect cited event IDs and remedies; instrument missing feedback/clocks rather than guessing associations.")
        try:
            result["user_view"] = first_use_views.flight_view(report)
        except InputError:
            result["user_view"] = {"overall_status": report.get("status", "unknown"),
                                   "view_status": "unavailable",
                                   "physical_feedback_association": {"status": "unknown"},
                                   "interpretation": "A compact boundary table requires a complete triage report. Original findings and coverage remain authoritative."}
    # A huge coverage/config object must not make concise responses unbounded.
    if len(_json(result).encode()) > SUMMARY_LIMIT and "user_view" in result:
        view = result["user_view"]
        result["user_view"] = {"overall_status": view.get("overall_status", "unknown"),
                               "details_omitted": True, "full_view_sha256": _sha(view)}
        association = view.get("physical_feedback_association")
        if isinstance(association, dict):
            result["user_view"]["physical_feedback_association"] = {
                "status": association.get("status", "unknown"),
                "details_omitted": True, "full_sha256": _sha(association)}
    if len(_json(result).encode()) > SUMMARY_LIMIT:
        for component in result.get("components", {}).values():
            coverage = component.get("coverage")
            component["coverage"] = {"preview_omitted": True, "sha256": _sha(coverage)}
        for asset in result.get("assets", []):
            changes = asset.get("associated_config_changes")
            asset["associated_config_changes"] = {"preview_omitted": True, "sha256": _sha(changes)}
    if len(_json(result).encode()) > SUMMARY_LIMIT:
        # Preserve every asset identity/verdict and the first severe source witness.
        result["first_source_witness"] = next(iter(result["findings_shown"]), None)
        result["findings_shown"] = []
        result["findings_omitted"] = total
        if "assets" in result:
            result["first_source_witness"] = next((a["first_source_witness"] for a in result["assets"]
                                                    if a["status"] == "changed" and a["first_source_witness"]), None)
            for asset in result["assets"]:
                asset.pop("associated_config_changes", None)
                asset.pop("first_source_witness", None)
            result["asset_detail_omitted"] = True
            result["affected_asset_ids"] = {"count": len(report.get("affected_asset_ids", [])),
                                             "names_shown_in_assets": True}
            for asset in result["assets"]:
                coverage = asset.get("coverage")
                asset["coverage"] = {"status": coverage.get("status", "unknown") if isinstance(coverage, dict) else "unknown",
                                     "preview_omitted": True, "sha256": _sha(coverage)}
                for key in ("criterion_intent", "comparison_basis"):
                    if key in asset and asset[key] == result.get(key):
                        asset[key] = {"same_as_report": "/" + key}
    if len(_json(result).encode()) > SUMMARY_LIMIT:
        raise InputError("Report summary exceeds the bounded interface; use the local CLI with --json")
    return result


def _view(report, response_format, *, local_cli_argv=None):
    if response_format not in ("concise", "detailed"):
        raise InputError("response_format must be concise or detailed")
    if response_format == "detailed" and len(_json(report).encode()) <= RESULT_LIMIT:
        return report
    return summarize(report, local_cli_argv=local_cli_argv)


def _authorize_tree(source, guard):
    """Authorize effective dataset root and all local link targets before checker reads."""
    authorized = guard(_local(source, "Dataset"))
    root = guard(str(dataset_root(authorized)))
    if not Path(root).is_dir():
        return root
    pending, seen, count = [(root, 0)], set(), 0
    while pending:
        directory, depth = pending.pop()
        if depth > 32:
            raise InputError("Dataset authorization exceeds depth 32; select a smaller explicit dataset")
        identity = Path(directory).stat()
        key = (identity.st_dev, identity.st_ino)
        if key in seen:
            continue
        seen.add(key)
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > MAX_LOCAL_ENTRIES:
                    raise InputError("Dataset authorization exceeds 20000 entries; select a smaller explicit dataset")
                checked = guard(entry.path)
                if Path(checked).is_dir():
                    pending.append((checked, depth + 1))
    return root


def _authorize_upgrade(manifest_path, guard):
    path = guard(_local(manifest_path, "Manifest"))
    manifest = _read_json(path)
    ui = upgrade_impact
    ui._object(manifest, {"schema_version", "kind", "contract", "assets"}, "Impact manifest")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["kind"] != "native_upgrade_impact" or type(manifest["assets"]) is not list
            or not 1 <= len(manifest["assets"]) <= ui.MAX_ASSETS):
        raise InputError("Unsupported impact manifest schema or asset count")
    references = [manifest["contract"]]
    for asset in manifest["assets"]:
        ui._object(asset, {"asset_id", "reference", "candidate"}, "User asset")
        ui._text(asset["asset_id"], "Asset identity")
        for side in ("reference", "candidate"):
            snapshot = asset[side]
            ui._object(snapshot, {"consumer", "revision", "source_files", "configs", "capture"}, "Snapshot")
            for field in ("source_files", "configs"):
                files = snapshot[field]
                if (type(files) is not dict or not (0 if field == "source_files" else 1)
                        <= len(files) <= ui.MAX_FILES):
                    raise InputError("Provide a bounded named snapshot file mapping")
                references.extend(files.values())
            references.append(snapshot["capture"])
    base = Path(path).parent
    # Complete the entire authorization pass before the assessment reads a contract,
    # config, native source or capture, even if an earlier asset is healthy.
    for reference in references:
        ui._object(reference, {"path", "sha256"}, "File identity")
        given = Path(_local(ui._text(reference["path"], "File path"), "Referenced artifact"))
        ui._sha(reference["sha256"])
        guard(str(given if given.is_absolute() else base / given))
    return path, _sha(manifest)


def register_mcp_tools(server, *, guard, allowed, annotations, allow_hub_datasets=False):
    """Register bounded evidence/readiness tools; guard remains the read authority.

    No arbitrary health config, training code, executable, output path or extra
    roots are accepted by these MCP schemas. The operator and request must both
    authorize public dataset fetches.
    """
    _strict_flag(allow_hub_datasets, "allow_hub_datasets")
    if not allowed:
        raise InputError("MCP tools require configured local read roots")
    health_annotations = annotations.model_copy(update={"openWorldHint": allow_hub_datasets})

    @server.tool(annotations=annotations)
    def choose_workflow(goal: Literal["dataset", "training", "runtime", "upgrade", "conversion", "deploy", "recovery", "sim"],
                        framework: Literal["unspecified", "groot", "lerobot", "openpi", "isaaclab", "other"] = "unspecified",
                        dataset: str | None = None, contract: str | None = None,
                        criterion_sha256: str | None = None, capture: str | None = None,
                        recording: str | None = None, manifest: str | None = None,
                        original_recording: str | None = None, converter_source: str | None = None,
                        converter_config: str | None = None, robot_command: str | None = None,
                        robot_config: str | None = None, openpi_config: str | None = None,
                        openpi_source_root: str | None = None,
                        checkpoint: str | None = None, pipeline_contract: str | None = None,
                        runtime_source_root: str | None = None,
                        lerobot_version: str | None = None) -> dict[str, Any]:
        """Start here when unsure which evidence you need. Explains checks available now versus retained-reference requirements and future capture. Inspects explicitly authorized path presence and an optional top-level upgrade manifest only. No assessment, fetching, code execution, writes or invented baseline; assessment_status is not_run."""
        from .workflow_readiness import choose_workflow as plan_workflow

        if lerobot_version is not None and (type(lerobot_version) is not str or not lerobot_version
                or len(lerobot_version) > 64 or not all(char.isprintable() for char in lerobot_version)):
            raise InputError('lerobot_version must be a nonempty version label of at most 64 printable characters.')
        if lerobot_version is not None and not (robot_command or robot_config):
            raise InputError('lerobot_version needs the actual robot_command or robot_config.')
        result = plan_workflow(goal, framework=framework, dataset=dataset, contract=contract,
            criterion_sha256=criterion_sha256, capture=capture, recording=recording,
            manifest=manifest, original_recording=original_recording,
            converter_source=converter_source, converter_config=converter_config,
            authorize=guard, allow_hub_datasets=allow_hub_datasets, robot_command=robot_command,
            robot_config=robot_config, openpi_config=openpi_config, openpi_source_root=openpi_source_root,
            checkpoint=checkpoint, pipeline_contract=pipeline_contract, runtime_source_root=runtime_source_root)
        for option in result.get('can_do_now', []):
            request = option.get('mcp_request') or {}
            if request.get('name') == 'preflight_check' and lerobot_version is not None:
                request['arguments']['lerobot_version'] = lerobot_version
                option['local_cli_argv'].extend(['--lerobot-version', lerobot_version])
        return result

    @server.tool(annotations=health_annotations)
    def dataset_health_card(source: str, consumer: str = "unspecified", fetch: Flag = False,
                            revision: str = "main", video_headers: Flag = True,
                            max_rows: RowBound = 200000, max_files: FileBound = 1024,
                            max_bytes: ByteBound = 64 * 1024 * 1024, units: OptionalFlag = None,
                            commanded_reach: Flag = False, groot_contract: Flag = False,
                            response_format: Literal["concise", "detailed"] = "concise") -> dict[str, Any]:
        """Bounded dataset-only health card. Units PASS identifies consistent recorded units, not robot compatibility: concise output names the modes and states no deployment comparison. For deploy or OpenPI/GR00T questions, use preflight_check with actual consumer selectors. Optional public fetch requires operator and request permission. No writes or checkpoint/deployment/config paths accepted."""
        _strict_flag(fetch, "fetch")
        config = _health_config({"video_headers": video_headers, "max_rows": max_rows, "max_files": max_files,
                                 "max_bytes": max_bytes, "units": units, "commanded_reach": commanded_reach,
                                 "groot_contract": groot_contract})
        if type(source) is not str or not source:
            raise InputError("An explicit dataset source is required")
        if source.startswith("hf://"):
            if not allow_hub_datasets or not fetch:
                raise InputError("Public health cards require operator allow_hub_datasets and explicit fetch=true")
            checked = source
        else:
            if fetch:
                raise InputError("fetch=true is reserved for an explicit public hf:// dataset")
            checked = _authorize_tree(source, guard)
        local_cli = ["roboclue", "health-card", checked, "--consumer", consumer, "--revision", revision]
        if fetch:
            local_cli.append("--fetch")
        for key, value in config.items():
            if value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if type(value) is bool:
                local_cli.append(flag if value else "--no-" + key.replace("_", "-"))
            else:
                local_cli.extend([flag, str(value)])
        local_cli.extend(["--output", "NEW.json", "--artifacts", "NEW_DIR"])
        return _view(health_cards.health_card(checked, consumer=consumer, config=config, fetch=fetch, revision=revision),
                     response_format, local_cli_argv=local_cli)

    @server.tool(annotations=annotations)
    def triage_flight_recording(recording: str, max_command_gap_s: Seconds = None,
                               max_feedback_gap_s: Seconds = None, max_camera_gap_s: Seconds = None,
                               max_latency_s: Seconds = None, tracking_tolerance: Tracking = None,
                               response_format: Literal["concise", "detailed"] = "concise") -> dict[str, Any]:
        """Triage a completed recorded software-command/feedback file, exact event IDs and declared clocks/units only. No feedback or synchronized clocks means UNKNOWN; no hardware control or causal mechanical diagnosis."""
        checked = guard(_local(recording, "Recording"))
        report = flight_recorder.triage_flight_recording(checked, max_command_gap_s=max_command_gap_s,
            max_feedback_gap_s=max_feedback_gap_s, max_camera_gap_s=max_camera_gap_s,
            max_latency_s=max_latency_s, tracking_tolerance=tracking_tolerance)
        return _view(report, response_format)

    @server.tool(annotations=annotations)
    def assess_upgrade_impact(manifest: str,
                              response_format: Literal["concise", "detailed"] = "concise") -> dict[str, Any]:
        """Assess named assets against one retained contract. Every source/config/capture/contract path must be under configured read roots; unsupported coverage stays UNKNOWN. No migration execution, writes or outgoing alerts."""
        checked, digest = _authorize_upgrade(manifest, guard)
        report = upgrade_impact.assess_upgrade_impact(checked)
        if _sha(_read_json(checked)) != digest:
            raise InputError("Upgrade manifest changed after path authorization")
        return _view(report, response_format,
                     local_cli_argv=["roboclue", "upgrade-impact", checked, "--output", "NEW.json", "--artifacts", "NEW_DIR"])

    @server.tool(annotations=annotations)
    def check_training_targets(contract: str, criterion_sha256: str, capture: str,
                               response_format: Literal["concise", "detailed"] = "concise") -> dict[str, Any]:
        """Offline target/mask capture check using an independently retained contract pin. Authorizes both local inputs before reading either. No training import/launch or live attachment; use installed Python API for that."""
        checked_contract = guard(_local(contract, "Contract"))
        checked_capture = guard(_local(capture, "Capture"))
        return _view(_check_training_capture(checked_contract, criterion_sha256, checked_capture), response_format)

    # Preserve original strict arguments and refuse unknown keys even when this
    # registrar is used directly, outside create_server's final schema pass.
    names = {"choose_workflow", "dataset_health_card", "triage_flight_recording", "assess_upgrade_impact", "check_training_targets"}
    for tool in server._tool_manager.list_tools():
        if tool.name in names:
            model = tool.fn_metadata.arg_model
            model.model_config["extra"] = "forbid"
            model.model_rebuild(force=True)
            tool.parameters = model.model_json_schema()


def _new_destination(path):
    destination = Path(_local(path, "Output")).expanduser()
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise InputError("Output requires a new path with an existing parent")
    return destination


def _parser():
    parser = argparse.ArgumentParser(description="RoboClue bounded evidence tools and guided first use")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start", add_help=False, help="Choose checks for the evidence you already have")
    commands.add_parser("demo-native", add_help=False, help="Replay genuine saved native target captures on CPU")
    commands.add_parser("preflight", add_help=False, help="Dataset preflight (same checker and options as policy-doctor preflight)")
    commands.add_parser("monitor", add_help=False, help="Recurring bounded dataset inventory scans and local cards")
    commands.add_parser("upgrade-watch", add_help=False, help="Owned-asset inventory, upstream watches and local alerts")
    health = commands.add_parser("health-card", help="Bounded dataset evidence card; optional SVG/CI artifacts")
    health.add_argument("source")
    health.add_argument("--consumer", default="unspecified")
    health.add_argument("--groot-version", choices=("n1.5", "n1.7"), help="Explicit GR00T dataset-contract version; enables that contract unless explicitly disabled")
    health.add_argument("--config", help="JSON object of existing allowlisted health-card preflight options")
    health.add_argument("--fetch", action="store_true")
    health.add_argument("--revision", default="main")
    health.add_argument("--max-rows", type=int)
    health.add_argument("--max-files", type=int)
    health.add_argument("--max-bytes", type=int)
    health.add_argument("--video-headers", action=argparse.BooleanOptionalAction, default=None)
    health.add_argument("--units", action=argparse.BooleanOptionalAction, default=None)
    health.add_argument("--commanded-reach", action=argparse.BooleanOptionalAction, default=None)
    health.add_argument("--groot-contract", action=argparse.BooleanOptionalAction, default=None)
    health.add_argument("--artifacts", help="Explicit new directory for health-card.json, badge.svg and summary.txt")
    flight = commands.add_parser("flight-triage", help="Recorded command/feedback/event triage")
    flight.add_argument("recording")
    for key in ("command-gap", "feedback-gap", "camera-gap", "latency"):
        flight.add_argument("--max-" + key + "-s", type=float)
    flight.add_argument("--tracking-tolerance", metavar="JSON_FILE_OR_OBJECT",
                        help="JSON file path or inline JSON object mapping joint to {unit, absolute} tolerance")
    upgrade = commands.add_parser("upgrade-impact", help="Retained native criterion against named assets")
    upgrade.add_argument("manifest")
    upgrade.add_argument("--artifacts", help="Explicit new directory for impact.json and alerts.md")
    training = commands.add_parser("check-training-targets", help="Offline retained-contract target/mask check")
    training.add_argument("--contract", required=True)
    training.add_argument("--criterion-sha256", required=True)
    training.add_argument("--capture", required=True)
    for command in (health, flight, upgrade, training):
        command.add_argument("--output", help="Explicit new JSON report file; existing paths are refused")
        command.add_argument("--json", action="store_true", help="Print the full unmodified evidence report as JSON")
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["preflight"]:
        from .cli import main as preflight_main

        return preflight_main(arguments)
    if arguments[:1] == ["start"]:
        from .workflow_readiness import main as readiness_main

        return readiness_main(arguments[1:])
    if arguments[:1] == ["demo-native"]:
        from .native_training_demo import main as demo_main

        return demo_main(arguments[1:])
    if arguments[:1] == ["monitor"]:
        from .health_monitor import main as monitor_main

        return monitor_main(arguments[1:])
    if arguments[:1] == ["upgrade-watch"]:
        from .upgrade_watch import main as watch_main

        return watch_main(arguments[1:])
    args = _parser().parse_args(arguments)
    try:
        output = _new_destination(args.output) if args.output else None
        artifacts = _new_destination(args.artifacts) if getattr(args, "artifacts", None) else None
        if output is not None and artifacts is not None and (output == artifacts or artifacts in output.parents):
            raise InputError("--output and --artifacts must use separate new destinations")
        if args.command == "health-card":
            config = _health_config(_read_json(args.config)) if args.config else {}
            for key in ("max_rows", "max_files", "max_bytes", "video_headers", "units", "commanded_reach", "groot_contract", "groot_version"):
                if getattr(args, key) is not None:
                    config[key] = getattr(args, key)
            report = health_cards.health_card(args.source, consumer=args.consumer, config=_health_config(config),
                                               fetch=args.fetch, revision=args.revision)
        elif args.command == "flight-triage":
            report = flight_recorder.triage_flight_recording(args.recording,
                max_command_gap_s=args.max_command_gap_s, max_feedback_gap_s=args.max_feedback_gap_s,
                max_camera_gap_s=args.max_camera_gap_s, max_latency_s=args.max_latency_s,
                tracking_tolerance=_tracking_tolerance_arg(args.tracking_tolerance)
                if args.tracking_tolerance is not None else None)
        elif args.command == "upgrade-impact":
            report = upgrade_impact.assess_upgrade_impact(args.manifest)
        else:
            report = _check_training_capture(args.contract, args.criterion_sha256, args.capture)
        outputs = {}
        if output is not None:
            with output.open("x", encoding="utf8") as stream:
                stream.write(_json(report) + "\n")
            outputs["report"] = str(output)
        if artifacts is not None:
            outputs.update(health_cards.write_health_card(report, artifacts) if args.command == "health-card"
                           else upgrade_impact.write_upgrade_alerts(report, artifacts))
        if args.json:
            print(_json(report))
        else:
            summary = summarize(report)
            print(f"{report['status'].upper()}: {report.get('code', report['kind'])}")
            if "user_view" in summary:
                print("Read this first: " + _json(summary["user_view"]))
            print("Counts: " + _json(summary["counts"]))
            for finding in summary["findings_shown"]:
                print(_json(finding))
            if summary.get("first_difference"):
                print("First source witness: " + _json(summary["first_difference"]))
            for asset in summary.get("assets", []):
                print(_json(asset))
            for key in ("criterion_intent", "comparison_basis"):
                if key in summary:
                    print(key + ": " + _json(summary[key]))
            print(summary.get("next", "Inspect the complete JSON evidence and rerun the same declared criteria."))
            if outputs:
                print("Outputs: " + _json(outputs))
            print(LIMITS)
        return {"pass": 0, "warning": 0, "healthy": 0, "error": 1, "changed": 1, "unknown": 3}[report["status"]]
    except (InputError, OSError, ValueError) as error:
        print(_json({"error": {"code": "invalid_input", "message": str(error)}}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
