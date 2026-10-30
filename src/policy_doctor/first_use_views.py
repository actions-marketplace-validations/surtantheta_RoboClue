"""Small read-only views of complete reports; these never create new verdicts.

At most eight rows, two event citations per row and 256 characters per string
keep either view below 32 KiB of ASCII JSON. Counts describe available evidence,
not physical incidents, whole-dataset coverage or successful repairs.
"""
from __future__ import annotations

import json
import math

from .artifacts import InputError

MAX_VIEW_BYTES = 32768
_STATES = ("error", "warning", "unknown", "pass")
_SOFTWARE = ("policy_to_request", "request_to_return")


def _rows(value):
    if type(value) is not int or not 1 <= value <= 8:
        raise InputError("max_rows must be an integer in 1..8")
    return value


def _value(value):
    if type(value) is str:
        bounded = value[:256]
        while len(json.dumps(bounded, ensure_ascii=True)) > 378:
            bounded = bounded[:-1]
        return bounded[:255] + "…" if bounded != value else bounded
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float) and abs(value) <= 1e100 and math.isfinite(value):
        return value
    return None


def _count(value):
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _finish(view):
    # Guard the advertised output bound if the schema grows later.
    if len(json.dumps(view, ensure_ascii=True, allow_nan=False).encode()) > MAX_VIEW_BYTES:
        raise InputError("Compact view exceeds its byte bound")
    return view


def flight_view(report, *, max_rows=4):
    """Expose declared software comparisons beside unverified physical feedback.

    Only explicit comparison_boundary annotations from triage identify software
    boundaries. Older reports retain UNKNOWN scope even if their IDs look like
    runtime events. The original report, counts and criterion remain untouched.
    """
    limit = _rows(max_rows)
    if report.get("kind") != "flight_recording_triage" or type(report.get("findings")) is not list:
        raise InputError("flight_view requires a complete flight_recording_triage report")
    findings = report["findings"]
    software = [f for f in findings if f.get("code") == "tracking"
                and f.get("details", {}).get("comparison_boundary") in _SOFTWARE]
    software.sort(key=lambda f: _STATES.index(f["status"]) if f.get("status") in _STATES else 2)
    rows = []
    for finding in software[:limit]:
        details = finding["details"]
        row = {"boundary": details["comparison_boundary"], "status": _value(finding.get("status")),
               "event_ids": [_value(e) for e in finding.get("event_ids", [])[:2]]}
        row.update({key: _value(details.get(key)) for key in
                    ("joint", "unit", "command", "observed", "absolute_residual", "threshold")})
        if finding.get("status") == "unknown":
            row["reason"] = _value(finding.get("message"))
        rows.append(row)
    unassociated = [f for f in findings if f.get("code") == "feedback_association" and f.get("status") == "unknown"]
    reasons = ["Software returns and caller declarations do not verify sensor acquisition, motor delivery or physical response."]
    if unassociated:
        reasons.append(f"{len(unassociated)} feedback event(s) have no associated recorded command.")
    if not software:
        reasons.append("No explicitly identified software comparison boundary is available in this report.")
    return _finish({
        "overall_status": _value(report.get("status")),
        "interpretation": "A selected comparison PASS means its declared values meet that criterion. Overall UNKNOWN means other checks lack evidence; physical execution remains unverified.",
        "software_comparisons": rows,
        "software_counts": {s: sum(f.get("status") == s for f in software) for s in ("pass", "warning", "unknown")},
        "software_comparisons_omitted": max(0, len(software) - len(rows)),
        "physical_feedback_association": {
            "status": "unknown", "reasons": reasons,
            "unassociated_feedback_events": len(unassociated),
            "unassociated_observation_returns": sum(f.get("details", {}).get("observation_boundary") == "observation_return" for f in unassociated),
            "event_ids": [_value(e) for f in unassociated[:limit] for e in f.get("event_ids", [])[:1]],
            "event_citations_omitted": max(0, len(unassociated) - limit),
        },
    })


def _byte_accounting(report, card):
    provenance = report.get("provenance", [])
    unique = {json.dumps(p, sort_keys=True): p for p in provenance}
    artifacts, headers = [], []
    for entry in unique.values():
        artifacts.append(_count(entry.get("downloaded_bytes")))
        headers.append(_count(entry.get("video_headers", {}).get("bytes")))
    artifact_bytes = sum(artifacts) if artifacts and None not in artifacts else None
    header_bytes = sum(headers) if headers and None not in headers else None
    config = card.get("request", {}).get("config", {})
    return {
        "configured_download_budget_bytes": _count(config.get("max_bytes", report.get("coverage", {}).get("max_download_bytes"))),
        "metadata_and_parquet_downloaded_bytes": artifact_bytes,
        "video_header_range_bytes": header_bytes,
        "combined_recorded_bytes": artifact_bytes + header_bytes if artifact_bytes is not None and header_bytes is not None else None,
        "scope": "Recorded intake and header-range bytes only; null means unavailable. Partial headers do not verify full-video hashes. No full-video decoding or repair is established.",
    }


def _video_view(components):
    part = components.get("video_index", {})
    coverage = part.get("coverage", {})
    declared = _count(coverage.get("declared_video_features"))
    status = part.get("status", "unknown")
    not_applicable = status == "pass" and coverage.get("status") == "not_applicable" and declared == 0
    return {
        "display_status": "N/A" if not_applicable else status.upper() if status in _STATES else "UNKNOWN",
        "reported_status": _value(status), "coverage_status": _value(coverage.get("status")),
        "declared_video_features": declared, "files_checked": _count(coverage.get("files_checked")),
        "interpretation": ("No video features were declared, so video index/header checks are N/A; no video content was validated."
                           if not_applicable else
                           "Reported video status covers only the listed index/header checks; decoded frames remain unassessed."),
    }


def _groot_view(report, card):
    config = card.get("request", {}).get("config", {})
    enabled = config.get("groot_contract")
    enabled = enabled if type(enabled) is bool else None
    selectors = ("groot_modality_config", "groot_config_name", "groot_version")
    supplied = True if any(config.get(key) is not None for key in selectors) else (
        False if all(key in config for key in selectors) else None)
    if supplied is None:
        legacy = report.get("coverage", {}).get("groot_contract_requested")
        supplied = legacy if type(legacy) is bool else None
    modality = config.get("groot_modality_config") is not None if "groot_modality_config" in config else None
    return {
        "enabled": enabled, "recipe_selectors_supplied": supplied,
        "modality_config_supplied": modality, "selected_version": _value(config.get("groot_version")),
        "reported_status": _value(report["components"].get("groot_contract", {}).get("status", "not_assessed")),
        "interpretation": "Enabled is the card's effective setting. Recipe selectors are a modality config, config name or version. Legacy groot_contract_requested refers to selectors; false does not mean disabled. Null means selection metadata is unavailable.",
    }


def health_view(report, *, max_rows=4):
    """Bounded maintainer issue table from a full health card or preflight report.

    Duplicate episode/camera/file/time evidence across components counts once.
    The table is an available-evidence list, never a count of distinct incidents.
    """
    limit = _rows(max_rows)
    card = report if report.get("kind") == "dataset_health_card" else {}
    data = card.get("report", report)
    if data.get("kind") != "dataset_preflight" or type(data.get("components")) is not dict:
        raise InputError("health_view requires a complete dataset health card or preflight report")
    components = data["components"]
    findings = [f for component in components.values() for f in component.get("findings", [])]
    if not findings:
        findings = data.get("findings", [])
    source = card.get("request", {}).get("source", data.get("source"))
    revision = card.get("source_revision", {}).get("value")
    issues = {}
    for finding in findings:
        if finding.get("status") not in ("error", "warning", "unknown"):
            continue
        episodes = finding.get("episodes") if finding.get("code") == "video.interval_past_end" else None
        for episode in episodes or [None]:
            if isinstance(episode, dict):
                row = {"code": _value(finding.get("code")), "status": _value(finding.get("status")),
                       "episode": _value(episode.get("episode_index")), "camera": _value(episode.get("video_key")),
                       "source_file": _value(episode.get("file")),
                       "conflicting_values": {k: _value(episode.get(k)) for k in
                                              ("episode_from_s", "episode_last_frame_s", "file_last_frame_s")},
                       "action": "Inspect the original cited clip and episode reference; verify any neighboring-file hypothesis separately, retain a corrected candidate, then rerun. No repair was applied."}
                # Full, untruncated identity prevents unrelated long paths from merging.
                key = json.dumps({"code": finding.get("code"), "episode": episode.get("episode_index"),
                                  "camera": episode.get("video_key"), "file": episode.get("file"),
                                  "times": [episode.get(k) for k in ("episode_from_s", "episode_last_frame_s", "file_last_frame_s")]}, sort_keys=True)
            else:
                message = finding.get("message")
                if message is None:
                    message = finding.get("reason")
                row = {"code": _value(finding.get("code")), "status": _value(finding.get("status")),
                       "episode": None, "camera": None, "source_file": None,
                       "conflicting_values": None, "message": _value(message),
                       "action": "Inspect the full finding and cited evidence; resolve missing evidence or verify a correction before rerunning."}
                key = json.dumps({"code": finding.get("code"), "message": message,
                                  "evidence": finding.get("evidence")}, sort_keys=True)
            previous = issues.get(key)
            if previous is None or _STATES.index(row["status"]) < _STATES.index(previous["status"]):
                issues[key] = row
    ordered = sorted(issues.values(), key=lambda row: _STATES.index(row["status"]))
    truncated = bool(data.get("findings_truncated")) or any(c.get("findings_truncated") for c in components.values())
    return _finish({
        "overall_status": _value(report.get("status")),
        "source": _value(source), "source_revision": _value(revision),
        "issues": ordered[:limit], "available_unique_issues": len(ordered),
        "issues_omitted": max(0, len(ordered) - limit), "source_findings_truncated": truncated,
        "count_scope": "Unique available issue evidence, deduplicated across components; not distinct incidents or whole-dataset defect counts.",
        "video_index": _video_view(components), "groot_contract": _groot_view(data, card),
        "byte_accounting": _byte_accounting(data, card),
        "limits": "Selected checks only. Whole-dataset quality, native consumer compatibility, decoded video, successful repair and physical outcomes remain unassessed.",
    })
