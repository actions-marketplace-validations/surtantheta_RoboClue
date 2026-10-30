"""Bounded adapters for derived native metadata proposals and offline replay."""

from __future__ import annotations

import os
from pathlib import Path

from . import bulk_video_recovery as engine
from . import video_recovery as vr
from .artifacts import InputError


def is_bulk_case(directory):
    """Select a case format using marker presence only; never follow marker links."""
    root = Path(directory)
    batch = any(os.path.lexists(root / name) for name in ("BATCH_MANIFEST.json", "BATCH_READY.json"))
    if batch and os.path.lexists(root / "manifest.json"):
        raise InputError("Ambiguous case: both batch and legacy case markers are present")
    return batch


def _integer(value, name, upper):
    if type(value) is not int or not 1 <= value <= upper:
        raise InputError(f"{name} must be a strict integer in [1,{upper}]")
    return value


def check_report_output(output, batch):
    """A report export must not mutate the batch's sealed evidence inventory."""
    if output is None:
        return
    requested = Path(output).expanduser().absolute()
    if os.path.lexists(requested):
        raise InputError("Report output must be a new file")
    target = requested.resolve()
    if target.is_relative_to(Path(batch).expanduser().resolve()):
        raise InputError("Report output must be outside the retained batch directory")


def _view(report, directory, replay=None):
    result = dict(report if replay is None else replay)
    result.update(kind="dataset_video_reference_batch", batch_kind=report["kind"],
                  batch_schema_version=report["batch_schema_version"], verifier_matches=True, directory=str(Path(directory).absolute()),
                  execution_status="completed" if replay is None else "reproduced",
                  proposal_status=report["status"], proposal_complete=report["proposal_complete"],
                  batch_status=report["status"], upstream_modified=False,
                  scope=report["scope"], request=report["request"])
    for key in ("attempted_associations", "verified_associations", "proposed_cells", "detection_coverage",
                "merge_errors", "episodes", "body_bytes_read", "recovery_body_bytes_read",
                "response_body_upper_bound", "origin_authenticated", "assurance"):
        if key in report:
            result[key] = report[key]
    if report["request"]["route"] == "pinned_public_intake":
        detection = vr._json(engine._seal_read(Path(directory).absolute(), "detection.json", vr.MAX_METADATA))
        result["findings"] = detection["findings"]
        statuses = {finding["status"] for finding in detection["findings"]}
        result["original_status"] = next((s for s in ("error", "unknown", "warning") if s in statuses),
                                         "pass" if report["detection_coverage"] == "complete" else "unknown")
    else:
        result.update(original_status="unknown", findings=[])
    result["next_actions"] = list(report["next_actions"])
    result["replay_command"] = ["policy-doctor", "reproduce-video-references", result["directory"], "--json"]
    return result


def recover_video_references(repo_id, revision, original_revision, destination, *, fetch=False,
                             max_associations=32, max_response_body_mb=512, max_intake_mb=16,
                             max_rows=100000, max_files=128):
    """Explicit public intake; return a new proposal without changing the original dataset."""
    if not isinstance(repo_id, str):
        raise InputError("Require a public dataset ORG/REPO")
    repo_id = repo_id.removeprefix("hf://")
    vr._request_spec(repo_id, revision, original_revision, 0, "probe")
    _integer(max_associations, "max_associations", 256)
    _integer(max_response_body_mb, "max_response_body_mb", 8192)
    _integer(max_intake_mb, "max_intake_mb", 300)
    _integer(max_rows, "max_rows", 200000)
    _integer(max_files, "max_files", 1000)
    if fetch is not True:
        raise InputError("Public batch recovery requires explicit --fetch or operator media authorization")
    report = engine.recover_all_video_references(repo_id, revision, original_revision, destination,
        fetch=True, max_episodes=max_associations, max_body_bytes=max_response_body_mb * 1024**2,
        max_intake_bytes=max_intake_mb * 1024**2, max_rows=max_rows, max_files=max_files)
    return _view(report, destination)


def reproduce_video_references(directory):
    """Verify offline evidence; reproduction success remains separate from proposal completeness."""
    root = vr._path(directory)
    try:
        seal = engine._seal_read(root, "BATCH_MANIFEST.json", engine.MAX_BATCH_MANIFEST)
        manifest = vr._json(seal)
        raw_report = engine._seal_read(root, "report.json", vr.MAX_METADATA)
        if vr._sha(raw_report) != manifest["files"]["report.json"]:
            raise InputError("Batch report differs from its retained seal")
        report = vr._json(raw_report)
        replay = engine.reproduce_bulk_proposal(root)
        result = _view(report, root, replay)
        if report["request"]["route"] == "pinned_public_intake":
            raw = engine._seal_read(root, "detection.json", vr.MAX_METADATA)
            if vr._sha(raw) != manifest["files"]["detection.json"] or vr._json(raw)["findings"] != result["findings"]:
                raise InputError("Batch detector changed during report preparation")
        if engine._seal_read(root, "BATCH_MANIFEST.json", engine.MAX_BATCH_MANIFEST) != seal:
            raise InputError("Batch seal changed during report preparation")
        return result
    except InputError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError("Malformed retained batch evidence cannot establish reproduction") from exc


def concise_bulk_report(result):
    lines = [f"Execution: {result['execution_status'].upper()}",
             f"Original detector: {result['original_status'].upper()}",
             f"Derived proposal: {result['proposal_status'].upper()}; complete: {result['proposal_complete']}",
             f"Associations: {result['verified_associations']}/{result['attempted_associations']} verified; "
             f"native cells: {result['proposed_cells']}; coverage: {result['detection_coverage']}",
             result['scope'], "Original dataset modified: false"]
    for entry in result.get("episodes", []):
        if entry.get("status") != "pass":
            lines.append(f"UNRESOLVED {entry.get('episode_index', '?')}/{entry.get('video_key', '?')}: "
                         + str(entry.get("reason", "Evidence unavailable")))
    lines.extend("Next: " + str(action) for action in result.get("next_actions", []))
    return "\n".join(lines) + "\n"
