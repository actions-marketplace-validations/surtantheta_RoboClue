"""Handoff view of a video-reference recovery: where the proposed metadata is, what changed, how to get all of it.

Read from bounded no-follow retained files, with content binding to a single-case replay or fresh batch snapshot
revalidation. It lives outside the modules a retained case's verifier seal binds, so existing seals stay unchanged.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from .artifacts import InputError

PATCH_PREVIEW_CELLS = 12
PATCH_BYTES = 96 * 1024  # the complete patch on an MCP request; leaves room under the 192 KiB response bound
MAX_CASE_CELLS = 16
OVERLAY_SCOPE = ("Metadata overlay, not a complete dataset: proposed_metadata/ holds only the changed native metadata "
                 "file(s). A complete dataset is the pinned revision's unchanged files (data/, videos/ and the other "
                 "meta/ files) with each listed dataset_path replaced by its proposed copy; RoboClue does not assemble "
                 "or verify that dataset.")
VALIDATION_SCOPE = {
    "replay": ("reproduce-video-references re-derives this proposal offline from the retained evidence: internal "
               "consistency only, with no decoder, network or native loader."),
    "native_consumer": ("Not run. Load the assembled dataset with the intended native consumer (for example LeRobot) "
                        "and decode the cited episodes' frames before training.")}
CASE_OVERLAY_SCOPE = ("Per-episode proposal, not a complete dataset: corrected_metadata.parquet changes only this "
                      "episode's cells in one native metadata file; other episodes' proposals for the same file are not "
                      "merged here (a batch from recover-video-references merges them) and the dataset's other files "
                      "are not included.")
CASE_VALIDATION_SCOPE = {
    "replay": "This replay re-derives the retained report and derived copy offline: internal consistency only.",
    "native_consumer": ("Not run. Load a complete dataset that uses the derived metadata with the intended native "
                        "consumer (for example LeRobot) and decode this episode's frames.")}
_CELL_KEYS = ("episode_index", "path", "row", "column", "old", "new")


def _cells(proposals):
    return [{key: cell.get(key) for key in _CELL_KEYS} for cell in proposals]


def _representative(proposals, limit):
    """Every cell of the first episode of each (file, camera) group, then the rest in order, up to limit."""
    def group(cell):
        return cell.get("path"), str(cell.get("column", "")).rsplit("/", 1)[0]

    first = {}
    for cell in proposals:
        first.setdefault(group(cell), cell.get("episode_index"))
    leading = [i for i, cell in enumerate(proposals) if first[group(cell)] == cell.get("episode_index")]
    chosen = set(leading[:limit])
    chosen |= set([i for i in range(len(proposals)) if i not in chosen][:limit - len(chosen)])
    return [proposals[i] for i in sorted(chosen)]


def _retained_patch(root: Path, result: dict):
    """Bind a bounded presentation snapshot to a complete offline batch replay.

    The sealed batch API's result carries counts, not content digests. A mutable replacement manifest is therefore
    not a binding to that result. Revalidate the snapshot with the unchanged kernel before displaying its cells.
    """
    from . import bulk_video_recovery as engine
    from . import video_recovery as vr
    from .bulk_workflow import reproduce_video_references

    snapshot, total = {}, 0

    def read(name, bound=vr.MAX_METADATA):
        nonlocal total
        raw = engine._seal_read(root, name, bound)
        total += len(raw)
        if total > vr.MAX_RESPONSE:
            raise InputError("Batch handoff snapshot exceeds its byte bound")
        snapshot[name] = raw
        return raw

    manifest_raw = read("BATCH_MANIFEST.json", engine.MAX_BATCH_MANIFEST)
    manifest = vr._json(manifest_raw)
    ready = vr._json(read("BATCH_READY.json", 4096))
    if not isinstance(ready, dict) or ready.get("manifest_sha256") != vr._sha(manifest_raw):
        raise InputError("Batch handoff seal changed after replay")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    raw = read("proposals.json")
    if not isinstance(files, dict) or vr._sha(raw) != files.get("proposals.json"):
        raise InputError("Batch proposals differ from their retained seal")
    proposals = vr._json(raw)
    if type(proposals) is not list or any(type(cell) is not dict for cell in proposals):
        raise InputError("Malformed retained batch proposals")
    report_raw = read("report.json")
    if vr._sha(report_raw) != files.get("report.json"):
        raise InputError("Batch handoff report differs from its retained seal")
    for name, digest in files.items():
        if name.startswith(("original_metadata/", "proposed_metadata/")):
            if vr._sha(read(name)) != digest:
                raise InputError("Batch handoff native metadata differs from its retained seal")
    # A fresh replay checks the proposal union against all full case replays and reconstructs the merged native
    # metadata. This also catches an attacker who updates READY along with a replacement manifest/proposal.
    replay = reproduce_video_references(root)
    operational = {"status", "execution_status", "next_actions", "handoff", "patch"}
    for key in result.keys() & replay.keys() - operational:
        if result[key] != replay[key]:
            raise InputError(f"Batch handoff {key} differs from the replayed snapshot")
    if replay.get("status") != "pass" or replay.get("reproduction_matches") is not True:
        raise InputError("Batch handoff snapshot did not reproduce")
    if replay.get("proposed_cells") != len(proposals):
        raise InputError("Batch handoff cell count differs from the replayed snapshot")
    for name, before in snapshot.items():
        if engine._seal_read(root, name, max(len(before), 1)) != before:
            raise InputError("Batch handoff evidence changed during replay")
    verification = {"manifest_sha256": vr._sha(manifest_raw), "report_sha256": vr._sha(report_raw),
                    "proposals_sha256": files["proposals.json"],
                    "scope": "Complete offline batch replay with an unchanged bounded no-follow handoff snapshot"}
    prior = result.get("handoff", {}).get("verification")
    if isinstance(prior, dict) and any(prior.get(key) != verification[key]
                                       for key in ("manifest_sha256", "report_sha256", "proposals_sha256")):
        raise InputError("Batch handoff differs from its earlier verified snapshot")
    return proposals, files, verification


def _patch(proposals, files, limit):
    """The complete retained patch, cut at a byte bound only when one is given (MCP)."""
    cells, size = [], 0
    for cell in _cells(proposals):
        size += len(json.dumps(cell, separators=(",", ":"))) + 1
        if limit is not None and size > limit:
            break
        cells.append(cell)
    return {"cells": cells, "cell_count": len(proposals), "truncated": len(cells) < len(proposals),
            "sha256": files.get("proposals.json"), "source": "the retained proposals.json"}


def batch_handoff(result: dict, *, include_patch: bool = False, patch_bytes: int | None = None) -> dict:
    """Add the proposed copies, a bounded before->after preview and the complete patch's location to a batch result."""
    if type(include_patch) is not bool:
        raise InputError("include_patch must be a strict boolean")
    root = Path(result["directory"])
    proposals, files, verification = _retained_patch(root, result)
    paths = sorted({str(cell.get("path")) for cell in proposals})
    listed = []
    for path in paths[:16]:
        proposed, original = "proposed_metadata/" + path, "original_metadata/" + path
        listed.append({"dataset_path": path,
                       "proposed_copy": str(root / proposed) if proposed in files else None,
                       "proposed_sha256": files.get(proposed),
                       "original_copy": str(root / original) if original in files else None,
                       "original_sha256": files.get(original),
                       "cells": sum(str(cell.get("path")) == path for cell in proposals)})
    preview = _representative(proposals, PATCH_PREVIEW_CELLS)
    result = dict(result)
    result["handoff"] = {
        "proposed_metadata_files": listed, "proposed_metadata_file_count": len(paths),
        "proposed_metadata_files_truncated": len(paths) > 16,
        "patch_preview": _cells(preview), "patch_preview_omitted": len(proposals) - len(preview),
        "complete_patch": {"path": str(root / "proposals.json"), "sha256": files.get("proposals.json"),
                           "cells": len(proposals),
                           "command": ["policy-doctor", "reproduce-video-references", str(root), "--patch", "--json"],
                           "mcp": "reproduce_dataset_video_references or reproduce_incident_case with include_patch=true"},
        "overlay_scope": OVERLAY_SCOPE, "validation_scope": dict(VALIDATION_SCOPE),
        "verification": verification}
    if proposals and result.get("proposal_complete"):
        result["next_actions"] = list(result.get("next_actions", [])) + [
            "Review the patch preview and the complete patch (" + str(root / "proposals.json") + "). To use it, start "
            "from a complete copy of the pinned revision and replace each listed dataset_path with its proposed copy; "
            "then load that dataset with the intended native consumer. Replay alone is not native validation."]
    if include_patch:
        result["patch"] = _patch(proposals, files, patch_bytes)
    return result


def case_handoff(result: dict, directory) -> dict:
    """Add the reproduced patch values and the derived copy's path to a single-episode replay, when it reproduced."""
    from . import video_recovery as vr

    result = dict(result)
    root = Path(directory).expanduser().resolve()
    cells, derived = [], None
    if result.get("status") == "pass" and result.get("reproduction_matches") is True:
        from . import bulk_video_recovery as engine

        # Reopening a mutable manifest does not bind it to the replay that just passed. The replay result binds the
        # canonical complete report, whose patch and original metadata hash let us re-derive the same native copy.
        # Every new presentation read uses the existing bounded no-follow reader, including root ancestors.
        snapshot = {}

        def read(name, bound=vr.MAX_METADATA):
            raw = engine._seal_read(root, name, bound)
            snapshot[name] = raw
            return raw

        manifest_raw = read("manifest.json")
        manifest = vr._json(manifest_raw)
        ready = vr._json(read("READY.json", 4096))
        if (type(manifest) is not dict or type(manifest.get("files")) is not dict
                or type(ready) is not dict or ready.get("manifest_sha256") != vr._sha(manifest_raw)):
            raise InputError("Recovery handoff seal changed after replay")
        files = manifest["files"]
        report_raw = read("report.json")
        report = vr._json(report_raw)
        report_sha = vr._sha(vr._encode(report))
        if (type(report) is not dict or files.get("report.json") != vr._sha(report_raw)
                or any(result.get(key) != report_sha for key in
                       ("recorded_report_sha256", "current_report_sha256"))):
            raise InputError("Recovery handoff report differs from the replayed content")
        plan_raw = read("repair_plan.json")
        plan = vr._json(plan_raw)
        if type(plan) is not list or any(type(cell) is not dict for cell in plan):
            raise InputError("Malformed retained repair plan")
        if vr._sha(plan_raw) != files.get("repair_plan.json") or plan != report.get("patch"):
            raise InputError("Repair plan differs from the replayed content")
        cells = plan
        if cells:
            evidence = report.get("evidence")
            metadata_sha = evidence.get("original_metadata_sha256") if isinstance(evidence, dict) else None
            candidates = [name for name, digest in files.items()
                          if name.startswith("payload/") and digest == metadata_sha]
            if not candidates or "corrected_metadata.parquet" not in files:
                raise InputError("Replayed native metadata is missing from the handoff")
            original = read(candidates[0])
            if vr._sha(original) != metadata_sha:
                raise InputError("Native metadata differs from the replayed content")
            expected = vr._corrected_metadata({"raw": original}, cells)
            corrected = read("corrected_metadata.parquet")
            if corrected != expected or vr._sha(corrected) != files["corrected_metadata.parquet"]:
                raise InputError("Derived metadata differs from the replayed content")
            derived = {"path": str(root / "corrected_metadata.parquet"), "dataset_path": cells[0].get("path"),
                       "sha256": vr._sha(corrected),
                       "content": "A complete copy of that one native metadata file with only the patch cells changed."}
        for name, raw in snapshot.items():
            if engine._seal_read(root, name, max(len(raw), 1)) != raw:
                raise InputError("Recovery handoff evidence changed while reading")
    result.update(patch=cells[:MAX_CASE_CELLS], patch_cells=len(cells), patch_truncated=len(cells) > MAX_CASE_CELLS,
                  derived_metadata=derived, overlay_scope=CASE_OVERLAY_SCOPE,
                  validation_scope=dict(CASE_VALIDATION_SCOPE))
    return result


def _cell_line(cell):
    return (f"  episode {json.dumps(cell.get('episode_index'))} {json.dumps(cell.get('path'))} row "
            f"{json.dumps(cell.get('row'))} {json.dumps(cell.get('column'))}: {json.dumps(cell.get('old'))} -> "
            f"{json.dumps(cell.get('new'))}")


def handoff_lines(result: dict) -> list[str]:
    handoff = result.get("handoff")
    if not isinstance(handoff, dict):
        return []
    lines = []
    for item in handoff.get("proposed_metadata_files", []):
        lines.append("Proposed metadata copy (overlay; only changed files): " + json.dumps(item.get("dataset_path"))
                     + " -> " + json.dumps(item.get("proposed_copy")) + f" ({item.get('cells')} cells)")
    preview = handoff.get("patch_preview", [])
    complete = handoff.get("complete_patch") or {}
    lines.append(f"Patch preview ({len(preview)} of {complete.get('cells', len(preview))} cells):")
    lines.extend(_cell_line(cell) for cell in preview)
    lines.append("Complete patch: " + json.dumps(complete.get("path")) + f" ({complete.get('cells')} cells, sha256 "
                 + str(complete.get("sha256")) + "); print it with "
                 + shlex.join(complete.get("command", [])[:-1] or ["policy-doctor"]))
    lines.append(str(handoff.get("overlay_scope", "")))
    validation = handoff.get("validation_scope") or {}
    lines.append("Replay: " + str(validation.get("replay", "")))
    lines.append("Native consumer validation: " + str(validation.get("native_consumer", "")))
    patch = result.get("patch")
    if isinstance(patch, dict):
        lines.append(f"Complete patch ({patch.get('cell_count')} cells):" if not patch.get("truncated") else
                     f"Patch ({len(patch.get('cells', []))} of {patch.get('cell_count')} cells; cut at the response "
                     "bound, the complete list is the retained file above):")
        lines.extend(_cell_line(cell) for cell in patch.get("cells", []))
    return lines
