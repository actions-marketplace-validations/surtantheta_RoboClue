"""Deterministic, metadata-only windows for the selected saved G1 context.

This does not assess rows, images, native execution or representative quality.
The later workflow must match this plan's metadata/context hashes before reuse.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from .artifacts import InputError
from .consumer_context import PROFILE, REVISION, _authorized_root, _json, load_consumer_numeric_parameters
from .consumer_samples import DATA_TEMPLATE, MAX_EPISODE_ROWS, MAX_METADATA, _integer
from .native_intake import _directory_fd, _stamp

MAX_EPISODES = 4096
HORIZON = 50
RULE = {
    "version": 1,
    "eligible": "metadata length >= 50; metadata order preserved",
    "episode_ranks": "distinct [0, n//3, n-1] within the eligible ordered list",
    "anchors": "distinct [0, (length-50)//2, length-50] for each selected episode",
    "allow_padding": False,
    "selection_scope": "Bounded deterministic selection, not representative quality or coverage of every row.",
}


def _finding(code, status, message):
    return {"code": code, "status": status, "message": message,
            "next": "Inspect the actual saved context and bounded v2.1 episode metadata; preserve the originals."}


def _status(findings, planning):
    return "ERROR" if any(f["status"] == "ERROR" for f in findings) else planning


def _read_metadata(root_fd, relative):
    """Read only the two fixed metadata leaves through the retained root FD."""
    meta_fd = os.open("meta", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    fd = None
    try:
        leaf = relative.removeprefix("meta/")
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=meta_fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA:
            raise InputError("Planning metadata must be bounded regular files.")
        chunks, size = [], 0
        while size <= MAX_METADATA:
            body = os.read(fd, min(65536, MAX_METADATA + 1 - size))
            if not body:
                break
            chunks.append(body)
            size += len(body)
        if size != before.st_size or _stamp(before) != _stamp(os.fstat(fd)):
            raise InputError("Planning metadata changed during reading.")
        current = os.stat(leaf, dir_fd=meta_fd, follow_symlinks=False)
        if _stamp(current) != _stamp(before):
            raise InputError("Planning metadata file identity changed during reading.")
        raw = b"".join(chunks)
        return raw, {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                     "size_bytes": size}, _stamp(before)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(meta_fd)


def _episodes(raw):
    lines = raw.splitlines()
    if not 1 <= len(lines) <= MAX_EPISODES:
        raise InputError("Episode metadata requires one to 4096 bounded records.")
    rows, seen = [], set()
    for position, line in enumerate(lines):
        item = _json(line)
        episode = _integer(item.get("episode_index"), 999999, "Stored episode ID")
        length = _integer(item.get("length"), MAX_EPISODE_ROWS, "Episode length")
        if episode in seen or length == 0:
            raise InputError("Episode metadata has duplicate IDs or nonpositive lengths.")
        seen.add(episode)
        rows.append({"metadata_position": position, "episode_index": episode, "length": length})
    return rows


def plan_consumer_samples(dataset_directory, context, *, read_roots=None):
    """Choose 1–9 stored-ID/physical-anchor selectors from ordinary metadata.

    All explicit roots are authorized before intake. A complete supported context
    is required; independent known capacity ERRORs remain visible even when the
    metadata selection is usable. Metadata failure never returns partial samples.
    """
    root = _authorized_root(dataset_directory, read_roots)
    if type(context) is not dict:
        raise InputError("A generated saved consumer context is required.")
    checkpoint = _authorized_root(context.get("checkpoint_directory"), read_roots)
    findings, readset, captured = [], [], {}
    context_status = "UNKNOWN"
    context_axes, context_artifacts = {}, {}
    interpretation = None
    planning = "UNKNOWN"
    selectors, selected_episodes = [], []
    counts = {"declared_episodes_total": None, "episodes_total": None, "eligible_episodes": None, "short_episodes": None,
              "selected_episodes": 0, "selected_windows": 0}
    root_fd = None

    def capture(relative):
        raw, artifact, stamp = _read_metadata(root_fd, relative)
        captured[relative] = (artifact, stamp)
        readset.append(artifact)
        return raw

    try:
        # This helper returns freshly revalidated status, never caller-edited
        # findings. Its statistical JSON parameters are not dataset row arrays.
        loaded = load_consumer_numeric_parameters(context, read_roots=read_roots)
        context_status = loaded["context_status"]
        context_axes = loaded["context_axes"]
        context_artifacts = context["artifacts"]
        findings.extend(dict(f, cause_scope="saved_native_context")
                        for f in loaded["context_findings"] if f["status"] == "ERROR")
        descriptor = loaded["descriptor"]
        if (type(descriptor) is not dict or descriptor.get("profile") != PROFILE
                or descriptor.get("native_revision") != REVISION
                or descriptor.get("numeric_context_complete") is not True
                or descriptor["modalities"]["action"]["delta_indices"] != list(range(HORIZON))
                or descriptor["modalities"]["state"]["delta_indices"] != [0]):
            raise InputError("A complete supported G1 numeric context with 50 offsets is required.")
        interpretation = descriptor["interpretation_sha256"]
        root_fd = _directory_fd(root)
        root_stamp = _stamp(os.fstat(root_fd))
        info = _json(capture("meta/info.json"))
        if (info.get("codebase_version") != "v2.1" or info.get("data_path") != DATA_TEMPLATE
                or type(info.get("features")) is not dict):
            raise InputError("Only the supported ordinary v2.1 episode layout is planned.")
        chunk_size = _integer(info.get("chunks_size"), 1000000, "chunks_size")
        if chunk_size == 0:
            raise InputError("chunks_size must be positive.")
        rows = _episodes(capture("meta/episodes.jsonl"))
        total = _integer(info.get("total_episodes"), MAX_EPISODES, "total_episodes")
        counts["declared_episodes_total"] = total
        if total != len(rows):
            findings.append(_finding("consumer.plan_episode_count_mismatch", "WARNING",
                "Declared info episode total differs from observed episode records; selection uses only the parsed records."))
        eligible = [row for row in rows if row["length"] >= HORIZON]
        ranks = list(dict.fromkeys([0, len(eligible) // 3, len(eligible) - 1])) if eligible else []
        for rank in ranks:
            row = eligible[rank]
            last = row["length"] - HORIZON
            anchors = list(dict.fromkeys([0, last // 2, last]))
            selected_episodes.append({**row, "eligible_rank": rank, "anchors": anchors})
            selectors.extend({"episode_index": row["episode_index"], "anchor_index": anchor,
                              "allow_padding": False} for anchor in anchors)
        for relative, (artifact, stamp) in captured.items():
            _, current, current_stamp = _read_metadata(root_fd, relative)
            if current != artifact or current_stamp != stamp:
                raise InputError("Planning metadata changed during intake.")
        current_root = _directory_fd(root)
        try:
            if (_stamp(os.fstat(current_root)) != root_stamp
                    or _stamp(os.fstat(root_fd)) != root_stamp):
                raise InputError("Dataset directory identity changed during planning.")
        finally:
            os.close(current_root)
        counts.update(episodes_total=len(rows), eligible_episodes=len(eligible),
                      short_episodes=len(rows) - len(eligible),
                      selected_episodes=len(selected_episodes), selected_windows=len(selectors))
        if not eligible:
            findings.append(_finding("consumer.plan_no_eligible_episode", "UNKNOWN",
                "No bounded metadata episode has the required 50 rows; no windows were invented."))
        else:
            planning = "PASS"
    except (InputError, OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        selectors, selected_episodes = [], []
        counts.update(selected_episodes=0, selected_windows=0)
        findings.append(_finding("consumer.sample_plan_unavailable", "UNKNOWN",
            "The saved context or stable bounded episode metadata could not establish a supported plan; no selectors were retained."))
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return {
        "schema_version": 1, "kind": "native_consumer_sample_plan",
        "requested_scope": "bounded metadata-only selected numeric windows",
        "status": _status(findings, planning), "planning_status": planning,
        "dataset_directory": str(root), "checkpoint_directory": str(checkpoint),
        "context_status": context_status, "context_axes": context_axes,
        "context_artifacts": context_artifacts, "interpretation_sha256": interpretation,
        "samples": selectors, "selected_episodes": selected_episodes, "counts": counts,
        "readset": readset, "rule": RULE.copy(), "findings": findings,
        "integration_requirement": (
            "Before running or accepting the later comparison, match these dataset metadata hashes "
            "and context artifact identities against its actual reads. A saved plan alone authorizes "
            "no stale or changed-input reuse. No workflow binding is implemented in this component."
        ),
        "bounds": {"max_episodes": MAX_EPISODES, "max_episode_rows": MAX_EPISODE_ROWS,
                   "metadata_file_bytes": MAX_METADATA, "max_windows": 9,
                   "source_body_allowance_bytes": 12 * 1024 * 1024,
                   "accounting": "Fixed input reader allowances only; not measured total filesystem I/O."},
        "evaluator": {"implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "boundaries": {key: "UNKNOWN" for key in (
            "actual_episode_rows", "selected_values", "representative_quality", "native_sample_parity",
            "full_processor_readiness", "model_target_effect", "historical_execution",
            "original_recording_association", "physical_truth")},
        "coverage": {"dataset_arrays_read": False, "media_read": False, "native_code_executed": False,
                     "scope": "Declared metadata lengths and IDs only; not actual data coverage."},
    }
