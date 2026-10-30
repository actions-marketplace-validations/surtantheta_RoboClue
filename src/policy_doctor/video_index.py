"""Video index checks for LeRobot v3 datasets: declared video features against episode columns, files and
real per-file frame counts read from MP4 headers only (no video payload is decoded or read).

Findings are scoped to what the pinned LeRobot source makes fatal:

- ``video.feature_without_index``: a declared video feature lacks the per-episode ``videos/<key>/*`` columns
  that ``LeRobotDatasetMetadata.get_video_file_path`` reads (src/lerobot/datasets/dataset_metadata.py:356-358).
- ``video.file_missing``: a referenced video file is absent from an authoritative pinned Hub listing.
- ``video.interval_past_end``: an episode's frames lie past the last sample of its video file. LeRobot shifts query
  timestamps by ``from_timestamp`` (src/lerobot/datasets/dataset_reader.py:437-438) and raises
  ``FrameTimestampError`` when no frame lies within ``tolerance_s`` (src/lerobot/datasets/video_utils.py:262-265).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .artifacts import LEROBOT_REVISION, read_json

LEROBOT = f"LeRobot {LEROBOT_REVISION[:7]}"
MAX_VIDEO_FILES = 512
MAX_EPISODE_ROWS = 200000
TOLERANCE_S = 1e-4  # LeRobotDataset default tolerance_s (src/lerobot/datasets/lerobot_dataset.py:61)
HEADER_INDEX = ".roboclue/video_headers.json"  # legacy path, deliberately never trusted or read


class _LocalMedia:
    """Seekable reader exposing the interface of ``video_recovery._Media``; reads only what the parser asks for."""

    def __init__(self, path: Path):
        self.media_extents: list = []
        self._stream = path.open("rb")
        self._before = os.fstat(self._stream.fileno())
        self.size = self._before.st_size
        self.path = path

    def read(self, offset, count):
        if type(offset) is not int or type(count) is not int or not 0 <= offset <= offset + count <= self.size:
            raise ValueError("MP4 read escapes the file")
        self._stream.seek(offset)
        data = self._stream.read(count)
        if len(data) != count:
            raise ValueError("Truncated MP4 read")
        return data

    def close(self):
        try:
            after = self.path.stat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(self._before, key) != getattr(after, key) for key in fields):
                raise ValueError("Video changed during header checking")
        finally:
            self._stream.close()


def header_timing(media) -> dict:
    """Sample count and presentation timing of the single video track, from the MP4 header boxes only."""
    from .video_recovery import _parse_mp4, _Unsupported

    try:
        table = _parse_mp4(media)
    except (_Unsupported, ValueError) as exc:
        return {"status": "unsupported", "reason": str(exc)[:200]}
    starts = [float(start) for start, _ in table["times"]]
    ends = [float(start + length) for start, length in table["times"]]
    return {"status": "parsed", "sample_count": table["sample_count"],
            "last_sample_start_s": max(starts) if starts else None, "end_s": max(ends) if ends else None}


def _video_keys(info: dict) -> list[str]:
    features = info.get("features") if isinstance(info, dict) else None
    if features is not None and (not isinstance(features, dict) or len(features) > 4096):
        raise ValueError("Video features require a bounded mapping")
    return sorted(k for k, f in (features or {}).items() if isinstance(f, dict) and f.get("dtype") == "video")


def _episode_rows(root: Path, keys: list[str], *, max_rows=MAX_EPISODE_ROWS, max_files=1024):
    """Read selected metadata columns only after the existing intake's footer/expansion preflight."""
    import pyarrow as pa

    from .windows import MAX_VALUES, _Intake, _integer, _Unknown

    if not _integer(max_rows, minimum=1, maximum=MAX_EPISODE_ROWS) or not _integer(max_files, minimum=1, maximum=1024):
        raise ValueError("Invalid native video metadata budgets")
    intake = _Intake(root, max_rows, max_files)
    try:
        files = intake.discover(root / "meta" / "episodes")
    except _Unknown as exc:
        raise ValueError(str(exc)) from exc
    rows, present, expanded, values = [], set(), 0, 0
    for path in files:
        try:
            pf, _ = intake.parquet(path, (), "episode_metadata", action=True)
            names = pf.schema_arrow.names
            if len(names) != len(set(names)) or any(k not in names for k in ("episode_index", "length")):
                raise ValueError("Missing or ambiguous episode_index/length metadata")
            if len(rows) + pf.metadata.num_rows > max_rows:
                raise ValueError("Episode metadata row budget exceeded before reading")
            wanted = ["episode_index", "length"] + [f"videos/{k}/{c}" for k in keys
                                                    for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
            columns = [c for c in wanted if c in names]
            pf, _ = intake.parquet(path, columns, "episode_metadata", action=True)
            for column in columns:
                field_type = pf.schema_arrow.field(column).type
                numeric = pa.types.is_integer(field_type) or pa.types.is_floating(field_type)
                valid = (numeric or pa.types.is_null(field_type)) if column.endswith("timestamp") else pa.types.is_integer(field_type)
                if not valid and not (pa.types.is_null(field_type) and column.startswith("videos/")):
                    raise ValueError("Invalid native video index/timestamp schema")
            # Literal feature names contain dots; _Intake's generic dotted-path accounting is for nested fields.
            # These selected columns have just been restricted to primitive types, so match their exact names.
            for group in range(pf.metadata.num_row_groups):
                row_group = pf.metadata.row_group(group)
                for number in range(row_group.num_columns):
                    column = row_group.column(number)
                    if column.path_in_schema in columns:
                        expanded += column.total_uncompressed_size
                        values += column.num_values
            if expanded > 8 * 1024 * 1024 or values > MAX_VALUES:
                raise ValueError("Video metadata expanded-byte/value budget exceeded before reading")
            present.update(columns)
            for batch in pf.iter_batches(batch_size=512, columns=columns):
                for row in batch.to_pylist():
                    if not _integer(row.get("episode_index")) or not _integer(row.get("length"), minimum=1):
                        raise ValueError("Invalid native episode_index/length values")
                    for column in columns:
                        value = row[column]
                        if value is None or column in ("episode_index", "length"):
                            continue
                        if column.endswith(("/chunk_index", "/file_index")):
                            valid = _integer(value)
                        else:
                            valid = type(value) in (int, float) and math.isfinite(value)
                        if not valid:
                            raise ValueError("Invalid native video index/timestamp values")
                    rows.append(row)
        except _Unknown as exc:
            raise ValueError(str(exc)) from exc
    intake.stable()
    return rows, present


def _relative_video_path(template, key, chunk, file):
    rel = template.format(video_key=key, chunk_index=chunk, file_index=file)
    native = PurePosixPath(rel)
    if (not rel or len(rel) > 512 or "\\" in rel or native.is_absolute() or str(native) != rel
            or any(part in (".", "..") for part in native.parts)):
        raise ValueError("Video path must remain a normalized relative dataset path")
    return rel


def check_video_index(root: str | Path, *, listing: set[str] | None = None, headers: dict | None = None,
                      max_rows=MAX_EPISODE_ROWS, max_files=1024, video_headers=True):
    """Return ``(findings, coverage)`` for the dataset at ``root``.

    ``listing`` (all file paths at a pinned Hub commit) makes file presence authoritative; ``headers`` maps video
    paths to ``header_timing`` results obtained by bounded range reads. Without them, local files are used.
    """
    from .windows import _Intake, _Unknown

    root = Path(root).expanduser().resolve()
    intake = _Intake(root, max_rows, max_files)
    details = {"status": "unknown", "declared_video_features": 0, "episode_rows": 0,
               "files_referenced": 0, "files_checked": 0, "files_missing": 0,
               "files_unsupported": 0, "files_unread": 0}
    findings = []

    def coverage(reason, status="unknown"):
        details.update(status=status, reason=reason)
        details["finding_counts"] = {severity: sum(item.get("status") == severity for item in findings)
                                     for severity in ("error", "warning", "unknown", "pass")}
        return {"video_index": reason, "video_index_details": details}

    if listing is not None and (not isinstance(listing, set) or len(listing) > 50000
                                or any(not isinstance(value, str) for value in listing)):
        return [], coverage("not checked: invalid supplied video listing")
    if headers is not None and (not isinstance(headers, dict) or len(headers) > MAX_VIDEO_FILES):
        return [], coverage("not checked: invalid supplied header observations")
    info_path = root / "meta" / "info.json"
    try:
        intake.artifact(info_path, "dataset_info")
        info = read_json(info_path)
        keys = _video_keys(info)
    except (OSError, ValueError, _Unknown) as exc:
        return [], coverage(f"not checked: {exc}")
    if isinstance(info, dict) and info.get("storage_format", "parquet") != "parquet":
        # Episode metadata shards left beside another declared backend are not its video index.
        return [], coverage("not checked: info.json declares a non-Parquet storage_format")
    details["declared_video_features"] = len(keys)
    if not keys:
        return [], coverage("no declared video features", "not_applicable")
    fps, template = info.get("fps"), info.get("video_path")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0 or not isinstance(template, str):
        return [], coverage("not checked: info.json lacks a usable finite fps or video_path")
    try:
        rows, present = _episode_rows(root, keys, max_rows=max_rows, max_files=max_files)
    except (OSError, ValueError) as exc:
        return [], coverage(f"not checked: {exc}")
    details["episode_rows"] = len(rows)
    if not rows:
        return [], coverage("not checked: no episode metadata rows")
    findings, checked, unsupported, absent_local, missing_listed, listed_unread = [], 0, [], 0, [], 0
    info_ref = {"path": str(info_path), "pointer": "/features"}
    overruns = []
    references = set()
    for key in keys:
        needed = [f"videos/{key}/{c}" for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
        lacking = [c for c in needed if c not in present]
        incomplete = [r["episode_index"] for r in rows if not lacking and any(r.get(c) is None for c in needed)]
        if lacking or incomplete:
            findings.append({
                "code": "video.feature_without_index", "status": "error", "video_key": key,
                "message": (f"info.json declares video feature {key}, but episode metadata "
                            + (f"has no {', '.join(c.split('/')[-1] for c in lacking)} column(s) for it"
                               if lacking else f"leaves its location empty for {len(incomplete)} episode(s)")
                            + f". {LEROBOT} reads videos/{key}/chunk_index and file_index to locate every frame "
                            "(src/lerobot/datasets/dataset_metadata.py:356-358), so loading those frames fails."),
                "episodes": incomplete[:16], "evidence": [dict(info_ref, value=key)]})
            continue
        by_file = defaultdict(list)
        for r in rows:
            by_file[(r[f"videos/{key}/chunk_index"], r[f"videos/{key}/file_index"])].append(r)
        files = sorted(by_file)
        starts = {f: min(r[f"videos/{key}/from_timestamp"] for r in by_file[f]) for f in files}
        for chunk, file in files:
            try:
                rel = _relative_video_path(template, key, chunk, file)
            except (KeyError, IndexError, ValueError):
                return findings, coverage("not checked: unsafe or unsupported video_path template")
            references.add(rel)
            details["files_referenced"] = len(references)
            if len(references) > min(MAX_VIDEO_FILES, max_files):
                return findings, coverage("not checked: native video file budget exceeded", "partial")
            timing = None
            if listing is not None and rel not in listing:
                missing_listed.append(rel)
                details["files_missing"] += 1
                continue
            elif video_headers and headers is not None and rel in headers:
                timing = headers[rel]
            elif listing is not None:
                listed_unread += 1  # present at the pinned commit; headers were not requested
                details["files_unread"] += 1
                continue
            elif not video_headers:
                details["files_unread"] += 1
                continue
            else:
                path = root / rel
                intake.safe(path)
                if path.is_file():
                    try:
                        media = _LocalMedia(path)
                        try:
                            timing = header_timing(media)
                        finally:
                            media.close()
                    except (OSError, ValueError) as exc:
                        timing = {"status": "unavailable", "reason": str(exc)[:200]}
                else:
                    absent_local += 1
                    details["files_unread"] += 1
                    continue
            if (not isinstance(timing, dict) or timing.get("status") != "parsed"
                    or type(timing.get("sample_count")) is not int or not 0 < timing["sample_count"] <= 100000
                    or type(timing.get("last_sample_start_s")) not in (int, float)
                    or not math.isfinite(timing["last_sample_start_s"])):
                unsupported.append(rel)
                details["files_unsupported"] += 1
                continue
            checked += 1
            details["files_checked"] += 1
            last = timing["last_sample_start_s"]
            follow = starts.get((chunk, file + 1))
            if follow is None:
                follow = starts.get((chunk + 1, 0))
            for r in by_file[(chunk, file)]:
                begin = r[f"videos/{key}/from_timestamp"]
                needed_last = begin + (r["length"] - 1) / fps
                duration = r["length"] / fps
                if not math.isfinite(needed_last) or not math.isfinite(duration):
                    unsupported.append(rel)
                    details["files_unsupported"] += 1
                    break
                if begin < -TOLERANCE_S or needed_last > last + TOLERANCE_S:
                    overruns.append({
                        "video_key": key, "episode_index": r["episode_index"], "file": rel,
                        "file_samples": timing["sample_count"], "file_last_frame_s": round(last, 6),
                        "episode_from_s": round(begin, 6), "episode_last_frame_s": round(needed_last, 6),
                        "next_file_starts_with_unreferenced_span_of_this_length":
                            follow is not None and abs(follow - duration) <= 1e-6})
    if missing_listed:
        findings.append({
            "code": "video.file_missing", "status": "error", "files": missing_listed[:16],
            "message": (f"{len(missing_listed)} video file(s) referenced by episode metadata are absent from the pinned "
                        f"repository listing (e.g. {missing_listed[0]}); frames in them cannot be decoded."),
            "evidence": [dict(info_ref, value="video_path")]})
    if overruns:
        hinted = sum(o["next_file_starts_with_unreferenced_span_of_this_length"] for o in overruns)
        findings.append({
            "code": "video.interval_past_end", "status": "error", "episodes": overruns[:32],
            "message": (f"{len(overruns)} episode/camera interval(s) need frames past the last sample of their video "
                        f"file (e.g. episode {overruns[0]['episode_index']}, {overruns[0]['video_key']}: last frame at "
                        f"{overruns[0]['episode_last_frame_s']} s, file ends at {overruns[0]['file_last_frame_s']} s). "
                        f"{LEROBOT} shifts query timestamps by from_timestamp (src/lerobot/datasets/dataset_reader.py:"
                        "437-438) and raises FrameTimestampError when no frame is within tolerance "
                        "(src/lerobot/datasets/video_utils.py:262-265)."
                        + (f" For {hinted} of them the next file begins with an unreferenced span of exactly that "
                           "episode's length: run recover-video-reference to verify whether the clip is there."
                           if hinted else "")),
            "evidence": [{"path": str(root / o["file"]), "value": {k: o[k] for k in
                          ("episode_index", "file_samples", "file_last_frame_s", "episode_last_frame_s")}}
                         for o in overruns[:4]]})
    parts = [f"checked {checked} video file(s)"]
    if unsupported:
        parts.append(f"{len(unsupported)} with unsupported MP4 layout not checked")
    if listed_unread:
        parts.append(f"{listed_unread} listed at the pinned commit but not read (add --video-headers to check frame counts)")
    if absent_local:
        parts.append(f"{absent_local} not present locally (download videos to check them)")
    if missing_listed:
        parts.append(f"{len(missing_listed)} missing from the pinned listing")
    intake.stable()
    state = "partial" if (unsupported or listed_unread or absent_local or details["files_unread"]) else "complete"
    return findings, coverage("; ".join(parts), state)
