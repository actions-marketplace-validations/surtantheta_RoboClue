"""Versioned batch orchestration for derived native metadata proposals.

PASS means every detected association has an independently replayed native-cell
proposal. It does not mean that the dataset, whole video, or policy is repaired.
"""

from __future__ import annotations

import importlib.metadata
import io
import os
import shutil
import stat
import tempfile
from fractions import Fraction
from pathlib import Path

from . import hub
from . import video_recovery as vr
from .artifacts import LEROBOT_REVISION, InputError
from .publication import publish_capture
from .video_index import check_video_index, header_timing

KIND = "robot_video_reference_batch_v2"
BATCH_SCHEMA_VERSION = 2
MAX_EPISODES = 256
MAX_BATCH_BYTES = 8 * 1024**3
MAX_BATCH_MANIFEST = 4 * 1024**2


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(vr._encode(value))


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise InputError("Batch evidence symlinks are refused")
        if path.is_file() and path.relative_to(root).as_posix() not in ("BATCH_MANIFEST.json", "BATCH_READY.json"):
            relative = path.relative_to(root).as_posix()
            result[relative] = vr._sha(_seal_read(root, relative, vr.MAX_TOTAL))
    return result


def _batch_fingerprint():
    fingerprint = vr._fingerprint()
    root = Path(__file__).parent
    names = ("bulk_video_recovery.py", "bulk_workflow.py", "video_index.py", "windows.py")
    fingerprint["module_sha256"].update({name: vr._sha((root / name).read_bytes()) for name in names})
    dependencies = {}
    for name in ("numpy", "pyarrow"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return dict(fingerprint, batch_schema_version=BATCH_SCHEMA_VERSION, detector_schema_version=2,
                dependency_versions=dependencies, native_reference_revision=LEROBOT_REVISION)


def _seal(root, *, verifier=None):
    raw = vr._encode({"schema_version": BATCH_SCHEMA_VERSION, "kind": KIND,
                      "verifier": verifier if verifier is not None else _batch_fingerprint(),
                      "files": _inventory(root)})
    (root / "BATCH_MANIFEST.json").write_bytes(raw)
    _write(root / "BATCH_READY.json", {"manifest_sha256": vr._sha(raw)})


def _seal_read(root, name, bound):
    """Open root ancestors and a regular seal without following links (Linux)."""
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        relative = vr._native_path(name)
        parts = (*root.parts[1:], *Path(relative).parts[:-1])
        for part in parts:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(Path(relative).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= bound:
            raise InputError(f"Batch seal is not a bounded regular file: {str(root / name)!r}")
        chunks, total = [], 0
        while True:
            block = os.read(descriptor, min(65536, bound - total + 1))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > bound:
                raise InputError(f"Batch seal exceeds its read bound: {str(root / name)!r}")
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if total != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
            raise InputError(f"Batch seal changed while reading: {str(root / name)!r}")
        return b"".join(chunks)
    except FileNotFoundError as exc:
        raise InputError(f"Batch seal input does not exist: {str(root / name)!r}. "
                         "Check the case directory and retained seal files, then rerun.") from exc
    except OSError as exc:
        raise InputError(f"Batch seal requires no-follow regular-file access: {str(root / name)!r}. "
                         "Check the case path and any links, then rerun.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _unseal(root):
    raw = _seal_read(root, "BATCH_MANIFEST.json", MAX_BATCH_MANIFEST)
    ready = vr._json(_seal_read(root, "BATCH_READY.json", 4096))
    manifest = vr._json(raw)
    if (type(manifest) is not dict or set(manifest) != {"schema_version", "kind", "verifier", "files"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != BATCH_SCHEMA_VERSION
            or manifest["kind"] != KIND or ready != {"manifest_sha256": vr._sha(raw)}):
        raise InputError("Batch evidence has an unsupported or malformed versioned seal")
    # Refuse identity drift before interpreting detector facts or native proposals.
    if manifest["verifier"] != _batch_fingerprint():
        raise InputError("Batch verifier identity differs; replay cannot establish a complete proposal")
    if manifest["files"] != _inventory(root):
        raise InputError("Batch evidence inventory differs from its retained manifest")
    return manifest


def _validate(repo_id, revision, original_revision, max_episodes, max_body_bytes):
    vr._request_spec(repo_id, revision, original_revision, 0, "probe")
    if type(max_episodes) is not int or not 1 <= max_episodes <= MAX_EPISODES:
        raise InputError("max_episodes must be a strict integer in [1,256]")
    if type(max_body_bytes) is not int or not 1 <= max_body_bytes <= MAX_BATCH_BYTES:
        raise InputError("max_body_bytes must be a strict integer in [1,8 GiB]")


class BodyBudget:
    """Shared per-call read budget, including bytes from failed acquisitions."""

    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def transport(self, root):
        budget = self

        class Response:
            def __init__(self, response):
                self.response, self.read_bytes = response, 0
                self.headers = response.headers
                self.status = response.status

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *args):
                return self.response.__exit__(*args)

            def read(self, size):
                length = self.headers.get("Content-Length")
                if length is not None and str(length).isdigit() and self.read_bytes == int(length):
                    return b""
                remaining = budget.limit - budget.used
                if remaining <= 0:
                    raise InputError("Batch response-body budget exhausted")
                raw = self.response.read(min(size, remaining))
                budget.used += len(raw)
                self.read_bytes += len(raw)
                if len(raw) > remaining:
                    raise InputError("Transport exceeded the requested bounded read size")
                return raw

        class Opener:
            def __init__(self, opener):
                self.opener = opener

            def open(self, request, timeout):
                if budget.used >= budget.limit:
                    raise InputError("Batch response-body budget exhausted before request")
                # OpenerDirector's second positional argument is request data,
                # not the timeout. Passing a number there turns this GET into
                # a request with an invalid integer body on real HTTP handlers.
                return Response(self.opener.open(request, timeout=timeout))

        transport = vr._Transport(root)
        transport.opener = Opener(transport.opener)
        return transport


def _metadata(root):
    """Bounded full schema snapshot, with global episode identity coverage."""
    _, pq = vr._arrow()
    info_path = root / "meta/info.json"
    if any(p.is_symlink() for p in (info_path, *info_path.parents)):
        raise InputError("Metadata symlinks are refused")
    if info_path.stat().st_size > vr.MAX_METADATA:
        raise InputError("Native info JSON exceeds metadata byte bound")
    info_raw = _seal_read(root, "meta/info.json", vr.MAX_METADATA)
    info = vr._json(info_raw)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise InputError("Batch recovery requires native v3 episode metadata")
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths or len(paths) > 16:
        raise InputError("Require 1..16 native episode metadata files")
    files, identities, size, total_expanded = {}, set(), 0, 0
    for path in paths:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise InputError("Metadata symlinks are refused")
        rel = path.relative_to(root).as_posix()
        vr._native_path(rel)
        size += path.stat().st_size
        if size > vr.MAX_METADATA:
            raise InputError("Combined native episode metadata exceeds recovery byte bound")
        raw = _seal_read(root, rel, vr.MAX_METADATA)
        from .native_parquet_guard import require_unencrypted_parquet

        require_unencrypted_parquet(io.BytesIO(raw))
        pf = pq.ParquetFile(io.BytesIO(raw))
        if pf.metadata.num_rows > vr.MAX_SAMPLES or pf.metadata.num_columns > 512:
            raise InputError("Native episode dimensions exceed recovery bounds")
        expanded = sum(pf.metadata.row_group(i).column(j).total_uncompressed_size
                       for i in range(pf.metadata.num_row_groups) for j in range(pf.metadata.num_columns))
        total_expanded += expanded
        if total_expanded > vr.MAX_RESPONSE:
            raise InputError("Expanded native episode metadata exceeds recovery bound")
        table = pf.read()
        if len(table.column_names) != len(set(table.column_names)) or "episode_index" not in table.column_names:
            raise InputError("Native episode identity schema is absent or ambiguous")
        ids = table["episode_index"].to_pylist()
        if any(type(i) is not int or not 0 <= i < vr.MAX_SAMPLES for i in ids):
            raise InputError("Native episode identities must be bounded integers")
        if len(set(ids)) != len(ids) or identities.intersection(ids):
            raise InputError("Duplicated native episode identities across metadata files")
        identities.update(ids)
        if len(identities) > vr.MAX_SAMPLES:
            raise InputError("Combined native episode rows exceed recovery bound")
        files[rel] = raw
    total = info.get("total_episodes")
    if type(total) is not int or total != len(identities):
        raise InputError("Complete native episode identity coverage requires matching total_episodes")
    return {"meta/info.json": info_raw, **files}


def _native_identity(root, files):
    # Only the bounded metadata under review; never hash caller video payloads.
    return {"meta/info.json": vr._sha(_seal_read(root, "meta/info.json", vr.MAX_METADATA)),
            **{name: vr._sha(_seal_read(root, name, vr.MAX_METADATA)) for name in files}}


def _case_proposals(case, request, files):
    """Replay full native evidence first; a hash-only report is never sufficient."""
    replay = vr.reproduce_video_recovery(case)
    if replay.get("status") != "pass" or replay.get("repair_verified") is not True:
        raise InputError("Full retained case does not replay a verified recovery proposal")
    report = vr._json(_seal_read(case, "report.json", vr.MAX_TOTAL))
    recorded = vr._json(_seal_read(case, "request.json", vr.MAX_METADATA))
    if recorded != request:
        raise InputError("Recovery case does not match the exact pinned batch request")
    ledger = vr._json(_seal_read(case, "ledger.json", vr.MAX_METADATA))
    for name, raw in files.items():
        matches = [entry for entry in ledger if entry["url"] == vr._url(request, name)
                   and entry["range"] is None]
        if len(matches) != 1 or matches[0]["sha256"] != vr._sha(raw):
            raise InputError("Case does not bind the same complete native metadata projection")
    selected = report.get("evidence", {}).get("original_metadata_path")
    if selected not in files or report["evidence"]["original_metadata_sha256"] != vr._sha(files[selected]):
        raise InputError("Case selected metadata hash differs from the batch snapshot")
    return report["patch"]


def _consistent_case_origins(case, previous):
    ledger = vr._json(_seal_read(case, "ledger.json", vr.MAX_METADATA))
    for other in [case, *previous]:
        other_ledger = vr._json(_seal_read(other, "ledger.json", vr.MAX_METADATA))
        if vr._origin_conflicts(case, ledger, other, other_ledger):
            raise InputError("Batch cases contradict bytes or complete sizes for the same pinned resource")


def _canonical_paths(value, root):
    if isinstance(value, str):
        prefix = str(root)
        return "native://retained" + value[len(prefix):] if value == prefix or value.startswith(prefix + "/") else value
    if isinstance(value, list):
        return [_canonical_paths(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_paths(item, root) for key, item in value.items()}
    return value


def _detection_record(root, findings, coverage, context):
    if not isinstance(context, dict) or set(context) != {"listing", "headers"}:
        raise InputError("Automatic batch requires explicit derived native intake context")
    listing = context["listing"]
    if not isinstance(listing, (set, list)) or any(type(path) is not str for path in listing):
        raise InputError("Derived native listing is malformed")
    return {"schema_version": 2, "findings": _canonical_paths(findings, root), "coverage": coverage,
            "derived_video_context": {"listing": sorted(listing), "headers": context["headers"]},
            "assurance": "retained_derived_header_facts_internal_consistency_only",
            "raw_header_response_replay": False, "origin_authenticated": False}


def _candidate_plan(detection, max_episodes):
    candidates, seen = [], set()
    truncated = False
    for finding in detection["findings"]:
        if finding["code"] == "video.interval_past_end":
            rows = finding["episodes"]
            truncated |= len(rows) >= 32  # native detector currently exposes a bounded preview
            for row in rows:
                identity = (row["episode_index"], row["video_key"])
                if identity in seen:
                    raise InputError("Detector repeated an episode/camera association")
                seen.add(identity)
                candidates.append(identity)
    candidates.sort()
    truncated |= len(candidates) > max_episodes
    state = detection["coverage"]["video_index_details"]["status"]
    if truncated:
        state = "truncated"
    if any(f["code"] != "video.interval_past_end" and f["status"] in ("error", "unknown")
           for f in detection["findings"]):
        state = "other_unresolved_findings"
    return candidates, candidates[:max_episodes], state


def _intake_case_consistency(case, request, provenance, context):
    if (provenance.get("repo_id") != request["repo_id"]
            or provenance.get("resolved_revision") != request["revision"]):
        raise InputError("Native intake provenance differs from the pinned batch identity")
    ledger = vr._json(_seal_read(case, "ledger.json", vr.MAX_METADATA))
    api = f"https://huggingface.co/api/datasets/{request['repo_id']}/revision/{request['revision']}?blobs=true"
    entries = [entry for entry in ledger if entry["url"] == api and entry["range"] is None]
    if len(entries) != 1 or entries[0]["sha256"] != provenance.get("api_sha256"):
        raise InputError("Recovery case contradicts the initial pinned intake listing")
    headers = context.get("headers") or {}
    transport = vr._OfflineTransport(case, ledger)
    prefix = f"https://huggingface.co/datasets/{request['repo_id']}/resolve/{request['revision']}/"
    paths = {entry["url"][len(prefix):] for entry in ledger
             if entry["url"].startswith(prefix) and entry["url"].endswith(".mp4")}
    for path in paths:
        fact = headers.get(path)
        if not isinstance(fact, dict) or fact.get("status") != "parsed":
            continue  # unavailable observations make no concrete timing assertion
        observed = header_timing(vr._Media(transport, prefix + path))
        if observed != fact:
            raise InputError("Recovery case header facts contradict the initial pinned intake observations")


def _projection_paths(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _retained_detection_consistency(root, request, report):
    path = root / "detection.json"
    if not path.is_file():
        raise InputError("Automatic batch is missing its retained detection record")
    detection = vr._json(_seal_read(root, "detection.json", vr.MAX_METADATA))
    if not isinstance(detection, dict) or "derived_video_context" not in detection:
        raise InputError("Automatic batch has no retained derived detector context")
    context = detection["derived_video_context"]
    if (not isinstance(context, dict) or set(context) != {"listing", "headers"}
            or not isinstance(context["listing"], list)
            or len(context["listing"]) != len(set(context["listing"]))):
        raise InputError("Malformed retained derived detector context")
    native = root / "original_metadata"
    findings, coverage = check_video_index(native, listing=set(context["listing"]), headers=context["headers"],
                                           max_rows=request["max_rows"], max_files=request["max_files"])
    observed = _detection_record(native, findings, coverage, context)
    if observed != detection:
        raise InputError("Retained findings/coverage differ from native detector on retained derived facts")
    _, attempted, state = _candidate_plan(observed, request["max_episodes"])
    actual = [(entry.get("episode_index"), entry.get("video_key")) for entry in report["episodes"]]
    if actual != attempted or report["detection_coverage"] != state:
        raise InputError("Batch entries/coverage omit or contradict the exact retained detector candidate plan")
    return context


def merge_proposals(files, proposals):
    """Exactly the union of distinct native cell proposals; schema and all other cells unchanged."""
    grouped, cells = {}, set()
    for change in proposals:
        if type(change) is not dict or set(change) != {"path", "row", "episode_index", "column", "old", "new"}:
            raise InputError("Malformed native cell proposal")
        path, row, column = change["path"], change["row"], change["column"]
        if path not in files or type(row) is not int or row < 0 or type(column) is not str:
            raise InputError("Native proposal escapes the retained metadata projection")
        cell = (path, row, column)
        if cell in cells:
            raise InputError("Duplicate or contradictory native cell proposals are refused")
        cells.add(cell)
        if change["old"] == change["new"]:
            raise InputError("No-op native cell proposal is refused")
        grouped.setdefault(path, []).append(change)
    pa, pq = vr._arrow()
    result = dict(files)
    for path, changes in grouped.items():
        rows = pq.read_table(io.BytesIO(files[path])).to_pylist()
        for change in changes:
            if change["row"] >= len(rows) or rows[change["row"]]["episode_index"] != change["episode_index"]:
                raise InputError("Native proposal row does not have its declared episode identity")
        table = pq.read_table(io.BytesIO(files[path]))
        after, columns = table, {}
        for change in changes:
            column = change["column"]
            position = table.schema.get_field_index(column)
            if position < 0:
                raise InputError("Proposal column is absent from native schema")
            values = columns.setdefault(column, table[column].to_pylist())
            if values[change["row"]] != change["old"]:
                raise InputError("Proposal old value differs from original native cell")
            values[change["row"]] = change["new"]
        for column, values in columns.items():
            position = table.schema.get_field_index(column)
            field = table.schema.field(position)
            after = after.set_column(position, field, pa.array(values, type=field.type))
        actual = {(i, key) for i, (left, right) in enumerate(zip(rows, after.to_pylist(), strict=True))
                  for key in left if left[key] != right[key]}
        expected = {(change["row"], change["column"]) for change in changes}
        if actual != expected or not table.schema.equals(after.schema, check_metadata=True):
            raise InputError("Merged metadata changed cells/schema outside the distinct proposal union")
        output = io.BytesIO()
        pq.write_table(after, output)
        raw = output.getvalue()
        if len(raw) > vr.MAX_METADATA or not pq.read_table(io.BytesIO(raw)).equals(after, check_metadata=True):
            raise InputError("Merged native metadata does not round-trip within the bound")
        result[path] = raw
    if proposals:
        info = vr._json(files["meta/info.json"])
        fps = Fraction(str(info["fps"]))
        rows = [row for name, raw in result.items() if name.endswith(".parquet")
                for row in pq.read_table(io.BytesIO(raw)).to_pylist()]
        changed = {(p["episode_index"], p["column"].split("/")[1]) for p in proposals}
        for episode, camera in changed:
            selected = next(row for row in rows if row["episode_index"] == episode)
            prefix = f"videos/{camera}/"
            first, last = vr._interval(selected, camera, fps)
            for other in rows:
                if other["episode_index"] == episode:
                    continue
                if any(other[prefix + name] != selected[prefix + name] for name in ("chunk_index", "file_index")):
                    continue
                other_first, other_last = vr._interval(other, camera, fps)
                if max(first, other_first) < min(last, other_last):
                    raise InputError("Merged proposals overlap another final native episode interval")
    return result


def _finish(staging, request, entries, files, proposals, coverage, *, detection=None, budget=None):
    errors = []
    try:
        merged = merge_proposals(files, proposals)
    except (InputError, ValueError, IndexError, KeyError) as exc:
        merged = {}
        errors.append(str(exc))
    for path, raw in merged.items():
        out = staging / "proposed_metadata" / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw)
    verified = sum(e["status"] == "pass" for e in entries)
    complete = bool(entries) and verified == len(entries) and coverage == "complete" and not errors
    report = {"kind": KIND, "batch_schema_version": BATCH_SCHEMA_VERSION, "verifier_matches": True, "status": "pass" if complete else "unknown",
              "proposal_complete": complete, "request": request, "episodes": entries,
              "verified_associations": verified, "attempted_associations": len(entries),
              "proposed_cells": len(proposals) if not errors else 0,
              "detection_coverage": coverage, "merge_errors": errors,
              "origin_authenticated": False, "assurance": "internal_consistency_only",
              "detector_coverage_replayed_from_raw_headers": False,
              "upstream_modified": False, "body_bytes_read": budget.used if budget else 0,
              "next_actions": (["Review the merged native metadata copy; test it with the intended native consumer"]
                               if complete else ["Review unresolved episode entries; no complete batch repair is claimed"]),
              "scope": "Native metadata reference proposals only; no whole-video, training or robot-performance claim"}
    _write(staging / "batch_request.json", request)
    _write(staging / "native_hashes.json", {p: vr._sha(raw) for p, raw in files.items()})
    _write(staging / "proposals.json", proposals)
    _write(staging / "report.json", report)
    if detection is not None:
        _write(staging / "detection.json", detection)
    _seal(staging)
    return report


def _destination(destination):
    target = vr._path(destination)
    if target.exists():
        raise InputError("Batch destination exists; choose a new directory")
    if any((p / "READY.json").exists() or (p / "BATCH_READY.json").exists() for p in target.parents):
        raise InputError("Batch destination cannot be nested inside retained evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def merge_retained_cases(repo_id, revision, original_revision, native_root, cases, destination,
                         *, max_episodes=32, max_body_bytes=1):
    """Offline explicit-case merge. No detector coverage claim and no network.

    The caller lists complete retained cases. Missing media/payload becomes an
    unresolved entry. This route never calls itself a complete automatic repair.
    """
    _validate(repo_id, revision, original_revision, max_episodes, max_body_bytes)
    verifier = _batch_fingerprint()
    if not isinstance(cases, list) or not cases or len(cases) > max_episodes:
        raise InputError("Explicit case list exceeds the bounded episode count")
    native_root = vr._path(native_root)
    if vr._path(destination).is_relative_to(native_root):
        raise InputError("Batch destination must be outside the caller's native dataset root")
    target = _destination(destination)
    files = _metadata(native_root)
    before = _native_identity(native_root, files)
    staging = Path(tempfile.mkdtemp(prefix=".bulk-recovery-", dir=target.parent))
    try:
        entries, proposals, identities, accepted = [], [], set(), []
        for index, case_value in enumerate(cases):
            entry = {"case": f"cases/{index:04d}", "status": "unknown"}
            case = vr._path(case_value)
            try:
                before_case = vr._inventory_checked(case)
                request = vr._json(_seal_read(case, "request.json", vr.MAX_METADATA))
                spec = vr._request_spec(repo_id, revision, original_revision,
                                        request["episode_index"], request["video_key"])
                identity = (spec["episode_index"], spec["video_key"])
                if identity in identities:
                    raise InputError("Duplicate episode/camera case is refused")
                identities.add(identity)
                changes = _case_proposals(case, spec, files)
                _consistent_case_origins(case, accepted)
                copied = staging / entry["case"]
                shutil.copytree(case, copied)
                if vr._inventory_checked(case) != before_case or vr._inventory_checked(copied) != before_case:
                    raise InputError("Case evidence changed while copying into the batch")
                proposals.extend(changes)
                accepted.append(copied)
                entry.update(status="pass", episode_index=spec["episode_index"], video_key=spec["video_key"])
            except (InputError, OSError, ValueError, KeyError) as exc:
                entry["reason"] = str(exc)
            entries.append(entry)
        request = {"repo_id": repo_id, "revision": revision, "original_revision": original_revision,
                   "route": "explicit_retained_cases", "max_episodes": max_episodes}
        report = _finish(staging, request, entries, files, proposals, "explicit_subset")
        for path, raw in files.items():
            out = staging / "original_metadata" / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
        if _native_identity(native_root, files) != before or _metadata(native_root) != files:
            raise InputError("Native caller input changed during offline batch merge")
        if _batch_fingerprint() != verifier:
            raise InputError("Batch verifier changed during acquisition")
        _seal(staging, verifier=verifier)
        publish_capture(staging, target)
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def recover_all_video_references(repo_id, revision, original_revision, destination, *,
                                 fetch=False, max_episodes=32, max_body_bytes=512*1024**2,
                                 max_intake_bytes=16*1024**2, max_rows=100000, max_files=128):
    """Real pinned Hub intake -> native detector -> independent full cases -> one proposal.

    Intake uses an explicit conservative reservation for the existing native
    intake's accepted metadata, API body and 96 MiB header read bound. Remaining
    response body reads are metered per response by the injected transport.
    """
    _validate(repo_id, revision, original_revision, max_episodes, max_body_bytes)
    verifier = _batch_fingerprint()
    if fetch is not True:
        raise InputError("Public batch intake requires explicit fetch=True")
    if type(max_intake_bytes) is not int or not 1 <= max_intake_bytes <= 300*1024**2:
        raise InputError("Require an explicit bounded native intake byte budget")
    reserved = max_intake_bytes + hub.MAX_API_BYTES + hub._HeaderBudget().total + 1024**2 + 1
    if max_body_bytes <= reserved:
        raise InputError("Batch body budget cannot cover the conservative native-intake reservation")
    target = _destination(destination)
    staging = Path(tempfile.mkdtemp(prefix=".bulk-recovery-", dir=target.parent))
    budget = BodyBudget(max_body_bytes-reserved)
    try:
        with hub.resolve_dataset_content(f"hf://{repo_id}", revision=revision, fetch=True,
                                         max_bytes=max_intake_bytes, video_headers=True,
                                         max_rows=max_rows, max_files=max_files) as view:
            files = _metadata(view.root)
            findings, coverage = check_video_index(view.root, **view.video_context,
                                                   max_rows=max_rows, max_files=max_files)
            _write(staging / "intake_provenance.json", view.provenance)
            detection = _detection_record(view.root, findings, coverage, view.video_context)
            _, candidates, state = _candidate_plan(detection, max_episodes)
            entries, proposals, accepted = [], [], []
            for index, (episode, camera) in enumerate(candidates[:max_episodes]):
                spec = vr._request_spec(repo_id, revision, original_revision, episode, camera)
                case = staging / "cases" / f"{index:04d}"
                entry = {"episode_index": episode, "video_key": camera,
                         "case": f"cases/{index:04d}", "status": "unknown"}
                try:
                    vr.recover_video_reference(repo_id, revision, original_revision, episode, camera,
                                                case, transport_factory=budget.transport)
                    changes = _case_proposals(case, spec, files)
                    _consistent_case_origins(case, accepted)
                    _intake_case_consistency(case, spec, view.provenance, view.video_context)
                    proposals.extend(changes)
                    accepted.append(case)
                    entry["status"] = "pass"
                except (InputError, OSError, ValueError, KeyError) as exc:
                    entry["reason"] = str(exc)
                entries.append(entry)
            request = {"repo_id": repo_id, "revision": revision, "original_revision": original_revision,
                       "route": "pinned_public_intake", "max_episodes": max_episodes,
                       "max_rows": max_rows, "max_files": max_files,
                       "max_body_bytes": max_body_bytes, "conservative_intake_reserved_bytes": reserved}
            report = _finish(staging, request, entries, files, proposals, state,
                             detection=detection, budget=budget)
            report["body_bytes_read"] = None  # intake reservation is not a body measurement
            report["recovery_body_bytes_read"] = budget.used
            report["response_body_upper_bound"] = reserved + budget.used
            _write(staging / "report.json", report)
            for path, raw in files.items():
                out = staging / "original_metadata" / path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(raw)
            if _batch_fingerprint() != verifier:
                raise InputError("Batch verifier changed during acquisition")
            _seal(staging, verifier=verifier)
            publish_capture(staging, target)
            return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def reproduce_bulk_proposal(directory):
    """Relocated offline native evidence replay, with exact union/schema checks."""
    root = vr._path(directory)
    before = _unseal(root)
    request = vr._json(_seal_read(root, "batch_request.json", vr.MAX_METADATA))
    report = vr._json(_seal_read(root, "report.json", vr.MAX_METADATA))
    automatic = request["route"] == "pinned_public_intake"
    context = _retained_detection_consistency(root, request, report) if automatic else None
    provenance = vr._json(_seal_read(root, "intake_provenance.json", vr.MAX_METADATA)) if automatic else None
    verified = sum(e.get("status") == "pass" for e in report["episodes"])
    complete = bool(report["episodes"]) and verified == len(report["episodes"]) and report["detection_coverage"] == "complete" and not report["merge_errors"]
    if (report["request"] != request or report["proposal_complete"] != complete
            or report["status"] != ("pass" if complete else "unknown")
            or report["verified_associations"] != verified
            or report["attempted_associations"] != len(report["episodes"])
            or request["route"] == "explicit_retained_cases" and report["detection_coverage"] != "explicit_subset"):
        raise InputError("Batch claimed summary differs from its retained entries and scope")
    hashes = vr._json(_seal_read(root, "native_hashes.json", vr.MAX_METADATA))
    files = {path: _seal_read(root, "original_metadata/" + vr._native_path(path), vr.MAX_METADATA) for path in hashes}
    if _projection_paths(root / "original_metadata") != set(files):
        raise InputError("Retained original native projection has extra or missing files")
    if {path: vr._sha(raw) for path, raw in files.items()} != hashes:
        raise InputError("Original native metadata hashes differ")
    proposals, accepted = [], []
    for entry in report["episodes"]:
        if entry["status"] == "pass":
            spec = vr._request_spec(request["repo_id"], request["revision"], request["original_revision"],
                                    entry["episode_index"], entry["video_key"])
            case = root / vr._native_path(entry["case"])
            proposals.extend(_case_proposals(case, spec, files))
            _consistent_case_origins(case, accepted)
            if automatic:
                _intake_case_consistency(case, spec, provenance, context)
            accepted.append(case)
    recorded = vr._json(_seal_read(root, "proposals.json", vr.MAX_METADATA))
    if proposals != recorded:
        raise InputError("Batch proposal union differs from independently replayed full cases")
    if report["proposed_cells"] != len(proposals):
        raise InputError("Batch claimed cell count differs from independently replayed proposals")
    merged = merge_proposals(files, proposals)
    if _projection_paths(root / "proposed_metadata") != set(merged):
        raise InputError("Merged native projection has extra or missing files")
    for path, raw in merged.items():
        if _seal_read(root, "proposed_metadata/" + path, vr.MAX_METADATA) != raw:
            raise InputError("Merged native metadata copy differs from full replay")
    if _unseal(root) != before:
        raise InputError("Batch evidence changed during replay")
    return {"status": "pass", "reproduction_matches": True,
            "verifier_matches": True, "batch_schema_version": BATCH_SCHEMA_VERSION,
            "proposal_complete": report["proposal_complete"], "batch_status": report["status"],
            "proposed_cells": len(proposals), "origin_authenticated": False,
            "retained_detector_consistency_verified": automatic,
            "detector_coverage_replayed_from_raw_headers": False,
            "assurance": "internal_consistency_only"}
