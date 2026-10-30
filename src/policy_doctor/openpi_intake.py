"""Bounded original OpenPI statistics intake; serving mode/origin are not inferred."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .artifacts import InputError
from .readiness_intake import ReadinessIntakeBudget

METADATA_LIMIT = 64 * 1024
STATISTICS_LIMIT = 1024 * 1024
_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,199}\Z")


@dataclass(frozen=True)
class OpenPIStatisticsSnapshot:
    """Original bytes/JSON plus scoped observations, never a normalization contract."""

    status: str
    document: dict | None
    raw: bytes | None
    metadata: dict | None
    metadata_raw: bytes | None
    provenance: dict
    observations: dict
    accounting: dict
    reasons: tuple[str, ...]


class _Unavailable(Exception):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _opener():
    return build_opener(_NoRedirect)


def _object(source):
    try:
        parts = urlsplit(source)
    except ValueError:
        raise InputError("Malformed OpenPI public source") from None
    if (parts.scheme != "gs" or parts.netloc != "openpi-assets" or parts.query
            or parts.fragment or "%" in parts.path or "\\" in parts.path):
        raise InputError("Only explicit official gs://openpi-assets statistics objects are supported")
    pieces = parts.path.removeprefix("/").split("/")
    if (not 5 <= len(pieces) <= 32 or pieces[0] != "checkpoints" or pieces[-1] != "norm_stats.json"
            or "assets" not in pieces[2:-2] or not all(_SEGMENT.fullmatch(x) for x in pieces)):
        raise InputError("OpenPI public input must select a bounded checkpoints/.../assets/.../norm_stats.json object")
    return "/".join(pieces)


def _check_budget(budget):
    if not isinstance(budget, ReadinessIntakeBudget):
        raise InputError("OpenPI intake requires a ReadinessIntakeBudget")
    for name in ("max_bytes", "max_files", "used_bytes", "logical_requests", "successful_body_bytes"):
        value = getattr(budget, name)
        if type(value) is not int or value < 0:
            raise InputError("OpenPI intake budget has an invalid ledger")
    if (not 1 <= budget.max_bytes <= 64 * 1024 ** 2 or not 1 <= budget.max_files <= 64
            or budget.used_bytes > budget.max_bytes or budget.logical_requests > budget.max_files
            or budget.successful_body_bytes > budget.used_bytes):
        raise InputError("OpenPI intake budget has an invalid ledger")


def _read_response(url, limit, generation=None):
    """No redirects; bounded application read and explicit complete-body identity."""
    deadline = time.monotonic() + 30
    request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "RoboClue-openpi-intake/1"})
    try:
        response = _opener().open(request, timeout=15)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        raise _Unavailable(f"OpenPI public intake refused HTTP {code}; redirects/fallback are disabled") from None
    except (URLError, OSError, TimeoutError, http.client.HTTPException, ValueError):
        raise _Unavailable("OpenPI public response could not be obtained") from None
    with response:
        if response.status != 200 or response.geturl() != url:
            raise _Unavailable("OpenPI public response was not the requested complete HTTP 200 object")
        if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
            raise _Unavailable("OpenPI public response used unsupported content encoding")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isascii() or not length.isdigit()
                                   or len(length) > 20 or int(length) > limit):
            raise _Unavailable("OpenPI public response exceeds its body bound")
        if generation is not None and response.headers.get("x-goog-generation") != generation:
            raise _Unavailable("OpenPI public response did not confirm the selected immutable generation")
        chunks, total = [], 0
        while True:
            if time.monotonic() >= deadline:
                raise _Unavailable("OpenPI public response exceeded its deadline checks")
            try:
                reader = getattr(response, "read1", response.read)
                chunk = reader(min(64 * 1024, limit + 1 - total))
            except (OSError, TimeoutError, http.client.HTTPException):
                raise _Unavailable("OpenPI public response could not be read completely") from None
            if not isinstance(chunk, bytes):
                raise _Unavailable("OpenPI transport returned a non-byte response")
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise _Unavailable("OpenPI public response exceeds its body bound")
            chunks.append(chunk)
        raw = b"".join(chunks)
        if length is not None and len(raw) != int(length):
            raise _Unavailable("OpenPI public response differs from its declared HTTP length")
        return raw


def _request(budget, url, *, limit, generation=None, exact_size=None):
    _check_budget(budget)
    if budget.logical_requests >= budget.max_files:
        raise _Unavailable("OpenPI intake exhausted its logical request bound")
    available = budget.max_bytes - budget.used_bytes
    if exact_size is not None and available < exact_size + 1:
        raise _Unavailable("OpenPI intake cannot reserve the declared object body within its remaining budget")
    limit = min(limit, available - 1)
    if limit < 1:
        raise _Unavailable("OpenPI intake cannot reserve a bounded response within its remaining budget")
    budget.logical_requests += 1
    budget.used_bytes += limit + 1
    # Failed reads retain the complete reservation, including the over-limit probe.
    raw = _read_response(url, limit, generation)
    if not isinstance(raw, bytes) or len(raw) > limit:
        raise _Unavailable("OpenPI transport returned an invalid or over-bound body")
    budget.used_bytes -= limit + 1 - len(raw)
    budget.successful_body_bytes += len(raw)
    return raw


def _json(raw):
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _Unavailable("OpenPI JSON is not valid UTF8") from None
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > 32:
                raise _Unavailable("OpenPI JSON exceeds the supported nesting bound")
        elif character in "]}":
            depth -= 1
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def integer(value):
        if len(value) > 128:
            raise ValueError("large integer")
        return int(value)
    try:
        document = json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, RecursionError, OverflowError):
        raise _Unavailable("OpenPI JSON is invalid or ambiguous within the supported parser bounds") from None
    todo, nodes = [document], 0
    while todo:
        value = todo.pop()
        nodes += 1
        if nodes > 100000:
            raise _Unavailable("OpenPI JSON exceeds the supported value-count bound")
        if isinstance(value, float) and not math.isfinite(value):
            raise _Unavailable("OpenPI JSON contains nonfinite values")
        if isinstance(value, dict):
            todo.extend(value.values())
        elif isinstance(value, list):
            todo.extend(value)
    if not isinstance(document, dict):
        raise _Unavailable("OpenPI statistics require an object within this intake route")
    return document


def _shape(value):
    if isinstance(value, list) and value:
        shapes = [_shape(x) for x in value]
        if all(x is not None and x == shapes[0] for x in shapes):
            return [len(value), *shapes[0]]
    elif type(value) in (int, float):
        try:
            if math.isfinite(value):
                return []
        except OverflowError:
            pass
    return None


def _observations(document):
    stats = document.get("norm_stats")
    if not isinstance(stats, dict) or not stats:
        return {}, ["The original JSON has no supported nonempty norm_stats mapping; native interpretation remains unknown"]
    facts, reasons = {}, []
    for role, values in stats.items():
        if not isinstance(values, dict):
            reasons.append(f"Statistics role {role!r} has an unsupported representation; original data is retained")
            continue
        shapes = {name: (None if value is None else _shape(value)) for name, value in values.items()}
        facts[role] = {"field_shapes": shapes, "null_fields": [k for k, v in values.items() if v is None]}
        if any(name not in ("mean", "std", "q01", "q99") for name in values):
            reasons.append(f"Statistics role {role!r} has native-extra fields; this intake does not decide their interpretation")
        if any(name not in values for name in ("mean", "std")):
            reasons.append(f"Statistics role {role!r} lacks a supported mean/std representation")
        for name, value in values.items():
            if name in ("mean", "std", "q01", "q99") and (value is not None or name in ("mean", "std")) and shapes[name] is None:
                reasons.append(f"Statistics field {role!r}/{name} has unsupported or nonrectangular numeric shape")
    extras = sorted(set(document) - {"norm_stats"})
    if extras:
        reasons.append("Original native-extra root fields are retained; their interpretation is not decided")
    return {"roles": facts, "root_extra_fields": extras,
            "scope": "Observed finite rectangular shapes only; mode, required quantiles, dimensions and training authority are not inferred"}, reasons


def _local(source, read_roots):
    if ".." in Path(source).parts:
        raise InputError("OpenPI local statistics parent components are refused before path rewriting")
    path = Path(os.path.abspath(Path(source).expanduser()))
    if read_roots is not None:
        roots = [Path(os.path.abspath(Path(root).expanduser())) for root in read_roots]
        if not roots or not any(path.is_relative_to(root) for root in roots):
            raise InputError("OpenPI local statistics are outside authorized read roots")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > STATISTICS_LIMIT:
                raise InputError("OpenPI local statistics must be a bounded regular no-follow file")
            chunks, total = [], 0
            while True:
                chunk = os.read(file_fd, min(64 * 1024, STATISTICS_LIMIT + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > STATISTICS_LIMIT:
                    raise InputError("OpenPI local statistics exceed the body bound")
            after = os.fstat(file_fd)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    or total != before.st_size):
                raise InputError("OpenPI local statistics changed during the bounded read")
            return path, b"".join(chunks)
        finally:
            os.close(file_fd)
    except OSError:
        raise InputError("OpenPI local statistics could not be opened as an authorized regular no-follow file") from None
    finally:
        os.close(fd)


def resolve_openpi_statistics(source, *, fetch=False, budget=None, read_roots=None, generation=None):
    """Read an actual selected original statistics file; no factory/mode inference."""
    if type(fetch) is not bool or not isinstance(source, str) or not source.strip() or "\0" in source or len(source) > 4096:
        raise InputError("OpenPI source must be nonempty text and fetch a strict boolean")
    if read_roots is not None and (not isinstance(read_roots, (list, tuple)) or len(read_roots) > 64
            or any(not isinstance(root, (str, Path)) or not str(root).strip()
                   or len(str(root)) > 4096 or "\0" in str(root) or ".." in Path(root).parts
                   for root in read_roots)):
        raise InputError("OpenPI read_roots must be a bounded sequence of explicit paths")
    if generation is not None and (not isinstance(generation, str)
            or not re.fullmatch(r"[1-9][0-9]{0,29}", generation)):
        raise InputError("OpenPI generation must be a positive ASCII decimal statistics-object generation")
    if generation is not None and "://" not in source:
        raise InputError("A GCS statistics-object generation cannot be applied to a local file")
    budget = ReadinessIntakeBudget(max_bytes=2 * 1024 ** 2, max_files=8) if budget is None else budget
    _check_budget(budget)
    raw = metadata_raw = document = metadata = None
    provenance = {"source": source, "requested_generation": generation,
                  "historical_training_authority": "not_authenticated"}
    observations, reasons = {}, []
    try:
        if "://" in source:
            name = _object(source)
            if not fetch:
                raise InputError("Public OpenPI statistics require explicit fetch authorization")
            metadata_url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o/" + quote(name, safe="")
            if generation is not None:
                metadata_url += "?generation=" + generation
            metadata_raw = _request(budget, metadata_url, limit=METADATA_LIMIT)
            metadata = _json(metadata_raw)
            selected_generation, size_text, md5_text = (metadata.get(k) for k in ("generation", "size", "md5Hash"))
            if (metadata.get("bucket") != "openpi-assets" or metadata.get("name") != name
                    or not isinstance(selected_generation, str) or not re.fullmatch(r"[1-9][0-9]{0,29}", selected_generation)
                    or not isinstance(size_text, str) or not re.fullmatch(r"[1-9][0-9]{0,9}", size_text)
                    or int(size_text) > STATISTICS_LIMIT or not isinstance(md5_text, str)):
                raise _Unavailable("OpenPI object metadata lacks the selected bounded immutable identity")
            if generation is not None and selected_generation != generation:
                raise _Unavailable("OpenPI object metadata differs from the explicitly requested statistics generation")
            try:
                expected_md5 = base64.b64decode(md5_text, validate=True)
            except (ValueError, TypeError):
                raise _Unavailable("OpenPI object metadata has an unsupported MD5 identity") from None
            if len(expected_md5) != 16:
                raise _Unavailable("OpenPI object metadata has an unsupported MD5 identity")
            size = int(size_text)
            provenance.update(bucket="openpi-assets", object=name, generation=selected_generation,
                              declared_size_bytes=size, declared_md5_base64=md5_text,
                              metadata_sha256=hashlib.sha256(metadata_raw).hexdigest())
            url = "https://storage.googleapis.com/openpi-assets/" + quote(name, safe="/") + "?generation=" + selected_generation
            raw = _request(budget, url, limit=size, exact_size=size, generation=selected_generation)
            if len(raw) != size or hashlib.md5(raw).digest() != expected_md5:
                raise _Unavailable("OpenPI object bytes differ from their generation metadata size/MD5 identity")
            provenance["content_identity_verified"] = True
        else:
            path, raw = _local(source, read_roots)
            provenance.update(source=str(path), kind="local_regular_file", local_body_bytes=len(raw))
        provenance.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        document = _json(raw)
        observations, reasons = _observations(document)
    except _Unavailable as exc:
        reasons.append(str(exc))
    if raw is not None:
        provenance.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    if metadata_raw is not None:
        provenance["metadata_sha256"] = hashlib.sha256(metadata_raw).hexdigest()
    return OpenPIStatisticsSnapshot("unknown" if reasons else "pass", document, raw, metadata, metadata_raw,
                                    provenance, observations, budget.summary(), tuple(reasons))
