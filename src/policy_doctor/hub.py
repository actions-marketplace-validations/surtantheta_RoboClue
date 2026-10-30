"""Explicit, public, JSON-only Hugging Face intake for the existing audit engine.

No Hub SDK, credentials, cache, weights, dataset rows, or remote code are used.
Resolve a mutable revision once; read a small allowlist only at that commit.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .artifacts import InputError
from .diagnostics import audit

MAX_API_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CONTENT_BYTES = 300 * 1024 * 1024
MAX_CONTENT_FILES = 4096  # the window audit's consistency pass reads at most this many shards per directory
MAX_PROVENANCE_FILES = 32
MODEL_FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")
# Dataset content for row-level checks: metadata plus Parquet rows, never videos or images.
_CONTENT = re.compile(r"meta/(?:info|stats)\.json|meta/tasks\.parquet|meta/episodes/[^/]+/[^/]+\.parquet|data/[^/]+/[^/]+\.parquet")
_CDN = re.compile(r"(?:[a-z0-9-]+\.)+hf\.co")
# One conservative character set per path segment: no spaces, backslashes, percent signs or non-ASCII.
_SAFE_NAME = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9_.=-]{0,127}/){1,3}[A-Za-z0-9][A-Za-z0-9_.=-]{0,127}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
DATASET_FILES = ("meta/info.json", "meta/stats.json")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_REV = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class HubUnavailable(InputError):
    """A public resource cannot be inspected; this is not a policy fault."""


class HubNotPublic(InputError):
    """The repository is private or gated; it is refused, never read with credentials."""


@dataclass
class HubSnapshot:
    root: Path
    provenance: dict
    omitted_states: dict[str, list[str]]
    video_context: dict | None = None


def _repo(source: str) -> str:
    if not isinstance(source, str) or not source.startswith("hf://"):
        raise InputError("Hub source must be hf://ORG/REPO; arbitrary URLs are not accepted")
    value = source[5:]
    parts = value.split("/")
    if len(parts) != 2 or any(not _PART.fullmatch(p) or ".." in p for p in parts):
        raise InputError("Hub source must be hf://ORG/REPO with no path, query, fragment or credentials")
    return value


def _revision(value: str) -> str:
    if not isinstance(value, str) or not _REV.fullmatch(value) or ".." in value:
        raise InputError("Hub revision must be a bounded branch/tag name or 40-character commit SHA")
    return value


def _validate_url(url: str, required_sha: str | None = None) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.fragment
            or parsed.username or parsed.password):
        raise InputError("Hub transport permits HTTPS on huggingface.co only")
    if required_sha is not None and required_sha not in parsed.path.split("/"):
        raise InputError("Hub download redirect lost the pinned revision")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    # Ignore ambient proxies, netrc, Hub tokens, cookies, and local SDK caches.
    return build_opener(ProxyHandler({}), _NoRedirect())


def _read_some(response, size: int) -> bytes:
    """Return the bytes already available (read1), so a slow stream cannot hold one read past the deadline."""
    reader = getattr(response, "read1", None)
    return reader(size) if callable(reader) else response.read(size)


def _request(url: str, *, limit: int, required_sha: str | None = None) -> bytes:
    opener = _opener()
    deadline = time.monotonic() + 30
    initial_path = urlsplit(url).path
    redirect_paths = {initial_path}
    if required_sha is not None:
        # A redirect may change HF's delivery route, never the selected file.
        prefix, marker, filename = initial_path.partition(f"/resolve/{required_sha}/")
        if marker:
            repository = prefix.removeprefix("/")
            repo_type = "datasets" if repository.startswith("datasets/") else "models"
            repository = repository.removeprefix("datasets/")
            redirect_paths.add(f"/api/resolve-cache/{repo_type}/{repository}/{required_sha}/{filename}")
            redirect_paths.add(f"/api/resolve-cache/{repo_type}/{repository}/{required_sha}/{quote(filename, safe='')}")
    for _ in range(5):
        if time.monotonic() >= deadline:
            raise HubUnavailable("Hub metadata request exceeded its time bound")
        _validate_url(url, required_sha)
        request = Request(url, headers={"Accept": "application/json", "Accept-Encoding": "identity",
                                        "User-Agent": "RoboClue-metadata-check/1"})
        try:
            response = opener.open(request, timeout=15)
        except (ValueError, http.client.InvalidURL) as exc:  # includes UnicodeEncodeError: never a traceback or a finding
            raise InputError(f"Hub request URL is not valid: {type(exc).__name__}") from exc
        except HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise InputError("Hub redirect has no target") from exc
                url = urljoin(url, location)
                _validate_url(url, required_sha)
                if urlsplit(url).path not in redirect_paths:
                    if required_sha is None:
                        raise HubUnavailable("Repository API redirects are not followed; supply the canonical repository ID") from exc
                    raise InputError("Hub redirect changed the selected metadata file or repository") from exc
                continue
            exc.close()
            error = HubUnavailable(f"Public Hub metadata unavailable (HTTP {exc.code}); no credentials used")
            error.http_status = exc.code
            raise error from exc
        except (URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            raise HubUnavailable("Public Hub metadata unavailable; no cached or authenticated fallback used") from exc
        with response:
            if getattr(response, "status", 200) != 200:
                raise HubUnavailable("Public Hub metadata did not return a complete HTTP 200 response")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > limit):
                raise InputError("Hub response exceeds the metadata byte bound")
            encoding = response.headers.get("Content-Encoding", "identity")
            if encoding not in ("identity", ""):
                raise InputError("Compressed Hub responses are not accepted by bounded metadata intake")
            commit = response.headers.get("X-Repo-Commit")
            if required_sha is not None and commit is not None and commit != required_sha:
                raise InputError("Hub response revision differs from pinned revision")
            chunks, total = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise HubUnavailable("Hub metadata request exceeded its time bound")
                try:
                    chunk = _read_some(response, min(65536, limit - total + 1))
                except (OSError, TimeoutError, http.client.HTTPException) as exc:
                    raise HubUnavailable("Hub metadata read failed; no partial metadata was audited") from exc
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise InputError("Hub response exceeds the metadata byte bound")
                chunks.append(chunk)
            if length is not None and total != int(length):
                raise InputError("Hub response length does not match its declared size")
            return b"".join(chunks)
    raise InputError("Hub metadata redirect count exceeds bound")


def _download_blob(url: str, target: Path, *, size: int, sha256: str) -> None:
    """Stream one content-addressed blob. Integrity comes from the pinned listing's SHA-256, not the CDN host.

    The pinned huggingface.co URL may redirect once, over HTTPS, to a Hugging Face CDN host (*.hf.co); the
    bytes must then match the listed size and SHA-256 exactly. No credentials, cookies or proxies are used.
    """
    _validate_url(url)
    opener = _opener()
    deadline = time.monotonic() + max(60.0, size / (256 * 1024))
    for hop in range(2):
        request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "RoboClue-metadata-check/1"})
        try:
            response = opener.open(request, timeout=30)
        except (ValueError, http.client.InvalidURL) as exc:  # includes UnicodeEncodeError: never a traceback or a finding
            raise InputError(f"Hub content URL is not valid: {type(exc).__name__}") from exc
        except HTTPError as exc:
            location = exc.headers.get("Location") if exc.code in (301, 302, 303, 307, 308) else None
            linked = (exc.headers.get("X-Linked-Etag") or "").strip('"')
            exc.close()
            if hop or not location:
                raise HubUnavailable(f"Public Hub content unavailable (HTTP {exc.code}); no credentials used") from exc
            redirected = urljoin(url, location)
            parts = urlsplit(redirected)
            if (parts.scheme != "https" or parts.username or parts.password or parts.port not in (None, 443)
                    or not _CDN.fullmatch(parts.hostname or "")):
                raise InputError("Hub content redirect left HTTPS Hugging Face hosts") from exc
            if linked and linked != sha256:
                raise InputError("Hub content redirect names a different blob than the pinned listing") from exc
            url = redirected
            continue
        except (URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            raise HubUnavailable("Public Hub content unavailable; no cached or authenticated fallback used") from exc
        digest, total = hashlib.sha256(), 0
        with response, open(target, "xb") as stream:
            if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                raise InputError("Compressed Hub content responses are not accepted")
            # read1 returns whatever has arrived, so the deadline is checked at least once per socket timeout.
            while chunk := _read_some(response, 1024 * 1024):
                if time.monotonic() >= deadline:
                    raise HubUnavailable("Hub content download exceeded its time bound")
                total += len(chunk)
                if total > size:
                    raise InputError("Hub content exceeds its listed size")
                digest.update(chunk)
                stream.write(chunk)
        if total != size or digest.hexdigest() != sha256:
            raise InputError("Hub content does not match the pinned listing's size and SHA-256")
        return
    raise InputError("Hub content redirect count exceeds bound")


class _HeaderBudget:
    """Bounds for opt-in MP4 header range reads across one dataset."""

    def __init__(self, files=256, requests=1500, total=96 * 1024 * 1024, seconds=600.0):
        self.files, self.requests, self.total = files, requests, total
        self.deadline = time.monotonic() + seconds
        self.used_requests = self.used_bytes = 0


class _HubRangeMedia:
    """Read-only MP4 access by HTTPS byte ranges at a pinned commit; interface of ``video_recovery._Media``.

    Each range must come back as an exact 206 with ``Content-Range`` total equal to the pinned listing's size.
    Header bytes are not verified against the file's LFS SHA-256 (partial reads cannot be); the provenance says so.
    """

    def __init__(self, url: str, size: int, budget: _HeaderBudget):
        if type(size) is not int or not 0 < size <= 2**53:
            raise InputError("Video requires a strict positive bounded listed size")
        self.url, self.size, self.budget = url, size, budget
        self.media_extents: list = []
        self.segments: list = []

    def read(self, offset, count):
        if type(offset) is not int or type(count) is not int or not 0 <= offset <= offset + count <= self.size:
            raise ValueError("MP4 read escapes the listed file size")
        if not count:
            return b""
        for start, data in self.segments:
            if start <= offset and offset + count <= start + len(data):
                return data[offset - start: offset - start + count]
        data = self._range(offset, offset + count - 1)
        self.segments.append((offset, data))
        return data[:count]

    def _range(self, first: int, last: int) -> bytes:
        budget = self.budget
        if budget.used_requests >= budget.requests or time.monotonic() >= budget.deadline:
            raise HubUnavailable("Video header intake exceeded its request or time bound")
        if budget.used_bytes + (last - first + 1) > budget.total:
            raise HubUnavailable("Video header intake exceeded its byte bound")
        url = self.url
        _validate_url(url)
        parts = urlsplit(url)
        if (parts.query or len(url) > 16384
                or not re.fullmatch(r"/datasets/[^/]+/[^/]+/resolve/[0-9a-f]{40}/.+", parts.path)):
            raise InputError("Video header reads require a canonical pinned Hub resource without query or fragment")
        opener = _opener()
        for hop in range(2):
            remaining = budget.deadline - time.monotonic()
            if budget.used_requests >= budget.requests or remaining <= 0:
                raise HubUnavailable("Video header intake exceeded its request or time bound")
            budget.used_requests += 1
            request = Request(url, headers={"Accept-Encoding": "identity", "Range": f"bytes={first}-{last}",
                                            "User-Agent": "RoboClue-metadata-check/1"})
            try:
                response = opener.open(request, timeout=min(30, remaining))
            except (ValueError, http.client.InvalidURL) as exc:
                raise InputError(f"Hub video URL is not valid: {type(exc).__name__}") from exc
            except HTTPError as exc:
                location = exc.headers.get("Location") if exc.code in (301, 302, 303, 307, 308) else None
                exc.close()
                if hop or not location:
                    raise HubUnavailable(f"Public Hub video header unavailable (HTTP {exc.code})") from exc
                redirected = urljoin(url, location)
                try:
                    parts = urlsplit(redirected)
                    allowed = (len(redirected) <= 16384 and parts.scheme == "https" and not parts.username
                               and not parts.password and not parts.fragment and parts.port in (None, 443)
                               and _CDN.fullmatch(parts.hostname or ""))
                except ValueError:
                    allowed = False
                if not allowed:
                    raise InputError("Hub video redirect left HTTPS Hugging Face hosts") from exc
                url = redirected
                continue
            except (URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
                raise HubUnavailable("Public Hub video header unavailable") from exc
            with response:
                expected = f"bytes {first}-{last}/{self.size}"
                if getattr(response, "status", None) != 206 or response.headers.get("Content-Range") != expected:
                    raise InputError("Hub video range response is not the exact requested range of the listed file")
                if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                    raise InputError("Compressed Hub video responses are not accepted")
                chunks, total = [], 0
                while True:
                    if time.monotonic() >= budget.deadline:
                        raise HubUnavailable("Video header intake exceeded its time bound")
                    chunk = _read_some(response, min(65536, last - first + 2 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    budget.used_bytes += len(chunk)
                    if total > last - first + 1 or time.monotonic() >= budget.deadline:
                        raise InputError("Hub video range exceeds its requested length or time bound")
                    chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != last - first + 1:
                raise InputError("Hub video range is truncated")
            return data
        raise InputError("Hub video redirect count exceeds bound")


def _video_header_intake(root: Path, repo: str, sha: str, sizes: dict, provenance: dict,
                         *, max_rows=200000, max_files=1024) -> dict:
    """Header timing for every video file the episode metadata references (opt-in; bounded range reads)."""
    from .artifacts import read_json
    from .video_index import _episode_rows, _relative_video_path, _video_keys, header_timing

    info = read_json(root / "meta" / "info.json")
    keys, template = _video_keys(info), info.get("video_path")
    if not keys or not isinstance(template, str):
        return {}
    rows, _ = _episode_rows(root, keys, max_rows=max_rows, max_files=max_files)
    wanted = set()
    for key in keys:
        columns = (f"videos/{key}/chunk_index", f"videos/{key}/file_index")
        for row in rows:
            if all(type(row.get(c)) is int for c in columns):
                try:
                    wanted.add(_relative_video_path(template, key, row[columns[0]], row[columns[1]]))
                except (KeyError, IndexError, ValueError):
                    continue  # the audit records unsupported paths; never fetch them
    budget = _HeaderBudget()
    headers = {}
    for rel in sorted(wanted):
        if rel not in sizes or not _SAFE_NAME.fullmatch(rel):
            continue  # absence is reported by the video index check against the listing
        if len(headers) >= min(budget.files, max_files):
            break  # unobserved listed files remain partial coverage, never a whole-video PASS
        url = f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{quote(rel, safe='/')}"
        try:
            headers[rel] = header_timing(_HubRangeMedia(url, sizes[rel], budget))
        except (InputError, ValueError, OSError, http.client.HTTPException) as exc:
            headers[rel] = {"status": "unavailable", "reason": str(exc)[:200]}
    provenance["video_headers"] = {
        "files": len(headers), "range_requests": budget.used_requests, "bytes": budget.used_bytes,
        "integrity": "MP4 header byte ranges at the pinned commit; each range's total size matches the listing, "
                     "but partial reads are not verified against the file's LFS SHA-256"}
    return headers


@contextmanager
def resolve_dataset_content(source: str, *, revision: str = "main", fetch: bool = False,
                            max_bytes: int = MAX_CONTENT_BYTES, video_headers: bool = False,
                            max_rows: int = 200000, max_files: int = 1024):
    """Yield a private copy of a public dataset's metadata and Parquet rows at one immutable commit.

    Explicit opt-in only. Video header ranges require a separate opt-in; no sample payload is fetched. Parquet and other
    LFS files are verified against the SHA-256 listed for the resolved commit, and small JSON files against the
    listed size and git blob ID.
    """
    repo = _repo(source)
    revision = _revision(revision)
    if fetch is not True:
        raise InputError("Hub network is disabled; explicitly enable fetch with --fetch for public dataset content")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 4 * 1024 ** 3:
        raise InputError("Dataset content byte limit must be an integer between 1 byte and 4 GiB")
    if (type(video_headers) is not bool or type(max_rows) is not int or not 1 <= max_rows <= 200000
            or type(max_files) is not int or not 1 <= max_files <= 1024):
        raise InputError("Require a boolean video_headers and bounded strict integer row/file budgets")
    api = f"https://huggingface.co/api/datasets/{repo}/revision/{quote(revision, safe='')}?blobs=true"
    api_bytes = _request(api, limit=MAX_API_BYTES)
    info = _json(api_bytes)
    sha = info.get("sha")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha) or (_SHA.fullmatch(revision) and sha != revision):
        raise InputError("Hub API did not resolve the requested immutable revision")
    if not isinstance(info.get("id"), str) or info["id"].lower() != repo.lower():
        raise HubUnavailable("Hub API repository identity differs from requested repository; aliases are not followed")
    if info.get("private") is not False or info.get("gated") not in (None, False):
        raise HubNotPublic("Only public ungated Hub repositories are accepted; no credentials are used")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > 50000 or any(not isinstance(x, dict) for x in siblings):
        raise InputError("Hub API has no complete bounded file listing")
    selected, names = [], set()
    for item in siblings:
        name = item.get("rfilename")
        if not isinstance(name, str) or not _CONTENT.fullmatch(name):
            continue
        if not _SAFE_NAME.fullmatch(name) or ".." in name.split("/"):
            raise InputError(f"Hub listing has a dataset file with an unsupported name: {json.dumps(name)[:200]}; "
                             "it is not downloaded, and the dataset is not checked without it")
        # Case-only duplicates collide on case-insensitive filesystems, so they count as repeats too.
        if name.casefold() in names:
            raise InputError(f"Hub listing repeats the file {name} (names are compared without case)")
        names.add(name.casefold())
        lfs = item.get("lfs")
        size, blob_id = item.get("size"), item.get("blobId")
        if type(size) is not int or size < 0:
            raise InputError(f"Hub listing lacks a size for {name}")
        if lfs is not None:
            if not isinstance(lfs, dict) or not isinstance(lfs.get("sha256"), str) or not _HEX64.fullmatch(lfs["sha256"]):
                raise InputError(f"Hub listing lacks a content SHA-256 for {name}")
            selected.append((name, size, lfs["sha256"], None))
        elif name.endswith(".parquet"):
            raise InputError(f"Hub listing lacks a content SHA-256 for {name}")
        elif not isinstance(blob_id, str) or not _HEX40.fullmatch(blob_id):
            raise InputError(f"Hub listing lacks a git blob ID for {name}")
        else:
            selected.append((name, size, None, blob_id))
    if not any(name == "meta/info.json" for name, *_ in selected):
        raise HubUnavailable("Repository has no meta/info.json; dataset format remains unknown")
    if len(selected) > MAX_CONTENT_FILES:
        raise InputError(f"Dataset lists {len(selected)} metadata and row files, above the {MAX_CONTENT_FILES}-file limit; "
                         "nothing was downloaded. Check a local copy instead")
    total = sum(size for _, size, _, _ in selected)
    if total > max_bytes:
        raise InputError(f"Dataset metadata and rows total {total} bytes, above the {max_bytes}-byte limit; "
                         "raise the limit explicitly or check a smaller dataset")
    provenance = {"source": source, "repo_id": repo, "repo_type": "dataset", "requested_revision": revision,
                  "resolved_revision": sha, "api_sha256": hashlib.sha256(api_bytes).hexdigest(),
                  "network": "explicit_public_https", "intake": "dataset_metadata_and_parquet_rows",
                  "integrity": "Parquet and LFS files match the listed SHA-256; other JSON files match the listed size "
                               "and git blob ID", "files": [], "downloaded_bytes": 0}
    with tempfile.TemporaryDirectory(prefix="policy-doctor-hub-data-") as temporary:
        root = Path(temporary)
        for name, size, blob, blob_id in sorted(selected):
            path = root / name
            url = f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{quote(name, safe='/')}"
            try:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if blob is None:
                    data = _request(url, limit=min(MAX_FILE_BYTES, size), required_sha=sha)
                    if len(data) != size or hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != blob_id:
                        raise InputError(f"Hub content for {name} does not match the pinned listing's size and git blob ID")
                    _json(data)
                    _write(root, name, data)
                    digest = hashlib.sha256(data).hexdigest()
                else:
                    _download_blob(url, path, size=size, sha256=blob)
                    digest = blob
            except (ValueError, OSError, http.client.HTTPException) as exc:  # URL, encoding, protocol, filesystem errors
                if isinstance(exc, InputError):
                    raise
                raise HubUnavailable(f"Could not fetch {name}: {type(exc).__name__}") from exc
            provenance["files"].append({"path": name, "size_bytes": size, "sha256": digest, "url": url})
            provenance["downloaded_bytes"] += size
        # The pinned listing makes video-file presence authoritative without downloading any video.
        listed = {}
        for item in siblings:
            name = item.get("rfilename")
            if not isinstance(name, str) or not name.startswith("videos/"):
                continue
            if name in listed:
                raise InputError("Pinned video listing repeats a resource")
            listed[name] = item.get("size")
        headers = {}
        if video_headers:
            try:
                headers = _video_header_intake(root, repo, sha, {k: v for k, v in listed.items() if type(v) is int},
                                               provenance, max_rows=max_rows, max_files=max_files)
            except (ValueError, OSError, http.client.HTTPException) as exc:
                provenance["video_headers"] = {"status": "unknown", "reason": str(exc)[:200]}
        yield HubSnapshot(root, provenance, {}, {"listing": set(listed), "headers": headers or None})


def portable_dataset_report(report: dict, view: HubSnapshot) -> dict:
    """Replace the private temporary root with the pinned hf:// identity and attach provenance."""
    source = view.provenance["source"] + "@" + view.provenance["resolved_revision"]
    report = _portable(report, [(str(view.root.resolve()), source), (str(view.root), source)])
    provenance = dict(view.provenance)
    files = provenance["files"]
    provenance["file_count"] = len(files)
    # The full manifest is pinned by digest; the report carries a bounded prefix, and cited files are in artifacts.
    provenance["files_manifest_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    provenance["files_truncated"] = len(files) > MAX_PROVENANCE_FILES
    provenance["files"] = files[:MAX_PROVENANCE_FILES]
    report["provenance"] = provenance
    if isinstance(report.get("artifacts"), list) and report.get("kind") == "episode_window_audit":
        # Paths are longer after the hf:// rewrite; bound the artifact list again at its final size.
        from .windows import bound_artifacts

        bound_artifacts(report, source, 224 * 1024)
    if len(json.dumps(report, allow_nan=False).encode()) > 224 * 1024:
        raise InputError("Fetched dataset report exceeds the bounded transport budget")
    return report


def _json(data: bytes) -> dict:
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise InputError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise InputError("Hub JSON contains a non-finite value")
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise InputError("Hub metadata JSON must be an object")
        finite(value)
        return value
    except InputError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InputError("Hub metadata is not bounded valid JSON") from exc


def _state_ref(value) -> str:
    if (not isinstance(value, str) or len(value) > 200 or "\\" in value or ":" in value
            or "%" in value or "?" in value or "#" in value or value.startswith("/")):
        raise InputError("Unsafe processor state reference")
    parts = value.split("/")
    if any(not p or p in (".", "..") for p in parts) or PurePosixPath(value).suffix != ".safetensors":
        raise InputError("Processor state reference must be a relative safetensors path")
    return value


def _write(root: Path, name: str, data: bytes) -> None:
    # name is selected from a fixed internal allowlist, never supplied by a URL.
    path = root / name
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _missing_repository(source: str, repo: str, repo_type: str, reason: str) -> str:
    """Anonymous Hub requests get 401 for private, gated and nonexistent repositories alike; name the possibilities.

    A model and a dataset can share an ID, so a repository of the other kind is reported as a possibility, never as
    proof that the requested kind or revision is absent. The actionable hint leads, so a clipped view keeps it.
    """
    other = "datasets" if repo_type == "model" else "models"
    unavailable = (f"No public {repo_type} was found at {source} at the requested revision ({reason}; anonymous "
                   "requests cannot tell whether it does not exist, is private or gated, or lacks that revision).")
    try:
        info = _json(_request(f"https://huggingface.co/api/{other}/{repo}", limit=MAX_API_BYTES))
    except InputError:
        info = {}
    if isinstance(info.get("id"), str) and info["id"].lower() == repo.lower():
        other_kind = "dataset" if repo_type == "model" else "model"
        if info.get("private") is not False or info.get("gated") not in (None, False):
            hint = (f"A gated or private {other_kind} repository exists at this ID; RoboClue refuses gated and private "
                    "repositories (no credentials are used), so it cannot be checked here either.")
        elif repo_type == "model":
            hint = (f"A public dataset repository exists at this ID; if you meant it, check it with "
                    f"`policy-doctor preflight {source} --fetch`, or pass it as --dataset beside a checkpoint.")
        else:
            hint = (f"A public model repository exists at this ID; if you meant it, pass it as the checkpoint "
                    f"(for example `policy-doctor check {source} --fetch`) and give a dataset repository here.")
        return f"{hint} {unavailable}"
    return f"{unavailable} Check the {repo_type} ID {repo} and the revision."


@contextmanager
def resolve_hub_snapshot(source: str, *, repo_type: str = "model", revision: str = "main",
                         fetch: bool = False):
    """Yield a private temporary JSON-only view and a portable intake receipt."""
    repo = _repo(source)
    revision = _revision(revision)
    if repo_type not in ("model", "dataset"):
        raise InputError("Hub repo_type must be model or dataset")
    if fetch is not True:
        raise InputError("Hub network is disabled; explicitly enable fetch with --fetch for public metadata")
    plural = "models" if repo_type == "model" else "datasets"
    api = f"https://huggingface.co/api/{plural}/{repo}/revision/{quote(revision, safe='')}"
    try:
        api_bytes = _request(api, limit=MAX_API_BYTES)
    except HubUnavailable as exc:
        if getattr(exc, "http_status", None) not in (401, 404):
            raise
        raise HubUnavailable(_missing_repository(source, repo, repo_type, str(exc))) from exc
    info = _json(api_bytes)
    sha = info.get("sha")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise InputError("Hub API did not resolve a valid immutable revision")
    if _SHA.fullmatch(revision) and sha != revision:
        raise InputError("Hub resolved revision differs from the requested commit")
    if not isinstance(info.get("id"), str) or info["id"].lower() != repo.lower():
        raise HubUnavailable("Hub API repository identity differs from requested repository; aliases are not followed")
    if info.get("private") is not False or info.get("gated") not in (None, False):
        raise HubNotPublic("Only public ungated Hub repositories are accepted; no credentials are used")
    siblings = info.get("siblings")
    if (not isinstance(siblings, list) or len(siblings) > 50000
            or any(not isinstance(x, dict) or not isinstance(x.get("rfilename"), str) for x in siblings)):
        raise InputError("Hub API has no complete bounded file listing")
    listed = [x["rfilename"] for x in siblings]
    if len(set(listed)) != len(listed):
        raise InputError("Hub API contains duplicate file paths")
    allowed = MODEL_FILES if repo_type == "model" else DATASET_FILES
    required = "config.json" if repo_type == "model" else "meta/info.json"
    if required not in listed:
        raise HubUnavailable(f"Repository has no supported {required}; metadata format remains unknown")
    prefix = "" if repo_type == "model" else "datasets/"
    provenance = {"source": source, "repo_id": repo, "repo_type": repo_type,
                  "requested_revision": revision, "resolved_revision": sha,
                  "api_sha256": hashlib.sha256(api_bytes).hexdigest(), "api_bytes": len(api_bytes),
                  "network": "explicit_public_https", "intake": "json_metadata_only",
                  "files": [], "downloaded_bytes": 0, "omitted_states": []}
    omitted = {}
    with tempfile.TemporaryDirectory(prefix="policy-doctor-hub-") as temporary:
        root = Path(temporary)
        for name in allowed:
            if name not in listed:
                continue
            remaining = MAX_METADATA_BYTES - provenance["downloaded_bytes"]
            url = f"https://huggingface.co/{prefix}{repo}/resolve/{sha}/{name}"
            data = _request(url, limit=min(MAX_FILE_BYTES, remaining), required_sha=sha)
            parsed = _json(data)
            if name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                steps = parsed.get("steps")
                if not isinstance(steps, list) or len(steps) > 100:
                    raise InputError("Hub processor steps must be a bounded list")
                for index, step in enumerate(steps):
                    if not isinstance(step, dict):
                        raise InputError("Hub processor steps must be objects")
                    if step.get("state_file"):
                        ref = _state_ref(step["state_file"])
                        present = ref in listed
                        provenance["omitted_states"].append({"processor": name, "pointer": f"/steps/{index}",
                            "path": ref, "listed_at_revision": present,
                            "reason": "Tensor state is outside JSON-only metadata intake"})
                        if present:
                            omitted.setdefault(name, []).append(f"/steps/{index}")
            _write(root, name, data)
            provenance["files"].append({"path": name, "size_bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest(), "url": url})
            provenance["downloaded_bytes"] += len(data)
        yield HubSnapshot(root, provenance, omitted)


def _portable(value, substitutions):
    if isinstance(value, str):
        for prefix, uri in substitutions:
            if value == prefix or value.startswith(prefix + "/"):
                return uri + value[len(prefix):]
        return value
    if isinstance(value, list):
        return [_portable(v, substitutions) for v in value]
    if isinstance(value, dict):
        return {k: _portable(v, substitutions) for k, v in value.items()}
    return value


def _withheld_unknown(report, view):
    if view is None:
        return
    for finding in report["findings"]:
        evidence = finding.get("evidence", [])
        withheld = any(e.get("pointer") in pointers and e.get("path") == str(view.root / name)
                       for name, pointers in view.omitted_states.items() for e in evidence)
        missing_stats = (finding["code"] == "normalization.statistics" and finding["status"] == "error"
                         and re.search(r": Missing (mean|std|min|max|q01|q99|q10|q90) statistics\.$",
                                       finding["message"]))
        if withheld and (finding["code"] == "processor.state_missing" or missing_stats):
            finding["status"] = "unknown"
            finding["code"] = ("processor.state_uninspected" if finding["code"] == "processor.state_missing"
                               else "normalization.statistics_uninspected")
            finding["message"] = "Referenced tensor state is listed at this commit and was not fetched by JSON-only intake."
            finding["remedy"] = "Use an explicit local checkpoint with its paired state files to inspect numerical statistics."


def _metadata_scope(report):
    unmodeled = any(f["code"].startswith("processor.unsupported_") for f in report["findings"])
    unmodeled_pre = any(f["code"] == "processor.unsupported_pre" for f in report["findings"])
    for finding in report["findings"]:
        if finding["code"] in ("dataset.feature_shape", "dataset.feature_missing"):
            if unmodeled and finding["status"] in ("error", "warning"):
                finding["status"] = "unknown"
                finding["message"] += " A declared processor is not modeled; its effect on this interface is unknown."
                finding["remedy"] = "Inspect the effective processor output before treating raw metadata differences as incompatible."
            elif finding["status"] == "error":
                finding["message"] = "Declared raw metadata mismatch: " + finding["message"]
        if (unmodeled_pre and finding["code"] == "action.processor_order_names"
                and finding["status"] == "error"
                and finding["message"].startswith(("dataset:", "checkpoint:"))):
            finding["status"] = "unknown"
            finding["message"] += " A declared preprocessor may alter this raw order; its effect is not modeled."
            finding["remedy"] = "Inspect a saved effective processor input/output before asserting an order contradiction."


def _audit_metadata(paths, views):
    try:
        return audit(paths[0], paths[1])
    except InputError as exc:
        # The local loader deliberately requires one explicit name per vector
        # coordinate. Public datasets also use group names / unnamed axes. Do
        # not reinterpret these fields, manufacture a reorder, or blame a repo.
        known = (str(exc) == "Feature names must be a list of nonempty strings (or a single-axis mapping)"
                 or str(exc) in ("action: names length does not match vector dimension",
                                 "observation.state: names length does not match vector dimension"))
        dataset_view = views[1]
        if not known or dataset_view is None:
            raise
        # Still validate the unmodified checkpoint independently. A checkpoint
        # error with the same wording is not swallowed as a dataset format gap.
        report = audit(paths[0])
        path = dataset_view.root / "meta/info.json"
        info = _json(path.read_bytes())
        receipt = next(f for f in dataset_view.provenance["files"] if f["path"] == "meta/info.json")
        evidence = []
        for key in ("action", "observation.state"):
            feature = info.get("features", {}).get(key)
            if isinstance(feature, dict) and "names" in feature:
                value = {k: feature.get(k) for k in ("shape", "names")}
                if len(json.dumps(value).encode()) > 4096:
                    value = {"detail": "Field exceeds bounded inline evidence; inspect the pinned original JSON."}
                evidence.append({"path": str(path), "sha256": receipt["sha256"],
                                 "pointer": "/features/" + key, "value": value})
        report["findings"].append({"code": "dataset.metadata_format_unsupported", "status": "unknown",
            "message": "Dataset names do not provide the explicit per-coordinate format required by this auditor; "
                       "checkpoint/dataset compatibility was not evaluated. This is an auditor coverage limit.",
            "evidence": evidence, "remedy": "Supply a source-backed per-coordinate interface declaration for a local audit; "
                "group labels or unnamed axes do not establish joint order.", "verification": str(exc)})
        report["artifacts"].extend({"path": str(dataset_view.root / f["path"]), "role": "uninterpreted_dataset_metadata",
                                     "sha256": f["sha256"], "size_bytes": f["size_bytes"]}
                                    for f in dataset_view.provenance["files"])
        return report


def _finish(report, provenance, substitutions, *, scope="deployment"):
    report.setdefault("schema_version", 1)
    report["kind"] = "checkpoint_check"
    report["provenance"] = provenance
    report["assessment_scope"] = scope
    report["deployment_status"] = "not_assessed"
    report["unassessed_findings"] = []
    if scope == "metadata":
        assessed = []
        for finding in report["findings"]:
            if finding["code"] == "deployment.missing" and finding["status"] == "unknown":
                report["unassessed_findings"].append(finding)
            else:
                assessed.append(finding)
        report["findings"] = assessed
        if not assessed:
            report["findings"].append({
                "code": "metadata.not_assessed", "status": "unknown",
                "message": "No declared interface or normalization comparison was assessed.",
                "evidence": [],
                "remedy": "Supply the policy's declared input/output features and applicable processor metadata; an empty assessment cannot establish compatibility.",
            })
    report["counts"] = {s: sum(f["status"] == s for f in report["findings"])
                        for s in ("error", "warning", "unknown", "pass")}
    report["status"] = next((s for s in ("error", "unknown", "warning") if report["counts"][s]), "pass")
    report["scope"] = "Declared checkpoint/dataset metadata; no weights, rows, inference or deployment were inspected."
    if scope == "metadata":
        report["scope"] = ("Metadata-only assessment; deployment is not assessed. "
                           "No model weights, rows, inference or robot execution were inspected.")
    return _portable(report, substitutions)


def check_checkpoint(source: str, dataset: str | None = None, *, fetch: bool = False,
                     revision: str = "main", dataset_revision: str = "main", scope: str = "deployment") -> dict:
    """Reuse the offline audit with explicit opt-in public Hub metadata views.

    This does not establish that a dataset trained a policy, physical compatibility,
    model competence, or numerical normalization validity when tensors are omitted.
    """
    if type(scope) is not str or scope not in ("deployment", "metadata"):
        raise InputError("Check scope must be deployment or metadata")
    if not isinstance(source, str) or not source:
        raise InputError("Checkpoint source must be a nonempty path or hf://ORG/REPO")
    _revision(revision)
    _revision(dataset_revision)
    for value in (source, dataset):
        if value is not None and (not isinstance(value, str) or not value):
            raise InputError("Checkpoint and dataset inputs must be nonempty paths or hf://ORG/REPO")
        if value and ("://" in value):
            _repo(value)
            if fetch is not True:
                raise InputError("Hub network is disabled; explicitly enable fetch with --fetch for public metadata")
    provenance, substitutions, unavailable = [], [], []
    with ExitStack() as stack:
        paths, views = [], []
        try:
            for value, kind, rev in ((source, "model", revision), (dataset, "dataset", dataset_revision)):
                if value is not None and value.startswith("hf://"):
                    try:
                        view = stack.enter_context(resolve_hub_snapshot(value, repo_type=kind, revision=rev, fetch=fetch))
                    except (HubUnavailable, HubNotPublic) as exc:
                        if kind != "dataset":
                            raise
                        # An unavailable dataset must not discard the policy's own checks.
                        unavailable.append((value, str(exc)))
                        paths.append(None)
                        views.append(None)
                        continue
                    provenance.append(view.provenance)
                    substitutions.append((str(view.root), value + "@" + view.provenance["resolved_revision"]))
                    paths.append(str(view.root))
                    views.append(view)
                else:
                    paths.append(value)
                    views.append(None)
            if views[0] is not None and not isinstance(_json((views[0].root / "config.json").read_bytes()).get("type"), str):
                raise HubUnavailable("Checkpoint metadata is outside the supported LeRobot policy-config format")
            report = _audit_metadata(paths, views)
            _withheld_unknown(report, views[0])
            _metadata_scope(report)
            for value, reason in unavailable:
                report["findings"].append({"code": "hub.metadata_unavailable", "status": "unknown",
                    "message": f"Dataset {value} could not be inspected: {reason} The policy was checked on its own.",
                    "evidence": [], "remedy": "Supply a public dataset ID or explicit local dataset metadata to compare interfaces."})
        except HubUnavailable as exc:
            report = {"findings": [{"code": "hub.metadata_unavailable", "status": "unknown",
                                    "message": str(exc), "evidence": [],
                                    "remedy": "Supply a supported public repository or explicit local metadata."}],
                      "artifacts": [], "limitations": []}
        return _finish(report, provenance, substitutions, scope=scope)
