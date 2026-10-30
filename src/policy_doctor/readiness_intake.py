"""Bounded public metadata and declared processor statistics for readiness.

Legacy JSON-only Hub tools retain their permissions and implementation. This
reader never fetches model parameters, imports repository code, or opens a
training-data reference automatically; the workflow chooses that reference.
"""

from __future__ import annotations

import hashlib
import http.client
import math
import re
import struct
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request

from . import hub
from .artifacts import InputError, finite_tree, step_kind

MODEL_JSON = (*hub.MODEL_FILES, "train_config.json", "processor_config.json",
              "embodiment_id.json", "statistics.json")
STATE_LIMIT = 4 * 1024 ** 2
_STATS = {"mean", "std", "min", "max", "q01", "q99", "q10", "q90", "q50", "count"}
_BOOKKEEPING = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "I64": 8, "I32": 4,
           "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1}


class UnsupportedProcessorStatistics(InputError):
    """A listed state cannot be numerically assessed within this reader's scope."""


@dataclass
class ReadinessIntakeBudget:
    """Shared response-body bound, including failed-request reservations.

    One byte is reserved for the existing transport's bounded over-limit probe.
    A failed call retains its entire reservation: its actual partial read is
    unknown. Logical requests exclude redirect hops and HTTP overhead.
    """

    max_bytes: int = 8 * 1024 ** 2
    max_files: int = 16
    used_bytes: int = field(default=0, init=False)
    logical_requests: int = field(default=0, init=False)
    successful_body_bytes: int = field(default=0, init=False)

    def __post_init__(self):
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= 64 * 1024 ** 2:
            raise InputError("Readiness max_bytes must be an integer between 1 byte and 64 MiB")
        if type(self.max_files) is not int or not 1 <= self.max_files <= 64:
            raise InputError("Readiness max_files must be an integer between 1 and 64")
    def request(self, url, *, limit, required_sha=None, lfs_sha256=None):
        if self.logical_requests >= self.max_files:
            raise InputError("Readiness intake exhausted its logical request/file bound")
        available = self.max_bytes - self.used_bytes
        limit = min(limit, available - 1)
        if limit < 1:
            raise InputError("Readiness intake exhausted its aggregate response-body bound")
        self.logical_requests += 1
        self.used_bytes += limit + 1
        data = (_lfs_request(url, limit=limit, required_sha=required_sha, sha256=lfs_sha256)
                if lfs_sha256 is not None else hub._request(url, limit=limit, required_sha=required_sha))
        if not isinstance(data, bytes) or len(data) > limit:
            raise InputError("Readiness transport returned an invalid or over-bound body")
        self.used_bytes -= limit + 1 - len(data)
        self.successful_body_bytes += len(data)
        return data

    def summary(self):
        return {"max_response_body_bytes": self.max_bytes, "max_logical_requests": self.max_files,
                "response_body_bytes_or_failed_reservations": self.used_bytes,
                "successful_response_body_bytes": self.successful_body_bytes,
                "logical_requests": self.logical_requests,
                "scope": "Response bodies and conservative failed-read reservations; excludes HTTP overhead."}


def _lfs_request(url, *, limit, required_sha, sha256):
    """One pinned HF request and at most one content-verified HTTPS CDN hop."""
    hub._validate_url(url, required_sha)
    opener, deadline = hub._opener(), time.monotonic() + 30
    for hop in range(2):
        if time.monotonic() >= deadline:
            raise hub.HubUnavailable("Processor-statistics request exceeded its time bound")
        request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "RoboClue-readiness/1"})
        try:
            response = opener.open(request, timeout=15)
        except HTTPError as exc:
            location = exc.headers.get("Location") if exc.code in (301, 302, 303, 307, 308) else None
            linked = (exc.headers.get("X-Linked-Etag") or "").strip('"')
            code = exc.code
            exc.close()
            if hop or not location:
                raise hub.HubUnavailable(f"Public processor statistics unavailable (HTTP {code})") from exc
            redirected = urljoin(url, location)
            parts = urlsplit(redirected)
            try:
                safe = (parts.scheme == "https" and not parts.username and not parts.password
                        and not parts.fragment and parts.port in (None, 443)
                        and hub._CDN.fullmatch(parts.hostname or ""))
            except ValueError:
                safe = False
            if not safe or (linked and linked != sha256):
                raise InputError("Processor-statistics redirect left allowed hosts or changed content identity") from exc
            url = redirected
            continue
        except (URLError, OSError, TimeoutError, http.client.HTTPException, ValueError) as exc:
            raise hub.HubUnavailable("Public processor statistics unavailable without authenticated fallback") from exc
        with response:
            if getattr(response, "status", 200) != 200:
                raise InputError("Processor statistics require a complete HTTP 200 response")
            length = response.headers.get("Content-Length")
            if length is not None and (not isinstance(length, str) or not length.isascii()
                                       or not length.isdigit() or len(length) > 20 or int(length) > limit):
                raise InputError("Processor-statistics response exceeds its body bound")
            if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                raise InputError("Compressed processor-statistics responses are not accepted")
            commit = response.headers.get("X-Repo-Commit")
            if commit is not None and commit != required_sha:
                raise InputError("Processor-statistics response names another repository revision")
            chunks, total = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise hub.HubUnavailable("Processor-statistics request exceeded its time bound")
                try:
                    chunk = hub._read_some(response, min(64 * 1024, limit + 1 - total))
                except (OSError, TimeoutError, http.client.HTTPException) as exc:
                    raise hub.HubUnavailable("Processor-statistics response body could not be read completely") from exc
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise InputError("Processor-statistics response exceeds its body bound")
                chunks.append(chunk)
            data = b"".join(chunks)
            if length is not None and len(data) != int(length):
                raise InputError("Processor-statistics response differs from its declared HTTP length")
            if hashlib.sha256(data).hexdigest() != sha256:
                raise InputError("Processor-statistics bytes differ from the pinned content identity")
            return data
    raise InputError("Processor-statistics redirect count exceeds its bound")


def _file_identity(entry, name):
    size = entry.get("size")
    lfs = entry.get("lfs")
    if isinstance(lfs, dict):
        oid, kind = lfs.get("sha256"), "sha256"
        if size is None:
            size = lfs.get("size")
        if lfs.get("size") != size:
            raise InputError(f"Pinned file size declarations disagree: {name}")
        pattern = r"[0-9a-f]{64}"
    else:
        oid, kind, pattern = entry.get("blobId"), "git_blob_sha1", r"[0-9a-f]{40}"
    if type(size) is not int or size < 0 or not isinstance(oid, str) or not re.fullmatch(pattern, oid):
        raise InputError(f"Pinned listing lacks a supported complete content identity: {name}")
    return size, kind, oid


def _verify_body(data, entry, name):
    size, kind, oid = _file_identity(entry, name)
    actual = (hashlib.sha256(data).hexdigest() if kind == "sha256" else
              hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest())
    if size != len(data) or actual != oid:
        raise InputError(f"Pinned file content identity differs from downloaded bytes: {name}")
    return {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "listed_content_id_kind": kind, "listed_content_id": oid}


def _state_name(value, *, selected=False):
    name = hub._state_ref(value)
    if not hub._SAFE_NAME.fullmatch("state/" + name):
        raise InputError("Processor statistics path has unsupported characters or depth")
    basename = PurePosixPath(name).name.lower()
    if selected and (basename.startswith(("model", "pytorch_model")) or "weights" in basename):
        raise InputError("Model parameter filenames are outside processor-statistics intake")
    return name


def _statistics_body(data, features):
    """Inspect the header's declared role before decoding bounded statistics."""
    if len(data) < 8:
        raise InputError("Processor statistics have no complete safetensors header")
    length = struct.unpack("<Q", data[:8])[0]
    if not 2 <= length <= min(64 * 1024, len(data) - 8):
        raise InputError("Processor statistics header is incomplete or exceeds 64 KiB")
    header = hub._json(data[8:8 + length])
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    if not entries or len(entries) > 256:
        raise InputError("Processor statistics must contain between 1 and 256 declared tensors")
    offsets, total = [], 0
    for key, entry in entries.items():
        feature, _, stat = key.rpartition(".")
        if stat not in _STATS or not isinstance(entry, dict):
            raise InputError("Processor state contains tensors outside its declared normalization statistics")
        # Native LeRobot serializes the whole dataset statistics dictionary in
        # each normalization state, including fields unused by that step. The
        # original smolvla_libero postprocessor proves this export behavior.
        # Interpretation still selects only the processor's effective features.
        if not (feature in features or feature in _BOOKKEEPING or
                feature.startswith(("observation.", "next."))):
            raise UnsupportedProcessorStatistics("Statistics include an unmodeled native feature namespace")
        shape, dtype, bounds = entry.get("shape"), entry.get("dtype"), entry.get("data_offsets")
        if isinstance(dtype, str) and dtype not in _DTYPES:
            raise UnsupportedProcessorStatistics(f"Declared statistics dtype {dtype[:32]} is not supported by this numeric reader")
        if (not isinstance(shape, list) or len(shape) > 4
                or any(type(n) is not int or not 0 <= n <= 1_000_000 for n in shape)
                or not isinstance(dtype, str) or dtype not in _DTYPES
                or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(n) is not int or n < 0 for n in bounds)):
            raise InputError("Processor statistics have unsupported shape, dtype or offsets")
        count = math.prod(shape)
        total += count
        if total > 1_000_000 or bounds[1] - bounds[0] != count * _DTYPES[dtype]:
            raise InputError("Processor statistics exceed the element bound or declared storage size")
        offsets.append(tuple(bounds))
    cursor = 0
    for start, end in sorted(offsets):
        if start != cursor or end < start:
            raise InputError("Processor statistics have overlapping or incomplete storage")
        cursor = end
    if cursor != len(data) - 8 - length:
        raise InputError("Processor statistics payload differs from its declared extent")
    from safetensors import SafetensorError
    from safetensors.numpy import load
    try:
        values = load(data)
        finite_tree({key: value.tolist() for key, value in values.items()})
    except (SafetensorError, ValueError, TypeError, OverflowError) as exc:
        raise InputError("Processor statistics cannot be read as bounded finite numeric tensors") from exc


@contextmanager
def resolve_readiness_snapshot(source: str, *, repo_type="model", revision="main", fetch=False,
                               budget: ReadinessIntakeBudget, processor_state=False):
    """Yield one immutable, content-verified metadata/statistics view.

    ``processor_state`` is a separate caller permission. It never expands the
    existing JSON-only tools, and applies only to explicitly named native
    normalizer/unnormalizer state files, not other processor state or weights.
    """
    repo, revision = hub._repo(source), hub._revision(revision)
    if repo_type not in ("model", "dataset") or type(fetch) is not bool or type(processor_state) is not bool:
        raise InputError("Readiness intake needs a supported repository type and strict permission booleans")
    if not isinstance(budget, ReadinessIntakeBudget):
        raise InputError("Readiness intake requires an explicit shared budget")
    if not fetch:
        raise InputError("Public readiness intake requires explicit fetch permission")
    if repo_type == "dataset" and processor_state:
        raise InputError("Processor-state permission applies only to checkpoint metadata")
    plural = "models" if repo_type == "model" else "datasets"
    api = f"https://huggingface.co/api/{plural}/{repo}/revision/{quote(revision, safe='')}?blobs=true"
    api_body = budget.request(api, limit=hub.MAX_API_BYTES)
    info = hub._json(api_body)
    sha = info.get("sha")
    if not isinstance(sha, str) or not hub._SHA.fullmatch(sha) or (hub._SHA.fullmatch(revision) and sha != revision):
        raise InputError("Readiness repository did not resolve the requested immutable revision")
    if not isinstance(info.get("id"), str) or info["id"].lower() != repo.lower():
        raise hub.HubUnavailable("Repository identity differs; aliases are not followed")
    if info.get("private") is not False or info.get("gated") not in (None, False):
        raise hub.HubNotPublic("Readiness intake accepts public ungated repositories without credentials")
    siblings = info.get("siblings")
    if (not isinstance(siblings, list) or len(siblings) > 50000 or
            any(not isinstance(row, dict) or not isinstance(row.get("rfilename"), str) for row in siblings)):
        raise InputError("Readiness repository has no complete bounded listing")
    listed = {row["rfilename"]: row for row in siblings}
    if len(listed) != len(siblings):
        raise InputError("Readiness repository listing contains duplicate paths")
    required = "config.json" if repo_type == "model" else "meta/info.json"
    if required not in listed:
        raise hub.HubUnavailable(f"Repository has no supported {required}")
    selected = [name for name in (MODEL_JSON if repo_type == "model" else hub.DATASET_FILES) if name in listed]
    prefix = "" if repo_type == "model" else "datasets/"
    provenance = {"source": source, "repo_id": repo, "repo_type": repo_type, "requested_revision": revision,
                  "resolved_revision": sha, "api_sha256": hashlib.sha256(api_body).hexdigest(),
                  "api_bytes": len(api_body), "network": "explicit_public_https",
                  "intake": "json_metadata_and_declared_processor_statistics" if processor_state else "json_metadata_only",
                  "files": [], "downloaded_bytes": 0, "omitted_states": []}
    omitted, states, references = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="roboclue-readiness-") as temporary:
        root = Path(temporary)

        def download(name, limit):
            size, kind, oid = _file_identity(listed[name], name)
            if size > limit:
                raise InputError(f"Selected readiness file exceeds its per-file bound: {name}")
            url = f"https://huggingface.co/{prefix}{repo}/resolve/{sha}/{name}"
            data = budget.request(url, limit=min(limit, max(size, 1)), required_sha=sha,
                                  lfs_sha256=oid if kind == "sha256" else None)
            record = _verify_body(data, listed[name], name)
            record["url"] = url
            return data, record

        for name in selected:
            data, record = download(name, hub.MAX_FILE_BYTES)
            parsed = hub._json(data)
            if name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                steps = parsed.get("steps")
                if not isinstance(steps, list) or len(steps) > 100:
                    raise InputError("Processor steps must be a bounded list")
                for index, step in enumerate(steps):
                    if not isinstance(step, dict):
                        raise InputError("Processor steps must be objects")
                    registry = step.get("registry_name") or step.get("class")
                    if not isinstance(registry, str) or not isinstance(step.get("config", {}), dict):
                        raise InputError("Processor steps need a declared name and configuration object")
                    if not step.get("state_file"):
                        continue
                    ref = _state_name(step["state_file"])
                    present = ref in listed
                    role = step_kind(registry)
                    config = step.get("config", {})
                    features = config.get("features") if isinstance(config, dict) else None
                    supported = role in ("normalize", "unnormalize") and isinstance(features, dict) and bool(features)
                    reference = {"processor": name, "pointer": f"/steps/{index}", "path": ref,
                                 "listed_at_revision": present}
                    references.setdefault(ref, []).append(reference)
                    if processor_state and supported and present:
                        _state_name(ref, selected=True)
                        if ref in states:
                            states[ref].update(features)
                        else:
                            states[ref] = set(features)
                    elif present:
                        omitted.setdefault(name, []).append(f"/steps/{index}")
                    if not (processor_state and supported and present):
                        provenance["omitted_states"].append({"processor": name, "pointer": f"/steps/{index}",
                            "path": ref, "listed_at_revision": present,
                            "reason": "Processor-state permission absent or normalization role/features unsupported" if present
                                      else "Referenced state is absent from the pinned listing"})
            hub._write(root, name, data)
            provenance["files"].append(record)
            provenance["downloaded_bytes"] += len(data)
        available_states = set()
        for name, features in sorted(states.items()):
            size, _, _ = _file_identity(listed[name], name)
            if size > STATE_LIMIT:
                provenance["omitted_states"].extend(dict(item, reason="Declared processor statistics exceed the 4 MiB per-file bound")
                                                    for item in references[name])
                continue
            data, record = download(name, STATE_LIMIT)
            try:
                _statistics_body(data, features)
            except UnsupportedProcessorStatistics as exc:
                record.update(retained_in_view=False, numerical_assessment="unknown", reason=str(exc))
                provenance["files"].append(record)
                provenance["downloaded_bytes"] += len(data)
                provenance["omitted_states"].extend(dict(item, reason=str(exc)) for item in references[name])
                continue
            hub._write(root, name, data)
            available_states.add(name)
            provenance["files"].append(record)
            provenance["downloaded_bytes"] += len(data)
        # One file can be referenced by several steps; bytes fetched for a known
        # normalizer are not simultaneously reported as withheld for another.
        provenance["omitted_states"] = [item for item in provenance["omitted_states"] if item["path"] not in available_states]
        omitted = {}
        for item in provenance["omitted_states"]:
            if item["listed_at_revision"] and item["pointer"] not in omitted.get(item["processor"], []):
                omitted.setdefault(item["processor"], []).append(item["pointer"])
        provenance["shared_budget_at_yield"] = budget.summary()
        yield hub.HubSnapshot(root, provenance, omitted)
