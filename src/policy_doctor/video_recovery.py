"""Bounded public native-video reference recovery and code-free offline replay.

Only verified original-clip identity can justify a metadata reference proposal.
No original files, remote repositories, robots, decoders or artifact code execute.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import io
import json
import math
import re
import shutil
import struct
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request

from . import __version__
from .artifacts import InputError
from .hub import _opener
from .incidents import _inside, _inventory
from .publication import publish_capture

KIND = "robot_video_reference_recovery"
MAX_RESPONSE = 32 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_METADATA = 4 * 1024 * 1024
MAX_MOOV = 2 * 1024 * 1024
MAX_SAMPLES = 100000
CDN_HOSTS = {
    "cas-bridge.xethub.hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
    "us.aws.cdn.hf.co",
}
_CRITERION = "out_of_bounds_interval_and_exact_next_shard_original_clip_identity_v1"
_VERIFICATION_SCOPE = "native_reference_association_and_retained_listing_internal_consistency_only"
_SCOPE = (
    "Verified native metadata-reference association for the selected episode only. "
    "A proposal is not an upstream dataset update, whole-video integrity, robot repair or policy-performance proof. "
    "Complete inputs match content IDs in a retained pinned Hub listing; complete target-media integrity is unverified. "
    "Offline replay checks internal consistency only; public origin is not authenticated by resealable local evidence. "
    "No artifact code, decoder, simulator or network executes during replay."
)
_MANIFEST_KEYS = {"schema_version", "kind", "files", "verifier"}


class _Unsupported(ValueError):
    """Truthful unresolved native input; never a partial repair PASS."""


def _encode(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise InputError("Recovery evidence requires finite JSON values") from exc


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise InputError("Duplicate JSON keys are refused")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
        _encode(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError("Invalid finite JSON recovery evidence") from exc


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise InputError("Require a nonempty local recovery-case path")
    path = Path(value).expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise InputError("Recovery case symlinks are refused")
    return path.resolve()


def _request_spec(repo_id, revision, original_revision, episode_index, video_key):
    if type(repo_id) is not str or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", repo_id
    ):
        raise InputError("Require a canonical public dataset ORG/REPO identifier")
    if any(part in (".", "..") for part in repo_id.split("/")):
        raise InputError("Unsafe dataset identifier")
    if any(type(v) is not str or not re.fullmatch(r"[0-9a-f]{40}", v) for v in (revision, original_revision)):
        raise InputError("Both revisions must be exact lowercase 40-character commit SHAs")
    if type(episode_index) is not int or not 0 <= episode_index < MAX_SAMPLES:
        raise InputError("Require an integer episode_index in [0,100000)")
    if (
        type(video_key) is not str
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", video_key)
        or video_key in (".", "..")
    ):
        raise InputError("Require a bounded native video feature key without path separators")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "original_revision": original_revision,
        "episode_index": episode_index,
        "video_key": video_key,
        "criterion": _CRITERION,
    }


def _native_path(value):
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or str(PurePosixPath(value)) != value
        or any(p in (".", "..") for p in PurePosixPath(value).parts)
    ):
        raise InputError("Require a normalized bounded native relative file path")
    return value


def _url(spec, path, *, original=False):
    path = _native_path(path)
    revision = spec["original_revision" if original else "revision"]
    return f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{revision}/{quote(path, safe='/')}"


class _Transport:
    """Credential-free pinned-origin requests; signed redirects are transient only."""

    def __init__(self, root):
        self.root, self.ledger, self.total = root, [], 0
        self.deadline = time.monotonic() + 120
        self.opener = _opener()
        (root / "payload").mkdir()

    def fetch(self, url, *, limit, byte_range=None):
        if len(self.ledger) >= 32 or time.monotonic() >= self.deadline:
            raise InputError("Recovery intake exceeds its request/time budget")
        if not 0 < limit <= MAX_RESPONSE:
            raise InputError("Invalid bounded recovery response limit")
        original = urlsplit(url)
        if (
            original.scheme != "https"
            or original.netloc != "huggingface.co"
            or original.fragment
            or original.username
            or original.password
        ):
            raise InputError("Recovery intake starts only at pinned public huggingface.co URLs")
        match = re.search(r"/resolve/([0-9a-f]{40})/", original.path)
        api_match = re.search(r"/(?:tree|revision)/([0-9a-f]{40})(?:/|$)", original.path)
        required_sha = (match or api_match).group(1) if match or api_match else None
        if required_sha is None:
            raise InputError("Recovery intake requires an immutable dataset path")
        canonical = url
        headers = {"Accept": "*/*", "Accept-Encoding": "identity", "User-Agent": "RoboClue-video-recovery/1"}
        if byte_range is not None:
            first, last = byte_range
            if (
                type(first) is not int
                or type(last) is not int
                or not 0 <= first <= last
                or last - first + 1 > limit
            ):
                raise InputError("Invalid bounded media byte range")
            headers["Range"] = f"bytes={first}-{last}"
        response = None
        try:
            for _ in range(5):
                parsed = urlsplit(url)
                if (
                    parsed.scheme != "https"
                    or parsed.username
                    or parsed.password
                    or parsed.fragment
                    or parsed.netloc not in {"huggingface.co", *CDN_HOSTS}
                    or len(url) > 16384
                ):
                    raise InputError("Recovery redirect host/transport is not allowlisted")
                if parsed.netloc == "huggingface.co" and required_sha not in parsed.path.split("/"):
                    raise InputError("Recovery redirect lost the immutable revision")
                if time.monotonic() >= self.deadline:
                    raise InputError("Recovery intake exceeded its deadline")
                try:
                    response = self.opener.open(Request(url, headers=headers), timeout=15)
                except HTTPError as exc:
                    if exc.code not in (301, 302, 303, 307, 308):
                        exc.close()
                        raise InputError(f"Public recovery input unavailable (HTTP {exc.code})") from exc
                    location = exc.headers.get("Location")
                    commit = exc.headers.get("X-Repo-Commit")
                    if commit is not None and commit != required_sha:
                        exc.close()
                        raise InputError("Hub redirect revision differs from the immutable request") from exc
                    exc.close()
                    if not location:
                        raise InputError("Public recovery redirect has no target") from exc
                    destination = urljoin(url, location)
                    next_url = urlsplit(destination)
                    if next_url.netloc == "huggingface.co":
                        allowed = {
                            original.path,
                            original.path.replace("/datasets/", "/api/resolve-cache/datasets/", 1).replace(
                                "/resolve/", "/"
                            ),
                        }
                        origin_prefix, marker, filename = original.path.partition(f"/resolve/{required_sha}/")
                        if marker:
                            repository = origin_prefix.removeprefix("/datasets/")
                            allowed.add(
                                f"/api/resolve-cache/datasets/{repository}/{required_sha}/{quote(filename, safe='')}"
                            )
                        if next_url.path not in allowed:
                            raise InputError(
                                "Recovery redirect changed the selected native resource"
                            ) from exc
                    elif parsed.netloc != "huggingface.co" and next_url.netloc != parsed.netloc:
                        raise InputError("Recovery CDN redirect changed delivery host") from exc
                    url = destination
                    continue
                break
            else:
                raise InputError("Recovery redirect count exceeds its bound")
            with response:
                status = getattr(response, "status", 200)
                if status != (206 if byte_range is not None else 200):
                    raise InputError("Recovery range/full response status is not exact")
                if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                    raise InputError("Encoded recovery responses are refused")
                commit = response.headers.get("X-Repo-Commit")
                if commit is not None and commit != required_sha:
                    raise InputError("Hub response differs from the pinned revision")
                if response.headers.get("Link"):
                    raise InputError(
                        "Paginated recovery metadata is unsupported; no incomplete listing is accepted"
                    )
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) > limit):
                    raise InputError("Recovery response exceeds its byte bound")
                content_range, total_size = response.headers.get("Content-Range"), None
                if byte_range is not None:
                    parsed_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
                    if not parsed_range or tuple(map(int, parsed_range.groups()[:2])) != byte_range:
                        raise InputError("Recovery Content-Range differs from the exact requested bytes")
                    total_size = int(parsed_range.group(3))
                    if total_size <= byte_range[1] or total_size > 2**53:
                        raise InputError("Invalid bounded media total size")
                chunks, count = [], 0
                while True:
                    if time.monotonic() >= self.deadline:
                        raise InputError("Recovery intake exceeded its deadline")
                    block = response.read(min(65536, limit - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > limit or self.total + count > MAX_TOTAL:
                        raise InputError("Recovery response/total byte budget exceeded")
                    chunks.append(block)
                raw = b"".join(chunks)
                if length is not None and len(raw) != int(length):
                    raise InputError("Recovery response length differs from its header")
                if byte_range is not None and len(raw) != byte_range[1] - byte_range[0] + 1:
                    raise InputError("Recovery range body is truncated")
            self.total += len(raw)
            name = f"payload/{len(self.ledger):03}.bin"
            (self.root / name).write_bytes(raw)
            entry = {
                "url": canonical,
                "range": list(byte_range) if byte_range is not None else None,
                "status": status,
                "total_size": total_size,
                "path": name,
                "size_bytes": len(raw),
                "sha256": _sha(raw),
            }
            self.ledger.append(entry)
            return raw, total_size
        except (URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            raise InputError("Public recovery input could not be read within its bounds") from exc


class _OfflineTransport:
    def __init__(self, root, ledger):
        self.root, self.ledger = root, ledger
        self.total = sum(entry["size_bytes"] for entry in ledger)
        self.used = set()

    def fetch(self, url, *, limit, byte_range=None):
        matches = [
            e
            for e in self.ledger
            if e["url"] == url and e["range"] == (list(byte_range) if byte_range is not None else None)
        ]
        if len(matches) != 1:
            raise InputError("Offline recovery requires exactly one retained native response for each read")
        entry = matches[0]
        raw = _inside(self.root, entry["path"]).read_bytes()
        if len(raw) > limit or len(raw) != entry["size_bytes"] or _sha(raw) != entry["sha256"]:
            raise InputError("Offline recovery response differs from retained bytes")
        self.used.add(entry["path"])
        return raw, entry["total_size"]


class _Media:
    def __init__(self, transport=None, url=None, raw=None):
        self.transport, self.url, self.raw = transport, url, raw
        self.segments = []
        if raw is not None:
            self.size = len(raw)
        else:
            head, self.size = transport.fetch(url, limit=64, byte_range=(0, 63))
            self.segments.append((0, head))
        self.media_extents = []

    def read(self, offset, count):
        if (
            type(offset) is not int
            or type(count) is not int
            or not 0 <= offset <= self.size
            or not 0 <= count <= MAX_RESPONSE
            or offset + count > self.size
        ):
            raise _Unsupported("Native media read escapes its bounded file")
        if self.raw is not None:
            return self.raw[offset : offset + count]
        for start, data in self.segments:
            if start <= offset and offset + count <= start + len(data):
                return data[offset - start : offset - start + count]
        if not count:
            return b""
        raw, size = self.transport.fetch(self.url, limit=count, byte_range=(offset, offset + count - 1))
        if size != self.size:
            raise InputError("Media total size changed between pinned range reads")
        for start, data in self.segments:
            left, right = max(start, offset), min(start + len(data), offset + count)
            if left < right and data[left - start : right - start] != raw[left - offset : right - offset]:
                raise InputError("Overlapping native media ranges disagree")
        self.segments.append((offset, raw))
        return raw


def _boxes(raw, start=0, end=None):
    end = len(raw) if end is None else end
    out, offset = [], start
    while offset < end:
        if len(out) >= 256 or end - offset < 8:
            raise _Unsupported("MP4 box count or exact extent is unsupported")
        size, kind = struct.unpack_from(">I4s", raw, offset)
        header = 8
        if size == 1:
            if end - offset < 16:
                raise _Unsupported("Truncated extended MP4 box")
            size, header = struct.unpack_from(">Q", raw, offset + 8)[0], 16
        if size == 0:
            size = end - offset
        if size < header or offset + size > end:
            raise _Unsupported("MP4 box escapes its parent extent")
        out.append((kind, offset, offset + header, offset + size))
        offset += size
    return out


def _one(raw, parent, kind, *, optional=False):
    values = [b for b in _boxes(raw, parent[2], parent[3]) if b[0] == kind]
    if not values and optional:
        return None
    if len(values) != 1:
        raise _Unsupported(f"MP4 requires exactly one {kind.decode(errors='replace')} box")
    return values[0]


def _full(raw, box, versions=(0,)):
    if box[3] - box[2] < 4 or raw[box[2]] not in versions or raw[box[2] + 1 : box[2] + 4] != b"\0\0\0":
        raise _Unsupported("Unsupported MP4 full-box version or flags")
    return raw[box[2]]


def _table(raw, box, width, *, versions=(0,)):
    _full(raw, box, versions)
    if box[3] - box[2] < 8:
        raise _Unsupported("Truncated MP4 table count")
    count = struct.unpack_from(">I", raw, box[2] + 4)[0]
    if count > MAX_SAMPLES or box[3] - box[2] != 8 + count * width:
        raise _Unsupported("MP4 table cardinality or exact byte extent is invalid")
    return count, box[2] + 8


def _parse_mp4(media):
    offset, moov, top_count, ftyp_count = 0, None, 0, 0
    while offset < media.size:
        top_count += 1
        if top_count > 32 or media.size - offset < 8:
            raise _Unsupported("Unsupported top-level MP4 structure")
        header = media.read(offset, 8)
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            size, header_size = struct.unpack(">Q", media.read(offset + 8, 8))[0], 16
        if not size:
            size = media.size - offset
        if size < header_size or offset + size > media.size:
            raise _Unsupported("MP4 top-level box escapes its file")
        if kind in (b"moof", b"sidx"):
            raise _Unsupported("Fragmented MP4 recovery is unsupported")
        if kind == b"ftyp":
            ftyp_count += 1
            if offset or not 16 <= size <= 256 or (size - header_size) % 4:
                raise _Unsupported("Unsupported native MP4 file-type declaration")
        if kind == b"mdat":
            media.media_extents.append((offset + header_size, offset + size))
        if kind == b"moov":
            if moov is not None or size > MAX_MOOV:
                raise _Unsupported("MP4 requires one bounded complete moov box")
            moov = media.read(offset, size)
        offset += size
    if moov is None or not media.media_extents or ftyp_count != 1:
        raise _Unsupported("MP4 lacks complete media metadata or payload extents")
    outer = _boxes(moov)
    if len(outer) != 1 or outer[0][0] != b"moov":
        raise _Unsupported("Invalid retained moov metadata")
    parent = outer[0]
    children = _boxes(moov, parent[2], parent[3])
    if any(b[0] == b"mvex" for b in children):
        raise _Unsupported("Fragmented MP4 sample tables are unsupported")
    tracks = [b for b in children if b[0] == b"trak"]
    if len(tracks) != 1:
        raise _Unsupported("Recovery supports exactly one native video track")
    track = tracks[0]
    mdia = _one(moov, track, b"mdia")
    hdlr = _one(moov, mdia, b"hdlr")
    if hdlr[3] - hdlr[2] < 12 or moov[hdlr[2] + 8 : hdlr[2] + 12] != b"vide":
        raise _Unsupported("Native track is not a video track")
    mdhd = _one(moov, mdia, b"mdhd")
    version = _full(moov, mdhd, (0, 1))
    expected_min = 36 if version else 24
    if mdhd[3] - mdhd[2] != expected_min:
        raise _Unsupported("Unsupported media-header exact extent")
    scale_offset = mdhd[2] + (20 if version else 12)
    timescale = struct.unpack_from(">I", moov, scale_offset)[0]
    duration = struct.unpack_from(">Q" if version else ">I", moov, scale_offset + 4)[0]
    if not timescale or duration > 2**53:
        raise _Unsupported("Invalid bounded native media timescale/duration")
    minf = _one(moov, mdia, b"minf")
    dref = _one(moov, _one(moov, minf, b"dinf"), b"dref")
    _full(moov, dref)
    if dref[3] - dref[2] < 8 or struct.unpack_from(">I", moov, dref[2] + 4)[0] != 1:
        raise _Unsupported("Require one self-contained native data reference")
    refs = _boxes(moov, dref[2] + 8, dref[3])
    if len(refs) != 1 or refs[0][0] != b"url " or moov[refs[0][2] : refs[0][3]] != b"\0\0\0\1":
        raise _Unsupported("External or ambiguous native media data references are unsupported")
    stbl = _one(moov, minf, b"stbl")
    stsd = _one(moov, stbl, b"stsd")
    _full(moov, stsd)
    if stsd[3] - stsd[2] < 8 or struct.unpack_from(">I", moov, stsd[2] + 4)[0] != 1:
        raise _Unsupported("Multiple native sample descriptions are unsupported")
    entries = _boxes(moov, stsd[2] + 8, stsd[3])
    if len(entries) != 1 or entries[0][0] not in (b"av01", b"avc1", b"hvc1", b"hev1", b"vp09"):
        raise _Unsupported("Unsupported native visual sample description")
    entry = entries[0]
    if entry[3] - entry[2] < 78 or struct.unpack_from(">H", moov, entry[2] + 6)[0] != 1:
        raise _Unsupported("Unsupported visual sample-entry extent/reference")
    width, height = struct.unpack_from(">HH", moov, entry[2] + 24)
    if not 0 < width <= 16384 or not 0 < height <= 16384:
        raise _Unsupported("Invalid bounded video dimensions")
    config_kind = {b"av01": b"av1C", b"avc1": b"avcC", b"hvc1": b"hvcC", b"hev1": b"hvcC", b"vp09": b"vpcC"}[
        entry[0]
    ]
    properties = _boxes(moov, entry[2] + 78, entry[3])
    configs = [b for b in properties if b[0] == config_kind]
    if len(configs) != 1 or configs[0][3] - configs[0][2] > 65536:
        raise _Unsupported("Missing or ambiguous bounded codec initialization")
    config = configs[0]
    configuration = moov[config[2] : config[3]]
    minimum = {b"av1C": 4, b"avcC": 7, b"hvcC": 23, b"vpcC": 12}[config_kind]
    if len(configuration) < minimum:
        raise _Unsupported("Codec initialization is empty or structurally truncated")
    if config_kind == b"av1C" and (
        configuration[0] != 0x81 or configuration[1] >> 5 > 2 or configuration[3] & 0xE0
    ):
        raise _Unsupported("Unsupported AV1 codec initialization marker/version/profile")
    if config_kind in (b"avcC", b"hvcC") and configuration[0] != 1:
        raise _Unsupported("Unsupported codec initialization version")
    codec = {
        "sample_entry": entry[0].decode(),
        "width": width,
        "height": height,
        "configuration_sha256": _sha(moov[config[1] : config[3]]),
        "presentation_properties": [
            {"type": b[0].decode(errors="replace"), "sha256": _sha(moov[b[1] : b[3]])}
            for b in properties
            if b[0] != b"btrt"
        ],
    }
    stsz = _one(moov, stbl, b"stsz")
    _full(moov, stsz)
    if stsz[3] - stsz[2] < 12:
        raise _Unsupported("Truncated native sample-size table")
    fixed, count = struct.unpack_from(">II", moov, stsz[2] + 4)
    if not 0 < count <= MAX_SAMPLES or stsz[3] - stsz[2] != 12 + (0 if fixed else 4 * count):
        raise _Unsupported("Invalid native sample-size cardinality/extent")
    sizes = [fixed] * count if fixed else list(struct.unpack_from(">" + "I" * count, moov, stsz[2] + 12))
    if any(not 0 < size <= MAX_RESPONSE for size in sizes):
        raise _Unsupported("Invalid or unbounded native sample size")
    stts = _one(moov, stbl, b"stts")
    n, start = _table(moov, stts, 8)
    durations = []
    for i in range(n):
        samples, delta = struct.unpack_from(">II", moov, start + 8 * i)
        if not samples or not delta or len(durations) + samples > count:
            raise _Unsupported("Invalid native decode-time table counts")
        durations.extend([delta] * samples)
    if len(durations) != count or sum(durations) != duration:
        raise _Unsupported("Native decode times differ from sample/media cardinality")
    ctts = _one(moov, stbl, b"ctts", optional=True)
    composition = [0] * count
    if ctts:
        n, start = _table(moov, ctts, 8, versions=(0, 1))
        composition = []
        code = ">Ii" if moov[ctts[2]] else ">II"
        for i in range(n):
            samples, delta = struct.unpack_from(code, moov, start + 8 * i)
            if not samples or len(composition) + samples > count:
                raise _Unsupported("Invalid native composition-time counts")
            composition.extend([delta] * samples)
        if len(composition) != count:
            raise _Unsupported("Native composition-time cardinality differs")
    edit_time, edit_duration = 0, None
    edts = _one(moov, track, b"edts", optional=True)
    if edts:
        elst = _one(moov, edts, b"elst")
        version = _full(moov, elst, (0, 1))
        n, start = _table(moov, elst, 20 if version else 12, versions=(0, 1))
        if n != 1:
            raise _Unsupported("Recovery requires one unit-rate movie edit")
        segment_duration, edit_time, integer, fraction = struct.unpack_from(
            ">Qqhh" if version else ">Iihh", moov, start
        )
        if not segment_duration or edit_time < 0 or integer != 1 or fraction:
            raise _Unsupported("Unsupported empty or non-unit native movie edit")
        mvhd = _one(moov, parent, b"mvhd")
        movie_version = _full(moov, mvhd, (0, 1))
        if mvhd[3] - mvhd[2] != (112 if movie_version else 100):
            raise _Unsupported("Unsupported movie timescale header extent")
        movie_scale = struct.unpack_from(">I", moov, mvhd[2] + (20 if movie_version else 12))[0]
        if not movie_scale:
            raise _Unsupported("Invalid movie edit timescale")
        edit_duration = Fraction(segment_duration, movie_scale)
    stsc = _one(moov, stbl, b"stsc")
    n, start = _table(moov, stsc, 12)
    mapping = [struct.unpack_from(">III", moov, start + 12 * i) for i in range(n)]
    if (
        not mapping
        or mapping[0][0] != 1
        or any(not b or description != 1 for _, b, description in mapping)
        or any(a[0] >= b[0] for a, b in zip(mapping, mapping[1:], strict=False))
    ):
        raise _Unsupported("Unsupported native sample-to-chunk map")
    offsets_boxes = [b for b in _boxes(moov, stbl[2], stbl[3]) if b[0] in (b"stco", b"co64")]
    if len(offsets_boxes) != 1:
        raise _Unsupported("Require one unambiguous native chunk-offset table")
    offsets_box = offsets_boxes[0]
    width = 4 if offsets_box[0] == b"stco" else 8
    n, start = _table(moov, offsets_box, width)
    offsets = list(struct.unpack_from(">" + ("I" if width == 4 else "Q") * n, moov, start))
    if (
        not offsets
        or mapping[-1][0] > len(offsets)
        or any(a >= b for a, b in zip(offsets, offsets[1:], strict=False))
    ):
        raise _Unsupported("Invalid native chunk-offset ordering/cardinality")
    samples, current, previous_end = [], 0, -1
    for chunk, chunk_offset in enumerate(offsets, 1):
        while current + 1 < len(mapping) and mapping[current + 1][0] <= chunk:
            current += 1
        amount = mapping[current][1]
        if len(samples) + amount > count:
            raise _Unsupported("Native chunks contain more samples than declared")
        chunk_end = chunk_offset + sum(sizes[len(samples) : len(samples) + amount])
        if chunk_offset < previous_end or not any(
            left <= chunk_offset < chunk_end <= right for left, right in media.media_extents
        ):
            raise _Unsupported("Native chunk/sample bytes escape media extents or overlap")
        position = chunk_offset
        for _ in range(amount):
            size = sizes[len(samples)]
            samples.append((position, size))
            position += size
        previous_end = chunk_end
    if len(samples) != count:
        raise _Unsupported("Native chunks do not cover every declared sample")
    sync = _one(moov, stbl, b"stss", optional=True)
    if sync:
        n, start = _table(moov, sync, 4)
        indices = list(struct.unpack_from(">" + "I" * n, moov, start))
        if (
            not indices
            or indices[0] != 1
            or indices[-1] > count
            or any(a >= b for a, b in zip(indices, indices[1:], strict=False))
        ):
            raise _Unsupported("Native sync-sample table does not establish an initial decode boundary")
    times, dts = [], 0
    for delta, offset in zip(durations, composition, strict=True):
        times.append((Fraction(dts + offset - edit_time, timescale), Fraction(delta, timescale)))
        dts += delta
    if edit_duration is not None and any(
        point < 0 or point + length > edit_duration for point, length in times
    ):
        raise _Unsupported("Native movie edit trims declared sample presentation intervals")
    return {
        "codec": codec,
        "samples": samples,
        "times": times,
        "sample_count": count,
        "media_duration": Fraction(duration, timescale),
    }


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputError("Install the dataset extra for native Parquet recovery metadata") from exc
    return pa, pq


def _complete_content(transport, spec, path, *, original=False, limit=MAX_METADATA):
    """Check complete bytes against a retained Hub listing, not an origin signature.

    Live intake relies on authenticated HTTPS to the allowlisted Hub. Offline replay
    rechecks only the supplied listing/body relationship; both can be resealed.
    Partial large-video ranges deliberately do not use this whole-blob verifier.
    """
    revision = spec["original_revision"] if original else spec["revision"]
    api = f"https://huggingface.co/api/datasets/{spec['repo_id']}/revision/{revision}?blobs=true"
    cache = getattr(transport, "content_listings", None)
    if cache is None:
        transport.content_listings = cache = {}
        transport.complete_content_identities = []
    if api not in cache:
        raw, _ = transport.fetch(api, limit=MAX_METADATA)
        listing = _json(raw)
        if (
            type(listing) is not dict
            or listing.get("id") != spec["repo_id"]
            or listing.get("sha") != revision
            or listing.get("private") is not False
            or listing.get("gated") not in (None, False)
            or type(listing.get("siblings")) is not list
            or not 1 <= len(listing["siblings"]) <= 50000
        ):
            raise InputError("Pinned content ID listing has invalid repository/revision/public identity")
        entries = {}
        for entry in listing["siblings"]:
            if type(entry) is not dict or type(entry.get("rfilename")) is not str:
                raise InputError("Malformed pinned content ID listing")
            name = _native_path(entry["rfilename"])
            if name.casefold() in entries:
                raise InputError("Duplicate paths in pinned content ID listing")
            entries[name.casefold()] = entry
        cache[api] = entries
    entry = cache[api].get(path.casefold())
    if entry is None or entry["rfilename"] != path:
        raise InputError("Selected native file is absent from pinned content ID listing")
    size = entry.get("size")
    if type(size) is not int or not 0 < size <= limit:
        raise InputError("Pinned content ID listing lacks a supported exact size")
    lfs = entry.get("lfs")
    if lfs is not None:
        digest = lfs.get("sha256") if type(lfs) is dict else None
        if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise InputError("Pinned content ID listing lacks an LFS SHA256")
        algorithm = "sha256"
    else:
        digest = entry.get("blobId")
        if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise InputError("Pinned content ID listing lacks a git blob ID")
        algorithm = "git_blob_sha1"
    raw, _ = transport.fetch(_url(spec, path, original=original), limit=limit)
    actual = _sha(raw) if lfs is not None else hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    if len(raw) != size or actual != digest:
        raise InputError("Complete native bytes differ from pinned listing content ID or size")
    identity = {"path": path, "revision": revision, "size_bytes": size,
                "algorithm": algorithm, "content_id": digest, "listing_url": api}
    transport.complete_content_identities.append(identity)
    return raw


def _metadata(transport, spec):
    _, pq = _arrow()
    info_raw = _complete_content(transport, spec, "meta/info.json")
    old_info_raw = _complete_content(transport, spec, "meta/info.json", original=True)
    info, old_info = _json(info_raw), _json(old_info_raw)
    if type(info) is not dict or type(old_info) is not dict:
        raise _Unsupported("Native dataset info must be an object")
    fps = info.get("fps")
    if (
        type(fps) not in (int, float)
        or not math.isfinite(fps)
        or not 0 < fps <= 1000
        or old_info.get("fps") != fps
    ):
        raise _Unsupported("Original and current native fps declarations must agree")
    key = spec["video_key"]
    for description in (info, old_info):
        features = description.get("features")
        if type(features) is not dict:
            raise _Unsupported("Native feature declarations must be an object")
        feature = features.get(key)
        if type(feature) is not dict or feature.get("dtype") != "video":
            raise _Unsupported("The selected key is not a native video feature in both revisions")
    if not str(info.get("codebase_version", "")).startswith("v3") or not str(
        old_info.get("codebase_version", "")
    ).startswith("v2.1"):
        raise _Unsupported("Recovery currently supports native v3 references to original v2.1 episode clips")
    tree_url = f"https://huggingface.co/api/datasets/{spec['repo_id']}/tree/{spec['revision']}/meta/episodes?recursive=true&limit=1000"
    tree_raw, _ = transport.fetch(tree_url, limit=MAX_METADATA)
    tree = _json(tree_raw)
    if type(tree) is not list or len(tree) > 128:
        raise _Unsupported("Native episode metadata listing exceeds its supported bound")
    paths = []
    for item in tree:
        if type(item) is not dict or item.get("type") not in ("file", "directory"):
            raise _Unsupported("Malformed native episode metadata listing")
        if item["type"] == "file":
            name = _native_path(item.get("path"))
            if not re.fullmatch(r"meta/episodes/chunk-\d{3}/file-\d{3}\.parquet", name):
                raise _Unsupported("Unsupported native episode metadata layout")
            size = item.get("size")
            if type(size) is not int or not 0 < size <= MAX_METADATA:
                raise _Unsupported("Native episode metadata exceeds its byte bound")
            paths.append(name)
    if not paths or len(paths) > 16 or len(set(paths)) != len(paths):
        raise _Unsupported("Require bounded unique native episode metadata files")
    selected, all_rows, metadata_bytes = None, [], 0
    columns = [
        "episode_index",
        "length",
        f"videos/{key}/chunk_index",
        f"videos/{key}/file_index",
        f"videos/{key}/from_timestamp",
        f"videos/{key}/to_timestamp",
    ]
    for path in sorted(paths):
        raw = _complete_content(transport, spec, path)
        metadata_bytes += len(raw)
        if metadata_bytes > MAX_METADATA:
            raise _Unsupported("Combined native episode metadata exceeds 4 MiB")
        try:
            from .native_parquet_guard import require_unencrypted_parquet

            require_unencrypted_parquet(io.BytesIO(raw))
            file = pq.ParquetFile(io.BytesIO(raw))
            if (
                file.metadata.num_rows > MAX_SAMPLES
                or file.metadata.num_columns > 512
                or file.metadata.num_row_groups > MAX_SAMPLES
            ):
                raise _Unsupported("Native episode metadata exceeds bounded dimensions")
            if not set(columns).issubset(file.schema_arrow.names):
                raise _Unsupported("Selected native camera interval columns are absent")
            expanded = sum(
                file.metadata.row_group(i).column(j).total_uncompressed_size
                for i in range(file.metadata.num_row_groups)
                for j in range(file.metadata.num_columns)
            )
            if expanded > MAX_RESPONSE:
                raise _Unsupported("Expanded native metadata exceeds 32 MiB")
            rows = file.read(columns=columns).to_pylist()
        except (ValueError, OSError, TypeError) as exc:
            raise _Unsupported("Native Parquet metadata cannot be read within supported bounds") from exc
        if len(all_rows) + len(rows) > MAX_SAMPLES:
            raise _Unsupported("Native episode metadata exceeds 100000 rows")
        for index, row in enumerate(rows):
            if any(type(row.get(k)) is not int or row[k] < 0 for k in columns[:4]):
                raise _Unsupported("Native episode identity/length/file columns must be nonnegative integers")
            if not row["length"] or any(
                type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                for k in columns[4:]
            ):
                raise _Unsupported("Native interval times/length are invalid")
            if row["episode_index"] == spec["episode_index"]:
                if selected is not None:
                    raise _Unsupported("Native episode identity is duplicated")
                selected = {"path": path, "row_index": index, "row": row, "raw": raw}
            all_rows.append(row)
    if len({r["episode_index"] for r in all_rows}) != len(all_rows) or selected is None:
        raise _Unsupported("Native episode identities are duplicated or selected episode is absent")
    return info, old_info, Fraction(str(fps)), selected, all_rows


def _video_path(template, *, video_key, chunk_index, file_index, episode_index=None, chunk_size=None):
    if type(template) is not str or len(template) > 256:
        raise _Unsupported("Native video path template is absent or unbounded")
    allowed = {"video_key", "chunk_index", "file_index", "episode_index", "episode_chunk"}
    import string

    try:
        fields = list(string.Formatter().parse(template))
        if episode_index is not None and not any(field == "episode_index" for _, field, _, _ in fields):
            raise _Unsupported("Original video path template must bind episode_index")
        if any(
            field is not None
            and (
                field not in allowed
                or conversion is not None
                or spec not in ("", "d", "03d", "06d")
                or field == "video_key"
                and spec
            )
            for _, field, spec, conversion in fields
        ):
            raise _Unsupported("Native video path template uses an unsupported field/format")
        result = template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_index=episode_index,
            episode_chunk=episode_index // chunk_size if episode_index is not None and chunk_size else None,
        )
    except _Unsupported:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise _Unsupported("Native video path template cannot be resolved safely") from exc
    return _native_path(result)


def _interval(row, key, fps):
    start, end = (
        Fraction(str(row[f"videos/{key}/{name}"])) * fps for name in ("from_timestamp", "to_timestamp")
    )
    first, last = round(start), round(end)
    if (
        abs(start - first) > Fraction(1, 100000)
        or abs(end - last) > Fraction(1, 100000)
        or last - first != row["length"]
        or not 0 <= first < last
    ):
        raise _Unsupported("Native interval is not a complete declared frame-grid interval")
    return first, last


def _evaluate(transport, spec):
    report = {
        "schema_version": 1,
        "kind": "video_reference_recovery",
        "status": "unknown",
        **spec,
        "original_status": "unknown",
        "repair_verified": False,
        "origin_verified": False,
        "verification_scope": _VERIFICATION_SCOPE,
        "patch": [],
        "unknowns": [],
        "scope": _SCOPE,
    }
    try:
        info, old_info, fps, selected, rows = _metadata(transport, spec)
        row, key = selected["row"], spec["video_key"]
        prefix = f"videos/{key}/"
        first, last = _interval(row, key, fps)
        chunk_index, file_index = row[prefix + "chunk_index"], row[prefix + "file_index"]
        current_path = _video_path(
            info.get("video_path"), video_key=key, chunk_index=chunk_index, file_index=file_index
        )
        current = _Media(transport, _url(spec, current_path))
        current_table = _parse_mp4(current)
        report["original_reference"] = {
            "path": current_path,
            "from_sample_inclusive": first,
            "to_sample_exclusive": last,
            "available_samples": current_table["sample_count"],
        }
        if last <= current_table["sample_count"]:
            report["original_status"] = "pass"
            raise _Unsupported("Selected original interval is in bounds; no recovery is justified")
        report["original_status"] = "error"
        report["original_reference"]["finding"] = "Declared interval exceeds native MP4 sample table"
        old_chunks = old_info.get("chunks_size")
        if type(old_chunks) is not int or not 0 < old_chunks <= MAX_SAMPLES:
            raise _Unsupported("Original native episode chunk size is unsupported")
        original_path = _video_path(
            old_info.get("video_path"),
            video_key=key,
            chunk_index=0,
            file_index=0,
            episode_index=spec["episode_index"],
            chunk_size=old_chunks,
        )
        original_raw = _complete_content(transport, spec, original_path, original=True, limit=MAX_RESPONSE)
        original = _Media(raw=original_raw)
        original_table = _parse_mp4(original)
        if original_table["sample_count"] != row["length"]:
            raise _Unsupported("Original short clip length differs from the selected native episode")
        if original_table["times"] != [
            (Fraction(i, 1) / fps, Fraction(1, 1) / fps) for i in range(row["length"])
        ]:
            raise _Unsupported(
                "Original clip presentation intervals do not establish the native zero-based frame grid"
            )
        next_index = file_index + 1
        chunks = info.get("chunks_size", 1000)
        if type(chunks) is not int or not 0 < chunks <= MAX_SAMPLES:
            raise _Unsupported("Native shard chunk size is unsupported")
        next_chunk = chunk_index
        if next_index >= chunks:
            next_chunk, next_index = chunk_index + 1, 0
        if next_chunk != chunk_index:
            raise _Unsupported("This recovery criterion does not alter chunk indices")
        target_path = _video_path(
            info.get("video_path"), video_key=key, chunk_index=next_chunk, file_index=next_index
        )
        target = _Media(transport, _url(spec, target_path))
        target_table = _parse_mp4(target)
        count = row["length"]
        if target_table["sample_count"] < count or original_table["codec"] != target_table["codec"]:
            raise _Unsupported(
                "Next native shard has insufficient samples or different codec/presentation metadata"
            )
        if original_table["times"] != target_table["times"][:count]:
            raise _Unsupported("Next native shard does not preserve every original presentation interval")
        target_samples = target_table["samples"][:count]
        if [n for _, n in original_table["samples"]] != [n for _, n in target_samples]:
            raise _Unsupported("Next native shard sample sizes differ from the original clip")
        end = max(offset + size for offset, size in target_samples)
        if end > MAX_RESPONSE:
            raise _Unsupported("Complete original clip identity requires an unbounded next-shard prefix")
        target.read(0, end)
        witnesses = []
        for index, ((a, n), (b, m)) in enumerate(zip(original_table["samples"], target_samples, strict=True)):
            left, right = original.read(a, n), target.read(b, m)
            if left != right:
                raise _Unsupported(
                    "Next native shard compressed sample identity differs from the original clip"
                )
            witnesses.append(
                {
                    "sample_index": index,
                    "original_offset": a,
                    "target_offset": b,
                    "size_bytes": n,
                    "sha256": _sha(left),
                    "presentation_time": str(original_table["times"][index][0]),
                    "duration": str(original_table["times"][index][1]),
                }
            )
        finish = Fraction(count, 1) / fps
        for other in rows:
            if other["episode_index"] == row["episode_index"]:
                continue
            if other[prefix + "chunk_index"] == next_chunk and other[prefix + "file_index"] == next_index:
                other_first, other_last = _interval(other, key, fps)
                if max(0, other_first) < min(count, other_last):
                    raise _Unsupported("Proposed recovered interval overlaps another declared native episode")
        patch = [
            {
                "path": selected["path"],
                "row": selected["row_index"],
                "episode_index": row["episode_index"],
                "column": prefix + name,
                "old": row[prefix + name],
                "new": value,
            }
            for name, value in [
                ("file_index", next_index),
                ("from_timestamp", 0.0),
                ("to_timestamp", float(finish)),
            ]
        ]
        report.update(
            status="pass",
            repair_verified=True,
            patch=patch,
            reason="Every original compressed sample, codec and presentation interval identifies the next native shard prefix",
            recovered_reference={
                "path": target_path,
                "from_sample_inclusive": 0,
                "to_sample_exclusive": count,
                "from_timestamp": 0.0,
                "to_timestamp": float(finish),
            },
            evidence={
                "sample_count": count,
                "all_samples_match": True,
                "codec_match": True,
                "timestamps_match": True,
                "original_clip_path": original_path,
                "original_clip_sha256": _sha(original_raw),
                "original_metadata_path": selected["path"],
                "original_metadata_sha256": _sha(selected["raw"]),
                "codec": original_table["codec"],
                "sample_witnesses": witnesses,
                "complete_content_identities": transport.complete_content_identities,
            },
            unknowns=[
                "Other episodes, complete target media integrity and policy performance remain unverified"
            ],
        )
        return report, selected
    except _Unsupported as exc:
        report.update(reason=str(exc), unknowns=[str(exc)])
        return report, None


def _corrected_metadata(selected, patch):
    pa, pq = _arrow()
    table = pq.read_table(io.BytesIO(selected["raw"]))
    before = table.to_pylist()
    after = table
    for change in patch:
        position = table.schema.get_field_index(change["column"])
        field = table.schema.field(position)
        values = table[change["column"]].to_pylist()
        if values[change["row"]] != change["old"]:
            raise InputError("Native metadata changed before its derived correction")
        values[change["row"]] = change["new"]
        after = after.set_column(position, field, pa.array(values, type=field.type))
    corrected = after.to_pylist()
    differences = {
        (i, key)
        for i, (left, right) in enumerate(zip(before, corrected, strict=True))
        for key in left
        if left[key] != right[key]
    }
    if differences != {(p["row"], p["column"]) for p in patch} or not table.schema.equals(
        after.schema, check_metadata=True
    ):
        raise InputError("Derived metadata changed cells/schema outside the reviewed proposal")
    output = io.BytesIO()
    pq.write_table(after, output)
    raw = output.getvalue()
    if len(raw) > MAX_METADATA or not pq.read_table(io.BytesIO(raw)).equals(after, check_metadata=True):
        raise InputError("Derived native metadata does not round-trip within its byte bound")
    return raw


def _fingerprint():
    root = Path(__file__).parent
    names = ("video_recovery.py", "hub.py", "incidents.py", "publication.py", "artifacts.py")
    try:
        version = importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "engine_version": __version__,
        "python_version": list(sys.version_info[:2]),
        "module_sha256": {name: _sha((root / name).read_bytes()) for name in names},
        "pyarrow_version": version,
    }


def _summary(report, **extra):
    output = {key: value for key, value in report.items() if key != "evidence"}
    if "evidence" in report:
        evidence = report["evidence"]
        witnesses = evidence["sample_witnesses"]
        output["evidence"] = {key: value for key, value in evidence.items() if key != "sample_witnesses"}
        output["evidence"].update(
            sample_witnesses=witnesses[:2], omitted_sample_witnesses=max(0, len(witnesses) - 2)
        )
    output.update(stored_full_report_sha256=_sha(_encode(report)), detail_complete=False, **extra)
    if len(_encode(output)) > 16384:
        raise InputError("Recovery preview exceeds its 16 KiB bound")
    return output


def _inventory_checked(root):
    inventory = _inventory(root, case=True)
    if sum(size for _, size in inventory.values()) > MAX_TOTAL:
        raise InputError("Published recovery case exceeds 64 MiB")
    for path in root.rglob("*"):
        if path.is_dir() and not any(
            name.startswith(path.relative_to(root).as_posix() + "/") for name in inventory
        ):
            raise InputError("Unbound empty recovery directories are refused")
    return inventory


def recover_video_reference(repo_id, revision, original_revision, episode_index, video_key, destination, *, transport_factory=None):
    """Explicit bounded public Hub intake; publish a proposal, never modify originals."""
    spec = _request_spec(repo_id, revision, original_revision, episode_index, video_key)
    target = _path(destination)
    if target.exists():
        raise InputError("Recovery destination already exists; choose a new directory")
    if any((p / "READY.json").exists() and (p / "manifest.json").exists() for p in target.parents):
        raise InputError("Recovery destination cannot be nested inside an existing retained case")
    _arrow()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".policy-doctor-video-recovery-", dir=target.parent))
    try:
        verifier = _fingerprint()
        transport = (transport_factory or _Transport)(staging)
        report, selected = _evaluate(transport, spec)
        (staging / "request.json").write_bytes(_encode(spec))
        (staging / "ledger.json").write_bytes(_encode(transport.ledger))
        (staging / "report.json").write_bytes(_encode(report))
        (staging / "repair_plan.json").write_bytes(_encode(report["patch"]))
        if selected is not None:
            (staging / "corrected_metadata.parquet").write_bytes(
                _corrected_metadata(selected, report["patch"])
            )
        if _fingerprint() != verifier:
            raise InputError("Recovery verifier changed during acquisition")
        inventory = _inventory_checked(staging)
        manifest = {
            "schema_version": 1,
            "kind": KIND,
            "verifier": verifier,
            "files": {name: digest for name, (digest, _) in sorted(inventory.items())},
        }
        raw = _encode(manifest)
        (staging / "manifest.json").write_bytes(raw)
        (staging / "READY.json").write_bytes(_encode({"schema_version": 1, "manifest_sha256": _sha(raw)}))
        _load(staging)
        response = _summary(report, case_directory=str(target), acquired_bytes=transport.total)
        publish_capture(staging, target)
        return response
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load(root):
    if not root.is_dir():
        raise InputError("Require a local video-recovery case directory")
    inventory = _inventory_checked(root)
    if not {
        "manifest.json",
        "READY.json",
        "request.json",
        "ledger.json",
        "report.json",
        "repair_plan.json",
    }.issubset(inventory):
        raise InputError("Recovery case is incomplete")
    manifest = _json((root / "manifest.json").read_bytes())
    ready = _json((root / "READY.json").read_bytes())
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_KEYS
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["kind"] != KIND
    ):
        raise InputError("Malformed video-recovery manifest")
    if type(manifest["verifier"]) is not dict:
        raise InputError("Malformed video-recovery verifier identity")
    if (
        type(ready) is not dict
        or set(ready) != {"schema_version", "manifest_sha256"}
        or type(ready["schema_version"]) is not int
        or ready["schema_version"] != 1
        or ready["manifest_sha256"] != inventory["manifest.json"][0]
    ):
        raise InputError("Video-recovery READY digest mismatch")
    files = manifest["files"]
    if type(files) is not dict or set(files) | {"manifest.json", "READY.json"} != set(inventory):
        raise InputError("Missing or unlisted video-recovery evidence")
    for name, digest in files.items():
        _inside(root, name)
        if type(digest) is not str or digest != inventory[name][0]:
            raise InputError("Video-recovery native evidence hash mismatch")
    request = _json((root / "request.json").read_bytes())
    if type(request) is not dict or set(request) != {
        "repo_id",
        "revision",
        "original_revision",
        "episode_index",
        "video_key",
        "criterion",
    }:
        raise InputError("Malformed frozen recovery request")
    if request != _request_spec(**{key: value for key, value in request.items() if key != "criterion"}):
        raise InputError("Frozen recovery criterion differs from the supported version")
    ledger = _json((root / "ledger.json").read_bytes())
    if type(ledger) is not list or not 1 <= len(ledger) <= 32:
        raise InputError("Malformed bounded recovery response ledger")
    paths, requests = set(), set()
    for entry in ledger:
        if type(entry) is not dict or set(entry) != {
            "url",
            "range",
            "status",
            "total_size",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise InputError("Malformed native recovery response entry")
        if type(entry["url"]) is not str:
            raise InputError("Recovery ledger may contain only canonical public Hub request URLs")
        parsed = urlsplit(entry["url"])
        prefixes = [
            f"/datasets/{request['repo_id']}/resolve/{revision}/"
            for revision in (request["revision"], request["original_revision"])
        ]
        tree = f"https://huggingface.co/api/datasets/{request['repo_id']}/tree/{request['revision']}/meta/episodes?recursive=true&limit=1000"
        listings = {
            f"https://huggingface.co/api/datasets/{request['repo_id']}/revision/{revision}?blobs=true"
            for revision in (request["revision"], request["original_revision"])
        }
        ordinary = (
            parsed.scheme == "https"
            and parsed.netloc == "huggingface.co"
            and not parsed.query
            and not parsed.fragment
            and not parsed.username
            and not parsed.password
            and any(parsed.path.startswith(p) for p in prefixes)
        )
        if entry["url"] != tree and entry["url"] not in listings and not ordinary:
            raise InputError("Recovery ledger URL differs from the frozen public native request scope")
        native = _inside(root, entry["path"])
        if not re.fullmatch(r"payload/\d{3}\.bin", entry["path"]) or entry["path"] in paths:
            raise InputError("Duplicate or invalid native recovery response file")
        paths.add(entry["path"])
        key = (entry["url"], _encode(entry["range"]))
        if key in requests:
            raise InputError("Duplicate native recovery requests are refused")
        requests.add(key)
        if type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= MAX_RESPONSE:
            raise InputError("Invalid retained recovery response byte size")
        selected_range = entry["range"]
        if selected_range is None:
            if type(entry["status"]) is not int or entry["status"] != 200 or entry["total_size"] is not None:
                raise InputError("Invalid retained complete-response evidence")
        elif (
            type(selected_range) is not list
            or len(selected_range) != 2
            or any(type(v) is not int for v in selected_range)
            or not 0 <= selected_range[0] <= selected_range[1]
            or type(entry["status"]) is not int
            or entry["status"] != 206
            or type(entry["total_size"]) is not int
            or not selected_range[1] < entry["total_size"] <= 2**53
            or entry["size_bytes"] != selected_range[1] - selected_range[0] + 1
        ):
            raise InputError("Invalid retained exact-range evidence")
        if (
            entry["path"] not in inventory
            or type(entry["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or inventory[entry["path"]] != (entry["sha256"], entry["size_bytes"])
            or not native.is_file()
        ):
            raise InputError("Recovery ledger differs from native payload bytes")
    allowed = {
        "request.json",
        "ledger.json",
        "report.json",
        "repair_plan.json",
        "corrected_metadata.parquet",
    } | paths
    if set(files) - allowed or paths != {name for name in files if name.startswith("payload/")}:
        raise InputError("Unexpected payload in video-recovery case")
    if _origin_conflicts(root, ledger, root, ledger):
        raise InputError("Conflicting retained bytes for the same canonical native URL")
    return manifest, request, ledger, inventory


def _reproduce(root):
    manifest, request, ledger, before = _load(root)
    expected = _json((root / "report.json").read_bytes())
    if type(expected) is not dict or expected.get("status") not in ("pass", "unknown"):
        raise InputError("Malformed retained recovery report")
    matches = _fingerprint() == manifest["verifier"]
    transport = _OfflineTransport(root, ledger)
    observed, selected = _evaluate(transport, request) if matches else (None, None)
    exact = observed is not None and _encode(observed) == _encode(expected)
    if matches:
        if transport.used != {entry["path"] for entry in ledger}:
            raise InputError("Recovery case contains native responses outside the replayed read set")
        if _json((root / "repair_plan.json").read_bytes()) != observed["patch"]:
            exact = False
        derived = root / "corrected_metadata.parquet"
        if selected is not None:
            if not derived.is_file() or derived.read_bytes() != _corrected_metadata(
                selected, observed["patch"]
            ):
                exact = False
        elif derived.exists():
            exact = False
    if _inventory_checked(root) != before or matches and _fingerprint() != manifest["verifier"]:
        raise InputError("Video-recovery evidence/verifier changed during replay")
    result = {
        "schema_version": 1,
        "kind": "video_reference_recovery_reproduction",
        "status": "unknown" if not matches else "pass" if exact else "error",
        "original_status": expected.get("original_status", "unknown"),
        "repair_verified": bool(exact and observed["repair_verified"]),
        "recovery_status": expected["status"],
        "episode_index": request["episode_index"],
        "reproduction_matches": exact,
        "verifier_matches": matches,
        "assurance": "internal_consistency_only",
        "origin_authenticated": False,
        "origin_verified": False,
        "verification_scope": _VERIFICATION_SCOPE,
        "recorded_report_sha256": _sha(_encode(expected)),
        "current_report_sha256": _sha(_encode(observed)) if observed is not None else None,
        "reason": "Exact complete retained report and derived native copy reproduced"
        if exact
        else "Verifier drift or complete report/correction mismatch",
        "scope": _SCOPE,
    }
    return result, request


def reproduce_video_recovery(directory):
    """Offline full-report and native correction replay; never fetch media."""
    return _reproduce(_path(directory))[0]


def _origin_conflicts(root, ledger, other_root, other_ledger):
    """Compare complete bodies and overlapping ranges, including within one case."""
    conflicts, cache = [], {}

    def body(directory, entry):
        key = (directory, entry["path"])
        if key not in cache:
            cache[key] = _inside(directory, entry["path"]).read_bytes()
        return cache[key]

    for left in ledger:
        for right in other_ledger:
            if left["url"] != right["url"]:
                continue
            a, b = left["range"], right["range"]
            left_start, left_end = (0, left["size_bytes"]) if a is None else (a[0], a[1] + 1)
            right_start, right_end = (0, right["size_bytes"]) if b is None else (b[0], b[1] + 1)
            left_total = left["size_bytes"] if a is None else left["total_size"]
            right_total = right["size_bytes"] if b is None else right["total_size"]
            start, end = max(left_start, right_start), min(left_end, right_end)
            reason = None
            if left_total != right_total:
                reason = "Different complete resource sizes for the same pinned URL"
            elif start < end and (
                body(root, left)[start - left_start : end - left_start]
                != body(other_root, right)[start - right_start : end - right_start]
            ):
                reason = "Different bytes in an overlapping canonical URL interval"
            if reason is not None:
                conflicts.append({"url": left["url"], "reason": reason,
                                  "overlap": [start, end] if start < end else None})
    return conflicts


def check_video_recovery_candidate(directory, candidate_directory):
    """Offline comparison of independently reproduced, distinct episode cases."""
    root, candidate = _path(directory), _path(candidate_directory)
    if root == candidate or root.is_relative_to(candidate) or candidate.is_relative_to(root):
        raise InputError("Require two separate non-nested retained recovery cases")
    _, _, left_ledger, left_before = _load(root)
    _, _, right_ledger, right_before = _load(candidate)
    first, a = _reproduce(root)
    second, b = _reproduce(candidate)
    conflicts = _origin_conflicts(root, left_ledger, candidate, right_ledger)
    independently_verified = (
        first["status"] == second["status"] == "pass"
        and first["repair_verified"] and second["repair_verified"]
    )
    left = _json((root / "report.json").read_bytes()) if independently_verified else {}
    right = _json((candidate / "report.json").read_bytes()) if independently_verified else {}
    x, y = left.get("recovered_reference", {}), right.get("recovered_reference", {})
    disjoint = bool(x and y) and (
        x["path"] != y["path"]
        or max(x["from_sample_inclusive"], y["from_sample_inclusive"])
        >= min(x["to_sample_exclusive"], y["to_sample_exclusive"])
    )
    lx, ly = left.get("evidence", {}), right.get("evidence", {})
    clips_distinct = bool(lx and ly) and lx["original_clip_sha256"] != ly["original_clip_sha256"]
    fields = ("repo_id", "revision", "original_revision", "video_key", "criterion")
    same = all(a[key] == b[key] for key in fields)
    distinct = a["episode_index"] != b["episode_index"]
    accepted = (
        first["status"] == second["status"] == "pass"
        and first["repair_verified"]
        and second["repair_verified"]
        and same
        and distinct
        and not conflicts
        and disjoint
        and clips_distinct
    )
    if _inventory_checked(root) != left_before or _inventory_checked(candidate) != right_before:
        raise InputError("Recovery evidence changed during cross-case origin comparison")
    return {
        "schema_version": 1,
        "kind": "video_reference_recovery_candidate",
        "status": "pass" if accepted else "unknown",
        "original_status": first["original_status"],
        "repair_verified": bool(accepted),
        "episode_index": a["episode_index"],
        "candidate_episode_index": b["episode_index"],
        "distinct_episode": distinct,
        "fixed_criterion_matches": same,
        "candidate_evaluated": bool(accepted),
        "shared_origin_bytes_consistent": not conflicts,
        "origin_conflicts": conflicts,
        "recovered_intervals_disjoint": disjoint,
        "original_clips_distinct": clips_distinct,
        "assurance": "internal_consistency_only",
        "origin_authenticated": False,
        "origin_verified": False,
        "verification_scope": _VERIFICATION_SCOPE,
        "reference_reproduction": first,
        "candidate_reproduction": second,
        "reason": "Both distinct original episode cases reproduce under the same frozen recovery criterion"
        if accepted
        else "Distinctness, criterion, shared origin bytes, target collision or independent recovery verification is unresolved",
        "scope": _SCOPE,
    }
