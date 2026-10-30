"""Bounded native recovery, full offline replay and distinct-case acceptance."""

import hashlib
import io
import json
import struct
from urllib.parse import urlsplit

import pytest

from policy_doctor import video_recovery as recovery
from policy_doctor.artifacts import InputError

REPO = "community/recorded"
REVISION = "2" * 40
ORIGINAL = "1" * 40
CAMERA = "observation.images.front"
EPISODES = "meta/episodes/chunk-000/file-000.parquet"


def box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def full(payload=b"", flags=0):
    return struct.pack(">I", flags) + payload


def movie(payloads=(b"original-frame-a", b"original-frame-b"), *, codec=b"\x81\0\0\0", shift=0):
    ftyp = box(b"ftyp", b"isom\0\0\0\0isomav01")
    mdat = box(b"mdat", b"".join(payloads))
    count = len(payloads)
    mvhd = bytearray(100)
    struct.pack_into(">II", mvhd, 12, 1000, count * 200)
    mdhd = bytearray(24)
    struct.pack_into(">II", mdhd, 12, 10240, count * 2048)
    visual = bytearray(78)
    struct.pack_into(">H", visual, 6, 1)
    struct.pack_into(">HH", visual, 24, 640, 480)
    sample = box(b"av01", bytes(visual) + box(b"av1C", codec))
    stsd = box(b"stsd", full(struct.pack(">I", 1) + sample))
    stsz = box(
        b"stsz", full(struct.pack(">II", 0, count) + struct.pack(">" + "I" * count, *map(len, payloads)))
    )
    stts = box(b"stts", full(struct.pack(">III", 1, count, 2048)))
    stsc = box(b"stsc", full(struct.pack(">IIII", 1, 1, count, 1)))
    stco = box(b"stco", full(struct.pack(">II", 1, len(ftyp) + 8 + shift)))
    stss = box(b"stss", full(struct.pack(">II", 1, 1)))
    stbl = box(b"stbl", stsd + stsz + stts + stsc + stco + stss)
    dinf = box(b"dinf", box(b"dref", full(struct.pack(">I", 1) + box(b"url ", full(flags=1)))))
    hdlr = box(b"hdlr", full(struct.pack(">I", 0) + b"vide" + b"\0" * 12))
    mdia = box(b"mdia", box(b"mdhd", bytes(mdhd)) + hdlr + box(b"minf", dinf + stbl))
    return ftyp + mdat + box(b"moov", box(b"mvhd", bytes(mvhd)) + box(b"trak", mdia))


class Response(io.BytesIO):
    def __init__(self, raw, status, headers):
        super().__init__(raw)
        self.status, self.headers = status, headers


class PublicSource:
    def __init__(self, *, bad_original=False, target_codec=None, healthy=False):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        self.requests = []
        self.mutate = None
        feature = {CAMERA: {"dtype": "video", "shape": [480, 640, 3]}}
        info = {
            "fps": 5,
            "codebase_version": "v3.0",
            "chunks_size": 1000,
            "features": feature,
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        old = {
            "fps": 5,
            "codebase_version": "v2.1",
            "chunks_size": 1000,
            "features": feature,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
        rows = [
            {
                "episode_index": i,
                "length": 2,
                f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": i,
                f"videos/{CAMERA}/from_timestamp": 0.0 if healthy else 0.4,
                f"videos/{CAMERA}/to_timestamp": 0.4 if healthy else 0.8,
                "untouched_label": f"original-{i}",
            }
            for i in range(2)
        ]
        stream = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), stream)
        self.metadata = stream.getvalue()
        self.files = {
            (REVISION, "meta/info.json"): recovery._encode(info),
            (ORIGINAL, "meta/info.json"): recovery._encode(old),
            (REVISION, EPISODES): self.metadata,
        }
        for index, payloads in enumerate(
            [
                (b"earlier-a", b"earlier-b"),
                (b"original-frame-a", b"original-frame-b"),
                (b"later-frame-a", b"later-frame-b"),
            ]
        ):
            self.files[(REVISION, f"videos/{CAMERA}/chunk-000/file-{index:03d}.mp4")] = movie(
                payloads, codec=target_codec or b"\x81\0\0\0"
            )
        for episode, payloads in enumerate(
            [(b"original-frame-a", b"original-frame-b"), (b"later-frame-a", b"later-frame-b")]
        ):
            self.files[(ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_{episode:06d}.mp4")] = movie(
                (b"wrong-a", b"wrong-b") if bad_original else payloads
            )

    def open(self, request, timeout):
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items())})
        parsed = urlsplit(request.full_url)
        if "/revision/" in parsed.path:
            revision = parsed.path.rsplit("/", 1)[1]
            siblings = []
            for (rev, path), data in sorted(self.files.items()):
                if rev != revision:
                    continue
                item = {"rfilename": path, "size": len(data)}
                if path.endswith((".mp4", ".parquet")):
                    item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
                else:
                    item["blobId"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                siblings.append(item)
            raw = recovery._encode(
                {"id": REPO, "sha": revision, "private": False, "gated": False, "siblings": siblings}
            )
        elif "/tree/" in parsed.path:
            raw = recovery._encode(
                [
                    {"type": "directory", "path": "meta/episodes/chunk-000", "size": 0},
                    {"type": "file", "path": EPISODES, "size": len(self.metadata)},
                ]
            )
            revision = REVISION
        else:
            prefix, relative = parsed.path.split("/resolve/")
            revision, path = relative.split("/", 1)
            assert prefix == f"/datasets/{REPO}"
            raw = self.files[(revision, path)]
        headers = {"X-Repo-Commit": revision, "Content-Length": str(len(raw))}
        status = 200
        selected = request.get_header("Range")
        if selected:
            start, end = map(int, selected.removeprefix("bytes=").split("-"))
            assert 0 <= start <= end < len(raw)
            headers["Content-Range"] = f"bytes {start}-{end}/{len(raw)}"
            raw = raw[start : end + 1]
            headers["Content-Length"] = str(len(raw))
            status = 206
        if self.mutate:
            raw, status, headers = self.mutate(request, raw, status, headers)
        return Response(raw, status, headers)


def create(monkeypatch, tmp_path, source=None, episode=0, name="case"):
    source = source or PublicSource()
    monkeypatch.setattr(recovery, "_opener", lambda: source)
    directory = tmp_path / name
    result = recovery.recover_video_reference(REPO, REVISION, ORIGINAL, episode, CAMERA, directory)
    return directory, result, source


def resign(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    raw = recovery._encode(manifest)
    (directory / "manifest.json").write_bytes(raw)
    (directory / "READY.json").write_bytes(
        recovery._encode({"schema_version": 1, "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    )


def test_exact_native_identity_produces_reviewable_three_cell_copy(monkeypatch, tmp_path):
    directory, result, source = create(monkeypatch, tmp_path)
    assert result["status"] == "pass" and result["original_status"] == "error" and result["repair_verified"]
    assert len(result["patch"]) == 3
    assert [p["new"] for p in result["patch"]] == [1, 0.0, 0.4]
    assert result["evidence"]["sample_count"] == 2
    assert (directory / "corrected_metadata.parquet").is_file()
    assert source.metadata == source.files[(REVISION, EPISODES)]
    pq = pytest.importorskip("pyarrow.parquet")
    before = pq.read_table(io.BytesIO(source.metadata)).to_pylist()
    after = pq.read_table(directory / "corrected_metadata.parquet").to_pylist()
    changed = {(i, k) for i, (a, b) in enumerate(zip(before, after, strict=True)) for k in a if a[k] != b[k]}
    assert changed == {(0, p["column"]) for p in result["patch"]}
    assert all("Authorization" not in r["headers"] and "Cookie" not in r["headers"] for r in source.requests)
    assert len(recovery._encode(result)) < 16384


def test_relocation_offline_full_replay_and_unchanged_distinct_case(monkeypatch, tmp_path):
    source = PublicSource()
    first, _, source = create(monkeypatch, tmp_path, source, episode=0, name="first")
    second, _, source = create(monkeypatch, tmp_path, source, episode=1, name="second")
    count = len(source.requests)
    moved = tmp_path / "colleague"
    first.rename(moved)
    monkeypatch.setattr(recovery, "_opener", lambda: pytest.fail("Offline replay attempted network"))
    assert recovery.reproduce_video_recovery(moved)["status"] == "pass"
    result = recovery.check_video_recovery_candidate(moved, second)
    assert result["status"] == "pass" and result["distinct_episode"]
    assert result["candidate_episode_index"] == 1
    assert len(source.requests) == count


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"bad_original": True}, "sample sizes"),
        ({"target_codec": b"\x81\x01\0\0"}, "codec"),
        ({"healthy": True}, "in bounds"),
    ],
)
def test_partial_or_unjustified_recovery_never_passes(monkeypatch, tmp_path, options, reason):
    directory, result, _ = create(monkeypatch, tmp_path, PublicSource(**options))
    assert result["status"] == "unknown" and not result["repair_verified"] and result["patch"] == []
    assert reason in result["reason"]
    assert not (directory / "corrected_metadata.parquet").exists()
    replay = recovery.reproduce_video_recovery(directory)
    assert replay["status"] == "pass" and not replay["repair_verified"]


@pytest.mark.parametrize("revision", ["main", "v3.0", "A" * 40, "a" * 39, "a" * 41, None])
def test_mutable_or_invalid_revision_refused_before_network(monkeypatch, tmp_path, revision):
    monkeypatch.setattr(recovery, "_opener", lambda: pytest.fail("Invalid input attempted network"))
    with pytest.raises(InputError):
        recovery.recover_video_reference(REPO, revision, ORIGINAL, 0, CAMERA, tmp_path / "case")
    assert not (tmp_path / "case").exists()


@pytest.mark.parametrize("mode", ["range_200", "bad_range", "short_body", "encoding", "wrong_commit"])
def test_transport_failure_never_publishes_partial_case(monkeypatch, tmp_path, mode):
    source = PublicSource()

    def mutate(request, raw, status, headers):
        if status == 206:
            if mode == "range_200":
                status = 200
            elif mode == "bad_range":
                headers["Content-Range"] = "bytes 1-64/500"
            elif mode == "short_body":
                raw = raw[:-1]
            elif mode == "encoding":
                headers["Content-Encoding"] = "gzip"
            elif mode == "wrong_commit":
                headers["X-Repo-Commit"] = "f" * 40
        return raw, status, headers

    source.mutate = mutate
    with pytest.raises(InputError):
        create(monkeypatch, tmp_path, source)
    assert not (tmp_path / "case").exists()


@pytest.mark.parametrize("mode", ["payload", "report", "ready", "extra", "empty", "symlink"])
def test_local_tamper_refused(monkeypatch, tmp_path, mode):
    directory, _, _ = create(monkeypatch, tmp_path)
    if mode == "payload":
        next((directory / "payload").iterdir()).write_bytes(b"changed")
    elif mode == "report":
        (directory / "report.json").write_text("{}")
    elif mode == "ready":
        (directory / "READY.json").write_text("{}")
    elif mode == "extra":
        (directory / "extra").write_text("unlisted")
    elif mode == "empty":
        (directory / "extra").mkdir()
    else:
        (directory / "extra").symlink_to(directory / "report.json")
    with pytest.raises(InputError):
        recovery.reproduce_video_recovery(directory)


def test_changed_full_finding_with_resigned_inventory_is_not_reproduced(monkeypatch, tmp_path):
    directory, _, _ = create(monkeypatch, tmp_path)
    report = json.loads((directory / "report.json").read_text())
    report["reason"] = "Changed conclusion with identical status and witness counts"
    (directory / "report.json").write_bytes(recovery._encode(report))
    resign(directory)
    result = recovery.reproduce_video_recovery(directory)
    assert result["status"] == "error" and not result["reproduction_matches"]


def test_same_episode_and_criterion_mismatch_are_not_later_reuse(monkeypatch, tmp_path):
    source = PublicSource()
    first, _, _ = create(monkeypatch, tmp_path, source, name="first")
    same, _, _ = create(monkeypatch, tmp_path, source, name="same")
    result = recovery.check_video_recovery_candidate(first, same)
    assert (
        result["status"] == "unknown" and not result["distinct_episode"] and not result["candidate_evaluated"]
    )
    with pytest.raises(InputError):
        recovery.check_video_recovery_candidate(first, first)


def test_verifier_drift_blocks_later_and_replay(monkeypatch, tmp_path):
    source = PublicSource()
    first, _, _ = create(monkeypatch, tmp_path, source, name="first")
    second, _, _ = create(monkeypatch, tmp_path, source, episode=1, name="second")
    original = recovery._fingerprint
    monkeypatch.setattr(recovery, "_fingerprint", lambda: original() | {"engine_version": "changed"})
    assert recovery.reproduce_video_recovery(first)["status"] == "unknown"
    later = recovery.check_video_recovery_candidate(first, second)
    assert later["status"] == "unknown" and not later["candidate_evaluated"]


def test_overwrite_and_nested_destinations_refused(monkeypatch, tmp_path):
    directory, _, source = create(monkeypatch, tmp_path)
    ready = (directory / "READY.json").read_bytes()
    with pytest.raises(InputError):
        create(monkeypatch, tmp_path, source)
    assert (directory / "READY.json").read_bytes() == ready
    with pytest.raises(InputError):
        recovery.recover_video_reference(REPO, REVISION, ORIGINAL, 0, CAMERA, directory / "nested")
    assert not (directory / "nested").exists()


def test_chunk_byte_escape_and_invalid_codec_refuse():
    with pytest.raises(recovery._Unsupported):
        recovery._parse_mp4(recovery._Media(raw=movie(shift=999)))
    with pytest.raises(recovery._Unsupported):
        recovery._parse_mp4(recovery._Media(raw=movie(codec=b"")))


def test_exact_percent_encoded_hf_cache_redirect_is_supported(monkeypatch, tmp_path):
    from urllib.error import HTTPError
    from urllib.parse import quote, unquote

    class CacheSource(PublicSource):
        def open(self, request, timeout):
            parsed = urlsplit(request.full_url)
            if "/resolve/" in parsed.path and parsed.path.endswith("/meta/info.json"):
                revision = parsed.path.split("/resolve/")[1].split("/")[0]
                location = f"/api/resolve-cache/datasets/{REPO}/{revision}/{quote('meta/info.json', safe='')}"
                raise HTTPError(
                    request.full_url,
                    307,
                    "Temporary Redirect",
                    {"Location": location, "X-Repo-Commit": revision},
                    None,
                )
            if "/api/resolve-cache/datasets/" in parsed.path:
                suffix = parsed.path.split(f"/api/resolve-cache/datasets/{REPO}/")[1]
                revision, path = suffix.split("/", 1)
                raw = self.files[(revision, unquote(path))]
                return Response(raw, 200, {"Content-Length": str(len(raw)), "X-Repo-Commit": revision})
            return super().open(request, timeout)

    directory, result, _ = create(monkeypatch, tmp_path, CacheSource())
    assert result["repair_verified"]
    ledger = json.loads((directory / "ledger.json").read_text())
    assert all("/api/resolve-cache/" not in entry["url"] for entry in ledger)
    assert recovery.reproduce_video_recovery(directory)["status"] == "pass"


def test_atomic_publication_race_preserves_other_owners_directory(monkeypatch, tmp_path):
    source = PublicSource()
    publish = recovery.publish_capture

    def race(staging, target):
        target.mkdir()
        (target / "other-owner").write_text("keep")
        publish(staging, target)

    monkeypatch.setattr(recovery, "publish_capture", race)
    with pytest.raises(InputError):
        create(monkeypatch, tmp_path, source)
    assert [p.name for p in (tmp_path / "case").iterdir()] == ["other-owner"]


def test_unused_listed_response_is_not_silently_accepted(monkeypatch, tmp_path):
    directory, _, _ = create(monkeypatch, tmp_path)
    ledger = json.loads((directory / "ledger.json").read_text())
    raw = b"not in the evaluated native read set"
    path = "payload/999.bin"
    (directory / path).write_bytes(raw)
    ledger.append(
        {
            "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/unused.json",
            "range": None,
            "status": 200,
            "total_size": None,
            "path": path,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    (directory / "ledger.json").write_bytes(recovery._encode(ledger))
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"][path] = hashlib.sha256(raw).hexdigest()
    (directory / "manifest.json").write_bytes(recovery._encode(manifest))
    resign(directory)
    with pytest.raises(InputError, match="outside the replayed read set"):
        recovery.reproduce_video_recovery(directory)
