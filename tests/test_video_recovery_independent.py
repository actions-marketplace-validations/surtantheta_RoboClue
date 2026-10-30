"""Independent adversarial checks; fixtures are synthetic container tables."""

import hashlib
import io
import json
import struct
from fractions import Fraction

import pytest

from policy_doctor import video_recovery as recovery
from policy_doctor.artifacts import InputError

SHA = "1" * 40
ORIGINAL_SHA = "2" * 40
URL = f"https://huggingface.co/datasets/example/native/resolve/{SHA}/video.mp4"


def box(kind, body):
    return struct.pack(">I4s", len(body) + 8, kind) + body


def tablebox(kind, rows, width):
    return box(
        kind,
        b"\0" * 4 + struct.pack(">I", len(rows)) + b"".join(struct.pack(">" + width, *row) for row in rows),
    )


def movie(
    *,
    payloads=(b"first-packet", b"second-packet"),
    codec=b"\x81\x00\x0c\x00",
    tracks=1,
    offset_shift=0,
    stsc=None,
    ctts=None,
    extra_stbl=b"",
    duplicate_codec=False,
    handler=b"vide",
    sync=None,
    empty_edit=False,
    external_reference=False,
    presentation_children=b"",
):
    ftyp = box(b"ftyp", b"isom" + b"\0" * 4 + b"isomav01")
    media = box(b"mdat", b"".join(payloads))
    data_offset = len(ftyp) + 8 + offset_shift
    sizes = box(
        b"stsz",
        b"\0" * 8 + struct.pack(">I", len(payloads)) + b"".join(struct.pack(">I", len(p)) for p in payloads),
    )
    timing = tablebox(b"stts", [(len(payloads), 2048)], "II")
    chunk = tablebox(b"stsc", stsc or [(1, len(payloads), 1)], "III")
    offsets = tablebox(b"stco", [(data_offset,)], "I")
    entry_header = bytearray(78)
    struct.pack_into(">H", entry_header, 6, 1)
    struct.pack_into(">HH", entry_header, 24, 640, 480)
    config = box(b"av1C", codec)
    desc = box(
        b"stsd",
        b"\0" * 4
        + struct.pack(">I", 1)
        + box(
            b"av01",
            bytes(entry_header) + config + (config if duplicate_codec else b"") + presentation_children,
        ),
    )
    composition = b"" if ctts is None else tablebox(b"ctts", ctts, "II")
    syncbox = b"" if sync is None else tablebox(b"stss", [(s,) for s in sync], "I")
    stbl = box(b"stbl", desc + sizes + timing + chunk + offsets + composition + syncbox + extra_stbl)
    url = box(b"url ", b"\0\0\0\0https://outside.invalid/video.mp4\0" if external_reference else b"\0\0\0\1")
    dref = box(b"dref", b"\0" * 4 + struct.pack(">I", 1) + url)
    dinf = box(b"dinf", dref)
    mdhd = box(b"mdhd", b"\0" * 12 + struct.pack(">II", 10240, len(payloads) * 2048) + b"\0" * 4)
    hdlr = box(b"hdlr", b"\0" * 8 + handler + b"\0" * 12)
    minf = box(b"minf", dinf + stbl)
    edts = box(b"edts", tablebox(b"elst", [(0, 0, 1, 0)], "Iihh")) if empty_edit else b""
    trak = box(b"trak", edts + box(b"mdia", mdhd + hdlr + minf))
    return ftyp + media + box(b"moov", trak * tracks)


def parse(raw):
    return recovery._parse_mp4(recovery._Media(raw=raw))


def test_independent_native_table_positive():
    parsed = parse(movie())
    assert parsed["sample_count"] == 2
    assert parsed["times"] == [(Fraction(0), Fraction(1, 5)), (Fraction(1, 5), Fraction(1, 5))]
    assert parsed["codec"]["sample_entry"] == "av01"
    assert parsed["codec"]["width"] == 640 and parsed["codec"]["height"] == 480


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tracks": 0},
        {"tracks": 2},
        {"offset_shift": -16},
        {"offset_shift": 1000},
        {"stsc": [(2, 2, 1)]},
        {"stsc": [(1, 1, 1)]},
        {"stsc": [(1, 3, 1)]},
        {"stsc": [(1, 2, 2)]},
        {"ctts": [(1, 0)]},
        {"duplicate_codec": True},
        {"handler": b"soun"},
        {"sync": [2]},
        {"extra_stbl": tablebox(b"stco", [(32,)], "I")},
    ],
)
def test_inconsistent_or_unsupported_native_tables_refused(kwargs):
    with pytest.raises((InputError, recovery._Unsupported)):
        parse(movie(**kwargs))


@pytest.mark.parametrize(
    "raw", [b"", b"\0" * 7, struct.pack(">I4s", 7, b"moov"), struct.pack(">I4s", 100, b"moov") + b"\0" * 20]
)
def test_truncated_or_invalid_container_cannot_parse(raw):
    with pytest.raises((InputError, recovery._Unsupported)):
        parse(raw)


def test_empty_codec_configuration_cannot_establish_identity():
    with pytest.raises((InputError, recovery._Unsupported)):
        parse(movie(codec=b""))


def test_external_media_reference_not_treated_as_internal_payload():
    with pytest.raises((InputError, recovery._Unsupported)):
        parse(movie(external_reference=True))


def test_empty_movie_edit_not_treated_as_complete_clip():
    with pytest.raises((InputError, recovery._Unsupported)):
        parse(movie(empty_edit=True))


def test_codec_and_time_changes_remain_visible_in_native_parser():
    original = parse(movie())
    assert original["codec"] != parse(movie(codec=b"\x81\x01\x0c\x00"))["codec"]
    assert original["times"] != parse(movie(ctts=[(2, 1024)]))["times"]


class Response(io.BytesIO):
    def __init__(self, raw, *, status=200, headers=None):
        super().__init__(raw)
        self.status = status
        self.headers = headers or {}
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


class Opener:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def open(self, request, **kwargs):
        self.requests.append(request)
        return self.response


def transport(tmp_path, monkeypatch, response):
    opener = Opener(response)
    monkeypatch.setattr(recovery, "_opener", lambda: opener)
    return recovery._Transport(tmp_path), opener


def test_transport_exact_range_and_bounded_offline_replay(tmp_path, monkeypatch):
    response = Response(
        b"ABCD", status=206, headers={"Content-Range": "bytes 4-7/100", "Content-Length": "4"}
    )
    intake, opener = transport(tmp_path, monkeypatch, response)
    raw, total = intake.fetch(URL, limit=4, byte_range=(4, 7))
    assert (raw, total) == (b"ABCD", 100)
    assert opener.requests[0].get_header("Range") == "bytes=4-7"
    assert opener.requests[0].get_header("Authorization") is None
    assert intake.total == 4

    def refuse(*a, **kw):
        pytest.fail("offline replay opened network")

    monkeypatch.setattr(recovery, "_opener", refuse)
    offline = recovery._OfflineTransport(tmp_path, intake.ledger)
    assert offline.fetch(URL, limit=4, byte_range=(4, 7)) == (b"ABCD", 100)
    (tmp_path / intake.ledger[0]["path"]).write_bytes(b"WXYZ")
    with pytest.raises(InputError):
        offline.fetch(URL, limit=4, byte_range=(4, 7))


@pytest.mark.parametrize(
    "status,headers,body",
    [
        (200, {"Content-Range": "bytes 4-7/100"}, b"ABCD"),
        (206, {"Content-Range": "bytes 0-3/100"}, b"ABCD"),
        (206, {"Content-Range": "bytes 4-7/7"}, b"ABCD"),
        (206, {"Content-Range": "bytes 4-7/100"}, b"ABC"),
        (206, {"Content-Range": "bytes 4-7/100", "Content-Length": "5"}, b"ABCDE"),
        (206, {"Content-Range": "bytes 4-7/100", "Content-Encoding": "gzip"}, b"ABCD"),
        (206, {"Content-Range": "bytes 4-7/100", "X-Repo-Commit": ORIGINAL_SHA}, b"ABCD"),
        (206, {"Content-Range": "bytes 4-7/100", "Link": "next-page"}, b"ABCD"),
    ],
)
def test_transport_rejects_wrong_or_incomplete_range(tmp_path, monkeypatch, status, headers, body):
    intake, _ = transport(tmp_path, monkeypatch, Response(body, status=status, headers=headers))
    with pytest.raises(InputError):
        intake.fetch(URL, limit=4, byte_range=(4, 7))
    assert intake.ledger == []


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/datasets/a/b/resolve/" + SHA + "/x",
        "https://outside.invalid/resolve/" + SHA + "/x",
        "https://user:password@huggingface.co/datasets/a/b/resolve/" + SHA + "/x",
        "https://huggingface.co/datasets/a/b/resolve/main/x",
    ],
)
def test_transport_invalid_origin_refused_before_open(tmp_path, monkeypatch, url):
    intake, opener = transport(tmp_path, monkeypatch, Response(b"ABCD"))
    with pytest.raises(InputError):
        intake.fetch(url, limit=4)
    assert not opener.requests


def test_request_spec_rejects_mutable_revisions_and_path_keys():
    for kwargs in [
        {"revision": "main"},
        {"original_revision": "v2.1"},
        {"episode_index": True},
        {"video_key": "../../private"},
        {"repo_id": "https://huggingface.co/a/b"},
    ]:
        values = {
            "repo_id": "example/native",
            "revision": SHA,
            "original_revision": ORIGINAL_SHA,
            "episode_index": 40,
            "video_key": "observation.images.camera",
        }
        values.update(kwargs)
        with pytest.raises(InputError):
            recovery._request_spec(**values)


def test_transport_enforces_aggregate_budget(tmp_path, monkeypatch):
    intake, _ = transport(tmp_path, monkeypatch, Response(b"ABCD", headers={"Content-Length": "4"}))
    intake.fetch(URL, limit=4)
    monkeypatch.setattr(recovery, "MAX_TOTAL", 6)
    intake.opener = Opener(Response(b"EFGH", headers={"Content-Length": "4"}))
    with pytest.raises(InputError):
        intake.fetch(URL.replace("video.mp4", "another.mp4"), limit=4)
    assert intake.total == 4 and len(intake.ledger) == 1


def test_media_refuses_changed_total_and_overlapping_bytes():
    class Fake:
        def __init__(self):
            self.calls = 0

        def fetch(self, url, *, limit, byte_range):
            self.calls += 1
            return (b"A" * 64, 100) if self.calls == 1 else (b"B" * 16, 100)

    media = recovery._Media(Fake(), URL)
    with pytest.raises(InputError, match="Overlapping"):
        media.read(60, 16)


@pytest.fixture
def case_factory(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def make(
        name,
        *,
        episode=40,
        repo="example/native",
        revision=SHA,
        original_revision=ORIGINAL_SHA,
        target_payloads=None,
        target_codec=b"\x81\x00\x0c\x00",
        target_ctts=None,
        target_presentation=None,
        camera="observation.images.camera",
    ):
        key = camera
        info = {
            "fps": 5,
            "codebase_version": "v3.0",
            "chunks_size": 1000,
            "features": {key: {"dtype": "video"}},
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        old = {
            **info,
            "codebase_version": "v2.1",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
        metadata = io.BytesIO()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "episode_index": episode,
                        "length": 2,
                        f"videos/{key}/chunk_index": 0,
                        f"videos/{key}/file_index": 0,
                        f"videos/{key}/from_timestamp": 0.4,
                        f"videos/{key}/to_timestamp": 0.8,
                    }
                ]
            ),
            metadata,
        )
        parquet = metadata.getvalue()
        meta_path = "meta/episodes/chunk-000/file-000.parquet"
        spec = recovery._request_spec(repo, revision, original_revision, episode, key)
        current = movie(payloads=(b"old-current-a", b"old-current-b"))
        presentation = box(b"fiel", b"\x01\x00")
        original = movie(presentation_children=presentation)
        target = movie(
            payloads=target_payloads or (b"first-packet", b"second-packet"),
            codec=target_codec,
            ctts=target_ctts,
            presentation_children=presentation if target_presentation is None else target_presentation,
        )
        tree = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/meta/episodes?recursive=true&limit=1000"
        bodies = {
            recovery._url(spec, "meta/info.json"): json.dumps(info).encode(),
            recovery._url(spec, "meta/info.json", original=True): json.dumps(old).encode(),
            tree: json.dumps([{"type": "file", "path": meta_path, "size": len(parquet)}]).encode(),
            recovery._url(spec, meta_path): parquet,
            recovery._url(spec, f"videos/{key}/chunk-000/file-000.mp4"): current,
            recovery._url(spec, f"videos/{key}/chunk-000/file-001.mp4"): target,
            recovery._url(spec, f"videos/chunk-000/{key}/episode_{episode:06d}.mp4", original=True): original,
        }
        for rev in (revision, original_revision):
            prefix = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/"
            siblings = []
            for url, body in list(bodies.items()):
                if not url.startswith(prefix):
                    continue
                path = url.removeprefix(prefix)
                item = {"rfilename": path, "size": len(body)}
                if path.endswith((".mp4", ".parquet")):
                    item["lfs"] = {"sha256": hashlib.sha256(body).hexdigest()}
                else:
                    item["blobId"] = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()
                siblings.append(item)
            bodies[f"https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true"] = recovery._encode(
                {"id": repo, "sha": rev, "private": False, "gated": False, "siblings": siblings}
            )
        seen = []

        class Server:
            def open(self, request, **kwargs):
                seen.append(request.full_url)
                body = bodies[request.full_url]
                byte_range = request.get_header("Range")
                if byte_range:
                    import re

                    first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", byte_range).groups())
                    return Response(
                        body[first : last + 1],
                        status=206,
                        headers={
                            "Content-Range": f"bytes {first}-{last}/{len(body)}",
                            "Content-Length": str(last - first + 1),
                        },
                    )
                return Response(body, headers={"Content-Length": str(len(body))})

        monkeypatch.setattr(recovery, "_opener", Server)
        directory = tmp_path / name
        result = recovery.recover_video_reference(repo, revision, original_revision, episode, key, directory)
        return directory, result, seen

    return make


def test_creator_offline_replay_does_not_accept_colliding_distinct_episode_case(case_factory, monkeypatch):
    first, result, seen = case_factory("forty")
    second, other, _ = case_factory("seventy-seven", episode=77)
    assert result["status"] == other["status"] == "pass"
    assert result["repair_verified"] is True and len(result["patch"]) == 3
    assert seen

    def refuse(*a, **kw):
        pytest.fail("offline case opened network")

    monkeypatch.setattr(recovery, "_opener", refuse)
    replay = recovery.reproduce_video_recovery(first)
    assert replay["status"] == "pass" and replay["repair_verified"]
    candidate = recovery.check_video_recovery_candidate(first, second)
    assert candidate["status"] == "unknown" and candidate["distinct_episode"]
    assert candidate["recovered_intervals_disjoint"] is False
    assert candidate["original_clips_distinct"] is False
    assert candidate["shared_origin_bytes_consistent"] is False


@pytest.mark.parametrize(
    "changed",
    [
        {"target_payloads": (b"wrong-packet", b"second-packet")},
        {"target_codec": b"\x81\x01\x0c\x00"},
        {"target_ctts": [(2, 1024)]},
    ],
)
def test_wrong_clip_codec_or_time_cannot_prove_repair(case_factory, changed):
    directory, result, _ = case_factory("unproven", **changed)
    assert result["status"] == "unknown" and not result["repair_verified"]
    assert not (directory / "corrected_metadata.parquet").exists()
    replay = recovery.reproduce_video_recovery(directory)
    assert replay["reproduction_matches"] and not replay["repair_verified"]
    other, _, _ = case_factory("other-unknown", episode=77, **changed)
    assert recovery.check_video_recovery_candidate(directory, other)["status"] == "unknown"


def test_same_episode_clone_is_not_distinct_case_reuse(case_factory, tmp_path):
    import shutil

    first, _, _ = case_factory("first")
    clone = tmp_path / "clone"
    shutil.copytree(first, clone)
    result = recovery.check_video_recovery_candidate(first, clone)
    assert result["status"] == "unknown" and not result["distinct_episode"]


@pytest.mark.parametrize(
    "change", [{"repo": "different/native"}, {"revision": "3" * 40}, {"original_revision": "4" * 40}]
)
def test_incompatible_frozen_provenance_not_accepted(case_factory, change):
    first, _, _ = case_factory("first")
    second, _, _ = case_factory("second", episode=77, **change)
    result = recovery.check_video_recovery_candidate(first, second)
    assert result["status"] == "unknown" and not result["fixed_criterion_matches"]


def test_retained_payload_tamper_is_refused(case_factory):
    directory, _, _ = case_factory("case")
    payload = next((directory / "payload").iterdir())
    data = bytearray(payload.read_bytes())
    data[0] ^= 1
    payload.write_bytes(data)
    with pytest.raises(InputError):
        recovery.reproduce_video_recovery(directory)


def test_retained_source_symlink_refused(case_factory, tmp_path):
    directory, _, _ = case_factory("case")
    payload = next((directory / "payload").iterdir())
    outside = tmp_path / "outside"
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)
    with pytest.raises(InputError):
        recovery.reproduce_video_recovery(directory)


def test_case_manifest_traversal_and_unlisted_files_refused(case_factory):
    directory, _, _ = case_factory("case")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"]["../outside"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InputError):
        recovery.reproduce_video_recovery(directory)


def test_creator_destination_refused_before_network(case_factory, monkeypatch):
    directory, _, _ = case_factory("case")
    monkeypatch.setattr(recovery, "_opener", lambda: pytest.fail("existing destination reached network"))
    with pytest.raises(InputError):
        recovery.recover_video_reference(
            "example/native", SHA, ORIGINAL_SHA, 40, "observation.images.camera", directory
        )


def test_redirect_signed_delivery_query_never_persists(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    from urllib.parse import urlsplit

    secret = "REVIEW_ONLY_SIGNATURE"
    host = sorted(recovery.CDN_HOSTS)[0]
    target = f"https://{host}/object?signature={secret}"
    responses = [
        HTTPError(URL, 302, "redirect", {"Location": target, "X-Repo-Commit": SHA}, None),
        Response(b"ABCD", headers={"Content-Length": "4"}),
    ]
    seen = []

    class Server:
        def open(self, request, **kwargs):
            seen.append(request)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(recovery, "_opener", Server)
    intake = recovery._Transport(tmp_path)
    assert intake.fetch(URL, limit=4) == (b"ABCD", None)
    assert len(seen) == 2 and urlsplit(seen[1].full_url).netloc == host
    assert all(item.get_header("Authorization") is None for item in seen)
    assert secret not in json.dumps(intake.ledger)
    assert intake.ledger[0]["url"] == URL


@pytest.mark.parametrize(
    "location",
    [
        "https://outside.invalid/object",
        "http://huggingface.co/object",
        f"https://huggingface.co/datasets/example/native/resolve/{ORIGINAL_SHA}/video.mp4",
    ],
)
def test_unsafe_redirect_refused_before_second_request(tmp_path, monkeypatch, location):
    from urllib.error import HTTPError

    seen = []

    class Server:
        def open(self, request, **kwargs):
            seen.append(request)
            if len(seen) > 1:
                pytest.fail("unsafe redirect opened")
            raise HTTPError(URL, 302, "redirect", {"Location": location}, None)

    monkeypatch.setattr(recovery, "_opener", Server)
    intake = recovery._Transport(tmp_path)
    with pytest.raises(InputError):
        intake.fetch(URL, limit=4)
    assert len(seen) == 1 and not intake.ledger


def test_deadline_exhaustion_refuses_request(tmp_path, monkeypatch):
    intake, opener = transport(tmp_path, monkeypatch, Response(b"ABCD"))
    monkeypatch.setattr(recovery.time, "monotonic", lambda: intake.deadline + 1)
    with pytest.raises(InputError):
        intake.fetch(URL, limit=4)
    assert not opener.requests


def test_case_byte_extra_file_refused(case_factory):
    directory, _, _ = case_factory("case")
    (directory / "unbound.txt").write_text("not in the immutable inventory")
    with pytest.raises(InputError):
        recovery.reproduce_video_recovery(directory)


def test_candidate_requires_both_successful_recovery_cases(case_factory):
    first, _, _ = case_factory("valid")
    second, _, _ = case_factory("invalid", episode=77, target_codec=b"\x81\x01\x0c\x00")
    result = recovery.check_video_recovery_candidate(first, second)
    assert result["status"] == "unknown" and not result["repair_verified"]


@pytest.mark.parametrize(
    "changed", [box(b"fiel", b"\x02\x01"), box(b"colr", b"nclx\x00\x01\x00\x01\x00\x01\x00")]
)
def test_presentation_properties_change_cannot_prove_identity(case_factory, changed):
    directory, report, _ = case_factory("changed-presentation", target_presentation=changed)
    assert report["status"] == "unknown" and not report["repair_verified"]
    assert recovery.reproduce_video_recovery(directory)["repair_verified"] is False


def test_bitrate_metadata_change_is_explicitly_nonsemantic():
    one = parse(movie(presentation_children=box(b"btrt", struct.pack(">III", 0, 100, 100))))
    two = parse(movie(presentation_children=box(b"btrt", struct.pack(">III", 0, 200, 150))))
    assert one["codec"] == two["codec"]


@pytest.mark.parametrize(
    "template",
    [
        "videos/{video_key:1000000000}.mp4",
        "videos/{file_index:1000000000d}.mp4",
        "videos/{video_key!r}.mp4",
        "videos/{file_index:{chunk_index}}.mp4",
        "videos/{unknown}.mp4",
        "videos/{file_index",
        "videos/{video_key.__class__}.mp4",
    ],
)
def test_native_path_template_is_bounded_before_formatting(template):
    with pytest.raises((InputError, recovery._Unsupported)):
        recovery._video_path(template, video_key="camera", chunk_index=0, file_index=1)


def test_incompatible_frozen_camera_not_accepted(case_factory):
    first, _, _ = case_factory("original-camera")
    second, _, _ = case_factory("other-camera", episode=77, camera="observation.images.other_camera")
    result = recovery.check_video_recovery_candidate(first, second)
    assert result["status"] == "unknown" and not result["fixed_criterion_matches"]
