"""Origin-ID, offline trust boundary, and cross-case contradiction regressions."""
import hashlib
import http.client
import io
import json
from urllib.parse import urlsplit

import pytest
from test_video_recovery import (
    CAMERA,
    EPISODES,
    ORIGINAL,
    REPO,
    REVISION,
    PublicSource,
    Response,
    movie,
    resign,
)

from policy_doctor import video_recovery as recovery
from policy_doctor.artifacts import InputError


class OriginSource(PublicSource):
    """Freeze content IDs independently of later served-body mutation."""

    def freeze_listings(self):
        self.listings = {}
        for rev in (REVISION, ORIGINAL):
            siblings = []
            for (revision, name), raw in sorted(self.files.items()):
                if revision != rev:
                    continue
                item = {"rfilename": name, "size": len(raw)}
                if name.endswith((".parquet", ".mp4")):
                    item["lfs"] = {"sha256": hashlib.sha256(raw).hexdigest()}
                else:
                    item["blobId"] = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
                siblings.append(item)
            self.listings[rev] = recovery._encode(
                {"id": REPO, "sha": rev, "private": False, "gated": False, "siblings": siblings}
            )

    def __init__(self):
        super().__init__()
        self.freeze_listings()

    def open(self, request, timeout):
        path = urlsplit(request.full_url).path
        if "/revision/" in path:
            self.requests.append({"url": request.full_url, "headers": dict(request.header_items())})
            rev = path.rsplit("/", 1)[1]
            raw = self.listings[rev]
            return Response(raw, 200, {"Content-Length": str(len(raw)), "X-Repo-Commit": rev})
        return super().open(request, timeout)


def create(monkeypatch, tmp_path, source, episode=0, name="case"):
    monkeypatch.setattr(recovery, "_opener", lambda: source)
    directory = tmp_path / name
    result = recovery.recover_video_reference(REPO, REVISION, ORIGINAL, episode, CAMERA, directory)
    return directory, result


def test_complete_inputs_match_pinned_ids_and_offline_trust_is_explicit(monkeypatch, tmp_path):
    directory, report = create(monkeypatch, tmp_path, OriginSource())
    assert report["repair_verified"]
    assert report["origin_verified"] is False
    assert "internal_consistency_only" in report["verification_scope"]
    identities = report["evidence"]["complete_content_identities"]
    assert len(identities) == 4
    assert {item["algorithm"] for item in identities} == {"sha256", "git_blob_sha1"}
    monkeypatch.setattr(recovery, "_opener", lambda: pytest.fail("offline request"))
    replay = recovery.reproduce_video_recovery(directory)
    assert replay["status"] == "pass"
    assert replay["origin_authenticated"] is False
    assert replay["origin_verified"] is False
    assert replay["assurance"] == "internal_consistency_only"
    assert "not authenticated" in replay["scope"]
    assert "complete target" in replay["scope"]


@pytest.mark.parametrize("role", ["metadata", "clip", "current_info", "original_info"])
def test_complete_body_different_from_frozen_content_id_is_refused(monkeypatch, tmp_path, role):
    source = OriginSource()
    selected = {"metadata": (REVISION, EPISODES),
                "clip": (ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_000000.mp4"),
                "current_info": (REVISION, "meta/info.json"),
                "original_info": (ORIGINAL, "meta/info.json")}[role]
    raw = source.files[selected]
    source.files[selected] = raw[:-1] + bytes([raw[-1] ^ 1])
    with pytest.raises(InputError, match="content ID"):
        create(monkeypatch, tmp_path, source)
    assert not (tmp_path / "case").exists()


@pytest.mark.parametrize("mode", ["missing", "malformed", "wrong_size", "wrong_repository", "wrong_revision"])
def test_listing_cannot_omit_or_lie_about_selected_identity(monkeypatch, tmp_path, mode):
    source = OriginSource()
    data = json.loads(source.listings[REVISION])
    entry = next(item for item in data["siblings"] if item["rfilename"] == "meta/info.json")
    if mode == "missing":
        del entry["blobId"]
    elif mode == "malformed":
        entry["blobId"] = "z" * 40
    elif mode == "wrong_size":
        entry["size"] += 1
    elif mode == "wrong_repository":
        data["id"] = "other/repository"
    elif mode == "wrong_revision":
        data["sha"] = "f" * 40
    source.listings[REVISION] = recovery._encode(data)
    with pytest.raises(InputError):
        create(monkeypatch, tmp_path, source)


def test_original_template_must_bind_episode_index(monkeypatch, tmp_path):
    source = OriginSource()
    key = (ORIGINAL, "meta/info.json")
    info = json.loads(source.files[key])
    info["video_path"] = f"videos/chunk-000/{CAMERA}/episode_000000.mp4"
    source.files[key] = recovery._encode(info)
    source.freeze_listings()
    _, report = create(monkeypatch, tmp_path, source)
    assert report["status"] == "unknown" and not report["repair_verified"]
    assert "episode_index" in report["reason"]


def test_incomplete_read_is_an_intake_refusal(monkeypatch, tmp_path):
    class Dropped(OriginSource):
        def open(self, request, timeout):
            response = super().open(request, timeout)
            def read(size=-1):
                raise http.client.IncompleteRead(b"short", 100)
            response.read = read
            return response
    with pytest.raises(InputError, match="read within its bounds"):
        create(monkeypatch, tmp_path, Dropped())
    assert not (tmp_path / "case").exists()


def test_same_pinned_url_cannot_describe_different_complete_bytes(monkeypatch, tmp_path):
    first = OriginSource()
    a, _ = create(monkeypatch, tmp_path, first, name="first")
    second = OriginSource()
    key = (REVISION, "meta/info.json")
    info = json.loads(second.files[key])
    info["unrelated_note"] = "different cached origin"
    second.files[key] = recovery._encode(info)
    second.freeze_listings()
    b, _ = create(monkeypatch, tmp_path, second, episode=1, name="second")
    result = recovery.check_video_recovery_candidate(a, b)
    assert result["status"] == "unknown" and not result["repair_verified"]
    assert result["shared_origin_bytes_consistent"] is False
    assert result["origin_conflicts"]


def test_same_pinned_url_overlapping_ranges_must_agree(monkeypatch, tmp_path):
    first = OriginSource()
    a, _ = create(monkeypatch, tmp_path, first, name="first")
    second = OriginSource()
    key = (REVISION, f"videos/{CAMERA}/chunk-000/file-001.mp4")
    second.files[key] = second.files[key].replace(b"isomav01", b"isomav02", 1)
    # Keep the complete-input listings identical: only partial media-range bytes differ.
    b, report = create(monkeypatch, tmp_path, second, episode=1, name="second")
    assert report["repair_verified"]
    result = recovery.check_video_recovery_candidate(a, b)
    assert result["status"] == "unknown"
    assert result["shared_origin_bytes_consistent"] is False
    assert all(item["url"].endswith("file-001.mp4") for item in result["origin_conflicts"])


def test_contradictory_overlapping_ranges_within_one_case_are_refused(monkeypatch, tmp_path):
    source = OriginSource()
    large = movie((b"a" * 80, b"b" * 90))
    source.files[(ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_000000.mp4")] = large
    source.files[(REVISION, f"videos/{CAMERA}/chunk-000/file-001.mp4")] = large
    source.freeze_listings()

    def mutate(request, raw, status, headers):
        selected = request.get_header("Range")
        if request.full_url.endswith("file-001.mp4") and selected and status == 206:
            first, last = map(int, selected.removeprefix("bytes=").split("-"))
            if first == 0 and last > 63:
                raw = raw[:20] + bytes([raw[20] ^ 1]) + raw[21:]
        return raw, status, headers

    source.mutate = mutate
    with pytest.raises(InputError, match="ranges disagree|Conflicting retained bytes"):
        create(monkeypatch, tmp_path, source)
    assert not (tmp_path / "case").exists()


def test_identical_clips_cannot_certify_colliding_recovered_intervals(monkeypatch, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = OriginSource()
    table = pq.read_table(io.BytesIO(source.metadata))
    rows = table.to_pylist()
    rows[1][f"videos/{CAMERA}/file_index"] = 0
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), out)
    source.metadata = out.getvalue()
    source.files[(REVISION, EPISODES)] = source.metadata
    source.files[(ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_000001.mp4")] = source.files[(ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_000000.mp4")]
    source.freeze_listings()
    a, x = create(monkeypatch, tmp_path, source, name="first")
    b, y = create(monkeypatch, tmp_path, source, episode=1, name="second")
    assert x["repair_verified"] and y["repair_verified"]
    result = recovery.check_video_recovery_candidate(a, b)
    assert result["status"] == "unknown" and result["distinct_episode"]
    assert result["shared_origin_bytes_consistent"] is True
    assert result["recovered_intervals_disjoint"] is False


def test_legitimate_distinct_targets_still_pass(monkeypatch, tmp_path):
    source = OriginSource()
    a, _ = create(monkeypatch, tmp_path, source, name="first")
    b, _ = create(monkeypatch, tmp_path, source, episode=1, name="second")
    result = recovery.check_video_recovery_candidate(a, b)
    assert result["status"] == "pass"
    assert result["origin_verified"] is False
    assert "internal_consistency_only" in result["verification_scope"]
    assert result["shared_origin_bytes_consistent"] is True
    assert result["recovered_intervals_disjoint"] is True


def test_identical_original_clips_at_distinct_targets_remain_ambiguous(monkeypatch, tmp_path):
    source = OriginSource()
    original = source.files[(ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_000000.mp4")]
    source.files[(ORIGINAL, f"videos/chunk-000/{CAMERA}/episode_000001.mp4")] = original
    source.files[(REVISION, f"videos/{CAMERA}/chunk-000/file-002.mp4")] = original
    source.freeze_listings()
    a, x = create(monkeypatch, tmp_path, source, name="first")
    b, y = create(monkeypatch, tmp_path, source, episode=1, name="second")
    assert x["repair_verified"] and y["repair_verified"]
    result = recovery.check_video_recovery_candidate(a, b)
    assert result["status"] == "unknown"
    assert result["shared_origin_bytes_consistent"] is True
    assert result["recovered_intervals_disjoint"] is True
    assert result["original_clips_distinct"] is False


def test_resealed_change_between_case_replays_is_refused(monkeypatch, tmp_path):
    source = OriginSource()
    a, _ = create(monkeypatch, tmp_path, source, name="first")
    b, _ = create(monkeypatch, tmp_path, source, episode=1, name="second")
    reproduce = recovery._reproduce

    def mutate_after_second(root):
        result = reproduce(root)
        if root == b:
            report = json.loads((a / "report.json").read_text())
            report["reason"] = "Resealed report changed after the first semantic replay"
            (a / "report.json").write_bytes(recovery._encode(report))
            resign(a)
        return result

    monkeypatch.setattr(recovery, "_reproduce", mutate_after_second)
    with pytest.raises(InputError, match="changed"):
        recovery.check_video_recovery_candidate(a, b)
