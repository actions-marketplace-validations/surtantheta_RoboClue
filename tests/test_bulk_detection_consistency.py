"""Independent review witnesses: resealed detection/entry/projection inconsistencies."""

import json
import os
import shutil

import pytest
from recovery_fixtures import CAMERA, EPISODES, ORIGINAL, REPO, REVISION, movie
from test_bulk_video_recovery import source

from policy_doctor import bulk_video_recovery as bulk
from policy_doctor import hub
from policy_doctor import video_recovery as vr
from policy_doctor.artifacts import InputError


def automatic(monkeypatch, tmp_path, **kwargs):
    source(monkeypatch, tmp_path)
    batch = tmp_path / "automatic"
    bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True, **kwargs)
    return batch


def load(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_bytes(vr._encode(obj))


def drop_one_complete_entry(batch):
    report = load(batch / "report.json")
    report["episodes"] = report["episodes"][:1]
    report.update(verified_associations=1, attempted_associations=1, proposed_cells=3)
    proposals = load(batch / "proposals.json")[:3]
    hashes = load(batch / "native_hashes.json")
    files = {p: (batch / "original_metadata" / p).read_bytes() for p in hashes}
    for path, raw in bulk.merge_proposals(files, proposals).items():
        (batch / "proposed_metadata" / path).write_bytes(raw)
    write(batch / "report.json", report)
    write(batch / "proposals.json", proposals)
    bulk._seal(batch)


def test_resealed_complete_union_cannot_omit_a_detected_association(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    drop_one_complete_entry(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_resealed_automatic_batch_requires_detection_record(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    (batch / "detection.json").unlink()
    bulk._seal(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_resealed_partial_detector_coverage_cannot_claim_complete_batch(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    detection = load(batch / "detection.json")
    detection["coverage"]["video_index_details"]["status"] = "partial"
    write(batch / "detection.json", detection)
    bulk._seal(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_honest_partial_attempt_replays_as_subset_not_complete(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path, max_episodes=1)
    moved = tmp_path / "relocated"
    shutil.move(batch, moved)
    monkeypatch.setattr(vr, "_opener", lambda: pytest.fail("Offline detector consistency used network"))
    replay = bulk.reproduce_bulk_proposal(moved)
    assert replay["reproduction_matches"]
    assert replay["batch_status"] == "unknown" and not replay["proposal_complete"]
    assert replay["detector_coverage_replayed_from_raw_headers"] is False
    assert replay["retained_detector_consistency_verified"] is True


def test_resealed_header_change_must_recompute_findings(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    detection = load(batch / "detection.json")
    context = detection.get("derived_video_context", {"headers": {"unbound": {}}})
    next(iter(context["headers"].values()))["last_sample_start_s"] = 999
    detection["derived_video_context"] = context
    write(batch / "detection.json", detection)
    bulk._seal(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_resealed_unsupported_header_cannot_keep_complete_coverage(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    detection = load(batch / "detection.json")
    context = detection.get("derived_video_context", {"headers": {"unbound": {}}})
    next(iter(context["headers"].values()))["status"] = "unsupported"
    detection["derived_video_context"] = context
    write(batch / "detection.json", detection)
    bulk._seal(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


@pytest.mark.parametrize("area", ["original_metadata", "proposed_metadata"])
def test_resealed_extra_native_projection_file_is_not_ignored(monkeypatch, tmp_path, area):
    batch = automatic(monkeypatch, tmp_path)
    original = batch / area / EPISODES
    extra = original.with_name("file-001.parquet")
    extra.write_bytes(original.read_bytes())
    bulk._seal(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_resealed_automatic_batch_without_derived_context_is_not_full_replay(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    detection = load(batch / "detection.json")
    detection.pop("derived_video_context", None)
    write(batch / "detection.json", detection)
    bulk._seal(batch)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_intake_listing_and_first_recovery_must_not_contradict_same_pin(monkeypatch, tmp_path):
    public, _ = source(monkeypatch, tmp_path)
    original_open, count = public.open, 0

    def changed_open(request, timeout):
        nonlocal count
        if "/revision/" + REVISION in request.full_url:
            count += 1
            if count == 2:
                public.files[(REVISION, f"videos/{CAMERA}/chunk-000/file-000.mp4")] = movie((b"changed-a", b"changed-b"))
        return original_open(request, timeout)

    monkeypatch.setattr(public, "open", changed_open)
    batch = tmp_path / "contradiction"
    report = bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True)
    assert report["status"] == "unknown" and not report["proposal_complete"]
    assert report["verified_associations"] == 0
    assert all("intake listing" in entry["reason"] for entry in report["episodes"])


def test_resealed_intake_listing_digest_mismatch_is_refused(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    provenance = load(batch / "intake_provenance.json")
    provenance["api_sha256"] = "0" * 64
    write(batch / "intake_provenance.json", provenance)
    bulk._seal(batch)
    with pytest.raises(InputError, match="intake listing"):
        bulk.reproduce_bulk_proposal(batch)


def test_overlapping_derived_header_facts_must_match_recovery_headers(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    original = hub._video_header_intake

    def inconsistent(*args, **kwargs):
        headers = original(*args, **kwargs)
        for fact in headers.values():
            if fact.get("status") == "parsed":
                fact["sample_count"] += 1  # preserve an overrun finding, but contradict its actual MP4 header
        return headers

    monkeypatch.setattr(hub, "_video_header_intake", inconsistent)
    report = bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, tmp_path / "inconsistent", fetch=True)
    assert report["status"] == "unknown" and report["verified_associations"] == 0
    assert all("header facts" in entry["reason"] for entry in report["episodes"])


@pytest.mark.parametrize("name", ["BATCH_MANIFEST.json", "BATCH_READY.json"])
def test_symlink_seal_is_refused_before_reading_outside_bytes(monkeypatch, tmp_path, name):
    batch = automatic(monkeypatch, tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes((batch / name).read_bytes())
    (batch / name).unlink()
    (batch / name).symlink_to(outside)
    outside_identity = (outside.stat().st_dev, outside.stat().st_ino)
    real_read, reads = os.read, []

    def denied_read(descriptor, size):
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == outside_identity:
            reads.append(size)
            pytest.fail("Outside seal bytes were read before symlink refusal")
        return real_read(descriptor, size)

    monkeypatch.setattr(os, "read", denied_read)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)
    assert not reads


def test_oversized_manifest_refused_before_reading_body(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    manifest = batch / "BATCH_MANIFEST.json"
    with manifest.open("r+b") as stream:
        stream.truncate(5 * 1024**2)
    identity = (manifest.stat().st_dev, manifest.stat().st_ino)
    original_read = os.read

    def deny_large_read(descriptor, size):
        observed = os.fstat(descriptor)
        assert (observed.st_dev, observed.st_ino) != identity, "Oversized manifest body was read"
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "read", deny_large_read)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)


def test_reroot_after_ready_cannot_make_inventory_read_outside_bytes(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "private.bin"
    secret.write_bytes(b"unrelated bytes must not be opened")
    identity = (secret.stat().st_dev, secret.stat().st_ino)
    original_seal_read, original_read = bulk._seal_read, os.read

    def reroot(root, name, bound):
        raw = original_seal_read(root, name, bound)
        if root == batch and name == "BATCH_READY.json":
            batch.rename(tmp_path / "original-batch")
            batch.symlink_to(outside, target_is_directory=True)
        return raw

    def deny_outside(descriptor, size):
        observed = os.fstat(descriptor)
        assert (observed.st_dev, observed.st_ino) != identity, "Inventory followed a replaced batch root"
        return original_read(descriptor, size)

    monkeypatch.setattr(bulk, "_seal_read", reroot)
    monkeypatch.setattr(os, "read", deny_outside)
    with pytest.raises(InputError):
        bulk.reproduce_bulk_proposal(batch)
