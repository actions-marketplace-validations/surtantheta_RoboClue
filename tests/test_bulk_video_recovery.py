"""Generated native inputs and actual detector/recovery/replay/Parquet engines."""

import io
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from recovery_fixtures import CAMERA, EPISODES, ORIGINAL, REPO, REVISION, PublicSource, movie

from policy_doctor import bulk_video_recovery as bulk
from policy_doctor import hub
from policy_doctor import video_recovery as vr
from policy_doctor.artifacts import InputError


def source(monkeypatch, tmp_path, **kwargs):
    public = PublicSource(**kwargs)
    info = vr._json(public.files[(REVISION, "meta/info.json")])
    info["total_episodes"] = 2
    public.files[(REVISION, "meta/info.json")] = vr._encode(info)
    monkeypatch.setattr(vr, "_opener", lambda: public)
    monkeypatch.setattr(hub, "_opener", lambda: public)
    native = tmp_path / "native"
    for (revision, path), raw in public.files.items():
        if revision == REVISION and path.startswith("meta/"):
            out = native / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
    return public, native


def cases(monkeypatch, tmp_path, **kwargs):
    public, native = source(monkeypatch, tmp_path, **kwargs)
    result = []
    for episode in range(2):
        case = tmp_path / f"case-{episode}"
        vr.recover_video_reference(REPO, REVISION, ORIGINAL, episode, CAMERA, case)
        result.append(case)
    return public, native, result


def test_two_recoveries_merge_six_cells_schema_and_all_other_values(monkeypatch, tmp_path):
    public, native, retained = cases(monkeypatch, tmp_path)
    before = {p: p.read_bytes() for p in native.rglob("*") if p.is_file()}
    monkeypatch.setattr(vr, "_opener", lambda: pytest.fail("Offline merge attempted network"))
    out = tmp_path / "batch"
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    assert result["status"] == "unknown" and result["detection_coverage"] == "explicit_subset"
    assert result["verified_associations"] == 2 and result["proposed_cells"] == 6
    original = pq.read_table(io.BytesIO(public.metadata))
    after = pq.read_table(out / "proposed_metadata" / EPISODES)
    assert after.schema.equals(original.schema, check_metadata=True)
    rows = after.to_pylist()
    for i, row in enumerate(rows):
        assert row[f"videos/{CAMERA}/file_index"] == i + 1
        assert row[f"videos/{CAMERA}/from_timestamp"] == 0
        assert row[f"videos/{CAMERA}/to_timestamp"] == 0.4
        assert row["untouched_label"] == f"original-{i}"
    assert all(p.read_bytes() == raw for p, raw in before.items())
    moved = tmp_path / "colleague"
    shutil.move(out, moved)
    replay = bulk.reproduce_bulk_proposal(moved)
    assert replay["status"] == "pass" and replay["proposed_cells"] == 6
    assert not replay["origin_authenticated"] and not replay["proposal_complete"]


def test_real_pinned_intake_detects_recovers_and_merges_without_forged_context(monkeypatch, tmp_path):
    public, _ = source(monkeypatch, tmp_path)
    out = tmp_path / "automatic"
    result = bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, out, fetch=True)
    assert result["status"] == "pass" and result["proposal_complete"]
    assert result["verified_associations"] == 2 and result["proposed_cells"] == 6
    assert result["response_body_upper_bound"] <= result["request"]["max_body_bytes"]
    assert result["body_bytes_read"] is None
    detection = json.loads((out / "detection.json").read_text())
    assert detection["coverage"]["video_index_details"]["files_checked"] == 2
    assert detection["findings"][0]["code"] == "video.interval_past_end"
    assert any("/revision/" in req["url"] for req in public.requests)
    monkeypatch.setattr(vr, "_opener", lambda: pytest.fail("Replay attempted network"))
    assert bulk.reproduce_bulk_proposal(out)["proposed_cells"] == 6


def test_max_episodes_preserves_partial_unknown(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    out = tmp_path / "partial"
    result = bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, out, fetch=True, max_episodes=1)
    assert result["status"] == "unknown" and not result["proposal_complete"]
    assert result["detection_coverage"] == "truncated" and result["proposed_cells"] == 3


def test_unrecoverable_original_keeps_full_unknown_cases(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path, bad_original=True)
    out = tmp_path / "unresolved"
    result = bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, out, fetch=True)
    assert result["status"] == "unknown" and result["verified_associations"] == 0
    assert all(entry["status"] == "unknown" for entry in result["episodes"])
    assert len(list((out / "cases").glob("*/READY.json"))) == 2


def test_missing_full_payload_is_unknown_not_hash_only_proof(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    next((retained[1] / "payload").iterdir()).unlink()
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, tmp_path / "partial")
    assert result["status"] == "unknown" and result["verified_associations"] == 1
    assert result["episodes"][1]["status"] == "unknown"


def test_duplicate_episode_case_preserves_unknown(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, [retained[0], retained[0]], tmp_path / "dup")
    assert result["verified_associations"] == 1 and result["status"] == "unknown"
    assert "Duplicate" in result["episodes"][1]["reason"]


def test_individually_valid_cases_with_contradictory_pinned_listings_are_unknown(monkeypatch, tmp_path):
    public, native = source(monkeypatch, tmp_path)
    first = tmp_path / "first"
    vr.recover_video_reference(REPO, REVISION, ORIGINAL, 0, CAMERA, first)
    # A second self-consistent recording at the same SHA claims a different unused video0 blob.
    # Its selected episode1 native association is still individually valid.
    public.files[(REVISION, f"videos/{CAMERA}/chunk-000/file-000.mp4")] = movie((b"changed-a", b"changed-b"))
    second = tmp_path / "second"
    vr.recover_video_reference(REPO, REVISION, ORIGINAL, 1, CAMERA, second)
    assert vr.reproduce_video_recovery(second)["repair_verified"]
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, [first, second], tmp_path / "batch")
    assert result["verified_associations"] == 1 and result["status"] == "unknown"
    assert "contradict" in result["episodes"][1]["reason"]


def test_mismatched_native_snapshot_never_merges(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    table = pq.read_table(native / EPISODES)
    rows = table.to_pylist()
    rows[0]["untouched_label"] = "other-revision"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), native / EPISODES)
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, tmp_path / "mismatch")
    assert result["verified_associations"] == 0
    assert all("same complete native metadata" in e["reason"] for e in result["episodes"])


@pytest.mark.parametrize("mode", ["duplicate", "conflict", "wrong_id", "wrong_old", "outside", "negative_row"])
def test_invalid_cell_proposals_refused_before_copy(monkeypatch, tmp_path, mode):
    public, native, retained = cases(monkeypatch, tmp_path)
    proposals = json.loads((retained[0] / "repair_plan.json").read_text())
    if mode in ("duplicate", "conflict"):
        other = dict(proposals[0])
        if mode == "conflict":
            other["new"] = 99
        proposals.append(other)
    elif mode == "wrong_id":
        proposals[0]["episode_index"] = 1
    elif mode == "wrong_old":
        proposals[0]["old"] = 77
    elif mode == "outside":
        proposals[0]["path"] = "../escape.parquet"
    else:
        proposals[0]["row"] = -1
    with pytest.raises(InputError):
        bulk.merge_proposals({EPISODES: public.metadata}, proposals)


def test_budget_tracks_failed_reads_and_never_exceeds_cap(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    budget = bulk.BodyBudget(10)
    with pytest.raises(InputError, match="budget"):
        vr.recover_video_reference(REPO, REVISION, ORIGINAL, 0, CAMERA, tmp_path / "small", transport_factory=budget.transport)
    assert budget.used == 10
    assert not (tmp_path / "small").exists()


def test_existing_destination_is_not_touched(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_bytes(b"keep")
    with pytest.raises(InputError, match="exists"):
        bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    assert (out / "keep").read_bytes() == b"keep"


def test_destination_inside_native_input_is_refused_before_creation(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    out = native / "new-parent" / "batch"
    with pytest.raises(InputError, match="outside"):
        bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    assert not out.parent.exists()


def test_tampered_merged_output_cannot_replay(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    out = tmp_path / "batch"
    bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    path = out / "proposed_metadata" / EPISODES
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(InputError, match="inventory"):
        bulk.reproduce_bulk_proposal(out)


def test_resealed_wrong_merge_still_fails_native_replay(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    out = tmp_path / "batch"
    bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    path = out / "proposed_metadata" / EPISODES
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["untouched_label"] = "forged"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    bulk._seal(out)
    with pytest.raises(InputError, match="differs from full replay"):
        bulk.reproduce_bulk_proposal(out)


def test_resealed_subset_cannot_claim_complete_automatic_recovery(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    out = tmp_path / "batch"
    bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    path = out / "report.json"
    report = json.loads(path.read_text())
    report.update(status="pass", proposal_complete=True, detection_coverage="complete")
    path.write_bytes(vr._encode(report))
    bulk._seal(out)
    with pytest.raises(InputError, match="claimed summary"):
        bulk.reproduce_bulk_proposal(out)


def test_native_info_mismatch_refuses_all_cases(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    path = native / "meta/info.json"
    info = json.loads(path.read_text())
    info["fps"] = 10
    path.write_bytes(vr._encode(info))
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, tmp_path / "mismatch")
    assert result["verified_associations"] == 0


def test_native_total_coverage_must_agree_before_merging(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    path = native / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_episodes"] = 3
    path.write_bytes(vr._encode(info))
    with pytest.raises(InputError, match="identity coverage"):
        bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, tmp_path / "mismatch")


def test_schema_metadata_survives_union_across_rows(monkeypatch, tmp_path):
    public, native = source(monkeypatch, tmp_path)
    table = pq.read_table(io.BytesIO(public.metadata)).replace_schema_metadata({b"native-note": b"keep"})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    public.metadata = buf.getvalue()
    public.files[(REVISION, EPISODES)] = public.metadata
    (native / EPISODES).write_bytes(public.metadata)
    retained = []
    for episode in range(2):
        case = tmp_path / f"case-{episode}"
        vr.recover_video_reference(REPO, REVISION, ORIGINAL, episode, CAMERA, case)
        retained.append(case)
    out = tmp_path / "batch"
    result = bulk.merge_retained_cases(REPO, REVISION, ORIGINAL, native, retained, out)
    assert result["proposed_cells"] == 6
    assert pq.read_table(out / "proposed_metadata" / EPISODES).schema.metadata == {b"native-note": b"keep"}


def test_two_proposals_cannot_create_an_overlapping_final_interval(monkeypatch, tmp_path):
    _, native, retained = cases(monkeypatch, tmp_path)
    files = bulk._metadata(native)
    first = json.loads((retained[0] / "repair_plan.json").read_text())
    second = json.loads((retained[1] / "repair_plan.json").read_text())
    second = second[1:]  # keep episode1 in file1 but propose its interval at the same prefix
    with pytest.raises(InputError, match="overlap"):
        bulk.merge_proposals(files, first + second)


@pytest.mark.parametrize("kwargs", [{"fetch": False}, {"fetch": True, "max_episodes": True},
                                     {"fetch": True, "max_body_bytes": 10}, {"fetch": True, "max_episodes": 257}])
def test_bad_budget_and_network_optin_refused_without_requests(monkeypatch, tmp_path, kwargs):
    public, _ = source(monkeypatch, tmp_path)
    with pytest.raises(InputError):
        bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, tmp_path / "out", **kwargs)
    assert not public.requests
