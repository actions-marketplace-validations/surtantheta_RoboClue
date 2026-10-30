import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.dataset import scan_dataset


def rows():
    return [{"action": [1., .5], "observation.state": [2., .5], "episode_index": 0,
             "frame_index": i, "timestamp": i / 30} for i in range(5)]


def parquet(case, records):
    root = case["dataset"] / "data/chunk-000"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), root / "file-000.parquet")


def test_healthy_values_and_explicit_coverage(case):
    parquet(case, rows())
    report = scan_dataset(str(case["dataset"]))
    assert report["status"] == "pass"
    assert report["coverage"]["rows_scanned"] == 5
    assert report["coverage"]["complete_parquet_scan"]
    assert not report["coverage"]["videos_checked"]


def test_faulty_vectors_and_times(case):
    data = rows()
    data[2]["action"] = [float("nan"), 1]
    data[3]["timestamp"] = data[2]["timestamp"]
    data[4]["frame_index"] = 20
    parquet(case, data)
    codes = {f["code"] for f in scan_dataset(str(case["dataset"]))["findings"]}
    assert {"dataset.vector_invalid", "dataset.time_not_increasing", "dataset.frame_sequence"} <= codes


def test_cap_never_claims_complete_scan(case):
    parquet(case, rows())
    report = scan_dataset(str(case["dataset"]), max_rows=2)
    assert report["status"] == "unknown"
    assert not report["coverage"]["complete_parquet_scan"]
    assert report["coverage"]["rows_scanned"] == 2


def test_incomplete_export_detects_metadata_count(case):
    parquet(case, rows())
    path = case["dataset"] / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_frames"] = 100
    case["put"](path, info)
    assert any(f["code"] == "dataset.total_frames" for f in scan_dataset(str(case["dataset"]))["findings"])


def test_metadata_only_download_has_no_value_claim(case):
    with pytest.raises(InputError, match="No data"):
        scan_dataset(str(case["dataset"]))


def test_valid_episode_reset_not_timestamp_failure(case):
    data = rows() + [dict(row, episode_index=1) for row in rows()]
    parquet(case, data)
    assert scan_dataset(str(case["dataset"]))["status"] == "pass"


def test_partial_scan_says_which_budget_stopped_it(case):
    # Live fetch of lerobot/pusht with the default budget ended unknown with no explanation.
    parquet(case, rows())
    report = scan_dataset(str(case["dataset"]), max_rows=2)
    assert "max_rows=2" in report["summary"] and "--max-rows" in report["summary"]
    complete = scan_dataset(str(case["dataset"]), max_rows=1000)
    assert "summary" not in complete


def test_explicit_other_backend_never_diagnoses_stale_parquet_values(case):
    data = rows()
    data[2]["action"] = [float("nan"), 1]
    data[3]["timestamp"] = data[2]["timestamp"]
    parquet(case, data)
    info_path = case["dataset"] / "meta/info.json"
    info = json.loads(info_path.read_text())
    assert scan_dataset(str(case["dataset"]))["status"] == "error"
    info_path.write_text(json.dumps(dict(info, storage_format="lance")))
    report = scan_dataset(str(case["dataset"]))
    assert report["status"] == "unknown" and report["coverage"]["rows_scanned"] == 0
    assert [f["code"] for f in report["findings"]] == ["dataset.storage_format_unassessed"]
    info_path.write_text(json.dumps(dict(info, storage_format="parquet")))
    assert scan_dataset(str(case["dataset"]))["status"] == "error"
