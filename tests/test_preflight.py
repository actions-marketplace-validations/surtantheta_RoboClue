"""Preflight must find actual row defects and disclose every unassessed component."""

import copy
import hashlib
import importlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError

engine = importlib.import_module("policy_doctor.preflight")


def dataset(tmp_path, *, bad_vector=False, bad_span=False, video=False):
    root = tmp_path / "native"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "fps": 30, "total_frames": 20, "total_episodes": 2,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
                         "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]}}}
    if video:
        info["features"]["observation.images.front"] = {"dtype": "video", "shape": [32, 32, 3]}
    rows = [{"index": i, "episode_index": i // 10, "frame_index": i % 10, "timestamp": (i % 10) / 30,
             "action": [float(i), float(i + 1)],
             "observation.state": [float(i), float(i + 2)]} for i in range(20)]
    meta = [{"episode_index": e, "length": 10, "dataset_from_index": e * 10,
             "dataset_to_index": 11 if bad_span and e == 0 else (e + 1) * 10,
             "data/chunk_index": 0, "data/file_index": 0} for e in range(2)]
    (root / "meta/info.json").write_text(json.dumps(info))
    statistics = {}
    for key in ("action", "observation.state"):
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        statistics[key] = {name: value.tolist() for name, value in {
            "mean": values.mean(0), "std": values.std(0), "min": values.min(0),
            "max": values.max(0), "count": np.asarray([len(values)])}.items()}
    (root / "meta/stats.json").write_text(json.dumps(statistics))
    if bad_vector:
        rows[7]["action"][0] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(meta), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_complete_native_dataset_has_scoped_pass_and_is_unchanged(tmp_path):
    root = dataset(tmp_path)
    before = hashes(root)
    result = engine.preflight(str(root))
    assert result["kind"] == "dataset_preflight" and result["status"] == "pass"
    assert result["components"]["numeric_rows"]["coverage"]["rows_scanned"] == 20
    assert result["components"]["episode_windows"]["coverage"]["complete"]
    assert result["components"]["video_index"]["coverage"]["status"] == "not_applicable"
    assert result["deployment_status"] == "not_assessed"
    assert "training" in result["scope"] and result["next_actions"]
    assert hashes(root) == before


def test_real_numeric_and_window_errors_keep_independent_components(tmp_path):
    result = engine.preflight(str(dataset(tmp_path, bad_vector=True, bad_span=True)))
    assert result["status"] == "error"
    assert result["components"]["numeric_rows"]["status"] == "error"
    assert result["components"]["episode_windows"]["status"] == "error"
    vector = next(f for f in result["findings"] if f["code"] == "dataset.vector_invalid")
    assert vector["status"] == "error" and vector["component"] == "numeric_rows"
    assert any(f["code"] == "episode.range_contradiction" for f in result["findings"])


def test_partial_rows_are_unknown_and_never_erase_a_known_error(tmp_path):
    root = dataset(tmp_path, bad_vector=True)
    small = engine.preflight(str(root), max_rows=3)
    assert small["status"] == "unknown"
    assert not small["components"]["numeric_rows"]["coverage"]["complete_parquet_scan"]
    result = engine.preflight(str(root), max_rows=8)
    assert result["status"] == "error"
    assert result["components"]["episode_windows"]["status"] == "unknown"


def test_declared_camera_without_index_is_error(tmp_path):
    result = engine.preflight(str(dataset(tmp_path, video=True)))
    assert result["status"] == "error"
    assert result["components"]["video_index"]["status"] == "error"
    assert "video.feature_without_index" in {f["code"] for f in result["findings"]}


def test_malformed_metadata_is_unknown_not_a_dataset_fault(tmp_path):
    root = dataset(tmp_path)
    (root / "meta/info.json").write_text("{bad json")
    result = engine.preflight(str(root))
    assert result["status"] == "unknown"
    assert all(c["status"] == "unknown" for c in result["components"].values())


@pytest.mark.parametrize("kwargs", [{"max_rows": True}, {"max_rows": 200001}, {"max_files": 0},
                                    {"max_bytes": False}, {"fetch": 1}, {"video_headers": "yes"},
                                    {"deployment": "hf://org/contract"}])
def test_invalid_arguments_are_refused_before_checks(tmp_path, kwargs):
    with pytest.raises(InputError):
        engine.preflight(str(tmp_path), **kwargs)


@pytest.mark.parametrize("source,checkpoint", [("hf://org/data", None), ("local", "hf://org/policy"),
                                               ("https://example.test/data", None)])
def test_network_requires_explicit_fetch_before_any_intake(monkeypatch, source, checkpoint):
    def forbidden(*args, **kwargs):
        pytest.fail("intake was reached without network consent")
    monkeypatch.setattr(engine.hub, "resolve_dataset_content", forbidden)
    monkeypatch.setattr(engine.hub, "resolve_hub_snapshot", forbidden)
    with pytest.raises(InputError):
        engine.preflight(source, checkpoint=checkpoint)


def test_hub_dataset_intake_once_and_temporary_paths_are_portable(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    calls = []

    @contextmanager
    def intake(source, **kwargs):
        calls.append((source, kwargs))
        yield SimpleNamespace(root=root, provenance={"source": source, "resolved_revision": "a" * 40,
            "files": [{"path": k, "sha256": v} for k, v in hashes(root).items()]},
            video_context=None)

    monkeypatch.setattr(engine.hub, "resolve_dataset_content", intake)
    result = engine.preflight("hf://org/data", fetch=True, revision="a" * 40)
    assert len(calls) == 1 and calls[0][1]["video_headers"] is True
    assert calls[0][1]["max_rows"] == 200000 and calls[0][1]["max_files"] == 1024
    assert result["status"] == "pass"
    assert str(root) not in json.dumps(result)
    assert result["provenance"][0]["resolved_revision"] == "a" * 40
    assert any(a["path"].startswith("hf://org/data@") for a in result["artifacts"])


def test_missing_checkpoint_does_not_hide_dataset_error(tmp_path):
    result = engine.preflight(str(dataset(tmp_path, bad_vector=True)), checkpoint=str(tmp_path / "absent"))
    assert result["status"] == "error"
    assert result["components"]["checkpoint_metadata"]["status"] == "unknown"


def test_invalid_dataset_metadata_does_not_hide_checkpoint_error(tmp_path, case):
    root = dataset(tmp_path)
    info = json.loads((root / "meta/info.json").read_text())
    info["features"]["action"]["shape"] = []
    (root / "meta/info.json").write_text(json.dumps(info))
    post = json.loads((case["checkpoint"] / "policy_postprocessor.json").read_text())
    post["steps"].reverse()
    (case["checkpoint"] / "policy_postprocessor.json").write_text(json.dumps(post))
    result = engine.preflight(str(root), checkpoint=str(case["checkpoint"]))
    component = result["components"]["checkpoint_metadata"]
    assert component["status"] == result["status"] == "error"
    assert any(f["status"] == "error" for f in component["findings"])
    assert any(f["code"] == "preflight.dataset_interface_unassessed" for f in component["findings"])


def test_hub_checkpoint_withheld_states_remain_unknown(tmp_path, case, monkeypatch):
    root = dataset(tmp_path)
    cp = case["checkpoint"]
    (cp / "pre.safetensors").unlink()
    calls = []

    @contextmanager
    def intake(source, **kwargs):
        calls.append((source, kwargs))
        yield SimpleNamespace(root=cp, provenance={"source": source, "resolved_revision": "b" * 40,
            "files": []}, omitted_states={"policy_preprocessor.json": ["/steps/1"]})

    monkeypatch.setattr(engine.hub, "resolve_hub_snapshot", intake)
    result = engine.preflight(str(root), checkpoint="hf://org/policy", fetch=True)
    assert len(calls) == 1
    assert calls[0][1]["repo_type"] == "model"
    assert any(f["code"] == "processor.state_uninspected" and f["status"] == "unknown"
               for f in result["findings"])
    assert not any(f["code"] == "processor.state_missing" for f in result["findings"])
    assert str(cp) not in json.dumps(result)


def test_deployment_without_checkpoint_is_retained_and_explicitly_unknown(tmp_path):
    root = dataset(tmp_path)
    contract = tmp_path / "deployment.json"
    contract.write_text(json.dumps({"schema_version": 1, "fps": 30, "features": {}}))
    result = engine.preflight(str(root), deployment=str(contract))
    assert result["status"] == result["deployment_status"] == "unknown"
    component = result["components"]["deployment_contract"]
    assert component["artifacts"][0]["sha256"] == hashlib.sha256(contract.read_bytes()).hexdigest()
    assert "checkpoint is required" in component["findings"][0]["message"]


def test_supplied_deployment_is_compared_without_metadata_filtering(tmp_path, case):
    root = dataset(tmp_path)
    dep = json.loads(case["deployment"].read_text())
    dep["execute_steps"] = 99
    case["deployment"].write_text(json.dumps(dep))
    result = engine.preflight(str(root), checkpoint=str(case["checkpoint"]), deployment=str(case["deployment"]))
    component = result["components"]["deployment_contract"]
    assert result["status"] == component["status"] == "error"
    assert any(f["code"] == "timing.execution_exceeds_chunk" and f["status"] == "error"
               for f in component["findings"])
    assert component["assessment_scope"] == "declared_deployment_contract"


def test_unread_video_headers_make_preflight_unknown_even_if_rows_pass(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    original = engine.audit_episode_windows

    def windows(*args, **kwargs):
        report = original(*args, **kwargs)
        report["coverage"]["video_index_details"] = {"status": "partial", "files_unread": 1}
        return report

    monkeypatch.setattr(engine, "audit_episode_windows", windows)
    result = engine.preflight(str(root), video_headers=False)
    assert result["status"] == "unknown"
    assert result["components"]["numeric_rows"]["status"] == "pass"


def test_hidden_video_errors_survive_window_finding_truncation(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    original = engine.audit_episode_windows

    def windows(*args, **kwargs):
        report = original(*args, **kwargs)
        report.update(status="error", findings_truncated=True)
        report["coverage"]["video_index_details"] = {
            "status": "complete", "finding_counts": {"error": 1, "warning": 0, "unknown": 0, "pass": 0}}
        return report

    monkeypatch.setattr(engine, "audit_episode_windows", windows)
    result = engine.preflight(str(root))
    assert result["status"] == result["components"]["video_index"]["status"] == "error"
    assert result["components"]["video_index"]["findings_truncated"]
    assert result["findings_truncated"]


def test_different_hashes_between_components_are_disclosed(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    original = engine.scan_dataset

    def scan(*args, **kwargs):
        report = copy.deepcopy(original(*args, **kwargs))
        report["artifacts"][0]["sha256"] = "0" * 64
        return report

    monkeypatch.setattr(engine, "scan_dataset", scan)
    result = engine.preflight(str(root))
    assert result["status"] == "unknown"
    assert result["components"]["input_stability"]["status"] == "unknown"
