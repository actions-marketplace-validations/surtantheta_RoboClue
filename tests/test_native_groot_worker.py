"""Provider intake/replay guards and opt-in parity against original native evidence."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from policy_doctor import native_groot_worker as worker


@pytest.fixture
def probe_request(tmp_path):
    source, model, dataset = (tmp_path / name for name in ("source", "model", "dataset"))
    source.mkdir()
    model.mkdir()
    (dataset / "meta").mkdir(parents=True)
    config = {"processor_kwargs": {**{key: False for key in worker.FLAGS},
              "modality_configs": {"gr1": {"state": {"modality_keys": ["arm"]}}}}}
    (model / "processor_config.json").write_text(json.dumps(config))
    (model / "statistics.json").write_text("{}")
    (dataset / "meta/modality.json").write_text(json.dumps({"state": {"arm": {"start": 0, "end": 2}}}))
    (dataset / "meta/info.json").write_text(json.dumps({"features": {"observation.state": {"shape": [2]}}}))
    return {"operation": "probe", "source_root": str(source), "model": str(model), "dataset": str(dataset),
            "output": str(tmp_path / "output"), "samples": [{"episode_index": 0, "frame_index": 1}],
            "raw_samples": [{"identity": {"episode_index": 0, "frame_index": 1, "index": 1,
                                          "timestamp": 0.1, "physical_row": 1},
                             "columns": {"observation.state": [0.25, 0.5]},
                             "source": {"path": "data/chunk-000/episode_000000.parquet", "sha256": "a" * 64}}]}


def test_preparation_preserves_original_identity_raw_values_and_explicit_slices(probe_request):
    _, _, _, bindings, rows, retained = worker._prepare(probe_request)
    assert rows == probe_request["raw_samples"]
    assert bindings == [{"key": "arm", "original_key": "observation.state", "start": 0, "end": 2,
                         "declared_column_shape": [2]}]
    assert retained["metadata/dataset/meta/modality.json"] == (Path(probe_request["dataset"]) / "meta/modality.json").read_bytes()


@pytest.mark.parametrize("mutation, reason", [
    (lambda r: r.update(operation="invoke_arbitrary_callable"), "operation"),
    (lambda r: r["raw_samples"][0]["identity"].update(index=True), "integer"),
    (lambda r: r["raw_samples"][0]["identity"].update(frame_index=9), "identity"),
    (lambda r: r["raw_samples"][0]["source"].update(path="../escape.parquet"), "traversal"),
    (lambda r: r["raw_samples"][0]["source"].update(path="/volatile/episode.parquet"), "Absolute"),
    (lambda r: r["raw_samples"][0]["source"].update(sha256="not-original-digest"), "SHA256"),
    (lambda r: r["raw_samples"][0]["columns"].update({"observation.state": [0.1]}), "width"),
    (lambda r: r["raw_samples"][0]["columns"].update({"observation.state": [0.1, "callable()"]}), "finite"),
    (lambda r: r["raw_samples"][0]["columns"].clear(), "width"),
])
def test_invalid_original_identity_and_generated_input_controls_refused_before_import(probe_request, mutation, reason):
    mutation(probe_request)
    result = worker.run(probe_request)
    assert result["execution_status"] == "error"
    assert reason.lower() in result["error"]["message"].lower()
    assert result["source_files"] == []
    assert not Path(probe_request["output"]).exists()


def test_no_equal_width_inference_when_original_mapping_missing(probe_request):
    (Path(probe_request["dataset"]) / "meta/modality.json").write_text('{"state": {}}')
    result = worker.run(probe_request)
    assert result["execution_status"] == "error"
    assert "mapping missing" in result["error"]["message"]


def test_missing_statistics_refused_as_missing_original_metadata(probe_request):
    (Path(probe_request["model"]) / "statistics.json").unlink()
    result = worker.run(probe_request)
    assert result["execution_status"] == "error"
    assert "statistics.json" in result["error"]["message"]
    # This intake refusal is distinct from the known native sin/cos path's
    # acceptance of statistics=None; the worker retains original metadata.


def test_duplicate_json_keys_and_symlinked_metadata_refused(probe_request, tmp_path):
    path = Path(probe_request["dataset"]) / "meta/modality.json"
    path.write_text('{"state": {}, "state": {}}')
    assert "Duplicate" in worker.run(probe_request)["error"]["message"]
    original = tmp_path / "original.json"
    original.write_text('{"state": {}}')
    path.unlink()
    path.symlink_to(original)
    assert "symlink" in worker.run(probe_request)["error"]["message"]


def test_refused_output_is_not_modified(probe_request):
    out = Path(probe_request["output"])
    out.mkdir()
    (out / "existing.txt").write_text("preserve")
    result = worker.run(probe_request)
    assert result["execution_status"] == "error"
    assert list(out.iterdir()) == [out / "existing.txt"]


def test_replay_rejects_changed_retained_bytes_without_import(tmp_path):
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "raw.json").write_text('{"raw_samples": []}')
    (capsule / "worker_manifest.json").write_text(json.dumps({"provider": worker.PROVIDER, "schema_version": 1,
        "retained_files": [{"path": "raw.json", "sha256": "0" * 64}]}))
    result = worker.run({"operation": "replay", "capsule_root": str(capsule), "source_root": str(tmp_path),
                         "output": str(tmp_path / "new-output")})
    assert result["execution_status"] == "error"
    assert "hash mismatch" in result["error"]["message"]


def test_request_response_bounded():
    result = worker.run({"operation": "probe", "oversized": "x" * worker.MAX_REQUEST})
    assert result["execution_status"] == "error"
    assert len(json.dumps(result).encode()) < worker.MAX_RESPONSE


def _native_configuration():
    names = ("SOURCE", "MODEL", "DATASET", "PYTHON", "ORACLE", "ROW0", "ROW1")
    values = {name: os.environ.get("POLICY_DOCTOR_GROOT_" + name) for name in names}
    if not all(values.values()):
        pytest.skip("Original native GR1 runtime/oracle paths not explicitly configured")
    return values


def test_actual_original_native_oracle_parity_and_relocated_replay(tmp_path):
    native = _native_configuration()
    rows = []
    for number in range(2):
        row = json.loads(Path(native[f"ROW{number}"]).read_text())
        raw = row["native_values"]
        rows.append({"identity": {"episode_index": raw["episode_index"], "frame_index": None,
                                  "index": raw["index"], "physical_row": number, "timestamp": raw["timestamp"]},
                     "selection_basis": "single_episode_physical_row",
                     "columns": {"observation.state": raw["observation.state"]},
                     "source": {"path": "data/chunk-000/episode_000000.parquet", "sha256": row["source_sha256"]}})
    probe_request = {"operation": "probe", "source_root": native["SOURCE"], "model": native["MODEL"],
               "dataset": native["DATASET"], "output": str(tmp_path / "probe"),
               "samples": [{"episode_index": 0, "physical_row": n} for n in range(2)], "raw_samples": rows}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def execute(payload):
        completed = subprocess.run([native["PYTHON"], "-B", worker.__file__], input=json.dumps(payload),
                                   text=True, capture_output=True, env=env, timeout=30, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert not completed.stderr
        return json.loads(completed.stdout)
    first = execute(probe_request)
    assert first["execution_status"] == "complete"
    oracle = json.loads(Path(native["ORACLE"]).read_text())
    for actual, expected in zip(first["samples"], oracle["rows"], strict=False):
        assert actual["identity"] == rows[actual["identity"]["physical_row"]]["identity"]
        route = actual["routes"][0]
        assert route["status"] == "complete"
        assert route["dimensions"]["native_declared_base"] == 29
        assert route["dimensions"]["native_declared_expanded"] == route["dimensions"]["observed_output"] == 58
        for key, value in route["output"].items():
            reference = expected["native_output"][key]
            assert value["sha256"] == reference["sha256_c_bytes"]
            assert (value["shape"], value["dtype"], value["finite"]) == (reference["shape"], reference["dtype"], True)
    relocated = tmp_path / "relocated-capsule"
    Path(probe_request["output"]).rename(relocated)
    replay = execute({"operation": "replay", "source_root": native["SOURCE"],
                      "capsule_root": str(relocated), "output": str(tmp_path / "replay")})
    assert replay["samples"] == first["samples"]
    assert replay["source_files"] == first["source_files"]
    assert not any(str(tmp_path) in json.dumps(sample) for sample in replay["samples"])
    for descriptor in replay["exports"]:
        contents = (tmp_path / "replay" / descriptor["path"]).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == descriptor["sha256"]
    (relocated / "inputs/raw_samples.json").write_text('{"raw_samples": []}')
    bad = subprocess.run([native["PYTHON"], "-B", worker.__file__], input=json.dumps({
        "operation": "replay", "source_root": native["SOURCE"], "capsule_root": str(relocated),
        "output": str(tmp_path / "bad-replay")}), text=True, capture_output=True, env=env, timeout=30)
    assert bad.returncode == 1
    assert "hash mismatch" in json.loads(bad.stdout)["error"]["message"]


def test_physical_row_selection_preserves_absent_native_frame(probe_request):
    probe_request['samples'] = [{'episode_index': 0, 'physical_row': 1}]
    row = probe_request['raw_samples'][0]
    row['identity']['frame_index'] = None
    row['selection_basis'] = 'single_episode_physical_row'
    prepared = worker._prepare(probe_request)
    assert prepared[4][0]['identity']['frame_index'] is None
    assert prepared[4][0]['identity']['physical_row'] == 1
    assert prepared[4][0]['selection_basis'] == 'single_episode_physical_row'


def test_physical_row_selection_requires_explicit_basis_and_matching_row(probe_request):
    probe_request['samples'] = [{'episode_index': 0, 'physical_row': 1}]
    probe_request['raw_samples'][0]['identity']['frame_index'] = None
    with pytest.raises(ValueError, match='explicit'):
        worker._prepare(probe_request)
    probe_request['raw_samples'][0]['selection_basis'] = 'single_episode_physical_row'
    probe_request['samples'][0]['physical_row'] = 0
    with pytest.raises(ValueError, match='does not match'):
        worker._prepare(probe_request)


def test_frame_selector_does_not_silently_alias_absent_frame_column(probe_request):
    probe_request['raw_samples'][0]['identity']['frame_index'] = None
    with pytest.raises(ValueError, match='frame_index'):
        worker._prepare(probe_request)


def test_original_parquet_receipt_is_verified_before_native_import(probe_request):
    path = Path(probe_request['dataset']) / probe_request['raw_samples'][0]['source']['path']
    path.parent.mkdir(parents=True)
    path.write_text('version https://git-lfs.github.com/spec/v1\nunit pointer is not original Parquet\n')
    result = worker.run(probe_request)
    assert result['execution_status'] == 'error'
    assert 'disagree with raw-sample source SHA256' in result['error']['message']
    assert result['source_files'] == []
