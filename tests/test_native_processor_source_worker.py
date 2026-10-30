"""Source-worker boundary tests. Generated fixtures are not native evidence."""
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).parents[1] / "src/policy_doctor/native_processor_source_worker.py"
SPEC = importlib.util.spec_from_file_location("isolated_native_processor_source_worker", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def model(tmp_path, **changes):
    root = tmp_path / "model"
    root.mkdir()
    cfg = {"type": "smolvla", "input_features": {"observation.state": {"type": "STATE", "shape": [9]}},
           "output_features": {"action": {"type": "ACTION", "shape": [5]}},
           "vlm_model_name": worker.TOKENIZER_ID, "pretrained_path": "organization/training-origin"}
    cfg.update(changes)
    (root / "config.json").write_text(json.dumps(cfg))
    for name in ("policy_preprocessor", "policy_postprocessor"):
        (root / (name + ".json")).write_text(json.dumps({"name": name, "steps": [{"registry_name": "to_batch_processor", "config": {}}]}))
    return root


def test_standalone_import_is_standard_library_only():
    code = "import runpy,sys;runpy.run_path(sys.argv[1]);assert not any(x=='torch' or x=='numpy' or x.startswith(('lerobot','policy_doctor')) for x in sys.modules)"
    result = subprocess.run([sys.executable, "-I", "-c", code, str(WORKER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for node in ast.parse(WORKER.read_text()).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module]
            assert not any(x.startswith(("torch", "numpy", "lerobot", "policy_doctor")) for x in names)


def test_model_validator_accepts_other_widths_and_origin_as_data(tmp_path):
    root = model(tmp_path)
    assert worker.validate_model(root)["input_features"]["observation.state"]["shape"] == [9]
    assert worker.validate_model(root)["pretrained_path"] == "organization/training-origin"


@pytest.mark.parametrize("changes", [{"type": "xvla"}, {"auto_map": {"AutoModel": "artifact.Code"}}, {"vlm_model_name": "arbitrary/tokenizer"}, {"tokenizer_max_length": 100000}, {"tokenizer_max_length": True}])
def test_model_validator_refuses_unsupported_family_and_code(tmp_path, changes):
    with pytest.raises(ValueError):
        worker.validate_model(model(tmp_path, **changes))


@pytest.mark.parametrize("entry", [
    {"class": "artifact.Run", "config": {}},
    {"registry_name": "to_batch_processor", "config": {"callback": "artifact.run"}},
    {"registry_name": "normalizer_processor", "config": {"stats": {"observation.state": {"mean": [1]}}}},
    {"registry_name": "normalizer_processor", "config": {}, "state_file": "../escape.safetensors"},
    {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "unapproved/repo"}},
])
def test_pipeline_constructor_and_reference_guards(tmp_path, entry):
    root = model(tmp_path)
    (root / "policy_preprocessor.json").write_text(json.dumps({"name": "policy_preprocessor", "steps": [entry]}))
    with pytest.raises(ValueError):
        worker._validate_pipeline(root, "policy_preprocessor.json")


def test_local_tokenizer_vocab_spellings_are_data_but_real_hooks_refused(tmp_path):
    root = model(tmp_path)
    tok = root / "tokenizer"
    tok.mkdir()
    (tok / "vocab.json").write_text(json.dumps({"module": 1, "auto_map": 2}))
    (tok / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {"module": 1, "auto_map": 2}}}))
    (tok / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GPT2Tokenizer"}))
    entry = {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "tokenizer", "max_length": 37}, "artifacts": {"tokenizer_name": "tokenizer"}}
    (root / "policy_preprocessor.json").write_text(json.dumps({"name": "policy_preprocessor", "steps": [entry]}))
    worker._validate_pipeline(root, "policy_preprocessor.json")
    worker._validate_pipeline(root, "policy_preprocessor.json", replay=True)
    (tok / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GPT2Tokenizer", "auto_map": {"AutoTokenizer": "artifact.Code"}}))
    with pytest.raises(ValueError):
        worker._validate_pipeline(root, "policy_preprocessor.json")


def test_original_cuda_declaration_is_not_executed_by_stdlib_validator(tmp_path):
    root = model(tmp_path)
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": [{"registry_name": "device_processor", "config": {"device": "cuda"}}]}))
    worker._validate_pipeline(root, "policy_preprocessor.json")


def request(tmp_path, **changes):
    req = {"operation": "probe", "source_root": str(tmp_path / "source"), "model": str(tmp_path / "model"),
           "dataset": str(tmp_path / "dataset"), "output": str(tmp_path / "output"),
           "samples": [{"episode_index": 4, "frame_index": 3}], "processor_source": "selected_checkpoint", "intent": None, "approved_sources": []}
    req.update(changes)
    return req


@pytest.mark.parametrize("samples", [[], [{"episode_index": 1, "frame_index": 0}] * 2,
    [{"episode_index": True, "frame_index": 0}], [{"episode_index": 1, "frame_index": -1}]])
def test_singular_auditor_scope_and_native_selector_boundaries(tmp_path, samples):
    with pytest.raises(ValueError):
        worker._validate_request(request(tmp_path, samples=samples))


def test_serialized_origin_requires_exact_explicit_approved_id(tmp_path):
    req = request(tmp_path, processor_source="serialized_pretrained_path", approved_sources=[{"repo_id": "org/allowed", "revision": "a" * 40, "model": str(tmp_path / "alternate")}])
    assert worker._resolve_consumer_source(req, {"pretrained_path": "org/allowed"})["repo_id"] == "org/allowed"
    for value in ["org/other", "/arbitrary/path", "../escape", None]:
        with pytest.raises(ValueError):
            worker._resolve_consumer_source(req, {"pretrained_path": value})


def test_selected_route_plan_reuses_one_honest_call_id():
    plan = worker._route_plan("selected_checkpoint")
    assert plan["consumer"] == plan["selected_reference"]
    assert worker._route_plan("fresh_training")["consumer"] != worker._route_plan("fresh_training")["selected_reference"]
    shared = {"call_id": plan["consumer"], "trace": {"path": "actual.json", "sha256": "a" * 64}}
    consumer = worker._route_alias(shared, "consumer")
    reference = worker._route_alias(shared, "selected_reference")
    assert consumer["name"] == "consumer" and reference["name"] == "selected_reference"
    assert consumer["call_id"] == reference["call_id"] and consumer["trace"] == reference["trace"]


@pytest.mark.parametrize("claim", [{"repo_id": "untrusted/claim"}, "line\nbreak", "x" * 513, 42])
def test_selected_identity_is_a_bounded_printable_label_not_freeform_json(tmp_path, claim):
    with pytest.raises(ValueError):
        worker._validate_request(request(tmp_path, selected_identity=claim))


def test_guard_failure_writes_bounded_fixed_receipt_without_native_imports(tmp_path):
    root = model(tmp_path)
    for name in ["source", "dataset"]:
        (tmp_path / name).mkdir()
    req = request(tmp_path, model=str(root), processor_source="serialized_pretrained_path")
    before = {name for name in sys.modules if name.startswith(("lerobot", "torch", "numpy"))}
    result = worker.run(req)
    assert result["execution_status"] == "error"
    receipt = Path(req["output"]) / "worker_receipt.json"
    assert receipt.stat().st_size <= worker.MAX_RESPONSE
    assert json.loads(receipt.read_text()) == result
    assert result["result"]["path"] == "result.json"
    assert {name for name in sys.modules if name.startswith(("lerobot", "torch", "numpy"))} == before


def test_invalid_existing_output_is_not_mutated(tmp_path):
    root = model(tmp_path)
    for name in ["source", "dataset"]:
        (tmp_path / name).mkdir()
    marker = root / "keep.txt"
    marker.write_text("original")
    result = worker.run(request(tmp_path, model=str(root), output=str(root)))
    assert result["execution_status"] == "error"
    assert marker.read_text() == "original"
    assert not (root / "worker_receipt.json").exists()
