import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from policy_doctor.artifacts import InputError, load_checkpoint, load_dataset, load_deployment, read_json


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_native_checkpoint_does_not_load_weights(tmp_path):
    write(tmp_path / "config.json", {"type": "act", "output_features": {"action": {"shape": [2], "type": "ACTION"}}})
    (tmp_path / "model.safetensors").write_bytes(b"not a real model")
    snapshot = load_checkpoint(tmp_path)
    assert snapshot.features["action"]["shape"] == [2]
    assert [x.role for x in snapshot.artifacts] == ["checkpoint_config"]


def test_native_processor_state_and_provenance(tmp_path):
    write(tmp_path / "config.json", {"type": "act"})
    save_file({"action.mean": np.array([1., 2.]), "action.std": np.ones(2)}, tmp_path / "stats.safetensors")
    write(tmp_path / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "config": {}, "state_file": "stats.safetensors"}]})
    snapshot = load_checkpoint(tmp_path)
    assert snapshot.post[0]["stats"]["action"]["mean"] == [1., 2.]
    assert len(snapshot.artifacts) == 3
    assert len(snapshot.artifacts[-1].sha256) == 64


@pytest.mark.parametrize("content", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '[]', '{'])
def test_invalid_json(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(InputError):
        read_json(path)


@pytest.mark.parametrize("bad_shape", [[0], [-1], [True], [], "2"])
def test_bad_feature_shape(tmp_path, bad_shape):
    write(tmp_path / "config.json", {"type": "act", "output_features": {"action": {"shape": bad_shape}}})
    with pytest.raises(InputError):
        load_checkpoint(tmp_path)


@pytest.mark.parametrize("reference", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_processor_traversal_rejected(tmp_path, reference):
    write(tmp_path / "config.json", {"type": "act"})
    write(tmp_path / "policy_preprocessor.json", {"steps": [{"registry_name": "normalizer_processor", "state_file": reference}]})
    with pytest.raises(InputError):
        load_checkpoint(tmp_path)


def test_processor_symlink_escape(tmp_path):
    root = tmp_path / "checkpoint"
    write(root / "config.json", {"type": "act"})
    (tmp_path / "outside.safetensors").write_bytes(b"x")
    (root / "link.safetensors").symlink_to(tmp_path / "outside.safetensors")
    write(root / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "state_file": "link.safetensors"}]})
    with pytest.raises(InputError):
        load_checkpoint(root)


def test_dataset_names_mapping(tmp_path):
    write(tmp_path / "meta/info.json", {"fps": 30, "features": {"action": {"shape": [2], "names": {"motors": ["a", "b"]}}}})
    assert load_dataset(tmp_path).features["action"]["names"] == ["a", "b"]


def test_ambiguous_names_rejected_and_duplicates_reported(tmp_path):
    for names in ({}, {"one": ["a"], "two": ["b"]}):
        write(tmp_path / "meta/info.json", {"fps": 30, "features": {"action": {"shape": [2], "names": names}}})
        with pytest.raises(InputError):
            load_dataset(tmp_path)
    # Repeated per-coordinate dataset labels: order is withheld, never silently accepted or crashed on.
    write(tmp_path / "meta/info.json", {"fps": 30, "features": {"action": {"shape": [2], "names": ["a", "a"]}}})
    action = load_dataset(tmp_path).features["action"]
    assert action["names"] is None and action["raw_names"] == ["a", "a"]
    assert action["coordinate_names_status"] == "duplicate"


def test_deployment_version_and_execution_steps(tmp_path):
    for obj in ({}, {"schema_version": 1, "execute_steps": 1.5}, {"schema_version": 1, "fps": -1}):
        with pytest.raises(InputError):
            load_deployment(write(tmp_path / "deploy.json", obj))


def test_missing_processor_state_is_retained(tmp_path):
    write(tmp_path / "config.json", {"type": "act"})
    write(tmp_path / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "state_file": "missing.safetensors"}]})
    assert load_checkpoint(tmp_path).post[0]["missing_state"]
