import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from policy_doctor import retained_metadata as retained
from policy_doctor.artifacts import InputError


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    write(root / "config.json", {"type": "act", "output_features": {
        "action": {"shape": [2], "type": "ACTION", "names": ["a", "b"]}}})
    return root


def normal(value, root):
    if isinstance(value, dict):
        return {normal(k, root): normal(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [normal(x, root) for x in value]
    return value.replace(str(root), "<ROOT>") if isinstance(value, str) else value


def test_large_weights_and_payload_directories_are_not_opened(checkpoint, tmp_path, monkeypatch):
    weights = checkpoint / "model.safetensors"
    with weights.open("wb") as stream:
        stream.truncate(4 * 1024 ** 3)
    write(checkpoint / "data/deep/episode.json", {"secret": "not metadata"})
    original = retained._read
    read = []

    def track(path, maximum):
        read.append(path)
        assert path.name != weights.name and "data" not in path.parts
        return original(path, maximum)

    monkeypatch.setattr(retained, "_read", track)
    destination = tmp_path / "copy"
    result = retained.snapshot_metadata(checkpoint, destination)
    assert {p.name for p in read} == {"config.json"}
    assert list(destination.iterdir()) == [destination / "config.json"]
    assert result["retained_files"] == [{"path": "config.json", "sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(), "size_bytes": (checkpoint / "config.json").stat().st_size}]
    assert {item["name"] for item in result["omitted_entries"]} == {"model.safetensors", "data"}
    assert result["entry"] == "." and result["loader_parity"]


def test_original_stats_and_nested_native_paths_retained(checkpoint, tmp_path):
    stats = checkpoint / "normalization/stats.safetensors"
    stats.parent.mkdir()
    save_file({"action.mean": np.array([1., 2.]), "action.std": np.ones(2)}, stats)
    write(checkpoint / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "config": {}, "state_file": "normalization/stats.safetensors"}]})
    destination = tmp_path / "copy"
    result = retained.snapshot_metadata(checkpoint / "config.json", destination)
    assert {item["path"] for item in result["retained_files"]} == {"config.json", "policy_postprocessor.json", "normalization/stats.safetensors"}
    assert (destination / "normalization/stats.safetensors").read_bytes() == stats.read_bytes()
    original = retained.evaluate_metadata(checkpoint)
    copied = retained.evaluate_metadata(destination)
    assert normal(original, checkpoint) == normal(copied, destination)


def test_missing_reference_keeps_existing_error_and_custom_unknown(checkpoint, tmp_path):
    write(checkpoint / "policy_preprocessor.json", {"steps": [{"registry_name": "new_community_processor", "config": {}, "state_file": "missing.safetensors"}]})
    destination = tmp_path / "copy"
    receipt = retained.snapshot_metadata(checkpoint, destination)
    assert "missing.safetensors" in receipt["missing_metadata"]
    before = retained.evaluate_metadata(checkpoint)
    after = retained.evaluate_metadata(destination)
    assert normal(before, checkpoint) == normal(after, destination)
    assert any(item["status"] == "unknown" for item in after["findings"])


def test_dataset_alias_stats_and_schema_are_not_rewritten(tmp_path):
    root = tmp_path / "dataset"
    write(root / "meta/info.json", {"fps": 30, "features": {"action": {"dtype": "float32", "shape": [2], "names": {"motors": ["a", "b"]}}}})
    write(root / "meta/stats.json", {"action": {"mean": [0, 0], "std": [1, 1]}})
    (root / "videos").mkdir()
    result = retained.snapshot_metadata(root / "meta/info.json", tmp_path / "copy", role="dataset")
    assert result["entry"] == "."
    for item in result["retained_files"]:
        assert (tmp_path / "copy" / item["path"]).read_bytes() == (root / item["path"]).read_bytes()


@pytest.mark.parametrize("bad", [{"fetch": True}, {"revision": "main"}, {"scope": "other"}, {"scope": None}, [], 2])
def test_parameters_reject_unknown_or_network_options(bad):
    with pytest.raises(InputError):
        retained.validate_metadata_parameters(bad)


def test_scope_is_explicit_and_offline(checkpoint, monkeypatch):
    calls = []
    monkeypatch.setattr(retained, "check_checkpoint", lambda *a, **kw: calls.append((a, kw)) or {})
    retained.evaluate_metadata(checkpoint)
    retained.evaluate_metadata(checkpoint, parameters={"scope": "deployment"})
    assert [call[1] for call in calls] == [{"fetch": False, "scope": "metadata"}, {"fetch": False, "scope": "deployment"}]


@pytest.mark.parametrize("reference", ["../escaped.safetensors", "/tmp/escaped.safetensors", "https://evil/x.safetensors", "a\\b.safetensors"])
def test_unsafe_stats_references_refused_before_loading(checkpoint, tmp_path, monkeypatch, reference):
    write(checkpoint / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "config": {}, "state_file": reference}]})
    monkeypatch.setattr(retained, "load_checkpoint", lambda *a: pytest.fail("unsafe source reached native loader"))
    with pytest.raises(InputError):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


@pytest.mark.parametrize("path", ["config.json", "policy_postprocessor.json", "stats.safetensors"])
def test_symlinks_refused(checkpoint, tmp_path, path):
    external = tmp_path / "external"
    external.write_text("{}")
    if path == "stats.safetensors":
        write(checkpoint / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "config": {}, "state_file": path}]})
    (checkpoint / path).unlink(missing_ok=True)
    (checkpoint / path).symlink_to(external)
    with pytest.raises(InputError, match="symlink"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")


def test_referenced_weights_are_never_read(checkpoint, tmp_path):
    (checkpoint / "model.safetensors").write_bytes(b"sensitive weights")
    write(checkpoint / "policy_postprocessor.json", {"steps": [{"registry_name": "unnormalizer_processor", "config": {}, "state_file": "model.safetensors"}]})
    with pytest.raises(InputError, match="Model weight"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")


@pytest.mark.parametrize("destination", ["checkpoint/nested", "checkpoint"])
def test_nested_or_existing_destination_refused(checkpoint, tmp_path, destination):
    with pytest.raises(InputError):
        retained.snapshot_metadata(checkpoint, tmp_path / destination)
    assert (checkpoint / "config.json").exists()


def test_existing_destination_not_modified(checkpoint, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "sentinel").write_text("keep")
    with pytest.raises(InputError, match="exists"):
        retained.snapshot_metadata(checkpoint, destination)
    assert (destination / "sentinel").read_text() == "keep"


def test_mutation_during_source_loader_refused(checkpoint, tmp_path, monkeypatch):
    original = retained.load_checkpoint

    def change(path):
        result = original(path)
        if path == checkpoint:
            write(path / "config.json", {"type": "changed"})
        return result

    monkeypatch.setattr(retained, "load_checkpoint", change)
    with pytest.raises(InputError, match="changed"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_absent_file_appears_during_capture_refused(checkpoint, tmp_path, monkeypatch):
    original = retained.load_checkpoint

    def change(path):
        result = original(path)
        if path == checkpoint:
            write(path / "policy_preprocessor.json", {"steps": []})
        return result

    monkeypatch.setattr(retained, "load_checkpoint", change)
    with pytest.raises(InputError, match="appeared"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")


def test_duplicate_json_keys_not_silently_repaired(checkpoint, tmp_path):
    (checkpoint / "config.json").write_text('{"type":"act","type":"other"}')
    with pytest.raises(InputError, match="Duplicate"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")


def test_omission_inventory_bounded_without_recursive_walk(checkpoint, tmp_path):
    for i in range(150):
        (checkpoint / f"unrelated-{i}").mkdir()
    receipt = retained.snapshot_metadata(checkpoint, tmp_path / "copy")
    assert len(receipt["omitted_entries"]) == retained.MAX_OMITTED_ENTRIES
    assert receipt["omitted_entries_truncated"]


def test_same_bytes_inode_replacement_during_capture_refused(checkpoint, tmp_path, monkeypatch):
    original = retained.load_checkpoint

    def replace(path):
        result = original(path)
        if path == checkpoint:
            content = (path / "config.json").read_bytes()
            (path / "replacement").write_bytes(content)
            (path / "replacement").replace(path / "config.json")
        return result

    monkeypatch.setattr(retained, "load_checkpoint", replace)
    with pytest.raises(InputError, match="changed"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")


def test_native_audit_scope_preserved_for_pair(checkpoint, tmp_path):
    dataset = tmp_path / "dataset"
    write(dataset / "meta/info.json", {"fps": 30, "features": {"action": {
        "dtype": "float32", "shape": [2], "names": ["b", "a"]}}})
    original = retained.evaluate_metadata(checkpoint, dataset)
    metadata = retained.evaluate_metadata(checkpoint, dataset, {"scope": "metadata"})
    deployment = retained.evaluate_metadata(checkpoint, dataset, {"scope": "deployment"})
    assert original == metadata
    assert metadata["assessment_scope"] == "metadata"
    assert deployment["assessment_scope"] == "deployment"
    assert any(item["code"] == "deployment.missing" for item in metadata["unassessed_findings"])
    assert any(item["code"] == "deployment.missing" for item in deployment["findings"])
    assert metadata["status"] == "unknown"  # Missing processors do not establish order semantics.


def test_total_snapshot_budget_refused_before_native_loader(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(retained, "MAX_RETAINED_BYTES", 3)
    monkeypatch.setattr(retained, "load_checkpoint", lambda *a: pytest.fail("over-budget native load"))
    with pytest.raises(InputError, match="budget"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_duplicate_native_path_alias_not_accepted(checkpoint, tmp_path):
    save_file({"action.mean": np.ones(2)}, checkpoint / "stats.safetensors")
    write(checkpoint / "policy_postprocessor.json", {"steps": [
        {"registry_name": "unnormalizer_processor", "state_file": "stats.safetensors"},
        {"registry_name": "unnormalizer_processor", "state_file": "./stats.safetensors"}]})
    with pytest.raises(InputError, match="alias|read set"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")


@pytest.mark.parametrize("role", [None, [], "other"])
def test_invalid_role_has_input_error(checkpoint, tmp_path, role):
    with pytest.raises(InputError, match="role"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy", role=role)


def test_missing_required_native_metadata_refused(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(InputError, match="Required"):
        retained.snapshot_metadata(root, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_hardlinked_stats_aliases_refused(checkpoint, tmp_path):
    save_file({"action.mean": np.ones(2)}, checkpoint / "stats.safetensors")
    (checkpoint / "alias.safetensors").hardlink_to(checkpoint / "stats.safetensors")
    write(checkpoint / "policy_postprocessor.json", {"steps": [
        {"registry_name": "unnormalizer_processor", "state_file": "stats.safetensors"},
        {"registry_name": "unnormalizer_processor", "state_file": "alias.safetensors"}]})
    with pytest.raises(InputError, match="alias"):
        retained.snapshot_metadata(checkpoint, tmp_path / "copy")
