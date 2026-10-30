"""Dataset labels may be readable without naming each vector coordinate."""

import hashlib
import json

import pytest

from policy_doctor.artifacts import InputError, load_checkpoint, load_dataset
from policy_doctor.diagnostics import audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def pair(root, *, names, dataset_width=3, policy_width=2):
    cp, ds = root / "checkpoint", root / "dataset"
    write(cp / "config.json", {"type": "act", "input_features": {
        "observation.state": {"type": "STATE", "shape": [policy_width]}}, "output_features": {
        "action": {"type": "ACTION", "shape": [2]}}})
    write(ds / "meta/info.json", {"fps": 30, "features": {
        "observation.state": {"shape": [dataset_width], "names": names},
        "action": {"shape": [2], "names": ["a", "b"]}}})
    write(ds / "meta/stats.json", {"observation.state": {"mean": [1] * dataset_width}})
    return cp, ds


@pytest.mark.parametrize("names", [["state"], {"axes": None}])
def test_unresolved_labels_preserve_raw_names_bytes_and_statistics(tmp_path, names):
    _, ds = pair(tmp_path, names=names)
    source = ds / "meta/info.json"
    before = source.read_bytes()
    snapshot = load_dataset(ds)
    feature = snapshot.features["observation.state"]
    assert feature["names"] is None and feature["raw_names"] == names
    assert feature["coordinate_names_status"] == "unresolved"
    assert feature["shape"] == [3]
    assert snapshot.config["features"]["observation.state"]["names"] == names
    assert snapshot.config["stats"]["observation.state"]["mean"] == [1, 1, 1]
    assert source.read_bytes() == before
    assert next(a for a in snapshot.artifacts if a.role == "dataset_info").sha256 == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize("names", [["state"], {"axes": None}])
def test_unresolved_names_keep_dimension_mismatch_and_evidence(tmp_path, names):
    cp, ds = pair(tmp_path, names=names)
    result = audit(str(cp), str(ds))
    shape = next(f for f in result["findings"] if f["code"] == "dataset.feature_shape"
                 and "observation.state" in f["message"])
    # ACT sizes its layers from the declared width (modeling_act.py:314): a real mismatch (validator round 2).
    assert shape["status"] == "error"
    unresolved = next(f for f in result["findings"] if f["code"] == "dataset.coordinate_names_unresolved")
    assert unresolved["status"] == "unknown"
    witness = unresolved["evidence"][0]
    assert witness["pointer"] == "/features/observation.state/names" and witness["value"] == names
    assert witness["sha256"] == hashlib.sha256((ds / "meta/info.json").read_bytes()).hexdigest()
    assert any(a["role"] == "dataset_stats" for a in result["artifacts"])


def test_equal_width_is_checked_without_claiming_joint_order(tmp_path):
    cp, ds = pair(tmp_path, names=["state"], dataset_width=2)
    result = audit(str(cp), str(ds))
    assert any(f["code"] == "dataset.feature_shape" and f["status"] == "pass" for f in result["findings"])
    assert any(f["code"] == "dataset.coordinate_names_unresolved" and f["status"] == "unknown"
               for f in result["findings"])


@pytest.mark.parametrize("names", [["a", "b", "c"], {"motors": ["a", "b", "c"]}])
def test_full_coordinate_names_still_retained(tmp_path, names):
    _, ds = pair(tmp_path, names=names)
    snapshot = load_dataset(ds)
    assert snapshot.features["observation.state"]["names"] == ["a", "b", "c"]
    assert "coordinate_names_status" not in snapshot.features["observation.state"]


@pytest.mark.parametrize("names", [["a", "a"], {}, {"x": ["a"], "y": ["b"]}, [], 12, False])
def test_unsupported_ambiguous_names_still_refused(tmp_path, names):
    _, ds = pair(tmp_path, names=names)
    with pytest.raises(InputError):
        load_dataset(ds)


def test_repeated_per_coordinate_labels_are_explicitly_duplicate(tmp_path):
    _, ds = pair(tmp_path, names=["a", "a", "b"])
    state = load_dataset(ds).features["observation.state"]
    assert state["coordinate_names_status"] == "duplicate" and state["names"] is None
    assert state["raw_names"] == ["a", "a", "b"]


@pytest.mark.parametrize("names", [["state"], {"axes": None}])
def test_checkpoint_names_remain_strict(tmp_path, names):
    cp, _ = pair(tmp_path, names=None)
    config = json.loads((cp / "config.json").read_text())
    config["input_features"]["observation.state"]["names"] = names
    write(cp / "config.json", config)
    with pytest.raises(InputError):
        load_checkpoint(cp)


def test_unresolved_vector_labels_do_not_reinterpret_image_axes(tmp_path):
    _, ds = pair(tmp_path, names=["state"])
    info = json.loads((ds / "meta/info.json").read_text())
    info["features"]["observation.images.camera"] = {"shape": [3, 240, 320], "names": ["channels", "height", "width"]}
    write(ds / "meta/info.json", info)
    snapshot = load_dataset(ds)
    assert snapshot.features["observation.images.camera"]["names"] == ["channels", "height", "width"]
    assert "coordinate_names_status" not in snapshot.features["observation.images.camera"]


def test_non_vector_state_not_projected_as_coordinates(tmp_path):
    _, ds = pair(tmp_path, names=["state"])
    info = json.loads((ds / "meta/info.json").read_text())
    info["features"]["observation.state"]["shape"] = [2, 3]
    write(ds / "meta/info.json", info)
    with pytest.raises(InputError, match="names length"):
        load_dataset(ds)


def test_source_cannot_inject_derived_coordinate_annotations(tmp_path):
    cp, ds = pair(tmp_path, names=["a", "b", "c"])
    info = json.loads((ds / "meta/info.json").read_text())
    feature = info["features"]["observation.state"]
    feature["coordinate_names_status"] = "unresolved"
    feature["raw_names"] = ["fake"]
    write(ds / "meta/info.json", info)
    snapshot = load_dataset(ds)
    assert "coordinate_names_status" not in snapshot.features["observation.state"]
    assert "raw_names" not in snapshot.features["observation.state"]
    assert snapshot.config["features"]["observation.state"]["raw_names"] == ["fake"]
    assert not any(f["code"] == "dataset.coordinate_names_unresolved" for f in audit(str(cp), str(ds))["findings"])


def test_unresolved_state_order_stays_unknown_while_action_order_is_checked(tmp_path):
    cp, ds = pair(tmp_path, names=["state"])
    write(cp / "policy_preprocessor.json", {"steps": [{"registry_name": "relative_actions_processor",
         "config": {"enabled": True, "action_names": ["a", "b"]}}]})
    write(cp / "policy_postprocessor.json", {"steps": [{"registry_name": "absolute_actions_processor",
         "config": {"enabled": True}}]})
    result = audit(str(cp), str(ds))
    order = [f for f in result["findings"] if f["code"] == "action.processor_order_names"]
    assert any(f["status"] == "unknown" and "dataset:observation.state" in f["message"] for f in order)
    assert any(f["status"] == "pass" and "dataset:action" in f["message"] for f in order)
    assert not any(f["status"] == "error" for f in order)
