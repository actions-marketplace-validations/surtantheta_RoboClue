import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.diagnostics import audit


def run(case):
    return audit(str(case["checkpoint"]), str(case["dataset"]), str(case["deployment"]))


def change(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def test_healthy_control(case):
    report = run(case)
    assert report["status"] == "pass", report["findings"]
    assert report["limitations"]
    assert all(a["sha256"] for a in report["artifacts"])


@pytest.mark.parametrize("field,value,code", [
    ("units", ["deg", "normalized"], "semantics.units"),
    ("frame", "tool", "semantics.frame"),
    ("command_representation", "relative", "semantics.command_representation"),
    ("model_output_representation", "absolute", "action.model_representation"),
    ("gripper_convention", "1_closed_0_open", "semantics.gripper_convention"),
    ("execute_steps", 5, "timing.execution_exceeds_chunk"),
    ("normalization_mapping", {"ACTION": "IDENTITY"}, "normalization.runtime_override"),
])
def test_detects_specific_integration_fault(case, field, value, code):
    change(case["deployment"], lambda d: d.update({field: value}))
    findings = [f for f in run(case)["findings"] if f["code"] == code and f["status"] == "error"]
    assert findings and findings[0]["evidence"]


def test_joint_order_is_checked_not_only_shape(case):
    change(case["deployment"], lambda d: d["features"]["action"].update(names=["gripper", "shoulder"]))
    assert any(f["code"] == "joints.order" and f["status"] == "error" for f in run(case)["findings"])


def test_camera_rename_requires_explicit_mapping(case):
    change(case["deployment"], lambda d: d["features"].update({"observation.images.wrist": d["features"].pop("observation.images.front")}))
    assert any(f["code"] == "deployment.feature_missing" for f in run(case)["findings"])
    change(case["deployment"], lambda d: d.update(rename_map={"observation.images.wrist": "observation.images.front"}))
    assert run(case)["status"] == "pass"


def test_relative_pair_and_order(case):
    path = case["checkpoint"] / "policy_postprocessor.json"
    change(path, lambda d: d["steps"].reverse())
    assert any(f["code"] == "action.processor_order" for f in run(case)["findings"])
    change(path, lambda d: d["steps"].pop(0))
    assert any(f["code"] == "action.relative_pair" for f in run(case)["findings"])


def test_missing_semantics_remain_unknown(case):
    change(case["deployment"], lambda d: d.pop("training_contract"))
    report = run(case)
    assert report["status"] == "unknown"
    assert not report["counts"]["error"]


def test_shape_fault(case):
    change(case["deployment"], lambda d: d["features"]["action"].update(shape=[3], names=["a", "b", "c"]))
    change(case["deployment"], lambda d: d.update(units=["rad", "rad", "normalized"]))
    change(case["deployment"], lambda d: d["training_contract"].update(units=["rad", "rad", "normalized"]))
    change(case["deployment"], lambda d: d.update(limits=[[-20, 20], [-20, 20], [0, 1]]))
    assert any(f["code"] == "deployment.feature_shape" and f["status"] == "error" for f in run(case)["findings"])


def test_cadence_change_is_warning_not_claimed_failure(case):
    change(case["deployment"], lambda d: d.update(fps=20, execute_steps=2))
    report = run(case)
    assert report["status"] == "warning"
    assert report["counts"]["error"] == 0


def test_renames_cannot_shadow_features(case):
    change(case["deployment"], lambda d: d.update(rename_map={"observation.state": "action"}))
    with pytest.raises(InputError, match="collides"):
        run(case)


def test_metadata_only_keeps_runtime_unknown(case):
    report = audit(str(case["checkpoint"]))
    assert report["status"] == "unknown"
    assert any(f["code"] == "deployment.missing" for f in report["findings"])
