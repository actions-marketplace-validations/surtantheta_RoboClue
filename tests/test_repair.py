import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.diagnostics import audit
from policy_doctor.repair import check_repair


def broken(case, tmp_path):
    value = json.loads(case["deployment"].read_text())
    value["units"] = ["deg", "normalized"]
    value["features"]["action"]["names"].reverse()
    return case["put"](tmp_path / "broken.json", value)


def test_candidate_fixes_only_declared_contract_with_same_checkpoint(case, tmp_path):
    before = broken(case, tmp_path)
    result = check_repair(str(case["checkpoint"]), str(before), str(case["deployment"]), str(case["dataset"]))
    assert result["status"] == "pass"
    assert result["before_status"] == "error"
    assert result["resolved_error_counts"] == {"joints.order": 1, "action.processor_order_names": 1, "semantics.units": 1}
    assert not result["introduced_error_counts"]
    assert len(result["configuration_changes"]) == 2


def test_deleting_evidence_is_not_an_accepted_repair(case, tmp_path):
    before = broken(case, tmp_path)
    candidate = json.loads(case["deployment"].read_text())
    candidate.pop("training_contract")
    path = case["put"](tmp_path / "candidate.json", candidate)
    result = check_repair(str(case["checkpoint"]), str(before), str(path), str(case["dataset"]))
    assert result["status"] == "unknown" and result["candidate_unknown_counts"]


def test_new_faults_are_reported_after_other_faults_clear(case, tmp_path):
    before = broken(case, tmp_path)
    candidate = json.loads(case["deployment"].read_text())
    candidate["execute_steps"] = 20
    path = case["put"](tmp_path / "candidate.json", candidate)
    result = check_repair(str(case["checkpoint"]), str(before), str(path), str(case["dataset"]))
    assert result["status"] == "error"
    assert result["introduced_error_counts"] == {"timing.execution_exceeds_chunk": 1}


def test_concurrent_checkpoint_change_invalidates_comparison(case, monkeypatch, tmp_path):
    calls = []
    def changing_audit(*args):
        report = audit(*args)
        if calls:
            report["artifacts"][0]["sha256"] = "0" * 64
        calls.append(True)
        return report

    monkeypatch.setattr("policy_doctor.repair.audit", changing_audit)
    with pytest.raises(InputError, match="changed between"):
        check_repair(str(case["checkpoint"]), str(case["deployment"]), str(case["deployment"]))
