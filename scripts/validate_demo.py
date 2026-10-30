"""End-to-end controlled usefulness check. No real-user or hardware result is implied."""

import json
from pathlib import Path

from policy_doctor.compare import compare_configs
from policy_doctor.diagnostics import audit
from policy_doctor.evidence import inspect_evidence
from policy_doctor.repair import check_repair
from policy_doctor.trace import trace_actions


def main():
    root = Path(__file__).parents[1] / "examples/demo"
    cp, ds = str(root / "checkpoint"), str(root / "dataset")
    healthy = audit(cp, ds, str(root / "healthy.json"))
    broken = audit(cp, ds, str(root / "broken.json"))
    expected = {"joints.order", "action.processor_order_names", "semantics.units", "timing.execution_exceeds_chunk"}
    detected = {f["code"] for f in broken["findings"] if f["status"] == "error"}
    assert healthy["status"] == "pass", healthy["findings"]
    assert detected == expected, detected
    assert all(f["remedy"] or f["code"] == "timing.execution_exceeds_chunk"
               for f in broken["findings"] if f["status"] == "error")
    trace = trace_actions(cp, str(root / "sample.json"), str(root / "healthy.json"))
    assert trace["status"] == "pass"
    differences = compare_configs(str(root / "healthy.json"), str(root / "broken.json"))
    assert len(differences["changes"]) == 3
    source = next(f for f in broken["findings"] if f["code"] == "semantics.units")["evidence"][1]
    verified = inspect_evidence(source["path"], source["pointer"], source["sha256"])
    assert verified["status"] == "pass" and verified["value"] == ["deg", "normalized"]
    repaired = check_repair(cp, str(root / "broken.json"), str(root / "healthy.json"), ds)
    assert repaired["status"] == "pass" and sum(repaired["resolved_error_counts"].values()) == 4
    result = {"status": "pass", "synthetic_fault_types": ["joint_order", "units", "execution_horizon"],
              "expected_diagnostic_codes": sorted(expected), "detected_diagnostic_codes": sorted(detected),
              "healthy_false_errors": healthy["counts"]["error"], "healthy_check_status": healthy["status"],
              "saved_trace_status": trace["status"], "configuration_changes": len(differences["changes"]),
              "evidence_verified": verified["status"], "candidate_repair_status": repaired["status"],
              "resolved_error_counts": repaired["resolved_error_counts"],
              "scope": "Known synthetic integration faults and a healthy control; no independent user study, policy improvement or hardware outcome."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
