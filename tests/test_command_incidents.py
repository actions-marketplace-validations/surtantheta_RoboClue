"""A processor mismatch becomes the same portable check on later native values."""

# ruff: noqa: F811
import hashlib
import json
import shutil

import pytest
from test_command_path import arrays_update, change, contract, packet  # noqa: F401

from policy_doctor.artifacts import InputError
from policy_doctor.capture import audit_capture
from policy_doctor.incidents import check_incident_candidate, export_incident_case, reproduce_incident_case
from policy_doctor.runtime_diff import compare_runtime_captures


def tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_relocated_command_contract_reuses_changed_values_and_preserves_unknown(packet, tmp_path):
    bad = packet("bad", mean=(13.0, -20.0))
    good = packet("good")
    later = packet("later", selected=(1.0, 2.0))
    recurrence = packet("recurrence", selected=(1.0, 2.0), mean=(13.0, -20.0))
    unknown = packet("unknown")
    change(unknown, lambda d: d["coverage"].update(complete_target_call_history=False))
    originals = {str(p): tree(p) for p in (bad, good, later, recurrence, unknown)}
    case = tmp_path / "case"
    result = export_incident_case(
        "act_command_path",
        bad,
        case,
        {},
        corrected=good,
        controls=[{"name": "incomplete", "source": unknown, "parameters": {}, "expected_status": "unknown"}],
    )
    assert (
        result["status"] == "pass" and result["checks"]["before"]["acceptance_contract"]["status"] == "error"
    )
    assert result["checks"]["after"]["acceptance_contract"]["status"] == "pass"
    manifest = json.loads((case / "manifest.json").read_text())
    assert manifest["parameters"] == dict(contract(), atol=1e-6, rtol=1e-5)
    assert manifest["controls"][0]["parameters"] == manifest["parameters"]
    assert "command_rules.py" in manifest["verifier"]["module_sha256"]
    for p in (bad, good, later, recurrence, unknown):
        assert tree(p) == originals[str(p)]
    moved = tmp_path / "colleague"
    shutil.move(case, moved)
    for p in (bad, good, unknown):
        shutil.rmtree(p)
    assert reproduce_incident_case(moved)["status"] == "pass"
    preserved = tree(moved)
    for p, status in ((later, "pass"), (recurrence, "error")):
        result = check_incident_candidate(moved, p)
        assert result["status"] == status and result["candidate_evaluated"]
        assert result["parameters"] == manifest["parameters"]
    assert tree(moved) == preserved


def test_weakened_control_and_numeric_only_fault_cannot_be_called_processor_repair(packet, tmp_path):
    bad = packet("bad", mean=(13.0, -20.0))
    good = packet("good")
    with pytest.raises(InputError, match="same processor"):
        export_incident_case(
            "act_command_path",
            bad,
            tmp_path / "weak",
            {},
            corrected=good,
            controls=[
                {
                    "name": "weakened",
                    "source": bad,
                    "parameters": {
                        "expected_processor": {"mode": "MEAN_STD", "mean": [13.0, -20.0], "std": [2.0, 4.0]}
                    },
                    "expected_status": "pass",
                }
            ],
        )
    change(bad, lambda d: d["ticks"][0]["postprocessor"]["steps"][0]["transform"].update(output="s"))
    with pytest.raises(InputError, match="contract violation"):
        export_incident_case("act_command_path", bad, tmp_path / "numeric", {}, corrected=good)
    assert not (tmp_path / "weak").exists() and not (tmp_path / "numeric").exists()


def test_existing_capture_and_comparison_routes_consume_same_supported_chain(packet):
    good = packet("good")
    bad = packet("bad", mean=(13.0, -20.0))
    assert audit_capture(str(good), command_contract=contract())["status"] == "pass"
    assert audit_capture(str(bad), command_contract=contract())["status"] == "error"
    assert audit_capture(str(good))["kind"] == "command_path_audit"
    assert compare_runtime_captures(str(good), str(bad))["first_difference"]["stage"] == "postprocessor"
    with pytest.raises(InputError, match="only expected_processor"):
        audit_capture(str(good), command_contract={"arbitrary_code": "echo foo"})
    with pytest.raises(InputError, match="dataset or layout"):
        audit_capture(str(good), expected_names=["fake"])
    with pytest.raises(InputError, match="max_query_gap requires"):
        audit_capture(str(good), max_query_gap=1)


def test_command_capture_root_symlink_is_not_hidden_by_common_router(packet, tmp_path):
    good = packet("good")
    alias = tmp_path / "alias"
    alias.symlink_to(good, target_is_directory=True)
    with pytest.raises(InputError, match="symlink"):
        audit_capture(str(alias), command_contract=contract())


def test_changed_verifier_prevents_later_candidate_read(packet, tmp_path):
    bad = packet("bad", mean=(13.0, -20.0))
    good = packet("good")
    case = tmp_path / "case"
    export_incident_case("act_command_path", bad, case, {}, corrected=good)
    data = json.loads((case / "manifest.json").read_text())
    data["verifier"]["engine_version"] = "another-version"
    raw = json.dumps(data).encode()
    (case / "manifest.json").write_bytes(raw)
    (case / "READY.json").write_text(
        json.dumps({"schema_version": 1, "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    )
    result = check_incident_candidate(case, tmp_path / "nonexistent")
    assert result["status"] == "unknown" and not result["candidate_evaluated"]
