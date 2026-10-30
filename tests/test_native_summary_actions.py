"""Cold expert regressions over actual retained native captures, not mock statuses."""

import copy
import hashlib
import json

import pytest

from policy_doctor.big_bets_frontdoor import _json, main, summarize
from policy_doctor.native_training_demo import run_demo
from policy_doctor.server import _summary


@pytest.fixture
def reports(tmp_path):
    demo = run_demo(tmp_path / "native")
    return {phase: json.loads((tmp_path / "native" / (phase + "-report.json")).read_text())
            for phase in demo["checks"]}


def test_actual_fault_diagnostics_have_error_counts_without_mutating_raw_report(reports):
    report = reports["fault"]
    before = copy.deepcopy(report)
    view = summarize(report)
    assert view["status"] == "error" and view["counts"]["error"] == view["findings_total"] == 4
    assert view["counts_scope"] == "all_findings"
    assert view["unclassified_available_findings"] == 0
    assert view["differing_target_values"] == 1400
    assert view["full_report_sha256"] == hashlib.sha256(_json(report).encode()).hexdigest()
    assert report == before


@pytest.mark.parametrize("phase", ["healthy", "corrected"])
def test_preserved_targets_do_not_receive_repair_instructions(reports, phase):
    view = summarize(reports[phase])
    assert view["status"] == "pass" and view["first_difference"] is None
    assert view["counts"]["error"] == 0
    assert "match" in view["next"] and "unchanged" in view["next"]
    assert "restore" not in view["next"] and "correction" not in view["next"]


def test_unknown_target_identity_requests_evidence_without_prescribing_repair(reports):
    view = summarize(reports["unknown"])
    assert view["status"] == "unknown" and view["message"]
    assert "missing" in view["next"] and "unchanged" in view["next"]
    assert "restore" not in view["next"]


def test_unfamiliar_native_diagnostic_remains_explicitly_unclassified(reports):
    report = reports["fault"]
    report["findings"][0]["code"] = "future_guard_diagnostic"
    view = summarize(report)
    assert view["counts"]["error"] == 3
    assert view["unclassified_available_findings"] == 1
    assert view["counts_scope"] == "classified_available_findings"
    assert view["status"] == "error"


def test_source_truncated_findings_are_not_claimed_fully_counted(reports):
    report = reports["fault"]
    report["findings"] = report["findings"][:2]
    report["findings_truncated"] = True
    view = summarize(report)
    assert view["counts"]["error"] == 2 and view["findings_total"] == 4
    assert view["findings_omitted"] == 2 and view["counts_scope"] == "available_findings"


def test_preflight_scope_preview_marks_omission_and_keeps_complete_text_hash():
    scope = "A long explicitly bounded scope. " * 40
    report = {"kind": "dataset_preflight", "status": "unknown", "scope": scope, "findings": []}
    view = _summary(report)
    assert view["scope_truncated"] is True and len(view["scope"]) <= 512
    assert view["full_scope_sha256"] == hashlib.sha256(scope.encode()).hexdigest()


def test_health_card_version_flag_uses_existing_explicit_config_semantics(tmp_path, capsys, monkeypatch):
    seen = {}
    def inspect(source, **kwargs):
        seen.update(kwargs)
        return {"kind": "dataset_health_card", "status": "unknown"}
    monkeypatch.setattr("policy_doctor.big_bets_frontdoor.health_cards.health_card", inspect)
    assert main(["health-card", str(tmp_path), "--groot-version", "n1.7", "--json"]) == 3
    assert seen["config"]["groot_version"] == "n1.7"
    assert json.loads(capsys.readouterr().out)["status"] == "unknown"
