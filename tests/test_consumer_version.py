"""Declared consumer labels cannot silently inherit an unqualified default."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from policy_doctor import consumer_version as consumer
from policy_doctor.artifacts import InputError


def _report(state="error", explicit=False):
    code = "units.joint_mode_mismatch" if state == "error" else "units.joint_mode_match"
    return {"schema_version": 1, "kind": "units_manifest", "status": state,
        "counts": {"error": int(state == "error"), "pass": int(state == "pass"), "warning": 0, "unknown": int(state == "unknown")},
        "manifest": {"features": {"observation.state": {"coordinates": [{"motor": "shoulder_pan", "unit": "RANGE_M100_100"}]}},
            "deployment": {"robot_type": "so101_follower", "program": "record", "unknown": None,
                "arms": {"": {"use_degrees": True, "mode": "DEGREES", "pointer": "/robot/use_degrees",
                    "origin": "set explicitly" if explicit else "the default of LeRobot146d847",
                    "basis": "old conditional basis"}}}},
        "findings": [{"code": code, "status": state, "message": "old conditional basis", "evidence": []}],
        "coverage": {"rows": 400}}


def test_omitted_label_preserves_legacy_report_and_marks_it_conditional():
    original = _report()
    saved = copy.deepcopy(original)
    result = consumer.apply_consumer_version(original)
    context = result.pop("consumer_version")
    assert result == saved and original == saved
    assert context["status"] == "conditional_pinned_profile" and context["fleet_verified"] is False
    assert context["declared_version"] is None


def test_known_label_selects_recorded_source_profile_not_actual_runtime_attestation():
    original = _report()
    saved = copy.deepcopy(original)
    result = consumer.apply_consumer_version(original, "0.6.2")
    context = result["consumer_version"]
    assert result["status"] == "error" and context["comparison_allowed"] is True
    assert context["tested_profile"] == "lerobot-0.6.2-so-146d847" and context["fleet_verified"] is False
    assert context["source_basis"]["source_revision"] == consumer.SOURCE_REVISION
    assert context["command_role"] == "collection_only" and "lerobot-rollout" in context["command_role_note"]
    assert "caller" in context["reason"].lower() and "does not attest" in context["reason"]
    assert "0.6.2" in result["manifest"]["deployment"]["arms"][""]["basis"]
    assert original == saved


@pytest.mark.parametrize("label", ["0.6.3", "0.6.20", "0.6.2.dev0", "v0.6.2", "latest", " 0.6.2"])
@pytest.mark.parametrize("state", ["pass", "error"])
def test_unqualified_labels_never_keep_an_active_legacy_pass_or_mismatch(label, state):
    original = _report(state)
    saved = copy.deepcopy(original)
    result = consumer.apply_consumer_version(original, label)
    assert result["status"] == "unknown" and result["consumer_version"]["comparison_allowed"] is False
    assert result["consumer_version"]["declared_version"] == label
    assert all(f["code"] not in {"units.joint_mode_match", "units.joint_mode_mismatch"} for f in result["findings"])
    assert result["findings"][-1]["code"] == "units.consumer_version_unqualified"
    assert result["conditional_comparison"]["findings"][0]["status"] == state
    assert result["manifest"]["features"] == saved["manifest"]["features"]
    arm = result["manifest"]["deployment"]["arms"][""]
    assert arm["mode"] is None and arm["use_degrees"] is None
    assert result["manifest"]["deployment"]["unknown"] is None  # Unqualified is not invalid input.
    assert original == saved


def test_explicit_boolean_does_not_qualify_semantics_for_an_unknown_version():
    original = _report(explicit=True)
    result = consumer.apply_consumer_version(original, "0.9.0")
    arm = result["manifest"]["deployment"]["arms"][""]
    assert arm["use_degrees"] is True and arm["mode"] is None
    assert result["status"] == "unknown" and result["counts"]["error"] == 0


def test_qualified_version_does_not_resolve_a_missing_actual_setting():
    original = _report("unknown")
    arm = original["manifest"]["deployment"]["arms"][""]
    arm.update(use_degrees=None, mode=None, basis="config_path was not read")
    result = consumer.apply_consumer_version(original, "0.6.2")
    assert result["status"] == "unknown" and result["manifest"]["deployment"]["arms"][""]["use_degrees"] is None


def test_independent_dataset_warning_or_error_is_not_hidden_by_an_unknown_consumer():
    original = _report()
    original["findings"] += [{"code": "units.timebase_contradicts_fps", "status": "warning", "message": "recorded timing warning"}]
    result = consumer.apply_consumer_version(original, "0.9.0")
    assert result["status"] == "unknown" and result["counts"]["warning"] == 1
    original["findings"].append({"code": "units.recorded_data_error", "status": "error", "message": "independent data contradiction"})
    result = consumer.apply_consumer_version(original, "0.9.0")
    assert result["status"] == "error" and result["counts"]["error"] == 1
    assert any(f["code"] == "units.recorded_data_error" for f in result["findings"])


def test_pre777_consumer_migration_error_is_not_asserted_for_an_unqualified_version():
    original = _report()
    original["findings"] = [{"code": "units.pre_pr777_calibration", "status": "error", "message": "legacy consumer mismatch"}]
    result = consumer.apply_consumer_version(original, "unknown-build")
    assert result["status"] == "unknown" and result["counts"]["error"] == 0


def test_known_profile_does_not_claim_every_robot_model():
    original = _report()
    original["manifest"]["deployment"]["robot_type"] = "custom_robot"
    result = consumer.apply_consumer_version(original, "0.6.2")
    assert result["status"] == "unknown" and result["consumer_version"]["tested_profile"] is None


def test_rollout_role_is_parsed_context_not_execution_proof():
    deployment = _report()["manifest"]["deployment"]
    deployment["program"] = "rollout"
    context = consumer.consumer_version_context("0.6.2", deployment=deployment)
    assert context["command_role"] == "policy_rollout" and context["fleet_verified"] is False
    assert "does not establish" in context["command_role_note"]


@pytest.mark.parametrize("label", [True, 0.6, "", "x" * 65, "0.6.2\nextra"])
def test_invalid_label_is_an_input_error(label):
    with pytest.raises(InputError):
        consumer.apply_consumer_version(_report(), label)


def test_a_supplied_source_is_read_only_and_never_imported(tmp_path, monkeypatch):
    # Generated source-reader control, not an additional qualified native profile.
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "PWNED"
    raw = ("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n").encode()
    (source / "critical.py").write_bytes(raw)
    monkeypatch.setattr(consumer, "CRITICAL_FILES", {"critical.py": hashlib.sha256(raw).hexdigest()})
    context = consumer.consumer_version_context("unqualified-label", deployment=_report()["manifest"]["deployment"], source_root=source)
    assert context["comparison_allowed"] is True and context["source_basis"]["matches_tested_profile"] is True
    assert context["fleet_verified"] is False and not marker.exists()
    (source / "critical.py").write_text("changed")
    context = consumer.consumer_version_context("0.6.2", deployment=_report()["manifest"]["deployment"], source_root=source)
    assert context["comparison_allowed"] is False  # Actual supplied mismatch overrides a matching label.


@pytest.mark.parametrize("special", ["symlink", "fifo"])
def test_source_verification_refuses_links_and_special_files(tmp_path, monkeypatch, special):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "critical.py"
    monkeypatch.setattr(consumer, "CRITICAL_FILES", {"critical.py": "a" * 64})
    if special == "symlink":
        other = tmp_path / "other.py"
        other.write_text("unused")
        target.symlink_to(other)
    else:
        os.mkfifo(target)
    with pytest.raises(InputError):
        consumer.verify_consumer_source(source)


def test_incomplete_source_is_unqualified_and_bounded_source_reads_fail_closed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(consumer, "CRITICAL_FILES", {"critical.py": "a" * 64})
    assert consumer.verify_consumer_source(source)["matches_tested_profile"] is False
    (source / "critical.py").write_bytes(b"x" * 20)
    monkeypatch.setattr(consumer, "MAX_SOURCE_BYTES", 8)
    with pytest.raises(InputError, match="byte limit"):
        consumer.verify_consumer_source(source)


def test_production_context_does_not_import_lerobot_or_torch():
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('lerobot', 'torch'):
        raise AssertionError('framework import attempted: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from policy_doctor.consumer_version import consumer_version_context
result = consumer_version_context('0.6.2', deployment={'robot_type': 'so101_follower', 'program': 'record'})
assert result['comparison_allowed'] and not result['fleet_verified']
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-B", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_fleet_version_is_a_typed_forwarded_automation_field(tmp_path):
    from policy_doctor import automation

    (tmp_path / "dataset").mkdir()
    check = {"id": "fleet", "check": "deployment-units", "artifact": "dataset", "robot_command": "lerobot-rollout --robot.type=so101_follower", "lerobot_version": "0.6.2"}
    (tmp_path / "checks.json").write_text(json.dumps({"schema_version": 1, "checks": [check]}))
    loaded = automation.load_manifest("checks.json", workspace=tmp_path)["checks"][0]
    argv = automation.command_for(loaded)
    assert argv[argv.index("--lerobot-version") + 1] == "0.6.2"
    check["lerobot_version"] = False
    (tmp_path / "checks.json").write_text(json.dumps({"schema_version": 1, "checks": [check]}))
    with pytest.raises(InputError):
        automation.load_manifest("checks.json", workspace=tmp_path)
