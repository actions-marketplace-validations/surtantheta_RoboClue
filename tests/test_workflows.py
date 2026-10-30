import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.bundle import export_bundle, reproduce_bundle
from policy_doctor.compare import compare_configs
from policy_doctor.template import deployment_template


def command(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    return subprocess.run([sys.executable, "-m", "policy_doctor", *map(str, args)], env=env,
                          text=True, capture_output=True, cwd=cwd, timeout=20)


def test_cli_exit_status_and_structured_output(case):
    common = [case["checkpoint"], "--dataset", case["dataset"], "--deployment", case["deployment"], "--json"]
    clean = command("audit", *common)
    assert clean.returncode == 0, clean.stderr
    assert json.loads(clean.stdout)["status"] == "pass"
    data = json.loads(case["deployment"].read_text())
    data["units"] = ["deg", "normalized"]
    case["put"](case["deployment"], data)
    bad = command("audit", *common)
    assert bad.returncode == 1
    assert json.loads(bad.stdout)["counts"]["error"] == 1
    incomplete = command("audit", case["checkpoint"], "--json")
    assert incomplete.returncode == 3
    invalid = command("inspect", "does-not-exist")
    assert invalid.returncode == 2
    assert json.loads(invalid.stderr)["error"]["code"] == "invalid_input"
    assert "Traceback" not in invalid.stderr


def test_cli_trace_and_no_overwrite(case, tmp_path):
    result = command("trace", case["checkpoint"], case["sample"], "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["commands"][1] == [13, .75]
    output = tmp_path / "report.json"
    assert command("inspect", case["checkpoint"], "--json", "--output", output).returncode == 0
    original = output.read_bytes()
    assert command("template", case["checkpoint"], "--output", output).returncode == 2
    assert output.read_bytes() == original


def test_cli_evidence_and_candidate_repair(case, tmp_path):
    from policy_doctor.artifacts import sha256

    evidence = command("evidence", case["deployment"], "--pointer", "/units", "--sha256", sha256(case["deployment"]), "--json")
    assert evidence.returncode == 0
    assert json.loads(evidence.stdout)["value"] == ["rad", "normalized"]
    broken = json.loads(case["deployment"].read_text())
    broken["units"] = ["deg", "normalized"]
    source = case["put"](tmp_path / "broken.json", broken)
    repaired = command("check-repair", case["checkpoint"], source, case["deployment"], "--dataset", case["dataset"], "--json")
    assert repaired.returncode == 0
    assert json.loads(repaired.stdout)["resolved_error_counts"] == {"semantics.units": 1}
    summary = command("check-repair", case["checkpoint"], source, case["deployment"], "--dataset", case["dataset"])
    assert "Before: error" in summary.stdout
    assert '"candidate_findings":' not in summary.stdout


def test_template_leaves_semantics_unknown(case):
    value = deployment_template(str(case["checkpoint"]))
    assert value["units"] is None and value["training_contract"]["units"] is None
    assert value["features"]["action"]["names"] is None


def test_diff_preserves_types_order_and_escaped_keys(case, tmp_path):
    a = case["put"](tmp_path / "a.json", {"x/y": {"a~b": 1}, "order": ["a", "b"], "type": 1})
    b = case["put"](tmp_path / "b.json", {"x/y": {"a~b": 2}, "order": ["b", "a"], "type": True})
    result = compare_configs(str(a), str(b))
    assert len(result["changes"]) == 3
    assert any(x["pointer"] == "/x~1y/a~0b" for x in result["changes"])


def test_bundle_portability_and_tamper_detection(case, tmp_path):
    bundle = tmp_path / "export"
    export_bundle(str(case["checkpoint"]), str(bundle), str(case["dataset"]), str(case["deployment"]))
    moved = tmp_path / "other-host" / "bundle"
    moved.parent.mkdir()
    shutil.move(str(bundle), moved)
    assert reproduce_bundle(str(moved))["status"] == "pass"
    assert command("reproduce", moved, "--json", cwd=tmp_path).returncode == 0
    config = moved / "checkpoint/config.json"
    config.write_text(config.read_text() + " ")
    with pytest.raises(InputError, match="hash mismatch"):
        reproduce_bundle(str(moved))


def test_bundle_will_not_overwrite(case, tmp_path):
    with pytest.raises(InputError, match="exists"):
        export_bundle(str(case["checkpoint"]), str(tmp_path))


def test_bundle_manifest_traversal_rejected(case, tmp_path):
    bundle = tmp_path / "bundle"
    export_bundle(str(case["checkpoint"]), str(bundle))
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"][0]["path"] = "../../secret"
    case["put"](bundle / "manifest.json", manifest)
    with pytest.raises(InputError, match="within"):
        reproduce_bundle(str(bundle))


@pytest.mark.parametrize("version", [None, "0.0.1"])
def test_bundle_engine_version_is_part_of_reproduction_contract(case, tmp_path, version):
    bundle = tmp_path / "bundle"
    export_bundle(str(case["checkpoint"]), str(bundle))
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["engine_version"] = version
    case["put"](bundle / "manifest.json", manifest)
    result = reproduce_bundle(str(bundle))
    assert result["status"] == "unknown" and not result["engine_version_matches"]
