"""Consumer-level automation cases, not tests of a mocked exit-code table."""
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from policy_doctor import automation
from policy_doctor.artifacts import InputError
from policy_doctor.product_demo import run_demo

ROOT = Path(__file__).resolve().parents[1]


def _write(path, data):
    path.write_text(json.dumps(data))


def _manifest(root, checks, name="checks.json"):
    _write(root / name, {"schema_version": 1, "checks": checks})
    return name


@pytest.fixture
def config_case(tmp_path):
    _write(tmp_path / "reviewed.json", {"camera_order": ["front", "wrist"], "action": "relative"})
    _write(tmp_path / "candidate.json", {"camera_order": ["wrist", "front"], "action": "relative"})
    check = {"id": "camera-order", "check": "config", "artifact": "candidate.json", "reference": "reviewed.json"}
    return tmp_path, check


def test_config_fault_then_correction_use_same_reviewed_reference(config_case):
    root, check = config_case
    name = _manifest(root, [check])
    reference = (root / "reviewed.json").read_bytes()
    before = automation.run_manifest(name, workspace=root, output=root / "before")
    assert before["exit_code"] == 1 and before["gate"] == "blocked"
    evidence = json.loads((root / "before/camera-order.json").read_text())
    assert evidence["report"]["equal"] is False and evidence["report"]["changes"]
    (root / "candidate.json").write_bytes(reference)
    after = automation.run_manifest(name, workspace=root, output=root / "after")
    assert after["exit_code"] == 0 and after["gate"] == "clear"
    assert (root / "reviewed.json").read_bytes() == reference
    assert before["manifest_sha256"] == after["manifest_sha256"]


def test_saved_targets_fault_correction_and_missing_coverage_keep_original_pin(tmp_path):
    demo = tmp_path / "demo"
    run_demo(demo)
    retained = {name: (demo / name).read_bytes() for name in ("source.json", "reference-contract.json", "criterion.sha256", "broken.json")}
    codes = []
    for phase in ("broken", "corrected", "unknown"):
        check = {"id": "targets", "check": "training", "artifact": "demo/" + phase + ".json",
                 "reference": "demo/reference-contract.json", "pin_file": "demo/criterion.sha256"}
        name = _manifest(tmp_path, [check])
        result = automation.run_manifest(name, workspace=tmp_path)
        codes.append(result["exit_code"])
        assert result["checks"][0]["gate"] == ("clear" if phase == "corrected" else "blocked")
    assert codes == [1, 0, 3]
    assert {n: (demo / n).read_bytes() for n in retained} == retained


def test_required_deployment_units_never_clears_missing_robot(config_case):
    root, _ = config_case
    name = _manifest(root, [{"id": "deploy", "check": "deployment-units", "artifact": "candidate.json"}])
    assert automation.main(["run", "--manifest", name, "--workspace", str(root)]) == 2


@pytest.mark.parametrize("change", [
    {"shell": "touch injected"}, {"args": ["--fail-on", "error"]}, {"check": "unknown"},
    {"artifact": "../candidate.json"}, {"artifact": "/etc/passwd"}, {"id": "result"},
    {"max_files": True}, {"max_bytes": 64 * 1024**2 + 1}, {"max_rows": 0},
    {"id": "::error::injected"}, {"id": "x\ninjected"}, {"reference": None},
])
def test_unrecognized_or_unsafe_manifest_refused_before_execution(config_case, monkeypatch, change):
    root, check = config_case
    check.update(change)
    name = _manifest(root, [check])
    from policy_doctor import product_cli

    monkeypatch.setattr(product_cli, "main", lambda *_: pytest.fail("invalid manifest reached a check"))
    assert automation.main(["run", "--manifest", name, "--workspace", str(root)]) == 2


@pytest.mark.parametrize("kind", ["link", "fifo"])
def test_special_manifest_and_inputs_do_not_hang_or_follow(config_case, kind):
    root, check = config_case
    unsafe = root / "unsafe.json"
    if kind == "link":
        unsafe.symlink_to(root / "candidate.json")
    else:
        os.mkfifo(unsafe)
    name = _manifest(root, [dict(check, artifact="unsafe.json")])
    assert automation.main(["run", "--manifest", name, "--workspace", str(root)]) == 2
    assert automation.main(["run", "--manifest", "unsafe.json", "--workspace", str(root)]) == 2


def test_symlinked_output_parent_and_existing_output_refused(config_case):
    root, check = config_case
    name = _manifest(root, [check])
    (root / "kept").mkdir()
    (root / "alias").symlink_to(root / "kept", target_is_directory=True)
    for output in (root / "kept", root / "alias/new", root / "missing/new"):
        with pytest.raises(InputError):
            automation.run_manifest(name, workspace=root, output=output)
    assert list((root / "kept").iterdir()) == []


def test_receipts_cannot_be_written_into_readonly_dataset_input(tmp_path):
    (tmp_path / "dataset/meta").mkdir(parents=True)
    _write(tmp_path / "dataset/meta/info.json", {"codebase_version": "v2.1", "features": {}})
    name = _manifest(tmp_path, [{"id": "data", "check": "dataset", "artifact": "dataset"}])
    with pytest.raises(InputError, match="inside an input"):
        automation.run_manifest(name, workspace=tmp_path, output=tmp_path / "dataset/receipts")
    assert not (tmp_path / "dataset/receipts").exists()


def test_whole_manifest_validated_before_first_check(config_case, monkeypatch):
    root, check = config_case
    bad = dict(check, id="second", shell="do something")
    name = _manifest(root, [check, bad])
    from policy_doctor import product_cli

    monkeypatch.setattr(product_cli, "main", lambda *_: pytest.fail("a later invalid check was not validated"))
    assert automation.main(["run", "--manifest", name, "--workspace", str(root)]) == 2


@pytest.mark.parametrize("check", [
    {"id": "data", "check": "dataset", "artifact": "hf://example/data", "fetch": True, "revision": "main"},
    {"id": "data", "check": "dataset", "artifact": "hf://example/data", "revision": "a" * 40},
    {"id": "data", "check": "dataset", "artifact": "hf://example/data", "fetch": True, "revision": "a" * 40},
])
def test_hub_requires_both_permissions_and_immutable_revision(tmp_path, monkeypatch, check):
    name = _manifest(tmp_path, [check])
    from policy_doctor import product_cli

    monkeypatch.setattr(product_cli, "main", lambda *_: pytest.fail("unpermitted Hub input reached network intake"))
    assert automation.main(["run", "--manifest", name, "--workspace", str(tmp_path)]) == 2
    if check.get("revision") == "main" or not check.get("fetch"):
        assert automation.main(["run", "--manifest", name, "--workspace", str(tmp_path), "--allow-hub"]) == 2


def test_dataset_cannot_use_error_only_to_clear_incomplete_coverage(tmp_path):
    (tmp_path / "dataset").mkdir()
    check = {"id": "data", "check": "dataset", "artifact": "dataset", "fail_on": "error"}
    name = _manifest(tmp_path, [check])
    with pytest.raises(InputError, match="fail_on=unknown"):
        automation.run_manifest(name, workspace=tmp_path)


def test_precommit_selects_candidate_reference_manifest_and_not_unrelated_json(config_case):
    root, check = config_case
    name = _manifest(root, [check])
    for selected in ("candidate.json", "reviewed.json", name):
        result = automation.run_manifest(name, workspace=root, filenames=[selected])
        assert result["exit_code"] == 1 and len(result["checks"]) == 1
    result = automation.run_manifest(name, workspace=root, filenames=["unrelated.json"])
    assert result["gate"] == "not_selected" and result["checks"] == []


def test_precommit_refuses_dataset_or_big_json(config_case):
    root, check = config_case
    name = _manifest(root, [dict(check, check="dataset")])
    with pytest.raises(InputError):
        automation.run_manifest(name, workspace=root, filenames=[])
    name = _manifest(root, [check])
    (root / "candidate.json").write_text('{"large":"' + "x" * automation.PRECOMMIT_FILE_LIMIT + '"}')
    result = automation.run_manifest(name, workspace=root, filenames=[])
    assert result["exit_code"] == 2
    assert "byte limit" in result["checks"][0]["error"]["message"]


def test_shell_punctuation_is_literal_in_config_filename(config_case):
    root, check = config_case
    name = "candidate$(touch PWNED);.json"
    (root / "candidate.json").rename(root / name)
    manifest = _manifest(root, [dict(check, artifact=name)])
    assert automation.run_manifest(manifest, workspace=root)["exit_code"] == 1
    assert not (root / "PWNED").exists()


def test_action_and_hook_use_checked_source_and_delimited_arguments():
    import yaml

    action = yaml.safe_load((ROOT / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    steps = action["runs"]["steps"]
    for step in steps:
        if "run" in step:
            assert subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True).returncode == 0
            assert "${{ inputs." not in step["run"] and "eval " not in step["run"]
    assert '"${GITHUB_ACTION_PATH}[dataset]"' in steps[1]["run"]
    assert '"${arguments[@]}"' in steps[2]["run"]
    hooks = yaml.safe_load((ROOT / ".pre-commit-hooks.yaml").read_text())
    assert hooks[0]["entry"].startswith("python -I -m roboclue.automation ")
    assert hooks[0]["entry"].endswith(" --") and hooks[0]["require_serial"] is True


def test_empty_duplicate_and_oversized_manifests_rejected(config_case):
    root, check = config_case
    for document in ({"schema_version": True, "checks": [check]}, {"schema_version": 1, "checks": []},
                     {"schema_version": 1, "checks": [check, copy.deepcopy(check)]}):
        _write(root / "checks.json", document)
        assert automation.main(["run", "--manifest", "checks.json", "--workspace", str(root)]) == 2
    (root / "checks.json").write_text('{"x":"' + "x" * automation.MANIFEST_LIMIT + '"}')
    assert automation.main(["run", "--manifest", "checks.json", "--workspace", str(root)]) == 2


def test_manifest_pin_is_exact_original_bytes(config_case):
    root, check = config_case
    name = _manifest(root, [check])
    result = automation.run_manifest(name, workspace=root)
    assert result["manifest_sha256"] == hashlib.sha256((root / name).read_bytes()).hexdigest()


def test_precommit_explains_the_changed_field_without_requiring_an_artifact_browser(config_case, capsys):
    root, check = config_case
    name = _manifest(root, [check])
    assert automation.main(["pre-commit", "--manifest", name, "--workspace", str(root), "--", "candidate.json"]) == 1
    output = capsys.readouterr().out
    assert "Problem:" in output and "camera_order" in output and "Next:" in output
    assert "{'" not in output
