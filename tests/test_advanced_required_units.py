"""Canonical advanced preflight cannot waive declared required unit evidence."""
import json

import pytest
from test_units_manifest import so_dataset

from policy_doctor import cli

COMMAND = 'lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0'
CANONICAL = {"prog": "roboclue advanced", "brand": "RoboClue", "command_prefix": ("roboclue", "advanced")}


def _call(root, *options, canonical=True):
    return cli.main(["preflight", str(root), "--json", "--no-video-headers", "--no-groot-contract", *options],
                    **(CANONICAL if canonical else {}))


@pytest.mark.parametrize("suffix", ["--robot.type=so101_folower", "--robot.use_degres=false", "--robot.usedegrees=false", "--robot.use_degrees=flase"])
def test_canonical_invalid_declaration_is_not_a_green_error_only_gate(tmp_path, suffix):
    root = so_dataset(tmp_path, "deg")
    command = COMMAND if suffix.startswith("--robot.use") else "lerobot-record --robot.port=/dev/ttyACM0"
    assert _call(root, "--robot-command", command + " " + suffix, "--fail-on", "error") == 2


def test_canonical_missing_metadata_with_declared_robot_is_invalid(tmp_path):
    assert _call(tmp_path, "--robot-command", COMMAND, "--fail-on", "error") == 2


@pytest.mark.parametrize("canonical", [True, False])
def test_explicit_required_units_needs_a_command_on_both_launchers(tmp_path, canonical):
    root = so_dataset(tmp_path, "deg")
    assert _call(root, "--require", "deployment-units", "--fail-on", "error", canonical=canonical) == 2


@pytest.mark.parametrize("canonical", [True, False])
def test_required_unknown_units_stays_unknown_with_error_threshold(tmp_path, capsys, canonical):
    root = so_dataset(tmp_path, "deg", data=False)
    assert _call(root, "--robot-command", COMMAND, "--require", "deployment-units", "--fail-on", "error", canonical=canonical) == 3
    report = json.loads(capsys.readouterr().out)
    required = report["coverage"]["required_assessments"]["deployment-units"]
    assert required["status"] == "unknown" and "reason" in required


def test_required_fault_then_corrected_command_keeps_full_report_and_names_scope(tmp_path, capsys):
    root = so_dataset(tmp_path, "m100")
    assert _call(root, "--robot-command", COMMAND, "--lerobot-version", "0.6.2", "--require", "deployment-units", "--fail-on", "error") == 1
    fault = json.loads(capsys.readouterr().out)
    assert fault["coverage"]["required_assessments"]["deployment-units"]["status"] == "error"
    assert _call(root, "--robot-command", COMMAND + " --robot.use_degrees=false", "--lerobot-version", "0.6.2", "--require", "deployment-units", "--fail-on", "error") == 0
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["components"]["units"]["status"] == "pass"
    assert corrected["coverage"]["required_assessments"]["deployment-units"]["status"] == "pass"
    assert corrected["status"] == "unknown"  # Unrelated missing preflight evidence stays truthful.


def test_canonical_selector_requires_coverage_without_an_extra_require_flag(tmp_path):
    root = so_dataset(tmp_path, "deg", data=False)
    assert _call(root, "--robot-command", COMMAND, "--fail-on", "error") == 3


def test_generic_legacy_error_filter_remains_a_filter_without_required_coverage(tmp_path):
    root = so_dataset(tmp_path, "deg", data=False)
    assert _call(root, "--robot-command", COMMAND, "--fail-on", "error", canonical=False) == 0


@pytest.mark.parametrize("canonical", [True, False])
def test_required_comparison_cannot_be_skipped(tmp_path, canonical):
    root = so_dataset(tmp_path, "deg")
    assert _call(root, "--robot-command", COMMAND, "--require", "deployment-units", "--no-units", canonical=canonical) == 2
