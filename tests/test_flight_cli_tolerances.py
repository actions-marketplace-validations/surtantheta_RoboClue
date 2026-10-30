"""Actual flight CLI tolerance forms over generated synchronous software controls."""

import json
import os
import subprocess
import sys

import pytest

from policy_doctor import big_bets_frontdoor as fd
from policy_doctor.artifacts import InputError
from policy_doctor.flight_recorder import triage_flight_recording
from policy_doctor.runtime_recorder import RuntimeRecorder, SignalSpec

KEY = "generated_axis.pos"
UNIT = "synthetic_position_unit"
TOLERANCE = {KEY: {"unit": UNIT, "absolute": 0.02}}


def cli(recording, argument, *, timeout=10):
    return subprocess.run(
        [sys.executable, "-m", "policy_doctor.big_bets_frontdoor", "flight-triage",
         str(recording), "--tracking-tolerance", argument, "--json"],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def record(path, *, fault=False):
    class GeneratedInterface:
        step = 0

        def send_action(self, action):
            offset = 0.5 if fault and self.step == 1 else 0.0
            self.step += 1
            return {KEY: action[KEY] + offset}

        def get_observation(self):
            return {KEY: 0.11 + self.step * 0.1}

    spec = SignalSpec((KEY,), (UNIT,), "joint_position")
    pairs = []
    with RuntimeRecorder(
        GeneratedInterface(), path, action_spec=spec, returned_spec=spec,
        observation_spec=spec, policy_spec=spec,
        provenance={"kind": "synthetic_software_control"},
    ) as runtime:
        for value in (0.1, 0.2, 0.3):
            runtime.get_observation()
            action = {KEY: value}
            policy_id = runtime.record_policy_command(action)
            runtime.send_action(action, policy_command_id=policy_id)
            pairs.append([runtime.last_send.request_id, runtime.last_send.returned_feedback_id])
    return pairs


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "healthy.jsonl"
    record(path)
    return path


def test_actual_cli_inline_file_and_python_agree_for_three_controls(tmp_path):
    tolerance_file = tmp_path / "tolerances.json"
    tolerance_file.write_text(json.dumps(TOLERANCE))
    for label in ("healthy", "fault", "corrected"):
        path = tmp_path / (label + ".jsonl")
        pairs = record(path, fault=label == "fault")
        reference = triage_flight_recording(path, tracking_tolerance=TOLERANCE)
        inline = cli(path, " \n" + json.dumps(TOLERANCE))
        file = cli(path, str(tolerance_file))
        assert json.loads(inline.stdout) == json.loads(file.stdout) == reference
        assert inline.stderr == file.stderr == ""
        expected_exit = 0 if reference["status"] == "warning" else 3
        assert inline.returncode == file.returncode == expected_exit
        comparisons = [f for f in reference["findings"] if f["code"] == "tracking"]
        warnings = [f for f in comparisons if f["status"] == "warning"]
        if label == "fault":
            assert len(warnings) == 1
            assert warnings[0]["event_ids"] == pairs[1]
            assert warnings[0]["details"]["comparison_boundary"] == "request_to_return"
            assert warnings[0]["details"]["absolute_residual"] == pytest.approx(0.5)
        else:
            assert not warnings
        assert any(f["code"] == "feedback_association" and f["status"] == "unknown"
                   for f in reference["findings"])
        assert reference["criteria"]["tracking_tolerance"] == TOLERANCE


@pytest.mark.parametrize("value", [
    '{"axis":',
    '{"axis":{"unit":"u","absolute":0},"axis":{"unit":"u","absolute":1}}',
    '{"axis":{"unit":"u","unit":"v","absolute":0}}',
    '{"axis":{"unit":"u","absolute":NaN}}',
    '{"axis":{"unit":"u","absolute":Infinity}}',
    '{"axis":{"unit":"u","absolute":1e999}}',
    "{'axis': {'unit': 'u', 'absolute': 0}}",
])
@pytest.mark.parametrize("form", ["inline", "file"])
def test_strict_json_refusals(recording, tmp_path, value, form):
    argument = value
    if form == "file":
        path = tmp_path / "invalid.json"
        path.write_text(value)
        argument = str(path)
    result = cli(recording, argument)
    assert result.returncode == 2 and result.stdout == ""
    message = json.loads(result.stderr)["error"]["message"]
    assert message.startswith("--tracking-tolerance expects a JSON file path or inline JSON object:")


@pytest.mark.parametrize("absolute", [True, False, -0.01, "0.02", None])
@pytest.mark.parametrize("form", ["inline", "file"])
def test_invalid_specs_remain_refused(recording, tmp_path, absolute, form):
    value = json.dumps({KEY: {"unit": UNIT, "absolute": absolute}})
    argument = value
    if form == "file":
        path = tmp_path / "invalid-spec.json"
        path.write_text(value)
        argument = str(path)
    result = cli(recording, argument)
    assert result.returncode == 2 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "invalid_input"


@pytest.mark.parametrize("value", ["[]", "true", "null", '"text"', "0"])
def test_file_requires_object(recording, tmp_path, value):
    path = tmp_path / "nonobject.json"
    path.write_text(value)
    result = cli(recording, str(path))
    assert result.returncode == 2
    assert "--tracking-tolerance expects a JSON file path or inline JSON object" in result.stderr
    assert "must be a JSON object" in result.stderr


def test_file_path_gates_and_fifo_refuse_promptly(recording, tmp_path):
    regular = tmp_path / "tolerances.json"
    regular.write_text(json.dumps(TOLERANCE))
    symlink = tmp_path / "link.json"
    symlink.symlink_to(regular)
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    directory_link = tmp_path / "linked-parent"
    directory_link.symlink_to(tmp_path, target_is_directory=True)
    for path in (tmp_path / "absent.json", tmp_path, symlink, fifo, directory_link / regular.name):
        result = cli(recording, str(path), timeout=5)
        assert result.returncode == 2 and result.stdout == ""
        assert "--tracking-tolerance expects a JSON file path or inline JSON object" in result.stderr


def test_inline_utf8_byte_bound_and_file_bound(tmp_path, monkeypatch):
    value = '{"é": {"unit": "u", "absolute": 0}}'
    monkeypatch.setattr(fd, "MAX_JSON_BYTES", len(value.encode("utf8")) - 1)
    with pytest.raises(InputError, match="--tracking-tolerance.*byte bound"):
        fd._tracking_tolerance_arg(value)
    path = tmp_path / "too-large.json"
    path.write_text(value)
    with pytest.raises(InputError, match="--tracking-tolerance.*byte bound"):
        fd._tracking_tolerance_arg(str(path))
    monkeypatch.setattr(fd, "MAX_JSON_BYTES", len(value.encode("utf8")))
    assert fd._tracking_tolerance_arg(value) == fd._tracking_tolerance_arg(str(path))


def test_help_names_both_forms():
    result = subprocess.run(
        [sys.executable, "-m", "policy_doctor.big_bets_frontdoor", "flight-triage", "--help"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0
    assert "JSON_FILE_OR_OBJECT" in result.stdout
    assert "JSON file path or inline JSON object" in result.stdout
