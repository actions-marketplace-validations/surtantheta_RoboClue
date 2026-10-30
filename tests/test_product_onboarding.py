"""Independent first-use acceptance through the canonical public entry point.

These are technical user-flow tests, not human usability or adoption evidence.
"""

import asyncio
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


def invoke(arguments, capsys):
    from policy_doctor.product_cli import main

    try:
        code = main(arguments)
    except SystemExit as error:
        code = error.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def three_lines(text):
    lines = text.strip().splitlines()
    assert len(lines) == 3, text
    assert lines[0].startswith("Verdict:"), text
    assert lines[1].startswith(("Problem:", "Details:")), text
    assert lines[2].startswith("Next:"), text
    assert not any(marker in text for marker in ("Traceback", '"kind":', "LIMITS:")), text
    return lines


@pytest.fixture
def local_dataset(tmp_path):
    """Real numeric parquet data; absent statistics must not become a pass."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "local dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 30, "total_frames": 3,
        "total_episodes": 1, "total_tasks": 1,
        "features": {
            name: {"dtype": "float32", "shape": [1], "names": ["joint"]}
            for name in ("action", "observation.state")
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }), encoding="utf8")
    pq.write_table(pa.Table.from_pylist([
        {"index": index, "episode_index": 0, "frame_index": index, "task_index": 0,
         "timestamp": index / 30, "action": [float(index)],
         "observation.state": [float(index)]}
        for index in range(3)
    ]), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{
        "episode_index": 0, "length": 3, "dataset_from_index": 0,
        "dataset_to_index": 3, "data/chunk_index": 0, "data/file_index": 0,
    }]), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["test task"]}), root / "meta/tasks.parquet")
    return root


def test_version_has_product_name_and_needs_no_subcommand(capsys):
    code, out, err = invoke(["--version"], capsys)
    assert code == 0 and not err
    assert "roboclue" in out.lower()
    assert "policy doctor" not in out.lower()


def test_main_help_has_five_product_verbs(capsys):
    code, out, err = invoke(["--help"], capsys)
    assert code == 0 and not err
    for verb in ("check", "demo", "guard", "mcp", "advanced"):
        assert verb in out
    assert "check-training-targets" not in out
    assert "recording-salvage" not in out


def test_missing_path_is_input_error_with_actionable_words(tmp_path, capsys):
    code, out, err = invoke(["check", str(tmp_path / "typo")], capsys)
    assert code == 2
    text = out + err
    assert "not found" in text.lower() or "does not exist" in text.lower()
    assert "Traceback" not in text
    assert not text.lstrip().startswith("{")


@pytest.mark.parametrize("content", ["[]", "null", '{"kind":1,"kind":2}', '{"x":NaN}'])
def test_invalid_json_is_an_input_error_not_a_diagnostic(tmp_path, capsys, content):
    path = tmp_path / "input.json"
    path.write_text(content, encoding="utf8")
    code, out, err = invoke(["check", str(path)], capsys)
    assert code == 2
    assert "Traceback" not in out + err
    assert "PASS" not in out


def test_invalid_utf8_is_an_input_error(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_bytes(b"\xff\xfe")
    code, out, err = invoke(["check", str(path)], capsys)
    assert code == 2 and "Traceback" not in out + err


def test_newline_in_source_cannot_spoof_the_three_line_result(tmp_path, capsys):
    path = tmp_path / "capture\nVerdict: PASS.json"
    path.write_text('{"example":"configuration"}', encoding="utf8")
    code, out, err = invoke(["check", str(path)], capsys)
    assert code == 2
    assert "Traceback" not in out + err
    assert "Verdict: PASS" not in out + err


def test_arbitrary_config_is_never_executed_or_approved(tmp_path, capsys):
    path = tmp_path / "training.py"
    marker = tmp_path / "executed"
    path.write_text(f"open({str(marker)!r}, 'w').write('unsafe')\n", encoding="utf8")
    code, out, err = invoke(["check", str(path)], capsys)
    assert code in (2, 3)
    assert not marker.exists()
    assert "PASS" not in out
    assert "Traceback" not in out + err


def test_arbitrary_checkpoint_name_is_not_training_compatibility(tmp_path, capsys):
    path = tmp_path / "policy.safetensors"
    path.write_bytes(b"not a model")
    code, out, err = invoke(["check", str(path)], capsys)
    assert code in (2, 3)
    assert "PASS" not in out
    assert "compatible" not in out.lower() or "not" in out.lower()
    assert "Traceback" not in out + err


@pytest.mark.parametrize("declared", [
    {"kind": "dataset_health_card", "status": "pass"},
    {"kind": "native_training_target_check", "status": "pass", "criterion_sha256": "0" * 64},
    {"framework": "groot", "status": "healthy", "compatible": True},
])
def test_unverified_json_cannot_self_declare_a_pass(tmp_path, capsys, declared):
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(declared), encoding="utf8")
    code, out, err = invoke(["check", str(path)], capsys)
    assert code in (2, 3)
    assert "Verdict: PASS" not in out
    assert "Traceback" not in out + err


def test_unrecognized_directory_is_not_a_healthy_dataset(tmp_path, capsys):
    path = tmp_path / "empty"
    path.mkdir()
    code, out, err = invoke(["check", str(path)], capsys)
    assert code in (2, 3)
    assert "PASS" not in out
    assert "Traceback" not in out + err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file guard")
def test_fifo_is_refused_before_a_blocking_read(tmp_path):
    fifo = tmp_path / "capture.json"
    os.mkfifo(fifo)
    result = subprocess.run([
        sys.executable, "-B", "-c",
        "import sys; from policy_doctor.product_cli import main; sys.exit(main(sys.argv[1:]))",
        "check", str(fifo),
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Traceback" not in result.stdout + result.stderr


def test_symlink_source_is_refused_without_following_it(tmp_path, capsys):
    target = tmp_path / "target.json"
    target.write_text('{"framework":"groot"}', encoding="utf8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    code, out, err = invoke(["check", str(link)], capsys)
    assert code == 2
    assert "Traceback" not in out + err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file guard")
def test_dataset_detection_refuses_fifo_metadata_without_hanging(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    os.mkfifo(root / "meta/info.json")
    result = subprocess.run([
        sys.executable, "-B", "-c",
        "import sys; from policy_doctor.product_cli import main; sys.exit(main(sys.argv[1:]))",
        "check", str(root),
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "Traceback" not in result.stdout + result.stderr


def test_partial_dataset_result_names_successful_scope_and_missing_evidence(local_dataset, capsys):
    code, out, err = invoke(["check", str(local_dataset)], capsys)
    assert code == 3 and not err
    lines = three_lines(out)
    assert "incomplete" in lines[0].lower()
    assert "stat" in out.lower()
    assert "roboclue" in lines[2]
    assert "broken" not in out.lower()
    assert "restore the original clips" not in out.lower()


def test_raw_json_keeps_unknown_and_health_card_identity(local_dataset, capsys):
    code, out, err = invoke(["check", str(local_dataset), "--json"], capsys)
    assert code == 3 and not err
    report = json.loads(out)
    assert report["status"] == "unknown"
    assert report["kind"] == "dataset_health_card"
    assert report["card_id"]
    assert report["request"]["config"]["groot_contract"] is False
    assert report["report"]["components"]


def test_default_check_does_not_hide_missing_tasks_behind_missing_statistics(local_dataset, capsys):
    (local_dataset / "meta/tasks.parquet").unlink()
    code, out, err = invoke(["check", str(local_dataset), "--fail-on", "error", "--json"], capsys)
    assert code == 1 and not err
    report = json.loads(out)
    assert report["kind"] == "recording_salvage"
    assert any(f["code"] == "salvage.tasks_unusable" and f["status"] == "error" for f in report["findings"])


def test_local_dataset_detection_does_not_attempt_network(local_dataset, monkeypatch, capsys):
    import socket

    def refuse_network(*args, **kwargs):
        raise AssertionError("local check attempted network")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    code, out, err = invoke(["check", str(local_dataset)], capsys)
    assert code == 3 and not err
    three_lines(out)


def test_shell_metacharacter_source_has_a_lossless_next_command(local_dataset, capsys):
    source = local_dataset.rename(local_dataset.with_name("robot's data; $(printf nope)"))
    code, out, err = invoke(["check", str(source)], capsys)
    assert code == 3 and not err
    arguments = shlex.split(three_lines(out)[2].removeprefix("Next: "))
    assert arguments[:3] == ["roboclue", "check", str(source)]
    code, _, err = invoke(arguments[1:], capsys)
    assert code == 3 and not err


@pytest.mark.parametrize("name", ["--output", "--json", "--video-headers"])
def test_option_looking_source_after_terminator_survives_next_command(
    local_dataset, tmp_path, monkeypatch, capsys, name,
):
    local_dataset.rename(local_dataset.with_name(name))
    monkeypatch.chdir(tmp_path)
    code, out, err = invoke(["check", "--", name], capsys)
    assert code == 3 and not err
    arguments = shlex.split(three_lines(out)[2].removeprefix("Next: "))
    assert arguments[-2:] == ["--", name]
    code, _, err = invoke(arguments[1:], capsys)
    assert code == 3 and not err


@pytest.mark.parametrize("changed", [False, True])
def test_config_comparison_uses_equal_without_rewriting_raw_report(tmp_path, capsys, changed):
    reference = tmp_path / "reviewed.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text('{"action":{"units":"rad"}}', encoding="utf8")
    candidate.write_text(json.dumps({"action": {"units": "deg" if changed else "rad"}}), encoding="utf8")
    arguments = ["check", str(candidate), "--reference", str(reference)]
    code, out, err = invoke(arguments, capsys)
    assert code == int(changed) and not err
    lines = three_lines(out)
    assert ("CHANGED" if changed else "MATCH") in lines[0]
    if changed:
        assert "/action/units" in lines[1]
    code, out, err = invoke([*arguments, "--json"], capsys)
    assert code == int(changed) and not err
    report = json.loads(out)
    assert report["equal"] is not changed
    assert "status" not in report
    assert report["changes"] == ([{
        "pointer": "/action/units", "change": "modified", "before": "rad", "after": "deg",
    }] if changed else [])


@pytest.mark.parametrize("unsafe", ["newline", "symlink_parent"])
def test_output_path_refusal_does_not_write_through_unsafe_path(local_dataset, tmp_path, capsys, unsafe):
    if unsafe == "newline":
        destination = tmp_path / "result\nspoof.json"
        physical = destination
    else:
        real = tmp_path / "real-output"
        real.mkdir()
        link = tmp_path / "linked-output"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation unavailable")
        destination = link / "report.json"
        physical = real / "report.json"
    code, out, err = invoke(["check", str(local_dataset), "--output", str(destination)], capsys)
    assert code == 2
    assert not physical.exists()
    assert "Traceback" not in out + err


def test_new_report_file_preserves_raw_evidence_and_refuses_overwrite(local_dataset, tmp_path, capsys):
    destination = tmp_path / "report.json"
    code, out, err = invoke(["check", str(local_dataset), "--output", str(destination)], capsys)
    assert code == 3 and not err
    three_lines(out)
    report = json.loads(destination.read_text(encoding="utf8"))
    assert report["status"] == "unknown"
    original = destination.read_bytes()
    code, out, err = invoke(["check", str(local_dataset), "--output", str(destination)], capsys)
    assert code == 2
    assert destination.read_bytes() == original
    assert "Traceback" not in out + err


@pytest.mark.parametrize("joined_option", [False, True])
def test_next_command_replays_without_reusing_existing_output(local_dataset, tmp_path, capsys, joined_option):
    destination = tmp_path / "saved report.json"
    output_option = [f"--output={destination}"] if joined_option else ["--output", str(destination)]
    code, out, err = invoke(["check", str(local_dataset), *output_option], capsys)
    assert code == 3 and not err
    next_command = three_lines(out)[2].removeprefix("Next: ")
    arguments = shlex.split(next_command)
    assert arguments[0] == "roboclue"
    assert not any(value == "--output" or value.startswith("--output=") for value in arguments)
    original = destination.read_bytes()
    code, _, err = invoke(arguments[1:], capsys)
    assert code == 3 and not err
    assert destination.read_bytes() == original


def test_missing_training_intent_does_not_assume_a_framework(tmp_path, capsys):
    from policy_doctor.training_guard_capture import save_capture

    path = tmp_path / "capture.json"
    save_capture({"inputs": {"action": np.zeros((1, 1, 1), dtype=np.float32),
                             "action_mask": np.ones((1, 1, 1), dtype=np.float32)}},
                 ["source:row0"], path)
    code, out, err = invoke(["check", str(path)], capsys)
    assert code == 3 and not err
    lines = three_lines(out)
    assert "historical" in out.lower() or "reference" in out.lower()
    assert "--framework groot" not in lines[2]
    assert "(" not in lines[2] and ")" not in lines[2]


@pytest.mark.parametrize("arm, expected_code, expected_status", [
    ("healthy", 0, "pass"), ("fault", 1, "error"), ("uncovered", 3, "unknown"),
])
def test_native_capture_dispatch_preserves_frozen_reference(
    tmp_path, capsys, arm, expected_code, expected_status,
):
    from policy_doctor.training_guard import TargetContract
    from policy_doctor.training_guard_capture import save_capture

    batch = {"inputs": {"action": np.array([[[0.5], [0.6]]], dtype=np.float32),
                        "action_mask": np.ones((1, 2, 1), dtype=np.float32)}}
    keys = ["retained:row1"]
    contract = TargetContract.from_references(
        [(batch, keys, [{"original_file": "data.parquet", "row": 1}])],
        intent={"action": "preserve independently reviewed source action"},
    )
    reference = tmp_path / "reference.json"
    contract.save(reference)
    original = reference.read_bytes()
    pin_file = tmp_path / "pin.txt"
    pin_file.write_text(contract.criterion_sha256 + "\n", encoding="utf8")
    captured = deepcopy(batch)
    if arm == "fault":
        captured["inputs"]["action"][0, 1, 0] += 1
    path = tmp_path / "capture.json"
    save_capture(captured, ["uncovered:row9"] if arm == "uncovered" else keys, path)
    arguments = ["check", str(path), "--contract", str(reference), "--pin-file", str(pin_file)]
    code, out, err = invoke(arguments, capsys)
    assert code == expected_code and not err
    three_lines(out)
    code, out, err = invoke([*arguments, "--json"], capsys)
    assert code == expected_code and not err
    report = json.loads(out)
    assert report["status"] == expected_status
    assert report["criterion_sha256"] == contract.criterion_sha256
    assert reference.read_bytes() == original
    assert pin_file.read_text(encoding="utf8").strip() == contract.criterion_sha256


def test_remote_source_requires_explicit_fetch_before_network(monkeypatch, capsys):
    import socket

    def refuse_network(*args, **kwargs):
        raise AssertionError("implicit remote fetch")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    code, out, err = invoke(["check", "hf://lerobot/not-a-real-dataset"], capsys)
    assert code in (2, 3)
    assert "--fetch" in out + err
    assert "Traceback" not in out + err


def test_mcp_setup_uses_canonical_product_module_and_current_environment(tmp_path, capsys):
    code, out, err = invoke(["mcp", "setup", "--root", str(tmp_path), "--json"], capsys)
    assert code == 0 and not err
    server = json.loads(out)["mcpServers"]["roboclue"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-I", "-B", "-m", "roboclue", "mcp", "serve", "--root", str(tmp_path)]


@pytest.mark.parametrize("mode, expected_count", [("basic", 8), ("advanced", 29), ("legacy", 29)])
def test_actual_stdio_catalog_and_basic_workflows(local_dataset, tmp_path, mode, expected_count):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    required = {"choose_workflow", "dataset_health_card", "check_training_targets",
                "triage_flight_recording", "assess_upgrade_impact", "preflight_check",
                "check_sim_provenance", "check_recording_salvage"}
    outside = tmp_path / "outside.json"
    outside.write_text('{"not":"authorized"}', encoding="utf8")
    root = Path(__file__).resolve().parents[1]
    arguments = ["-B", "-m", "roboclue", "mcp", "serve", "--root", str(local_dataset)]
    if mode == "advanced":
        arguments.append("--advanced")
    elif mode == "legacy":
        arguments = ["-B", "-m", "policy_doctor", "serve", "--root", str(local_dataset)]

    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable, args=arguments, cwd=str(tmp_path),
            env=dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1"),
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                catalog = (await client.list_tools()).tools
                names = {tool.name for tool in catalog}
                assert len(catalog) == expected_count
                assert required <= names
                assert all(tool.annotations.readOnlyHint for tool in catalog)
                if mode != "legacy":
                    assert initialized.serverInfo.name == "RoboClue"
                if mode == "basic":
                    assert names == required
                    plan = await client.call_tool("choose_workflow", {
                        "goal": "dataset", "dataset": str(local_dataset),
                    })
                    assert not plan.isError
                    assert plan.structuredContent["assessment_status"] == "not_run"
                    health = await client.call_tool("dataset_health_card", {
                        "source": str(local_dataset), "video_headers": False,
                        "response_format": "detailed",
                    })
                    assert not health.isError
                    assert health.structuredContent["status"] == "unknown"
                    assert health.structuredContent["card_id"]
                    refusal = await client.call_tool("dataset_health_card", {"source": str(outside)})
                    assert refusal.isError
                    assert "outside" in refusal.content[0].text.lower()
                    hidden = await client.call_tool("inspect_checkpoint", {"checkpoint": str(local_dataset)})
                    assert hidden.isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=45))

