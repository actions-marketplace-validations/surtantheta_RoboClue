"""Evidence-based first contact without a fabricated reference or assessment."""

import asyncio
import json
import os

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.big_bets_frontdoor import main as frontdoor
from policy_doctor.native_training_demo import run_demo
from policy_doctor.workflow_readiness import choose_workflow, main


def test_no_reference_routes_to_real_capture_instead_of_clearing_suspect_targets(tmp_path):
    before = list(tmp_path.iterdir())
    result = choose_workflow("training", framework="groot")
    assert result["readiness"] == "needs_reference"
    assert result["assessment_status"] == "not_run"
    assert result["retrospective_target_correctness"] == "unknown"
    assert result["missing_evidence"] == ["contract", "criterion_sha256", "capture"]
    assert "capture_gr00t_reference" in " ".join(result["next_steps"])
    assert "suspect" in " ".join(result["next_steps"])
    assert all(not action["executed"] for action in result["can_do_now"])
    assert list(tmp_path.iterdir()) == before


@pytest.mark.parametrize("framework,helper", [("groot", "capture_gr00t_reference"),
    ("lerobot", "capture_lerobot_reference"), ("openpi", "capture_openpi_reference")])
def test_native_guides_are_framework_specific(framework, helper):
    result = choose_workflow("training", framework=framework)
    assert helper in " ".join(result["next_steps"])
    assert result["capture_guide"].startswith("docs/")


def test_dataset_check_is_available_without_a_reference(tmp_path):
    result = choose_workflow("training", dataset=str(tmp_path))
    assert result["readiness"] == "needs_reference"
    action, = result["can_do_now"]
    assert action["mcp_request"] == {"name": "dataset_health_card", "arguments": {"source": str(tmp_path)}}
    assert result["assessment_status"] == "not_run"


def test_public_reference_plan_does_not_fetch_or_imply_permission():
    result = choose_workflow("dataset", dataset="hf://lerobot/pusht")
    assert result["public_access"]["fetch_performed"] is False
    assert result["public_access"]["operator_permission_enabled"] is False
    assert result["can_do_now"][0]["mcp_request"]["arguments"]["fetch"] is True
    assert result["available_evidence"]["dataset"] == "unresolved_public_reference"


def test_real_demo_files_route_to_check_under_the_existing_pin(tmp_path):
    demo = run_demo(tmp_path / "demo")
    # Public resources are saved native observations; the readiness plan itself
    # does not run the checker or transform those observations.
    contract = demo["artifacts"]["reference_contract"]
    capture = str(tmp_path / "demo" / "inputs" / "fault.json")
    result = choose_workflow("training", framework="groot", contract=contract,
        criterion_sha256=demo["criterion_sha256"], capture=capture)
    assert result["readiness"] == "inputs_present" and result["assessment_status"] == "not_run"
    request = result["can_do_now"][0]["mcp_request"]
    assert request["name"] == "check_training_targets"
    assert request["arguments"]["criterion_sha256"] == demo["criterion_sha256"]


def test_runtime_does_not_promote_software_evidence_to_a_physical_cause(tmp_path):
    recording = tmp_path / "flight.jsonl"
    recording.write_text("not a real recording yet")
    result = choose_workflow("runtime", recording=str(recording))
    assert result["readiness"] == "inputs_present"
    assert result["physical_cause"] == "unknown"
    assert result["assessment_status"] == "not_run"


def test_watch_config_explains_absent_native_comparisons(tmp_path):
    config = tmp_path / "watch.json"
    config.write_text(json.dumps({"kind": "owned_upgrade_watch", "assets": [{"asset_id": "saved-demo"}]}))
    result = choose_workflow("upgrade", framework="isaaclab", manifest=str(config))
    assert result["readiness"] == "needs_reference"
    assert result["missing_evidence"] == ["retained native comparisons for 1 asset(s)"]
    assert result["can_do_now"][0]["local_cli_argv"][1:3] == ["upgrade-watch", "scan-once"]
    assert "mcp_request" not in result["can_do_now"][0]


def test_native_manifest_presence_does_not_validate_its_nested_paths(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"kind": "native_upgrade_impact", "contract": {"path": "/unread/outside"}}))
    result = choose_workflow("upgrade", manifest=str(manifest))
    assert result["readiness"] == "inputs_present"
    assert result["assessment_status"] == "not_run"
    assert result["can_do_now"][0]["mcp_request"]["name"] == "assess_upgrade_impact"


@pytest.mark.parametrize("goal", ["dataset", "training", "runtime", "upgrade", "conversion"])
def test_empty_goals_always_explain_a_truthful_next_step(goal):
    result = choose_workflow(goal)
    assert result["readiness"] in {"needs_reference", "needs_inputs"}
    assert result["assessment_status"] == "not_run"
    assert result["next_steps"] and result["can_do_now"]
    assert "status" not in result


@pytest.mark.parametrize("arguments", [{"goal": "bad"}, {"goal": True},
    {"goal": "training", "framework": "automatic"}, {"goal": "training", "criterion_sha256": "guess"},
    {"goal": "training", "criterion_sha256": True}, {"goal": "dataset", "dataset": ""},
    {"goal": "dataset", "dataset": "https://somewhere/data"},
    {"goal": "dataset", "dataset": "hf://org/repo/nested"}])
def test_invalid_requests_are_input_errors(arguments):
    with pytest.raises(InputError):
        choose_workflow(**arguments)


def test_missing_path_names_input_and_does_not_report_an_assessment(tmp_path, capsys):
    assert frontdoor(["start", "--goal", "dataset", "--dataset", str(tmp_path / "typo"), "--json"]) == 2
    output = json.loads(capsys.readouterr().err)
    assert "dataset was not found" in output["error"]["message"]


def test_special_file_and_symlink_do_not_hang_or_expand_access(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(InputError, match="special"):
        choose_workflow("runtime", recording=str(fifo))
    (tmp_path / "link").symlink_to(fifo)
    with pytest.raises(InputError, match="symlink"):
        choose_workflow("runtime", recording=str(tmp_path / "link"))


def test_json_plan_exit_zero_means_plan_only(capsys):
    assert main(["--goal", "training", "--framework", "groot", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["readiness"] == "needs_reference" and value["assessment_status"] == "not_run"


def test_mcp_readiness_is_strict_read_only_and_authorized(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError

    from policy_doctor.server import create_server

    async def check():
        server = create_server([str(tmp_path)])
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert len(tools) == 29
        tool = tools["choose_workflow"]
        assert tool.annotations.readOnlyHint and not tool.annotations.openWorldHint
        assert tool.inputSchema["additionalProperties"] is False
        result = await server.call_tool("choose_workflow", {"goal": "training", "framework": "groot"})
        assert result[1]["readiness"] == "needs_reference"
        for arguments in [{"goal": "dataset", "dataset": str(tmp_path.parent)},
                          {"goal": "training", "output": str(tmp_path / "forbidden.json")},
                          {"goal": "training", "allow_hub_datasets": True}]:
            with pytest.raises(ToolError):
                await server.call_tool("choose_workflow", arguments)
        assert not (tmp_path / "forbidden.json").exists()

    asyncio.run(check())
