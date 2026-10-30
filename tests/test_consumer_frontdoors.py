"""Generated ordinary folders through Python, CLI and actual MCP stdio.

These fixtures exercise frontdoor behavior, not a historical model checkpoint or
native/model execution. Actual native-output source parity has separate evidence.
"""

import asyncio
import copy
import hashlib
import importlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path

import pytest
from test_consumer_context import PROFILE, native_layout
from test_consumer_samples import ordinary as ordinary

from policy_doctor.artifacts import InputError
from policy_doctor.cli import human, main
from policy_doctor.server import _summary, create_server

engine = importlib.import_module("policy_doctor.preflight")


def edit(root, name, change):
    path = root / name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


@pytest.fixture
def consumer_job(request, tmp_path):
    sample_fixture = request.getfixturevalue("ordinary")
    dataset, _, values, write = sample_fixture
    # The selector fixture also advertises ID0, whose Parquet is intentionally
    # absent. A frontdoor automatic plan needs a coherent ordinary source.
    (dataset / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 72, "length": 60}) + "\n")
    edit(dataset, "meta/info.json", lambda info: info.update(
        total_episodes=1, total_frames=60, total_tasks=1, fps=50))
    values["index"] = list(range(60))
    values["timestamp"] = [row / 50 for row in range(60)]
    values["task_index"] = [0] * 60
    # Separate the first representation effect (coordinate1) from the first
    # final normalization effect (coordinate0) so their locators cannot alias.
    values["observation.state"][0][15] = 0.0
    write()
    (dataset / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "generated numeric comparison"}) + "\n")
    reference = tmp_path / "reference_processor"
    candidate = tmp_path / "candidate_processor"
    config, statistics = native_layout()
    config["processor_kwargs"]["clip_outliers"] = False
    statistics[PROFILE]["relative_action"] = {
        key: {name: [copy.deepcopy(vector) for _ in range(50)]
              for name, vector in statistics[PROFILE]["action"][key].items()}
        for key in ("left_arm", "right_arm")
    }
    for relative in statistics[PROFILE]["relative_action"].values():
        relative["min"] = [[-3.0] * 7 for _ in range(50)]
        relative["max"] = [[3.0] * 7 for _ in range(50)]
    for directory in (reference, candidate):
        directory.mkdir()
        (directory / "processor_config.json").write_text(json.dumps(config))
        (directory / "statistics.json").write_text(json.dumps(statistics))
    edit(candidate, "processor_config.json", lambda c: c["processor_kwargs"].update(use_relative_action=True))
    inactive = tmp_path / "inactive_processor"
    shutil.copytree(reference, inactive)
    edit(inactive, "statistics.json", lambda s: s[PROFILE]["state"]["left_arm"]["q01"].__setitem__(0, -1.5))
    return {"source": str(dataset), "checkpoint": str(candidate),
            "consumer_reference_checkpoint": str(reference), "read_roots": [str(tmp_path)],
            "inactive_checkpoint": str(inactive)}


def request(job, **changes):
    # This saved-processor fixture models the supported N1.7 G1 consumer.
    # The separate version tests exercise omitted-version N1.5/N1.7 checks.
    return {"groot_version": "n1.7",
            **{key: value for key, value in job.items() if key != "inactive_checkpoint"}, **changes}


def report(job, **changes):
    return engine.preflight(**request(job, **changes))


def cli_args(job):
    return ["preflight", job["source"], "--checkpoint", job["checkpoint"],
            "--groot-version", "n1.7",
            "--consumer-reference-checkpoint", job["consumer_reference_checkpoint"]]


def parquet_identity(job):
    path = Path(job["source"]) / "data/chunk-000/episode_000072.parquet"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_source_witness(compact, expected_hash):
    witness = compact["first_value_divergence"]
    assert witness["stage"] == "representation"
    assert witness["field"] == "action.left_arm"
    assert witness["coordinate"] == [0, 1]
    assert witness["requested_selection"]["episode_index"] == 72
    for side in ("reference", "candidate"):
        assert witness[side + "_row"]["caller_selected_index"] == 0
        mapping = witness["source_mapping"][side]
        assert mapping["column"] == "action"
        assert mapping["declared_slice"] == [15, 22]
        assert mapping["source_coordinate"] == 15 + witness["coordinate"][1]
        source = witness[side + "_source"]
        member = next(item for item in source["files"] if item["path"].endswith(".parquet"))
        assert member["path"] == "data/chunk-000/episode_000072.parquet"
        assert member["sha256"] == expected_hash
    assert compact["first_final_value_difference"]["field"] == "action.left_arm"
    assert compact["first_final_value_difference"]["coordinate"] == [0, 0]
    final = compact["first_final_value_source"]
    assert final["stage"] == "clipped"
    assert final["coordinate"] == [0, 0]
    assert final["source_identity_from"] == "first_value_divergence"
    for side in ("reference", "candidate"):
        assert final[side + "_row"]["caller_selected_index"] == 0
        assert final["source_mapping"][side]["column"] == "action"
        assert final["source_mapping"][side]["source_coordinate"] == 15
        assert witness["source_mapping"][side]["source_coordinate"] == 16


def test_python_ordinary_folder_auto_plan_observes_change_without_hiding_unknowns(consumer_job):
    result = report(consumer_job)
    part = result["components"]["consumer_comparison"]
    assert part["status"] == "pass"
    assert part["numeric_projection_value_effect"] == "CHANGED"
    assert part["coverage"]["fully_projected_windows"] == 3
    windows = part["comparison"]["samples"]
    assert [item["requested_selection"]["anchor_index"] for item in windows] == [0, 5, 10]
    assert all(item["requested_selection"]["episode_index"] == 72 for item in windows)
    assert result["status"] == "unknown"
    separate = result["components"]["checkpoint_metadata"]
    assert separate["status"] == "unknown"
    assert separate["assessment_scope"] == "lerobot_policy_interface"
    assert all(value == "UNKNOWN" for value in part["boundaries"].values())
    assert_source_witness(_summary(result)["components"]["consumer_comparison"], parquet_identity(consumer_job))


def test_python_requested_equality_changes_only_the_supported_invariant_verdict(consumer_job):
    result = report(consumer_job, consumer_require_equal=True)
    part = result["components"]["consumer_comparison"]
    assert part["status"] == result["status"] == "error"
    assert part["numeric_projection_value_effect"] == "CHANGED"
    assert any(f["code"] == "consumer.selected_value_invariant_changed" for f in part["findings"])
    assert result["components"]["checkpoint_metadata"]["status"] == "unknown"


def test_inactive_saved_statistic_identity_change_does_not_invent_numeric_harm(consumer_job):
    result = report(consumer_job, checkpoint=consumer_job["inactive_checkpoint"], consumer_require_equal=True)
    part = result["components"]["consumer_comparison"]
    assert part["status"] == "pass"
    assert part["numeric_projection_value_effect"] == "EQUAL"
    compact = _summary(result)["components"]["consumer_comparison"]
    assert compact["artifact_identity_changed"] is True
    assert compact["active_interpretation_changed"] is False
    assert compact["first_value_divergence"] is None
    assert compact["first_final_value_difference"] is None
    assert result["status"] == "unknown"


@pytest.mark.parametrize("form", ["human", "json"])
def test_cli_default_and_json_show_changed_effect_while_broad_preflight_stays_unknown(consumer_job, capsys, form):
    code = main(cli_args(consumer_job) + (["--json"] if form == "json" else []))
    output = capsys.readouterr().out
    assert code == 3
    if form == "json":
        result = json.loads(output)
        assert result["status"] == "unknown"
        assert result["components"]["consumer_comparison"]["status"] == "pass"
        assert result["components"]["consumer_comparison"]["numeric_projection_value_effect"] == "CHANGED"
    else:
        assert output.startswith("RoboClue preflight: UNKNOWN")
        assert "selected numeric values: CHANGED" in output
        assert "consumer comparison: pass" in output
        assert "representation action.left_arm" in output
        assert "episode_000072.parquet" in output
        assert parquet_identity(consumer_job) in output
        assert "full native processor and model target: UNKNOWN" in output


def test_cli_equality_flag_exits_error_and_explicit_selector_reaches_actual_window(consumer_job, capsys):
    code = main(cli_args(consumer_job) + ["--consumer-sample", "72:5", "--consumer-require-equal", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    part = result["components"]["consumer_comparison"]
    assert part["status"] == "error" and part["numeric_projection_value_effect"] == "CHANGED"
    assert part["coverage"]["requested_windows"] == 1
    assert part["comparison"]["samples"][0]["requested_selection"]["anchor_index"] == 5


@pytest.mark.parametrize("selector", ["72", "72:5:1", "72:true", "-1:5", "72:1.0", "７２:5"])
def test_cli_bad_selectors_are_actionable_input_errors(consumer_job, capsys, selector):
    assert main(cli_args(consumer_job) + ["--consumer-sample=" + selector, "--json"]) == 2
    output = capsys.readouterr()
    assert "consumer-sample" in output.err
    assert not output.out


def test_concise_preserves_value_source_witness_under_many_unrelated_findings(consumer_job):
    result = report(consumer_job, consumer_require_equal=True)
    result["findings"] += [{"status": "error", "code": "generated.unrelated." + str(i),
                            "message": "Other independent finding " * 50, "evidence": []} for i in range(100)]
    result["next_actions"] += ["Other independent action " * 100 for _ in range(20)]
    concise = _summary(result)
    assert len(json.dumps(concise, allow_nan=False).encode()) <= 8192
    assert concise["status"] == "error"
    assert concise["findings_omitted"] > 0
    compact = concise["components"]["consumer_comparison"]
    assert compact["status"] == "error" and compact["numeric_projection_value_effect"] == "CHANGED"
    assert compact["first_error"]["code"] == "consumer.selected_value_invariant_changed"
    assert_source_witness(compact, parquet_identity(consumer_job))
    assert "selected numeric values: CHANGED" in human(result)


@pytest.mark.parametrize("change", [
    {"consumer_require_equal": 1}, {"consumer_require_equal": "false"},
    {"consumer_samples": []}, {"consumer_samples": [{"episode_index": True, "anchor_index": 0}]},
    {"consumer_samples": [{"episode_index": 72, "anchor_index": False}]},
    {"consumer_samples": [{"episode_index": 72, "anchor_index": 1.0}]},
    {"consumer_samples": [{"episode_index": 72, "anchor_index": 0, "allow_padding": 1}]},
    {"consumer_samples": [{"episode_index": 72, "anchor_index": 0, "extra": "ignored?"}]},
    {"consumer_samples": [{"episode_index": 72, "anchor_index": 4096}]},
    {"consumer_samples": [{"episode_index": 1000000, "anchor_index": 0}]},
])
def test_python_new_arguments_refuse_before_any_source_body_or_network_read(consumer_job, monkeypatch, change):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid consumer arguments must be rejected before reads/network")

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(socket, "create_connection", forbidden)
        patch.setattr(engine.hub, "resolve_dataset_content", forbidden)
        with pytest.raises(InputError):
            engine.preflight(**request(consumer_job, **change))


def test_all_consumer_paths_are_authorized_before_reads_or_optional_hub_intake(consumer_job, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Outside-root consumer paths must be rejected before intake")

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(socket, "create_connection", forbidden)
        patch.setattr(engine.hub, "resolve_dataset_content", forbidden)
        with pytest.raises(InputError, match="outside"):
            engine.preflight(**request(consumer_job,
                consumer_reference_checkpoint=str(tmp_path.parent / "unauthorized_checkpoint")))
        with pytest.raises(InputError):
            engine.preflight(**request(consumer_job, source="hf://generated/dataset", fetch=True))


def test_unsupported_processor_remains_unknown_and_lerobot_audit_scope_is_accurate(consumer_job):
    edit(Path(consumer_job["checkpoint"]), "processor_config.json",
         lambda config: config.update(processor_class="UnsupportedGeneratedProcessor"))
    result = report(consumer_job)
    assert result["status"] == "unknown"
    assert result["components"]["consumer_comparison"]["status"] == "unknown"
    separate = result["components"]["checkpoint_metadata"]
    assert separate["status"] == "unknown"
    assert separate["assessment_scope"] == "lerobot_policy_interface"
    message = " ".join(f["message"] for f in separate["findings"])
    assert "LeRobot" in message and "GR00T" in message
    assert "No config.json" not in message
    assert "UNKNOWN" in message


def test_actual_mcp_stdio_default_concise_and_detailed_reach_saved_consumer_comparison(consumer_job, tmp_path):
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import stdio_client

    source = str(Path(__file__).resolve().parents[1] / "src")
    bootstrap = ("import sys,runpy;sys.dont_write_bytecode=True;sys.path.insert(0," + repr(source) + ");"
                 "sys.argv=['policy_doctor','serve','--root'," + repr(str(tmp_path)) + "];"
                 "runpy.run_module('policy_doctor',run_name='__main__')")
    arguments = {key: value for key, value in request(consumer_job).items() if key != "read_roots"}
    expected_hash = parquet_identity(consumer_job)

    async def exercise():
        parameters = mcp.StdioServerParameters(command=sys.executable,
            args=["-I", "-B", "-c", bootstrap], cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
        async with stdio_client(parameters) as (read, write):
            async with mcp.ClientSession(read, write) as client:
                await client.initialize()
                tool = next(t for t in (await client.list_tools()).tools if t.name == "preflight_check")
                assert tool.inputSchema["additionalProperties"] is False
                properties = tool.inputSchema["properties"]
                assert properties["consumer_require_equal"]["type"] == "boolean"
                schemas = properties["consumer_samples"].get("anyOf", [properties["consumer_samples"]])
                array = next(s for s in schemas if s.get("type") == "array")
                assert array["items"]["additionalProperties"] is False
                assert array["items"]["properties"]["episode_index"]["type"] == "integer"
                short = await client.call_tool("preflight_check", arguments)
                assert not short.isError
                body = short.structuredContent
                assert body["status"] == "unknown"
                assert len(json.dumps(body, allow_nan=False).encode()) <= 8192
                compact = body["components"]["consumer_comparison"]
                assert compact["status"] == "pass" and compact["numeric_projection_value_effect"] == "CHANGED"
                assert_source_witness(compact, expected_hash)
                full = await client.call_tool("preflight_check", {**arguments, "response_format": "detailed"})
                assert not full.isError
                assert full.structuredContent["components"]["consumer_comparison"]["coverage"]["fully_projected_windows"] == 3
                equal = await client.call_tool("preflight_check", {**arguments, "consumer_require_equal": True})
                assert not equal.isError and equal.structuredContent["status"] == "error"
                assert equal.structuredContent["components"]["consumer_comparison"]["numeric_projection_value_effect"] == "CHANGED"
                inactive = await client.call_tool("preflight_check", {**arguments,
                    "checkpoint": consumer_job["inactive_checkpoint"], "consumer_require_equal": True})
                assert not inactive.isError
                part = inactive.structuredContent["components"]["consumer_comparison"]
                assert part["status"] == "pass" and part["numeric_projection_value_effect"] == "EQUAL"
                assert part["artifact_identity_changed"] is True and part["active_interpretation_changed"] is False
                for changes in [
                    {"consumer_require_equal": 1},
                    {"consumer_samples": [{"episode_index": True, "anchor_index": 0}]},
                    {"consumer_samples": [{"episode_index": 72, "anchor_index": 0, "allow_padding": 1}]},
                    {"consumer_samples": [{"episode_index": 72, "anchor_index": 0, "extra": True}]},
                    {"consumer_reference_checkpoint": str(tmp_path.parent / "outside_checkpoint")},
                    {"unknown_consumer_option": True},
                ]:
                    rejected = await client.call_tool("preflight_check", {**arguments, **changes})
                    assert rejected.isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=60))


@pytest.mark.parametrize("source_case, expected", [("hub", "literal local path"), ("local", "outside")])
def test_mcp_bad_consumer_authority_does_not_reach_hub_or_body_intake(
    consumer_job, tmp_path, monkeypatch, source_case, expected
):
    pytest.importorskip("mcp")
    server = create_server([str(tmp_path)], allow_hub_datasets=True)

    intake_calls = []

    def forbidden(*args, **kwargs):
        intake_calls.append((args, kwargs))
        raise AssertionError("Invalid consumer authorization must stop before Hub/local intake")

    async def exercise():
        with monkeypatch.context() as patch:
            patch.setattr(os, "open", forbidden)
            patch.setattr(Path, "read_bytes", forbidden)
            patch.setattr(Path, "read_text", forbidden)
            patch.setattr(engine.hub, "resolve_dataset_content", forbidden)
            patch.setattr(socket, "create_connection", forbidden)
            with pytest.raises(Exception, match=expected):
                await server.call_tool("preflight_check", {
                    "source": "hf://generated/dataset" if source_case == "hub" else consumer_job["source"],
                    "checkpoint": consumer_job["checkpoint"],
                    "consumer_reference_checkpoint": str(tmp_path.parent / "outside_checkpoint")})
    asyncio.run(exercise())
    assert intake_calls == []


@pytest.mark.parametrize("selector", ["9" * 5000 + ":5", "72:" + "9" * 5000], ids=["huge-episode", "huge-anchor"])
def test_cli_huge_decimal_selection_is_invalid_input_before_engine_reads(monkeypatch, capsys, selector):
    def forbidden(*args, **kwargs):
        raise AssertionError("An oversized selector must be rejected before the engine can read anything")

    monkeypatch.setattr(engine, "preflight", forbidden)
    code = main(["preflight", "/unread-generated-dataset", "--checkpoint", "/unread-generated-checkpoint",
                 "--consumer-reference-checkpoint", "/unread-generated-reference",
                 "--consumer-sample", selector, "--json"])
    captured = capsys.readouterr()
    assert code == 2 and not captured.out
    error = json.loads(captured.err)["error"]
    assert error["code"] == "invalid_input"
    assert "--consumer-sample" in error["message"]
    assert len(error["message"]) < 400


@pytest.mark.parametrize("episode, anchor", [("72", "5"), ("0", "0")], ids=["small-values", "all-zero"])
def test_cli_long_leading_zeros_normalize_before_integer_conversion(monkeypatch, capsys, episode, anchor):
    observed = []

    def capture(source, **kwargs):
        observed.append({"source": source, **kwargs})
        return {"status": "pass", "findings": []}

    monkeypatch.setattr(engine, "preflight", capture)
    selector = "0" * 5000 + episode + ":" + "0" * 5000 + anchor
    assert main(["preflight", "/unread-generated-dataset", "--checkpoint", "/unread-generated-checkpoint",
                 "--consumer-reference-checkpoint", "/unread-generated-reference",
                 "--consumer-sample", selector, "--json"]) == 0
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out)["status"] == "pass"
    assert len(observed) == 1
    assert observed[0]["consumer_samples"] == [{"episode_index": int(episode), "anchor_index": int(anchor)}]
    assert type(observed[0]["consumer_samples"][0]["episode_index"]) is int
    assert type(observed[0]["consumer_samples"][0]["anchor_index"]) is int
