"""Preflight, CLI and MCP wiring of the I8 units, frames and timebase manifest."""

import asyncio
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_consumer_frontdoors as consumer_fixtures
import test_preflight as preflight_fixtures
import test_units_manifest as units_fixtures

from policy_doctor import readiness_workflow
from policy_doctor import units_manifest as units_module
from policy_doctor.artifacts import InputError
from policy_doctor.cli import human, main
from policy_doctor.server import SUMMARY_LIMIT, _summary

engine = importlib.import_module("policy_doctor.preflight")
plain_dataset, hashes = preflight_fixtures.dataset, preflight_fixtures.hashes
so_dataset, command = units_fixtures.so_dataset, units_fixtures.command
consumer_job, ordinary = consumer_fixtures.consumer_job, consumer_fixtures.ordinary
SRC = Path(__file__).resolve().parents[1] / "src"


def robot_config(path, use_degrees=False, policy=None):
    path.write_text(json.dumps({"robot": {"type": "so101_follower", "port": "/dev/x", "use_degrees": use_degrees},
                                "fps": 30, "policy": {"path": str(policy or path.parent / "policy")}}))
    return path


def structured(result):
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and isinstance(result[-1], dict):
        return result[-1]
    return json.loads(result[0].text)


def cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", "-m", "policy_doctor", *map(str, args)], capture_output=True,
                          text=True, env=env, cwd=cwd, timeout=300)


# ---------------------------------------------------------------- engine

def test_plain_preflight_is_unchanged_and_units_is_opt_in(tmp_path):
    root = plain_dataset(tmp_path)
    plain = engine.preflight(str(root))
    assert plain["status"] == "pass" and "units" not in plain["components"]
    assert "units_requested" not in plain["coverage"] and engine.UNITS_LIMITS not in plain["scope"]
    explicit = engine.preflight(str(root), units=True)
    part = explicit["components"]["units"]
    assert part["kind"] == "units_manifest" and part["title"] == "Units, frames and timebase"
    assert part["compared_with"] == "dataset only (no robot command or config)"
    assert part["units_limits"] == engine.UNITS_LIMITS and explicit["coverage"]["units_requested"] == "explicit"
    assert explicit["status"] == engine._status([c["status"] for c in explicit["components"].values()])


def test_robot_command_runs_the_units_comparison_automatically(tmp_path):
    root = so_dataset(tmp_path, "m100")  # RANGE_M100_100 values on a robot whose default is degrees
    result = engine.preflight(str(root), robot_command=command(), video_headers=False)
    part = result["components"]["units"]
    assert {"robot_readiness", "units"} <= result["components"].keys()
    assert part["status"] == result["status"] == "error"
    assert part["compared_with"] == "so101_follower from the robot command"
    assert result["coverage"]["units_requested"] == "robot command or config"
    assert "deployment_contract" not in result["components"] and result["deployment_status"] == "not_assessed"
    mismatch = next(f for f in result["findings"] if f["code"] == "units.joint_mode_mismatch")
    assert mismatch["component"] == "units" and "config_so_follower.py:42" in mismatch["message"]
    assert "the default of LeRobot 146d847" in mismatch["message"]
    assert "other LeRobot versions may default differently" in mismatch["message"]
    assert "the policy is not read" in mismatch["remedy"]
    assert any("the policy is not read" in action for action in result["next_actions"])
    assert "146d847" in part["units_limits"] and "--policy.path" in part["units_limits"]
    matched = engine.preflight(str(root), robot_command=command("--robot.use_degrees=false"), video_headers=False)
    assert matched["components"]["units"]["status"] == "pass"
    assert "units.joint_mode_match" in {f["code"] for f in matched["findings"]}


def test_a_supplied_deployment_contract_is_still_compared_with_units(case):
    # Regression: the manifest's robot side must never replace the caller's deployment contract.
    for options in ({"units": True}, {"robot_command": command()}):
        result = engine.preflight(str(case["dataset"]), checkpoint=str(case["checkpoint"]),
                                  deployment=str(case["deployment"]), video_headers=False, **options)
        part = result["components"]["deployment_contract"]
        assert part["assessment_scope"] == "declared_deployment_contract" and "units" in result["components"]
        assert part["input_sources"] == [str(case["checkpoint"]), str(case["dataset"]), str(case["deployment"])]
        assert not any("__fspath__" in f.get("message", "") for f in result["findings"])
        assert result["deployment_status"] == part["status"]


def test_robot_config_is_authorized_like_readiness_and_compared(tmp_path):
    root = so_dataset(tmp_path, "m100")
    config = robot_config(tmp_path / "rollout.json", use_degrees=False)
    result = engine.preflight(str(root), robot_config=str(config), video_headers=False)
    part = result["components"]["units"]
    assert part["status"] == "pass" and part["compared_with"] == "so101_follower from the robot config"
    assert part["input_sources"] == [str(root), str(config)]
    outside = robot_config(tmp_path.parent / (tmp_path.name + "_outside.json"))
    with pytest.raises(InputError, match="outside"):
        engine.preflight(str(root), robot_config=str(outside), read_roots=[str(tmp_path)], video_headers=False)


def test_units_co_run_with_consumer_comparison_and_the_exclusion_holds(request, tmp_path):
    job = request.getfixturevalue("consumer_job")
    result = engine.preflight(**consumer_fixtures.request(job, units=True))
    assert {"consumer_comparison", "units"} <= result["components"].keys()
    assert result["components"]["consumer_comparison"]["numeric_projection_value_effect"] == "CHANGED"
    concise = _summary(result)
    consumer_fixtures.assert_source_witness(concise["components"]["consumer_comparison"],
                                            consumer_fixtures.parquet_identity(job))
    assert concise["components"]["units"]["limits"] == engine.UNITS_LIMITS
    assert len(json.dumps(concise, allow_nan=False).encode()) <= SUMMARY_LIMIT
    with pytest.raises(InputError, match="separate preflight calls"):
        engine.preflight(**consumer_fixtures.request(job, units=True, robot_command=command()))


def test_hub_sources_never_read_more_and_say_what_was_not_run(tmp_path, monkeypatch):
    root, calls = plain_dataset(tmp_path), []

    @contextmanager
    def intake(source, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(root=root, provenance={"source": source, "resolved_revision": "a" * 40,
            "files": [{"path": k, "sha256": v} for k, v in hashes(root).items()]}, video_context=None)

    monkeypatch.setattr(engine.hub, "resolve_dataset_content", intake)
    monkeypatch.setattr(units_module, "units_manifest", lambda *a, **k: pytest.fail("no local manifest of a Hub view"))
    monkeypatch.setattr(readiness_workflow, "check_robot_readiness",
                        lambda **k: {"status": "pass", "findings": [], "axes": {}, "provenance": []})
    explicit = engine.preflight("hf://org/data", fetch=True, revision="a" * 40, units=True)
    assert explicit["components"]["units"]["status"] == "unknown"
    assert "requires a local dataset" in explicit["components"]["units"]["findings"][0]["message"]
    robot = engine.preflight("hf://org/data", fetch=True, revision="a" * 40, robot_command=command(),
                             max_bytes=64 * 1024 ** 2)
    assert "units" not in robot["components"] and "local dataset only" in robot["coverage"]["units_not_run"]
    # The readiness reservation is unchanged: the dataset allocation is the same as without the manifest.
    assert calls[-1]["max_bytes"] == 64 * 1024 ** 2 - 8 * 1024 ** 2
    assert "units: not run." in human(robot) and "local dataset only" in _summary(robot)["units_not_run"]


def test_without_the_dataset_extra_the_reason_names_it(tmp_path, monkeypatch, capsys):
    root = so_dataset(tmp_path, "m100")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    result = engine.preflight(str(root), units=True)
    part = result["components"]["units"]
    assert part["status"] == "unknown" and part["coverage"]["values_read"] is False
    assert "policy-deployment-doctor[dataset]" in part["coverage"]["reason"]
    assert "values not read: pyarrow is not installed" in human(result)
    assert "policy-deployment-doctor[dataset]" in _summary(result)["components"]["units"]["values_not_read"]
    assert main(["preflight", str(root), "--units", "--json"]) in (1, 3)
    assert json.loads(capsys.readouterr().out)["components"]["units"]["coverage"]["values_read"] is False


def test_matching_hashes_listed_once_and_a_changed_file_reaches_input_stability(tmp_path, monkeypatch):
    root = plain_dataset(tmp_path)
    result = engine.preflight(str(root), units=True)
    part = result["components"]["units"]
    assert "input_stability" not in result["components"]
    assert part["coverage"]["files_with_matching_hash_in_other_components"] >= 1
    original = units_module.units_manifest

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        report["coverage"]["files"][0]["sha256"] = "0" * 64
        return report

    monkeypatch.setattr(units_module, "units_manifest", changed)
    unstable = engine.preflight(str(root), units=True)
    assert unstable["components"]["input_stability"]["status"] == unstable["status"] == "unknown"


@pytest.mark.parametrize("value", [1, "true", 0])
def test_units_is_a_strict_boolean(tmp_path, value):
    with pytest.raises(InputError, match="strict booleans"):
        engine.preflight(str(tmp_path), units=value)


# ---------------------------------------------------------------- actual entry points

def test_entry_point_help_and_exit_codes(tmp_path):
    out = cli("preflight", "--help")
    text = " ".join(out.stdout.split()).replace("- ", "-")
    assert out.returncode == 0 and "--units" in text
    assert "runs automatically with --robot-command or --robot-config. The policy is not read" in text
    plain = plain_dataset(tmp_path)
    mismatch = so_dataset(tmp_path, "m100", name="m100")
    assert cli("preflight", plain).returncode == 0
    report = json.loads(cli("preflight", plain, "--units", "--json").stdout)
    assert report["components"]["units"]["compared_with"].startswith("dataset only")
    out = cli("preflight", mismatch, "--robot-command", command(), "--no-video-headers")
    assert out.returncode == 1, out.stderr
    assert "Units, frames and timebase: error" in out.stdout and "compared with: so101_follower" in out.stdout
    assert "the policy is not read" in out.stdout
    config = robot_config(tmp_path / "rollout.json", use_degrees=False)
    report = json.loads(cli("preflight", mismatch, "--robot-config", config, "--no-video-headers", "--json").stdout)
    assert report["components"]["units"]["status"] == "pass"
    markdown = cli("preflight", mismatch, "--units", "--markdown")
    assert markdown.stdout.startswith("# RoboClue dataset preflight") and "Units, frames and timebase" in markdown.stdout
    for args in (["--units=yes"], ["--robot-command", command(), "--robot-config", config]):
        out = cli("preflight", plain, *args)
        assert out.returncode == 2 and not out.stdout


def test_actual_stdio_server_schema_and_units(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = so_dataset(tmp_path, "m100")

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-B", "-m", "policy_doctor", "serve", "--root",
                                                                     str(tmp_path)],
                                       env=dict(os.environ, PYTHONPATH=str(SRC)))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                tool = next(t for t in tools if t.name == "preflight_check")
                units_schema = tool.inputSchema["properties"]["units"]
                assert {"type": "boolean"} in units_schema["anyOf"] and {"type": "null"} in units_schema["anyOf"]
                assert units_schema.get("default") is None
                assert "units" not in tool.inputSchema.get("required", [])
                assert "the policy is not read" in tool.description
                short = await client.call_tool("preflight_check", {"source": str(root), "robot_command": command()})
                assert not short.isError
                body = short.structuredContent
                assert len(json.dumps(body, allow_nan=False).encode()) <= SUMMARY_LIMIT
                units = body["components"]["units"]
                assert units["status"] == body["status"] == "error"
                assert units["compared_with"] == "so101_follower from the robot command"
                assert units["limits"] == engine.UNITS_LIMITS
                assert units["counts"]["error"] >= 1 and body["counts"]["error"] >= units["counts"]["error"]
                alone = (await client.call_tool("preflight_check", {"source": str(root), "units": True})).structuredContent
                assert alone["components"]["units"]["compared_with"].startswith("dataset only")
                skipped = (await client.call_tool("preflight_check", {"source": str(root), "robot_command": command(),
                                                                      "units": False})).structuredContent
                assert "units" not in skipped["components"] and skipped["units_skipped"].startswith("units=false")
                auto = (await client.call_tool("preflight_check", {"source": str(root), "robot_command": command(),
                                                                   "units": None})).structuredContent
                assert auto["components"]["units"]["status"] == "error"
                for value in ("yes", 1):
                    rejected = await client.call_tool("preflight_check", {"source": str(root), "units": value})
                    assert rejected.isError and "strict booleans" in rejected.content[0].text
    asyncio.run(asyncio.wait_for(run(), timeout=120))


# ---------------------------------------------------------------- wiring review 2: U1-U3

def wide_dataset(tmp_path, width):
    import numpy as np

    state, action = units_fixtures.arm(400, "m100", seed=1), units_fixtures.arm(400, "m100", seed=2)
    env = ([f"env_{j:05d}" for j in range(width)], np.random.default_rng(3).normal(size=(400, width)))
    return units_fixtures.write(tmp_path, {"observation.state": state, "action": action,
                                           "observation.environment_state": env}, name=f"w{width}")


@pytest.mark.parametrize("width", [256, 4096])
def test_a_wide_non_motor_feature_is_summarized_and_never_refused(tmp_path, width):
    """Review 2 U1/V2: about 340 B per coordinate; W=4096 gave exit 2 where 28c4a2cd worked."""
    root = wide_dataset(tmp_path, width)
    out = cli("preflight", root, "--robot-command", command(), "--no-video-headers", "--json")
    assert out.returncode == 1, out.stderr  # the joint-mode mismatch, never exit 2
    report = json.loads(out.stdout)
    part = report["components"]["units"]
    assert part["status"] == "error" and not part.get("reduced")
    assert len(json.dumps(part).encode()) <= engine.UNITS_MAX_BYTES
    features = part["manifest"]["features"]
    summary = features["observation.environment_state"]["coordinates_summary"]
    assert summary["total"] == width and summary["listed"] + summary["not_listed"] == width
    assert sum(item["count"] for item in summary["not_listed_by_status_and_unit"]) == summary["not_listed"]
    motors = [c for c in features["observation.state"]["coordinates"] if c.get("motor")]
    assert len(motors) == 6 and all(c["status"].startswith("decided") for c in motors if c["motor"] != "gripper")
    assert any(f["code"] == "units.joint_mode_mismatch" for f in part["findings"])
    assert "summarized: observation.environment_state" in cli("preflight", root, "--robot-command", command(),
                                                              "--no-video-headers").stdout
    assert part["coverage"]["manifest_bounded"] is True


def test_no_units_skips_the_automatic_manifest(tmp_path):
    root = so_dataset(tmp_path, "m100")
    skipped = engine.preflight(str(root), robot_command=command(), units=False)
    assert "units" not in skipped["components"] and "robot_readiness" in skipped["components"]
    assert skipped["coverage"]["units_skipped"] == "units=false (--no-units)"
    assert engine.preflight(str(root), robot_command=command())["components"]["units"]["status"] == "error"
    out = cli("preflight", root, "--robot-command", command(), "--no-units", "--no-video-headers")
    assert "Units, frames and timebase" not in out.stdout and "units: skipped (units=false (--no-units))" in out.stdout
    help_text = " ".join(cli("preflight", "--help").stdout.split()).replace("- ", "-")
    assert "--no-units Skip the units manifest that otherwise runs automatically with --robot-command" in help_text
    assert cli("preflight", root, "--units", "--no-units").returncode == 2


def test_units_mismatch_stays_visible_behind_many_groot_errors(tmp_path):
    """Review 2 U2/B6: with 16 or 60 GR00T errors the mismatch was in neither the human view nor the concise one."""
    import test_preflight_review_fixes as review

    root = so_dataset(tmp_path, "m100")
    (root / "meta/modality.json").write_text(json.dumps(review.wide_modality(30)))
    report = engine.preflight(str(root), robot_command=command())
    groot = report["components"]["groot_contract"]
    assert groot["counts"]["error"] >= 30 and report["components"]["units"]["status"] == "error"
    text = human(report)
    assert "units.joint_mode_mismatch" in text and "--robot.use_degrees=false" in text
    shown = [f["code"] for f in _summary(report)["findings_shown"]]
    assert "units.joint_mode_mismatch" in shown[:3]


def test_robot_config_mismatch_names_the_config_not_a_command(tmp_path):
    """Review 2 U3: with --robot-config the text said 'in this lerobot-rollout command'."""
    root = so_dataset(tmp_path, "m100")
    config = robot_config(tmp_path / "rollout.json", use_degrees=True)
    report = engine.preflight(str(root), robot_config=str(config))
    mismatch = next(f for f in report["findings"] if f["code"] == "units.joint_mode_mismatch")
    assert "in this robot config" in mismatch["message"] and "lerobot-" not in mismatch["message"]
    by_command = engine.preflight(str(root), robot_command=command())
    assert "in this lerobot-record command" in next(f["message"] for f in by_command["findings"]
                                                     if f["code"] == "units.joint_mode_mismatch")


def test_automatic_units_never_make_a_robot_preflight_refuse(tmp_path, monkeypatch):
    root = wide_dataset(tmp_path, 512)
    without = len(json.dumps(engine.preflight(str(root), robot_command=command(), units=False)).encode())
    full = engine.preflight(str(root), robot_command=command())
    stages = []
    for limit in range(len(json.dumps(full).encode()) + 1, without + 1000, -1500):
        monkeypatch.setattr(engine, "REPORT_LIMIT", limit)
        report = engine.preflight(str(root), robot_command=command())  # never raises above the units-free size
        assert len(json.dumps(report).encode()) <= limit
        part = report["components"].get("units")
        stages.append("left_out" if part is None else "omitted" if part.get("omitted_at_report_bound")
                      else "reduced" if part.get("reduced") else "full")
        assert report["status"] == "error" and (part is None or part["status"] == "error")  # never hidden
    assert "full" in stages and "reduced" in stages, stages
