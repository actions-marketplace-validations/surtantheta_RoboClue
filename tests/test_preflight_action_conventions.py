"""Preflight, CLI and MCP wiring of the I12 action-convention fingerprint."""

import asyncio
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_action_conventions as ac
import test_preflight as preflight_fixtures

from policy_doctor.artifacts import InputError
from policy_doctor.cli import human
from policy_doctor.server import SUMMARY_LIMIT, _summary

engine = importlib.import_module("policy_doctor.preflight")
openpi_source = ac.openpi_source  # the pinned openpi 215abfb source fixture, registered here
SRC = Path(__file__).resolve().parents[1] / "src"
ENV = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
LEROBOT = "LeRobot relative-action processor"


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-B", "-m", "policy_doctor", "preflight", *map(str, args)],
                          capture_output=True, text=True, env=ENV, cwd=cwd, timeout=300)


def groot_checkpoint(folder, reps=("RELATIVE", "ABSOLUTE"), tags=("new_embodiment",)):
    folder.mkdir()
    ac.groot_config(folder / "processor_config.json", list(reps), tags=tags)
    return folder


def libero_like(root):
    """Three incremental (delta) coordinates and three the data cannot decide, plus a gripper: openpi's LIBERO case."""
    rng = np.random.default_rng(6)
    pairs = []
    for s in ac.states(dims=6):
        a = np.column_stack([(s[1:-1, :3] - s[:-2, :3]) * 20, rng.normal(0, 1, (len(s) - 2, 3))])
        pairs.append((a, s[:-2]))
    return ac.write_dataset(root, gripper=True, pairs=pairs)[0]


def verdicts(report):
    return report["components"]["action_conventions"]["verdicts"]


def consumer_status(part):
    """The worst status among consumer findings, which the display must show (fingerprint findings excluded)."""
    statuses = {f["status"] for f in part["findings"] if f["code"] not in engine.FINGERPRINT_CODES}
    return next(s for s in ("error", "warning", "unknown", "pass") if s in statuses)


# ---------------------------------------------------------------- triggers

def test_lerobot_relative_checkpoint_runs_it_and_a_plain_checkpoint_does_not(tmp_path):
    names = ["a", "b", "c"]
    root = ac.write_dataset(tmp_path / "ds", "absolute", names=names)[0]
    relative = ac.write_checkpoint(tmp_path / "relative", names)
    result = engine.preflight(str(root), checkpoint=str(relative))
    part = result["components"]["action_conventions"]
    assert part["kind"] == "action_conventions" and part["title"] == "Action convention"
    assert verdicts(result)["consumers"] == [LEROBOT] and verdicts(result)["consumer"] == consumer_status(part)
    assert ("convention.relative_on_absolute", "pass") in {(f["code"], f["status"]) for f in part["findings"]}
    assert part["input_sources"] == [str(root), str(relative)]
    assert result["coverage"]["action_conventions_consumers"] == [LEROBOT]
    assert engine.CONVENTION_LIMITS in result["scope"]
    plain = engine.preflight(str(root), checkpoint=str(ac.write_checkpoint(tmp_path / "plain", names, relative=False)))
    assert "action_conventions" not in plain["components"] and "action_conventions_consumers" not in plain["coverage"]
    assert "action_conventions" not in engine.preflight(str(root))["components"]


def test_declared_deltas_under_a_relative_step_are_an_error(tmp_path):
    names = ["delta_a", "delta_b", "delta_c"]
    root = ac.write_dataset(tmp_path / "ds", "delta", names=names)[0]
    result = engine.preflight(str(root), checkpoint=str(ac.write_checkpoint(tmp_path / "cp", names)))
    part = result["components"]["action_conventions"]
    assert part["status"] == "error" and verdicts(result)["consumer"] == "error"
    assert any(f["code"] == "convention.relative_on_incremental" and f["status"] == "error" and
               f["component"] == "action_conventions" for f in result["findings"])
    assert result["status"] == "error"


def test_groot_processor_checkpoint_and_embodiment(tmp_path):
    root = ac.write_dataset(tmp_path / "ds", "absolute", gripper=True, modality=ac.MODALITY)[0]
    single = groot_checkpoint(tmp_path / "single")
    result = engine.preflight(str(root), checkpoint=str(single), groot_contract=False)
    assert verdicts(result)["consumers"] == ["GR00T processor_config.json"]
    assert ("convention.relative_on_absolute", "pass") in {
        (f["code"], f["status"]) for f in result["components"]["action_conventions"]["findings"]}
    multi = groot_checkpoint(tmp_path / "multi", tags=("a", "b"))
    ambiguous = engine.preflight(str(root), checkpoint=str(multi), groot_contract=False)
    assert ("convention.groot_embodiment", "unknown") in {
        (f["code"], f["status"]) for f in ambiguous["components"]["action_conventions"]["findings"]}
    chosen = engine.preflight(str(root), checkpoint=str(multi), groot_embodiment="a", groot_contract=False)
    assert verdicts(chosen)["consumer"] == consumer_status(chosen["components"]["action_conventions"])
    assert ("convention.relative_on_absolute", "pass") in {
        (f["code"], f["status"]) for f in chosen["components"]["action_conventions"]["findings"]}
    lerobot = ac.write_checkpoint(tmp_path / "lerobot", ["a", "b", "c", "gripper"])
    with pytest.raises(InputError, match='groot-embodiment.*requires a GR00T checkpoint'):
        engine.preflight(str(root), checkpoint=str(lerobot), groot_embodiment="a", groot_contract=False)
    with pytest.raises(InputError, match="supply the checkpoint"):
        engine.preflight(str(root), groot_embodiment="a")


def test_checkpoint_config_json_path_and_processor_link_out_of_the_checkpoint(tmp_path, monkeypatch):
    names = ["a", "b", "c"]
    root = ac.write_dataset(tmp_path / "ds", "absolute", names=names)[0]
    relative = ac.write_checkpoint(tmp_path / "relative", names)
    by_file = engine.preflight(str(root), checkpoint=str(relative / "config.json"))
    assert verdicts(by_file)["consumers"] == [LEROBOT]
    # A processor_config.json linking out of the checkpoint folder is never opened, and the reason is shown.
    outside = ac.groot_config(tmp_path / "outside.json", ["RELATIVE", "ABSOLUTE"])
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "processor_config.json").symlink_to(outside)
    opened = []
    original = engine.read_json
    monkeypatch.setattr(engine, "read_json", lambda path: opened.append(Path(path)) or original(path))
    result = engine.preflight(str(root), checkpoint=str(linked), groot_contract=False)
    part = result["components"]["action_conventions"]
    assert part["status"] == "unknown" and "links outside the checkpoint folder" in part["findings"][0]["message"]
    assert outside not in opened and outside.resolve() not in opened
    assert verdicts(result)["consumers"] == [] and verdicts(result)["consumer"] == "unknown"
    assert verdicts(result)["dataset_fingerprint"] == "not assessed"
    assert result["status"] == "unknown"


def test_openpi_libero_shows_consumer_pass_and_a_partly_undecided_fingerprint(tmp_path, openpi_source):
    root = libero_like(tmp_path / "ds")
    result = engine.preflight(str(root), openpi_config="pi0_libero", openpi_source_root=str(openpi_source))
    part = result["components"]["action_conventions"]
    assert part["status"] == "unknown"  # the overall semantics are kept ...
    assert verdicts(result)["consumer"] == "pass"  # ... and the display separates the two verdicts
    assert verdicts(result)["dataset_fingerprint"] == "partly undecided"
    assert verdicts(result)["consumers"] == ["openpi config 'pi0_libero'"]
    assert part["input_sources"] == [str(root), str(Path(openpi_source).resolve())]
    text = human(result)
    assert "  Action convention: unknown" in text
    assert "    delta transform consistency: pass; dataset fingerprint: partly undecided" in text
    assert "    declared transform recipes: openpi config 'pi0_libero'" in text and engine.CONVENTION_LIMITS in text
    concise = _summary(result)["components"]["action_conventions"]
    assert concise["consumer"] == "pass" and concise["dataset_fingerprint"] == "partly undecided"
    assert concise["limits"] == engine.CONVENTION_LIMITS


@pytest.mark.parametrize("kwargs,match", [
    ({"openpi_config": "pi0_libero"}, "supply both"),
    ({"openpi_source_root": "/tmp"}, "supply both"),
    ({"openpi_config": "", "openpi_source_root": "/tmp"}, "nonempty name"),
    ({"openpi_config": "x" * 129, "openpi_source_root": "/tmp"}, "nonempty name"),
    ({"groot_embodiment": 3, "checkpoint": "/tmp"}, "nonempty name"),
    ({"openpi_config": "pi0_libero", "openpi_source_root": "hf://org/openpi"}, "local directory"),
])
def test_invalid_consumer_selectors_are_refused_before_checks(tmp_path, monkeypatch, kwargs, match):
    monkeypatch.setattr(engine, "scan_dataset", lambda *a, **k: pytest.fail("checks must not start"))
    with pytest.raises(InputError, match=match):
        engine.preflight(str(tmp_path), **kwargs)


def test_openpi_source_root_is_authorized(tmp_path, openpi_source):
    root = libero_like(tmp_path / "roots" / "ds")
    with pytest.raises(InputError, match="outside the configured read roots"):
        engine.preflight(str(root), openpi_config="pi0_libero", openpi_source_root=str(openpi_source),
                         read_roots=[str(tmp_path / "roots")])
    with pytest.raises(InputError, match="not a directory"):
        engine.preflight(str(root), openpi_config="pi0_libero", openpi_source_root=str(tmp_path / "absent"))


def test_hub_source_with_a_consumer_is_unknown(tmp_path, openpi_source, monkeypatch):
    root = libero_like(tmp_path / "ds")

    @contextmanager
    def intake(source, **kwargs):
        yield SimpleNamespace(root=root, provenance={"source": source, "resolved_revision": "a" * 40, "files": [
            {"path": k, "sha256": v} for k, v in preflight_fixtures.hashes(root).items()]}, video_context=None)

    monkeypatch.setattr(engine.hub, "resolve_dataset_content", intake)
    result = engine.preflight("hf://org/data", fetch=True, revision="a" * 40, openpi_config="pi0_libero",
                              openpi_source_root=str(openpi_source))
    part = result["components"]["action_conventions"]
    assert part["status"] == "unknown" and "local dataset and checkpoint" in part["findings"][0]["message"]


def test_without_the_dataset_extra_the_fingerprint_names_it(tmp_path, openpi_source, monkeypatch):
    root = libero_like(tmp_path / "ds")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    result = engine.preflight(str(root), openpi_config="pi0_libero", openpi_source_root=str(openpi_source))
    part = result["components"]["action_conventions"]
    assert part["status"] == "unknown" and verdicts(result)["dataset_fingerprint"] == "undecided"
    assert any("policy-deployment-doctor[dataset]" in f["message"] for f in part["findings"])


def test_matching_hashes_listed_once_and_status_counts(tmp_path):
    names = ["a", "b", "c"]
    root = ac.write_dataset(tmp_path / "ds", "delta", names=names)[0]
    result = engine.preflight(str(root), checkpoint=str(ac.write_checkpoint(tmp_path / "cp", names)))
    assert "input_stability" not in result["components"]
    assert result["status"] == engine._status([c["status"] for c in result["components"].values()])
    assert result["counts"] == {s: sum(f["status"] == s for f in result["findings"]) for s in engine.PRIORITY}


# ---------------------------------------------------------------- actual entry points

def test_entry_point_help_and_exit_codes(tmp_path, openpi_source):
    out = cli("--help")
    text = " ".join(out.stdout.split()).replace("- ", "-")
    assert out.returncode == 0 and "--openpi-config NAME" in text and "--openpi-source-root PATH" in text
    assert "--groot-embodiment TAG" in text and "nothing is imported or run" in text
    root = libero_like(tmp_path / "ds")
    out = cli(root, "--openpi-config", "pi0_libero", "--openpi-source-root", openpi_source)
    assert "delta transform consistency: pass; dataset fingerprint: partly undecided" in out.stdout
    report = json.loads(cli(root, "--openpi-config", "pi0_libero", "--openpi-source-root", openpi_source,
                            "--json").stdout)
    assert out.returncode == {"error": 1, "unknown": 3}.get(report["status"], 0)
    names = ["delta_a", "delta_b", "delta_c"]
    error = ac.write_dataset(tmp_path / "err", "delta", names=names)[0]
    out = cli(error, "--policy", ac.write_checkpoint(tmp_path / "cp", names))
    assert out.returncode == 1 and "Action convention: error" in out.stdout
    assert "delta transform consistency: error" in out.stdout
    for args in (["--openpi-config", "pi0_libero"], ["--groot-embodiment", "a"],
                 ["--openpi-config", "pi0_libero", "--openpi-source-root", tmp_path / "absent"]):
        out = cli(root, *args)
        assert out.returncode == 2 and not out.stdout, args


def test_actual_stdio_server_schema_and_calls(tmp_path, openpi_source):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = libero_like(tmp_path / "ds")

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-B", "-m", "policy_doctor", "serve", "--root",
                                                                     str(tmp_path)], env=ENV)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                tool = next(t for t in tools if t.name == "preflight_check")
                for name in ("openpi_config", "openpi_source_root", "groot_embodiment"):
                    schema = tool.inputSchema["properties"][name]
                    assert {"type": "string"} in schema["anyOf"] and {"type": "null"} in schema["anyOf"]
                    assert name not in tool.inputSchema.get("required", [])
                body = (await client.call_tool("preflight_check", {
                    "source": str(root), "openpi_config": "pi0_libero",
                    "openpi_source_root": str(openpi_source)})).structuredContent
                assert len(json.dumps(body, allow_nan=False).encode()) <= SUMMARY_LIMIT
                part = body["components"]["action_conventions"]
                assert part["consumer"] == "pass" and part["dataset_fingerprint"] == "partly undecided"
                for changes, expected in (({"openpi_source_root": "/etc"}, "outside"),
                                          ({"openpi_source_root": "hf://org/x"}, "local directory"),
                                          ({"openpi_source_root": None}, "supply both")):
                    rejected = await client.call_tool("preflight_check", {
                        "source": str(root), "openpi_config": "pi0_libero",
                        "openpi_source_root": str(openpi_source), **changes})
                    assert rejected.isError and expected in rejected.content[0].text, changes
    asyncio.run(asyncio.wait_for(run(), timeout=120))


def test_fingerprint_label_keeps_undecided_and_warning_apart():
    def component(statuses, conventions):
        return {"findings": [{"code": code, "status": status} for code, status in statuses],
                "fingerprint": {"groups": [{"coordinates": [{"convention": c} for c in conventions]}]}}

    cases = [
        ([("convention.fingerprint", "unknown"), ("convention.rotation_declared_conflict", "warning"),
          ("convention.relative_on_incremental", "pass")], ["incremental", "unknown"], "pass", "partly undecided, warning"),
        ([("convention.fingerprint", "unknown")], ["unknown"], "not compared", "undecided"),
        ([("convention.fingerprint", "pass"), ("convention.lerobot_passthrough", "pass")], ["absolute"], "pass", "pass"),
        ([("convention.relative_on_incremental", "error")], [], "error", "not assessed"),
    ]
    for findings, conventions, consumer, fingerprint in cases:
        result = engine._convention_verdicts(component(findings, conventions), ["x"])
        assert (result["consumer"], result["dataset_fingerprint"]) == (consumer, fingerprint)
