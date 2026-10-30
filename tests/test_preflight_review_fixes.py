"""Fixes from the independent review of the GR00T contract / residual triage wiring (ad9f0360)."""

import asyncio
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import test_commanded_reach as reach_fixtures
import test_groot_contract as groot_fixtures

from policy_doctor import commanded_reach as reach_module
from policy_doctor.artifacts import InputError
from policy_doctor.cli import human
from policy_doctor.groot_contract import check_groot_dataset
from policy_doctor.server import SUMMARY_LIMIT, _summary, create_server

engine = importlib.import_module("policy_doctor.preflight")
SRC = Path(__file__).resolve().parents[1] / "src"
ENV = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")


@pytest.fixture
def reach(monkeypatch):
    monkeypatch.setattr(reach_fixtures, "RNG", np.random.default_rng(7))
    return reach_fixtures


def structured(result):
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and isinstance(result[-1], dict):
        return result[-1]
    return json.loads(result[0].text)


def cli(*args, timeout=120):
    return subprocess.run([sys.executable, "-B", "-m", "policy_doctor", "preflight", *map(str, args)],
                          capture_output=True, text=True, env=ENV, timeout=timeout)


def wide_modality(n):
    modality = json.loads(json.dumps(groot_fixtures.MODALITY))
    for i in range(n):
        modality["state"][f"extra_state_group_{i:05d}"] = {"start": 6 + i, "end": 7 + i}
        modality["action"][f"extra_action_group_{i:05d}"] = {"start": 6 + i, "end": 7 + i}
    return modality


# ---------------------------------------------------------------- 1. directory links back into data/ never hang

@pytest.mark.parametrize("links", [{"l0": "."}, {"l0": ".", "l1": "."}, {"up": "..", "l0": "."}],
                         ids=["one-self-link", "two-self-links", "parent-and-self-links"])
def test_directory_link_loops_are_walked_once(tmp_path, reach, links):
    plain = reach._dataset(tmp_path / "plain")
    looped = tmp_path / "looped"
    shutil.copytree(plain, looped)
    for name, target in links.items():
        (looped / "data" / name).symlink_to(target)
    outputs = {}
    for label, root in (("plain", plain), ("looped", looped)):
        out = cli(root, "--commanded-reach", "--units", "--json", timeout=120)  # the fixed walk returns in seconds
        assert out.returncode in (0, 1, 3), out.stderr
        outputs[label] = json.loads(out.stdout)["components"]
    for name in ("commanded_reach", "units"):
        before, after = outputs["plain"][name], outputs["looped"][name]
        assert after["status"] == before["status"]
        assert [(f["code"], f["status"], f["message"]) for f in after["findings"]] == \
            [(f["code"], f["status"], f["message"]) for f in before["findings"]]
    reach_plain, reach_looped = outputs["plain"]["commanded_reach"], outputs["looped"]["commanded_reach"]
    assert reach_looped["coverage"]["rows"] == reach_plain["coverage"]["rows"] > 1000
    assert reach_looped["coverage"]["data_files_read"] == reach_plain["coverage"]["data_files_read"] == 1
    assert reach_looped["joints"] == reach_plain["joints"]


def test_the_walk_is_bounded_and_files_stay_contained(tmp_path, reach, monkeypatch):
    root = reach._dataset(tmp_path / "ds")
    for i in range(3):
        (root / "data" / f"chunk-extra-{i}").mkdir()
    monkeypatch.setattr(reach_module, "MAX_FOLDERS", 2)
    with pytest.raises(InputError, match="More than 2 data directories"):
        reach_module.check_commanded_reach(root)
    monkeypatch.undo()
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copyfile(root / "data" / "chunk-000" / "file-000.parquet", outside / "file-999.parquet")
    (root / "data" / "linked").symlink_to(outside)
    with pytest.raises(InputError, match="escapes"):  # a followed link out of the dataset is still refused
        reach_module.check_commanded_reach(root)


def test_a_looping_dataset_does_not_freeze_the_stdio_server(tmp_path, reach):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = reach._dataset(tmp_path / "ds")
    for name in ("l0", "l1"):
        (root / "data" / name).symlink_to(".")

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-B", "-m", "policy_doctor", "serve", "--root",
                                                                     str(tmp_path)], env=ENV)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                first = await asyncio.wait_for(client.call_tool("preflight_check", {
                    "source": str(root), "commanded_reach": True}), 60)
                assert not first.isError and first.structuredContent["components"]["commanded_reach"]["status"] == "pass"
                second = await asyncio.wait_for(client.call_tool("preflight_check", {"source": str(root)}), 60)
                assert not second.isError
    asyncio.run(asyncio.wait_for(run(), timeout=150))


# ---------------------------------------------------------------- 2. no outside path is echoed

def test_out_of_root_metadata_target_is_not_disclosed(tmp_path):
    roots, outside = tmp_path / "roots", tmp_path / "outside"
    outside.mkdir()
    root = groot_fixtures._dataset(roots / "ds")
    secret = outside / "SECRET_TARGET_modality.json"
    shutil.move(root / "meta" / "modality.json", secret)
    (root / "meta" / "modality.json").symlink_to(secret)
    report = engine.preflight(str(root), read_roots=[str(roots)])
    part = report["components"]["groot_contract"]
    message = part["findings"][0]["message"]
    assert part["status"] == "unknown" and "outside the configured read roots" in message
    assert "repository cache directory" in message
    assert str(outside) not in json.dumps(report) and "SECRET_TARGET" not in json.dumps(report)
    concise = structured(asyncio.run(create_server([str(roots)]).call_tool("preflight_check", {"source": str(root)})))
    assert str(outside) not in json.dumps(concise) and "SECRET_TARGET" not in json.dumps(concise)


# ---------------------------------------------------------------- 3. an automatic contract never refuses a plain preflight

def nonpass_keys(findings):
    return {(f["groot_version"], f["code"], f["status"]) for f in findings if f["status"] != "pass"}


def uncapped_contract_findings(root):
    """Both default profiles before the combined dispatcher's example cap; keep the full-count oracle independent."""
    return [{**finding, "groot_version": version} for version in ("n1.5", "n1.7")
            for finding in check_groot_dataset(root, groot_version=version)["findings"]]


@pytest.mark.parametrize("groups", [200, 2000])
def test_wide_modality_is_bounded_and_never_refused(tmp_path, groups):
    root = groot_fixtures._dataset(tmp_path / "ds", wide_modality(groups))
    out = cli(root, "--json")
    assert out.returncode == 1, out.stderr  # the contract's errors, never exit 2 at the report bound
    report = json.loads(out.stdout)
    part = report["components"]["groot_contract"]
    full = uncapped_contract_findings(root)
    assert len(json.dumps(report).encode()) <= engine.REPORT_LIMIT
    assert len(part["findings"]) <= engine.GROOT_MAX_FINDINGS and part["findings_truncated"] is True
    assert part["schema_version"] == 3 and part["versions_run"] == ["n1.5", "n1.7"]
    assert part["total_findings"] == sum(part["counts"].values()) == len(full) > len(part["findings"])
    # One example of every non-passing code and status is kept first; the remaining slots go to the most severe.
    assert nonpass_keys(part["findings"]) == nonpass_keys(full) and part["codes_without_example"] == []
    assert sum(n for by in part["finding_codes"].values() for n in by.values()) == part["total_findings"]
    fill = [f["status"] for f in part["findings"]]
    for key in nonpass_keys(part["findings"]):
        fill.remove(key[2])
    dropped_errors = part["counts"]["error"] - sum(f["status"] == "error" for f in part["findings"])
    assert dropped_errors == 0 or all(status == "error" for status in fill)  # the fill is the most severe
    assert report["findings_truncated"] is True and report["total_findings"] >= part["total_findings"]
    text = human(report)
    counts = part["counts"]
    assert (f"findings: {len(part['findings'])} of {part['total_findings']} kept in this report" in text
            and f"component counts cover all: {counts['error']} error, {counts['warning']} warning, "
                f"{counts['unknown']} unknown, {counts['pass']} pass" in text)
    assert "by code: " in text
    concise = _summary(report)
    assert concise["components"]["groot_contract"]["findings_total"] == part["total_findings"]
    assert concise["components"]["groot_contract"]["findings_truncated"] is True


def test_capped_contract_keeps_an_example_of_every_code_and_status(tmp_path):
    """Review 2 D2/B1: 60 bad groups plus a malformed tasks.jsonl kept 50 state-slice errors and dropped every
    groot.action_slice, groot.tasks and groot.config example although they were counted (B2: the config action too)."""
    root = groot_fixtures._dataset(tmp_path / "ds", wide_modality(60))
    tasks = root / "meta" / "tasks.jsonl"
    tasks.write_text(tasks.read_text() + "not json\n")
    full = uncapped_contract_findings(root)
    assert {("n1.7", "groot.tasks", "error"), ("n1.7", "groot.action_slice", "error"),
            ("n1.7", "groot.state_slice", "error"), ("n1.7", "groot.config", "unknown")} <= nonpass_keys(full)
    out = cli(root, "--json")
    report = json.loads(out.stdout)
    part = report["components"]["groot_contract"]
    assert out.returncode == 1 and part["findings_truncated"] is True
    assert len(part["findings"]) == engine.GROOT_MAX_FINDINGS < part["total_findings"] == len(full)
    assert nonpass_keys(part["findings"]) == nonpass_keys(full) and part["codes_without_example"] == []
    tally = {}
    for finding in full:
        tally.setdefault(finding["code"], {}).setdefault(finding["status"], 0)
        tally[finding["code"]][finding["status"]] += 1
    assert part["finding_codes"] == tally
    assert part["counts"] == {status: sum(f["status"] == status for f in full) for status in engine.PRIORITY}
    assert any("Supply the GR00T training modality config" in a for a in report["next_actions"])
    text = human(report)
    assert "one example of every code and status first" in text and f"groot.tasks {tally['groot.tasks']['error']} error" in text
    concise = _summary(report)["components"]["groot_contract"]
    assert concise["finding_codes"]["groot.tasks"] == tally["groot.tasks"] and concise["findings_truncated"] is True
    for n in (10, 40):  # B2: the config action survives truncation
        many = engine.preflight(str(groot_fixtures._dataset(tmp_path / f"na{n}", wide_modality(n))))
        assert any("Supply the GR00T training modality config" in a for a in many["next_actions"]), n


def test_bound_keeps_examples_when_bytes_force_fewer_findings(monkeypatch):
    findings = ([{"code": "a", "status": "error", "message": "x" * 400}] * 30
                + [{"code": "a", "status": "warning", "message": "w"},
                   {"code": "b", "status": "warning", "message": "y"}, {"code": "c", "status": "unknown", "message": "z"},
                   {"code": "p", "status": "pass", "message": "ok"}])
    monkeypatch.setattr(engine, "GROOT_MAX_BYTES", 4000)
    part = engine._bound_contract({"findings": list(findings), "counts": {}})
    kept = {(f["code"], f["status"]) for f in part["findings"]}
    assert {("a", "error"), ("a", "warning"), ("b", "warning"), ("c", "unknown")} <= kept and ("p", "pass") not in kept
    assert part["codes_without_example"] == [] and len(json.dumps(part).encode()) <= 4000
    assert part["finding_codes"] == {"a": {"error": 30, "warning": 1}, "b": {"warning": 1}, "c": {"unknown": 1},
                                     "p": {"pass": 1}}
    small = engine._bound_contract({"findings": findings[30:], "counts": {}})
    assert small["findings_truncated"] is False and small["finding_codes"]["b"] == {"warning": 1}


@pytest.mark.parametrize("size", [100_000, 400_000])
def test_one_huge_finding_is_shortened_not_dropped(tmp_path, size):
    """Review 2 D3/B5: a 100k- or 400k-character group name made one finding take the whole budget (400k: omitted,
    --fail-on error passed)."""
    modality = json.loads(json.dumps(groot_fixtures.MODALITY))
    modality["state"]["g" * size] = {"start": 6, "end": 7}
    root = groot_fixtures._dataset(tmp_path / "ds", modality)
    out = cli(root, "--groot-version", "n1.7", "--json")
    report = json.loads(out.stdout)
    part = report["components"]["groot_contract"]
    assert out.returncode == 1 and part["status"] == "error" and not part.get("reduced")
    assert len(json.dumps(part).encode()) < 64 * 1024
    long = next(f for f in part["findings"] if "message" in f.get("text_truncated", []))
    assert long["message"].startswith("state.ggg") and len(long["message_sha256"]) == 64
    assert any(item.endswith("/pointer") for item in long["text_truncated"])
    assert cli(root, "--groot-version", "n1.7", "--fail-on", "error").returncode == 1


def size(value):
    return len(json.dumps(value, allow_nan=False).encode())


def test_size_bound_sweep_keeps_the_error_until_only_unknown_fits(tmp_path, monkeypatch):
    """Review 2 D1 (B3/B4 pad sweeps): as the bound tightens, the automatic contract goes bounded -> reduced (examples
    kept) -> a status stub -> left out. Its ERROR and counts survive every stage, so --fail-on error always fails."""
    root = groot_fixtures._dataset(tmp_path / "ds", wide_modality(400))
    full = engine.preflight(str(root))
    whole = full["components"]["groot_contract"]
    without = size(engine.preflight(str(root), groot_contract=False))
    limits = {size(full) + 1, *range(without - 400, without + 9000, 200),
              *range(without + 9000, size(full), max(1, (size(full) - without) // 30))}
    stages = []
    for limit in sorted(limits, reverse=True):
        monkeypatch.setattr(engine, "REPORT_LIMIT", limit)
        try:
            report = engine.preflight(str(root))
        except InputError as exc:
            assert "768 KiB" in str(exc) and "--max-files" in str(exc)
            stages.append("refused")
            continue
        assert size(report) <= limit
        part = report["components"].get("groot_contract")
        stage = ("left_out" if part is None else "omitted" if part.get("omitted_at_report_bound")
                 else "reduced" if part.get("reduced") else "bounded")
        assert report["status"] == "error"  # every stage keeps the verdict
        if stage in ("bounded", "reduced"):
            assert part["status"] == "error"
            assert part["counts"] == whole["counts"] and part["total_findings"] == whole["total_findings"]
            assert nonpass_keys(part["findings"]) == nonpass_keys(whole["findings"])
        if stage == "omitted":
            assert part["status"] == "error" and part["counts"] == whole["counts"]
            assert part["findings"][0]["status"] == "error" and engine.GROOT_OMITTED in part["findings"][0]["message"]
            assert part["contract_statement"] == engine.GROOT_STUB
        if stage == "left_out":
            entry = report["coverage"]["components_left_out_at_size_bound"]["groot_contract"]
            assert entry["status"] == "error" and entry["counts"] == whole["counts"] and "left out" in entry["reason"]
            assert "left out" in human(report)
        from policy_doctor.cli import main

        if stage in ("omitted", "left_out"):
            monkeypatch.setattr(engine, "preflight", lambda *a, _r=report, **k: copy.deepcopy(_r))
            assert main(["preflight", str(root), "--json", "--fail-on", "error"]) == 1
            monkeypatch.undo()
            monkeypatch.setattr(engine, "REPORT_LIMIT", limit)
        stages.append(stage)
    order = ["bounded", "reduced", "omitted", "left_out", "refused"]
    assert [order.index(s) for s in stages] == sorted(order.index(s) for s in stages)  # monotonic as it tightens
    assert set(order) <= set(stages), stages
    for text in (engine.GROOT_REDUCED, engine.GROOT_OMITTED):
        assert "--groot-config" not in text and "--max-files" in text  # the remedy works at these sizes
    reduced_limit = next(limit for limit, stage in zip(sorted(limits, reverse=True), stages, strict=True) if stage == "reduced")
    monkeypatch.setattr(engine, "REPORT_LIMIT", reduced_limit)
    report = engine.preflight(str(root))
    assert report["components"]["groot_contract"]["contract_statement"] == engine.GROOT_REDUCED
    assert any("smaller --max-files" in a for a in report["next_actions"] if "modality config" in a)
    text = human(report)
    assert "GR00T dataset contract: error" in text and "Reduced form" in text and "by code: " in text
    concise = _summary(report)["components"]["groot_contract"]
    assert concise["status"] == "error" and concise["finding_codes"] and _summary(report)["components"]
    monkeypatch.setattr(engine, "REPORT_LIMIT", without + 2000)  # the automatic contract would be reduced here
    with pytest.raises(InputError, match="768 KiB.*--max-files"):  # a requested check is refused with the remedy
        engine.preflight(str(root), groot_modality_config=str(groot_fixtures._config(tmp_path)))


def test_fail_on_honours_component_statuses(tmp_path, monkeypatch, capsys):
    from policy_doctor.cli import main

    root = groot_fixtures._dataset(tmp_path / "ds", wide_modality(3))
    report = engine.preflight(str(root))
    assert report["components"]["groot_contract"]["status"] == "error"
    report["findings"] = [f for f in report["findings"] if f["status"] not in ("error", "unknown")]  # none listed
    monkeypatch.setattr(engine, "preflight", lambda *a, **k: copy.deepcopy(report))
    assert main(["preflight", str(root), "--json", "--fail-on", "error"]) == 1
    assert main(["preflight", str(root), "--json", "--fail-on", "unknown"]) == 1
    report["components"] = {name: dict(part, status="pass") for name, part in report["components"].items()}
    report["status"] = "pass"
    assert main(["preflight", str(root), "--json", "--fail-on", "error"]) == 0
    capsys.readouterr()


def test_opt_out_skips_the_automatic_contract(tmp_path):
    root = groot_fixtures._dataset(tmp_path / "ds")
    config = groot_fixtures._config(tmp_path)
    report = engine.preflight(str(root), groot_contract=False)
    assert "groot_contract" not in report["components"]
    assert report["coverage"]["groot_contract_skipped"].startswith("groot_contract=false")
    with pytest.raises(InputError, match="cannot be combined"):
        engine.preflight(str(root), groot_contract=False, groot_modality_config=str(config))
    for value in (0, "false", None):
        with pytest.raises(InputError, match="strict booleans"):
            engine.preflight(str(root), groot_contract=value)
    out = cli(root, "--no-groot-contract", "--json")
    assert "groot_contract" not in json.loads(out.stdout)["components"]
    out = cli(root, "--no-groot-contract", "--groot-config", config)
    assert out.returncode == 2 and "cannot be combined" in out.stderr
    help_text = " ".join(cli("--help").stdout.split()).replace("- ", "-")
    assert "--no-groot-contract Skip the GR00T dataset contract that otherwise runs automatically" in help_text


def test_mcp_opt_out_schema_and_applies_to_in_concise(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = groot_fixtures._dataset(tmp_path / "ds", version="v3.0")  # automatic contract: groot.format error

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-B", "-m", "policy_doctor", "serve", "--root",
                                                                     str(tmp_path)], env=ENV)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                schema = next(t for t in tools if t.name == "preflight_check").inputSchema["properties"]["groot_contract"]
                assert schema["type"] == "boolean" and schema["default"] is True
                body = (await client.call_tool("preflight_check", {"source": str(root)})).structuredContent
                assert len(json.dumps(body, allow_nan=False).encode()) <= SUMMARY_LIMIT
                groot = [f for f in body["findings_shown"] if f["code"].startswith("groot.")]
                assert groot and all(f["applies_to"] for f in groot)
                assert any(f["code"] == "groot.format" and f["applies_to"] == "fine-tuning and direct loaders"
                           for f in groot)
                skipped = (await client.call_tool("preflight_check", {"source": str(root), "groot_contract": False}))
                assert not skipped.isError and "groot_contract" not in skipped.structuredContent["components"]
                rejected = await client.call_tool("preflight_check", {"source": str(root), "groot_contract": "no"})
                assert rejected.isError and "strict booleans" in rejected.content[0].text
    asyncio.run(asyncio.wait_for(run(), timeout=120))
