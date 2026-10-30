"""Real CLI engines and FastMCP argument/authorization boundaries; no robot runs."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from policy_doctor import big_bets_frontdoor as fd
from policy_doctor import upgrade_impact as ui
from policy_doctor.artifacts import InputError
from policy_doctor.flight_recorder import FlightRecorder
from policy_doctor.health_cards import verify_health_card
from policy_doctor.training_guard import TargetContract
from policy_doctor.training_guard_capture import save_capture


def digest(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: {"dtype": "float32", "shape": [1], "names": ["a"]}
                for key in ("action", "observation.state")}
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30,
        "total_frames": 3, "total_episodes": 1, "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}))
    (root / "meta/stats.json").write_text(json.dumps({key: {"mean": [1.], "std": [(2 / 3) ** .5], "min": [0.], "max": [2.]}
                                                       for key in features}))
    pq.write_table(pa.Table.from_pylist([{"index": i, "episode_index": 0, "frame_index": i,
        "timestamp": i / 30, "action": [float(i)], "observation.state": [float(i)]} for i in range(3)]),
        root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, "length": 3, "dataset_from_index": 0,
        "dataset_to_index": 3, "data/chunk_index": 0, "data/file_index": 0}]),
        root / "meta/episodes/chunk-000/file-000.parquet")
    return root


@pytest.fixture
def targets(tmp_path):
    root = tmp_path / "targets"
    root.mkdir()
    batch = {"inputs": {"action": np.array([[[.5], [.6]]], dtype=np.float32),
                         "action_mask": np.ones((1, 2, 1), dtype=np.float32)}}
    keys = ["original:episode0:row1"]
    contract = TargetContract.from_references([(batch, keys, [{"original_file": "recording.hdf5", "row": 1}])],
                                               intent={"action": "preserve selected source target"})
    contract_path = root / "contract.json"
    contract.save(contract_path)
    result = {"contract": contract_path, "pin": contract.criterion_sha256}
    for name in ("pass", "error", "unknown"):
        observed = copy.deepcopy(batch)
        if name == "error":
            observed["inputs"]["action"][0, 1, 0] += 1
        path = root / f"{name}.json"
        save_capture(observed, ["uncovered"] if name == "unknown" else keys, path)
        result[name] = path
    return result


@pytest.fixture
def flight(tmp_path):
    path = tmp_path / "flight.jsonl"
    with FlightRecorder(path, clocks={"host": {"unit": "s", "domain": "test_epoch"}},
                        provenance={"kind": "controlled_software_test"}) as recorder:
        for i in range(2):
            recorder.command(f"c{i}", timestamp=i * .1, clock="host", stream="policy",
                             joint_names=["a"], units=["rad"], values=[.5], semantics="position")
    return path


@pytest.fixture
def upgrade(tmp_path, monkeypatch):
    root = tmp_path / "upgrade"
    root.mkdir()
    source = root / "native.py"
    source.write_text("# controlled source identity; never executed\n")
    monkeypatch.setattr(ui, "ISAACLAB_SOURCES", {"test-native": {"loader.py": digest(source)["sha256"]}})
    config = root / "recipe.json"
    config.write_text('{"mode":"XYZW"}')
    fields = {"/data/demo_0/root_pose": np.array([[0., 0., 0., 1.]], dtype=np.float32)}
    contract = ui.BoundaryContract.from_reference(fields, intent={"orientation": "XYZW"})
    contract_path = root / "contract.json"
    contract.save(contract_path)
    captures = {}
    for name in ("good", "bad"):
        path = root / f"{name}.json"
        ui.save_boundary_capture(fields if name == "good" else {"/data/demo_0/root_pose": np.array([[1., 0., 0., 0.]], dtype=np.float32)}, path)
        captures[name] = path

    def snapshot(path):
        return {"consumer": "isaaclab-hdf5-xyzw", "revision": "test-native", "source_files": {"loader.py": digest(source)},
                "configs": {"recipe": digest(config)}, "capture": digest(path)}

    manifest = {"schema_version": 1, "kind": "native_upgrade_impact", "contract": digest(contract_path),
                "assets": [{"asset_id": "my-asset", "reference": snapshot(captures["good"]),
                            "candidate": snapshot(captures["bad"])}]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def invoke(*args):
    return subprocess.run([sys.executable, "-m", "policy_doctor.big_bets_frontdoor", *map(str, args)],
                          capture_output=True, text=True, timeout=45)


def run_main(capsys, *args):
    code = fd.main(list(map(str, args)))
    output = capsys.readouterr()
    return code, output


def test_cli_health_full_json_new_output_and_opt_in_bound_badge(dataset, tmp_path):
    output, artifacts = tmp_path / "card.json", tmp_path / "artifacts"
    run = invoke("health-card", dataset, "--consumer", "private-ci", "--json", "--output", output,
                 "--artifacts", artifacts)
    assert run.returncode == 0, run.stderr
    card = json.loads(run.stdout)
    assert card["status"] == "pass" and verify_health_card(card)
    assert card == json.loads(output.read_text()) == json.loads((artifacts / "health-card.json").read_text())
    assert "bounded checks" in (artifacts / "badge.svg").read_text()
    assert not (dataset / "badge.svg").exists()
    before = output.read_bytes()
    refused = invoke("health-card", dataset, "--output", output)
    assert refused.returncode == 2 and "new path" in refused.stderr
    assert output.read_bytes() == before


def test_cli_health_unknown_and_actual_error_are_distinct(dataset, tmp_path):
    partial = invoke("health-card", dataset, "--max-rows", 1, "--json")
    assert partial.returncode == 3 and json.loads(partial.stdout)["status"] == "unknown"
    data = dataset / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(data).to_pylist()
    rows[0]["action"] = [float("nan")]
    pq.write_table(pa.Table.from_pylist(rows), data)
    broken = invoke("health-card", dataset, "--max-rows", 1, "--json")
    assert broken.returncode == 1 and json.loads(broken.stdout)["status"] == "error"
    assert not list(tmp_path.glob("badge.svg"))


@pytest.mark.parametrize("status,code", [("pass", 0), ("error", 1), ("unknown", 3)])
def test_cli_training_real_reports_first_witness_and_exit(targets, tmp_path, status, code):
    path = tmp_path / f"report-{status}.json"
    run = invoke("check-training-targets", "--contract", targets["contract"], "--criterion-sha256", targets["pin"],
                 "--capture", targets[status], "--output", path, "--json")
    assert run.returncode == code, run.stderr
    report = json.loads(run.stdout)
    assert report == json.loads(path.read_text()) and report["status"] == status
    if status == "error":
        assert report["first_difference"]["source_key"] == "original:episode0:row1"
        assert report["first_difference"]["coordinate"] == [0, 1, 0]


def test_cli_flight_summary_retains_unknown_and_healthy_timing(flight):
    run = invoke("flight-triage", flight, "--max-command-gap-s", .2)
    assert run.returncode == 3, run.stderr
    assert "UNKNOWN" in run.stdout and "UNKNOWN" in run.stdout.splitlines()[-1]
    full = invoke("flight-triage", flight, "--max-command-gap-s", .2, "--json")
    assert json.loads(full.stdout)["counts"]["pass"] == 1


def test_cli_upgrade_observed_regression_artifacts_and_corrected_control(upgrade, tmp_path, capsys):
    path, manifest = upgrade
    artifacts = tmp_path / "upgrade-alerts"
    code, output = run_main(capsys, "upgrade-impact", path, "--json", "--artifacts", artifacts)
    report = json.loads(output.out)
    assert code == 1 and report["status"] == "changed"
    assert report["criterion_intent"] == {"orientation": "XYZW"}
    assert report["assets"][0]["code"] == "retained_boundary_criterion_violated"
    assert "causal_attribution" in report["comparison_basis"]
    assert "my-asset" in (artifacts / "alerts.md").read_text()
    assert report == json.loads((artifacts / "impact.json").read_text())
    manifest["assets"][0]["candidate"] = manifest["assets"][0]["reference"]
    path.write_text(json.dumps(manifest))
    code, output = run_main(capsys, "upgrade-impact", path, "--json")
    assert code == 0 and json.loads(output.out)["status"] == "healthy"
    manifest["assets"][0]["candidate"] = {**manifest["assets"][0]["candidate"], "revision": "unqualified"}
    path.write_text(json.dumps(manifest))
    code, output = run_main(capsys, "upgrade-impact", path, "--json")
    assert code == 3 and json.loads(output.out)["status"] == "unknown"


@pytest.mark.parametrize("config", [{"max_rows": True}, {"max_rows": 0}, {"video_headers": "false"},
                                    {"units": 1}, {"max_files": 1025}, {"config": {"arbitrary": "path"}}])
def test_cli_invalid_config_is_input_error_before_checker(tmp_path, monkeypatch, capsys, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(fd.health_cards, "health_card", lambda *a, **k: pytest.fail("checker reached"))
    code, output = run_main(capsys, "health-card", "missing", "--config", path, "--json")
    assert code == 2 and json.loads(output.err)["error"]["code"] == "invalid_input"


def test_cli_special_destinations_refused_before_reads(tmp_path, monkeypatch, capsys):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    monkeypatch.setattr(fd.health_cards, "health_card", lambda *a, **k: pytest.fail("checker reached"))
    for output in (fifo, link, tmp_path / "missing" / "out.json"):
        code, captured = run_main(capsys, "health-card", "missing", "--output", output)
        assert code == 2 and "new path" in captured.err


@pytest.fixture
def mcp_server(tmp_path):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    def make(*, roots=None, allow_hub=False):
        allowed = [Path(path).resolve() for path in (roots or [tmp_path])]
        checked = []

        def guard(path):
            candidate = Path(path).resolve()
            checked.append(str(candidate))
            if not any(candidate.is_relative_to(root) for root in allowed):
                raise InputError("outside configured read roots")
            from policy_doctor.fs_guard import refuse_special_file

            refuse_special_file(candidate, "artifact input")
            if not candidate.exists():
                raise InputError("missing input")
            return str(candidate)

        server = FastMCP("Four private bounded evidence tools")
        fd.register_mcp_tools(server, guard=guard, allowed=allowed, allow_hub_datasets=allow_hub,
            annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
        return server, checked
    return make


def mcp_call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    # FastMCP returns content plus structured output in current SDK versions.
    if isinstance(result, tuple):
        content, structured = result
        if structured:
            return structured
    else:
        content = result
    return json.loads(next(item.text for item in content if getattr(item, "type", None) == "text"))


def test_real_fastmcp_schema_readonly_no_execution_config_or_outputs(mcp_server):
    server, _ = mcp_server()
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"choose_workflow", "dataset_health_card", "triage_flight_recording", "assess_upgrade_impact", "check_training_targets"}
    for tool in tools:
        assert tool.annotations.readOnlyHint and not tool.annotations.destructiveHint
        assert not tool.annotations.openWorldHint
        assert tool.inputSchema["additionalProperties"] is False
    health = next(tool for tool in tools if tool.name == "dataset_health_card")
    assert health.inputSchema["properties"]["fetch"]["type"] == "boolean"
    assert not {"config", "deployment", "checkpoint", "output", "robot_command", "readiness"} & health.inputSchema["properties"].keys()
    training = next(tool for tool in tools if tool.name == "check_training_targets")
    assert set(training.inputSchema["properties"]) == {"contract", "criterion_sha256", "capture", "response_format"}


def test_real_fastmcp_all_four_engines_and_concise_witnesses(mcp_server, dataset, flight, targets, upgrade):
    server, checked = mcp_server()
    card = mcp_call(server, "dataset_health_card", {"source": str(dataset)})
    assert card["kind"] == "dataset_health_card" and card["status"] == "pass"
    assert card["components"]["numeric_rows"]["coverage"]["rows_scanned"] == 3
    assert card["local_cli_argv"][:3] == ["roboclue", "health-card", str(dataset)]
    assert "--artifacts" in card["local_cli_argv"]
    assert str(dataset / "meta/info.json") in checked
    timing = mcp_call(server, "triage_flight_recording", {"recording": str(flight), "max_command_gap_s": .2})
    assert timing["kind"] == "flight_recording_triage" and timing["status"] == "unknown"
    regression = mcp_call(server, "assess_upgrade_impact", {"manifest": str(upgrade[0])})
    assert regression["kind"] == "upgrade_impact" and regression["status"] == "changed"
    assert regression["criterion_intent"] == regression["assets"][0]["criterion_intent"] == {"orientation": "XYZW"}
    assert regression["comparison_basis"] == regression["assets"][0]["comparison_basis"]
    assert regression["local_cli_argv"][:3] == ["roboclue", "upgrade-impact", str(upgrade[0])]
    assert regression["assets"][0]["first_source_witness"]["field"] == "/data/demo_0/root_pose"
    target = mcp_call(server, "check_training_targets", {"contract": str(targets["contract"]),
        "criterion_sha256": targets["pin"], "capture": str(targets["error"])})
    assert target["status"] == "error"
    assert target["first_difference"]["source_key"] == "original:episode0:row1"
    for report in (card, timing, regression, target):
        assert len(json.dumps(report).encode()) <= fd.SUMMARY_LIMIT
        assert "physical success" in report["limits"]


@pytest.mark.parametrize("field,value", [("fetch", "false"), ("fetch", 1), ("max_rows", True), ("max_rows", 0),
                                         ("max_files", 1025), ("video_headers", 0), ("groot_contract", "true"),
                                         ("units", "false"), ("config", {"checkpoint": "/outside"})])
def test_real_fastmcp_strict_flags_bounds_and_arbitrary_config_refused(mcp_server, dataset, monkeypatch, field, value):
    server, _ = mcp_server()
    monkeypatch.setattr(fd.health_cards, "health_card", lambda *a, **k: pytest.fail("checker reached"))
    with pytest.raises(ToolError):
        mcp_call(server, "dataset_health_card", {"source": str(dataset), field: value})


def test_mcp_health_alias_and_nested_links_authorized_before_checker(mcp_server, dataset, tmp_path, monkeypatch):
    server, checked = mcp_server(roots=[dataset / "meta"])
    monkeypatch.setattr(fd.health_cards, "health_card", lambda *a, **k: pytest.fail("checker reached"))
    with pytest.raises(Exception, match="outside"):
        mcp_call(server, "dataset_health_card", {"source": str(dataset / "meta/info.json")})
    server, checked = mcp_server(roots=[dataset])
    outside = tmp_path / "outside.json"
    outside.write_text("sensitive outside artifact")
    (dataset / "meta/external.json").symlink_to(outside)
    with pytest.raises(Exception, match="outside"):
        mcp_call(server, "dataset_health_card", {"source": str(dataset)})
    assert str(outside) in checked


@pytest.mark.parametrize("location", ["contract", "source_files", "configs", "capture"])
def test_mcp_every_manifest_resource_authorized_before_engine(mcp_server, upgrade, tmp_path, monkeypatch, location):
    path, manifest = upgrade
    outside = tmp_path / "outside.json"
    outside.write_text("outside artifact never read")
    replacement = {"path": str(outside), "sha256": "0" * 64}
    if location == "contract":
        manifest["contract"] = replacement
    elif location == "capture":
        manifest["assets"][0]["candidate"][location] = replacement
    else:
        manifest["assets"][0]["candidate"][location] = {"outside": replacement}
    path.write_text(json.dumps(manifest))
    server, checked = mcp_server(roots=[path.parent])
    monkeypatch.setattr(fd.upgrade_impact, "assess_upgrade_impact", lambda *a, **k: pytest.fail("engine reached"))
    with pytest.raises(Exception, match="outside"):
        mcp_call(server, "assess_upgrade_impact", {"manifest": str(path)})
    assert str(outside) in checked


def test_mcp_training_authorizes_both_before_contract_read(mcp_server, targets, tmp_path, monkeypatch):
    server, _ = mcp_server(roots=[targets["contract"].parent])
    outside = tmp_path / "outside-capture"
    outside.write_text("not a capture")
    monkeypatch.setattr(fd.training_guard_capture, "check_capture", lambda *a, **k: pytest.fail("engine reached"))
    with pytest.raises(Exception, match="outside"):
        mcp_call(server, "check_training_targets", {"contract": str(targets["contract"]),
            "criterion_sha256": targets["pin"], "capture": str(outside)})


@pytest.mark.parametrize("operator,fetch,permitted", [(False, True, False), (True, False, False), (True, True, True)])
def test_mcp_public_fetch_requires_operator_and_request(mcp_server, monkeypatch, operator, fetch, permitted):
    server, _ = mcp_server(allow_hub=operator)
    called = []

    def checker(source, **kwargs):
        called.append((source, kwargs))
        return {"kind": "dataset_health_card", "status": "unknown", "report": {}, "request": {}, "limitations": "bounded public intake"}

    monkeypatch.setattr(fd.health_cards, "health_card", checker)
    if permitted:
        report = mcp_call(server, "dataset_health_card", {"source": "hf://org/data", "fetch": fetch})
        assert report["status"] == "unknown" and called[0][1]["fetch"] is True
    else:
        with pytest.raises(Exception, match="operator"):
            mcp_call(server, "dataset_health_card", {"source": "hf://org/data", "fetch": fetch})
        assert called == []


def test_summary_preserves_severe_error_after_many_unknowns_and_coverage():
    findings = [{"status": "unknown", "code": "uncovered", "message": "missing"}] * 100
    findings.append({"status": "error", "code": "bad.source", "evidence": [{"path": "original.hdf5", "pointer": "/data/demo0/actions/1"}],
                     "remedy": "Restore intended original source and rerun the same check."})
    report = {"kind": "dataset_health_card", "status": "error", "report": {"findings": findings,
        "components": {"numeric_rows": {"status": "error", "coverage": {"complete": False, "rows_scanned": 3}}}}}
    summary = fd.summarize(report)
    assert summary["status"] == "error" and summary["counts"]["error"] == 1
    assert summary["findings_shown"][0]["code"] == "bad.source"
    assert summary["findings_shown"][0]["evidence"][0]["path"] == "original.hdf5"
    assert summary["components"]["numeric_rows"]["coverage"]["complete"] is False
    assert summary["findings_omitted"] == 97


def test_mcp_detailed_is_full_valid_health_card(mcp_server, dataset):
    server, _ = mcp_server()
    report = mcp_call(server, "dataset_health_card", {"source": str(dataset), "response_format": "detailed"})
    assert verify_health_card(report) and "summary" not in report


def test_large_upgrade_summary_preserves_all_verdicts_first_source_intent_and_basis():
    intent = {"orientation": "Caller-adapted canonical all-XYZW reference; raw boundary may use field-specific conventions."}
    basis = "Retained criterion comparison; no upstream causal bug claim."
    assets = []
    for i in range(32):
        assets.append({"asset_id": str(i) + "x" * 500, "status": "changed", "code": "retained_boundary_criterion_violated",
            "coverage": {"status": "covered", "notes": ["coverage " + "a" * 600] * 32},
            "candidate_check": {"status": "error", "first_difference": {"field": "inputs.action",
                "source_key": "original:episode0:row1", "coordinate": [0, 1, 0], "expected": .5, "observed": 1.5}},
            "reference_check": {"status": "pass"}, "native_boundary_delta": {"status": "different"},
            "associated_config_changes": [{"name": "config " + "b" * 500}] * 32,
            "criterion_intent": intent, "comparison_basis": basis})
    report = {"kind": "upgrade_impact", "status": "changed", "assets": assets,
              "criterion_intent": intent, "comparison_basis": basis,
              "affected_asset_ids": [asset["asset_id"] for asset in assets], "counts": {"changed": 32, "healthy": 0, "unknown": 0}}
    summary = fd.summarize(report)
    assert len(json.dumps(summary).encode()) <= fd.SUMMARY_LIMIT
    assert len(summary["assets"]) == 32 and all(asset["status"] == "changed" for asset in summary["assets"])
    assert summary["criterion_intent"] == intent and summary["comparison_basis"] == basis
    assert all(asset["criterion_intent"] in (intent, {"same_as_report": "/criterion_intent"}) for asset in summary["assets"])
    witnesses = [asset.get("first_source_witness") for asset in summary["assets"]]
    witness = next((item for item in witnesses if item), summary.get("first_source_witness"))
    assert witness["source_key"] == "original:episode0:row1"


@pytest.mark.parametrize("value", [True, 0, -1, "0.1"])
def test_mcp_flight_strict_seconds_refused(mcp_server, flight, value):
    server, _ = mcp_server()
    with pytest.raises(ToolError):
        mcp_call(server, "triage_flight_recording", {"recording": str(flight), "max_latency_s": value})


def test_mcp_upgrade_relative_references_use_manifest_directory(mcp_server, upgrade):
    path, manifest = upgrade
    for reference in [manifest["contract"], *(item for asset in manifest["assets"]
        for side in ("reference", "candidate") for item in [*asset[side]["source_files"].values(),
            *asset[side]["configs"].values(), asset[side]["capture"]])]:
        reference["path"] = str(Path(reference["path"]).relative_to(path.parent))
    path.write_text(json.dumps(manifest))
    server, checked = mcp_server(roots=[path.parent])
    report = mcp_call(server, "assess_upgrade_impact", {"manifest": str(path)})
    assert report["status"] == "changed" and str(path.parent / "native.py") in checked
