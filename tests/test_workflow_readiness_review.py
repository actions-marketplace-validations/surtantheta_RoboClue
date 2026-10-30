"""Independent adversarial review of plans, read authority and concise evidence."""
import asyncio
import copy
import hashlib
import json
import os
import signal

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from policy_doctor import big_bets_frontdoor as frontdoor
from policy_doctor import workflow_readiness as readiness
from policy_doctor.artifacts import InputError
from policy_doctor.cli import parser
from policy_doctor.server import create_server


def call(server, arguments):
    return asyncio.run(server.call_tool("choose_workflow", arguments))[1]


@pytest.mark.parametrize("argument", ["contract", "capture", "recording", "manifest", "original_recording", "converter_config"])
def test_every_explicit_file_selector_is_authorized_even_when_irrelevant_to_goal(tmp_path, argument):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"kind":"native_upgrade_impact"}')
    with pytest.raises(ToolError, match="outside"):
        call(create_server([str(root)]), {"goal": "training", argument: str(outside)})


@pytest.mark.parametrize("argument", ["contract", "capture", "recording", "manifest", "original_recording", "converter_config"])
def test_every_file_selector_refuses_fifo_before_reading(tmp_path, argument):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    signal.alarm(3)
    try:
        with pytest.raises(InputError, match="special"):
            readiness.choose_workflow("training", **{argument: str(fifo)})
        with pytest.raises(ToolError, match="special|regular"):
            call(create_server([str(tmp_path)]), {"goal": "training", argument: str(fifo)})
    finally:
        signal.alarm(0)


@pytest.mark.parametrize("ancestor", [False, True])
def test_mcp_preserves_cli_materialized_input_rule_for_symlinks(tmp_path, ancestor):
    target = tmp_path / "real"
    target.mkdir()
    recording = target / "recording.jsonl"
    recording.write_text("unvalidated saved software evidence")
    alias = tmp_path / "alias"
    alias.symlink_to(target if ancestor else recording)
    selected = alias / recording.name if ancestor else alias
    with pytest.raises(InputError, match="symlink"):
        readiness.choose_workflow("runtime", recording=str(selected))
    with pytest.raises(ToolError, match="symlink"):
        call(create_server([str(tmp_path)]), {"goal": "runtime", "recording": str(selected)})


def test_symlink_to_outside_root_never_becomes_plan_authority(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"kind":"native_upgrade_impact"}')
    alias = root / "manifest.json"
    alias.symlink_to(outside)
    with pytest.raises(ToolError, match="outside|symlink"):
        call(create_server([str(root)]), {"goal": "upgrade", "manifest": str(alias)})


def test_upgrade_plan_reads_only_authorized_top_manifest_not_nested_fifo(tmp_path, monkeypatch):
    nested = tmp_path / "nested-fifo"
    os.mkfifo(nested)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"kind": "native_upgrade_impact", "contract": {"path": str(nested)},
                                    "assets": [{"candidate": {"capture": {"path": "/outside/unopened"}}}]}))
    original = readiness._read_small
    reads = []

    def tracked(path, **kwargs):
        reads.append(str(path))
        return original(path, **kwargs)

    monkeypatch.setattr(readiness, "_read_small", tracked)
    signal.alarm(3)
    try:
        result = call(create_server([str(tmp_path)]), {"goal": "upgrade", "manifest": str(manifest)})
    finally:
        signal.alarm(0)
    assert reads == [str(manifest)]
    assert result["assessment_status"] == "not_run"
    assert all(not row["executed"] for row in result["can_do_now"])
    assert "File presence does not validate" in result["limits"]


@pytest.mark.parametrize("raw", [
    b'{"kind":"owned_upgrade_watch","kind":"native_upgrade_impact"}',
    b'{"kind":"owned_upgrade_watch","assets":[NaN]}',
    b'{"kind":"owned_upgrade_watch","assets":[1e999]}',
    b'[]', b'{broken', b'\xff',
])
def test_manifest_duplicate_nonfinite_nonobject_and_invalid_bytes_refused(tmp_path, raw):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(raw)
    with pytest.raises(InputError):
        readiness.choose_workflow("upgrade", manifest=str(manifest))


def test_oversized_manifest_refused_without_inspecting_nested_data(tmp_path):
    manifest = tmp_path / "large.json"
    manifest.write_bytes(b' ' * (4 * 1024 * 1024 + 1))
    with pytest.raises(InputError, match="byte bound"):
        readiness.choose_workflow("upgrade", manifest=str(manifest))


def test_invalid_unicode_path_is_an_input_error():
    with pytest.raises(InputError):
        readiness.choose_workflow("runtime", recording="\ud800")


def test_public_conversion_recommendation_requests_cli_fetch_explicitly(tmp_path):
    original = tmp_path / "source.hdf5"
    original.write_bytes(b"not opened by planning")
    config = tmp_path / "recipe.yaml"
    config.write_text("not parsed by planning")
    source = tmp_path / "converter"
    source.mkdir()
    result = readiness.choose_workflow("conversion", dataset="hf://example/generated",
        original_recording=str(original), converter_source=str(source), converter_config=str(config))
    action, = [row for row in result["can_do_now"] if row.get("mcp_request", {}).get("name") == "preflight_check"]
    parsed = parser().parse_args(action["local_cli_argv"][1:])
    assert parsed.source == "hf://example/generated"
    assert parsed.fetch is True
    assert result["public_access"]["fetch_performed"] is False


def health_card_with_truncated_evidence():
    findings = [{"code": "generated.check", "status": "error", "message": "Retained contradiction"},
                {"code": "generated.coverage", "status": "unknown", "message": "Missing evidence"}]
    return {"kind": "dataset_health_card", "status": "error", "scope": "Generated supported checks",
            "request": {}, "report": {"kind": "dataset_preflight", "status": "error", "findings": findings,
                "total_findings": 17, "findings_truncated": True, "counts_scope": "available_findings",
                "counts": {"error": 1, "unknown": 1, "warning": 0, "pass": 0},
                "components": {"numeric_rows": {"status": "error", "findings": copy.deepcopy(findings),
                    "total_findings": 17, "findings_truncated": True}}}}


def test_summary_omission_count_includes_already_truncated_source_findings():
    report = health_card_with_truncated_evidence()
    original = copy.deepcopy(report)
    summary = frontdoor.summarize(report)
    assert report == original
    assert summary["status"] == "error"
    assert summary["counts"] == report["report"]["counts"]
    assert summary["findings_total"] == 17
    assert summary["findings_omitted"] == 15
    assert summary.get("counts_scope") == "available_findings"
    assert summary["full_report_sha256"] == hashlib.sha256(frontdoor._json(report).encode()).hexdigest()


def test_user_view_size_fallback_preserves_verdict_hash_and_physical_unknown(monkeypatch):
    report = {"kind": "flight_recording_triage", "status": "warning", "counts": {"warning": 1, "unknown": 2, "pass": 3},
              "findings": [{"code": "tracking", "status": "warning", "message": "Software difference"}]}
    view = {"overall_status": "warning", "software_comparisons": ["x" * 40000],
            "physical_feedback_association": {"status": "unknown", "reasons": ["No physical acquisition evidence"]}}
    monkeypatch.setattr(frontdoor.first_use_views, "flight_view", lambda _: view)
    summary = frontdoor.summarize(report)
    assert len(frontdoor._json(summary).encode()) <= frontdoor.SUMMARY_LIMIT
    assert summary["status"] == "warning" and summary["counts"] == report["counts"]
    assert summary["user_view"]["physical_feedback_association"]["status"] == "unknown"
    assert summary["user_view"]["details_omitted"] is True
    assert summary["user_view"]["full_view_sha256"] == hashlib.sha256(frontdoor._json(view).encode()).hexdigest()


@pytest.mark.parametrize("arguments", [
    {"goal": "runtime", "recording": True}, {"goal": "runtime", "recording": {}},
    {"goal": "training", "criterion_sha256": 3}, {"goal": "training", "framework": ["groot"]},
    {"goal": "dataset", "dataset": "https://example.org/export"},
    {"goal": "dataset", "dataset": "hf://example/generated/nested"},
    {"goal": "dataset", "allow_hub_datasets": True},
])
def test_mcp_malformed_arguments_never_coerce_into_a_ready_plan(tmp_path, arguments):
    with pytest.raises(ToolError):
        call(create_server([str(tmp_path)]), arguments)


@pytest.mark.parametrize("public", [False, True])
def test_recommended_mcp_actions_fit_actual_discovered_tool_schemas(tmp_path, public):
    supplied = tmp_path / "supplied.json"
    supplied.write_text('{"kind":"native_upgrade_impact"}')
    inputs = {"dataset": "hf://example/generated" if public else str(tmp_path),
              "contract": str(supplied), "capture": str(supplied), "recording": str(supplied),
              "manifest": str(supplied), "original_recording": str(supplied),
              "converter_source": str(tmp_path), "converter_config": str(supplied),
              "criterion_sha256": "a" * 64}
    server = create_server([str(tmp_path)])
    schemas = {tool.name: tool.inputSchema for tool in asyncio.run(server.list_tools())}
    for goal in readiness.GOALS:
        result = readiness.choose_workflow(goal, **inputs)
        assert result["assessment_status"] == "not_run"
        for action in result["can_do_now"]:
            request = action.get("mcp_request")
            if request is None:
                continue
            schema = schemas[request["name"]]
            assert set(request["arguments"]) <= set(schema["properties"])
            assert set(schema.get("required", [])) <= set(request["arguments"])
            assert action["executed"] is False
