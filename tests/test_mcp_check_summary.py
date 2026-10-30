"""Concise MCP reports retain verdicts, provenance and an explicit drill-down."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.check_report import concise_check, concise_check_summary

SHA = "1" * 40
DS_SHA = "2" * 40


def report():
    return {"schema_version": 1, "kind": "checkpoint_check", "status": "error",
        "counts": {"error": 2, "warning": 0, "unknown": 3, "pass": 1},
        "scope": "Declared metadata only; deployment was not inspected.",
        "assessment_scope": "metadata", "deployment_status": "not_assessed",
        "unassessed_findings": [{"code": "deployment.missing", "status": "unknown"}],
        "findings": [{"code": "dataset.feature_shape", "status": status,
            "message": "Observed shape declaration differs; inspect the exact evidence.",
            "evidence": [{"path": f"hf://org/data@{DS_SHA}/meta/info.json", "sha256": "a" * 64,
                "pointer": "/features/action", "value": {"shape": [2], "values": list(range(1000))}}]}
             for status in ["unknown", "error", "pass", "error", "unknown", "unknown"]],
        "provenance": [{"repo_type": "model", "source": "hf://org/model", "resolved_revision": SHA},
                       {"repo_type": "dataset", "source": "hf://org/data", "resolved_revision": DS_SHA}]}


def summarize(result):
    return concise_check_summary(result, source="hf://org/model", dataset="hf://org/data", scope="metadata")


def test_summary_preserves_full_verdict_counts_and_bounded_original_witness():
    full = report()
    summary = summarize(full)
    assert summary["status"] == full["status"] and summary["counts"] == full["counts"]
    assert len(json.dumps(summary).encode("utf-8")) <= 8192
    assert summary["shown_findings"] == 3 and summary["omitted_findings"] == 3
    assert len(summary["findings"]) == 3
    assert [f["status"] for f in summary["findings"]] == ["error", "error", "unknown"]
    witness = summary["findings"][0]["evidence"][0]
    assert witness["sha256"] == "a" * 64 and witness["pointer"] == "/features/action"
    assert witness["path"] == f"hf://org/data@{DS_SHA}/meta/info.json"
    assert witness["value_preview"]["truncated"]
    assert summary["detail_complete"] is False and summary["unassessed_findings_count"] == 1


def test_digest_binds_original_report_and_replay_pins_resolved_revisions():
    full = report()
    summary = summarize(full)
    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    assert summary["full_report"]["sha256"] == digest
    request = summary["detailed_request"]["arguments"]
    assert request == {"source": "hf://org/model", "dataset": "hf://org/data", "revision": SHA,
        "dataset_revision": DS_SHA, "scope": "metadata", "response_format": "detailed"}
    assert "new read" in summary["detailed_request_note"].lower()
    assert "receipts may differ" in summary["detailed_request_note"].lower()


def test_oversized_witness_is_omitted_without_truncating_its_identity():
    full = report()
    for finding in full["findings"]:
        finding["evidence"][0]["pointer"] = "/" + "x" * 10000
    summary = summarize(full)
    assert len(json.dumps(summary).encode()) <= 8192
    assert all(not f["evidence"] and f["omitted_evidence"] == 1 for f in summary["findings"])
    assert summary["status"] == "error"


def test_unresolved_revision_cannot_claim_pinned_replay():
    full = report()
    full["provenance"] = []
    summary = summarize(full)
    assert summary["detailed_request_pin_status"] == "unresolved"
    assert summary["unresolved_revisions"] == ["hf://org/model", "hf://org/data"]
    assert summary["detailed_request"]["arguments"]["revision"] == "main"


def test_local_replay_keeps_local_paths_and_requested_scope():
    summary = concise_check_summary(report(), source="/tmp/checkpoint", dataset="/tmp/dataset", scope="metadata")
    assert summary["detailed_request_pin_status"] == "local"
    assert summary["detailed_request"]["arguments"]["source"] == "/tmp/checkpoint"
    assert summary["detailed_request"]["arguments"]["dataset"] == "/tmp/dataset"


def test_request_larger_than_bound_is_refused_without_false_passing():
    with pytest.raises(InputError, match="detailed"):
        concise_check_summary(report(), source="/" + "x" * 9000)


def test_metadata_pass_terminal_view_still_names_unassessed_deployment():
    full = report()
    full.update(status="pass", findings=[], counts={"error": 0, "warning": 0, "unknown": 0, "pass": 6})
    text = concise_check(full)
    assert "PASS" in text and "NOT ASSESSED" in text
    assert "unassessed" in text.lower() and "1" in text


def test_terminal_view_never_cuts_a_finding_mid_word():
    # Round-1 beta (Arjun): the key sentence ended "...the saved statistics that wo".
    full = report()
    message = ("observation.state: the config declares [6] but the dataset has [9]. This may be a stale declaration "
               "inherited from a pretrained base; the saved statistics that would decide it are not available. ") * 3
    full.update(status="warning", findings=[{"code": "dataset.feature_shape", "status": "warning", "message": message,
                                             "evidence": [], "remedy": "Audit a local download. " * 20}])
    text = concise_check(full)
    line = next(row for row in text.splitlines() if row.startswith("WARNING dataset.feature_shape: "))
    assert line.endswith("not available. ...") and "the saved statistics that would decide it" in line
    nxt = next(row for row in text.splitlines() if row.startswith("Next: "))
    assert nxt.endswith("download. ...") and "Showing 1/1" in text


def test_real_stdio_formats_typed_catalog_and_pinned_drilldown(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = Path(__file__).resolve().parents[1]
    # The original metadata view is generated once by this independent HTTP stub.
    # The actual Hub loader, audit, MCP transport and formatters run unchanged.
    code = r'''
import json
from policy_doctor import hub
from policy_doctor.server import create_server
def request(url, *, limit, required_sha=None):
    if '/api/models/' in url:
        return json.dumps({'id':'org/model','sha':'1'*40,'private':False,'siblings':[
            {'rfilename':'config.json'},{'rfilename':'policy_preprocessor.json'}]}).encode()
    if '/api/datasets/' in url:
        return json.dumps({'id':'org/data','sha':'2'*40,'private':False,'siblings':[{'rfilename':'meta/info.json'}]}).encode()
    if url.endswith('/config.json'):
        value={'type':'act','input_features':{'observation.state':{'type':'STATE','shape':[2]}}}
    elif url.endswith('/policy_preprocessor.json'):
        value={'steps':[{'registry_name':'custom_'+('机器人🙂'*40)+str(i),'config':{'note':'x'*1500}} for i in range(10)]}
    elif url.endswith('/meta/info.json'):
        value={'fps':30,'features':{'observation.state':{'shape':[2]}}}
    else:
        raise AssertionError('Unexpected URL '+url)
    return json.dumps(value).encode()
hub._request=request
create_server(['ROOT'],allow_hub=True).run(transport='stdio')
'''.replace("ROOT", str(tmp_path))

    async def run():
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        params = StdioServerParameters(command=sys.executable, args=["-c", code], cwd=str(tmp_path), env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                catalog = await client.list_tools()
                assert len(catalog.tools) == 30
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact',
                        'check_training_targets'} <= {item.name for item in catalog.tools}
                tool = next(t for t in catalog.tools if t.name == "check_policy_metadata")
                fields = tool.inputSchema["properties"]
                assert fields["response_format"]["enum"] == ["concise", "detailed"]
                assert fields["response_format"]["default"] == "detailed"
                assert fields["scope"]["enum"] == ["deployment", "metadata"]
                arguments = {"source": "hf://org/model", "dataset": "hf://org/data"}
                detailed = await client.call_tool(tool.name, arguments)
                assert not detailed.isError
                full = detailed.structuredContent
                concise = await client.call_tool(tool.name, dict(arguments, response_format="concise"))
                assert not concise.isError
                summary = concise.structuredContent
                assert len(json.dumps(summary).encode()) <= 8192
                assert len(concise.content[0].text.encode("utf-8")) <= 8192
                assert "MCP envelope" in summary["size_contract"]
                assert summary["counts"] == full["counts"] and summary["status"] == full["status"]
                assert summary["omitted_findings"] > 0
                assert len(json.dumps(full).encode()) > len(json.dumps(summary).encode())
                original_digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                assert summary["full_report"]["sha256"] == original_digest
                replay = await client.call_tool(tool.name, summary["detailed_request"]["arguments"])
                assert not replay.isError
                replayed = replay.structuredContent
                assert replayed["findings"] == full["findings"]
                assert replayed["artifacts"] == full["artifacts"]
                assert replayed["status"] == full["status"] and replayed["counts"] == full["counts"]
                assert [p["requested_revision"] for p in replayed["provenance"]] == [SHA, DS_SHA]
                for before, after in zip(full["provenance"], replayed["provenance"], strict=True):
                    assert {k: v for k, v in before.items() if k != "requested_revision"} == {
                        k: v for k, v in after.items() if k != "requested_revision"}
                print(json.dumps({"catalog_count": len(catalog.tools), "detailed_json_bytes": len(json.dumps(full).encode()),
                    "concise_json_bytes": len(json.dumps(summary).encode()),
                    "concise_actual_text_bytes": len(concise.content[0].text.encode()),
                    "original_full_report_sha256": original_digest,
                    "pinned_revisions": [p["resolved_revision"] for p in replayed["provenance"]],
                    "evidence_matches_pinned_new_read": True}))
                bad = await client.call_tool(tool.name, dict(arguments, response_format="invalid"))
                assert bad.isError

    asyncio.run(run())


def test_multibyte_previews_keep_bound_and_never_change_status():
    full = report()
    for finding in full["findings"]:
        finding["message"] = "机器人🙂" * 1000
        finding["evidence"][0]["value"] = "机器人🙂" * 1000
    summary = summarize(full)
    assert len(json.dumps(summary).encode()) <= 8192
    assert len(json.dumps(summary, ensure_ascii=False).encode()) <= 8192
    assert summary["status"] == "error" and summary["counts"] == full["counts"]
    assert any(f["message_truncated"] for f in summary["findings"])


def test_cli_metadata_scope_command_exit_and_unassessed_notice(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    demo = subprocess.run([sys.executable, "-m", "policy_doctor", "demo", str(tmp_path / "example"), "--kind", "policy"],
                          capture_output=True, text=True, env=env)
    assert demo.returncode == 0, demo.stderr
    checked = subprocess.run([sys.executable, "-m", "policy_doctor", "check", str(tmp_path / "example/checkpoint"),
        "--dataset", str(tmp_path / "example/corrected_dataset"), "--scope", "metadata"],
        capture_output=True, text=True, env=env)
    assert checked.returncode == 0, checked.stderr
    assert "PASS" in checked.stdout and "NOT ASSESSED" in checked.stdout
