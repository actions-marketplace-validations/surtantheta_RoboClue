import asyncio
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import sha256


def test_real_stdio_server_protocol_and_tools(case):
    parquet = case["dataset"] / "data/episode.parquet"
    parquet.parent.mkdir()
    pq.write_table(pa.Table.from_pylist([{"action": [1., .5], "observation.state": [10., .5],
                                        "episode_index": 0, "frame_index": 0, "timestamp": 0.}]), parquet)
    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=["-m", "policy_doctor", "serve", "--root", str(case["checkpoint"].parent)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src")))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact',
                        'check_training_targets'} <= {tool.name for tool in tools}
                assert all(t.annotations.readOnlyHint for t in tools)
                audit = await session.call_tool("audit_deployment", {k: str(case[k]) for k in ("checkpoint", "dataset", "deployment")})
                assert not audit.isError
                body = audit.structuredContent or json.loads(audit.content[0].text)
                assert body["status"] == "pass"
                evidence = next(f for f in body["findings"] if f["code"] == "semantics.units")["evidence"][1]
                verified = await session.call_tool("verify_finding_evidence", {
                    "path": evidence["path"], "pointer": evidence["pointer"], "expected_sha256": evidence["sha256"]})
                assert not verified.isError and verified.structuredContent["value"] == ["rad", "normalized"]
                stale = await session.call_tool("verify_finding_evidence", {
                    "path": evidence["path"], "pointer": evidence["pointer"], "expected_sha256": "0" * 64})
                assert not stale.isError and stale.structuredContent["status"] == "error"
                assert "value" not in stale.structuredContent
                broken = json.loads(case["deployment"].read_text())
                broken["units"] = ["deg", "normalized"]
                broken_path = case["put"](case["checkpoint"].parent / "broken.json", broken)
                repair = await session.call_tool("check_candidate_repair", {
                    "checkpoint": str(case["checkpoint"]), "before": str(broken_path),
                    "candidate": str(case["deployment"]), "dataset": str(case["dataset"])})
                assert not repair.isError and repair.structuredContent["status"] == "pass"
                assert repair.structuredContent["resolved_error_counts"] == {"semantics.units": 1}
                assert sha256(case["deployment"]) == evidence["sha256"]
                trace = await session.call_tool("trace_saved_actions", {k: str(case[k]) for k in ("checkpoint", "sample", "deployment")})
                assert not trace.isError
                body = trace.structuredContent or json.loads(trace.content[0].text)
                assert body["commands"][0] == [11, .5]
                for name, args in [
                    ("inspect_checkpoint", {"checkpoint": str(case["checkpoint"])}),
                    ("inspect_dataset", {"dataset": str(case["dataset"])}),
                    ("make_deployment_template", {"checkpoint": str(case["checkpoint"])}),
                    ("compare_configurations", {"before": str(case["deployment"]), "after": str(case["deployment"])}),
                ]:
                    assert not (await session.call_tool(name, args)).isError
                outside = await session.call_tool("inspect_checkpoint", {"checkpoint": "/etc"})
                assert outside.isError
                assert "outside" in outside.content[0].text
                scanned = await session.call_tool("scan_dataset_values", {"dataset": str(case["dataset"])})
                assert not scanned.isError
                assert scanned.structuredContent["status"] == "pass"
                parquet.unlink()
                missing = await session.call_tool("scan_dataset_values", {"dataset": str(case["dataset"])})
                assert missing.isError and "No data" in missing.content[0].text

    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
