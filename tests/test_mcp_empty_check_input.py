"""Explicitly empty MCP paths must not acquire authority or disappear."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import hub
from policy_doctor.demo import create_policy_demo
from policy_doctor.server import create_server


@pytest.mark.parametrize("response_format", ["detailed", "concise"])
def test_actual_stdio_empty_dataset_is_rejected_instead_of_checkpoint_only_pass(tmp_path, response_format):
    demo = create_policy_demo(tmp_path / "generated")

    async def run():
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "policy_doctor", "serve", "--root", str(tmp_path), "--allow-hub"],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")))
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool("check_policy_metadata", {
                    "source": demo["checkpoint"], "dataset": "", "scope": "metadata",
                    "response_format": response_format})
                assert response.isError, response.structuredContent or response.content
                assert "nonempty" in response.content[0].text.lower()

    asyncio.run(asyncio.wait_for(run(), timeout=30))


@pytest.mark.parametrize("source,dataset", [("hf://org/model", ""), ("", "hf://org/data")])
@pytest.mark.parametrize("response_format", ["detailed", "concise"])
def test_empty_paths_refuse_before_hub_or_local_intake(tmp_path, monkeypatch, source, dataset, response_format):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("An empty original path cannot authorize an intake")

    monkeypatch.setattr(hub, "check_checkpoint", forbidden)

    async def run():
        server = create_server([str(tmp_path), str(Path.cwd())], allow_hub=True)
        with pytest.raises(Exception, match="nonempty"):
            await server.call_tool("check_policy_metadata", {
                "source": source, "dataset": dataset, "scope": "metadata", "response_format": response_format})

    asyncio.run(run())
    assert calls == []
