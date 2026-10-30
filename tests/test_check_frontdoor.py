"""Exercise the new front door and its operator-owned network boundary."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import hub
from policy_doctor.cli import main
from policy_doctor.demo import create_policy_demo
from policy_doctor.server import create_server


def test_cli_requires_fetch_before_any_network(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('No network was authorized')

    monkeypatch.setattr(hub, '_request', forbidden)
    assert main(['check', 'hf://org/policy', '--dataset', 'hf://org/dataset']) == 2
    assert 'fetch' in capsys.readouterr().err.lower()


@pytest.mark.parametrize('source,dataset', [
    ('hf://org/policy', '/etc'), ('/etc', 'hf://org/data'),
])
def test_mcp_local_roots_checked_before_network(tmp_path, monkeypatch, source, dataset):
    def forbidden(*args, **kwargs):
        raise AssertionError('Both local roots must be authorized before Hub intake')

    monkeypatch.setattr(hub, 'check_checkpoint', forbidden)
    server = create_server([str(tmp_path)], allow_hub=True)

    async def run():
        with pytest.raises(Exception, match='outside.*read roots'):
            await server.call_tool('check_policy_metadata', {'source': source, 'dataset': dataset})

    asyncio.run(run())


def test_real_stdio_opt_in_and_local_evidence(tmp_path):
    demo = create_policy_demo(tmp_path / 'example')

    async def run(enabled):
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path)] + (['--allow-hub'] if enabled else []),
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert len(tools) == (30 if enabled else 29)
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {tool.name for tool in tools}
                found = [tool for tool in tools if tool.name == 'check_policy_metadata']
                if not enabled:
                    assert not found
                    return
                assert found[0].annotations.readOnlyHint and found[0].annotations.openWorldHint
                for dataset, expected in [('dataset', 'error'), ('corrected_dataset', 'unknown')]:
                    response = await session.call_tool('check_policy_metadata', {
                        'source': demo['checkpoint'], 'dataset': demo[dataset]})
                    assert not response.isError
                    body = response.structuredContent or json.loads(response.content[0].text)
                    assert body['kind'] == 'checkpoint_check' and body['status'] == expected
                    assert body['provenance'] == []
                    errors = [f for f in body['findings'] if f['status'] == 'error']
                    assert len(errors) == (2 if expected == 'error' else 0)
                    assert all(f['code'] == 'action.processor_order_names' for f in errors)
                denied = await session.call_tool('check_policy_metadata', {
                    'source': 'hf://org/policy', 'dataset': '/etc'})
                assert denied.isError and 'outside' in denied.content[0].text

    async def exercise():
        await run(False)
        await run(True)

    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
