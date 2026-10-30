"""Exercise selection and truthful coverage through the actual CLI and MCP transport."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SRC = str(Path(__file__).parents[1] / 'src')


@pytest.fixture
def recordings(tmp_path):
    first, second = tmp_path / 'original.npz', tmp_path / 'candidate.npz'
    values = dict(step=np.arange(4), force_tcp=np.zeros((4, 2, 3)),
                  rgb=np.zeros((4, 2, 2, 2, 3), dtype=np.uint8))
    np.savez(first, **values)
    values['force_tcp'][2, 1, 0] = 7
    np.savez(second, **values)
    return first, second


def cli(first, second, *args):
    return subprocess.run([sys.executable, '-m', 'policy_doctor', 'compare-rollouts',
                           str(first), str(second), *args], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=SRC), timeout=30)


def test_cli_selected_native_values_and_coverage(recordings):
    first, second = recordings
    result = cli(first, second, '--streams', 'force_tcp', '--json')
    assert result.returncode == 3, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'unknown' and report['observed_difference']
    assert not report['comparison_complete']
    assert report['causal_conclusion'] == report['alignment']['sample_phase'] == 'unknown'
    assert report['first_observed_difference']['coordinate'] == [1, 0]
    for path, artifact in zip(recordings, report['artifacts'], strict=True):
        assert artifact['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        selection = artifact['stream_selection']
        assert selection['selected'] == ['force_tcp']
        assert set(selection['excluded']) == {'step', 'rgb'}
        assert not selection['inventory']['rgb']['values_decoded']
    # A real caller-supplied ID permits only the explicitly selected comparison.
    result = cli(first, second, '--streams', 'step', 'force_tcp', '--alignment-key', 'step', '--json')
    report = json.loads(result.stdout)
    assert result.returncode == 1 and report['status'] == 'different'
    assert report['first_observed_difference']['sample_id'] == 2
    for mode in ((), ('--markdown',)):
        result = cli(first, second, '--streams', 'force_tcp', *mode)
        assert result.returncode == 3
        assert 'Selected' in result.stdout and 'rgb' in result.stdout
        assert 'excluded' in result.stdout.lower()


def test_cli_selection_refusals_and_unchanged_default(recordings):
    first, second = recordings
    for args in [('--json',), ('--streams', 'absent', '--json'),
                 ('--streams', 'force_tcp', 'force_tcp', '--json')]:
        result = cli(first, second, *args)
        assert result.returncode == 2


def test_mcp_selected_streams_raw_validation_and_read_roots(recordings):
    first, second = recordings

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(first.parent)],
            env=dict(os.environ, PYTHONPATH=SRC))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                catalog = await client.list_tools()
                assert len(catalog.tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in catalog.tools}
                tool = next(t for t in catalog.tools if t.name == 'compare_rollout_recordings')
                assert 'streams' in tool.inputSchema['properties']
                args = dict(reference=str(first), candidate=str(second), streams=['force_tcp'])
                result = await client.call_tool(tool.name, args)
                assert not result.isError
                report = result.structuredContent
                assert report['status'] == 'unknown' and report['observed_difference']
                assert report['artifacts'][0]['stream_selection']['excluded'] == ['step', 'rgb']
                result = await client.call_tool(tool.name, dict(args, candidate=str(first)))
                assert not result.isError and not result.structuredContent['observed_difference']
                assert result.structuredContent['status'] == 'unknown'
                for selection in (True, [], ['force_tcp', True], ['absent'], ['force_tcp', 'force_tcp']):
                    result = await client.call_tool(tool.name, dict(args, streams=selection))
                    assert result.isError, selection
                result = await client.call_tool(tool.name, dict(args, reference='/etc/passwd'))
                assert result.isError and 'outside' in result.content[0].text
                result = await client.call_tool(tool.name, dict(args, reference=str(first.parent)))
                assert result.isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=45))
