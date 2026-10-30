import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_cli_investigation_and_recording_comparison(case):
    root = case['checkpoint'].parent
    first, second = root / 'first.npz', root / 'second.npz'
    np.savez(first, step=np.arange(4), actions=np.zeros((4, 2)))
    np.savez(second, step=np.arange(4), actions=np.array([[0., 0.], [0., 0.], [1., 0.], [1., 0.]]))
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'investigate', str(root), '--json'],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['summary'].startswith('Found 1 checkpoint')
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'compare-rollouts', str(first), str(second),
                             '--alignment-key', 'step', '--json'], env=environment, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)['first_observed_difference']['row'] == 2


def test_actual_mcp_new_workflows_and_scope(case):
    root = case['checkpoint'].parent
    first, second = root / 'first.npz', root / 'second.npz'
    np.savez(first, step=np.arange(4), actions=np.zeros((4, 2)))
    np.savez(second, step=np.arange(4), actions=np.ones((4, 2)))
    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(root)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('investigate_run_directory', {'directory': str(root)})
                assert not result.isError and result.structuredContent['status'] == 'pass'
                result = await client.call_tool('compare_rollout_recordings', {
                    'reference': str(first), 'candidate': str(second), 'alignment_key': 'step'})
                assert not result.isError and result.structuredContent['status'] == 'different'
                assert result.structuredContent['first_observed_difference']['row'] == 0
                for tool, args in [('investigate_run_directory', {'directory': '/etc'}),
                                   ('compare_rollout_recordings', {'reference': '/etc/passwd', 'candidate': str(second)})]:
                    result = await client.call_tool(tool, args)
                    assert result.isError and 'outside' in result.content[0].text
    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
