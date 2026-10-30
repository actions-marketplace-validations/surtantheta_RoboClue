import asyncio
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_generated_controller_incident_and_actual_cli_mcp(tmp_path):
    project = Path(__file__).parents[1]
    create = runpy.run_path(str(project / 'examples/create_controller_showcase.py'))['create_showcase']
    root = tmp_path / 'incident'
    summary = create(root)
    assert summary['same_waypoint_positions_and_times']
    assert summary['candidate_position_at_750ms'] == [1.03125]
    assert summary['valid_reordering'] == summary['corrected_status'] == 'pass'
    assert summary['missing_initial_state'] == 'unknown'
    environment = dict(os.environ, PYTHONPATH=str(project / 'src'))
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'compare-trajectories',
        str(root / 'reference.json'), str(root / 'candidate.json'), '--times', '0', '.25', '.75', '1', '--markdown'],
        env=environment, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert 'Same waypoint positions and times: True' in result.stdout
    assert '1.03125' in result.stdout and 'Execution status: **unknown**' in result.stdout
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'sample-trajectory',
        str(root / 'missing_initial_state.json'), '--times', '0', '1.5', '--json'],
        env=environment, capture_output=True, text=True)
    assert result.returncode == 3 and json.loads(result.stdout)['status'] == 'unknown'

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(root)], env=environment)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('sample_ros_trajectory', {
                    'trajectory': str(root / 'candidate.json'), 'sample_times': [.75], 'position_limits': {'shoulder': [0., 1.]}})
                assert not result.isError and result.structuredContent['status'] == 'error'
                assert result.structuredContent['samples'][0]['positions'] == [1.03125]
                result = await client.call_tool('compare_ros_trajectories', {
                    'reference': str(root / 'reference.json'), 'candidate': str(root / 'candidate.json'), 'sample_times': [.25, .75]})
                assert not result.isError and result.structuredContent['status'] == 'different'
                for tool, arguments in [
                    ('sample_ros_trajectory', {'trajectory': '/etc/passwd', 'sample_times': [.5]}),
                    ('compare_ros_trajectories', {'reference': '/etc/passwd', 'candidate': str(root / 'candidate.json'), 'sample_times': [.5]}),
                    ('sample_ros_trajectory', {'trajectory': str(root / 'candidate.json'), 'sample_times': [True]})]:
                    result = await client.call_tool(tool, arguments)
                    assert result.isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
