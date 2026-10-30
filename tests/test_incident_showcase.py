import asyncio
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.reports import markdown_report
from policy_doctor.runtime_diff import compare_runtime_captures


def test_showcase_has_corrected_replay_and_unknown_control(tmp_path):
    project = Path(__file__).parents[1]
    create = runpy.run_path(str(project / 'examples/create_incident_showcase.py'))['create_showcase']
    destination = tmp_path / 'showcase'
    summary = create(destination)
    assert summary['first_command_difference']['reference_value'] == 11.
    assert summary['first_command_difference']['candidate_value'] == 13.
    assert summary['unknown_control_first_difference'] == {'stream': 'state', 'row': 4}
    assert summary['investigation_status'] == 'pass'
    with pytest.raises(FileExistsError):
        create(destination)
    report = compare_runtime_captures(str(destination / 'reference/capture'), str(destination / 'candidate/capture'))
    assert all(len(c['evidence']) == 2 for c in report['changes'])
    assert '| /post/0/statistics/action/mean |' in markdown_report(report)
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'compare-captures',
        str(destination / 'reference/capture'), str(destination / 'candidate/capture'), '--markdown'],
        env=dict(os.environ, PYTHONPATH=str(project / 'src')), capture_output=True, text=True)
    assert result.returncode == 1 and 'Captured runtime comparison' in result.stdout

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(destination)],
            env=dict(os.environ, PYTHONPATH=str(project / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('compare_runtime_captures', {
                    'reference': str(destination / 'reference/capture'), 'candidate': str(destination / 'candidate/capture')})
                assert not result.isError and result.structuredContent['status'] == 'different'
                result = await client.call_tool('compare_runtime_captures', {
                    'reference': '/etc', 'candidate': str(destination / 'candidate/capture')})
                assert result.isError and 'outside' in result.content[0].text
    asyncio.run(asyncio.wait_for(exercise(), timeout=30))


def test_report_escapes_untrusted_artifact_data():
    # Use a minimal structured investigation to exercise both field and evidence rendering.
    report = {'status': 'unknown', 'scope': '<script>alert(1)</script>', 'findings': [
        {'status': 'unknown', 'code': '[click](https://example.invalid)', 'message': '<img src=x onerror=x>',
         'evidence': [{'path': 'x|y`z', 'pointer': '/x', 'sha256': '123'}]}],
        'next_actions': [], 'coverage': {}, 'intake_problems': [{'path': 'capture', 'reason': 'digest mismatch'}]}
    rendered = markdown_report(report)
    assert '<script>' not in rendered and '<img' not in rendered
    assert '\\[click\\]' in rendered and 'x\\|y\\`z' in rendered
    assert 'digest mismatch' in rendered
