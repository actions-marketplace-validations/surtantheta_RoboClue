"""Public MCP short answers preserve units scope and fail closed on omitted evidence."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import pytest
import test_units_manifest as fixtures
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.deployment_answer import BRIEF_LIMIT, BRIEF_RESULT_LIMIT, deployment_answer
from policy_doctor.preflight import preflight

SRC = Path(__file__).parents[1] / 'src'


def raw(root, *, corrected=False):
    return preflight(str(root), robot_command=fixtures.command('--robot.use_degrees=false' if corrected else ''),
                     video_headers=False, groot_contract=False)


@pytest.mark.parametrize('corrected', [False, True])
def test_short_answer_preserves_original_and_names_both_unit_sides(tmp_path, corrected):
    root = fixtures.so_dataset(tmp_path, 'm100')
    report = raw(root, corrected=corrected)
    original = copy.deepcopy(report)
    answer = deployment_answer(report, source=str(root), robot_command=fixtures.command(
        '--robot.use_degrees=false' if corrected else ''))
    assert report == original
    assert answer['status'] == report['status']
    assert answer['components']['units']['status'] == ('pass' if corrected else 'error')
    assert answer['components']['units']['compared_with'] == 'so101_follower from the robot command'
    assert 'RANGE_M100_100' in json.dumps(answer['components']['units']['detected_units'])
    assert answer['components']['units']['declared_units'] == ['RANGE_M100_100' if corrected else 'DEGREES']
    assert answer['components']['units']['limits'] == report['components']['units']['units_limits']
    assert len(json.dumps(answer).encode()) <= BRIEF_LIMIT
    assert answer['framework_context']['runtime_revalidated'] is False
    assert 'consumer environment unverified' in answer['framework_context']['environment']
    assert "response_format='detailed'" in answer['details']
    assert answer['finding']['evidence']
    assert all('sha256' in item for item in answer['finding']['evidence'])
    if not corrected:
        assert answer['finding']['code'] == 'units.joint_mode_mismatch'
        assert answer['finding']['rule_ids'] == ['RC001']
        assert '--robot.use_degrees=false' in answer['next']


def test_oversize_settings_or_evidence_requires_details_without_hiding_error(tmp_path):
    root = fixtures.so_dataset(tmp_path, 'm100')
    for corrected in (False, True):
        report = raw(root, corrected=corrected)
        report['components']['units']['findings'][0]['evidence'].append({
            'path': 'x' * 5000, 'pointer': '/complete/required/condition', 'sha256': 'a' * 64})
        answer = deployment_answer(report, source=str(root), robot_command=fixtures.command())
        assert answer['answer_status'] == 'details_required'
        assert answer['status'] == ('error' if report['status'] == 'error' else 'unknown')
        assert answer['report_status'] == report['status']
        assert answer['units_status'] == report['components']['units']['status']
        assert 'No concise approval' in answer['scope']
        assert len(json.dumps(answer).encode()) <= BRIEF_LIMIT


def test_a_skipped_required_comparison_never_becomes_a_short_pass():
    for status in ('pass', 'unknown', 'error'):
        answer = deployment_answer({'status': status, 'components': {}, 'coverage': {'units_not_run': 'x' * 6000}},
                                   source='/dataset', robot_command='recorder')
        assert answer['status'] == ('error' if status == 'error' else 'unknown')
        assert answer['report_status'] == status and answer['units_status'] == 'unknown'
        assert len(json.dumps(answer).encode()) <= BRIEF_LIMIT


def test_actual_default_sdk_short_units_and_detailed_evidence(tmp_path):
    root = fixtures.so_dataset(tmp_path, 'm100')

    async def run():
        params = StdioServerParameters(command=sys.executable, args=['-B', '-m', 'roboclue', 'mcp', 'serve',
                                                                    '--root', str(tmp_path)],
                                       env=dict(os.environ, PYTHONPATH=str(SRC)))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            assert len((await session.list_tools()).tools) == 8
            resources = await session.list_resources()
            assert 'roboclue://rules' in {str(resource.uri) for resource in resources.resources}
            catalog = json.loads((await session.read_resource('roboclue://rules')).contents[0].text)
            assert catalog['kind'] == 'rule_catalog'
            unit_rule = next(rule for rule in catalog['rules'] if rule['id'] == 'RC001')
            assert 'units.joint_mode_mismatch' in unit_rule['diagnostic_codes']
            assert unit_rule['upstream'] and unit_rule['frameworks']
            arguments = {'source': str(root), 'robot_command': fixtures.command(), 'video_headers': False,
                         'lerobot_version': '0.6.2', 'groot_contract': False}
            bad = await session.call_tool('preflight_check', arguments)
            assert not bad.isError
            assert bad.structuredContent['kind'] == 'deployment_units_answer'
            assert bad.structuredContent['status'] == 'error'
            assert len(bad.content[0].text.encode()) <= BRIEF_LIMIT
            assert len(json.dumps(bad.structuredContent).encode()) <= BRIEF_LIMIT
            assert len(json.dumps(bad.model_dump(mode='json', exclude_none=True)).encode()) <= BRIEF_RESULT_LIMIT
            assert 'units.joint_mode_mismatch' in bad.content[0].text
            assert bad.structuredContent['finding']['rule_ids'] == ['RC001']
            assert 'RC001' in bad.content[0].text
            assert '--robot.use_degrees=false' in bad.content[0].text
            full = await session.call_tool('preflight_check', dict(arguments, response_format='detailed'))
            assert not full.isError and full.structuredContent['kind'] == 'dataset_preflight'
            assert full.structuredContent['components']['units']['status'] == 'error'
            assert 'manifest' in full.structuredContent['components']['units']
            corrected = await session.call_tool('preflight_check', dict(arguments,
                robot_command=fixtures.command('--robot.use_degrees=false')))
            assert not corrected.isError
            assert corrected.structuredContent['components']['units']['status'] == 'pass'
            assert corrected.structuredContent['status'] != 'error'
            assert len(json.dumps(corrected.model_dump(mode='json', exclude_none=True)).encode()) <= BRIEF_RESULT_LIMIT
            skipped = await session.call_tool('preflight_check', dict(arguments, units=False))
            assert not skipped.isError
            assert skipped.structuredContent['units_status'] == 'unknown'
            assert skipped.structuredContent['status'] != 'pass'

    asyncio.run(asyncio.wait_for(run(), timeout=120))
