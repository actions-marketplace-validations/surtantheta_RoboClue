"""Explicit fleet labels select source profiles, never the MCP Python environment."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import pytest
import test_mcp_r3_frontdoor as fixtures
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.deployment_answer import attach_consumer_version
from policy_doctor.preflight import preflight

SRC = Path(__file__).parents[1] / 'src'


@pytest.mark.parametrize('version,corrected,expected', [('0.6.2', False, 'error'), ('0.6.2', True, 'pass'),
                                                     ('0.6.3', False, 'unknown'), ('0.6.3', True, 'unknown')])
def test_only_explicit_qualified_profile_keeps_active_comparison(tmp_path, version, corrected, expected):
    root = fixtures.dataset(tmp_path)
    command = fixtures.command()
    if not corrected:
        command = command.replace('--robot.use_degrees=false', '')
    original = preflight(str(root), robot_command=command, video_headers=False, groot_contract=False)
    before = copy.deepcopy(original)
    result = attach_consumer_version(original, version)
    assert original == before
    units = result['components']['units']
    assert units['status'] == expected
    assert units['manifest']['features'] == original['components']['units']['manifest']['features']
    assert units['consumer_version']['declared_version'] == version
    assert units['consumer_version']['fleet_verified'] is False
    if version == '0.6.2':
        assert units['consumer_version']['tested_profile'] == 'lerobot-0.6.2-so-146d847'
        assert units['consumer_version']['command_role'] == 'collection_only'
    else:
        assert 'conditional_comparison' in units
        assert not any(finding['code'] in ('units.joint_mode_mismatch', 'units.joint_mode_match')
                       for finding in result['findings'] if finding.get('component') == 'units')
        assert all(arm.get('mode') is None for arm in units['manifest']['deployment']['arms'].values())


def test_unknown_consumer_cannot_hide_a_dataset_error(tmp_path):
    root = fixtures.dataset(tmp_path)
    report = preflight(str(root), robot_command=fixtures.command(), video_headers=False, groot_contract=False)
    report['components']['dataset_statistics'] = {'status': 'error'}
    report['findings'].append({'component': 'dataset_statistics', 'code': 'dataset.statistics_inconsistent',
                               'status': 'error', 'message': 'Saved mean is outside its range.'})
    report['status'] = 'error'
    result = attach_consumer_version(report, '9.9.9')
    assert result['status'] == 'error'
    assert result['components']['units']['status'] == 'unknown'
    assert any(finding['code'] == 'dataset.statistics_inconsistent' and finding['status'] == 'error'
               for finding in result['findings'])


def test_actual_sdk_fleet_profile_scope_and_two_camera_budget(tmp_path):
    root = fixtures.dataset(tmp_path)

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'roboclue', 'mcp', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(SRC)))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            tools = (await client.list_tools()).tools
            assert len(tools) == 8
            schema = next(tool for tool in tools if tool.name == 'preflight_check').inputSchema
            assert schema['properties']['lerobot_version']['default'] is None
            assert 'Caller-declared' in schema['properties']['lerobot_version']['description']
            base = {'source': str(root), 'robot_command': fixtures.command(('up', 'side')),
                    'groot_contract': False, 'video_headers': False}
            known = await client.call_tool('preflight_check', dict(base, lerobot_version='0.6.2'))
            assert not known.isError, known
            body = known.structuredContent
            assert body.get('answer_status') != 'details_required', body
            assert body['components']['units']['status'] == 'pass'
            assert body['camera_keys']['status'] == 'pass'
            assert body['consumer']['declared'] == '0.6.2'
            assert body['consumer']['profile'] == 'lerobot-0.6.2-so-146d847'
            assert body['consumer']['role'] == 'collection_only'
            assert body['consumer']['fleet_verified'] is False
            assert len(json.dumps(known.model_dump(mode='json', exclude_none=True)).encode()) <= 2560
            assert len(json.dumps(body).encode()) <= 2048
            assert '0.6.2 declared' in known.content[0].text
            full = await client.call_tool('preflight_check', dict(base, lerobot_version='0.6.2', response_format='detailed'))
            assert not full.isError
            context = full.structuredContent['components']['units']['consumer_version']
            assert context['source_basis']['source_revision'].startswith('146d847')
            assert context['fleet_verified'] is False
            unsupported = await client.call_tool('preflight_check', dict(base, lerobot_version='0.6.3'))
            assert not unsupported.isError
            unknown = unsupported.structuredContent
            assert unknown['status'] != 'pass'
            assert unknown.get('units_status', unknown.get('components', {}).get('units', {}).get('status')) == 'unknown'
            assert unknown['consumer']['profile'] is None
            assert unknown['consumer']['reason']
            assert '--robot.use_degrees=true' not in unsupported.content[0].text
            assert len(json.dumps(unsupported.model_dump(mode='json', exclude_none=True)).encode()) <= 2560
            no_version = await client.call_tool('preflight_check', base)
            assert not no_version.isError
            assert no_version.structuredContent['consumer']['declared'] is None
            assert no_version.structuredContent['components']['units']['status'] == 'unknown'
            assert 'actual robot environment' in no_version.content[0].text
            missing = await client.call_tool('preflight_check', {'source': str(root), 'lerobot_version': '0.6.2'})
            assert missing.isError and 'robot_command' in missing.content[0].text
            for bad in ('', True, 123):
                refused = await client.call_tool('preflight_check', dict(base, lerobot_version=bad))
                assert refused.isError

    asyncio.run(asyncio.wait_for(run(), timeout=120))
