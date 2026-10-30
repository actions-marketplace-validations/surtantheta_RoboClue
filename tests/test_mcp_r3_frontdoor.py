"""Round-three default MCP paths, camera scope, transport size and canonical launchers."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import test_units_manifest as fixtures
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.deployment_answer import BRIEF_RESULT_LIMIT, attach_camera_comparison
from policy_doctor.mcp_presentation import evidence_tool_result
from policy_doctor.product_mcp import create_product_server
from policy_doctor.server import _summary, create_server
from policy_doctor.setup_client import client_configuration

SRC = Path(__file__).parents[1] / 'src'


def dataset(tmp_path):
    root = fixtures.so_dataset(tmp_path, 'm100')
    info = root / 'meta/info.json'
    value = json.loads(info.read_text())
    for name in ('up', 'side'):
        value['features']['observation.images.' + name] = {'dtype': 'video', 'shape': [480, 640, 3], 'names': ['height', 'width', 'channels']}
    info.write_text(json.dumps(value))
    return root


def command(names=None):
    value = fixtures.command('--robot.use_degrees=false')
    if names is not None:
        cameras = {name: {'type': 'opencv', 'index_or_path': i, 'width': 640, 'height': 480, 'fps': 30}
                   for i, name in enumerate(names)}
        value += ' --robot.cameras=' + shlex.quote(json.dumps(cameras))
    return value


def strip_descriptions(value):
    if isinstance(value, dict):
        return {key: strip_descriptions(item) for key, item in value.items() if key != 'description'}
    if isinstance(value, list):
        return [strip_descriptions(item) for item in value]
    return value


def test_product_guidance_keeps_every_typed_parameter_contract(tmp_path):
    legacy = create_server([str(tmp_path)])
    product = create_product_server([str(tmp_path)])

    async def schemas():
        return (await legacy.list_tools()), (await product.list_tools())

    old, new = asyncio.run(schemas())
    before = next(tool for tool in old if tool.name == 'preflight_check')
    after = next(tool for tool in new if tool.name == 'preflight_check')
    assert strip_descriptions(before.inputSchema) == strip_descriptions(after.inputSchema)
    assert len(after.description) < 800
    assert len(after.inputSchema['properties']['source']['description']) > 20
    assert 'never executed' in after.description
    assert 'parsed' in after.inputSchema['properties']['robot_command']['description']


def test_camera_report_join_does_not_mutate_original_report():
    original = {'status': 'pass', 'components': {'units': {'status': 'pass'}}, 'findings': [], 'coverage': {}}
    before = copy.deepcopy(original)
    camera = {'status': 'error', 'requested': True, 'compared': True,
              'findings': [{'code': 'camera.keys_mismatch', 'status': 'error', 'message': 'Missing camera up.'}]}
    joined = attach_camera_comparison(original, camera)
    assert original == before
    assert joined['status'] == 'error'
    assert joined['components']['units']['status'] == 'pass'
    assert joined['components']['camera_keys']['status'] == 'error'
    assert joined['interface_scope']['joint order'] == 'not_assessed'


def test_nonduplicated_detail_preserves_every_structured_value():
    report = {'kind': 'dataset_preflight', 'status': 'warning', 'scope': 'Declarations only; no native execution.',
              'findings': [{'code': 'camera.extra_keys', 'status': 'warning', 'message': 'Extra camera may be deliberate.',
                            'evidence': [{'path': 'meta/info.json', 'pointer': '/features', 'sha256': 'a' * 64,
                                          'value': list(range(4000))}]}]}
    before = copy.deepcopy(report)
    response = evidence_tool_result(report, detailed=True)
    assert response.structuredContent == report == before
    assert 'Extra camera may be deliberate.' in response.content[0].text
    assert 'a' * 64 in response.content[0].text
    assert '3999' not in response.content[0].text
    assert 'structuredContent' in response.content[0].text or 'StructuredContent' in response.content[0].text


def test_action_warning_is_explicit_in_the_concise_component():
    report = {'kind': 'dataset_preflight', 'status': 'unknown', 'components': {
        'action_conventions': {'status': 'warning', 'title': 'Action conventions', 'verdicts': {'consumer': 'pass'},
            'findings': [{'code': 'convention.rotation_declared_conflict', 'status': 'warning',
                          'message': 'Named quaternion values are not unit norm.'}]}}}
    result = _summary(report)
    part = result['components']['action_conventions']
    assert part['consumer'] == 'pass'
    assert part['warning_codes'] == ['convention.rotation_declared_conflict']
    assert part.get('warning_codes_omitted', 0) == 0
    assert part['pipeline_compatibility'] == 'not_assessed'


def test_actual_default_sdk_distinguishes_camera_declarations_and_names_scope(tmp_path):
    root = dataset(tmp_path)

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'roboclue', 'mcp', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(SRC)))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            assert (await client.initialize()).serverInfo.name == 'RoboClue'
            assert len((await client.list_tools()).tools) == 8
            results = []
            for names in (None, ('up', 'side'), ('front',), ('up',), ()):
                response = await client.call_tool('preflight_check', {'source': str(root), 'robot_command': command(names),
                    'lerobot_version': '0.6.2', 'groot_contract': False, 'video_headers': False})
                assert not response.isError, response
                body = response.structuredContent
                assert len(json.dumps(response.model_dump(mode='json', exclude_none=True)).encode()) <= BRIEF_RESULT_LIMIT
                assert {'image size', 'frame rate/fps', 'joint order'} <= set(body['not_assessed'])
                assert all(term in response.content[0].text for term in ('image size', 'frame rate/fps', 'joint order'))
                results.append(body)
            absent, correct, wrong, partial, empty = results
            assert absent['camera_keys']['status'] == 'warning'
            assert absent['camera_keys']['compared'] is True
            assert absent['camera_keys']['defaulted'] is True
            assert absent['unresolved']
            assert correct['camera_keys']['status'] == 'pass'
            assert correct['camera_keys']['compared'] is True
            assert correct['unresolved']
            assert correct.get('answer_status') != 'details_required'
            assert correct['camera_keys']['rule_ids'] == ['RC007']
            assert len(correct['camera_keys']['metadata_sha256']) == 64
            assert len(correct['camera_keys']['declaration_sha256']) == 64
            assert wrong['status'] == 'error'
            assert wrong['camera_keys']['status'] == 'error'
            assert wrong['camera_keys']['rule_ids'] == ['RC007']
            assert wrong['camera_keys']['missing'] == ['side', 'up']
            assert wrong != correct
            assert 'camera.keys_mismatch' in json.dumps(wrong)
            for advisory in (partial, empty):
                assert advisory['status'] != 'error'
                assert advisory['camera_keys']['status'] == 'warning'
                assert 'camera.partial_keys' in json.dumps(advisory)
            bad_flag = await client.call_tool('preflight_check', {'source': str(root),
                'robot_command': command() + ' --robot.use_degres=false'})
            assert bad_flag.isError and 'use_degres' in bad_flag.content[0].text
            full = await client.call_tool('preflight_check', {'source': str(root), 'robot_command': command(('front',)),
                'lerobot_version': '0.6.2', 'groot_contract': False, 'video_headers': False, 'response_format': 'detailed'})
            assert not full.isError
            assert full.structuredContent['components']['camera_keys']['status'] == 'error'
            assert 'camera.keys_mismatch' in full.content[0].text
            assert full.content[0].text.lstrip().startswith('RoboClue')
            assert len(full.content[0].text.encode()) < len(json.dumps(full.structuredContent).encode())
            card = await client.call_tool('dataset_health_card', {'source': str(root), 'units': True})
            assert not card.isError
            recap = card.content[0].text
            assert '-100..100 across the calibrated range' in recap
            assert 'No robot command/config was compared' in recap
            assert 'RANGE_M100_100' not in recap
            assert card.structuredContent['kind'] == 'dataset_health_card'

    asyncio.run(asyncio.wait_for(run(), timeout=120))


def test_canonical_advanced_setup_and_serve_keep_legacy_explicitly_compatible(tmp_path):
    text = client_configuration([str(tmp_path)], clients=('json',), canonical=True)
    config, _ = json.JSONDecoder().raw_decode(text[text.index('{'):])
    definition = config['mcpServers']['roboclue']
    assert definition['command'] == sys.executable
    assert definition['args'][:4] == ['-m', 'roboclue', 'advanced', 'serve']
    assert 'policy_doctor' not in definition['args']
    env = dict(os.environ, PYTHONPATH=str(SRC))
    cli = subprocess.run([sys.executable, '-B', '-m', 'roboclue', 'advanced', 'setup', '--root', str(tmp_path), '--client', 'json'],
                         capture_output=True, text=True, env=env, timeout=30)
    assert cli.returncode == 0, cli.stderr
    printed, _ = json.JSONDecoder().raw_decode(cli.stdout[cli.stdout.index('{'):])
    assert printed['mcpServers']['roboclue']['args'][:4] == ['-m', 'roboclue', 'advanced', 'serve']

    async def names():
        results = []
        for args in (['-m', 'roboclue', 'advanced', 'serve'], ['-m', 'roboclue', 'mcp', 'serve', '--advanced'],
                     ['-m', 'policy_doctor', 'serve']):
            params = StdioServerParameters(command=sys.executable, args=['-B', *args, '--root', str(tmp_path)], env=env)
            async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
                results.append(((await client.initialize()).serverInfo.name, len((await client.list_tools()).tools)))
        return results

    assert asyncio.run(asyncio.wait_for(names(), timeout=120)) == [('RoboClue', 29), ('RoboClue', 29), ('Policy Deployment Doctor', 29)]


@pytest.mark.parametrize('canonical', [1, 'yes', None])
def test_canonical_launcher_flag_is_not_coerced(tmp_path, canonical):
    from policy_doctor.artifacts import InputError

    with pytest.raises(InputError, match='strict boolean'):
        client_configuration([str(tmp_path)], canonical=canonical)
