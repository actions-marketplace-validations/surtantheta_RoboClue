"""Actual pinned TrainConfig repack inputs, not guessed dataset conventions."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import test_action_conventions as fixtures

from policy_doctor.openpi_context import check_openpi_context

openpi_source = fixtures.openpi_source


def metadata(tmp_path, keys, version='v3.0'):
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True, exist_ok=True)
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': version,
        'features': {key: {'dtype': 'float32', 'shape': [1]} for key in keys}}))
    return root


@pytest.mark.parametrize('name,keys', [
    ('pi0_libero', ['image', 'wrist_image', 'state', 'actions']),
    ('pi0_libero_low_mem_finetune', ['image', 'wrist_image', 'state', 'actions']),
    ('pi0_fast_libero', ['image', 'wrist_image', 'state', 'actions']),
    ('pi0_fast_libero_low_mem_finetune', ['image', 'wrist_image', 'state', 'actions']),
    ('pi05_libero', ['image', 'wrist_image', 'state', 'actions']),
    ('pi0_aloha', ['observation.images.top', 'observation.state', 'action']),
    ('pi05_aloha', ['observation.images.top', 'observation.state', 'action']),
    ('pi0_aloha_towel', ['observation.images.top', 'observation.state', 'action']),
    ('pi0_aloha_pen_uncap', ['observation.images.cam_high', 'observation.images.cam_left_wrist',
        'observation.images.cam_right_wrist', 'observation.state', 'action']),
    ('pi05_aloha_pen_uncap', ['observation.images.cam_high', 'observation.images.cam_left_wrist',
        'observation.images.cam_right_wrist', 'observation.state', 'action']),
    ('pi05_droid_finetune', ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left',
        'joint_position', 'gripper_position', 'actions']),
])
def test_actual_selected_config_keys_and_overrides(tmp_path, openpi_source, name, keys):
    root = metadata(tmp_path, keys)
    report = check_openpi_context(root, config_name=name, source_root=openpi_source)
    assert report['status'] == 'pass', report
    assert report['required_dataset_keys'] == sorted(keys)
    assert report['missing_keys'] == []
    assert report['backend'] == 'lerobot'
    assert report['consumer_lerobot_pin']['revision'] == '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
    assert report['dataset_codebase_version'] == 'v3.0'
    assert 'not an observed installed loader' in report['consumer_lerobot_pin']['scope']
    assert 'Loader-version compatibility' in report['scope']
    assert 'loader-version compatibility or training readiness' in report['findings'][0]['message']
    for reference in report['findings'][0]['evidence']:
        assert len(reference['sha256']) == 64


def test_libero_public_schema_gets_named_missing_keys_before_delta_pass(tmp_path, openpi_source):
    root = metadata(tmp_path, ['observation.images.image', 'observation.state', 'action'])
    before = hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest()
    report = check_openpi_context(root, config_name='pi0_libero', source_root=openpi_source)
    assert report['status'] == 'error'
    assert report['missing_keys'] == ['actions', 'image', 'state', 'wrist_image']
    assert report['prompt_from_task'] is True
    assert report['declared_injected_keys'] == ['prompt']
    assert 'prompt' not in report['missing_keys']
    assert 'do not rename public data blindly' in report['findings'][0]['remedy']
    assert hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest() == before


def test_kwargs_repack_override_does_not_inherit_aloha_default_top(tmp_path, openpi_source):
    root = metadata(tmp_path, ['observation.images.top', 'observation.state', 'action'])
    report = check_openpi_context(root, config_name='pi0_aloha_pen_uncap', source_root=openpi_source)
    assert report['status'] == 'error'
    assert report['missing_keys'] == ['observation.images.cam_high', 'observation.images.cam_left_wrist',
                                      'observation.images.cam_right_wrist']
    assert 'observation.images.top' not in report['raw_source_keys']


@pytest.mark.parametrize('name', ['pi0_droid', 'pi05_droid', 'pi0_fast_droid', 'not_a_builtin'])
def test_dynamic_or_no_repack_never_receives_an_empty_key_pass(tmp_path, openpi_source, name):
    root = metadata(tmp_path, ['action', 'observation.state'])
    report = check_openpi_context(root, config_name=name, source_root=openpi_source)
    assert report['status'] == 'unknown'
    assert report['findings'][0]['code'] == 'openpi.input_context_unresolved'
    assert report['consumer_lerobot_pin']['revision'].startswith('0cf8648')


@pytest.mark.parametrize('name', ['pi0_fast_full_droid_finetune', 'pi05_full_droid_finetune'])
def test_rlds_backend_is_explicit_and_not_compared_to_lerobot_metadata(tmp_path, openpi_source, name):
    root = metadata(tmp_path, ['observation.state', 'action'])
    report = check_openpi_context(root, config_name=name, source_root=openpi_source)
    assert report['status'] == 'unknown'
    assert report['backend'] == 'rlds'
    assert report['raw_source_keys'] == []
    assert report['findings'][0]['code'] == 'openpi.rlds_backend'


@pytest.mark.parametrize('relative', ['src/openpi/training/config.py', 'src/openpi/transforms.py', 'uv.lock'])
def test_drifted_source_cannot_make_a_context_claim(tmp_path, openpi_source, relative):
    root = metadata(tmp_path, ['image', 'wrist_image', 'state', 'actions'])
    with (openpi_source / relative).open('a') as stream:
        stream.write('\n# drift\n')
    report = check_openpi_context(root, config_name='pi0_libero', source_root=openpi_source)
    assert report['status'] == 'unknown'
    assert report['consumer_lerobot_pin'] is None
    assert report['raw_source_keys'] == []
    assert 'fingerprint drift' in report['findings'][0]['message']


def test_preflight_retains_original_fingerprint_and_adds_raw_key_error(tmp_path, openpi_source):
    from test_action_conventions import write_dataset

    from policy_doctor.preflight import preflight

    root, *_ = write_dataset(tmp_path / 'real_rows', 'delta', dims=6, gripper=True)
    report = preflight(str(root), openpi_config='pi0_libero', openpi_source_root=str(openpi_source),
                       groot_contract=False, video_headers=False)
    assert report['status'] == 'error'
    assert report['components']['openpi_context']['status'] == 'error'
    assert report['components']['action_conventions']['verdicts']['consumer'] == 'pass'
    assert any(finding['code'] == 'openpi.raw_key_missing' and finding['component'] == 'openpi_context'
               for finding in report['findings'])
    assert report['coverage']['openpi_raw_input_context_requested'] is True


def test_actual_default_mcp_routes_openpi_raw_key_error_before_scoped_delta_pass(tmp_path, openpi_source):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from test_action_conventions import write_dataset

    root, *_ = write_dataset(tmp_path / 'native_rows', 'delta', dims=6, gripper=True)

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'roboclue', 'mcp', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            arguments = {'source': str(root), 'openpi_config': 'pi0_libero',
                'openpi_source_root': str(openpi_source), 'groot_contract': False, 'video_headers': False}
            brief = await client.call_tool('preflight_check', arguments)
            assert not brief.isError, brief
            body = brief.structuredContent
            assert body['status'] == 'error'
            assert body['components']['openpi_context']['status'] == 'error'
            assert body['components']['action_conventions']['delta_transform_consistency']['status'] == 'pass'
            assert any(finding['code'] == 'openpi.raw_key_missing' for finding in body['findings_shown'])
            assert 'openpi.raw_key_missing' in brief.content[0].text
            assert len(json.dumps(body).encode()) <= 8192
            detailed = await client.call_tool('preflight_check', dict(arguments, response_format='detailed'))
            assert not detailed.isError
            context = detailed.structuredContent['components']['openpi_context']
            assert context['consumer_lerobot_pin']['revision'].startswith('0cf8648')
            assert context['missing_keys'] == ['actions', 'image', 'state', 'wrist_image']

    asyncio.run(asyncio.wait_for(run(), timeout=120))
