"""One command reaches the readiness engine without executing robot code."""
import asyncio
import json
import os
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.readiness_workflow import check_robot_readiness, prepare_readiness


def config(case, *, camera='front', policy=None):
    path = case['checkpoint'].parent / 'robot.json'
    path.write_text(json.dumps({'strategy': {'type': 'base'}, 'fps': 30,
        'robot': {'type': 'so100_follower', 'cameras': {camera: {'type': 'opencv', 'width': 32, 'height': 32, 'fps': 30}}},
        'policy': {'path': str(policy or case['checkpoint'])}}))
    return path


def healthy(case):
    from policy_doctor.robot_interface import from_config
    path = config(case)
    names = from_config(path)['action_names']
    for filename in ('config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        p = case['checkpoint'] / filename
        data = json.loads(p.read_text())
        if filename == 'config.json':
            data['input_features']['observation.state']['shape'] = [len(names)]
            data['output_features']['action']['shape'] = [len(names)]
        else:
            for step in data['steps']:
                cfg = step['config']
                if 'action_names' in cfg:
                    cfg['action_names'] = names
                if 'features' in cfg:
                    for key in ('action', 'observation.state'):
                        cfg['features'][key]['shape'] = [len(names)]
        p.write_text(json.dumps(data))
    pre = case['checkpoint'] / 'policy_preprocessor.json'
    data = json.loads(pre.read_text())
    data['steps'][:0] = [{'registry_name': 'device_processor', 'config': {'device': 'cpu'}},
                         {'registry_name': 'rename_observations_processor', 'config': {'rename_map': {}}}]
    pre.write_text(json.dumps(data))
    values = {f'{key}.{stat}': np.ones(len(names)) for key in ('action', 'observation.state') for stat in ('mean', 'std')}
    for name in ('pre.safetensors', 'post.safetensors'):
        save_file(values, case['checkpoint'] / name)
    p = case['dataset'] / 'meta/info.json'
    data = json.loads(p.read_text())
    for key in ('action', 'observation.state'):
        data['features'][key].update(shape=[len(names)], names=names)
    p.write_text(json.dumps(data))
    return path


def test_real_config_reaches_all_axes_and_cli(case, capsys):
    path = healthy(case)
    report = check_robot_readiness(robot_config=str(path), dataset=str(case['dataset']))
    assert report['status'] == 'pass'
    assert len(report['axes']) == 10 and report['axes']['units']['status'] == 'unknown'
    assert report['axes']['normalization']['status'] == 'cleared'
    assert main(['readiness', '--robot-config', str(path), '--dataset', str(case['dataset'])]) == 0
    out = capsys.readouterr().out
    assert 'robot readiness: PASS' in out and 'normalization: cleared' in out


def test_actual_command_embedded_policy_and_conflict_before_network(case, monkeypatch):
    from policy_doctor import hub
    monkeypatch.setattr(hub, '_request', lambda *a, **k: pytest.fail('network must not run'))
    command = f'lerobot-rollout --robot.type=so100_follower --policy.path={shlex.quote(str(case["checkpoint"]))}'
    report = check_robot_readiness(robot_command=command)
    assert report['kind'] == 'robot_readiness' and report['status'] == 'error'
    conflict = check_robot_readiness(robot_command=command, checkpoint='hf://org/other', fetch=True)
    assert conflict['status'] == 'error' and conflict['findings'][0]['code'] == 'readiness.policy_reference_conflict'


def test_missing_local_state_remains_error(case):
    path = healthy(case)
    (case['checkpoint'] / 'pre.safetensors').unlink()
    report = check_robot_readiness(robot_config=str(path), dataset=str(case['dataset']))
    assert report['status'] == 'error'
    assert any(f['status'] == 'error' and 'normalization' in f['code'] for f in report['findings'])


def test_dataset_failure_preserves_independent_camera_error(case):
    path = config(case, camera='other')
    report = check_robot_readiness(robot_config=str(path), dataset=str(case['dataset'].parent / 'missing'))
    assert report['status'] == 'error'
    assert report['axes']['cameras']['status'] == 'not_cleared'


def test_mutable_auto_follow_is_metadata_only_and_preserves_camera_error(case, monkeypatch):
    from policy_doctor import hub, readiness_intake
    from policy_doctor import readiness as engine
    path = config(case, camera='other')
    (case['checkpoint'] / 'train_config.json').write_text(json.dumps({'dataset': {'repo_id': 'org/training'}}))
    calls = []
    original = engine.readiness
    def recording(cp, robot, dataset=None, **kwargs):
        calls.append(dataset)
        return original(cp, robot, dataset, **kwargs)
    @contextmanager
    def resolve(source, **kwargs):
        assert source == 'hf://org/training'
        yield hub.HubSnapshot(case['dataset'], {'source': source, 'resolved_revision': 'a' * 40}, {})
    monkeypatch.setattr(engine, 'readiness', recording)
    monkeypatch.setattr(readiness_intake, 'resolve_readiness_snapshot', resolve)
    report = check_robot_readiness(robot_config=str(path), fetch=True)
    assert calls == [None]
    assert report['status'] == 'error' and report['dataset_context']['not_historical_training_evidence']
    assert 'contextual_comparison' not in report and 'current_dataset_metadata' in report


def test_extracted_outside_path_refused_before_any_intake(case, monkeypatch):
    from policy_doctor import hub
    monkeypatch.setattr(hub, '_request', lambda *a, **k: pytest.fail('network must not run'))
    with pytest.raises(InputError, match='outside'):
        check_robot_readiness(robot_command='lerobot-rollout --policy.path=/etc',
            dataset='hf://org/data', fetch=True, read_roots=[str(case['checkpoint'].parent)],
            base_directory=str(case['checkpoint'].parent))


def test_ambiguous_bare_native_ref_does_not_probe_unknown_cwd(case):
    report = check_robot_readiness(robot_command='lerobot-rollout --policy.path=org/model',
        read_roots=[str(case['checkpoint'])], base_directory=None)
    assert report['status'] == 'unknown' and report['findings'][0]['code'] == 'readiness.reference_ambiguous'


def test_native_local_directory_precedes_bare_hub_id(tmp_path):
    (tmp_path / 'org/model').mkdir(parents=True)
    plan = prepare_readiness(robot_command='lerobot-rollout --policy.path=org/model', base_directory=str(tmp_path))
    assert plan['checkpoint'] == str(tmp_path / 'org/model')
    assert 'local_path' in plan['policy_interpretation']


@pytest.mark.parametrize('options', [{'fetch': 1}, {'processor_state': 'true'}, {'max_bytes': True}, {'max_files': True}])
def test_strict_bounds(case, options):
    with pytest.raises(InputError):
        check_robot_readiness(robot_config=str(config(case)), **options)


def test_actual_stdio_keeps_axes_guards_and_permission_separation(case):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    path = healthy(case)
    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(path.parent)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                tool = next(t for t in tools if t.name == 'check_robot_readiness')
                assert tool.inputSchema['additionalProperties'] is False
                args = {'robot_config': str(path), 'dataset': str(case['dataset'])}
                for mode in ('concise', 'detailed'):
                    result = await client.call_tool(tool.name, dict(args, response_format=mode))
                    assert not result.isError
                    report = result.structuredContent
                    assert report['status'] == 'pass' and len(report['axes']) == 10
                for args in ({'robot_config': str(path), 'max_files': True},
                             {'robot_config': str(path), 'max_download_mb': True},
                             {'robot_config': str(path), 'exec': True},
                             {'robot_command': 'lerobot-rollout --policy.path=/etc'},
                             {'robot_command': 'lerobot-rollout --policy.path=org/model', 'checkpoint': 'hf://org/model'}):
                    result = await client.call_tool(tool.name, args)
                    assert result.isError
    asyncio.run(run())


def test_preflight_embeds_same_robot_axes_and_conflict(case):
    from policy_doctor.preflight import preflight
    path = healthy(case)
    report = preflight(str(case['dataset']), robot_config=str(path), video_headers=False)
    assert report['components']['robot_readiness']['status'] == 'pass'
    assert len(report['components']['robot_readiness']['axes']) == 10
    conflict = preflight(str(case['dataset']), checkpoint=str(case['checkpoint'].parent / 'other'),
        robot_config=str(path), video_headers=False)
    assert conflict['components']['robot_readiness']['status'] == 'error'


def test_preflight_impossible_remote_allocation_never_opens_http(case, monkeypatch):
    from policy_doctor import hub
    from policy_doctor.preflight import preflight
    monkeypatch.setattr(hub, '_request', lambda *a, **k: pytest.fail('network must not run'))
    report = preflight('hf://org/data', robot_config=str(config(case, policy='hf://org/model')),
        fetch=True, max_bytes=1)
    assert report['status'] == 'unknown'
    assert report['components']['dataset_intake']['status'] == 'unknown'
    assert report['components']['robot_readiness']['status'] == 'unknown'


def test_cli_unknown_axes_obey_fail_on_unknown(case, capsys):
    path = healthy(case)
    args = ['readiness', '--robot-config', str(path), '--fail-on', 'unknown', '--json']
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'
    assert main(['readiness', '--robot-config', str(path), '--dataset', str(case['dataset']), '--fail-on', 'unknown', '--json']) == 0


def test_setup_emits_separate_state_permission(case, capsys):
    from policy_doctor.setup_client import client_configuration
    root = str(case['checkpoint'].parent)
    text = client_configuration([root], allow_hub_processor_state=True, clients=('json',))
    data = json.loads(text.split('\n', 1)[1].split('\n\n# Notes')[0])
    args = data['mcpServers']['roboclue']['args']
    assert '--allow-hub-processor-state' in args
    assert '--allow-hub' not in args and '--allow-hub-datasets' not in args
    assert main(['setup', '--root', root, '--allow-hub-processor-state', '--client', 'json']) == 0
    assert '--allow-hub-processor-state' in capsys.readouterr().out
    with pytest.raises(InputError, match='booleans'):
        client_configuration([root], allow_hub_processor_state=1)


def test_command_text_is_never_shell_executed(case):
    marker = case['checkpoint'].parent / 'must-not-exist'
    command = ('lerobot-rollout --robot.type=so100_follower --policy.path='
               + shlex.quote(str(case['checkpoint'])) + '; touch ' + shlex.quote(str(marker)))
    report = check_robot_readiness(robot_command=command)
    assert report['kind'] == 'robot_readiness' and not marker.exists()


def test_extracted_symlink_input_refused_before_network(case, monkeypatch):
    from policy_doctor import hub
    alias = case['checkpoint'].parent / 'linked'
    alias.symlink_to(case['checkpoint'], target_is_directory=True)
    monkeypatch.setattr(hub, '_request', lambda *a, **k: pytest.fail('network must not run'))
    with pytest.raises(InputError, match='symlink'):
        check_robot_readiness(robot_command='lerobot-rollout --policy.path=' + shlex.quote(str(alias)),
            dataset='hf://org/data', fetch=True, read_roots=[str(alias.parent)], base_directory=str(alias.parent))


def test_operator_permissions_do_not_transitively_grant_state_or_rows(case, monkeypatch):
    from policy_doctor import readiness_workflow
    from policy_doctor.server import create_server
    seen = []
    def capture(**kwargs):
        seen.append(kwargs)
        return {'kind': 'robot_readiness', 'status': 'unknown', 'findings': [], 'axes': {}}
    monkeypatch.setattr(readiness_workflow, 'check_robot_readiness', capture)
    async def run():
        for opts, expected in [({}, (False, False)), ({'allow_hub': True}, (True, False)),
                               ({'allow_hub_datasets': True}, (False, False)),
                               ({'allow_hub_processor_state': True}, (True, True))]:
            server = create_server([str(case['checkpoint'].parent)], **opts)
            await server.call_tool('check_robot_readiness', {'robot_command': 'lerobot-rollout', 'response_format': 'detailed'})
            assert (seen[-1]['fetch'], seen[-1]['processor_state']) == expected
    asyncio.run(run())
    with pytest.raises(InputError, match='boolean'):
        create_server([str(case['checkpoint'].parent)], allow_hub_processor_state=1)


def test_dataset_alias_effective_root_authorized_before_intake(case, monkeypatch):
    from policy_doctor import hub
    metadata = case['dataset'] / 'meta/info.json'
    monkeypatch.setattr(hub, '_request', lambda *a, **k: pytest.fail('network must not run'))
    with pytest.raises(InputError, match='outside'):
        check_robot_readiness(robot_command='lerobot-rollout --policy.path=org/model',
            checkpoint='hf://org/model', dataset=str(metadata), fetch=True,
            read_roots=[str(metadata.parent)], base_directory=str(metadata.parent))


def test_native_local_config_file_precedes_hub_and_matches_directory(tmp_path):
    folder = tmp_path / 'org'
    folder.mkdir()
    path = folder / 'config.json'
    path.write_text('{}')
    plan = prepare_readiness(robot_command='lerobot-rollout --policy.path=org/config.json',
        checkpoint=str(folder), base_directory=str(tmp_path))
    assert plan['embedded_policy'] == str(path) and not plan['policy_conflict']


def test_failed_preflight_snapshot_is_not_reacquired(case, monkeypatch):
    from policy_doctor import hub, readiness_intake
    from policy_doctor.preflight import preflight
    calls = []
    @contextmanager
    def unavailable(*args, **kwargs):
        calls.append(args[0])
        raise InputError('first pinned dataset intake unavailable')
        yield
    monkeypatch.setattr(hub, 'resolve_dataset_content', unavailable)
    monkeypatch.setattr(readiness_intake, 'resolve_readiness_snapshot',
        lambda *a, **k: pytest.fail('failed shared dataset must not be fetched again'))
    path = healthy(case)
    data = json.loads(path.read_text())
    data['robot']['cameras'] = {'other': {'type': 'opencv', 'width': 32, 'height': 32, 'fps': 30}}
    path.write_text(json.dumps(data))
    report = preflight('hf://org/data', robot_config=str(path), fetch=True)
    assert calls == ['hf://org/data']
    ready = report['components']['robot_readiness']
    assert ready['status'] == 'error' and ready['axes']['cameras']['status'] == 'not_cleared'
    assert any(f['code'] == 'readiness.dataset_snapshot_unavailable' for f in ready['findings'])


def test_concise_bound_keeps_every_axis_and_discloses_reason_omissions():
    from policy_doctor.server import _summary
    value = {'kind': 'robot_readiness', 'status': 'unknown', 'summary': 'S' * 500,
        'scope': 'x' * 512, 'findings': [], 'next_actions': ['R' * 400 for _ in range(6)],
        'axes': {f'axis{i}': {'status': 'unknown', 'reasons': ['Q' * 1000] * 4} for i in range(16)},
        'dataset_context': {'source': 'x' * 400}}
    report = _summary(value)
    assert report['status'] == 'unknown' and len(report['axes']) == 16
    assert len(json.dumps(report).encode()) <= 8192
    assert any(axis['reasons_omitted'] for axis in report['axes'].values())


def test_listed_withheld_state_is_unknown_before_axes_but_camera_error_survives(case, monkeypatch):
    from policy_doctor import hub, readiness_intake
    path = healthy(case)
    data = json.loads(path.read_text())
    data['policy']['path'] = 'org/policy'
    data['robot']['cameras'] = {'other': {'type': 'opencv', 'width': 32, 'height': 32, 'fps': 30}}
    path.write_text(json.dumps(data))
    (case['checkpoint'] / 'pre.safetensors').unlink()
    (case['checkpoint'] / 'post.safetensors').unlink()
    @contextmanager
    def resolve(source, **kwargs):
        yield hub.HubSnapshot(case['checkpoint'], {'source': source, 'resolved_revision': 'a' * 40},
            {'policy_preprocessor.json': ['/steps/3'], 'policy_postprocessor.json': ['/steps/0']})
    monkeypatch.setattr(readiness_intake, 'resolve_readiness_snapshot', resolve)
    report = check_robot_readiness(robot_config=str(path), checkpoint='hf://org/policy',
        dataset=str(case['dataset']), fetch=True, base_directory=str(path.parent))
    assert report['status'] == 'error' and report['axes']['cameras']['status'] == 'not_cleared'
    assert report['axes']['normalization']['status'] == 'unknown'
    assert report['axes']['state']['status'] == 'unknown' and report['axes']['action']['status'] == 'unknown'
    assert not any(f['status'] == 'error' and f['code'] == 'normalization.statistics' for f in report['findings'])


def test_embedded_model_permission_does_not_require_or_grant_public_rows(case, monkeypatch):
    from policy_doctor import hub, readiness_intake
    from policy_doctor.server import create_server
    path = healthy(case)
    data = json.loads(path.read_text())
    data['policy']['path'] = 'org/policy'
    path.write_text(json.dumps(data))
    calls = []
    @contextmanager
    def resolve(source, **kwargs):
        calls.append((source, kwargs['fetch'], kwargs['processor_state']))
        yield hub.HubSnapshot(case['checkpoint'], {'source': source, 'resolved_revision': 'a' * 40, 'files': []}, {})
    monkeypatch.setattr(readiness_intake, 'resolve_readiness_snapshot', resolve)
    async def run():
        args = {'source': str(case['dataset']), 'robot_config': str(path),
                'checkpoint': 'hf://org/policy', 'response_format': 'detailed'}
        for permission, expected_state in [({'allow_hub': True}, False),
                                           ({'allow_hub_processor_state': True}, True)]:
            server = create_server([str(path.parent)], **permission)
            response = await server.call_tool('preflight_check', args)
            report = response[1] if isinstance(response, tuple) else response
            assert report['components']['robot_readiness']['status'] == 'pass'
            assert calls[-1] == ('hf://org/policy', True, expected_state)
            before = len(calls)
            with pytest.raises(Exception, match='allow-hub-datasets'):
                await server.call_tool('preflight_check', dict(args, source='hf://org/data'))
            assert len(calls) == before
        server = create_server([str(path.parent)], allow_hub_datasets=True)
        with pytest.raises(Exception, match='robot-policy intake'):
            await server.call_tool('preflight_check', args)
    asyncio.run(run())


def test_mcp_read_root_never_establishes_native_command_cwd(case):
    from policy_doctor.server import create_server
    async def run():
        server = create_server([str(case['checkpoint'].parent)])
        result = await server.call_tool('check_robot_readiness', {
            'robot_command': 'lerobot-rollout --policy.path=org/model', 'response_format': 'detailed'})
        report = result[1] if isinstance(result, tuple) else result
        assert report['status'] == 'unknown'
        assert report['findings'][0]['code'] == 'readiness.reference_ambiguous'
        assert report['relative_path_base'] is None
        for value in ('relative/path', '/etc'):
            with pytest.raises(Exception, match='absolute|outside'):
                await server.call_tool('check_robot_readiness', {'robot_command': 'lerobot-rollout',
                    'command_directory': value})
    asyncio.run(run())


def test_explicit_authorized_command_directory_preserves_native_local_selection(case):
    from policy_doctor.server import create_server
    path = healthy(case)
    relative = case['checkpoint'].relative_to(path.parent)
    async def run():
        server = create_server([str(path.parent)])
        result = await server.call_tool('check_robot_readiness', {
            'robot_command': 'lerobot-rollout --robot.type=so100_follower --policy.path=' + str(relative),
            'command_directory': str(path.parent), 'dataset': str(case['dataset']), 'response_format': 'detailed'})
        report = result[1] if isinstance(result, tuple) else result
        assert report['relative_path_base'] == str(path.parent)
        assert report['checkpoint']['root'] == str(case['checkpoint'])
    asyncio.run(run())
