"""Interface authorization/dispatch controls; native evidence parity is separate."""
import asyncio
import json
import os
import sys
import types
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.server import create_server

KIND = 'robot_video_reference_recovery'


def fake_engine(monkeypatch):
    calls = []
    module = types.ModuleType('policy_doctor.video_recovery')
    for name in ('recover_video_reference', 'reproduce_video_recovery', 'check_video_recovery_candidate'):
        def invoke(*args, _name=name):
            calls.append((_name, args))
            return {'kind': KIND, 'status': 'unknown', 'original_status': 'error'}
        setattr(module, name, invoke)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_media_network_creator_requires_separate_explicit_operator_optin(tmp_path):
    with pytest.raises(InputError, match='case root'):
        create_server([str(tmp_path)], allow_hub_media=True)
    for options, expected in [({}, 29), ({'allow_hub': True}, 30),
                              ({'case_root': str(tmp_path)}, 31),
                              ({'case_root': str(tmp_path), 'allow_hub_media': True}, 34)]:
        tools = create_server([str(tmp_path)], **options)._tool_manager._tools
        assert len(tools) == expected
        assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
        assert ('recover_dataset_video_reference' in tools) == bool(options.get('allow_hub_media'))
    tool = tools['recover_dataset_video_reference']
    assert tool.annotations.readOnlyHint is False and tool.annotations.openWorldHint is True
    assert tool.annotations.destructiveHint is False and tool.annotations.idempotentHint is False


@pytest.mark.parametrize('destination', ['empty', 'root', 'outside', 'existing', 'symlink', 'symlink_parent'])
def test_media_destination_refused_before_any_network_creator_call(tmp_path, monkeypatch, destination):
    calls = fake_engine(monkeypatch)
    output, outside = tmp_path / 'output', tmp_path / 'outside'
    output.mkdir()
    outside.mkdir()
    (output / 'existing').mkdir()
    (output / 'link').symlink_to(output / 'missing')
    (output / 'parentlink').symlink_to(outside, target_is_directory=True)
    paths = {'empty': '', 'root': output, 'outside': outside / 'new', 'existing': output / 'existing',
             'symlink': output / 'link', 'symlink_parent': output / 'parentlink' / 'new'}
    tool = create_server([str(output)], case_root=str(output), allow_hub_media=True)._tool_manager._tools['recover_dataset_video_reference']
    with pytest.raises(InputError):
        tool.fn('org/dataset', 'a' * 40, 'b' * 40, 40, 'observation.images.camera', str(paths[destination]))
    assert calls == []
    assert not (outside / 'new').exists()


def test_media_creator_argument_routing_and_cli_unknown_exit(tmp_path, monkeypatch, capsys):
    calls = fake_engine(monkeypatch)
    output = tmp_path / 'output'
    output.mkdir()
    tool = create_server([str(output)], case_root=str(output), allow_hub_media=True)._tool_manager._tools['recover_dataset_video_reference']
    tool.fn('org/data', 'a' * 40, 'b' * 40, 40, 'observation.images.camera', str(output / 'new'))
    assert calls[-1] == ('recover_video_reference', ('org/data', 'a' * 40, 'b' * 40, 40, 'observation.images.camera', str(output / 'new')))
    status = main(['recover-video-reference', 'hf://org/data', str(output / 'cli'), '--revision', 'a' * 40,
                   '--original-revision', 'b' * 40, '--episode-index', '77', '--video-key', 'observation.images.camera', '--json'])
    assert status == 3 and json.loads(capsys.readouterr().out)['original_status'] == 'error'
    assert calls[-1] == ('recover_video_reference', ('org/data', 'a' * 40, 'b' * 40, 77, 'observation.images.camera', str(output / 'cli')))


def test_both_offline_candidate_paths_authorized_before_engine_intake(tmp_path, monkeypatch):
    calls = fake_engine(monkeypatch)
    case, later, outside = [tmp_path / p for p in ('case', 'later', 'outside')]
    for path in (case, later, outside):
        path.mkdir()
        (path / 'manifest.json').write_text(json.dumps({'kind': KIND}))
    tools = create_server([str(case), str(later)])._tool_manager._tools
    tools['reproduce_incident_case'].fn(str(case))
    tools['check_incident_candidate'].fn(str(case), str(later))
    assert [x[0] for x in calls] == ['reproduce_video_recovery', 'check_video_recovery_candidate']
    before = len(calls)
    for path, candidate in [(outside, later), (case, outside), (case, '')]:
        with pytest.raises(InputError):
            tools['check_incident_candidate'].fn(str(path), str(candidate))
    assert len(calls) == before


def test_cli_shared_replay_and_candidate_frontdoors_dispatch_without_network(tmp_path, monkeypatch, capsys):
    calls = fake_engine(monkeypatch)
    case = tmp_path / 'case'
    case.mkdir()
    (case / 'manifest.json').write_text(json.dumps({'kind': KIND}))
    assert main(['reproduce-case', str(case), '--json']) == 3
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'
    assert main(['check-case', str(case), str(tmp_path / 'later'), '--json']) == 3
    capsys.readouterr()
    assert calls == [('reproduce_video_recovery', (str(case),)),
                     ('check_video_recovery_candidate', (str(case), str(tmp_path / 'later')))]


def test_discovery_recognizes_video_case_without_calling_its_engine(tmp_path, monkeypatch):
    calls = fake_engine(monkeypatch)
    case = tmp_path / 'case'
    case.mkdir()
    (case / 'manifest.json').write_text(json.dumps({'kind': KIND}))
    (case / 'READY.json').write_text('{}')
    from policy_doctor.investigation import investigate_run
    result = investigate_run(str(case))
    assert result['status'] == 'unknown' and result['audits'] == []
    assert result['inventory']['retained_checks'] == [str(case)]
    assert result['next_actions'][0]['tool'] == 'reproduce_incident_case'
    assert calls == []


def test_actual_stdio_catalog_exposes_media_creator_only_with_operator_flag(tmp_path):
    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path), '--case-root', str(tmp_path), '--allow-hub-media'],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                tools = {t.name: t for t in (await client.list_tools()).tools}
                assert len(tools) == 34
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
                assert tools['recover_dataset_video_reference'].annotations.openWorldHint is True
                refused = await client.call_tool('recover_dataset_video_reference', {
                    'repo_id': 'org/data', 'revision': 'a' * 40, 'original_revision': 'b' * 40,
                    'episode_index': 40, 'video_key': 'observation.images.camera', 'destination': ''})
                assert refused.isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
