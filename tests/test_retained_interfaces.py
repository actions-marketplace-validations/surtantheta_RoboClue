import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.demo import create_policy_demo
from policy_doctor.retained import retain_check

SRC = str(Path(__file__).resolve().parents[1] / 'src')


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'policy_doctor', *map(str, args)],
        env=dict(os.environ, PYTHONPATH=SRC), capture_output=True, text=True, timeout=30)


def recording(path, *, changed=False, missing=False):
    path.write_text('frame_index,' + ('other' if missing else 'action') + '\n0,0\n1,' + ('9' if changed else '1') + '\n2,2\n')


def test_metadata_existing_check_can_save_and_replay_then_test_new_inputs(tmp_path):
    original = tmp_path / 'native'
    create_policy_demo(str(original))
    case = tmp_path / 'retained'
    result = cli('check', original / 'checkpoint', '--dataset', original / 'dataset', '--scope', 'metadata', '--case', case)
    assert result.returncode == 1, result.stderr
    assert 'Saved check: pass; finding: error' in result.stdout
    assert 'action.processor_order_names' in result.stdout
    assert 'reproduce-case' in result.stdout and 'check-case' in result.stdout
    corrected = tmp_path / 'later'
    shutil.copytree(original / 'corrected_dataset', corrected)
    shutil.rmtree(original)
    moved = tmp_path / 'moved case'
    case.rename(moved)
    replay = cli('reproduce-case', moved, '--json')
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout)['original_status'] == 'error'
    later = cli('check-case', moved, corrected, '--json')
    assert later.returncode == 0, later.stderr
    assert json.loads(later.stdout)['candidate_status'] == 'pass'


def test_recording_existing_command_retains_finding_exit_and_frozen_reference(tmp_path):
    reference, candidate = tmp_path / 'reference.csv', tmp_path / 'candidate.csv'
    recording(reference)
    recording(candidate, changed=True)
    case = tmp_path / 'recording-case'
    result = cli('compare-rollouts', reference, candidate, '--alignment-key', 'frame_index', '--case', case, '--json')
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)['original_status'] == 'different'
    reference.unlink()
    candidate.unlink()
    replay = cli('reproduce-case', case)
    assert replay.returncode == 0, replay.stderr
    for name, flags, code in [('healthy', {}, 0), ('again', {'changed': True}, 1), ('missing', {'missing': True}, 3)]:
        path = tmp_path / f'{name}.csv'
        recording(path, **flags)
        checked = cli('check-case', case, path, '--json')
        assert checked.returncode == code, checked.stderr
        assert json.loads(checked.stdout)['candidate_evaluated'] is True


def test_explicit_retention_command_and_output_conflicts_are_honest(tmp_path):
    reference = tmp_path / 'reference.csv'
    recording(reference)
    case = tmp_path / 'case'
    saved = cli('retain-check', 'recording-equivalence', reference, case, '--reference', reference,
                '--alignment-key', 'frame_index', '--json')
    assert saved.returncode == 0, saved.stderr
    assert json.loads(saved.stdout)['original_status'] == 'pass'
    refused = cli('compare-rollouts', reference, reference, '--case', tmp_path / 'not-created', '--output', tmp_path / 'report.json')
    assert refused.returncode == 2 and not (tmp_path / 'not-created').exists()
    refused = cli('check', 'hf://lerobot/smolvla_base', '--fetch', '--case', tmp_path / 'no-network')
    assert refused.returncode == 2 and not (tmp_path / 'no-network').exists()
    for args in [('reproduce-case', ''), ('check-case', case, '')]:
        refused = cli(*args)
        assert refused.returncode == 2


def test_actual_mcp_optin_retention_uses_existing_read_tools_and_guards(tmp_path):
    inputs, outputs, outside = [tmp_path / name for name in ('inputs', 'outputs', 'outside')]
    for path in (inputs, outputs, outside):
        path.mkdir()
    reference, candidate = inputs / 'reference.csv', inputs / 'candidate.csv'
    recording(reference)
    recording(candidate, changed=True)
    native = inputs / 'policy'
    create_policy_demo(str(native))

    async def exercise():
        for enabled in (False, True):
            args = ['-m', 'policy_doctor', 'serve', '--root', str(inputs), '--root', str(outputs)]
            if enabled:
                args += ['--case-root', str(outputs)]
            params = StdioServerParameters(command=sys.executable, args=args, env=dict(os.environ, PYTHONPATH=SRC))
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
                    assert len(tools) == (31 if enabled else 29)
                    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
                    assert ('retain_artifact_check' in tools) is enabled
                    if not enabled:
                        continue
                    assert tools['retain_artifact_check'].annotations.readOnlyHint is False
                    request = {'check': 'recording_equivalence', 'source': str(candidate),
                               'reference': str(reference), 'destination': str(outputs / 'case'),
                               'parameters': {'alignment_key': 'frame_index'}}
                    for changed in ({'destination': str(outside / 'case')}, {'source': ''}, {'reference': ''},
                                    {'source': str(outside / 'private.csv')}, {'dataset': ''},
                                    {'parameters': {'alignment_key': 'frame_index', 'shell': 'ignored'}}):
                        refused = await client.call_tool('retain_artifact_check', dict(request, **changed))
                        assert refused.isError, changed
                    made = await client.call_tool('retain_artifact_check', request)
                    assert not made.isError, made.content
                    assert made.structuredContent['original_status'] == 'different'
                    replay = await client.call_tool('reproduce_incident_case', {'directory': str(outputs / 'case')})
                    assert not replay.isError and replay.structuredContent['reproduction_matches']
                    checked = await client.call_tool('check_incident_candidate', {'directory': str(outputs / 'case'), 'candidate': str(reference)})
                    assert not checked.isError and checked.structuredContent['candidate_status'] == 'pass'
                    for tool, arguments in [('reproduce_incident_case', {'directory': ''}),
                                            ('check_incident_candidate', {'directory': str(outputs / 'case'), 'candidate': ''})]:
                        refused = await client.call_tool(tool, arguments)
                        assert refused.isError
                    metadata = await client.call_tool('retain_artifact_check', {'check': 'policy_metadata',
                        'source': str(native / 'checkpoint'), 'dataset': str(native / 'dataset'),
                        'destination': str(outputs / 'metadata')})
                    assert not metadata.isError, metadata.content
                    later = await client.call_tool('check_incident_candidate', {'directory': str(outputs / 'metadata'),
                        'candidate': str(native / 'corrected_dataset')})
                    assert not later.isError and later.structuredContent['candidate_status'] == 'pass'

    asyncio.run(exercise())
    assert not (outside / 'case').exists()


def test_metadata_alias_candidate_authorizes_effective_dataset_root(tmp_path):
    native = tmp_path / 'native'
    create_policy_demo(str(native))
    case = tmp_path / 'case'
    retain_check('policy_metadata', native / 'checkpoint', case, dataset=native / 'dataset')
    from policy_doctor.artifacts import InputError
    from policy_doctor.server import create_server

    tools = create_server([str(case), str(native / 'corrected_dataset/meta')])._tool_manager._tools
    try:
        tools['check_incident_candidate'].fn(str(case), str(native / 'corrected_dataset/meta/info.json'))
    except InputError as error:
        assert 'outside' in str(error)
    else:
        raise AssertionError('A metadata alias must not expand beyond an authorized root')
