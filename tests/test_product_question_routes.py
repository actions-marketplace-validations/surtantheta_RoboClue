"""Question-aware product routing against real engines and actual MCP stdio.

Generated controls reproduce the observed file/metadata/units fault signatures.
They are not human trials or additional native framework qualification.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import test_action_conventions as convention_fixtures
import test_groot_contract_n15 as groot_fixtures
import test_preflight_action_conventions as openpi_fixtures
import test_salvage as salvage_fixtures
import test_units_manifest as units_fixtures
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.groot_contract import check_groot_dataset
from policy_doctor.product_cli import main
from policy_doctor.salvage import check_salvage
from policy_doctor.sim_guards import check_sim_provenance
from policy_doctor.units_manifest import units_manifest

openpi_source = convention_fixtures.openpi_source
SRC = Path(__file__).parents[1] / 'src'
BASIC = {'choose_workflow', 'dataset_health_card', 'check_training_targets',
         'triage_flight_recording', 'assess_upgrade_impact', 'preflight_check',
         'check_sim_provenance', 'check_recording_salvage'}


def invoke(capsys, *arguments):
    try:
        code = main([str(a) for a in arguments])
    except SystemExit as error:
        code = error.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def plain(text):
    lines = text.strip().splitlines()
    assert len(lines) == 3, text
    assert lines[0].startswith('Verdict:') and lines[1].startswith(('Problem:', 'Details:')) and lines[2].startswith('Next:'), text
    assert not lines[1].removeprefix('Problem: ').startswith('{'), text
    assert 'policy-doctor' not in text and 'policy-deployment-doctor' not in text
    return lines


def component(report, kind):
    pending = [report]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if item.get('kind') == kind:
                return item
            pending.extend(item.get('components', {}).values())
            if isinstance(item.get('report'), dict):
                pending.append(item['report'])
    raise AssertionError(f'{kind} absent from real report: {report}')


def hashes(root):
    root = Path(root)
    files = [root] if root.is_file() else [p for p in root.rglob('*') if p.is_file()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def isaac(path, *, version=1):
    with h5py.File(path, 'w') as handle:
        if version is not None:
            handle.attrs['format_version'] = version
        data = handle.create_group('data')
        data.attrs['env_args'] = json.dumps({'env_name': 'Isaac-Control-v0', 'type': 2})
        group = data.create_group('demo_0')
        group.create_dataset('actions', data=np.zeros((2, 7)))
        # Identity XYZW is intentionally ambiguous from upright robot values.
        group.create_dataset('initial_state/articulation/robot/root_pose', data=np.array([[0, 0, 0, 0., 0, 0, 1.]]))
    return path


@pytest.mark.parametrize('extension', ['.hdf5', '.h5'])
def test_hdf5_front_door_reaches_real_simulator_check(tmp_path, capsys, extension):
    source = isaac(tmp_path / ('source' + extension), version=None)
    before = hashes(source)
    expected = check_sim_provenance(str(source))
    code, out, err = invoke(capsys, 'check', source, '--json')
    assert not err and code == {'pass': 0, 'warning': 1, 'unknown': 3, 'error': 1}[expected['status']]
    observed = component(json.loads(out), 'sim_provenance')
    assert observed['status'] == expected['status']
    assert [(f['code'], f['status']) for f in observed['findings']] == [(f['code'], f['status']) for f in expected['findings']]
    assert hashes(source) == before
    _, out, err = invoke(capsys, 'check', source)
    assert not err
    lines = plain(out)
    assert 'format_version' in out or 'quaternion' in out.lower(), lines
    assert 'video-headers' not in lines[2]


def test_sim_pair_correction_preserves_original_comparison_and_unknown(tmp_path, capsys):
    source = isaac(tmp_path / 'source.hdf5')
    wrong = isaac(tmp_path / 'merged.hdf5', version=None)
    fixed = isaac(tmp_path / 'fixed.hdf5')
    before = {**hashes(source), **hashes(wrong), **hashes(fixed)}
    code, out, err = invoke(capsys, 'check', source, '--converted', wrong, '--json')
    assert code == 1 and not err
    bad = component(json.loads(out), 'sim_provenance')
    assert any(f['code'] == 'isaaclab.conversion_quaternion_order' and f['status'] == 'error' for f in bad['findings'])
    code, out, err = invoke(capsys, 'check', source, '--converted', fixed, '--json')
    good = component(json.loads(out), 'sim_provenance')
    expected = check_sim_provenance(str(source), converted=str(fixed))
    assert not err and good['status'] == expected['status']
    assert any(f['code'] == 'isaaclab.conversion_quaternion_order' and f['status'] == 'pass' for f in good['findings'])
    assert code == {'pass': 0, 'warning': 1, 'unknown': 3, 'error': 1}[expected['status']]
    assert {**hashes(source), **hashes(wrong), **hashes(fixed)} == before
    _, out, err = invoke(capsys, 'check', source, '--converted', fixed)
    lines = plain(out)
    assert not err and ('match' in out.lower() or 'pass' in lines[0].lower() or 'consistent' in out.lower())
    assert 'video-headers' not in lines[2]


@pytest.mark.parametrize('corrected', [False, True])
def test_so101_question_compares_real_units_not_missing_videos(tmp_path, capsys, corrected):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    command = units_fixtures.command(*(['--robot.use_degrees=false'] if corrected else []))
    before = hashes(root)
    expected = units_manifest(root, robot_command=command)
    code, out, err = invoke(capsys, 'check', root, '--robot-command', command, '--lerobot-version', '0.6.2', '--json')
    assert not err and code == (0 if corrected else 1)
    observed = component(json.loads(out), 'units_manifest')
    assert observed['status'] == expected['status'] == ('pass' if corrected else 'error')
    assert [(f['code'], f['status']) for f in observed['findings']] == [(f['code'], f['status']) for f in expected['findings']]
    _, out, err = invoke(capsys, 'check', root, '--robot-command', command, '--lerobot-version', '0.6.2')
    assert not err
    lines = plain(out)
    assert '-100..100' in out
    assert 'video-headers' not in lines[2] and 'video evidence' not in lines[1].lower()
    if corrected:
        assert any(term in out.lower() for term in ('match', 'consistent', 'pass'))
        assert 'PARTIAL' not in lines[0]
    else:
        assert 'degrees' in out and '--robot.use_degrees=false' in out
    assert hashes(root) == before


def test_robot_command_is_parsed_never_executed(tmp_path, capsys):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    marker = tmp_path / 'should-not-exist'
    command = units_fixtures.command() + '; touch ' + str(marker)
    code, _, err = invoke(capsys, 'check', root, '--robot-command', command)
    assert code in (1, 2, 3)
    assert not marker.exists() and 'Traceback' not in err


@pytest.mark.parametrize('via_modality', [False, True])
def test_groot_config_selector_runs_named_version_contract(tmp_path, capsys, via_modality):
    root = groot_fixtures._dataset(tmp_path / 'dataset')
    source = root / 'meta/modality.json' if via_modality else root
    before = hashes(root)
    expected = check_groot_dataset(root, config_name='so100_dualcam', groot_version='n1.5')
    code, out, err = invoke(capsys, 'check', source, '--groot-version', 'n1.5', '--groot-config-name', 'so100_dualcam', '--json')
    assert not err and code == {'pass': 0, 'warning': 0, 'unknown': 3, 'error': 1}[expected['status']]
    report = component(json.loads(out), 'groot_dataset_contract')
    assert report['groot_version'] == 'n1.5' and report['modality_config_name'] == 'so100_dualcam'
    assert [(f['code'], f['status']) for f in report['findings']] == [(f['code'], f['status']) for f in expected['findings']]
    _, out, err = invoke(capsys, 'check', source, '--groot-version', 'n1.5', '--groot-config-name', 'so100_dualcam')
    assert not err
    lines = plain(out)
    assert 'GR00T' in out and 'n1.5' in out.lower()
    assert 'video-headers' not in lines[2]
    assert hashes(root) == before


def test_groot_wrong_config_does_not_prescribe_editing_public_data(tmp_path, capsys):
    root = groot_fixtures._dataset(tmp_path / 'dataset')
    modality = root / 'meta/modality.json'
    value = json.loads(modality.read_text())
    del value['annotation']['human.task_description']
    value['annotation']['human.coarse_action'] = {'original_key': 'task_index'}
    modality.write_text(json.dumps(value))
    before = hashes(root)
    code, out, err = invoke(capsys, 'check', root, '--groot-version', 'n1.5', '--groot-config-name', 'so100_dualcam')
    assert code == 1 and not err
    lines = plain(out)
    assert 'config' in out.lower()
    assert not any(phrase in out.lower() for phrase in ('add annotation', 'edit modality', 'rewrite modality'))
    assert 'video-headers' not in lines[2]
    _, machine, err = invoke(capsys, 'check', root, '--groot-version', 'n1.5', '--groot-config-name', 'so100_dualcam', '--json')
    assert not err
    report = component(json.loads(machine), 'groot_dataset_contract')
    language = next(f for f in report['findings'] if f['code'] == 'groot.language_key' and f['status'] == 'error')
    assert not language['remedy'].lower().startswith('add annotation')
    assert 'config' in language['remedy'].lower()
    assert hashes(root) == before


@pytest.mark.parametrize('fault', ['orphan', 'truncated'])
def test_interrupted_native_dataset_routes_to_salvage(tmp_path, capsys, fault):
    if fault == 'orphan':
        root = salvage_fixtures.write(tmp_path / 'dataset',
            [salvage_fixtures.rows_for([(0, 5), (1, 6), (2, 7)])],
            [salvage_fixtures.meta_for([(0, 5), (1, 6)])], 2, 11)
    else:
        root = salvage_fixtures.write(tmp_path / 'dataset',
            [salvage_fixtures.rows_for([(0, 5)]), salvage_fixtures.rows_for([(1, 6)], start=5)],
            [salvage_fixtures.meta_for([(0, 5)])], 2, 11)
        salvage_fixtures.strip_footer(root / 'data/chunk-000/file-001.parquet')
    expected = check_salvage(root)
    before = hashes(root)
    code, out, err = invoke(capsys, 'check', root, '--json')
    assert not err and code == 1
    report = component(json.loads(out), 'recording_salvage')
    assert report['status'] == expected['status'] == 'error'
    assert any(f['code'] == 'salvage.resume_hazard' and f['status'] == 'error' for f in report['findings'])
    _, out, err = invoke(capsys, 'check', root)
    assert not err
    lines = plain(out)
    assert 'resume' in out.lower()
    assert 'Passed: dataset statistics' not in out
    assert 'video-headers' not in lines[2]
    assert hashes(root) == before


def test_explicit_recovery_runs_on_even_healthy_recording(tmp_path, capsys):
    root = salvage_fixtures.healthy(tmp_path)
    code, out, err = invoke(capsys, 'check', root, '--question', 'recovery', '--json')
    assert not err and code == 0
    assert component(json.loads(out), 'recording_salvage')['status'] == 'pass'


def test_openpi_question_preserves_delta_scope_not_pipeline_certification(tmp_path, capsys, openpi_source):
    root = openpi_fixtures.libero_like(tmp_path / 'libero')
    before = hashes(root)
    code, out, err = invoke(capsys, 'check', root, '--openpi-config', 'pi0_libero', '--openpi-source-root', openpi_source, '--json')
    assert not err and code == 1  # The scoped delta match does not waive missing raw adapter keys.
    report = component(json.loads(out), 'action_conventions')
    assert any(f['code'] == 'convention.relative_on_incremental' and f['status'] == 'pass' for f in report['findings'])
    _, out, err = invoke(capsys, 'check', root, '--openpi-config', 'pi0_libero', '--openpi-source-root', openpi_source)
    assert not err
    lines = plain(out)
    assert any(term in out.lower() for term in ('delta', 'incremental', 'transform'))
    assert 'pipeline compatible' not in out.lower() and 'pipeline is compatible' not in out.lower()
    assert 'input keys: MISMATCH' in out and 'missing raw keys' in out
    assert 'video-headers' not in lines[2]
    assert hashes(root) == before


async def mcp_session(root, outside, calls):
    params = StdioServerParameters(command=sys.executable,
        args=['-B', '-m', 'roboclue', 'mcp', 'serve', '--root', str(root)],
        env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1'))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        initialized = await session.initialize()
        assert initialized.serverInfo.name == 'RoboClue'
        tools = (await session.list_tools()).tools
        assert {tool.name for tool in tools} == BASIC
        assert all(tool.annotations.readOnlyHint and not tool.annotations.destructiveHint for tool in tools)
        results = []
        for name, arguments in calls:
            response = await session.call_tool(name, arguments)
            assert not response.isError, response
            assert isinstance(response.structuredContent, dict)
            results.append({'report': response.structuredContent, 'text': response.content[0].text})
        refused = await session.call_tool('preflight_check', {'source': str(outside)})
        assert refused.isError
        return results


def test_default_mcp_can_answer_units_question_and_preserves_uncompared_scope(tmp_path):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    outside = tmp_path / 'outside'
    outside.mkdir()
    responses = asyncio.run(mcp_session(root, outside, [
        ('dataset_health_card', {'source': str(root), 'units': True}),
        ('preflight_check', {'source': str(root), 'robot_command': units_fixtures.command(), 'lerobot_version': '0.6.2'}),
        ('preflight_check', {'source': str(root), 'robot_command': units_fixtures.command('--robot.use_degrees=false'), 'lerobot_version': '0.6.2'}),
    ]))
    named = json.dumps(responses[0])
    assert 'RANGE_M100_100' in named
    assert any(phrase in named.lower() for phrase in ('dataset only', 'dataset-only', 'no robot', 'not compared', 'uncompared', 'does not compare')), named
    bad = json.dumps(responses[1])
    assert 'units.joint_mode_mismatch' in bad and '--robot.use_degrees=false' in bad
    assert responses[1]['report']['status'] == 'error'
    assert responses[2]['report']['components']['units']['status'] == 'pass'


def test_no_explain_command_loop_and_no_python_dict_headline(tmp_path, capsys):
    root = salvage_fixtures.write(tmp_path / 'dataset',
        [salvage_fixtures.rows_for([(0, 5), (1, 6), (2, 7)])],
        [salvage_fixtures.meta_for([(0, 5), (1, 6)])], 2, 11)
    _, out, err = invoke(capsys, 'check', root)
    assert not err
    command = shlex.split(plain(out)[2].removeprefix('Next: '))
    assert command[0] == 'roboclue'
    # A supported Next command can be copied without assuming shell prose is code.
    if '--explain' in command:
        _, explained, err = invoke(capsys, *command[1:])
        assert not err
        first = explained.splitlines()[:3]
        assert '--explain' not in first[2], first


def test_default_mcp_openpi_consistency_is_not_pipeline_approval(tmp_path, openpi_source):
    root = openpi_fixtures.libero_like(tmp_path / 'libero')
    response, = asyncio.run(mcp_session(tmp_path, tmp_path.parent / 'outside-read-root', [
        ('preflight_check', {'source': str(root), 'openpi_config': 'pi0_libero',
                             'openpi_source_root': str(openpi_source)}),
    ]))
    text = json.dumps(response).lower()
    part = response['report']['components']['action_conventions']
    assert part['consumer'] == 'pass'
    assert 'delta-transform choice' in part['consumer_scope'].lower()
    assert response['report']['components']['openpi_context']['status'] == 'error'
    # A consumer PASS here concerns the documented DeltaActions convention only.
    # Repacking keys and native loader-version compatibility have not been run.
    assert 'pipeline compatible' not in text and 'pipeline is compatible' not in text
    assert 'repack' in text
    assert 'version' in text
    assert any(term in text for term in ('unassessed', 'not assessed', 'not checked', 'not verified', 'not established'))


def test_explicit_dataset_question_does_not_print_a_python_dictionary(tmp_path, capsys):
    root = salvage_fixtures.write(tmp_path / 'dataset',
        [salvage_fixtures.rows_for([(0, 5), (1, 6), (2, 7)])],
        [salvage_fixtures.meta_for([(0, 5), (1, 6)])], 2, 11)
    _, out, err = invoke(capsys, 'check', root, '--question', 'dataset')
    assert not err
    lines = plain(out)
    assert "{'observed':" not in lines[1]


def test_required_deployment_units_reject_missing_robot_context(tmp_path, capsys):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    code, out, err = invoke(capsys, 'check', root, '--require', 'deployment-units', '--fail-on', 'error')
    assert code == 2
    assert 'robot' in (out + err).lower()
    assert 'PASS' not in out


def test_required_deployment_units_never_approve_misspelled_robot_type(tmp_path, capsys):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    code, out, err = invoke(capsys, 'check', root, '--robot-command',
        units_fixtures.command(robot='so101_folower'), '--require', 'deployment-units', '--fail-on', 'error')
    assert code in (2, 3)
    assert 'PASS' not in out and 'Traceback' not in out + err


def test_required_deployment_units_refuse_misspelled_units_flag(tmp_path, capsys):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    code, out, err = invoke(capsys, 'check', root, '--robot-command',
        units_fixtures.command('--robot.use_degree=false'), '--require', 'deployment-units', '--fail-on', 'error')
    assert code == 2
    assert 'use_degree' in out + err or 'unknown' in (out + err).lower()
    assert 'PASS' not in out


def test_required_deployment_units_refuse_non_dataset_directory(tmp_path, capsys):
    root = tmp_path / 'empty'
    root.mkdir()
    code, out, err = invoke(capsys, 'check', root, '--robot-command', units_fixtures.command(),
        '--require', 'deployment-units', '--fail-on', 'error')
    assert code == 2
    assert 'Traceback' not in out + err


def test_required_correct_units_pass_despite_unrelated_missing_evidence(tmp_path, capsys):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    command = units_fixtures.command('--robot.use_degrees=false')
    code, out, err = invoke(capsys, 'check', root, '--robot-command', command, '--lerobot-version', '0.6.2',
        '--require', 'deployment-units', '--fail-on', 'error', '--json')
    assert code == 0 and not err
    report = component(json.loads(out), 'units_manifest')
    assert report['status'] == 'pass'
    assert not (root / 'meta/stats.json').exists()  # stats are outside this required question


def test_required_unknown_is_not_filtered_to_success_by_fail_on_error(tmp_path, capsys):
    state = units_fixtures.arm(400, 'm100', seed=1, cal=units_fixtures.COINCIDENT)
    action = units_fixtures.arm(400, 'm100', seed=2, cal=units_fixtures.COINCIDENT)
    root = units_fixtures.write(tmp_path, {'observation.state': state, 'action': action})
    expected = units_manifest(root, robot_command=units_fixtures.command())
    assert expected['status'] == 'unknown'
    code, out, err = invoke(capsys, 'check', root, '--robot-command', units_fixtures.command(),
        '--require', 'deployment-units', '--fail-on', 'error', '--json')
    assert not err and code == 3
    assert component(json.loads(out), 'units_manifest')['status'] == 'unknown'


def test_default_mcp_deployment_chooser_leads_to_available_preflight(tmp_path):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    response, = asyncio.run(mcp_session(root, tmp_path / 'outside-read-root', [
        ('choose_workflow', {'goal': 'deploy', 'framework': 'lerobot', 'dataset': str(root),
                             'robot_command': units_fixtures.command()}),
    ]))
    text = json.dumps(response)
    assert response['report']['assessment_status'] == 'not_run'
    assert 'preflight_check' in text
    assert 'RuntimeRecorder' not in text
