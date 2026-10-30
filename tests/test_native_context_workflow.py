"""Recorded comparison selection, portable excerpt, and retained future check."""

import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.incidents import (
    check_incident_candidate,
    export_incident_case,
    reproduce_incident_case,
)
from policy_doctor.investigation import investigate_run

PARAMETERS = {'equal_fields': ['/terminations/success', '/actions', '/sim/dt']}
REQUIREMENT = {'kind': 'native-run-context', 'parameters': PARAMETERS}


def native(root, *, tolerance=0.0, offset=0, missing=False):
    root.mkdir()
    config = {'actions': {'body': {'scale': 1.0}}, 'sim': {'dt': 1 / 120},
              'terminations': {'success': {'func': 'example:success', 'tolerance': tolerance}},
              'scene': {'asset': f'/different/checkouts/{offset}/robot.usd'}}
    if missing:
        del config['terminations']
    (root / 'env_cfg.json').write_text(json.dumps(config))
    (root / 'log_0.json').write_text('[{}, {}]')
    with h5py.File(root / 'data.hdf5', 'w') as file:
        group = file.create_group('data/demo_0')
        group.attrs['success'] = True
        group.attrs['num_samples'] = 4
        group.create_dataset('actions', data=np.arange(32, dtype=np.float32).reshape(4, 8) + offset)
        group.create_dataset('states/robot/joint_pos', data=np.ones((4, 7), dtype=np.float32))
        group.create_dataset('obs/camera_rgb', data=np.zeros((4, 8, 8, 3), dtype=np.uint8), compression='gzip')
    return root


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def test_whole_native_context_lifecycle_survives_original_removal_and_move(tmp_path):
    originals = tmp_path / 'originals'
    originals.mkdir()
    before = native(originals / 'before', tolerance=.05)
    after = native(originals / 'after')
    unknown = native(originals / 'unknown', missing=True)
    later = native(tmp_path / 'later', offset=100)
    snapshot = hashes(originals)
    investigation = investigate_run(str(before), reference=str(after), requirements=REQUIREMENT)
    assert investigation['status'] == 'error'
    recipe = investigation['case_recipe']['arguments']
    assert recipe['parameters'] == PARAMETERS
    case = tmp_path / 'case'
    result = export_incident_case(**recipe, destination=case, controls=[
        {'name': 'missing criterion', 'source': str(unknown), 'parameters': PARAMETERS, 'expected_status': 'unknown'}])
    assert result['status'] == 'pass'
    assert snapshot == hashes(originals)
    for role in ('before', 'after', 'controls/0'):
        with h5py.File(case / role / 'data.hdf5') as selected:
            assert 'data/demo_0/actions' in selected
            assert 'data/demo_0/obs/camera_rgb' not in selected
            assert selected['data/demo_0/actions'].shape == (4, 8)
        receipt = json.loads((case / role / 'evidence_selection.json').read_text())
        assert 'derived' in json.dumps(receipt).lower()
        assert 'camera_rgb' in json.dumps(receipt)
    shutil.rmtree(originals)
    moved = tmp_path / 'colleague packet'
    case.rename(moved)
    replay = reproduce_incident_case(moved)
    assert replay['status'] == 'pass'
    assert replay['checks']['before']['status'] == 'error'
    assert replay['checks']['after']['status'] == 'pass'
    assert replay['checks']['controls'][0]['check']['status'] == 'unknown'
    reuse = check_incident_candidate(moved, later)
    assert reuse['case_verified'] and reuse['candidate_evaluated']
    assert reuse['status'] == 'pass' and reuse['parameters'] == PARAMETERS
    assert reuse['candidate']['unverified']
    with h5py.File(moved / 'after/data.hdf5', 'a') as file:
        file['data/demo_0/actions'][0, 0] = 42
    with pytest.raises(InputError, match='hash'):
        reproduce_incident_case(moved)


def test_cli_human_markdown_missing_and_case_use_same_requirement(tmp_path, capsys):
    before, after = native(tmp_path / 'before', tolerance=.05), native(tmp_path / 'after')
    requirement = tmp_path / 'requirement.json'
    requirement.write_text(json.dumps(REQUIREMENT))
    argv = ['investigate', str(before), '--reference', str(after), '--requirements', str(requirement)]
    for extra in ([], ['--markdown'], ['--json']):
        assert main([*argv, *extra]) == 1
        output = capsys.readouterr()
        assert output.out and not output.err
        assert '/terminations/success' in output.out
    assert main(['investigate', str(before), '--requirements', str(requirement), '--markdown']) == 3
    output = capsys.readouterr()
    assert output.out and not output.err
    case = tmp_path / 'case'
    assert main(['create-case', 'native-run-context', str(before), str(case), '--corrected', str(after),
                 '--requirements', str(requirement), '--json']) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)['applied_requirements'] == REQUIREMENT and not output.err
    assert main(['reproduce-case', str(case), '--markdown']) == 0
    output = capsys.readouterr()
    assert 'behavioral' in output.out and not output.err


def test_real_mcp_sdk_native_case_and_read_root_guards(tmp_path):
    before, after = native(tmp_path / 'before', tolerance=.05), native(tmp_path / 'after')
    outputs = tmp_path / 'outputs'
    outputs.mkdir()
    case = outputs / 'sdk-case'

    async def run():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path), '--case-root', str(outputs)], env=env)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('investigate_run_directory',
                {'directory': str(before), 'reference': str(after), 'requirements': REQUIREMENT})
            assert not result.isError
            payload = json.loads(result.content[0].text)
            assert payload['status'] == 'error' and payload['case_recipe']
            result = await session.call_tool('create_incident_case',
                dict(payload['case_recipe']['arguments'], destination=str(case)))
            assert not result.isError and json.loads(result.content[0].text)['status'] == 'pass'
            for tool, arguments in [('reproduce_incident_case', {'directory': str(case)}),
                                    ('check_incident_candidate', {'directory': str(case), 'candidate': str(after)})]:
                result = await session.call_tool(tool, arguments)
                assert not result.isError and json.loads(result.content[0].text)['status'] == 'pass'
            result = await session.call_tool('investigate_run_directory',
                {'directory': str(before), 'reference': '/etc', 'requirements': REQUIREMENT})
            assert result.isError

    asyncio.run(run())


def test_parent_run_recipe_resolves_unique_native_task_and_executes(tmp_path):
    left, right = tmp_path / 'run-before', tmp_path / 'run-after'
    left.mkdir()
    right.mkdir()
    before, after = native(left / 'Task', tolerance=.05), native(right / 'Task')
    report = investigate_run(str(left), reference=str(right), requirements=REQUIREMENT)
    recipe = report['case_recipe']['arguments']
    assert recipe['source'] == str(before) and recipe['corrected'] == str(after)
    assert export_incident_case(**recipe, destination=tmp_path / 'case')['status'] == 'pass'


def test_multi_task_or_config_only_comparison_does_not_advertise_unusable_recipe(tmp_path):
    left, right = tmp_path / 'run-before', tmp_path / 'run-after'
    left.mkdir()
    right.mkdir()
    for task in ('TaskA', 'TaskB'):
        native(left / task, tolerance=.05)
        native(right / task)
    report = investigate_run(str(left), reference=str(right), requirements=REQUIREMENT)
    assert report['status'] == 'error' and report['case_recipe'] is None and report['next_actions']
    before, after = native(tmp_path / 'before', tolerance=.05), native(tmp_path / 'after')
    (before / 'data.hdf5').unlink()
    (after / 'data.hdf5').unlink()
    report = investigate_run(str(before), reference=str(after), requirements=REQUIREMENT)
    assert report['status'] == 'error' and report['case_recipe'] is None and report['next_actions']
