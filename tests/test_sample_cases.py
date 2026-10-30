"""One native message is sufficient to retain and reproduce a sample finding."""
import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import cases
from policy_doctor.artifacts import InputError


def native(tmp_path, suffix='.json', first_second=0):
    data = {'joint_trajectory': {'joint_names': ['shoulder'], 'points': [
        {'positions': [0.0], 'velocities': [4.0], 'time_from_start': {'sec': first_second, 'nanosec': 0}},
        {'positions': [1.0], 'velocities': [0.0], 'time_from_start': {'sec': first_second + 1, 'nanosec': 0}}]},
        'multi_dof_joint_trajectory': {'joint_names': [], 'points': []}}
    path = tmp_path / f'original{suffix}'
    if suffix == '.json':
        path.write_text(json.dumps(data))
    else:
        yaml = pytest.importorskip('yaml')
        path.write_text('# Keep original comments and wrapper.\n' + yaml.safe_dump(data))
    return path


def export(source, destination, times=None, **kwargs):
    return cases.export_sample_case(str(source), str(destination), [.75] if times is None else times, **kwargs)


def reseal(root, mutate):
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    raw = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    manifest_path.write_bytes(raw)
    (root / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


@pytest.mark.parametrize('suffix', ['.json', '.yaml', '.yml'])
def test_one_native_input_moves_and_reproduces_without_original(tmp_path, suffix):
    source = native(tmp_path, suffix)
    raw = source.read_bytes()
    root = tmp_path / 'case'
    result = export(source, root)
    assert result['kind'] == 'exported_trajectory_sample_case'
    assert result['status'] == result['sample_status'] == 'pass'
    assert result['sampling']['samples'][0]['positions'] == [1.03125]
    assert (root / f'trajectory{suffix}').read_bytes() == raw
    assert not any(p.name.startswith(('reference', 'candidate')) for p in root.iterdir())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['kind'] == 'native_trajectory_sample_case'
    assert set(manifest['files']) == {f'trajectory{suffix}', 'expected.json'}
    assert ('pyyaml_version' in manifest['verifier']) == (suffix != '.json')
    moved = tmp_path / 'colleague'
    shutil.move(root, moved)
    source.unlink()
    replay = cases.reproduce_trajectory_case(str(moved))
    assert replay['status'] == 'pass' and replay['reproduction_matches'] and replay['verifier_matches']
    summary = replay['checks']['sample']
    assert summary['samples'][0]['positions'] == [1.03125]
    assert summary['artifacts'][0]['message_pointer'] == '/joint_trajectory'
    assert summary['artifacts'][0]['path'] == f'trajectory{suffix}'
    assert summary['execution_status'] == summary['physical_cause'] == 'unknown'
    assert 'comparison_complete' not in summary


@pytest.mark.parametrize('scenario', ['unknown', 'bounds'])
def test_reproduced_sample_does_not_relabel_diagnostic_as_healthy(tmp_path, scenario):
    source = native(tmp_path, first_second=1 if scenario == 'unknown' else 0)
    options = {} if scenario == 'unknown' else {'position_limits': {'shoulder': [0., 1.]}}
    result = export(source, tmp_path / 'case', [.5, .75], **options)
    expected = 'unknown' if scenario == 'unknown' else 'error'
    assert result['status'] == 'pass' and result['sample_status'] == expected
    replay = cases.reproduce_trajectory_case(str(tmp_path / 'case'))
    assert replay['status'] == 'pass'
    assert replay['checks']['sample']['status'] == expected
    if scenario == 'bounds':
        assert replay['checks']['sample']['sampled_limit_violation_count'] == 1
    else:
        assert all(s['status'] == 'unknown' for s in replay['checks']['sample']['samples'])


def test_changed_native_bytes_refuse_before_reproduction(tmp_path):
    source = native(tmp_path)
    root = tmp_path / 'case'
    export(source, root)
    with (root / 'trajectory.json').open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(InputError, match='hash mismatch'):
        cases.reproduce_trajectory_case(str(root))


def test_tampered_findings_do_not_count_as_reproduced(tmp_path):
    source = native(tmp_path)
    root = tmp_path / 'case'
    export(source, root)
    expected_path = root / 'expected.json'
    expected = json.loads(expected_path.read_text())
    expected['sample']['samples'][0]['positions'] = [999.]
    raw = json.dumps(expected).encode()
    expected_path.write_bytes(raw)
    reseal(root, lambda m: m['files'].update({'expected.json': hashlib.sha256(raw).hexdigest()}))
    result = cases.reproduce_trajectory_case(str(root))
    assert result['status'] == 'error' and not result['reproduction_matches']


@pytest.mark.parametrize('field', ['engine_version', 'module_sha256', 'pyyaml_version'])
def test_stale_verifier_remains_unknown_even_when_findings_match(tmp_path, field):
    root = tmp_path / 'case'
    export(native(tmp_path, '.yaml' if field == 'pyyaml_version' else '.json'), root)
    value = {'trajectory.py': '0' * 64} if field == 'module_sha256' else 'old'
    reseal(root, lambda m: m['verifier'].update({field: value}))
    result = cases.reproduce_trajectory_case(str(root))
    assert result['status'] == 'unknown' and result['reproduction_matches'] and not result['verifier_matches']


@pytest.mark.parametrize('mutation', ['command', 'extra_parameter', 'traversal', 'wrong_kind', 'duplicate_role', 'has_control'])
def test_only_explicit_declarative_sample_manifest_is_allowed(tmp_path, mutation):
    root = tmp_path / 'case'
    export(native(tmp_path), root)

    def mutate(manifest):
        if mutation == 'command':
            manifest['command'] = 'do not execute this'
        elif mutation == 'extra_parameter':
            manifest['parameters']['python'] = 'do not execute this'
        elif mutation == 'traversal':
            manifest['files']['../trajectory.json'] = manifest['files'].pop('trajectory.json')
        elif mutation == 'wrong_kind':
            manifest['kind'] = 'arbitrary_runner'
        elif mutation == 'duplicate_role':
            manifest['files']['trajectory.yaml'] = manifest['files']['trajectory.json']
        else:
            manifest['has_control'] = False

    reseal(root, mutate)
    with pytest.raises(InputError):
        cases.reproduce_trajectory_case(str(root))


def test_symlinked_native_case_input_refuses(tmp_path):
    source = native(tmp_path)
    root = tmp_path / 'case'
    export(source, root)
    (root / 'trajectory.json').unlink()
    (root / 'trajectory.json').symlink_to(source)
    with pytest.raises(InputError, match='symlink'):
        cases.reproduce_trajectory_case(str(root))


def test_extra_native_role_outside_manifest_refuses(tmp_path):
    root = tmp_path / 'case'
    export(native(tmp_path), root)
    (root / 'trajectory.yaml').write_bytes((root / 'trajectory.json').read_bytes())
    with pytest.raises(InputError):
        cases.reproduce_trajectory_case(str(root))


def test_export_does_not_overwrite_existing_destination(tmp_path):
    root = tmp_path / 'case'
    root.mkdir()
    sentinel = root / 'sentinel'
    sentinel.write_text('keep')
    with pytest.raises(InputError, match='already exists'):
        export(native(tmp_path), root)
    assert sentinel.read_text() == 'keep' and len(list(root.iterdir())) == 1


def test_export_snapshots_mutable_parameters_before_evaluation(tmp_path, monkeypatch):
    source = native(tmp_path)
    times, limits = [.75], {'shoulder': [0., 1.]}
    evaluate = cases._evaluate_sample

    def interleave(root, parameters):
        report = evaluate(root, parameters)
        times[:] = [0.]
        limits['shoulder'][1] = 100.
        return report

    monkeypatch.setattr(cases, '_evaluate_sample', interleave)
    export(source, tmp_path / 'case', times, position_limits=limits)
    result = cases.reproduce_trajectory_case(str(tmp_path / 'case'))
    assert result['status'] == 'pass' and result['checks']['sample']['status'] == 'error'


def test_file_changes_during_sample_evaluation_refuse(tmp_path, monkeypatch):
    root = tmp_path / 'case'
    export(native(tmp_path), root)
    evaluate = cases._evaluate_sample

    def interleave(directory, parameters):
        result = evaluate(directory, parameters)
        with (directory / 'trajectory.json').open('ab') as stream:
            stream.write(b' ')
        return result

    monkeypatch.setattr(cases, '_evaluate_sample', interleave)
    with pytest.raises(InputError, match='changed while'):
        cases.reproduce_trajectory_case(str(root))


def test_large_sample_report_reproduction_has_bounded_preview(tmp_path):
    source = native(tmp_path)
    times = [i / 100 for i in range(101)]
    export(source, tmp_path / 'case', times)
    result = cases.reproduce_trajectory_case(str(tmp_path / 'case'))
    summary = result['checks']['sample']
    assert summary['sample_count'] == 101 and len(summary['samples']) == 8 and summary['samples_truncated']
    assert summary['excursion_count'] > len(summary['waypoint_envelope_excursions'])
    assert len(summary['waypoint_envelope_excursions']) == 8


def test_existing_sdk_runner_reproduces_single_case_readonly_and_guards_root(tmp_path):
    root = tmp_path / 'case'
    export(native(tmp_path), root, position_limits={'shoulder': [0., 1.]})
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))

    async def exercise():
        parameters = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(root)], env=env)
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('reproduce_trajectory_case', {'directory': str(root)})
                assert not result.isError
                assert result.structuredContent['status'] == 'pass'
                assert result.structuredContent['checks']['sample']['status'] == 'error'
                assert result.structuredContent['checks']['sample']['sampled_limit_violation_count'] == 1
                denied = await client.call_tool('reproduce_trajectory_case', {'directory': str(tmp_path)})
                assert denied.isError

    asyncio.run(asyncio.wait_for(exercise(), 30))
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
