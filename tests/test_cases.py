import asyncio
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import InputError
from policy_doctor.cases import export_trajectory_case, reproduce_trajectory_case
from policy_doctor.cli import human
from policy_doctor.demo import create_controller_demo


@pytest.fixture
def native_case(tmp_path):
    root = tmp_path / 'native'
    create_controller_demo(str(root))
    destination = tmp_path / 'case'
    export_trajectory_case(str(root / 'reference.json'), str(root / 'candidate.json'), str(destination), [.25, .75],
                           control=str(root / 'corrected.json'), position_limits={'shoulder': [0, 1]})
    return root, destination


def recommit(root, mutate):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))
    (root / 'READY.json').write_text(json.dumps({'schema_version': 1,
        'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))


def test_case_survives_move_and_missing_originals_with_literal_native_bytes(native_case, tmp_path):
    source, case = native_case
    assert (case / 'reference.json').read_bytes() == (source / 'reference.json').read_bytes()
    assert (case / 'candidate.json').read_bytes() == (source / 'candidate.json').read_bytes()
    expected = (case / 'expected.json').read_text()
    assert str(tmp_path) not in expected
    shutil.rmtree(source)
    moved = tmp_path / 'a colleague' / 'case'
    moved.parent.mkdir()
    case.rename(moved)
    before = {p.name: p.read_bytes() for p in moved.iterdir()}
    report = reproduce_trajectory_case(str(moved))
    assert report['status'] == 'pass' and report['reproduction_matches']
    # A bounded candidate error is reproduced; it is never relabeled a repair.
    assert report['checks']['candidate']['status'] == 'error'
    assert report['checks']['candidate']['sampling']['candidate']['sampled_limit_violations'][0]['value'] == 1.03125
    # Hermite at t=.25: h01*1 + h10*4 = .15625 + .140625*4.
    assert report['checks']['candidate']['first_position_difference']['candidate_values'] == [0.71875]
    assert report['checks']['control']['status'] == 'pass'
    assert report['checks']['candidate']['execution_status'] == 'unknown'
    assert before == {p.name: p.read_bytes() for p in moved.iterdir()}


@pytest.mark.parametrize('name', ['reference.json', 'candidate.json', 'expected.json', 'control.json', 'manifest.json', 'READY.json'])
def test_case_rejects_modified_or_incomplete_input(native_case, name):
    _, case = native_case
    (case / name).unlink()
    with pytest.raises(InputError):
        reproduce_trajectory_case(str(case))


@pytest.mark.parametrize('name', ['reference.json', 'candidate.json', 'expected.json', 'control.json'])
def test_case_rejects_changed_bytes(native_case, name):
    _, case = native_case
    with (case / name).open('ab') as f:
        f.write(b' ')
    with pytest.raises(InputError, match='hash mismatch'):
        reproduce_trajectory_case(str(case))


def test_case_detects_wrong_recorded_answer_even_when_all_hashes_are_refreshed(native_case):
    _, case = native_case
    path = case / 'expected.json'
    expected = json.loads(path.read_text())
    expected['candidate']['status'] = 'pass'
    path.write_text(json.dumps(expected))
    recommit(case, lambda m: m['files'].update({'expected.json': hashlib.sha256(path.read_bytes()).hexdigest()}))
    report = reproduce_trajectory_case(str(case))
    assert report['status'] == 'error' and not report['reproduction_matches']


@pytest.mark.parametrize('alter', ['engine_version', 'numpy_version', 'module_sha256'])
def test_verifier_drift_never_silently_passes(native_case, alter):
    _, case = native_case
    recommit(case, lambda m: m['verifier'].update({alter: 'changed'}))
    report = reproduce_trajectory_case(str(case))
    assert report['reproduction_matches'] and report['status'] == 'unknown'
    assert not report['verifier_matches']


@pytest.mark.parametrize('mutate', [
    lambda m: m.update({'command': 'touch should-never-exist'}),
    lambda m: m.update({'kind': 'run_shell'}),
    lambda m: m['parameters'].update({'shell': 'touch should-never-exist'}),
    lambda m: m['files'].update({'../outside.json': '0' * 64}),
    lambda m: m.update({'schema_version': True}),
    lambda m: m.update({'schema_version': 2}),
    lambda m: m.update({'has_control': 'yes'}),
])
def test_only_fixed_declarative_operations_are_accepted(native_case, mutate):
    _, case = native_case
    recommit(case, mutate)
    with pytest.raises(InputError):
        reproduce_trajectory_case(str(case))
    assert not (case / 'should-never-exist').exists()


@pytest.mark.parametrize('name', ['reference.json', 'expected.json', 'manifest.json', 'READY.json'])
def test_reject_external_symlinks_before_reading(native_case, tmp_path, name):
    _, case = native_case
    path = case / name
    outside = tmp_path / ('outside-' + name)
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(InputError, match='symlinks'):
        reproduce_trajectory_case(str(case))


def test_unknown_control_is_retained_and_not_converted_to_success(native_case, tmp_path):
    source, _ = native_case
    destination = tmp_path / 'unknown'
    export_trajectory_case(str(source / 'missing_initial_state.json'), str(source / 'missing_initial_state.json'),
                           str(destination), [0, 1.5])
    report = reproduce_trajectory_case(str(destination))
    assert report['status'] == 'pass'
    assert report['checks']['candidate']['status'] == 'unknown'
    assert report['checks']['candidate']['sampled_references_equal'] is None


def test_existing_dangling_destination_is_preserved(native_case, tmp_path):
    source, _ = native_case
    target = tmp_path / 'reserved'
    target.symlink_to(tmp_path / 'absent')
    with pytest.raises(InputError, match='already exists'):
        export_trajectory_case(str(source / 'reference.json'), str(source / 'candidate.json'), str(target), [.75])
    assert target.is_symlink() and not (tmp_path / 'absent').exists()


def test_native_names_that_resemble_evidence_keys_remain_valid(tmp_path):
    p = tmp_path / 'native.json'
    p.write_text(json.dumps({'joint_names': ['path', 'sha256'], 'points': [
        {'positions': [0, 0], 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1, 1], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}))
    destination = tmp_path / 'case'
    export_trajectory_case(str(p), str(p), str(destination), [.5], position_limits={'path': [0, 1], 'sha256': [0, 1]})
    assert reproduce_trajectory_case(str(destination))['status'] == 'pass'


def test_failed_export_cleans_staging_and_does_not_publish_partial_case(native_case, tmp_path):
    source, _ = native_case
    destination = tmp_path / 'invalid-export'
    with pytest.raises(InputError, match='increase strictly'):
        export_trajectory_case(str(source / 'reference.json'), str(source / 'candidate.json'), str(destination), [.75, .25])
    assert not destination.exists()
    assert not list(tmp_path.glob('.policy-doctor-case-*'))


def test_concurrent_existing_destination_is_never_replaced(native_case, tmp_path, monkeypatch):
    import policy_doctor.cases as cases
    source, _ = native_case
    destination = tmp_path / 'raced'
    publish = cases.publish_capture

    def race(staging, target):
        target.mkdir()
        (target / 'colleague.txt').write_text('retain')
        publish(staging, target)

    monkeypatch.setattr(cases, 'publish_capture', race)
    with pytest.raises(InputError, match='already exists'):
        export_trajectory_case(str(source / 'reference.json'), str(source / 'candidate.json'), str(destination), [.75])
    assert (destination / 'colleague.txt').read_text() == 'retain'
    assert not (destination / 'READY.json').exists()


def test_change_during_evaluation_is_refused(native_case, monkeypatch):
    import policy_doctor.cases as cases
    _, case = native_case
    evaluate = cases._evaluate

    def change(root, parameters, has_control):
        report = evaluate(root, parameters, has_control)
        with (root / 'candidate.json').open('ab') as stream:
            stream.write(b' ')
        return report

    monkeypatch.setattr(cases, '_evaluate', change)
    with pytest.raises(InputError, match='changed while'):
        reproduce_trajectory_case(str(case))


@pytest.mark.parametrize('mutate', ['times', 'limits'])
def test_export_snapshots_mutable_caller_parameters(native_case, tmp_path, monkeypatch, mutate):
    import policy_doctor.cases as cases
    source, _ = native_case
    times, limits = [.25, .75], {'shoulder': [0, 1]}
    evaluate = cases._evaluate

    def interleave(root, parameters, has_control):
        result = evaluate(root, parameters, has_control)
        if mutate == 'times':
            times[:] = [1]
        else:
            limits['shoulder'][1] = 100
        return result

    monkeypatch.setattr(cases, '_evaluate', interleave)
    destination = tmp_path / 'snapshot'
    cases.export_trajectory_case(str(source / 'reference.json'), str(source / 'candidate.json'), str(destination), times,
                                 position_limits=limits)
    monkeypatch.setattr(cases, '_evaluate', evaluate)
    result = cases.reproduce_trajectory_case(str(destination))
    assert result['status'] == 'pass' and result['reproduction_matches']
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['parameters']['sample_times'] == [.25, .75]
    assert manifest['parameters']['position_limits'] == {'shoulder': [0, 1]}


@pytest.mark.parametrize('name', ["case$(printf DIFFERENT)", "case`printf DIFFERENT`", "case with 'quote' and spaces"])
def test_displayed_rerun_command_preserves_literal_shell_paths(native_case, tmp_path, name):
    source, _ = native_case
    exported = export_trajectory_case(str(source / 'reference.json'), str(source / 'candidate.json'), str(tmp_path / name), [.75])
    guidance = next(line for line in human(exported).splitlines() if line.startswith('Rerun: '))
    command = shlex.join([sys.executable, '-m', 'policy_doctor']) + guidance.removeprefix('Rerun: policy-doctor')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))
    result = subprocess.run(['/bin/sh', '-c', command], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'Recorded findings reproduced: True' in result.stdout


def test_large_case_has_a_bounded_first_report_and_full_recorded_evidence(native_case, tmp_path):
    source, _ = native_case
    destination = tmp_path / 'many-violations'
    export_trajectory_case(str(source / 'reference.json'), str(source / 'reference.json'), str(destination),
                           [i / 1000 for i in range(1, 1000)], position_limits={'shoulder': [0, 0]})
    report = reproduce_trajectory_case(str(destination))
    assert len(json.dumps(report).encode()) < 192 * 1024
    sampling = report['checks']['candidate']['sampling']['candidate']
    assert sampling['sampled_limit_violation_count'] == 999
    assert len(sampling['sampled_limit_violations']) == 8 and sampling['sampled_limit_violations_truncated']
    expected = json.loads((destination / 'expected.json').read_text())
    assert len(expected['candidate']['candidate']['sampled_limit_violations']) == 999


def test_real_cli_export_and_stdio_mcp_reproduction(native_case, tmp_path):
    source, case = native_case
    project = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(project / 'src'))
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'export-trajectory-case',
        str(source / 'reference.json'), str(source / 'candidate.json'), str(tmp_path / 'cli-case'),
        '--control', str(source / 'corrected.json'), '--times', '.75', '--json'], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'reproduce-case', str(tmp_path / 'cli-case')],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0 and 'Candidate comparison: different' in result.stdout
    assert 'Recorded findings reproduced: True' in result.stdout

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(case)], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('reproduce_trajectory_case', {'directory': str(case)})
                assert not result.isError and result.structuredContent['reproduction_matches']
                assert result.structuredContent['checks']['candidate']['status'] == 'error'
                denied = await client.call_tool('reproduce_trajectory_case', {'directory': str(tmp_path)})
                assert denied.isError
    asyncio.run(asyncio.wait_for(exercise(), 30))
