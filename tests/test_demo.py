import json

import pytest

from policy_doctor.cli import human, main
from policy_doctor.demo import create_controller_demo
from policy_doctor.trajectory import compare_joint_trajectories


def test_packaged_demo_cli_has_checkable_incident_and_controls(tmp_path, capsys):
    destination = tmp_path / 'new incident'
    assert main(['demo', str(destination), '--json']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == 'pass'  # Successful creation, not a clean candidate.
    assert summary['candidate_position_at_750ms'] == [1.03125]
    assert summary['candidate_bounds_status'] == 'error'
    assert summary['corrected_status'] == summary['valid_reordering'] == 'pass'
    assert summary['missing_initial_state'] == summary['execution_status'] == summary['physical_cause'] == 'unknown'
    report = json.loads((destination / 'comparison_report.json').read_text())
    assert report['same_waypoint_positions_and_times'] and not report['sampled_references_equal']
    assert report['first_position_difference']['reference_values'] == [0.25]
    assert report['first_position_difference']['candidate_values'] == [0.71875]
    assert all((destination / f'{name}_report.md').is_file() for name in summary['report_paths'])
    assert '1.03125' in (destination / 'README.md').read_text()
    assert 'Physical cause: **unknown**' in (destination / 'README.md').read_text()


@pytest.mark.parametrize('kind', ['empty_directory', 'populated_directory', 'file', 'symlink', 'dangling_symlink'])
def test_demo_cli_refuses_existing_paths_without_changes(tmp_path, capsys, kind):
    destination = tmp_path / 'existing'
    target = tmp_path / 'untouched'
    target.mkdir()
    if kind.endswith('directory'):
        destination.mkdir()
        if kind == 'populated_directory':
            (destination / 'reference.json').write_text('keep me')
    elif kind == 'file':
        destination.write_text('keep me')
    elif kind == 'symlink':
        destination.symlink_to(target, target_is_directory=True)
    else:
        target.rmdir()
        destination.symlink_to(target, target_is_directory=True)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    assert main(['demo', str(destination)]) == 2
    assert json.loads(capsys.readouterr().err)['error']['code'] == 'invalid_input'
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')) == before
    if kind == 'populated_directory':
        assert (destination / 'reference.json').read_text() == 'keep me'
    if kind == 'file':
        assert destination.read_text() == 'keep me'


def test_demo_markdown_escapes_destination_and_exposes_controls(tmp_path, capsys):
    assert main(['demo', str(tmp_path / '<incident|name>'), '--markdown']) == 0
    output = capsys.readouterr().out
    assert '1.03125' in output and 'Missing initial state | unknown' in output
    assert '<incident|name>' not in output and '&lt;incident\\|name&gt;' in output
    assert 'intentionally violates supplied fixture bounds' in output


def test_demo_default_output_explains_creation_and_next_evidence(tmp_path, capsys):
    assert main(['demo', str(tmp_path / 'incident')]) == 0
    output = capsys.readouterr().out
    assert 'Demo created:' in output and 'candidate: 1.03125' in output
    assert 'missing initial state: unknown' in output and '/README.md' in output


def test_default_comparison_exposes_first_discrepancy_concisely(tmp_path, capsys):
    root = tmp_path / 'incident'
    create_controller_demo(root)
    assert main(['compare-trajectories', str(root / 'reference.json'), str(root / 'candidate.json'),
                 '--times', '0', '.25', '.5', '.75', '1']) == 1
    output = capsys.readouterr().out
    assert len(output) < 2500 and 'Policy Doctor: different' in output
    assert 'Same waypoint positions and times: True' in output
    assert 'First positions difference at 250000000 ns:' in output
    assert '"shoulder"[0]: reference 0.25; candidate 0.71875' in output
    assert 'Execution status: unknown' in output and '--json' in output and 'SHA-256' in output


@pytest.mark.parametrize('field', ['accelerations', 'effort'])
def test_default_comparison_retains_nonposition_differences(tmp_path, capsys, field):
    message = {'joint_names': ['joint'], 'points': [
        {'positions': [0.], 'velocities': [0.], 'accelerations': [0.], 'effort': [0.],
         'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1.], 'velocities': [0.], 'accelerations': [0.], 'effort': [0.],
         'time_from_start': {'sec': 1, 'nanosec': 0}}]}
    reference, candidate = tmp_path / 'a.json', tmp_path / 'b.json'
    reference.write_text(json.dumps(message))
    message['points'][0][field] = [2.]
    candidate.write_text(json.dumps(message))
    assert main(['compare-trajectories', str(reference), str(candidate), '--times', '0']) == 1
    output = capsys.readouterr().out
    assert f'First {field} difference at 0 ns:' in output
    assert '"joint"[0]: reference 0.0; candidate 2.0' in output
    assert len(output) < 2500


def test_default_comparison_keeps_incomplete_sampling_visible(tmp_path, capsys):
    root = tmp_path / 'incident'
    create_controller_demo(root)
    assert main(['compare-trajectories', str(root / 'missing_initial_state.json'),
                 str(root / 'missing_initial_state.json'), '--times', '0', '1.5']) == 3
    output = capsys.readouterr().out
    assert 'Comparison complete: False; sampled reference equality: None' in output
    assert 'reference sampling/bounds: unknown'.lower() in output.lower()
    assert 'equal sampled references are not established' in output


def test_concise_comparison_preserves_equal_but_invalid_bounds(tmp_path):
    root = tmp_path / 'incident'
    create_controller_demo(root)
    result = compare_joint_trajectories(str(root / 'candidate.json'), str(root / 'candidate.json'), [.75],
                                       position_limits={'shoulder': [0., 1.]})
    output = human(result)
    assert output.startswith('Policy Doctor: error\n')
    assert 'sampled reference equality: True' in output
    assert 'Reference sampling/bounds: error' in output and 'Candidate sampling/bounds: error' in output
    assert 'declared bounds: supplied' in output


def test_concise_comparison_shows_changed_joint_beyond_preview(tmp_path, capsys):
    message = {'joint_names': [f'j{i}' for i in range(9)], 'points': [
        {'positions': [0.] * 9, 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1.] * 9, 'time_from_start': {'sec': 1, 'nanosec': 0}}]}
    reference, candidate = tmp_path / 'a.json', tmp_path / 'b.json'
    reference.write_text(json.dumps(message))
    message['points'][0]['positions'][8] = .5
    candidate.write_text(json.dumps(message))
    assert main(['compare-trajectories', str(reference), str(candidate), '--times', '0']) == 1
    output = capsys.readouterr().out
    assert '"j8"[8]: reference 0.0; candidate 0.5' in output
