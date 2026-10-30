import copy
import json

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.reports import markdown_report
from policy_doctor.trajectory import compare_joint_trajectories, sample_joint_trajectory


def native(tmp_path, *, velocities=None, accelerations=None, name='trajectory.json'):
    points = [{'positions': [p], 'time_from_start': {'sec': t, 'nanosec': 0}}
              for t, p in [(0, 0.), (1, 1.)]]
    for field, values in [('velocities', velocities), ('accelerations', accelerations)]:
        if values is not None:
            for point, value in zip(points, values, strict=True):
                point[field] = [value]
    data = {'header': {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''},
            'joint_names': ['shoulder'], 'points': points}
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path, data


def test_linear_cubic_and_quintic_use_native_derivative_fields(tmp_path):
    path, _ = native(tmp_path)
    report = sample_joint_trajectory(str(path), [0., .25, .5, .75, 1.])
    assert report['status'] == 'pass'
    assert report['samples'][1]['positions'] == [.25]
    assert report['samples'][1]['velocities'] == [1.]
    assert report['samples'][1]['interpolation'] == 'linear'
    path, _ = native(tmp_path, velocities=[4., 0.])
    report = sample_joint_trajectory(str(path), [.75], position_limits={'shoulder': [0., 1.]})
    assert report['status'] == 'error'
    assert report['samples'][0]['positions'] == [1.03125]
    assert report['excursion_count'] == 1
    assert report['waypoint_envelope_excursions'][0]['waypoint_envelope'] == [0., 1.]
    assert report['sampled_limit_violations'][0]['evidence_kind'] == 'derived_sample'
    assert report['execution_status'] == report['physical_cause'] == 'unknown'
    path, _ = native(tmp_path, velocities=[0., 0.], accelerations=[0., 0.])
    report = sample_joint_trajectory(str(path), [.25])
    # Quintic smoothstep: 10*t^3 - 15*t^4 + 6*t^5.
    assert report['samples'][0]['positions'] == [0.103515625]
    assert report['samples'][0]['interpolation'] == 'quintic'


def test_native_none_forwards_next_waypoint_and_terminal_retains_supplied_derivatives(tmp_path):
    path, data = native(tmp_path, velocities=[4., 2.])
    data['points'][1]['effort'] = [3.]
    path.write_text(json.dumps(data))
    report = sample_joint_trajectory(str(path), [0., .75, 1., 2.], interpolation_method='none')
    assert report['samples'][0]['positions'] == [1.]
    assert report['samples'][0]['velocities'] == [2.]
    assert report['samples'][1]['accelerations'] == []
    assert report['samples'][2]['velocities'] == [2.]
    assert report['samples'][2]['accelerations'] == [0.]
    assert report['samples'][3]['effort'] == [3.]


def test_initial_state_is_not_fabricated(tmp_path):
    path, data = native(tmp_path)
    for point in data['points']:
        point['time_from_start']['sec'] += 1
    path.write_text(json.dumps(data))
    result = sample_joint_trajectory(str(path), [0., 1.5])
    assert result['status'] == 'unknown' and not result['sampling_complete']
    assert 'positions' not in result['samples'][0]
    assert result['samples'][1]['positions'] == [.5]


def test_effort_is_interpolated_separately(tmp_path):
    path, data = native(tmp_path)
    data['points'][0]['effort'], data['points'][1]['effort'] = [2.], [6.]
    path.write_text(json.dumps(data))
    assert sample_joint_trajectory(str(path), [.25])['samples'][0]['effort'] == [3.]


def test_same_waypoint_positions_different_interpolated_path_and_correction(tmp_path):
    a, _ = native(tmp_path, name='reference.json')
    b, _ = native(tmp_path, velocities=[4., 0.], name='candidate.json')
    report = compare_joint_trajectories(str(a), str(b), [0., .25, .5, .75, 1.])
    assert report['status'] == 'different' and report['same_waypoint_positions_and_times']
    assert report['first_position_difference']['time_from_start_ns'] == 250000000
    assert report['first_position_difference']['candidate_values'] == [.71875]
    assert set(report['different_sample_fields']) == {'positions', 'velocities', 'accelerations'}
    assert compare_joint_trajectories(str(a), str(a), [0., .25, 1.])['status'] == 'pass'
    assert all(a['sha256'] for a in report['artifacts'])


def test_equal_references_do_not_hide_declared_sample_bound_violations(tmp_path):
    path, _ = native(tmp_path, velocities=[4., 0.])
    report = compare_joint_trajectories(str(path), str(path), [.75], position_limits={'shoulder': [0., 1.]})
    assert report['sampled_references_equal'] is True
    assert report['status'] == report['reference_sampling_status'] == report['candidate_sampling_status'] == 'error'


def test_joint_order_alignment_is_by_native_name(tmp_path):
    a, data = native(tmp_path, name='a.json')
    data['joint_names'] = ['shoulder', 'elbow']
    for point in data['points']:
        point['positions'] += [2 * point['positions'][0]]
    a.write_text(json.dumps(data))
    reordered = copy.deepcopy(data)
    reordered['joint_names'].reverse()
    for point in reordered['points']:
        point['positions'].reverse()
    b = tmp_path / 'b.json'
    b.write_text(json.dumps(reordered))
    result = compare_joint_trajectories(str(a), str(b), [0., .25, 1.])
    assert result['status'] == 'pass' and result['same_waypoint_positions_and_times']
    reordered['joint_names'][0] = 'other'
    b.write_text(json.dumps(reordered))
    assert compare_joint_trajectories(str(a), str(b), [.5])['status'] == 'unknown'


@pytest.mark.parametrize('mutation', ['duplicate_joint', 'duplicate_time', 'bad_width', 'no_positions',
    'boolean_value', 'nanosec_range', 'negative_time', 'large_integer', 'unknown_field', 'no_velocity'])
def test_malformed_or_unsupported_native_trajectories_refuse(tmp_path, mutation):
    path, data = native(tmp_path)
    if mutation == 'duplicate_joint':
        data['joint_names'] *= 2
    elif mutation == 'duplicate_time':
        data['points'][1]['time_from_start']['sec'] = 0
    elif mutation == 'bad_width':
        data['points'][0]['positions'] = [0., 1.]
    elif mutation == 'no_positions':
        data['points'][0]['positions'] = []
    elif mutation == 'boolean_value':
        data['points'][0]['positions'] = [True]
    elif mutation == 'nanosec_range':
        data['points'][0]['time_from_start']['nanosec'] = 1000000000
    elif mutation == 'negative_time':
        data['points'][0]['time_from_start']['sec'] = -1
    elif mutation == 'large_integer':
        data['points'][0]['positions'] = [2**53+1]
    elif mutation == 'unknown_field':
        data['points'][0]['command'] = [0.]
    else:
        for point in data['points']:
            point['accelerations'] = [0.]
    path.write_text(json.dumps(data))
    with pytest.raises(InputError):
        sample_joint_trajectory(str(path), [.5])


@pytest.mark.parametrize('times', [[], [True], [-.1], [0., 0.], [0., 1e-12], [float('inf')]])
def test_invalid_sampling_requests_refuse(tmp_path, times):
    path, _ = native(tmp_path)
    with pytest.raises(InputError):
        sample_joint_trajectory(str(path), times)


def test_overflow_does_not_escape_as_nan_or_raw_exception(tmp_path):
    path, data = native(tmp_path)
    data['points'][0]['positions'] = [-1e308]
    data['points'][1]['positions'] = [1e308]
    path.write_text(json.dumps(data))
    with pytest.raises(InputError, match='overflow'):
        sample_joint_trajectory(str(path), [.5])


def test_mixed_endpoint_velocity_uses_source_linear_rule(tmp_path):
    path, data = native(tmp_path)
    data['points'][0]['velocities'] = [100.]
    path.write_text(json.dumps(data))
    row = sample_joint_trajectory(str(path), [.25])['samples'][0]
    assert np.array_equal(row['positions'], [.25]) and row['interpolation'] == 'linear'


@pytest.mark.parametrize('field', ['accelerations', 'effort'])
def test_report_exposes_derivative_or_effort_only_discrepancy(tmp_path, field):
    a, data = native(tmp_path, velocities=[0., 0.], accelerations=[0., 0.], name='reference.json')
    for point in data['points']:
        point['effort'] = [0.]
    a.write_text(json.dumps(data))
    data['points'][0][field] = [1.]
    b = tmp_path / 'candidate.json'
    b.write_text(json.dumps(data))
    result = compare_joint_trajectories(str(a), str(b), [0.])
    assert set(result['different_sample_fields']) == {field}
    assert result['first_position_difference'] is None
    rendered = markdown_report(result).replace('\\', '')
    row = next(line for line in rendered.splitlines() if line.startswith(f'| {field} |'))
    assert '[0.0]' in row and '[1.0]' in row
