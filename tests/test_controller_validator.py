"""Independent controller-adapter regressions and semantic controls."""
import copy
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.trajectory import compare_joint_trajectories, sample_joint_trajectory


def write_trajectory(tmp_path, data, name='trajectory.json'):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def base_trajectory():
    return {'joint_names': ['a', 'b'], 'points': [
        {'positions': [0., 2.], 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1., 3.], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}


@pytest.mark.parametrize('location', ['position', 'sample_time', 'limit'])
def test_huge_integer_refuses_with_input_error(tmp_path, location):
    data = base_trajectory()
    kwargs, times = {}, [.5]
    if location == 'position':
        data['points'][0]['positions'][0] = 10**400
    elif location == 'sample_time':
        times = [10**400]
    else:
        kwargs['position_limits'] = {'a': [0, 10**400]}
    with pytest.raises(InputError):
        sample_joint_trajectory(write_trajectory(tmp_path, data), times, **kwargs)


def test_native_none_forwards_next_waypoint_at_left_boundary(tmp_path):
    path = write_trajectory(tmp_path, base_trajectory())
    result = sample_joint_trajectory(path, [0., .5, 1., 2.], interpolation_method='none')
    assert [row['positions'] for row in result['samples']] == [[1., 3.]] * 4
    assert result['samples'][0]['velocities'] == []
    assert result['samples'][-1]['velocities'] == [0., 0.]
    assert result['execution_status'] == 'unknown'


def test_named_joint_permutation_is_equivalent(tmp_path):
    first = base_trajectory()
    second = copy.deepcopy(first)
    second['joint_names'].reverse()
    for point in second['points']:
        point['positions'].reverse()
    a = write_trajectory(tmp_path, first, 'a.json')
    b = write_trajectory(tmp_path, second, 'b.json')
    result = compare_joint_trajectories(a, b, [.25, .75, 1.])
    assert result['status'] == 'pass'
    assert result['same_waypoint_positions_and_times']


def test_missing_initial_state_does_not_invent_a_sample(tmp_path):
    data = base_trajectory()
    for point in data['points']:
        point['time_from_start']['sec'] += 1
    result = sample_joint_trajectory(write_trajectory(tmp_path, data), [0., 1., 1.5])
    assert result['status'] == 'unknown'
    assert result['samples'][0]['status'] == 'unknown'
    assert not result['sampling_complete']


def test_nanosecond_rounding_collision_refuses(tmp_path):
    with pytest.raises(InputError, match='collide'):
        sample_joint_trajectory(write_trajectory(tmp_path, base_trajectory()), [0., 1e-12])


def test_acceleration_without_endpoint_velocities_refuses(tmp_path):
    data = base_trajectory()
    for point in data['points']:
        point['accelerations'] = [0., 0.]
    with pytest.raises(InputError, match='Acceleration'):
        sample_joint_trajectory(write_trajectory(tmp_path, data), [.5])


@pytest.mark.parametrize('field', ['accelerations', 'effort'])
def test_markdown_preserves_nonposition_differences(tmp_path, field):
    from policy_doctor.reports import markdown_report

    first = base_trajectory()
    for point in first['points']:
        point['velocities'] = [0., 0.]
        point['accelerations'] = [0., 0.]
        point['effort'] = [0., 0.]
    second = copy.deepcopy(first)
    second['points'][0][field][0] = 1.
    a = write_trajectory(tmp_path, first, 'first.json')
    b = write_trajectory(tmp_path, second, 'second.json')
    report = compare_joint_trajectories(a, b, [0.])
    assert report['status'] == 'different'
    assert report['first_position_difference'] is None
    assert field in report['different_sample_fields']
    assert field in markdown_report(report).lower()
