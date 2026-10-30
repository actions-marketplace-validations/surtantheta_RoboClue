"""Packaged, generated native-message demo; no optional extras or robot execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .reports import markdown_report
from .trajectory import compare_joint_trajectories, sample_joint_trajectory


def create_controller_demo(destination: str | Path) -> dict:
    # mkdir must see the original leaf, including an existing dangling symlink.
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    destination = destination.resolve()
    def write(path, value):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    baseline = {'header': {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''},
        'joint_names': ['shoulder'], 'points': [
            {'positions': [0.], 'time_from_start': {'sec': 0, 'nanosec': 0}},
            {'positions': [1.], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}
    candidate = copy.deepcopy(baseline)
    candidate['points'][0]['velocities'] = [4.]
    candidate['points'][1]['velocities'] = [0.]
    for name, value in [('reference', baseline), ('candidate', candidate), ('corrected', baseline)]:
        write(destination / f'{name}.json', value)
    times = [0., .25, .5, .75, 1.]
    limits = {'shoulder': [0., 1.]}
    reports = {
        'comparison': compare_joint_trajectories(str(destination / 'reference.json'), str(destination / 'candidate.json'), times),
        'candidate_bounds': sample_joint_trajectory(str(destination / 'candidate.json'), times, position_limits=limits),
        'corrected': compare_joint_trajectories(str(destination / 'reference.json'), str(destination / 'corrected.json'), times, position_limits=limits)}
    # Healthy named-joint reorder: the same physical channel values in another native order.
    ordered = copy.deepcopy(baseline)
    ordered['joint_names'] += ['elbow']
    for point in ordered['points']:
        point['positions'] += [point['positions'][0] * 2]
    reordered = copy.deepcopy(ordered)
    reordered['joint_names'].reverse()
    for point in reordered['points']:
        point['positions'].reverse()
    write(destination / 'ordered.json', ordered)
    write(destination / 'reordered.json', reordered)
    reports['valid_reordering'] = compare_joint_trajectories(str(destination / 'ordered.json'), str(destination / 'reordered.json'), times)
    # Missing initial state prevents inventing a startup path.
    later = copy.deepcopy(baseline)
    for point in later['points']:
        point['time_from_start']['sec'] += 1
    write(destination / 'missing_initial_state.json', later)
    write(destination / 'position_limits.json', limits)
    parameters = {'sample_times': [.75], 'interpolation_method': 'splines', 'position_limits': limits}
    write(destination / 'incident_controls.json', {'controls': [
        {'name': 'healthy reference', 'source': 'reference.json', 'parameters': parameters, 'expected_status': 'pass'},
        {'name': 'missing initial state', 'source': 'missing_initial_state.json', 'parameters': parameters, 'expected_status': 'unknown'}]})
    reports['unknown_initial_state'] = sample_joint_trajectory(str(destination / 'missing_initial_state.json'), [0., 1., 1.5])
    for name, report in reports.items():
        write(destination / f'{name}_report.json', report)
        with (destination / f'{name}_report.md').open('x', encoding='utf-8') as stream:
            stream.write(markdown_report(report))
    summary = {'kind': 'generated_trajectory_demo', 'status': 'pass', 'destination': str(destination),
        'summary': 'Same waypoint positions and times, different sampled references after adding endpoint velocities.',
        'report_paths': {name: str(destination / f'{name}_report.md') for name in reports},
        'scope': 'Generated native-message arithmetic fixture, inspired by a documented issue mechanism. Not original incident/hardware reproduction.',
        'same_waypoint_positions_and_times': reports['comparison']['same_waypoint_positions_and_times'],
        'first_position_difference': reports['comparison']['first_position_difference'],
        'candidate_position_at_750ms': reports['candidate_bounds']['samples'][3]['positions'],
        'candidate_bounds_status': reports['candidate_bounds']['status'],
        'corrected_status': reports['corrected']['status'], 'valid_reordering': reports['valid_reordering']['status'],
        'missing_initial_state': reports['unknown_initial_state']['status'],
        'execution_status': 'unknown', 'physical_cause': 'unknown',
        'upstream_reference': reports['comparison']['upstream_reference'],
        'public_mechanism': 'https://github.com/ros-controls/ros2_controllers/issues/1298#issuecomment-2380303228'}
    assert summary['candidate_position_at_750ms'] == [1.03125]
    assert summary['candidate_bounds_status'] == 'error'
    assert summary['corrected_status'] == summary['valid_reordering'] == 'pass'
    assert summary['missing_initial_state'] == 'unknown'
    with (destination / 'README.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown_report(summary))
    write(destination / 'summary.json', summary)
    return summary



def create_policy_demo(destination: str | Path) -> dict:
    """Create a synthetic metadata-order fault and its corrected dataset; no weights or network."""
    import shlex

    import numpy as np
    from safetensors.numpy import save_file

    # Preserve refusal for every existing leaf, including dangling symlinks.
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    destination = destination.resolve()
    checkpoint = destination / 'checkpoint'
    checkpoint.mkdir()

    def write(path, value):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

    names = ['shoulder', 'gripper']
    state = {'type': 'STATE', 'shape': [2]}
    action = {'type': 'ACTION', 'shape': [2]}
    features = {'observation.state': state, 'action': action}
    mapping = {'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}
    write(checkpoint / 'config.json', {
        'type': 'act', 'input_features': {'observation.state': state},
        'output_features': {'action': action}, 'normalization_mapping': mapping,
        'chunk_size': 4, 'n_action_steps': 4})
    normalization = {'features': features, 'norm_map': mapping, 'eps': 1e-8}
    write(checkpoint / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'relative_actions_processor',
         'config': {'enabled': True, 'action_names': names, 'exclude_joints': ['gripper']}},
        {'registry_name': 'normalizer_processor', 'config': normalization, 'state_file': 'pre.safetensors'}]})
    write(checkpoint / 'policy_postprocessor.json', {'steps': [
        {'registry_name': 'unnormalizer_processor', 'config': normalization, 'state_file': 'post.safetensors'},
        {'registry_name': 'absolute_actions_processor', 'config': {'enabled': True}}]})
    stats = {'action.mean': np.array([1., .5], dtype=np.float32),
             'action.std': np.array([2., .25], dtype=np.float32),
             'observation.state.mean': np.array([10., .5], dtype=np.float32),
             'observation.state.std': np.array([2., .25], dtype=np.float32)}
    for filename in ('pre.safetensors', 'post.safetensors'):
        save_file(stats, checkpoint / filename, metadata={'synthetic': 'true'})
    for folder, dataset_names in [('dataset', list(reversed(names))), ('corrected_dataset', names)]:
        metadata = destination / folder / 'meta'
        metadata.mkdir(parents=True)
        write(metadata / 'info.json', {'codebase_version': 'v3.0', 'robot_type': 'synthetic_arm',
            'fps': 30, 'synthetic': True,
            'features': {key: dict(value, names=dataset_names) for key, value in features.items()}})
    summary = {'kind': 'generated_policy_demo', 'status': 'pass', 'synthetic': True,
        'destination': str(destination), 'checkpoint': str(checkpoint),
        'dataset': str(destination / 'dataset'), 'corrected_dataset': str(destination / 'corrected_dataset'),
        'next_command': f'policy-doctor check {shlex.quote(str(checkpoint))} '
                        f'--dataset {shlex.quote(str(destination / "dataset"))}',
        'scope': 'Generated metadata and normalization statistics, not robot recordings or a trained policy. '
                 'Creation succeeds; the first check intentionally reports an order contradiction. '
                 'Corrected names do not establish deployment compatibility.'}
    write(destination / 'summary.json', summary)
    with (destination / 'README.md').open('x', encoding='utf-8') as stream:
        stream.write('# Synthetic policy metadata example\n\n'
            'All checkpoint metadata, normalization statistics and dataset declarations here are generated. '
            'There are no model weights, recordings or robot execution.\n\n'
            'The saved relative-action processor names are `[shoulder, gripper]`; both dataset action and '
            'state names are reversed. Check the contradiction with:\n\n'
            f'```sh\n{summary["next_command"]}\n```\n\n'
            'Use `corrected_dataset` with the same checkpoint to clear the name-order errors. '
            'Deployment semantics remain unknown; a corrected metadata check is not deployment proof.\n')
    return summary
