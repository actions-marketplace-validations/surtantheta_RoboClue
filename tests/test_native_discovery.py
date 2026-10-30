import json
import shlex

import yaml

from policy_doctor.cli import human
from policy_doctor.demo import create_controller_demo
from policy_doctor.investigation import investigate_run
from policy_doctor.trajectory import sample_joint_trajectory


def test_packaged_demo_reaches_existing_trajectory_tools(tmp_path):
    folder = tmp_path / 'incident'
    create_controller_demo(folder)
    report = investigate_run(str(folder))
    candidates = report['inventory']['trajectories']
    candidate = next(c for c in candidates if c['path'].endswith('/candidate.json'))
    assert candidate['point_count'] == 2
    assert candidate['derivative_points']['velocities'] == 2
    action = next(a for a in report['next_actions'] if a.get('arguments', {}).get('trajectory') == candidate['path'])
    sampled = sample_joint_trajectory(**action['arguments'])
    assert sampled['excursion_count'] > 0
    assert candidate['execution_status'] == 'unknown'
    assert report['status'] == 'unknown'
    assert not any(a['tool'] == 'compare_ros_trajectories' and a['arguments'] for a in report['next_actions'])


def test_invalid_native_candidate_retains_intake_reason(tmp_path):
    (tmp_path / 'message.json').write_text(json.dumps({'joint_names': ['j'], 'points': []}))
    report = investigate_run(str(tmp_path))
    candidate = report['inventory']['trajectories'][0]
    assert candidate['needs_selection_or_adapter']
    assert report['intake_problems']
    assert not any(a['arguments'] for a in report['next_actions'])


def test_dispatch_evidence_routes_without_associating_recording(tmp_path):
    evidence = tmp_path / 'trace.json'
    evidence.write_text(json.dumps({'schema_version': 1, 'kind': 'action_dispatch_evidence'}))
    report = investigate_run(str(tmp_path))
    assert report['inventory']['dispatch_evidence'] == [str(evidence)]
    action = next(a for a in report['next_actions'] if a['tool'] == 'audit_action_boundary')
    assert action['arguments'] == {'evidence': str(evidence)}
    assert report['status'] == 'unknown'  # Discovery does not validate the declared evidence.


def test_unrelated_json_is_not_trajectory(tmp_path):
    (tmp_path / 'report.json').write_text(json.dumps({'kind': 'joint_trajectory_samples', 'samples': []}))
    report = investigate_run(str(tmp_path))
    assert not report['inventory']['trajectories']
    assert not report['next_actions']


def test_existing_yaml_wrapper_routes_without_manual_conversion(tmp_path):
    source = tmp_path / 'moveit.yaml'
    source.write_text(yaml.safe_dump({'joint_trajectory': {'joint_names': ['j'], 'points': [
        {'positions': [0], 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}}))
    report = investigate_run(str(tmp_path))
    candidate = report['inventory']['trajectories'][0]
    assert candidate['message_pointer'] == '/joint_trajectory'
    action = next(a for a in report['next_actions'] if a['tool'] == 'sample_ros_trajectory')
    assert sample_joint_trajectory(**action['arguments'])['samples'][1]['positions'] == [.25]
    assert report['status'] == 'unknown'


def test_native_times_outside_sampler_range_do_not_offer_invalid_action(tmp_path):
    source = tmp_path / 'long.json'
    source.write_text(json.dumps({'joint_names': ['j'], 'points': [
        {'positions': [1], 'time_from_start': {'sec': 3601, 'nanosec': 0}}]}))
    report = investigate_run(str(tmp_path))
    assert report['inventory']['trajectories'][0]['needs_selection_or_adapter']
    assert not any(a['arguments'] for a in report['next_actions'])


def test_first_human_result_is_concise_and_command_preserves_path(tmp_path):
    source = tmp_path / 'space;$(touch NOT_EXECUTED).json'
    source.write_text(json.dumps({'joint_names': ['j'], 'points': [
        {'positions': [0], 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}))
    report = investigate_run(str(tmp_path))
    text = human(report)
    command = next(line.removeprefix('Suggested check: ') for line in text.splitlines() if line.startswith('Suggested check: '))
    assert shlex.split(command)[2] == str(source)
    assert len(text) < 2200
    assert '"schema_version"' not in text
    sampled = sample_joint_trajectory(**report['next_actions'][0]['arguments'])
    assert len(human(sampled)) < 1800
    assert 'Execution and physical cause: unknown' in human(sampled)
