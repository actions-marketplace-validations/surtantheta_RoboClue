"""Read a public issue's native trajectory and check numerical behavior; no robot.

Explicit network access fetches only GitHub issue1298. PyYAML is needed for its
fenced ROS message; the product's native JSON reader has no YAML dependency.
Original issue content stays in the selected local work directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from policy_doctor.trajectory import compare_joint_trajectories, sample_joint_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path, help='New directory for fetched source and native messages')
    parser.add_argument('--output', type=Path, required=True, help='New compact evidence file')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    url = 'https://api.github.com/repos/ros-controls/ros2_controllers/issues/1298'
    request = urllib.request.Request(url, headers={'User-Agent': 'policy-doctor-local-validation'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(128001)
    if len(raw) > 128000:
        raise SystemExit('Public issue exceeds bounded intake')
    issue = json.loads(raw)
    (args.work / 'issue.json').write_bytes(raw)
    blocks = re.findall(r'```[^\n]*\n(.*?)```', issue['body'], re.S)
    selected = next((block for block in blocks if 'joint_trajectory:' in block), None)
    if selected is None:
        raise SystemExit('Native trajectory block not present; source changed')
    if any(isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.TagToken)) for token in yaml.scan(selected)):
        raise SystemExit('YAML aliases or explicit tags are refused')
    native = yaml.safe_load(selected)['joint_trajectory']
    trajectory = args.work / 'trajectory.json'
    trajectory.write_text(json.dumps(native, indent=2, allow_nan=False) + '\n')
    end = native['points'][-1]['time_from_start']
    times = np.linspace(0, end['sec'] + end['nanosec']/1e9, 1301).tolist()
    sampled = sample_joint_trajectory(str(trajectory), times)
    # This is an explicit arithmetic counterfactual, not a claimed physical repair.
    positions_only = copy.deepcopy(native)
    for point in positions_only['points']:
        point.pop('velocities', None)
        point.pop('accelerations', None)
    counterfactual = args.work / 'positions_only.json'
    counterfactual.write_text(json.dumps(positions_only, indent=2) + '\n')
    compared = compare_joint_trajectories(str(trajectory), str(counterfactual), times)
    joint = native['joint_names'].index('shoulder_pan_joint')
    waypoint = [p['positions'][joint] for p in native['points']]
    values = [row['positions'][joint] for row in sampled['samples'] if row['status'] == 'sampled']
    result = {'scope': 'Publicly supplied native trajectory sampled under pinned Jazzy rules; no original host, controller, hardware or historical-version replay.',
        'url': 'https://github.com/ros-controls/ros2_controllers/issues/1298',
        'issue_updated_at': issue['updated_at'], 'issue_response_sha256': hashlib.sha256(raw).hexdigest(),
        'native_trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        'transformation': 'Fenced joint_trajectory YAML parsed to native JSON without numeric edits.',
        'native_points': len(native['points']), 'joints': native['joint_names'], 'sample_rows': len(times),
        'shoulder_waypoint_range': [min(waypoint), max(waypoint)],
        'shoulder_sampled_reference_range': [min(values), max(values)],
        'waypoint_envelope_excursion_count': sampled['excursion_count'],
        'positions_only_counterfactual': {'same_waypoint_positions_and_times': compared['same_waypoint_positions_and_times'],
            'sampled_references_equal': compared['sampled_references_equal'], 'scope': 'Derived field-removal comparison only, not a recommended universal fix.'},
        'upstream_reference': sampled['upstream_reference'], 'execution_status': 'unknown'}
    assert result['native_points'] == 32 and sampled['sampling_complete']
    assert -.77 < min(values) < -.75
    assert compared['same_waypoint_positions_and_times'] and compared['status'] == 'different'
    # Native endpoint quarter samples for the optional unchanged C++ body gate.
    cases = []
    for index, (a, b) in enumerate(zip(native['points'], native['points'][1:], strict=False)):
        t0, t1 = [p['time_from_start']['sec']*10**9 + p['time_from_start']['nanosec'] for p in (a, b)]
        for numerator in (1, 2, 3):
            cases.append({'name': f'public-segment-{index}-quarter-{numerator}', 't0_ns': t0, 't1_ns': t1,
                'sample_ns': t0 + (t1-t0)*numerator//4,
                'a': {field: a.get(field, []) for field in ('positions', 'velocities', 'accelerations', 'effort')},
                'b': {field: b.get(field, []) for field in ('positions', 'velocities', 'accelerations', 'effort')}})
    (args.work / 'reference_cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
