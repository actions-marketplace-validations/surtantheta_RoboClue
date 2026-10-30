# Polynomial coefficient equations adapted from trajectory.cpp:
# Copyright 2017 Open Source Robotics Foundation, Inc.
# Licensed under Apache-2.0; see LICENSE-APACHE-2.0 and THIRD_PARTY_NOTICES.md.
# Local implementation adds bounded native text intake, explicit unknowns and evidence reports.
"""Offline sampling of position-bearing ROS JointTrajectory messages, without ROS.

Semantics pinned to ros2_controllers Jazzy 8ce934536c6404750fb8aafc0b62b7d365049143.
This models sampled references, never command acceptance, PID or physical execution.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import MAX_JSON_BYTES, InputError, _pairs, finite_tree

REVISION = '8ce934536c6404750fb8aafc0b62b7d365049143'
REFERENCE = {'repository': 'https://github.com/ros-controls/ros2_controllers', 'revision': REVISION,
    'file': 'joint_trajectory_controller/src/trajectory.cpp',
    'sha256': 'a979c1bca2527a912571f5e1ed29e5887d7bf92aa226f6918a125a462821537d',
    'url': f'https://github.com/ros-controls/ros2_controllers/blob/{REVISION}/joint_trajectory_controller/src/trajectory.cpp'}
FIELDS = ('positions', 'velocities', 'accelerations', 'effort')
SCOPE = ('Offline sampled trajectory references under the selected pinned interpolation rules. '
         'No ROS node, controller, PID, driver or robot is run. Units, active controller configuration, '
         'acceptance, preemption, completed execution and physical cause remain unverified.')
MAX_DOCUMENT_DEPTH = 64
MAX_DOCUMENT_NODES = 500000
MAX_YAML_TOKENS = 1000000


def _document_tree(data):
    """Bound decoded structure, including wrapper fields not numerically modeled."""
    pending, seen = [(data, 0)], 0
    while pending:
        value, depth = pending.pop()
        seen += 1
        if depth > MAX_DOCUMENT_DEPTH or seen > MAX_DOCUMENT_NODES:
            raise InputError('Native document nesting/node budget exceeded')
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise InputError('Native document mapping keys must be strings')
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif value is not None and type(value) not in (str, bool, int, float):
            raise InputError('Native document contains unsupported scalar types')
    finite_tree(data)


def _yaml_document(text):
    try:
        import yaml
    except ImportError as exc:
        raise InputError("YAML intake requires PyYAML; install 'policy-deployment-doctor[ros]'") from exc

    class NativeSafeLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            pairs = []
            for key, value in node.value:
                if not isinstance(key, yaml.ScalarNode) or key.tag != 'tag:yaml.org,2002:str':
                    raise InputError('YAML mapping keys must be strings; merge keys are unsupported')
                pairs.append((self.construct_object(key, deep=deep), self.construct_object(value, deep=deep)))
            return _pairs(pairs)

    depth = count = 0
    try:
        # Scan before composition: aliases and deep structures never reach the loader.
        for token in yaml.scan(text, Loader=yaml.SafeLoader):
            count += 1
            if count > MAX_YAML_TOKENS:
                raise InputError('YAML token budget exceeded')
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken, yaml.tokens.TagToken)):
                raise InputError('YAML anchors, aliases and explicit tags are unsupported')
            if isinstance(token, (yaml.tokens.BlockMappingStartToken, yaml.tokens.BlockSequenceStartToken,
                                  yaml.tokens.FlowMappingStartToken, yaml.tokens.FlowSequenceStartToken)):
                depth += 1
                if depth > MAX_DOCUMENT_DEPTH:
                    raise InputError('YAML nesting budget exceeded')
            elif isinstance(token, (yaml.tokens.BlockEndToken, yaml.tokens.FlowMappingEndToken,
                                    yaml.tokens.FlowSequenceEndToken)):
                depth -= 1
        loader = NativeSafeLoader(text)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()
    except (yaml.YAMLError, RecursionError) as exc:
        raise InputError(f'Cannot read native YAML trajectory: {exc}') from exc


def _read_document(path):
    """Read one bounded native JSON/YAML document without selecting a message."""
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix not in ('.json', '.yaml', '.yml'):
        raise InputError('Native trajectory extension must be .json, .yaml or .yml')
    try:
        if not source.is_file() or source.stat().st_size > MAX_JSON_BYTES:
            raise InputError('Trajectory must be a bounded native JSON/YAML file')
        with source.open('rb') as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise InputError('Trajectory exceeds native document byte budget')
        text = raw.decode('utf-8')
        # A .json file is never reinterpreted as YAML after a JSON error.
        data = json.loads(text, object_pairs_hook=_pairs) if suffix == '.json' else _yaml_document(text)
        _document_tree(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise InputError(f'Cannot read native {"JSON" if suffix == ".json" else "YAML"} trajectory: {exc}') from exc
    return data, {'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'size_bytes': len(raw), 'format': 'JSON' if suffix == '.json' else 'YAML'}


def _select_message(data):
    if not isinstance(data, dict):
        raise InputError('Expected one native trajectory message object')
    if 'joint_trajectory' in data:
        allowed = {'joint_trajectory', 'multi_dof_joint_trajectory'}
        key, multi, kind = 'joint_trajectory', 'multi_dof_joint_trajectory', 'moveit_msgs/msg/RobotTrajectory'
    elif 'trajectory' in data:
        allowed = {'trajectory', 'multi_dof_trajectory', 'path_tolerance', 'component_path_tolerance',
                   'goal_tolerance', 'component_goal_tolerance', 'goal_time_tolerance'}
        key, multi, kind = 'trajectory', 'multi_dof_trajectory', 'control_msgs/action/FollowJointTrajectory.Goal'
    else:
        return data, '', 'trajectory_msgs/msg/JointTrajectory', []
    if set(data) - allowed:
        raise InputError('Ambiguous or unsupported native trajectory wrapper fields')
    if multi in data:
        value = data[multi]
        if (not isinstance(value, dict) or set(value) - {'header', 'joint_names', 'points'} or
                value.get('joint_names', []) != [] or value.get('points', []) != []):
            raise InputError('Nonempty or invalid multi_dof trajectory is unsupported; it cannot be discarded')
    return data[key], '/' + key, kind, sorted(set(data) - {key, multi})


def _number(value):
    if type(value) is int and abs(value) > 2**53:
        raise InputError('Integer would lose precision in a float64 trajectory field')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise InputError('Require finite real numbers, not booleans')
    return float(value)


def _duration(value):
    if not isinstance(value, dict) or set(value) != {'sec', 'nanosec'}:
        raise InputError('Native duration requires sec and nanosec')
    sec, nano = value['sec'], value['nanosec']
    if type(sec) is not int or type(nano) is not int or not 0 <= nano < 10**9 or not 0 <= sec <= 3600:
        raise InputError('Require nonnegative native duration <=3600 seconds and canonical nanosec')
    total = sec * 10**9 + nano
    if total > 3600 * 10**9:
        raise InputError('Trajectory duration exceeds 3600 seconds')
    return total


def _load(path):
    data, artifact = _read_document(path)
    data, pointer, kind, ignored = _select_message(data)
    artifact.update(format=kind + ' ' + artifact['format'], message_pointer=pointer,
                    wrapper_fields_not_modeled=ignored)
    if not isinstance(data, dict) or set(data) - {'header', 'joint_names', 'points'}:
        raise InputError('Expected native JointTrajectory fields header/joint_names/points')
    names, points = data.get('joint_names'), data.get('points')
    if (not isinstance(names, list) or not 1 <= len(names) <= 64 or
            any(not isinstance(n, str) or not n or len(n) > 256 for n in names) or len(set(names)) != len(names)):
        raise InputError('Require 1–64 unique nonempty native joint names')
    if not isinstance(points, list) or not 1 <= len(points) <= 10000 or len(points) * len(names) > 200000:
        raise InputError('Trajectory point/joint budget exceeded or no points')
    times, parsed = [], []
    for point in points:
        if not isinstance(point, dict) or set(point) - {*FIELDS, 'time_from_start'}:
            raise InputError('Unsupported native trajectory point fields')
        times.append(_duration(point.get('time_from_start')))
        arrays = {}
        for field in FIELDS:
            values = point.get(field, [])
            if not isinstance(values, list) or len(values) not in (0, len(names)):
                raise InputError('Each native point field must be empty or match joint_names width')
            arrays[field] = np.array([_number(v) for v in values], dtype=np.float64)
        if not len(arrays['positions']):
            raise InputError('Positions required; derivative integration needs initial state and is unsupported')
        parsed.append(arrays)
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise InputError('Native time_from_start must be strictly increasing')
    header = data.get('header', {'stamp': {'sec': 0, 'nanosec': 0}, 'frame_id': ''})
    if not isinstance(header, dict) or set(header) - {'stamp', 'frame_id'} or not isinstance(header.get('frame_id', ''), str):
        raise InputError('Invalid native trajectory header')
    stamp = header.get('stamp', {'sec': 0, 'nanosec': 0})
    if (not isinstance(stamp, dict) or set(stamp) != {'sec', 'nanosec'} or
            type(stamp['sec']) is not int or type(stamp['nanosec']) is not int or
            not -(2**31) <= stamp['sec'] < 2**31 or not 0 <= stamp['nanosec'] < 10**9):
        raise InputError('Invalid native header timestamp')
    return names, times, parsed, header, artifact


def _interpolate(a, b, elapsed, duration):
    """Position-bearing linear/cubic/quintic Hermite sampling; no integration."""
    velocity = bool(len(a['velocities']) and len(b['velocities']))
    acceleration = bool(len(a['accelerations']) and len(b['accelerations']))
    if acceleration and not velocity:
        raise InputError('Acceleration without both endpoint velocities is unsupported')
    p, end = a['positions'], b['positions']
    coefficients = [p, (end - p) / duration]
    degree = 'linear'
    if velocity:
        v, w = a['velocities'], b['velocities']
        coefficients = [p, v, (-3*p + 3*end - 2*v*duration - w*duration) / duration**2,
                         (2*p - 2*end + v*duration + w*duration) / duration**3]
        degree = 'cubic'
    if acceleration:
        c, d = a['accelerations'], b['accelerations']
        coefficients = [p, v, c / 2,
            (-20*p + 20*end - 3*c*duration**2 + d*duration**2 - 12*v*duration - 8*w*duration) / (2*duration**3),
            (30*p - 30*end + 3*c*duration**2 - 2*d*duration**2 + 16*v*duration + 14*w*duration) / (2*duration**4),
            (-12*p + 12*end - c*duration**2 + d*duration**2 - 6*v*duration - 6*w*duration) / (2*duration**5)]
        degree = 'quintic'
    zero = np.zeros_like(p)
    output = {'positions': sum((x * elapsed**i for i, x in enumerate(coefficients)), zero.copy()),
              'velocities': sum((i*x*elapsed**(i-1) for i, x in enumerate(coefficients) if i), zero.copy()),
              'accelerations': sum((i*(i-1)*x*elapsed**(i-2) for i, x in enumerate(coefficients) if i > 1), zero.copy()),
              'effort': zero.copy()}
    if len(a['effort']) and len(b['effort']):
        output['effort'] = a['effort'] + elapsed * (b['effort'] - a['effort']) / duration
    return output, degree


def sample_joint_trajectory(trajectory: str, sample_times: list[float], *, interpolation_method: str = 'splines',
                            position_limits: dict | None = None) -> dict:
    if interpolation_method not in ('splines', 'none'):
        raise InputError("Use native interpolation_method 'splines' or 'none'")
    if not isinstance(sample_times, list) or not 1 <= len(sample_times) <= 2000:
        raise InputError('Require 1–2000 relative sample times in seconds')
    requested = [_number(v) for v in sample_times]
    if any(t < 0 or t > 3600 for t in requested) or any(b <= a for a, b in zip(requested, requested[1:], strict=False)):
        raise InputError('Sample times must increase strictly within [0,3600] seconds')
    sample_ns = [round(t * 10**9) for t in requested]
    if any(b <= a for a, b in zip(sample_ns, sample_ns[1:], strict=False)):
        raise InputError('Sample times collide after rounding to integer nanoseconds')
    names, times, points, header, artifact = _load(trajectory)
    if len(names) * len(sample_ns) > 100000:
        raise InputError('Sample/joint output budget exceeded')
    limits = {}
    if position_limits is not None:
        if not isinstance(position_limits, dict) or not position_limits or set(position_limits) - set(names):
            raise InputError('Position limits must name present joints')
        for name, pair in position_limits.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise InputError('Each position limit requires [lower,upper]')
            bounds = [_number(v) for v in pair]
            if bounds[0] > bounds[1]:
                raise InputError('Position limits must be ordered')
            limits[name] = bounds
    samples, violations, excursions, excursion_count = [], [], [], 0
    for requested_time, ns in zip(requested, sample_ns, strict=True):
        row = {'requested_seconds': requested_time, 'time_from_start_ns': ns, 'role': 'sampled_reference'}
        if ns < times[0]:
            samples.append(dict(row, status='unknown', reason='Missing pre-trajectory state; no initial point is invented.'))
            continue
        index = bisect.bisect_right(times, ns) - 1
        if index == len(times) - 1:
            output = {field: points[index][field] if len(points[index][field]) else np.zeros(len(names)) for field in FIELDS}
            degree, segment = 'last_point', [index]
        elif interpolation_method == 'none':
            output = points[index + 1]
            degree, segment = 'next_waypoint', [index + 1]
        else:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                output, degree = _interpolate(points[index], points[index + 1], (ns-times[index])/10**9,
                                               (times[index+1]-times[index])/10**9)
            segment = [index, index + 1]
        if any(not np.isfinite(v).all() for v in output.values()):
            raise InputError('Sampling overflow; no non-finite references are returned')
        row.update(status='sampled', interpolation=degree, source_points=segment,
                   **{field: v.tolist() for field, v in output.items()})
        samples.append(row)
        if len(segment) == 2:
            left, right = (points[i]['positions'] for i in segment)
            lower, upper = np.minimum(left, right), np.maximum(left, right)
            for joint, value, low, high in zip(names, row['positions'], lower, upper, strict=True):
                tolerance = 1e-12 * max(1., abs(low), abs(high))
                if value < low - tolerance or value > high + tolerance:
                    excursion_count += 1
                    if len(excursions) < 64:
                        excursions.append({'time_from_start_ns': ns, 'joint': joint, 'value': value,
                            'waypoint_envelope': [float(low), float(high)], 'source_points': segment,
                            'evidence_kind': 'derived_sample'})
        for joint, value in zip(names, row['positions'], strict=True):
            if joint in limits and not limits[joint][0] <= value <= limits[joint][1]:
                violations.append({'time_from_start_ns': ns, 'joint': joint, 'value': value, 'bounds': limits[joint],
                    'evidence_kind': 'derived_sample', 'source_points': segment})
    complete = all(s['status'] == 'sampled' for s in samples)
    return {'schema_version': 1, 'kind': 'joint_trajectory_samples',
        'status': 'error' if violations else 'pass' if complete else 'unknown', 'scope': SCOPE,
        'joint_names': names, 'header': header, 'interpolation_method': interpolation_method,
        'header_treatment': 'Samples use time_from_start; header time is preserved, not mapped to a live clock.',
        'sample_time_treatment': 'Requested seconds rounded to nearest integer nanosecond.',
        'samples': samples, 'sampling_complete': complete, 'position_limits': limits, 'sampled_limit_violations': violations,
        'waypoint_envelope_excursions': excursions, 'excursion_count': excursion_count,
        'excursions_truncated': excursion_count > len(excursions),
        'excursion_scope': 'Review hints: sampled references outside adjacent waypoint position envelopes can be intentional. These envelopes are not physical joint limits; roundoff tolerance is 1e-12 times max(1,abs(endpoints)).',
        'limits_scope': 'User-supplied numeric reference bounds at sampled times only; no physical safety or between-sample certification.',
        'waypoint_times_ns': times, 'waypoint_positions': [p['positions'].tolist() for p in points],
        'upstream_reference': REFERENCE.copy(), 'artifacts': [artifact],
        'execution_status': 'unknown', 'physical_cause': 'unknown',
        'unsupported': ['derivative-only goals/integration', 'initial-state interpolation before first waypoint',
                        'continuous-joint wrapping', 'PID/feed-forward control law', 'goal acceptance/preemption/tolerances']}


def compare_joint_trajectories(reference: str, candidate: str, sample_times: list[float], *,
        reference_method: str = 'splines', candidate_method: str = 'splines', position_limits: dict | None = None) -> dict:
    a = sample_joint_trajectory(reference, sample_times, interpolation_method=reference_method, position_limits=position_limits)
    b = sample_joint_trajectory(candidate, sample_times, interpolation_method=candidate_method, position_limits=position_limits)
    same_joints = set(a['joint_names']) == set(b['joint_names'])
    permutation = [b['joint_names'].index(n) for n in a['joint_names']] if same_joints else None
    same_positions = None
    differences, first = {}, None
    if same_joints:
        same_positions = (a['waypoint_times_ns'] == b['waypoint_times_ns'] and
            np.array_equal(a['waypoint_positions'], np.asarray(b['waypoint_positions'])[:, permutation]))
        for left, right in zip(a['samples'], b['samples'], strict=True):
            if left['status'] != 'sampled' or right['status'] != 'sampled':
                continue
            for field in FIELDS:
                x, y = np.asarray(left[field]), np.asarray(right[field])
                if y.size:
                    y = y[permutation]
                if x.shape != y.shape or not np.array_equal(x, y):
                    differences.setdefault(field, {'time_from_start_ns': left['time_from_start_ns'],
                        'reference_values': x.tolist(), 'candidate_values': y.tolist()})
            if first is None and 'positions' in differences:
                first = dict(differences['positions'], joint_names=a['joint_names'])
    complete = bool(same_joints and a['sampling_complete'] and b['sampling_complete'])
    return {'schema_version': 1, 'kind': 'joint_trajectory_comparison',
        'status': 'error' if a['status'] == 'error' or b['status'] == 'error' else 'unknown' if not complete else 'different' if differences else 'pass', 'scope': SCOPE,
        'comparison_complete': complete, 'same_waypoint_positions_and_times': bool(same_positions) if same_positions is not None else None,
        'reference_sampling_status': a['status'], 'candidate_sampling_status': b['status'],
        'sampled_references_equal': not bool(differences) if complete else None,
        'joint_alignment': 'verified_by_native_name' if same_joints else 'unknown: different joint sets',
        'first_position_difference': first, 'different_sample_fields': differences,
        'reference': a, 'candidate': b, 'artifacts': a['artifacts'] + b['artifacts'],
        'upstream_reference': REFERENCE.copy(), 'execution_status': 'unknown', 'physical_cause': 'unknown',
        'comparison_rule': 'Exact float64 sampled arithmetic after joint-name alignment; no physical tolerance is inferred.'}
