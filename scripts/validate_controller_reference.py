#!/usr/bin/env python3
"""Compile a pinned upstream interpolation body with explicit stand-ins; no ROS execution.

Offline usage:
  python scripts/validate_controller_reference.py --source /path/to/trajectory.cpp --output result.json
Optional --cases JSON is a list of {t0_ns,t1_ns,sample_ns,a:{positions,...},b:{positions,...}}.
The source must exactly match the documented upstream pin. Compiler execution is a developer
validation action, separate from the read-only MCP and ordinary artifact intake.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

COMMIT = '8ce934536c6404750fb8aafc0b62b7d365049143'
SOURCE_SHA256 = 'a979c1bca2527a912571f5e1ed29e5887d7bf92aa226f6918a125a462821537d'
FIELDS = ('positions', 'velocities', 'accelerations', 'effort')
PREFIX = r'''
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>
namespace rclcpp {
struct Duration {
  std::int64_t ns;
  double seconds() const { return static_cast<double>(ns) / 1e9; }
  static Duration from_seconds(double s) { return {static_cast<std::int64_t>(std::llround(s * 1e9))}; }
};
struct Time { std::int64_t ns; };
Duration operator-(Time a, Time b) { return {a.ns - b.ns}; }
}
namespace trajectory_msgs::msg {
struct JointTrajectoryPoint { std::vector<double> positions, velocities, accelerations, effort; };
}
namespace joint_trajectory_controller {
class Trajectory { public: static void interpolate_between_points(
 const rclcpp::Time&, const trajectory_msgs::msg::JointTrajectoryPoint&,
 const rclcpp::Time&, const trajectory_msgs::msg::JointTrajectoryPoint&,
 const rclcpp::Time&, trajectory_msgs::msg::JointTrajectoryPoint&); };
'''
SUFFIX = r'''
}
void read_vector(std::vector<double>& v) {
  std::size_t n; std::cin >> n; v.resize(n); for (double& x : v) std::cin >> x;
}
void read_point(trajectory_msgs::msg::JointTrajectoryPoint& p) {
  read_vector(p.positions); read_vector(p.velocities); read_vector(p.accelerations); read_vector(p.effort);
}
void write_vector(const std::vector<double>& v) {
  std::cout << v.size(); for (double x : v) std::cout << " " << x; std::cout << "\n";
}
int main() {
  std::int64_t t0, t1, t;
  while (std::cin >> t0 >> t1 >> t) {
    trajectory_msgs::msg::JointTrajectoryPoint a, b, out;
    read_point(a); read_point(b); if (!std::cin) return 2;
    joint_trajectory_controller::Trajectory::interpolate_between_points({t0}, a, {t1}, b, {t}, out);
    std::cout << std::setprecision(17);
    write_vector(out.positions); write_vector(out.velocities); write_vector(out.accelerations); write_vector(out.effort);
  }
}
'''


def default_cases():
    cases = []
    for mode, time_ns, expected in [('linear', 250000000, .25), ('cubic', 250000000, .15625),
                                    ('overshoot', 750000000, 1.03125), ('quintic', 250000000, .103515625)]:
        a, b = {'positions': [0.]}, {'positions': [1.]}
        if mode != 'linear':
            a['velocities'], b['velocities'] = ([4.] if mode == 'overshoot' else [0.]), [0.]
        if mode == 'quintic':
            a['accelerations'], b['accelerations'] = [0.], [0.]
        cases.append({'name': mode, 't0_ns': 0, 't1_ns': 1000000000, 'sample_ns': time_ns,
                      'a': a, 'b': b, 'expected_position': [expected]})
    rng = random.Random(20261018)
    for mode in range(3):
        for index in range(32):
            dim = 1 if index % 2 else 3
            # Large absolute epochs exercise integer subtraction before double seconds.
            t0 = 1700000000000000000 + index * 10000000000
            duration = rng.choice([50000000, 123456789, 1000000000, 5000000000])
            fraction = rng.choice([-.1, 0., .25, .5, .9, 1., 1.1])
            points = []
            for _ in range(2):
                p = {'positions': [rng.uniform(-2, 2) for _ in range(dim)],
                     'effort': [rng.uniform(-5, 5) for _ in range(dim)]}
                if mode >= 1:
                    p['velocities'] = [rng.uniform(-3, 3) for _ in range(dim)]
                if mode == 2:
                    p['accelerations'] = [rng.uniform(-4, 4) for _ in range(dim)]
                points.append(p)
            cases.append({'name': f'random-{mode}-{index}', 't0_ns': t0, 't1_ns': t0 + duration,
                          'sample_ns': t0 + round(duration * fraction), 'a': points[0], 'b': points[1]})
    return cases


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 5000:
        raise ValueError('Provide between 1 and 5000 cases')
    for case in cases:
        times = [case[k] for k in ('t0_ns', 't1_ns', 'sample_ns')]
        if any(type(t) is not int or abs(t) >= 2**62 for t in times) or times[1] <= times[0]:
            raise ValueError('Cases need bounded integer nanoseconds and strictly positive segment duration')
        dim = len(case['a']['positions'])
        if not 1 <= dim <= 64:
            raise ValueError('Reference cases need 1–64 position coordinates')
        for point in (case['a'], case['b']):
            for field in FIELDS:
                values = point.get(field, [])
                if not isinstance(values, list) or len(values) not in (0, dim):
                    raise ValueError('Point fields must be empty or match position width')
                if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
                    raise ValueError('Reference values must be finite numbers')
            if len(point['positions']) != dim:
                raise ValueError('Both endpoints require complete positions')


def check_implementation(results):
    from policy_doctor import trajectory

    module = Path(trajectory.__file__)
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    checked, skipped, failures, maximum_error = 0, 0, [], 0.
    with tempfile.TemporaryDirectory(prefix='policy-doctor-controller-cases-') as directory:
        path = Path(directory) / 'trajectory.json'
        for result in results:
            case = result['case']
            relative = case['sample_ns'] - case['t0_ns']
            duration = case['t1_ns'] - case['t0_ns']
            if not 0 <= relative < duration or duration > 3600 * 10**9:
                skipped += 1  # Full sampler terminal/initial behavior differs from this body.
                continue
            dim = len(case['a']['positions'])
            points = []
            for endpoint, ns in [('a', 0), ('b', duration)]:
                points.append(dict(case[endpoint], time_from_start={'sec': ns // 10**9, 'nanosec': ns % 10**9}))
            epoch = case['t0_ns']
            data = {'header': {'stamp': {'sec': epoch // 10**9, 'nanosec': epoch % 10**9}, 'frame_id': ''},
                    'joint_names': [f'joint_{index}' for index in range(dim)], 'points': points}
            path.write_text(json.dumps(data))
            sampled = trajectory.sample_joint_trajectory(str(path), [relative / 10**9])['samples'][0]
            checked += 1
            for field in FIELDS:
                expected, actual = result['sampled'][field], sampled[field]
                for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                    maximum_error = max(maximum_error, abs(left - right))
                    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10):
                        failures.append({'case': case.get('name'), 'field': field, 'coordinate': index,
                                         'upstream_body': left, 'implementation': right})
    if hashlib.sha256(module.read_bytes()).hexdigest() != digest:
        raise ValueError('Implementation source changed during differential gate')
    return {'checked_cases': checked, 'skipped_initial_or_terminal_cases': skipped,
            'rule': 'math.isclose rel_tol=1e-10, abs_tol=1e-10 on every supported sampled field',
            'maximum_absolute_error': maximum_error, 'failures': failures,
            'module_sha256': digest, 'pass': not failures,
            'scope': 'Segment arithmetic only; header epoch preserved but not used for live-clock association.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--check-implementation', action='store_true',
                        help='Compare interior segment samples against installed policy_doctor API')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        parser.error(f'Upstream source hash differs from pinned {COMMIT}; refusing compilation')
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error('A local C++17 compiler is required for this optional developer gate')
    text = raw.decode('utf-8')
    body = text[text.index('void Trajectory::interpolate_between_points('):
                text.index('void Trajectory::deduce_from_derivatives(')]
    harness = PREFIX + body + SUFFIX
    cases = json.loads(args.cases.read_text()) if args.cases else default_cases()
    validate_cases(cases)
    lines = []
    for case in cases:
        lines.append(' '.join(str(case[k]) for k in ('t0_ns', 't1_ns', 'sample_ns')))
        for point in (case['a'], case['b']):
            for field in FIELDS:
                values = point.get(field, [])
                lines.append(' '.join([str(len(values)), *(repr(v) for v in values)]))
    with tempfile.TemporaryDirectory(prefix='policy-doctor-controller-reference-') as directory:
        cpp, binary = Path(directory) / 'reference.cpp', Path(directory) / 'reference'
        cpp.write_text(harness)
        subprocess.run([compiler, '-std=c++17', '-O2', str(cpp), '-o', str(binary)],
                       check=True, capture_output=True, text=True, timeout=60)
        completed = subprocess.run([str(binary)], input='\n'.join(lines) + '\n',
                                   check=True, capture_output=True, text=True, timeout=30)
    rows = iter(completed.stdout.splitlines())
    results = []
    for case in cases:
        sampled = {}
        for field in FIELDS:
            tokens = next(rows).split()
            values = [float(v) for v in tokens[1:]]
            if int(tokens[0]) != len(values) or not all(math.isfinite(v) for v in values):
                raise ValueError('Invalid native reference output')
            sampled[field] = values
        if 'expected_position' in case and sampled['positions'] != case['expected_position']:
            raise ValueError(f"Analytic control failed: {case.get('name')}")
        results.append({'case': case, 'sampled': sampled})
    if next(rows, None) is not None:
        raise ValueError('Unexpected trailing reference output')
    result = {'schema_version': 1, 'repository': 'ros-controls/ros2_controllers', 'commit': COMMIT,
              'source_sha256': digest, 'body_sha256': hashlib.sha256(body.encode()).hexdigest(),
              'harness_sha256': hashlib.sha256(harness.encode()).hexdigest(),
              'compiler': compiler, 'case_count': len(results), 'analytic_controls': 4 if not args.cases else 'supplied',
              'scope': 'Unmodified upstream interpolation function with integer-nanosecond Time/Duration and numeric message stand-ins. No ROS, controller, hardware, trajectory lifecycle or execution validation.',
              'results': results}
    if args.check_implementation:
        result['implementation_comparison'] = check_implementation(results)
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    if args.check_implementation and not result['implementation_comparison']['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
