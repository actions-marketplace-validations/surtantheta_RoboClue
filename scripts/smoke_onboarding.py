"""Installed core-only first result and controls; run outside the checkout.

No MCP, dataset readers, ROS, source examples or policy weights are required.
This is an automated correctness/onboarding gate, not a user-effort benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import policy_doctor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    package = Path(policy_doctor.__file__).resolve()
    assert not package.is_relative_to(project / 'src'), 'Use a non-editable installed package'
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    calls = []
    with tempfile.TemporaryDirectory(prefix='robot-onboarding-') as folder:
        work = Path(folder)
        destination = work / 'incident'

        def cli(*arguments, expected=0):
            command = [sys.executable, '-I', '-m', 'policy_doctor', *map(str, arguments), '--json']
            started = time.monotonic()
            result = subprocess.run(command, cwd=work, env=environment, capture_output=True, text=True, timeout=30)
            calls.append({'arguments': list(map(str, arguments)), 'exit': result.returncode,
                          'seconds': time.monotonic() - started})
            assert result.returncode == expected, (arguments, result.stdout, result.stderr)
            return json.loads(result.stdout if expected != 2 else result.stderr)

        summary = cli('demo', destination)
        assert summary['candidate_position_at_750ms'] == [1.03125]
        assert summary['execution_status'] == summary['physical_cause'] == 'unknown'
        first_result_seconds = calls[0]['seconds']
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
        assert cli('sample-trajectory', destination / 'candidate.json', '--times', '.75')['samples'][0]['positions'] == [1.03125]
        compare = cli('compare-trajectories', destination / 'reference.json', destination / 'candidate.json', '--times', '.75', expected=1)
        assert compare['same_waypoint_positions_and_times'] and not compare['sampled_references_equal']
        assert cli('compare-trajectories', destination / 'reference.json', destination / 'corrected.json', '--times', '.75')['status'] == 'pass'
        assert cli('compare-trajectories', destination / 'ordered.json', destination / 'reordered.json', '--times', '.75')['status'] == 'pass'
        assert cli('sample-trajectory', destination / 'missing_initial_state.json', '--times', '0', '1.5', expected=3)['status'] == 'unknown'
        assert cli('demo', destination, expected=2)['error']['code'] == 'invalid_input'
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()} == before
        assert '1.03125' in (destination / 'README.md').read_text()
        moved = work / 'moved incident'
        shutil.move(destination, moved)
        assert cli('compare-trajectories', moved / 'reference.json', moved / 'candidate.json', '--times', '.75', expected=1)['status'] == 'different'
        exported = cli('export-trajectory-case', moved / 'reference.json', moved / 'candidate.json', work / 'portable',
                       '--control', moved / 'corrected.json', '--times', '.75')
        assert exported['checks'] == {'candidate': 'different', 'control': 'pass'}
        shared = work / 'shared incident'
        incident = cli('create-case', 'trajectory-sampling', moved / 'candidate.json', shared,
            '--corrected', moved / 'corrected.json', '--times', '.75',
            '--position-limits', moved / 'position_limits.json', '--controls', moved / 'incident_controls.json')
        assert incident['checks']['before']['status'] == 'error' and incident['checks']['after']['status'] == 'pass'
        assert [c['check']['status'] for c in incident['checks']['controls']] == ['pass', 'unknown']
        retained = work / 'retained incident'
        shared.rename(retained)
        assert cli('reproduce-case', retained)['lifecycle_verified']
        for name, expected, status in [('corrected', 0, 'pass'), ('candidate', 1, 'error'), ('missing_initial_state', 3, 'unknown')]:
            checked = cli('check-case', retained, moved / f'{name}.json', expected=expected)
            assert checked['case_verified'] and checked['candidate_evaluated'] and checked['status'] == status
        shutil.rmtree(moved)
        assert cli('reproduce-case', retained)['reproduction_matches']
        portable = work / 'colleague case'
        (work / 'portable').rename(portable)
        reproduced = cli('reproduce-case', portable)
        assert reproduced['reproduction_matches'] and reproduced['verifier_matches']
        assert reproduced['checks']['candidate']['status'] == 'different'
        with (portable / 'candidate.json').open('ab') as stream:
            stream.write(b' ')
        assert cli('reproduce-case', portable, expected=2)['error']['code'] == 'invalid_input'
        from policy_doctor.dispatch import make_dispatch_event

        trace = work / 'dispatch.json'
        trace.write_text(json.dumps({'schema_version': 1, 'kind': 'action_dispatch_evidence',
            'recording_sha256': None, 'episode_index': None,
            'events': [make_dispatch_event('tick', driver_input={'joint.pos': 2.}, driver_return={'joint.pos': .5})]}))
        assert cli('audit-dispatch', trace, expected=3)['driver_change_events'] == 1
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'safetensors')}
    optional_versions = {}
    for name in ('mcp', 'pyarrow', 'h5py', 'torch', 'transformers'):
        try:
            optional_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional_versions[name] = None
    record = {'status': 'pass', 'version': policy_doctor.__version__, 'python': sys.version.split()[0],
              'installed_package': str(package), 'versions': versions, 'optional_versions': optional_versions,
              'first_result_command_seconds': first_result_seconds,
              'scope': 'Automated installed-package correctness; excludes installation time, human diagnosis effort and community adoption.',
              'cli_calls': calls,
              'installed_source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.parent.glob('*.py'))}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
