"""Repeatable installed-wheel CLI/MCP check, intended to run outside the checkout."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import policy_doctor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--without-mcp', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    package = Path(policy_doctor.__file__).resolve()
    assert not package.is_relative_to(project / 'src'), 'Run with an installed wheel, not an editable install'
    environment = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    checks = []
    with tempfile.TemporaryDirectory(prefix='policy-doctor-smoke-') as folder:
        work = Path(folder)
        demo = work / 'demo'
        shutil.copytree(project / 'examples/demo', demo)

        def cli(*arguments, expected=0):
            result = subprocess.run([sys.executable, '-m', 'policy_doctor', *map(str, arguments), '--json'],
                cwd=work, env=environment, capture_output=True, text=True, timeout=30)
            assert result.returncode == expected, (arguments, result.returncode, result.stderr)
            return json.loads(result.stdout)

        for name, expected in [('healthy', 0), ('broken', 1)]:
            report = cli('audit', demo / 'checkpoint', '--dataset', demo / 'dataset',
                         '--deployment', demo / f'{name}.json', expected=expected)
            assert report['status'] == ('pass' if name == 'healthy' else 'error')
        checks.append('CLI healthy/broken controls')
        source = next(f for f in report['findings'] if f['code'] == 'semantics.units')['evidence'][1]
        assert cli('evidence', source['path'], '--pointer', source['pointer'], '--sha256', source['sha256'])['value'] == ['deg', 'normalized']
        assert cli('trace', demo / 'checkpoint', demo / 'sample.json', '--deployment', demo / 'healthy.json')['status'] == 'pass'
        repair = cli('check-repair', demo / 'checkpoint', demo / 'broken.json', demo / 'healthy.json', '--dataset', demo / 'dataset')
        assert repair['status'] == 'pass' and not repair['introduced_error_counts']
        cli('bundle', demo / 'checkpoint', work / 'bundle', '--dataset', demo / 'dataset', '--deployment', demo / 'broken.json')
        assert cli('reproduce', work / 'bundle')['status'] == 'pass'
        checks.extend(['JSON evidence', 'saved decoding', 'candidate repair', 'portable bundle reproduction'])

        from types import SimpleNamespace

        from safetensors.numpy import load_file

        from policy_doctor.capture import capture_runtime

        class Loaded:
            def __init__(self, side):
                self.config = json.loads((demo / 'checkpoint' / f'policy_{side}processor.json').read_text())
                self.states = {s['state_file'][:-12]: load_file(str(demo / 'checkpoint' / s['state_file']))
                               for s in self.config['steps'] if s.get('state_file')}
                self.steps = [SimpleNamespace() for _ in self.config['steps']]
            def get_config(self):
                return self.config
            def state_dict(self):
                return self.states
        pre, post = Loaded('pre'), Loaded('post')
        post.steps[-1].relative_step = pre.steps[0]
        captured = capture_runtime(work / 'snapshot',
            policy_config=json.loads((demo / 'checkpoint/config.json').read_text()),
            preprocessor=pre, postprocessor=post, deployment=json.loads((demo / 'healthy.json').read_text()))
        assert captured['status'] == 'pass'
        assert cli('audit-capture', work / 'snapshot')['capture']['integrity'] == 'verified'
        checks.append('installed capture API, topology, commit marker and CLI verification')
        assert cli('investigate', work / 'snapshot')['status'] == 'pass'
        assert cli('compare-captures', work / 'snapshot', work / 'snapshot')['equal']
        import h5py
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        recordings = [work / 'reference.parquet', work / 'candidate.parquet']
        for path, actions in zip(recordings, [[[1., 2.], [3., 4.]], [[1., 2.], [4., 4.]]], strict=True):
            pq.write_table(pa.table({'episode_index': [0, 0], 'frame_index': [0, 1], 'action': actions}), path)
        compared = cli('compare-rollouts', *recordings, '--alignment-key', 'frame_index', expected=1)
        assert compared['first_observed_difference']['row'] == 1
        native_h5 = work / 'native.hdf5'
        with h5py.File(native_h5, 'w') as file:
            file['data/demo_0/actions'] = np.ones((3, 2))
            file['data/demo_0/step'] = np.arange(3)
        assert cli('compare-rollouts', native_h5, native_h5, '--alignment-key', 'step')['status'] == 'pass'
        checks.extend(['one-folder investigation', 'verified runtime comparison', 'native LeRobot Parquet and robomimic HDF5 comparison'])
        from policy_doctor.dispatch import make_dispatch_event

        boundary = work / 'dispatch.json'
        def boundary_for(recording):
            boundary.write_text(json.dumps({'schema_version': 1, 'kind': 'action_dispatch_evidence',
                'recording_sha256': hashlib.sha256(recording.read_bytes()).hexdigest(), 'episode_index': 0,
                'events': [make_dispatch_event(str(i), frame_index=i, driver_return={'j0': a, 'j1': b},
                    recorded_role='driver_return') for i, (a, b) in enumerate([(1., 2.), (3., 4.)])]}))
        boundary_for(recordings[0])
        assert cli('audit-dispatch', boundary, '--recording', recordings[0], '--episode', '0', '--action-names', 'j0', 'j1')['status'] == 'pass'
        boundary_for(recordings[1])
        assert cli('audit-dispatch', boundary, '--recording', recordings[1], '--episode', '0', '--action-names', 'j0', 'j1', expected=1)['first_recording_contradiction']['frame_index'] == 1
        checks.append('native action-boundary association, healthy control and declared-role contradiction')

        packaged_demo = cli('demo', work / 'packaged-demo')
        assert packaged_demo['candidate_position_at_750ms'] == [1.03125]
        checks.append('packaged demo without source example scripts')

        trajectories = [work / 'linear.json', work / 'cubic.json']
        for path, velocities in zip(trajectories, [None, [4., 0.]], strict=True):
            points = []
            for index, position in enumerate([0., 1.]):
                point = {'positions': [position], 'time_from_start': {'sec': index, 'nanosec': 0}}
                if velocities is not None:
                    point['velocities'] = [velocities[index]]
                points.append(point)
            path.write_text(json.dumps({'joint_names': ['joint'], 'points': points}))
        sampled = cli('sample-trajectory', trajectories[1], '--times', '0.75')
        assert sampled['samples'][0]['positions'] == [1.03125]
        compared = cli('compare-trajectories', *trajectories, '--times', '0.75', expected=1)
        assert compared['same_waypoint_positions_and_times'] and not compared['sampled_references_equal']
        checks.append('native ROS trajectory CLI sampling and derivative-only reference difference')
        portable_case = work / 'native-case'
        cli('export-trajectory-case', *trajectories, portable_case, '--control', trajectories[0], '--times', '.75')
        reproduced = cli('reproduce-case', portable_case)
        assert reproduced['reproduction_matches'] and reproduced['checks']['candidate']['status'] == 'different'
        assert reproduced['checks']['control']['status'] == 'pass'
        checks.append('portable native trajectory case with separate reproduction/comparison verdicts')
        shared_case = work / 'shared-incident'
        packed = work / 'packaged-demo'
        incident = cli('create-case', 'trajectory-sampling', packed / 'candidate.json', shared_case,
            '--corrected', packed / 'corrected.json', '--times', '.75',
            '--position-limits', packed / 'position_limits.json', '--controls', packed / 'incident_controls.json')
        assert incident['checks']['before']['status'] == 'error' and incident['checks']['after']['status'] == 'pass'
        assert cli('reproduce-case', shared_case)['lifecycle_verified']
        assert cli('check-case', shared_case, packed / 'candidate.json', expected=1)['case_verified']
        checks.append('shared native incident lifecycle, healthy/unknown controls and retained recurrence check')

        if not args.without_mcp:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            async def exercise():
                for root, reject in [(demo / 'dataset/meta', True), (work, False)]:
                    params = StdioServerParameters(command=sys.executable,
                        args=['-m', 'policy_doctor', 'serve', '--root', str(root)], cwd=str(work), env=environment)
                    async with stdio_client(params) as (read, write):
                        async with ClientSession(read, write) as client:
                            await client.initialize()
                            names = {tool.name for tool in (await client.list_tools()).tools}
                            assert len(names) == 29 and 'compare_runtime_captures' in names
                            assert {'dataset_health_card', 'triage_flight_recording',
                                    'assess_upgrade_impact', 'check_training_targets'} <= names
                            result = await client.call_tool('inspect_dataset', {'dataset': str(demo / 'dataset/meta/info.json')})
                            assert result.isError == reject
                            if reject:
                                assert 'outside' in result.content[0].text
                            else:
                                assert result.structuredContent['kind'] == 'dataset'
                                result = await client.call_tool('audit_deployment', {
                                    'checkpoint': str(demo / 'checkpoint'), 'dataset': str(demo / 'dataset'),
                                    'deployment': str(demo / 'healthy.json')})
                                assert not result.isError and result.structuredContent['status'] == 'pass'
                                result = await client.call_tool('audit_runtime_capture', {'capture': str(work / 'snapshot')})
                                assert not result.isError and result.structuredContent['capture']['integrity'] == 'verified'
                                for name, arguments, status in [
                                    ('investigate_run_directory', {'directory': str(work / 'snapshot')}, 'pass'),
                                    ('compare_runtime_captures', {'reference': str(work / 'snapshot'), 'candidate': str(work / 'snapshot')}, 'pass'),
                                    ('compare_rollout_recordings', {'reference': str(recordings[0]), 'candidate': str(recordings[1]), 'alignment_key': 'frame_index'}, 'different'),
                                    ('sample_ros_trajectory', {'trajectory': str(trajectories[1]), 'sample_times': [0.75]}, 'pass'),
                                    ('compare_ros_trajectories', {'reference': str(trajectories[0]), 'candidate': str(trajectories[1]), 'sample_times': [0.75]}, 'different'),
                                    ('reproduce_trajectory_case', {'directory': str(portable_case)}, 'pass'),
                                    ('reproduce_incident_case', {'directory': str(shared_case)}, 'pass'),
                                    ('check_incident_candidate', {'directory': str(shared_case), 'candidate': str(packed / 'candidate.json')}, 'error'),
                                    ('audit_action_boundary', {'evidence': str(boundary), 'recording': str(recordings[1]), 'episode': 0, 'action_names': ['j0', 'j1']}, 'error')]:
                                    result = await client.call_tool(name, arguments)
                                    assert not result.isError and result.structuredContent['status'] == status
            asyncio.run(asyncio.wait_for(exercise(), timeout=30))
            checks.append('real MCP discovery, allowed access, read-root rejection and healthy audit')
    versions = {}
    for name in ('numpy', 'safetensors', 'mcp', 'pyarrow', 'h5py'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.parent.glob('*.py'))}
    record = {'status': 'pass', 'version': policy_doctor.__version__, 'python': sys.version.split()[0],
              'installed_package': str(package), 'versions': versions, 'checks': checks, 'installed_source_hashes': sources}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
