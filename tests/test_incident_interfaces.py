import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_window_interfaces import native_dataset


def cli(*args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    return subprocess.run([sys.executable, '-m', 'policy_doctor', *map(str, args)],
                          env=env, text=True, capture_output=True, timeout=30)


def trajectory(path, *, faulty=False, start=0):
    points = []
    for i in range(2):
        point = {'positions': [float(i)], 'time_from_start': {'sec': start + i, 'nanosec': 0}}
        if faulty:
            point['velocities'] = [4. if i == 0 else 0.]
        points.append(point)
    path.write_text(json.dumps({'joint_names': ['joint'], 'points': points}))


def test_window_cli_creates_and_relocates_shared_case_without_repair_code(tmp_path):
    original, case = tmp_path / 'original', tmp_path / 'case with spaces'
    native_dataset(original)
    hashes = {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in original.rglob('*') if p.is_file()}
    created = cli('create-case', 'episode-windows', original, case, '--apply-metadata-corrections',
                  '--episode-index', '0', '--anchor-index', '1', '--delta-indices', '0', '1', '2')
    assert created.returncode == 0, created.stderr
    assert 'Original finding: error' in created.stdout
    assert 'Corrected check: pass' in created.stdout
    assert 'reproduce-case' in created.stdout
    moved = tmp_path / 'moved'
    case.rename(moved)
    reproduced = cli('reproduce-case', moved, '--json')
    assert reproduced.returncode == 0, reproduced.stderr
    result = json.loads(reproduced.stdout)
    assert result['kind'] == 'incident_case_reproduction'
    assert result['reproduction_matches'] and result['lifecycle_verified']
    assert result['checks']['before']['status'] == 'error'
    assert result['checks']['after']['status'] == 'pass'
    assert hashes == {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.rglob('*') if p.is_file()}


def test_ros_cli_same_case_command_retains_healthy_and_unknown_controls(tmp_path):
    before, after, late = [tmp_path / n for n in ['before.json', 'after.json', 'late.json']]
    trajectory(before, faulty=True)
    trajectory(after)
    trajectory(late, start=1)
    bounds = tmp_path / 'bounds.json'
    bounds.write_text(json.dumps({'joint': [-.1, 1.01]}))
    params = {'sample_times': [.75], 'interpolation_method': 'splines', 'position_limits': {'joint': [-.1, 1.01]}}
    controls = tmp_path / 'controls.json'
    controls.write_text(json.dumps({'controls': [
        {'name': 'healthy', 'source': str(after), 'parameters': params, 'expected_status': 'pass'},
        {'name': 'initial state absent', 'source': str(late), 'parameters': params, 'expected_status': 'unknown'}]}))
    case = tmp_path / 'case'
    made = cli('create-case', 'trajectory-sampling', before, case, '--corrected', after,
               '--times', '.75', '--position-limits', bounds, '--controls', controls, '--markdown')
    assert made.returncode == 0, made.stderr
    assert 'Verified native incident case' in made.stdout
    checked = cli('reproduce-case', case)
    assert checked.returncode == 0, checked.stderr
    assert 'Original finding: error' in checked.stdout and 'Corrected check: pass' in checked.stdout
    assert 'initial state absent' in checked.stdout and 'unknown; expected unknown' in checked.stdout


def test_creation_report_output_option_refused_before_any_write(tmp_path):
    original, case = tmp_path / 'original', tmp_path / 'case'
    native_dataset(original)
    injected_report = original / 'report.json'
    result = cli('create-case', 'episode-windows', original, case,
                 '--apply-metadata-corrections', '--output', injected_report)
    assert result.returncode == 2
    assert not case.exists() and not injected_report.exists()


def test_retained_cli_check_applies_same_parameters_to_later_native_artifacts(tmp_path):
    before, after, late = [tmp_path / n for n in ['before.json', 'after.json', 'late.json']]
    trajectory(before, faulty=True)
    trajectory(after)
    trajectory(late, start=1)
    bounds = tmp_path / 'bounds.json'
    bounds.write_text(json.dumps({'joint': [-.1, 1.01]}))
    case = tmp_path / 'retained'
    made = cli('create-case', 'trajectory-sampling', before, case, '--corrected', after,
               '--times', '.75', '--position-limits', bounds, '--json')
    assert made.returncode == 0, made.stderr
    initial_files = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in tmp_path.rglob('*') if p.is_file()}
    for path, status, exit_code in [(after, 'pass', 0), (before, 'error', 1), (late, 'unknown', 3)]:
        checked = cli('check-case', case, path, '--json')
        assert checked.returncode == exit_code, checked.stderr
        result = json.loads(checked.stdout)
        assert result['case_verified'] and result['candidate_evaluated']
        assert result['status'] == status == result['candidate']['status']
        assert result['parameters']['sample_times'] == [.75]
    markdown = cli('check-case', case, before, '--markdown')
    assert markdown.returncode == 1 and 'Retained incident regression' in markdown.stdout
    assert initial_files == {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in tmp_path.rglob('*') if p.is_file()}


def test_cli_reuse_drift_cannot_certify_or_inspect_a_candidate(tmp_path):
    before, after = tmp_path / 'before.json', tmp_path / 'after.json'
    trajectory(before, faulty=True)
    trajectory(after)
    limits = tmp_path / 'bounds.json'
    limits.write_text(json.dumps({'joint': [-.1, 1.01]}))
    case = tmp_path / 'case'
    made = cli('create-case', 'trajectory-sampling', before, case, '--corrected', after,
               '--times', '.75', '--position-limits', limits, '--json')
    assert made.returncode == 0, made.stderr
    manifest = json.loads((case / 'manifest.json').read_text())
    manifest['verifier']['engine_version'] = 'a-different-verifier'
    (case / 'manifest.json').write_text(json.dumps(manifest))
    digest = hashlib.sha256((case / 'manifest.json').read_bytes()).hexdigest()
    (case / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': digest}))
    checked = cli('check-case', case, tmp_path / 'does-not-exist.json', '--json')
    assert checked.returncode == 3, checked.stderr
    result = json.loads(checked.stdout)
    assert not result['case_verified'] and not result['candidate_evaluated']
    assert result['candidate'] is None


def test_case_cli_dispatch_rejects_escaping_manifest(tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text('{"kind":"robot_incident_case"}')
    case = tmp_path / 'case'
    case.mkdir()
    (case / 'manifest.json').symlink_to(outside)
    checked = cli('reproduce-case', case, '--json')
    assert checked.returncode == 2
    assert 'escapes' in checked.stderr


def test_mcp_case_creation_is_opt_in_scoped_and_strict(tmp_path):
    inputs, outputs, outside = tmp_path / 'inputs', tmp_path / 'outputs', tmp_path / 'outside'
    inputs.mkdir()
    outputs.mkdir()
    native_dataset(inputs / 'dataset')
    native_dataset(outside)
    alias = outputs / 'dangling'
    alias.symlink_to(outputs / 'not-created', target_is_directory=True)
    (outputs / 'escape').symlink_to(outside, target_is_directory=True)
    params = {'episode_index': 0, 'anchor_index': 1, 'delta_indices': [0, 1, 2]}

    async def run():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        for enabled in [False, True]:
            args = ['-m', 'policy_doctor', 'serve', '--root', str(inputs), '--root', str(outputs)]
            if enabled:
                args.extend(['--case-root', str(outputs)])
            server = StdioServerParameters(command=sys.executable, args=args, env=env)
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = {t.name: t for t in (await session.list_tools()).tools}
                    assert len(listed) == (31 if enabled else 29)
                    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(listed)
                    assert listed['reproduce_incident_case'].annotations.readOnlyHint
                    assert ('create_incident_case' in listed) == enabled
                    if not enabled:
                        assert all(t.annotations.readOnlyHint for t in listed.values())
                        continue
                    assert not listed['create_incident_case'].annotations.readOnlyHint
                    request = {'check': 'episode_windows', 'source': str(inputs / 'dataset'),
                               'destination': str(outputs / 'valid'), 'parameters': params,
                               'apply_metadata_corrections': True}
                    refused = [
                        {'destination': str(outside / 'case')}, {'destination': str(outputs)},
                        {'destination': str(alias)}, {'destination': str(outputs / 'escape/case')},
                        {'source': str(outside)}, {'corrected': str(outside)},
                        {'controls': [{'name': 'outside', 'source': str(outside), 'parameters': {}, 'expected_status': 'pass'}]},
                        {'apply_metadata_corrections': 1}, {'parameters': dict(params, anchor_index=True)},
                        {'destination': str(inputs / 'dataset/new-case')},
                    ]
                    for changed in refused:
                        result = await session.call_tool('create_incident_case', dict(request, **changed))
                        assert result.isError, changed
                    made = await session.call_tool('create_incident_case', request)
                    assert not made.isError, made.content
                    assert made.structuredContent['checks']['after']['status'] == 'pass'
                    again = await session.call_tool('create_incident_case', request)
                    assert again.isError
                    checked = await session.call_tool('reproduce_incident_case', {'directory': str(outputs / 'valid')})
                    assert not checked.isError and checked.structuredContent['lifecycle_verified']
                    for candidate, expected in [(inputs / 'dataset', 'error'), (outputs / 'valid/after', 'pass')]:
                        result = await session.call_tool('check_incident_candidate', {
                            'directory': str(outputs / 'valid'), 'candidate': str(candidate)})
                        assert not result.isError, result.content
                        assert result.structuredContent['status'] == expected
                        assert result.structuredContent['case_verified']
                    result = await session.call_tool('check_incident_candidate', {
                        'directory': str(outputs / 'valid'), 'candidate': str(outside)})
                    assert result.isError
                    denied = await session.call_tool('reproduce_incident_case', {'directory': str(outside)})
                    assert denied.isError
        server = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(outputs), '--root', str(outside / 'meta')], env=env)
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool('check_incident_candidate', {
                    'directory': str(outputs / 'valid'), 'candidate': str(outside / 'meta/info.json')})
                assert result.isError
    asyncio.run(run())
    assert alias.is_symlink() and not (outputs / 'not-created').exists()
    assert not (outside / 'case').exists()
