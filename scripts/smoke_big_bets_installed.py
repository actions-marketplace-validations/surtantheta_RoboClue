"""Installed-wheel CLI/MCP smoke using generated software fixtures only.

Run from outside the checkout in the target venv. A real wheel and independent
SHA256 are required; this script never treats a source checkout as installed proof.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

NEW_TOOLS = {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'}
SCOPE = ('Generated small software fixtures and installed CLI/SDK interfaces only. No native trainer, '
         'robot, hardware, public dataset, historical producer or physical benefit is exercised.')
MAX_OUTPUT = 4 * 1024**2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def tree(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob('*')) if path.is_file()}


def installed_origin(wheel, expected_sha256, expected_prefix):
    import policy_doctor

    prefix = Path(expected_prefix).resolve()
    assert Path(sys.prefix).resolve() == prefix, 'Use the explicitly selected installed environment'
    assert digest(wheel) == expected_sha256, 'Wheel SHA256 differs from the independently supplied pin'
    package = Path(policy_doctor.__file__).resolve().parent
    assert package.is_relative_to(prefix), 'Product import is outside the expected installed prefix'
    distribution = importlib.metadata.distribution('roboclue')
    direct = distribution.read_text('direct_url.json')
    if direct:
        assert not json.loads(direct).get('dir_info', {}).get('editable'), 'Editable product installs are not accepted'
    with zipfile.ZipFile(wheel) as archive:
        members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                   if name.startswith('policy_doctor/') and not name.endswith('/')}
    actual = {}
    for path in package.rglob('*'):
        if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_file():
            assert stat.S_ISREG(path.lstat().st_mode), 'Installed package contains a file link/special file'
            actual['policy_doctor/' + path.relative_to(package).as_posix()] = digest(path)
    assert members and actual == members, 'Installed package differs from the supplied wheel member set/bytes'
    checked_records = 0
    for item in distribution.files or []:
        if str(item).startswith('policy_doctor/') and '__pycache__' not in item.parts and item.suffix not in ('.pyc', '.pyo'):
            assert item.hash and item.hash.mode == 'sha256', 'Package RECORD lacks a SHA256'
            path = Path(distribution.locate_file(item)).resolve()
            assert path.is_relative_to(prefix)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
            assert encoded == item.hash.value, 'Installed package RECORD differs'
            checked_records += 1
    assert checked_records == len(members)
    executable = Path(sys.executable).parent / ('roboclue.exe' if os.name == 'nt' else 'roboclue')
    assert executable.is_file(), 'The installed roboclue console entrypoint is missing'
    return {'prefix': str(prefix), 'package': str(package), 'wheel_sha256': expected_sha256,
            'package_files_byte_verified': len(members), 'record_files_verified': checked_records,
            'version': distribution.version, 'roboclue': str(executable)}


def fixtures(root):
    """Use actual public product APIs; labels deliberately assert software origin."""
    import numpy as np

    from policy_doctor.flight_recorder import FlightRecorder
    from policy_doctor.training_guard import TargetContract
    from policy_doctor.training_guard_capture import save_capture

    inputs, outputs, outside = root / 'inputs', root / 'outputs', root / 'outside'
    for path in (inputs, outputs, outside):
        path.mkdir()
    dataset = inputs / 'dataset'
    (dataset / 'meta/episodes/chunk-000').mkdir(parents=True)
    (dataset / 'data/chunk-000').mkdir(parents=True)
    features = {key: {'dtype': 'float32', 'shape': [1], 'names': ['a']}
                for key in ('action', 'observation.state')}
    row_values = np.arange(3, dtype=np.float64)
    write_json(dataset / 'meta/info.json', {'codebase_version': 'v3.0', 'fps': 30,
        'total_frames': 3, 'total_episodes': 1, 'features': features,
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'})
    write_json(dataset / 'meta/stats.json', {key: {'mean': [1.], 'std': [float(np.std(row_values, ddof=0))], 'min': [0.], 'max': [2.]}
                                            for key in features})
    try:
        importlib.metadata.version('pyarrow')
    except importlib.metadata.PackageNotFoundError:
        rows_available = False
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows_available = True
        pq.write_table(pa.Table.from_pylist([{'index': i, 'episode_index': 0, 'frame_index': i,
            'timestamp': i / 30, 'action': [float(value)], 'observation.state': [float(value)]}
            for i, value in enumerate(row_values)]),
            dataset / 'data/chunk-000/file-000.parquet')
        pq.write_table(pa.Table.from_pylist([{'episode_index': 0, 'length': 3, 'dataset_from_index': 0,
            'dataset_to_index': 3, 'data/chunk_index': 0, 'data/file_index': 0}]),
            dataset / 'meta/episodes/chunk-000/file-000.parquet')
    recording = inputs / 'software-commands.jsonl'
    with FlightRecorder(recording, clocks={'host': {'unit': 's', 'domain': 'generated_software_epoch'}},
                        provenance={'kind': 'generated_software_smoke', 'no_robot_feedback': True}) as recorder:
        for index in range(2):
            recorder.command(f'generated-command-{index}', timestamp=index * .1, clock='host', stream='policy',
                             joint_names=['a'], units=['rad'], values=[.5], semantics='position')
    batch = {'inputs': {'action': np.array([[[.5], [.6]]], dtype=np.float32),
                        'action_mask': np.ones((1, 2, 1), dtype=np.float32)}}
    keys = ['generated-software-source:sample0']
    bank = TargetContract.from_references([(batch, keys, [{'kind': 'generated_software_fixture', 'row': 0}])],
        intent={'purpose': 'Generated software smoke; preserve exact target/mask values only',
                'roboclue_model_inputs': {'fields': ['observation.state', 'observation.images.front'],
                                         'declarations': {'units': 'operator-declared radians'}}})
    contract = inputs / 'software-contract.json'
    bank.save(contract)
    captures = {}
    for name in ('pass', 'error', 'unknown'):
        candidate = copy.deepcopy(batch)
        if name == 'error':
            candidate['inputs']['action'][0, 1, 0] += 1
        path = inputs / ('software-target-' + name + '.json')
        save_capture(candidate, ['generated-uncovered-source'] if name == 'unknown' else keys, path)
        captures[name] = path
    config = inputs / 'software-declaration.json'
    write_json(config, {'version': 'generated-unqualified', 'fixture_origin': 'software only'})

    def identity(path):
        return {'path': str(path), 'sha256': digest(path)}

    def snapshot(capture):
        return {'consumer': 'generated-unqualified-software-consumer', 'revision': 'generated-unqualified',
                'source_files': {}, 'configs': {'software_declaration': identity(config)}, 'capture': identity(capture)}

    upgrade = {'schema_version': 1, 'kind': 'native_upgrade_impact', 'contract': identity(contract),
               'assets': [{'asset_id': 'generated-owned-asset', 'reference': snapshot(captures['pass']),
                           'candidate': snapshot(captures['error'])}]}
    manifest = inputs / 'upgrade-unsupported.json'
    write_json(manifest, upgrade)
    bad_pin = copy.deepcopy(upgrade)
    bad_pin['contract']['sha256'] = '0' * 64
    write_json(inputs / 'upgrade-wrong-pin.json', bad_pin)
    shutil.copytree(dataset, outside / 'dataset')
    for path in (recording, contract, captures['pass'], manifest):
        shutil.copyfile(path, outside / path.name)
    nested_outside = copy.deepcopy(upgrade)
    nested_outside['assets'][0]['candidate']['capture'] = identity(outside / captures['pass'].name)
    write_json(inputs / 'upgrade-outside-capture.json', nested_outside)
    write_json(inputs / 'monitor-local.json', {'schema_version': 1, 'datasets': [
        {'id': 'generated-data', 'source': str(dataset), 'consumer': 'generated-software-smoke'}]})
    write_json(inputs / 'monitor-offline.json', {'schema_version': 1, 'datasets': [
        {'id': 'unverified-public-input', 'source': 'hf://generated/no-network-smoke', 'revision': 'a' * 40}]})
    write_json(inputs / 'watch-offline.json', {'schema_version': 1, 'kind': 'owned_upgrade_watch',
        'watches': [{'watch_id': 'declared-upstream', 'repository': 'isaac-sim/IsaacLab', 'kind': 'commit', 'ref': 'main'}],
        'assets': [{'asset_id': 'generated-owned-file', 'consumer': 'isaaclab-hdf5-xyzw', 'watch_id': 'declared-upstream',
                    'files': {'declaration': str(config)},
                    'version': {'path': str(config), 'format': 'json', 'selector': '/version'}}]})
    return {'inputs': inputs, 'outputs': outputs, 'outside': outside, 'dataset': dataset,
            'rows_available': rows_available, 'health_status': 'pass' if rows_available else 'unknown',
            'flight': recording, 'contract': contract, 'pin': bank.criterion_sha256, 'captures': captures,
            'upgrade': manifest}


def imported_closure(prefix):
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == 'policy_doctor' or name.startswith('policy_doctor.'):
            path = getattr(module, '__file__', None)
            assert path and Path(path).resolve().is_relative_to(prefix), name + ' escaped installed prefix'
            result[name] = {'path': str(Path(path).resolve()), 'sha256': digest(path)}
    return result


def execute(args, result):
    checkout = Path(__file__).resolve().parents[1]
    if (checkout / 'src/policy_doctor').is_dir():
        assert not Path.cwd().resolve().is_relative_to(checkout), 'Run smoke from outside the checkout'
    origin = installed_origin(args.wheel, args.wheel_sha256, args.expected_prefix)
    result['installed_origin'] = origin
    work = args.work_dir.absolute() if args.work_dir else Path(tempfile.mkdtemp(prefix='roboclue-installed-smoke-'))
    if (checkout / 'src/policy_doctor').is_dir():
        assert not work.resolve().is_relative_to(checkout), 'Retain fixture work outside the checkout'
    if args.work_dir:
        work.mkdir(parents=True, exist_ok=False)
    result['work_dir'] = str(work)
    data = fixtures(work)
    initial = tree(data['inputs'])
    environment = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME')}
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1',
                       CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    commands = result['commands'] = []

    def cli(*arguments, expected=0, full_json=True):
        number = len(commands)
        command = [origin['roboclue'], *map(str, arguments)]
        if full_json:
            command.append('--json')
        stdout, stderr = work / f'command-{number:02d}.stdout', work / f'command-{number:02d}.stderr'
        with stdout.open('x') as output, stderr.open('x') as error:
            child = subprocess.run(command, cwd=work, env=environment, stdout=output, stderr=error, timeout=90)
        item = {'argv': command, 'exit': child.returncode, 'expected_exit': expected,
                'stdout': str(stdout), 'stderr': str(stderr)}
        commands.append(item)
        assert stdout.stat().st_size <= MAX_OUTPUT and stderr.stat().st_size <= MAX_OUTPUT
        assert child.returncode == expected, (item, stderr.read_text()[-2048:])
        if expected == 2:
            return (stdout.read_text() + stderr.read_text())
        return json.loads(stdout.read_text())

    output = data['outputs']
    expected_health_exit = 0 if data['rows_available'] else 3
    health = cli('health-card', data['dataset'], '--consumer', 'generated-software-smoke',
                 '--output', output / 'health.json', '--artifacts', output / 'health-artifacts',
                 expected=expected_health_exit)
    assert health['status'] == data['health_status']
    assert json.loads((output / 'health-artifacts/health-card.json').read_text()) == health
    assert 'bounded checks' in (output / 'health-artifacts/badge.svg').read_text()
    if data['rows_available']:
        assert health['report']['components']['numeric_rows']['coverage']['rows_scanned'] == 3
        stale_dataset = work / 'dataset-stale-std'
        shutil.copytree(data['dataset'], stale_dataset, ignore=shutil.ignore_patterns('stats.json'))
        stale_stats = json.loads((data['dataset'] / 'meta/stats.json').read_text())
        stale_stats['action']['std'] = [1.]
        write_json(stale_dataset / 'meta/stats.json', stale_stats)
        stale = cli('health-card', stale_dataset, expected=1)
        assert any(finding['code'] == 'dataset.statistics_rows_mismatch'
                   and finding.get('feature') == 'action' and finding.get('metric') == 'std'
                   for finding in stale['report']['findings'])
        result['stale_std_negative_control'] = 'refused against the same three numeric rows'
    partial = cli('health-card', data['dataset'], '--max-rows', 1, expected=3)
    assert partial['status'] == 'unknown'
    flight = cli('flight-triage', data['flight'], '--max-command-gap-s', '.2', expected=3)
    assert flight['status'] == 'unknown' and flight['counts']['pass'] >= 1
    upgrade = cli('upgrade-impact', data['upgrade'], '--artifacts', output / 'upgrade-artifacts', expected=3)
    assert upgrade['status'] == 'unknown' and upgrade['source_input_postguard'] == 'pass'
    assert upgrade['assets'][0]['coverage']['status'] == 'unknown'
    assert (output / 'upgrade-artifacts/alerts.md').is_file()
    assert 'retained' in cli('upgrade-impact', data['inputs'] / 'upgrade-wrong-pin.json', expected=2).lower()
    for status, code in (('pass', 0), ('error', 1), ('unknown', 3)):
        trained = cli('check-training-targets', '--contract', data['contract'], '--criterion-sha256', data['pin'],
                      '--capture', data['captures'][status], expected=code)
        assert trained['status'] == status
        assert trained['additional_model_input_coverage']['status'] == 'unknown'
    invalid = cli('check-training-targets', '--contract', data['contract'], '--criterion-sha256', '0' * 64,
                  '--capture', data['captures']['pass'], expected=2)
    assert any(word in invalid.lower() for word in ('criterion', 'sha256', 'retained'))
    missing = data['inputs'] / 'missing'
    for command in (('health-card', missing), ('flight-triage', missing), ('upgrade-impact', missing),
                    ('check-training-targets', '--contract', data['contract'], '--criterion-sha256', data['pin'],
                     '--capture', missing)):
        cli(*command, expected=2)
    for _ in range(2):
        monitored = cli('monitor', 'scan-once', '--inventory', data['inputs'] / 'monitor-local.json',
                        '--state', output / 'monitor-state', '--timeout', '30', expected=expected_health_exit,
                        full_json=False)
        assert monitored['status'] == data['health_status']
    offline = cli('monitor', 'scan-once', '--inventory', data['inputs'] / 'monitor-offline.json',
                  '--state', output / 'monitor-state', expected=3, full_json=False)
    assert offline['status'] == 'unknown'
    state = json.loads((output / 'monitor-state/state.json').read_text())
    assert len(state['history']) == 3 and state['current']['status'] == 'unknown'
    assert (output / 'monitor-state/index.html').is_file()
    scans = []
    for _ in range(2):
        watched = cli('upgrade-watch', 'scan-once', '--config', data['inputs'] / 'watch-offline.json',
                      '--state-dir', output / 'watch-state', '--offline', expected=3)
        report = json.loads(Path(watched['report']).read_text())
        assert report['offline'] and report['status'] == 'unknown' and not report['previous_results_reused']
        assert all(not observation['previous_observations_reused'] for observation in report['observations'])
        assert Path(watched['alerts']).is_file()
        scans.append(report['scan_id'])
    assert len(set(scans)) == 2
    result['cli'] = {'engine_commands': ['health-card', 'flight-triage', 'upgrade-impact', 'check-training-targets'],
                     'recurrent_commands': ['monitor', 'upgrade-watch'], 'dataset_rows_available': data['rows_available'],
                     'health_status': data['health_status'], 'flight_status': 'unknown (software commands only)',
                     'upgrade_status': 'unknown (generated consumer is intentionally unqualified)',
                     'target_statuses': ['pass', 'error', 'unknown'], 'additional_model_inputs': 'unknown'}
    if args.with_mcp:
        result['mcp'] = asyncio.run(asyncio.wait_for(_mcp(data, work, environment), timeout=90))
    else:
        result['mcp'] = {'status': 'not_run', 'reason': '--without-mcp explicitly selected'}
    assert tree(data['inputs']) == initial, 'CLI/SDK smoke modified generated inputs'
    result['inputs_unchanged'] = True
    result['input_sha256'] = initial
    result['imported_product_closure'] = imported_closure(Path(args.expected_prefix).resolve())
    assert digest(args.wheel) == args.wheel_sha256
    result['status'] = 'PASS_INSTALLED_BIG_BETS_SOFTWARE_SMOKE'


async def _mcp(data, work, environment):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    def body(response):
        assert not response.isError, response.content
        return response.structuredContent or json.loads(response.content[0].text)

    calls = []
    params = StdioServerParameters(command=sys.executable,
        args=['-I', '-B', '-m', 'policy_doctor', 'serve', '--root', str(data['inputs'])], cwd=str(work), env=environment)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert len(tools) == 29 and NEW_TOOLS | {'choose_workflow'} <= set(tools)
            for name in NEW_TOOLS:
                assert tools[name].annotations.readOnlyHint is True
                assert tools[name].annotations.destructiveHint is False
                assert tools[name].inputSchema['additionalProperties'] is False
            requests = {
                'dataset_health_card': {'source': str(data['dataset'])},
                'triage_flight_recording': {'recording': str(data['flight']), 'max_command_gap_s': .2},
                'assess_upgrade_impact': {'manifest': str(data['upgrade'])},
                'check_training_targets': {'contract': str(data['contract']), 'criterion_sha256': data['pin'],
                                           'capture': str(data['captures']['pass'])}}
            expected = {'dataset_health_card': data['health_status'], 'triage_flight_recording': 'unknown',
                        'assess_upgrade_impact': 'unknown', 'check_training_targets': 'pass'}
            for name, arguments in requests.items():
                for format_name in ('concise', 'detailed'):
                    result = body(await client.call_tool(name, {**arguments, 'response_format': format_name}))
                    assert result['status'] == expected[name]
                    if name == 'check_training_targets':
                        assert result['additional_model_input_coverage']['status'] == 'unknown'
                    calls.append({'tool': name, 'format': format_name, 'status': result['status']})
                forbidden = work / ('forbidden-' + name + '.json')
                refused = await client.call_tool(name, {**arguments, 'output': str(forbidden)})
                assert refused.isError and not forbidden.exists()
            outsiders = {
                'dataset_health_card': {'source': str(data['outside'] / 'dataset')},
                'triage_flight_recording': {'recording': str(data['outside'] / data['flight'].name)},
                'assess_upgrade_impact': {'manifest': str(data['outside'] / data['upgrade'].name)},
                'check_training_targets': {'contract': str(data['outside'] / data['contract'].name),
                                           'criterion_sha256': data['pin'], 'capture': str(data['captures']['pass'])}}
            for name, arguments in outsiders.items():
                denied = await client.call_tool(name, arguments)
                assert denied.isError and 'outside' in str(denied.content).lower(), (name, denied)
            nested = await client.call_tool('assess_upgrade_impact', {'manifest': str(data['inputs'] / 'upgrade-outside-capture.json')})
            assert nested.isError and 'outside' in str(nested.content).lower()
            denied = await client.call_tool('check_training_targets', {**requests['check_training_targets'],
                'capture': str(data['outside'] / data['captures']['pass'].name)})
            assert denied.isError and 'outside' in str(denied.content).lower()
            for status in ('error', 'unknown'):
                result = body(await client.call_tool('check_training_targets', {**requests['check_training_targets'],
                    'capture': str(data['captures'][status])}))
                assert result['status'] == status
            refused = await client.call_tool('check_training_targets', {**requests['check_training_targets'], 'criterion_sha256': '0' * 64})
            assert refused.isError and any(word in str(refused.content).lower() for word in ('criterion', 'sha256', 'retained'))
            for name, arguments in requests.items():
                altered = dict(arguments)
                altered[{'dataset_health_card': 'source', 'triage_flight_recording': 'recording',
                         'assess_upgrade_impact': 'manifest', 'check_training_targets': 'capture'}[name]] = str(data['inputs'] / 'missing')
                assert (await client.call_tool(name, altered)).isError
            no_network = await client.call_tool('dataset_health_card', {'source': 'hf://generated/no-network-smoke', 'fetch': True})
            assert no_network.isError and 'operator' in str(no_network.content).lower()
    return {'status': 'pass', 'transport': 'actual MCP SDK stdio', 'default_tool_count': 29,
            'tools': sorted(NEW_TOOLS), 'calls': calls, 'outside_and_nested_denials': True,
            'unknown_and_bad_path_controls': True, 'wrong_criterion_pin_refused': True,
            'additional_model_input_coverage_unknown': True, 'output_arguments_refused': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--expected-prefix', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--work-dir', type=Path, help='new retained directory outside the checkout')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with-mcp', dest='with_mcp', action='store_true', default=True)
    group.add_argument('--without-mcp', dest='with_mcp', action='store_false')
    args = parser.parse_args()
    assert not args.output.exists(), 'Refuse overwriting an existing smoke report'
    result = {'schema_version': 1, 'status': 'FAILED', 'scope': SCOPE, 'final_wheel_required': True}
    try:
        execute(args, result)
    except BaseException as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
        write_json(args.output, result)
        raise
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
