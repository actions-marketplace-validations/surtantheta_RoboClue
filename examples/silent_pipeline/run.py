#!/usr/bin/env python3
"""Synthetic selected-numeric pipeline demo using the installed CLI and MCP server.

No robot recordings, models, weights, training or full native processor execution.
All displayed verdicts and source witnesses come from actual RoboClue responses.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
PROFILE = 'unitree_g1_full_body_with_waist_height_nav_cmd'
GROOT_VERSION = 'n1.7'
STATES = {'left_leg': 6, 'right_leg': 6, 'waist': 3, 'left_arm': 7, 'right_arm': 7,
          'left_hand': 7, 'right_hand': 7}
ACTIONS = {'left_arm': 7, 'right_arm': 7, 'left_hand': 7, 'right_hand': 7, 'waist': 3,
           'base_height_command': 1, 'navigate_command': 3}
CASES = ('healthy', 'shifted_labels', 'inactive_statistic', 'incomplete_processor')
SELECTORS = [{'episode_index': 72, 'anchor_index': 0}]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    reference = root / 'reference'
    dataset = reference / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'data/chunk-000').mkdir(parents=True)
    declarations = {
        'video': {'delta_indices': [0], 'modality_keys': ['ego_view']},
        'language': {'delta_indices': [0], 'modality_keys': ['annotation.human.task_description']},
        'state': {'delta_indices': [0], 'modality_keys': list(STATES)},
        'action': {'delta_indices': list(range(50)), 'modality_keys': list(ACTIONS), 'action_configs': [
            {'rep': 'RELATIVE' if key in ('left_arm', 'right_arm') else 'ABSOLUTE',
             'type': 'NON_EEF', 'format': 'DEFAULT', 'state_key': None} for key in ACTIONS]},
    }
    config = {'processor_class': 'Gr00tN1d7Processor', 'processor_kwargs': {
        'modality_configs': {PROFILE: declarations}, 'max_state_dim': 64, 'max_action_dim': 64,
        'max_action_horizon': 50, 'use_percentiles': False, 'use_relative_action': False,
        'clip_outliers': False, 'apply_sincos_state_encoding': False, 'use_mean_std': False,
        'exclude_state': False, 'state_dropout_prob': 0.0}}
    stats = {PROFILE: {kind: {key: {'min': [-2.0] * width, 'max': [2.0] * width,
        'q01': [-1.0] * width, 'q99': [1.0] * width, 'mean': [0.5] * width, 'std': [0.25] * width}
        for key, width in widths.items()} for kind, widths in [('state', STATES), ('action', ACTIONS)]}}
    stats[PROFILE]['relative_action'] = {key: {name: [copy.deepcopy(vector) for _ in range(50)]
        for name, vector in stats[PROFILE]['action'][key].items()} for key in ('left_arm', 'right_arm')}
    for group in stats[PROFILE]['relative_action'].values():
        group['min'] = [[-3.0] * 7 for _ in range(50)]
        group['max'] = [[3.0] * 7 for _ in range(50)]
    checkpoint = reference / 'checkpoint'
    write(checkpoint / 'processor_config.json', config)
    write(checkpoint / 'statistics.json', stats)
    info = {'codebase_version': 'v2.1', 'chunks_size': 1000, 'fps': 50,
        'total_frames': 60, 'total_episodes': 1, 'total_tasks': 1,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'features': {key: {'dtype': 'float32', 'shape': [width]} for key, width in [
            ('observation.state', 43), ('action', 43), ('teleop.base_height_command', 1), ('teleop.navigate_command', 3)]}}
    slices = {'left_leg': (0, 6), 'right_leg': (6, 12), 'waist': (12, 15), 'left_arm': (15, 22),
              'left_hand': (22, 29), 'right_arm': (29, 36), 'right_hand': (36, 43)}
    modality = {kind: {key: {'start': start, 'end': end} for key, (start, end) in slices.items()}
                for kind in ('state', 'action')}
    modality['action'].update({
        'base_height_command': {'start': 0, 'end': 1, 'original_key': 'teleop.base_height_command'},
        'navigate_command': {'start': 0, 'end': 3, 'original_key': 'teleop.navigate_command'}})
    write(dataset / 'meta/info.json', info)
    write(dataset / 'meta/modality.json', modality)
    (dataset / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 72, 'length': 60}) + '\n')
    (dataset / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'synthetic numeric pipeline control'}) + '\n')

    def table(shift):
        rows = {'episode_index': pa.array([72] * 60, type=pa.int64()),
            'frame_index': pa.array(range(60), type=pa.int64()), 'index': pa.array(range(60), type=pa.int64()),
            'task_index': pa.array([0] * 60, type=pa.int64()),
            'timestamp': pa.array([frame / 50 for frame in range(60)], type=pa.float64())}
        vectors = {'observation.state': [[0.01 * frame + 0.001 * col for col in range(43)] for frame in range(60)],
            'action': [[0.02 * (frame + shift) + 0.001 * col for col in range(43)] for frame in range(60)],
            'teleop.base_height_command': [[0.005 * frame + 0.125] for frame in range(60)],
            'teleop.navigate_command': [[0.005 * frame + offset for offset in (0.25, 0.5, 0.75)] for frame in range(60)]}
        rows.update({name: pa.array(value, type=pa.list_(pa.float32())) for name, value in vectors.items()})
        return pa.table(rows)

    parquet = Path('data/chunk-000/episode_000072.parquet')
    pq.write_table(table(0), dataset / parquet)
    for name in CASES:
        own = root / name
        shutil.copytree(reference, own)
        if name == 'shifted_labels':
            # Finite labels are generated one frame ahead; states and auxiliary commands stay unchanged.
            pq.write_table(table(1), own / 'dataset' / parquet)
        elif name == 'inactive_statistic':
            changed = copy.deepcopy(stats)
            changed[PROFILE]['state']['left_arm']['q01'][0] = -1.5
            (own / 'checkpoint/statistics.json').write_text(json.dumps(changed, indent=2, sort_keys=True) + '\n')
        elif name == 'incomplete_processor':
            (own / 'checkpoint/statistics.json').unlink()
    write(root / 'PROVENANCE.json', {'kind': 'entirely_synthetic_analytic_numeric_data',
        'recordings_or_private_data': False, 'downloaded_payloads': False,
        'dataset_format': 'LeRobot v2.1 ordinary numeric folders',
        'state_formula': '0.01 * frame + 0.001 * coordinate',
        'reference_action_formula': '0.02 * frame + 0.001 * coordinate',
        'shifted_action_formula': '0.02 * (frame + 1) + 0.001 * coordinate',
        'data_storage': 'finite float32 vectors; 60 rows; stored episode ID 72',
        'selection': SELECTORS, 'selected_profile': PROFILE, 'groot_version': GROOT_VERSION,
        'selected_scope': '43 state coordinates, 35 selected action coordinates, 50-step action window',
        'not_claimed': ['original recording fidelity', 'physical joint truth', 'full native processor parity',
                        'final model targets', 'training success', 'robot behavior']})


def request(root, name):
    return {'source': str(root / name / 'dataset'), 'checkpoint': str(root / name / 'checkpoint'),
        'consumer_reference_dataset': str(root / 'reference/dataset'),
        'consumer_reference_checkpoint': str(root / 'reference/checkpoint'),
        'consumer_samples': SELECTORS, 'consumer_require_equal': True, 'groot_version': GROOT_VERSION}


def cli_arguments(arguments):
    command = ['preflight', arguments['source'], '--checkpoint', arguments['checkpoint'],
        '--consumer-reference-dataset', arguments['consumer_reference_dataset'],
        '--consumer-reference-checkpoint', arguments['consumer_reference_checkpoint'],
        '--groot-version', arguments['groot_version']]
    for selected in arguments['consumer_samples']:
        command.extend(['--consumer-sample', f"{selected['episode_index']}:{selected['anchor_index']}"])
    return [*command, '--consumer-require-equal', '--json']


def payload(result):
    if isinstance(result.structuredContent, dict):
        return result.structuredContent
    return json.loads(next(item.text for item in result.content if item.type == 'text'))


def validate_controls(cli, mcp_results):
    """Demo acceptance only; keep these expectations separate from product responses."""
    expected = {'healthy': ('EQUAL', 'pass'), 'shifted_labels': ('CHANGED', 'error'),
                'inactive_statistic': ('EQUAL', 'pass'), 'incomplete_processor': ('UNKNOWN', 'unknown')}
    checks = []
    for name, (effect, status) in expected.items():
        actual = {'cli_effect': cli[name].get('numeric_projection_value_effect'),
                  'mcp_effect': mcp_results[name].get('numeric_projection_value_effect'),
                  'cli_component_status': cli[name].get('component_status'),
                  'mcp_component_status': mcp_results[name].get('component_status'),
                  'sdk_is_error': mcp_results[name].get('sdk_is_error')}
        checks.append({'control': name, 'expected': {'selected_effect_both_channels': effect, 'component_status_both_channels': status},
            'actual': actual, 'demonstrated': actual['cli_effect'] == actual['mcp_effect'] == effect
                and actual['cli_component_status'] == actual['mcp_component_status'] == status and actual['sdk_is_error'] is False})
    inactive = mcp_results['inactive_statistic']
    checks.append({'control': 'inactive_statistic_identity_is_not_active_effect',
        'expected': {'artifact_identity_changed': True, 'active_interpretation_changed': False},
        'actual': {key: inactive.get(key) for key in ('artifact_identity_changed', 'active_interpretation_changed')},
        'demonstrated': inactive.get('artifact_identity_changed') is True and inactive.get('active_interpretation_changed') is False})
    witness = mcp_results['shifted_labels'].get('first_value_divergence') or {}
    checks.append({'control': 'shifted_labels_has_actual_source_witness',
        'expected': {'stage': 'raw', 'field': 'action.left_arm', 'source_column': 'action'},
        'actual': witness,
        'demonstrated': witness.get('stage') == 'raw' and witness.get('field') == 'action.left_arm'
            and all(witness.get('source_mapping', {}).get(side, {}).get('column') == 'action' for side in ('reference', 'candidate'))})
    return {'kind': 'demo_control_validation_not_product_verdict', 'all_controls_demonstrated': all(item['demonstrated'] for item in checks), 'checks': checks}


async def mcp_calls(root, env):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    results = {}
    with tempfile.TemporaryDirectory(prefix='roboclue-synthetic-mcp-') as cwd, (root / 'reports/mcp.stderr.txt').open('x') as errors:
        server = StdioServerParameters(command=sys.executable,
            args=['-I', '-B', '-X', f'pycache_prefix={root}/absent-pycache-mcp', '-m', 'policy_doctor',
                  'serve', '--root', str(root)], env=env, cwd=cwd)
        async with stdio_client(server, errlog=errors) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=datetime.timedelta(seconds=90)) as client:
                initialize = await client.initialize()
                write(root / 'reports/mcp.initialize.json', initialize.model_dump(mode='json'))
                write(root / 'reports/mcp.tools.json', (await client.list_tools()).model_dump(mode='json'))
                for name in CASES:
                    arguments = request(root, name)
                    write(root / 'reports' / f'{name}.mcp.request.json', {'name': 'preflight_check', 'arguments': arguments})
                    start = time.monotonic()
                    result = await client.call_tool('preflight_check', arguments)
                    write(root / 'reports' / f'{name}.mcp.response.json', result.model_dump(mode='json'))
                    if result.isError:
                        results[name] = {'sdk_is_error': True, 'machine_seconds': time.monotonic() - start}
                        continue
                    body = payload(result)
                    write(root / 'reports' / f'{name}.mcp.payload.json', body)
                    component = body.get('components', {}).get('consumer_comparison', {})
                    results[name] = {'sdk_is_error': False, 'machine_seconds': time.monotonic() - start,
                        'overall_status': body.get('status'), 'component_status': component.get('status'),
                        'numeric_projection_value_effect': component.get('numeric_projection_value_effect'),
                        'artifact_identity_changed': component.get('artifact_identity_changed'),
                        'active_interpretation_changed': component.get('active_interpretation_changed'),
                        'first_value_divergence': component.get('first_value_divergence'),
                        'first_final_value_difference': component.get('first_final_value_difference'),
                        'first_final_value_source': component.get('first_final_value_source'),
                        'first_error': component.get('first_error'), 'first_unresolved': component.get('first_unresolved'),
                        'full_native_processor_and_model_target': component.get('full_native_processor_and_model_target')}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused.')
    parser.add_argument('--prepare-only', action='store_true', help='Generate the four folders and commands without invoking RoboClue.')
    args = parser.parse_args()
    expanded = args.output.expanduser()
    if os.path.lexists(expanded):
        parser.error('Output already exists; choose a new path to preserve earlier results.')
    root = expanded.resolve()
    try:
        import pyarrow  # noqa: F401
        version = importlib.metadata.version('policy-deployment-doctor')
        if not args.prepare_only:
            import mcp  # noqa: F401
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        parser.error('Install this project with [dataset,mcp] extras in the active Python environment: ' + str(error))
    root.mkdir(parents=True)
    fixture(root)
    inputs = [p for directory in [root / 'reference', *(root / name for name in CASES)] for p in directory.rglob('*') if p.is_file()]
    input_hashes = {str(p.relative_to(root)): {'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in inputs}
    write(root / 'input_manifest.json', input_hashes)
    for name in CASES:
        arguments = request(root, name)
        (root / name / 'commands.sh').write_text('#!/bin/sh\n' + shlex.join(['policy-doctor', *cli_arguments(arguments)]) + '\n')
        write(root / name / 'mcp_request.json', {'name': 'preflight_check', 'arguments': arguments})
    (root / 'MCP_SETUP.txt').write_text('Use the same installed environment. Start:\n' +
        shlex.join(['policy-doctor', 'serve', '--root', str(root)]) +
        '\nCall preflight_check with one of the four case mcp_request.json argument objects.\nLocal read-only inputs only; no Hub or native-runtime permission is needed.\n')
    if args.prepare_only:
        print(json.dumps({'generated_directory': str(root), 'product_calls': 'not run', 'provenance': 'synthetic'}, indent=2))
        return 0
    (root / 'reports').mkdir()
    env = dict(os.environ, PYTHONPATH='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    cli = {}
    with tempfile.TemporaryDirectory(prefix='roboclue-synthetic-cli-') as cwd:
        for name in CASES:
            command = [sys.executable, '-I', '-B', '-X', f'pycache_prefix={root}/absent-pycache-cli-{name}',
                       '-m', 'policy_doctor', *cli_arguments(request(root, name))]
            start = time.monotonic()
            result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=90)
            (root / 'reports' / f'{name}.cli.stdout.txt').write_text(result.stdout)
            (root / 'reports' / f'{name}.cli.stderr.txt').write_text(result.stderr)
            write(root / 'reports' / f'{name}.cli.invocation.json', {'argv': command, 'returncode': result.returncode,
                'machine_seconds': time.monotonic() - start, 'cwd': cwd})
            body = json.loads(result.stdout)
            write(root / 'reports' / f'{name}.cli.json', body)
            component = body.get('components', {}).get('consumer_comparison', {})
            cli[name] = {'returncode': result.returncode, 'overall_status': body.get('status'),
                        'component_status': component.get('status'),
                        'numeric_projection_value_effect': component.get('numeric_projection_value_effect')}
    mcp_results = asyncio.run(asyncio.wait_for(mcp_calls(root, env), timeout=240))
    unchanged = all(digest(root / name) == identity['sha256'] for name, identity in input_hashes.items())
    report = {'kind': 'actual_installed_synthetic_numeric_demo', 'product_version': version,
        'provenance': 'Entirely synthetic analytic arrays; see PROVENANCE.json',
        'criterion': 'Explicit equality of this selected numeric window; CHANGED alone is observational.',
        'input_files_unchanged': unchanged, 'cli': cli, 'mcp': mcp_results,
        'network_scope': 'Only local inputs and stdio; no network instrumentation or network-traffic claim.',
        'scope': 'Selected supported saved-processor arithmetic only; original recording fidelity, full native/model target, training and robot behavior remain unproved.',
        'reports_directory': str(root / 'reports')}
    write(root / 'results.json', report)
    for name in CASES:
        value = mcp_results[name]
        print(f"{name}: selected values {value.get('numeric_projection_value_effect')}; comparison {value.get('component_status')}; overall {value.get('overall_status')}; CLI exit {cli[name]['returncode']}")
        witness = value.get('first_value_divergence')
        if witness:
            print('  actual source witness: ' + json.dumps(witness, sort_keys=True))
    print('Actual reports: ' + str(root / 'results.json'))
    print('Synthetic selected numeric demonstration; full native/model/training outcome remains unproved.')
    validation = validate_controls(cli, mcp_results)
    write(root / 'demo_validation.json', validation)
    print('Demo controls demonstrated: ' + str(validation['all_controls_demonstrated']) + '; see ' + str(root / 'demo_validation.json'))
    return 0 if unchanged and validation['all_controls_demonstrated'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
