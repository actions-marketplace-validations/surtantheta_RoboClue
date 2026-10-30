"""Reconstruct documented failure mechanisms and healthy controls, without inference."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from policy_doctor.artifacts import InputError
from policy_doctor.diagnostics import audit
from policy_doctor.trace import trace_actions


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def padded(root, width):
    raw = {'observation.state': {'type': 'STATE', 'shape': [7]}, 'action': {'type': 'ACTION', 'shape': [7]}}
    mapping = {'STATE': 'MIN_MAX', 'ACTION': 'MIN_MAX'}
    write(root / 'checkpoint/config.json', {'type': 'evo1', 'input_features': {'observation.state': raw['observation.state']},
          'output_features': {'action': raw['action']}, 'normalization_mapping': mapping, 'chunk_size': 2, 'n_action_steps': 2})
    features = {k: dict(v, shape=[24]) for k, v in raw.items()}
    norm = {'features': features, 'norm_map': mapping}
    write(root / 'checkpoint/policy_preprocessor.json', {'steps': [
        {'registry_name': 'evo1_pad_state_processor', 'config': {'max_state_dim': 24}},
        {'registry_name': 'evo1_pad_action_processor', 'config': {'max_action_dim': 24}},
        {'registry_name': 'normalizer_processor', 'config': norm, 'state_file': 'pre.safetensors'}]})
    write(root / 'checkpoint/policy_postprocessor.json', {'steps': [
        {'registry_name': 'unnormalizer_processor', 'config': dict(norm, features={'action': features['action']}), 'state_file': 'post.safetensors'},
        {'registry_name': 'evo1_action_processor', 'config': {'action_dim': 7}}]})
    stats = {f'{key}.{stat}': np.full(width, value, dtype=np.float32)
             for key in raw for stat, value in [('min', -1.), ('max', 1.)]}
    for side in ('pre', 'post'):
        save_file(stats, str(root / f'checkpoint/{side}.safetensors'))
    report = audit(str(root / 'checkpoint'))
    write(root / 'audit.json', report)
    errors = [f['code'] for f in report['findings'] if f['status'] == 'error']
    if width == 24:
        assert report['status'] == 'unknown' and not errors, errors
        normalized = 2 * (np.pad(np.arange(7.), (0, 17)) + np.ones(24)) / 2 - 1
        assert normalized.shape == (24,) and np.isfinite(normalized).all()
    else:
        assert 'normalization.statistics' in errors and 'processor.feature_shape' not in errors
    return {'statistics_width': width, 'status': report['status'], 'error_codes': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists():
        raise SystemExit('Choose a new work directory to preserve prior evidence')
    args.work.mkdir(parents=True)
    demo = Path(__file__).resolve().parents[1] / 'examples/demo'
    cases = {'valid_padding': padded(args.work / 'valid-padding', 24),
             'broken_padding_statistics': padded(args.work / 'broken-padding', 7)}
    nested = args.work / 'nested-names'
    shutil.copytree(demo, nested)
    source = nested / 'checkpoint/policy_preprocessor.json'
    config = json.loads(source.read_text())
    config['steps'][0]['config']['action_names'] = [['shoulder', 'gripper']]
    write(source, config)
    try:
        audit(str(nested / 'checkpoint'))
    except InputError as exc:
        cases['nested_names'] = {'status': 'refused', 'message': str(exc)}
    else:
        raise AssertionError('Ambiguous nested names must not be flattened silently')
    override = args.work / 'valid-override'
    shutil.copytree(demo, override)
    for side in ('pre', 'post'):
        path = override / f'checkpoint/{side}.safetensors'
        stats = load_file(str(path))
        stats['action.mean'] = np.array([2., .5], dtype=stats['action.mean'].dtype)
        stats['action.std'] = np.array([.5, .25], dtype=stats['action.std'].dtype)
        save_file(stats, str(path))
    sample = json.loads((override / 'sample.json').read_text())
    sample['expected_commands'] = [[12., .5], [12.5, .75], [13., .25]]
    write(override / 'sample.json', sample)
    report = audit(str(override / 'checkpoint'), str(override / 'dataset'), str(override / 'healthy.json'))
    trace = trace_actions(str(override / 'checkpoint'), str(override / 'sample.json'), str(override / 'healthy.json'))
    assert report['status'] == trace['status'] == 'pass'
    cases['valid_paired_override'] = {'status': report['status'], 'trace_status': trace['status'], 'commands': trace['commands']}
    record = {'status': 'pass', 'cases': cases,
              'primary_sources': ['https://github.com/huggingface/lerobot/issues/4006',
                                  'https://github.com/huggingface/lerobot/issues/3283',
                                  'https://huggingface.co/docs/lerobot/main/en/bring_your_own_policies'],
              'scope': 'Mechanistic reconstructions and controls, not complete original incidents or user diagnosis-time evidence.'}
    write(args.output, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
