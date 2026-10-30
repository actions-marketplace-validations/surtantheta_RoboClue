"""Portable captures and offline checks for the native target guard.

Live delivery protection belongs to GuardedCollator. A capture preserves target
bytes and source keys; it does not authenticate how those values were produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .conversion_fidelity import _read_small
from .training_guard import (
    MAX_RECORDS,
    MAX_VALUES,
    TargetContract,
    _batch_arrays,
    _descriptor,
    _encode,
    _keys,
    _restore,
)

MAX_CAPTURE_BYTES = 4 * 1024**2


def _write_new(path, raw):
    try:
        with Path(path).open('xb') as stream:
            stream.write(raw)
    except OSError as error:
        raise InputError(f'Output requires a new writable file: {str(path)!r}') from error


def save_capture(batch, source_keys, path):
    """Save only the actual target/mask and associated original source keys."""
    action, mask = _batch_arrays(batch)
    keys = _keys(source_keys, len(action))
    if len(keys) > MAX_RECORDS:
        raise InputError('Capture exceeds the supported sample count')
    raw = _encode({'schema_version': 1, 'kind': 'native_training_target_capture',
                   'source_keys': list(keys), 'samples': [
                       {'action': _descriptor(a), 'action_mask': _descriptor(m)}
                       for a, m in zip(action, mask, strict=True)]})
    if len(raw) > MAX_CAPTURE_BYTES:
        raise InputError('Capture exceeds 4 MiB; use a smaller declared cohort')
    _write_new(path, raw)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise InputError('Duplicate target capture JSON key')
        result[key] = value
    return result


def load_capture(path):
    raw, _ = _read_small(str(path), limit=MAX_CAPTURE_BYTES)
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))
    except InputError:
        raise
    except (ValueError, RecursionError, UnicodeError) as error:
        raise InputError('Malformed native target capture JSON') from error
    if (type(value) is not dict or set(value) != {'schema_version', 'kind', 'source_keys', 'samples'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['kind'] != 'native_training_target_capture'
            or type(value['samples']) is not list or not 1 <= len(value['samples']) <= MAX_RECORDS):
        raise InputError('Unsupported native target capture schema')
    keys = _keys(value['source_keys'], len(value['samples']))
    actions, masks = [], []
    total = 0
    for record in value['samples']:
        if type(record) is not dict or set(record) != {'action', 'action_mask'}:
            raise InputError('Capture requires action and action_mask for every sample')
        action, mask = _restore(record['action']), _restore(record['action_mask'])
        total += action.size
        if total > MAX_VALUES:
            raise InputError('Capture exceeds the supported target-value budget')
        if (action.shape != mask.shape or (actions and (
                action.shape != actions[0].shape or action.dtype != actions[0].dtype
                or mask.dtype != masks[0].dtype))):
            raise InputError('Capture samples must preserve one batch shape and dtype without coercion')
        actions.append(action)
        masks.append(mask)
    return {'inputs': {'action': np.stack(actions), 'action_mask': np.stack(masks)}}, keys


def check_capture(contract_path, criterion_sha256, capture_path):
    """Check a retained capture against an independently supplied criterion pin."""
    try:
        contract = TargetContract.load(contract_path, expected_sha256=criterion_sha256)
    except InputError as error:
        raise InputError(f'--contract / --criterion-sha256: {error}') from error
    try:
        batch, keys = load_capture(capture_path)
    except InputError as error:
        raise InputError(f'--capture: {error}') from error
    return contract.check(batch, keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Check a native target capture against a fixed criterion.')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check')
    check.add_argument('--contract', required=True, help='Previously frozen target contract JSON')
    check.add_argument('--criterion-sha256', required=True, help='Pin retained independently before the change')
    check.add_argument('--capture', required=True, help='Capture of actual collator targets and source keys')
    check.add_argument('--output', required=True, help='New file for the complete evidence report')
    check.add_argument('--json', action='store_true', help='Emit a compact machine-readable summary')
    args = parser.parse_args(argv)
    try:
        report = check_capture(args.contract, args.criterion_sha256, args.capture)
        try:
            _write_new(args.output, _encode(report))
        except InputError as error:
            raise InputError(f'--output: {error}') from error
        result = {key: report[key] for key in ('status', 'code', 'criterion_sha256', 'samples_checked',
                  'differing_target_values', 'differing_mask_values', 'scope')}
        result['report'] = str(args.output)
        exit_code = {'pass': 0, 'error': 1, 'unknown': 3}[report['status']]
    except (InputError, OSError) as error:
        result = {'status': 'input_error', 'code': 'invalid_guard_input', 'message': str(error)}
        exit_code = 2
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"{result['status'].upper()}: {result['code']}")
        print(result.get('message', 'Full evidence: ' + str(args.output)))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
