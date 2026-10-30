"""A short, offline demonstration of a real target check and scoped correction.

The source rows are deliberately generated. This is an onboarding example, not
an observed incident, native trainer run or repair recommendation for user data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .training_guard import TargetContract
from .training_guard_capture import check_capture, save_capture


def _new_output(output):
    if any(ord(character) < 32 or ord(character) == 127 for character in os.fsdecode(output)):
        raise InputError('Demo output path must not contain control characters')
    # Check the user's raw ancestry before normalizing `..`, so an intervening
    # symlink cannot disappear from the refusal policy through lexical cleanup.
    path = Path(output)
    if not path.is_absolute():
        path = Path.cwd() / path
    for item in [*reversed(path.parents), path]:
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError as error:
            if item != path:
                raise InputError('Demo output parent must already exist') from error
            continue
        if stat.S_ISLNK(mode):
            raise InputError('Demo output cannot contain symlink components')
        if item == path:
            raise InputError('Demo requires a new output directory; choose another --output')
        if not stat.S_ISDIR(mode):
            raise InputError('Demo output parent must be a plain directory')
    path = Path(os.path.abspath(path))
    try:
        path.mkdir(mode=0o700)
    except OSError as error:
        raise InputError('Demo requires a writable new output directory') from error
    return path


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _batch(values):
    action = np.asarray(values, dtype=np.float32).reshape(3, 1, 1)
    return {'inputs': {'action': action, 'action_mask': np.ones_like(action)}}


def run_demo(output='./roboclue-demo'):
    """Create a new, synthetic three-row dataset and verify a source-derived fix.

    The intended relative targets and criterion pin are retained before the
    broken candidate is checked. Only the demonstration's separate output copy
    is corrected; neither the original candidate nor its reference is changed.
    """
    path = _new_output(output)
    source = {
        'kind': 'synthetic_target_demo_source', 'schema_version': 1,
        'origin': 'Generated three-row onboarding example; no observed incident.',
        'unit': 'radians', 'rows': [
            {'frame': 0, 'position': 0.5, 'next_position': 0.75},
            {'frame': 1, 'position': 1.0, 'next_position': 1.25},
            {'frame': 2, 'position': 1.5, 'next_position': 1.75},
        ],
    }
    _write(path / 'source.json', source)
    source_hash = hashlib.sha256((path / 'source.json').read_bytes()).hexdigest()
    keys = ['synthetic-demo:episode0:frame' + str(row['frame']) for row in source['rows']]
    intent = {
        'objective': 'Preserve the declared relative-position target: next_position - position.',
        'origin': 'Synthetic onboarding example, not an inferred intended recipe.',
        'units': 'radians', 'source_sha256': source_hash,
        'covered_rows': 3, 'covered_action_dimensions': 1,
        'reviewed_expected_targets': [0.25, 0.25, 0.25],
    }
    reference = TargetContract.from_references(
        [(_batch([0.25, 0.25, 0.25]), keys, [
            {'source_key': key, 'source_sha256': source_hash, 'synthetic': True} for key in keys
        ])], intent=intent,
    )
    contract_path = path / 'reference-contract.json'
    reference.save(contract_path)
    pin = reference.criterion_sha256
    with (path / 'criterion.sha256').open('x', encoding='ascii') as stream:
        stream.write(pin + '\n')
    reference_bytes = contract_path.read_bytes()
    # The broken exporter copied absolute positions into a relative-action field.
    save_capture(_batch([row['next_position'] for row in source['rows']]), keys, path / 'broken.json')
    broken_bytes = (path / 'broken.json').read_bytes()
    checks = {'fault': check_capture(contract_path, pin, path / 'broken.json')}
    # Apply the declared recipe to the source, rather than copying the reference
    # array into the candidate or re-learning a bank after the failure.
    corrected = _batch([row['next_position'] - row['position'] for row in source['rows']])
    save_capture(corrected, keys, path / 'corrected.json')
    checks['corrected'] = check_capture(contract_path, pin, path / 'corrected.json')
    save_capture(_batch([0.25, 0.25, 0.25]), keys, path / 'healthy.json')
    checks['healthy'] = check_capture(contract_path, pin, path / 'healthy.json')
    save_capture(corrected, [key + ':uncovered' for key in keys], path / 'unknown.json')
    checks['unknown'] = check_capture(contract_path, pin, path / 'unknown.json')
    if ({phase: report['status'] for phase, report in checks.items()}
            != {'fault': 'error', 'corrected': 'pass', 'healthy': 'pass', 'unknown': 'unknown'}
            or checks['fault']['differing_target_values'] != 3):
        raise InputError('Demo control did not reproduce its expected result; inspect the new output directory')
    if (contract_path.read_bytes() != reference_bytes
            or (path / 'criterion.sha256').read_text(encoding='ascii').strip() != pin
            or (path / 'broken.json').read_bytes() != broken_bytes
            or hashlib.sha256((path / 'source.json').read_bytes()).hexdigest() != source_hash):
        raise InputError('Demo evidence changed unexpectedly; no verification result can be claimed')
    for phase, report in checks.items():
        _write(path / (phase + '-report.json'), report)
    command = shlex.join([
        'roboclue', 'check', str(path / 'corrected.json'), '--reference', str(contract_path),
        '--pin-file', str(path / 'criterion.sha256'),
    ])
    result = {
        'schema_version': 1, 'kind': 'roboclue_product_demo', 'status': 'completed',
        'covered_check_status': 'pass', 'data_origin': 'synthetic',
        'problem': 'Absolute positions were exported where the declared recipe requires position changes.',
        'correction': 'Compute next_position - position from the unchanged source into a separate copy.',
        'criterion_sha256': pin, 'reference_created_before_candidate_checks': True,
        'references_regenerated_after_fault': False, 'original_candidate_unchanged': True,
        'source_unchanged': True, 'checks': checks,
        'scope': 'Three generated rows, one joint, one-step relative targets; exact target/mask comparison.',
        'unassessed': ['your intended recipe', 'other source rows', 'training', 'model quality', 'robot behavior'],
        'model_forward_calls': 0, 'optimizer_steps': 0,
        'artifacts': {'output': str(path), 'source': str(path / 'source.json'),
                      'original': str(path / 'broken.json'), 'corrected': str(path / 'corrected.json'),
                      'reference': str(contract_path), 'pin': str(path / 'criterion.sha256'),
                      'summary': str(path / 'summary.json')},
        'verify_command': command,
        'next_step': 'Run roboclue guard init to prepare a reviewed reference before your next pipeline change.',
    }
    _write(path / 'summary.json', result)
    return result


def render_demo(report, *, explain=False):
    """Render the result without replacing the underlying guard reports."""
    lines = [
        'PASS: synthetic demo caught 3 wrong action targets; the corrected copy passes the same check.',
        'Cause: absolute positions were used instead of position changes; the fix subtracts the current position.',
        'Next: ' + report['verify_command'],
    ]
    if explain:
        lines.extend([
            '', 'The criterion was pinned before checking the fault and was not regenerated.',
            'The original source and broken candidate remain unchanged.',
            'A source outside the retained reference is UNKNOWN, even with numerically equal targets.',
            'Only generated example targets are checked; training and robot behavior are unassessed.',
            'Recheck the corrected copy:', report['verify_command'],
        ])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='roboclue demo', description='Catch and correct a synthetic target error in seconds, offline.')
    parser.add_argument('--output', default='./roboclue-demo', help='New evidence directory; existing files are never overwritten')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--json', action='store_true', help='Print the full report as JSON')
    mode.add_argument('--explain', action='store_true', help='Add the evidence boundary and exact replay command')
    args = parser.parse_args(argv)
    try:
        report = run_demo(args.output)
    except (InputError, OSError) as error:
        if args.json:
            print(json.dumps({'status': 'error', 'kind': 'demo_input_error', 'message': str(error)}))
        else:
            print(f'Cannot run demo: {error}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else render_demo(report, explain=args.explain))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
