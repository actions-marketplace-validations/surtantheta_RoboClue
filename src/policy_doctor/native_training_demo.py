"""CPU replay of retained public GR00T targets; never executes a native model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .training_guard import TargetContract

MANIFEST_SHA256 = '0e5302b2f4c46f1edcd9c67bd4f58d0f2ca082fae19bbb029fb76ec76281c97d'
ORIGINAL_CRITERION_SHA256 = '758d5c621e178a3d5cfbda6995a02b1fbd35b00cf1e695d41815929e52b5f898'
_FIXTURE_NAMES = frozenset({'reference.json', 'healthy.json', 'fault.json', 'corrected.json',
                            'unknown.json', 'provenance.json', 'NOTICE.txt', 'LICENSE-CC-BY-4.0.txt'})
_SCOPE = ('Saved public native target/mask capture replay only. No model forward, optimizer step, '
          'new native capture, GPU run, physical correctness or inferred intended recipe.')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fixtures():
    return resources.files('policy_doctor').joinpath('data', 'native_training_demo')


def _verified_fixtures():
    """Authenticate every bundled input before creating a new runtime criterion."""
    directory = _fixtures()
    try:
        manifest_raw = directory.joinpath('manifest.json').read_bytes()
        if _sha(manifest_raw) != MANIFEST_SHA256:
            raise InputError('Saved native demo manifest differs from its retained source pin')
        manifest = json.loads(manifest_raw)
        if (manifest['kind'] != 'saved_native_demo_integrity_manifest'
                or manifest['schema_version'] != 1 or set(manifest['files']) != _FIXTURE_NAMES):
            raise InputError('Unsupported saved native demo integrity manifest')
        result = {}
        for name, entry in manifest['files'].items():
            raw = directory.joinpath(name).read_bytes()
            if len(raw) != entry['bytes'] or _sha(raw) != entry['sha256']:
                raise InputError(f'Saved native demo fixture integrity check failed: {name}')
            result[name] = raw
        result['manifest.json'] = manifest_raw
        return result
    except InputError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise InputError('Saved native demo resources are missing or malformed; reinstall the package') from error


def _capture(raw, *, reference=False):
    value = json.loads(raw)
    expected_kind = 'saved_native_reference_projection' if reference else 'native_training_target_capture'
    if (value.get('kind') != expected_kind or value.get('schema_version') != 1
            or len(value.get('source_keys', [])) != 2 or len(value.get('samples', [])) != 2
            or len(set(value['source_keys'])) != 2):
        raise InputError('Unsupported saved native target capture')
    arrays = {field: [] for field in ('action', 'action_mask')}
    for sample in value['samples']:
        for field in arrays:
            desc = sample[field]
            if desc['dtype'] not in ('<f4', '<f8') or desc['shape'] != [50, 64]:
                raise InputError('Saved native capture has an unsupported target schema')
            array = np.asarray(desc['data'], dtype=desc['dtype'])
            if (list(array.shape) != desc['shape'] or not np.isfinite(array).all()
                    or _sha(array.tobytes(order='C')) != desc['sha256']):
                raise InputError('Saved native capture array differs from its retained hash')
            arrays[field].append(array)
    batch = {'inputs': {field: np.stack(values) for field, values in arrays.items()}}
    return value, batch


def _output_path(output):
    path = Path(os.path.abspath(os.fspath(output)))
    # lstat detects dangling symlinks and special files without opening them.
    for item in [*reversed(path.parents), path]:
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError as error:
            if item != path:
                raise InputError('Saved native demo output parent must already exist') from error
            continue
        if stat.S_ISLNK(mode):
            raise InputError('Saved native demo output cannot contain symlink components')
        if item == path:
            raise InputError('Saved native demo requires a new output directory; no files are overwritten')
        if not stat.S_ISDIR(mode):
            raise InputError('Saved native demo output parent must be a plain directory')
    directory = _fixtures()
    # Source and installed wheels both expose a concrete resource directory.
    if isinstance(directory, Path) and (path.is_relative_to(directory.resolve())
                                       or directory.resolve().is_relative_to(path)):
        raise InputError('Saved native demo output overlaps its packaged input fixtures')
    return path


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run_demo(output):
    """Replay healthy, fault, correction and uncovered controls into a NEW directory.

    A distinct criterion is bound to this installed verifier and NumPy runtime.
    Only the frozen declared intended reference supplies its targets. The original
    native criterion is immutable and is never loaded with relaxed runtime pins.
    """
    path = _output_path(output)
    fixture_bytes = _verified_fixtures()
    reference, reference_batch = _capture(fixture_bytes['reference.json'], reference=True)
    provenance = json.loads(fixture_bytes['provenance.json'])
    if provenance['original_criterion_sha256'] != ORIGINAL_CRITERION_SHA256:
        raise InputError('Saved native source provenance differs from the original criterion pin')
    contract = TargetContract.from_references(
        [(reference_batch, reference['source_keys'], reference['provenance'])],
        intent={
            'objective': 'Replay the frozen declared native startup target/mask reference',
            'original_declared_intent': reference['intent'],
            'reference_projection_sha256': _sha(fixture_bytes['reference.json']),
            'original_native_criterion_sha256': ORIGINAL_CRITERION_SHA256,
            'scope': _SCOPE,
        },
    )
    try:
        path.mkdir(mode=0o700)  # atomic refusal of an existing destination
    except OSError as error:
        raise InputError('Saved native demo requires a writable new output directory') from error
    # Pin is independently retained before any candidate is decoded or checked.
    contract_path = path / 'reference-contract.json'
    pin_path = path / 'criterion.sha256'
    contract.save(contract_path)
    pin = contract.criterion_sha256
    with pin_path.open('x', encoding='ascii') as stream:
        stream.write(pin + '\n')
    contract = TargetContract.load(contract_path, expected_sha256=pin)
    inputs = path / 'inputs'
    inputs.mkdir()
    for name, raw in fixture_bytes.items():
        with (inputs / name).open('xb') as stream:
            stream.write(raw)
    checks, report_paths = {}, {}
    expected = {'healthy': 'pass', 'fault': 'error', 'corrected': 'pass', 'unknown': 'unknown'}
    for phase, expected_status in expected.items():
        capture, batch = _capture(fixture_bytes[phase + '.json'])
        report = contract.check(batch, capture['source_keys'])
        # These are actual guard reports; expected controls do not alter verdicts.
        checks[phase] = report
        report_paths[phase] = str(path / (phase + '-report.json'))
        _write_json(Path(report_paths[phase]), report)
        if report['status'] != expected_status:
            raise InputError(f'Saved native replay control failed: {phase}; inspect {report_paths[phase]}')
        if (phase == 'fault' and (report['differing_target_values'] != 1400
                or sum(f['field'] == 'inputs.action_mask' and f['code'] == 'target_schema_mismatch'
                       for f in report['findings']) != 2)):
            raise InputError('Saved native fault no longer reproduces the retained target/mask findings')
        # Revalidate using the retained pin; never replace the reference after fault.
        contract = TargetContract.load(contract_path, expected_sha256=pin)
        if pin_path.read_text(encoding='ascii').strip() != pin:
            raise InputError('Saved native demo independently retained criterion pin changed')
    summary = {
        'schema_version': 1, 'kind': 'saved_native_training_demo', 'status': 'completed',
        'overall_status': 'unknown', 'scope': _SCOPE,
        'message': ('Saved native capture replay completed: healthy and corrected targets pass; '
                    'fault has 1400 changed targets and two mask dtype mismatches; '
                    'uncovered sources stay UNKNOWN. Wider research correctness remains unassessed.'),
        'criterion_sha256': pin, 'original_criterion_sha256': ORIGINAL_CRITERION_SHA256,
        'criterion_relationship': ('New replay criterion for this installed verifier/NumPy runtime, '
                                   'distinct from the original native qualification criterion.'),
        'reference_created_before_candidate_checks': True,
        'references_regenerated_after_fault': False,
        'fixture_manifest_sha256': MANIFEST_SHA256,
        'numpy': np.__version__, 'model_forward_calls': 0, 'optimizer_steps': 0,
        'checks': checks, 'source_provenance': provenance,
        'unassessed': ['intended recipe for your run', 'other source samples', 'model learning',
                       'new native framework execution', 'hardware behavior', 'physical correctness'],
        'artifacts': {'output': str(path), 'reference_contract': str(contract_path),
                      'criterion_pin': str(pin_path), 'inputs': str(inputs),
                      'reports': report_paths, 'summary': str(path / 'summary.json')},
        'next_step': ('Use your own explicitly declared intended recipe to capture a reference before '
                      'changing preprocessing; keep its pin independently. This saved replay cannot '
                      'infer an intended recipe or reconstruct missing history.'),
    }
    _write_json(path / 'summary.json', summary)
    return summary


def _print_summary(report):
    checks = report['checks']
    healthy, fault = checks['healthy'], checks['fault']
    corrected, unknown = checks['corrected'], checks['unknown']
    mask_dtype_findings = sum(
        finding.get('field') == 'inputs.action_mask'
        and finding.get('code') == 'target_schema_mismatch'
        and finding.get('expected_dtype') != finding.get('observed_dtype')
        for finding in fault['findings']
    )
    same_pin = corrected['criterion_sha256'] == report['criterion_sha256']
    print('Saved native capture replay completed; expected controls reproduced (exit 0).')
    print(f"Healthy {healthy['status'].upper()}: {healthy['samples_checked']} covered samples.")
    print(f"Fault {fault['status'].upper()}: {fault['differing_target_values']:,} changed target values; "
          f'{mask_dtype_findings} mask dtype findings.')
    print(f"Corrected {corrected['status'].upper()}: {corrected['samples_checked']} covered samples; "
          + ('same retained pin.' if same_pin else 'different criterion pin.'))
    print(f"Uncovered {unknown['status'].upper()}: {len(unknown['uncovered_source_keys'])} source keys "
          f"outside the reference; {unknown['samples_checked']} samples paired.")
    print(f"Overall research {report['overall_status'].upper()}: saved target/mask comparison only; "
          'no new native/model execution, training or physical correctness claim.')
    print(f"Criterion (new replay runtime): {report['criterion_sha256']}")
    print(f"Reference contract: {report['artifacts']['reference_contract']}")
    print(f"Retained pin file: {report['artifacts']['criterion_pin']}")
    print(f"Full summary and provenance: {report['artifacts']['summary']}")
    print(f"Saved target inputs: {report['artifacts']['inputs']}")
    for phase, path in report['artifacts']['reports'].items():
        print(f'{phase.capitalize()} check report: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory under an existing plain parent')
    parser.add_argument('--json', action='store_true', help='Print the full saved summary as JSON')
    args = parser.parse_args(argv)
    try:
        report = run_demo(args.output)
    except InputError as error:
        print(f'Saved native demo refused: {error}', file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_summary(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
