"""Private native fault/control qualification using the installed guard package.

The immutable original criterion is supplied by the caller. Separate intentional
objectives get separately declared reference banks; corrections never mint one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class TimedContract:
    def __init__(self, contract):
        self.contract = contract
        self.seconds = []

    def check(self, batch, keys):
        started = time.perf_counter()
        try:
            return self.contract.check(batch, keys)
        finally:
            self.seconds.append(time.perf_counter() - started)


def run_case(binding, checkpoint, out, label, contract=None, count=2):
    import numpy as np
    import run_native as native

    from policy_doctor.training_guard import TargetContractViolation
    from policy_doctor.training_guard_capture import save_capture

    kwargs = read(checkpoint / 'processor_config.json')['processor_kwargs']
    measured = TimedContract(contract) if contract is not None else None
    start = time.perf_counter()
    session = native.create_session(binding, checkpoint, out, label, measured,
        expected_normalization=(kwargs['use_percentiles'], kwargs['use_mean_std']), max_samples=count)
    construction = time.perf_counter() - start
    refs, reports = [], []
    delivered = 0
    try:
        for index in range(count):
            started = time.perf_counter()
            blocked = None
            try:
                batch = session.next()
                delivered += 1
            except TargetContractViolation as error:
                blocked = error.report
                batch = session.tap.last_batch
            latency = time.perf_counter() - started
            records = session.tap.last_records
            keys = [record.source_key for record in records]
            assert batch is session.tap.last_batch, 'Guard must return the actual original collator object'
            action = batch['inputs']['action'].detach().numpy()
            mask = batch['inputs']['action_mask'].detach().numpy()
            assert action.shape == mask.shape == (1, 50, 64) and str(action.dtype) == 'float32'
            assert np.isfinite(action).all() and np.isfinite(mask).all() and np.isin(mask, [0, 1]).all()
            capture_path = out / (label + f'-capture-{index}.json')
            save_capture(batch, keys, capture_path)
            report = blocked if blocked is not None else session.guard.last_report if measured else None
            cli_result = None
            if contract is not None:
                report_path = out / (label + f'-cli-report-{index}.json')
                response = subprocess.run([str(Path(sys.prefix) / 'bin/roboclue-target-guard'), 'check',
                    '--contract', str(contract_paths[contract.criterion_sha256]),
                    '--criterion-sha256', contract.criterion_sha256, '--capture', str(capture_path),
                    '--output', str(report_path), '--json'], capture_output=True, text=True, timeout=15)
                cli_result = json.loads(response.stdout)
                assert response.returncode == {'pass': 0, 'error': 1, 'unknown': 3}[report['status']]
                assert cli_result['status'] == report['status'] and read(report_path) == report
                assert len(response.stdout.encode()) < 2048 and not response.stderr
            refs.append((batch, keys, [record.provenance for record in records]))
            reports.append({'native_collator_returned': True, 'finite': True, 'delivered': blocked is None,
                'batch': native.summary(batch), 'source_keys': keys, 'guard': report,
                'capture': str(capture_path), 'capture_sha256': sha(capture_path),
                'cli': cli_result, 'next_seconds': latency,
                'guard_seconds': measured.seconds[-1] if measured else None})
            if blocked is not None:
                break
    finally:
        try:
            cleanup = session.close()
        finally:
            write(out / (label + '-cleanup.json'), session.cleanup or {'status': 'FAILED_CLEANUP'})
    context = read(out / (label + '-native-context.json'))['processor']
    loaded = {key: context[key] for key in ('use_relative_action', 'use_percentiles', 'use_mean_std',
               'checkpoint_config_sha256', 'checkpoint_statistics_sha256', 'loaded_SAP_statistics_sha256')}
    result = {'label': label, 'construction_seconds': construction, 'delivered': delivered,
              'loaded_processing': loaded,
              'reports': reports, 'cleanup': cleanup}
    write(out / (label + '-result.json'), result)
    return refs, result


def checkpoints(binding, out):
    import run_native as native

    result = native.owned_checkpoints(binding, out)

    def clone(name, source):
        target = out / 'checkpoints' / name
        target.mkdir()
        for filename in ('processor_config.json', 'statistics.json', 'embodiment_id.json'):
            with (target / filename).open('xb') as stream:
                stream.write((source / filename).read_bytes())
        result[name] = target
        return target

    def replace(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

    bounds = clone('active_bounds', result['baseline'])
    stats = read(bounds / 'statistics.json')
    tag = next(iter(stats))
    for key in ('min', 'max'):
        stats[tag]['relative_action']['left_arm'][key][0][0] += 0.125
    replace(bounds / 'statistics.json', stats)
    percentiles = clone('percentiles', result['baseline'])
    config = read(percentiles / 'processor_config.json')
    config['processor_kwargs']['use_percentiles'] = True
    replace(percentiles / 'processor_config.json', config)
    corrected_bounds = clone('corrected_bounds', bounds)
    restored = read(corrected_bounds / 'statistics.json')
    baseline_stats = read(result['baseline'] / 'statistics.json')
    for key in ('min', 'max'):
        restored[tag]['relative_action']['left_arm'][key][0][0] = baseline_stats[tag]['relative_action']['left_arm'][key][0][0]
    assert restored == baseline_stats
    replace(corrected_bounds / 'statistics.json', restored)
    corrected_percentiles = clone('corrected_percentiles', percentiles)
    corrected_config = read(corrected_percentiles / 'processor_config.json')
    corrected_config['processor_kwargs']['use_percentiles'] = False
    assert corrected_config == read(result['baseline'] / 'processor_config.json')
    replace(corrected_percentiles / 'processor_config.json', corrected_config)
    write(out / 'correction-deltas.json', {
        'relative_mode': {'file': 'processor_config.json', 'pointer': '/processor_kwargs/use_relative_action', 'before': False, 'after': True},
        'active_bounds': {'file': 'statistics.json', 'pointers': [f'/{tag}/relative_action/left_arm/{key}/0/0' for key in ('min', 'max')], 'injected_offset': 0.125},
        'percentiles': {'file': 'processor_config.json', 'pointer': '/processor_kwargs/use_percentiles', 'before': True, 'after': False}})
    return result


contract_paths = {}


def execute(args, binding, out):
    import run_native as native

    from policy_doctor.artifacts import InputError
    from policy_doctor.training_guard import TargetContract

    contract = TargetContract.load(args.contract, expected_sha256=args.criterion_sha256)
    contract_paths[args.criterion_sha256] = Path(args.contract)
    if args.phase == 'matrix':
        cp = checkpoints(binding, out)
        write(out / 'pre-fault-criterion.json', {'criterion_sha256': args.criterion_sha256,
            'contract': args.contract, 'source': 'Retained 25 October native pilot, never refreshed',
            'intent': contract.document['intent']})
        # Freeze the two intentional objectives before assessing them. These banks
        # never replace the original criterion for any faulty/corrected pair.
        legal = {}
        for label, key, relative, percentiles in (
                ('absolute', 'relative_false', False, False), ('percentiles', 'percentiles', True, True)):
            intent = dict(native.intent(relative), use_percentiles=percentiles,
                          objective='Explicit intentional ' + label + ' reference recipe')
            write(out / (label + '-declared-intent.json'), intent)
            refs, accepted = run_case(binding, cp[key], out, label + '-reference')
            assert accepted['delivered'] == 2
            alternative = TargetContract.from_references(refs, intent=intent)
            path = out / (label + '-contract.json')
            alternative.save(path)
            contract_paths[alternative.criterion_sha256] = path
            write(out / (label + '-criterion-freeze.json'), {'criterion_sha256': alternative.criterion_sha256,
                'contract': str(path), 'intent': intent})
            legal[label] = {'path': str(path), 'pin': alternative.criterion_sha256, 'checkpoint': str(cp[key])}
        cases = [('relative_mode', cp['relative_false'], cp['corrected']),
                 ('active_bounds', cp['active_bounds'], cp['corrected_bounds']),
                 ('percentiles', cp['percentiles'], cp['corrected_percentiles'])]
        plan = {'original_contract': args.contract, 'criterion_sha256': args.criterion_sha256,
            'cases': [{'label': name, 'fault': str(fault), 'corrected': str(corrected)} for name, fault, corrected in cases],
            'legal': legal, 'baseline': str(cp['baseline']), 'inactive': str(cp['inactive_q01']),
            'files': {str(directory / filename): sha(directory / filename) for directory in cp.values()
                      for filename in ('processor_config.json', 'statistics.json', 'embodiment_id.json')}}
        write(out / 'replay-plan.json', plan)
    else:
        assert args.replay_plan and args.replay_plan_sha256 and sha(args.replay_plan) == args.replay_plan_sha256
        plan = read(args.replay_plan)
        assert plan['criterion_sha256'] == args.criterion_sha256 and plan['original_contract'] == args.contract
        assert all(sha(path) == digest for path, digest in plan['files'].items())
        cases = [(record['label'], Path(record['fault']), Path(record['corrected'])) for record in plan['cases']]
        legal = plan['legal']
    _, healthy = run_case(binding, Path(plan['baseline']), out, 'healthy', contract)
    assert healthy['delivered'] == 2 and all(r['guard']['status'] == 'pass' for r in healthy['reports'])
    results = []
    for label, fault, corrected in cases:
        _, rejected = run_case(binding, fault, out, label + '-fault', contract)
        report = rejected['reports'][0]['guard']
        assert rejected['delivered'] == 0 and report['status'] == 'error' and report['differing_target_values'] > 0
        assert report['criterion_sha256'] == args.criterion_sha256 and report['first_difference']['provenance']['source']
        _, fixed = run_case(binding, corrected, out, label + '-corrected', contract)
        assert fixed['delivered'] == 2 and all(r['guard']['status'] == 'pass' for r in fixed['reports'])
        results.append({'case': label, 'fault_checkpoint': str(fault), 'corrected_checkpoint': str(corrected),
                        'correction_criterion_sha256': args.criterion_sha256, 'fault': rejected, 'corrected': fixed})
    _, inactive = run_case(binding, Path(plan['inactive']), out, 'inactive-statistic', contract)
    assert inactive['delivered'] == 2 and all(r['guard']['status'] == 'pass' for r in inactive['reports'])
    for label, item in legal.items():
        alternative = TargetContract.load(item['path'], expected_sha256=item['pin'])
        contract_paths[item['pin']] = Path(item['path'])
        _, accepted = run_case(binding, Path(item['checkpoint']), out, 'intentional-' + label, alternative)
        assert accepted['delivered'] == 2 and all(r['guard']['status'] == 'pass' for r in accepted['reports'])
    _, uncovered = run_case(binding, Path(plan['baseline']), out, 'coverage-boundary', contract, count=3)
    assert uncovered['delivered'] == 2 and uncovered['reports'][-1]['guard']['status'] == 'unknown'
    assert uncovered['reports'][-1]['guard']['code'] == 'uncovered_source_sample'
    try:
        TargetContract.load(args.contract, expected_sha256='0' * 64)
    except InputError as error:
        drift = {'status': 'input_error', 'message': str(error), 'native_execution_started': False}
    else:
        raise AssertionError('Changed criterion pin was accepted')
    assert sha(args.contract) == args.criterion_sha256
    times = [r['guard_seconds'] for case in results for arm in ('fault', 'corrected')
             for r in case[arm]['reports'] if r['guard_seconds'] is not None]
    return {'status': 'PASS_INSTALLED_NATIVE_GUARD_MATRIX' if args.phase == 'matrix' else 'PASS_FRESH_NATIVE_MATRIX_REPLAY',
        'criterion_sha256': args.criterion_sha256, 'fault_cases': results,
        'valid_intentional_objectives': list(legal), 'inactive_change_passed': True,
        'uncovered_sample': uncovered, 'criterion_drift': drift, 'guard_check_seconds': {
            'observations': len(times), 'median': statistics.median(times), 'maximum': max(times)},
        'scope': 'Declared two-sample native target/mask preservation. Injected controls; no optimizer, weights, full-dataset or physical claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--controls', required=True)
    parser.add_argument('--controls-sha256', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--criterion-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('matrix', 'replay'), required=True)
    parser.add_argument('--replay-plan')
    parser.add_argument('--replay-plan-sha256')
    args = parser.parse_args()
    assert os.environ.get('PBS_JOBID') and sys.flags.isolated and sys.dont_write_bytecode
    assert not os.path.lexists(sys.pycache_prefix)
    assert sha(args.controls) == args.controls_sha256 and sha(args.binding) == args.binding_sha256
    controls = read(args.controls)
    assert all(sha(path) == digest for path, digest in controls.items())
    binding = read(args.binding)
    sys.path[:0] = [binding['native_source'], str(Path(__file__).parent)]
    import run_native as native
    import torch

    import policy_doctor.training_guard as core
    import policy_doctor.training_guard_capture as capture

    assert Path(core.__file__).is_relative_to(Path(sys.prefix)) and Path(capture.__file__).is_relative_to(Path(sys.prefix))
    assert sha(core.__file__) == 'e98d881b5b3318c78fd6b433f7b95a17cc6d3ceaa928bda1ab03c40e9a7b9bc1'
    assert sha(capture.__file__) == sha(Path(__file__).parents[2] / 'src/policy_doctor/training_guard_capture.py')
    native.verify(binding)
    sys.addaudithook(native.audit)
    torch.set_num_threads(1)
    args.out.mkdir(mode=0o700)
    result = {'status': 'FAILED_PRESERVED'}
    try:
        result = execute(args, binding, args.out)
    except BaseException as error:
        result.update(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        native.verify(binding)
        assert all(sha(path) == digest for path, digest in controls.items())
        assert sha(args.controls) == args.controls_sha256 and sha(args.binding) == args.binding_sha256
        assert not os.path.lexists(sys.pycache_prefix)
        result.update(installed_core=core.__file__, installed_capture=capture.__file__,
                      binding_sha256=args.binding_sha256, controls_sha256=args.controls_sha256,
                      source_runtime_input_postguards='pass', job=os.environ['PBS_JOBID'])
        write(args.out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('fault_cases', 'uncovered_sample', 'traceback')}))
    return 0 if result['status'].startswith('PASS_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
