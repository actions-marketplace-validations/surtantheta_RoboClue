"""Run the delivered native fault/correction/control matrix in fresh processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new directory outside this packet')
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(packet):
        parser.error('--output must name a new directory outside the packet')
    verify = runpy.run_path(str(packet / 'install.py'))['verify_packet']
    manifest = verify(packet)
    output.mkdir(mode=0o700)
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'GR00T_DIT_SDPA_MODE': 'math',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}
    results, commands, pin = {}, [], None
    for mode, code, outcome in [('reference', 0, 'reference_frozen_before_fault'),
                               ('native-fault', 0, 'trained'), ('guard-fault', 1, 'target_blocked'),
                               ('corrected', 0, 'trained'), ('healthy', 0, 'trained'),
                               ('replay', 0, 'trained'), ('unknown', 3, 'target_blocked')]:
        target = output / mode
        command = [sys.executable, '-I', '-B', str(packet / 'train.py'), '--mode', mode, '--output', str(target)]
        if pin is not None:
            command += ['--contract', str(output / 'reference/contract.json'), '--criterion-sha256', pin]
        started = time.monotonic()
        with (output / (mode + '.stdout')).open('x') as stdout, (output / (mode + '.stderr')).open('x') as stderr:
            child = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=300)
        item = {'mode': mode, 'exit': child.returncode, 'seconds': time.monotonic() - started, 'command': command}
        commands.append(item)
        (output / 'commands.json').write_text(json.dumps(commands, indent=2))
        result = json.loads((target / 'result.json').read_text())
        assert child.returncode == code and result['status'] == outcome, (mode, item, result['status'])
        cleanup = json.loads((target / 'cleanup.json').read_text())
        assert cleanup['executor_shutdown'] and cleanup['source_hooks_restored'] and not cleanup['background_errors']
        if mode == 'reference':
            pin = (target / 'criterion.sha256').read_text().strip()
            assert hashlib.sha256((target / 'contract.json').read_bytes()).hexdigest() == pin == result['criterion_sha256']
            (target / 'contract.json').chmod(0o444)
            (target / 'criterion.sha256').chmod(0o444)
            (output / 'retained-before-candidates.sha256').write_text(pin + '\n')
        assert result['criterion_sha256'] == pin
        assert hashlib.sha256((output / 'reference/contract.json').read_bytes()).hexdigest() == pin
        steps = 0 if mode in ('reference', 'guard-fault', 'unknown') else 2
        assert result['ledger']['optimizer_steps'] == result['ledger']['model_forward_calls'] == steps
        if mode == 'guard-fault':
            assert result['blocked_report']['status'] == 'error'
            assert result['blocked_report']['differing_target_values'] > 0
            assert {'option': 'use_relative_action', 'expected': True, 'observed': False} in result['blocked_report']['configuration_changes']
        if mode == 'unknown':
            assert result['blocked_report']['status'] == 'unknown'
        if mode in ('corrected', 'replay'):
            guard = result['guard']
            assert guard['checked_collations'] == guard['checked_loss_inputs'] == 1
            assert guard['unassessed_collations'] == guard['unassessed_loss_inputs'] == 1 and guard['closed']
        assert result['native_shard_min_samples'] > 1
        assert result['total_model_parameters'] > 1_000_000_000
        if mode != 'reference':
            assert result['initial_active_parameter_slice'] == results['reference']['initial_active_parameter_slice']
            assert result['pretrained_embedding_witness'] == results['reference']['pretrained_embedding_witness']
        results[mode] = result
        (output / 'progress.json').write_text(json.dumps({'last_passed': mode, 'criterion_sha256': pin}, indent=2))
    hashes = {results[mode]['final_active_parameter_slice']['sha256'] for mode in ('corrected', 'healthy', 'replay')}
    assert len(hashes) == 1
    assert results['native-fault']['final_active_parameter_slice']['sha256'] not in hashes
    fault = json.loads((output / 'guard-fault/checkpoint/processor_config.json').read_text())
    corrected = json.loads((output / 'corrected/checkpoint/processor_config.json').read_text())
    assert fault['processor_kwargs']['use_relative_action'] is False
    fault['processor_kwargs']['use_relative_action'] = True
    assert fault == corrected
    for name in ('statistics.json', 'embodiment_id.json'):
        assert (output / 'guard-fault/checkpoint' / name).read_bytes() == (output / 'corrected/checkpoint' / name).read_bytes()
    verify(packet)
    summary = {'status': 'PASS_PORTABLE_NATIVE_GR00T_INTEGRATION', 'profile': manifest['profile'],
               'criterion_sha256': pin, 'commands': commands,
               'wrong_native_optimizer_steps': 2, 'guarded_fault_optimizer_steps': 0,
               'guarded_fault_model_forward_calls': 0, 'corrected_optimizer_steps': 2,
               'different_target_values': results['guard-fault']['blocked_report']['differing_target_values'],
               'healthy_corrected_replay_active_slice_sha256': next(iter(hashes)),
               'all_native_shards_have_multiple_samples': True, 'human_trial': False,
               'scope': 'Pinned native startup target and installation workflow; no trained-policy or physical competence claim.'}
    with (output / 'verification.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
