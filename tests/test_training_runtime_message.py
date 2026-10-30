"""Runtime mismatch stays refused while naming the actual required dependency."""
import hashlib
import json

import numpy as np

from policy_doctor.big_bets_frontdoor import main, summarize
from policy_doctor.training_guard import TargetContract
from policy_doctor.training_guard_capture import save_capture


def inputs(tmp_path):
    batch = {'inputs': {'action': np.zeros((1, 2, 1), dtype=np.float32),
                        'action_mask': np.ones((1, 2, 1), dtype=np.float64)}}
    bank = TargetContract.from_references([(batch, ['generated-source'], [{'generated_unit_control': True}])],
                                          intent={'generated_control': True})
    document = bank.document
    document['verifier']['numpy'] = '0.0.unsupported-control'
    raw = json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    contract, capture = tmp_path / 'contract.json', tmp_path / 'capture.json'
    contract.write_bytes(raw)
    save_capture(batch, ['generated-source'], capture)
    return contract, capture, hashlib.sha256(raw).hexdigest()


def test_runtime_mismatch_names_numpy_and_preserves_criterion(tmp_path, capsys):
    contract, capture, pin = inputs(tmp_path)
    assert main(['check-training-targets', '--contract', str(contract), '--criterion-sha256', pin,
                 '--capture', str(capture), '--json']) == 2
    report = json.loads(capsys.readouterr().err)
    message = report['error']['message']
    assert 'requires NumPy 0.0.unsupported-control' in message and np.__version__ in message
    assert 'criterion_sha256 unchanged' in message
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == pin


def test_invalid_pin_keeps_original_pin_error_priority(tmp_path, capsys):
    contract, capture, _pin = inputs(tmp_path)
    assert main(['check-training-targets', '--contract', str(contract), '--criterion-sha256', '0' * 64,
                 '--capture', str(capture), '--json']) == 2
    message = json.loads(capsys.readouterr().err)['error']['message']
    assert 'criterion' in message.lower() and 'requires NumPy' not in message


def test_offline_target_pass_explicitly_leaves_additional_live_input_witnesses_unassessed(tmp_path, capsys):
    batch = {'inputs': {'action': np.zeros((1, 2, 1), dtype=np.float32),
                        'action_mask': np.ones((1, 2, 1), dtype=np.bool_)}}
    bank = TargetContract.from_references([(batch, ['source'], [{'generated_control': True}])],
        intent={'roboclue_model_inputs': {'fields': ['observation.images.front'],
                                          'declarations': {'units': {'action': None}}}})
    path, capture = tmp_path / 'live-contract.json', tmp_path / 'targets.json'
    bank.save(path)
    save_capture(batch, ['source'], capture)
    assert main(['check-training-targets', '--contract', str(path), '--criterion-sha256', bank.criterion_sha256,
                 '--capture', str(capture), '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'pass' and report['differing_target_values'] == 0
    coverage = report['additional_model_input_coverage']
    assert coverage['status'] == 'unknown' and coverage['declarations_assessed'] is False
    assert coverage['declared_fields'] == ['observation.images.front']
    assert summarize(report)['additional_model_input_coverage'] == coverage
