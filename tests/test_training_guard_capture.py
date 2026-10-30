import json
import os

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import TargetContract
from policy_doctor.training_guard_capture import check_capture, load_capture, main, save_capture


def reference(tmp_path):
    batch = {'inputs': {'action': np.arange(6, dtype=np.float32).reshape(1, 2, 3),
                        'action_mask': np.ones((1, 2, 3), dtype=np.float64)}}
    contract = TargetContract.from_references([(batch, ['original-row'], [{'episode': 48, 'row': 105}])],
                                               intent={'transformation': 'declared native reference'})
    contract_path = tmp_path / 'contract.json'
    contract.save(contract_path)
    return batch, contract, contract_path


@pytest.mark.parametrize(('case', 'exit_code', 'status'), [
    ('healthy', 0, 'pass'), ('target', 1, 'error'), ('mask', 1, 'error'), ('uncovered', 3, 'unknown'),
])
def test_cli_fixed_criterion_and_distinct_outcomes(tmp_path, capsys, case, exit_code, status):
    batch, contract, contract_path = reference(tmp_path)
    if case == 'target':
        batch['inputs']['action'][0, 0, 1] += 0.25
    elif case == 'mask':
        batch['inputs']['action_mask'][0, 0, 1] = 0
    keys = ['another-row' if case == 'uncovered' else 'original-row']
    capture = tmp_path / 'native.json'
    save_capture(batch, keys, capture)
    output = tmp_path / 'report.json'
    assert main(['check', '--contract', str(contract_path), '--criterion-sha256', contract.criterion_sha256,
                 '--capture', str(capture), '--output', str(output), '--json']) == exit_code
    compact = json.loads(capsys.readouterr().out)
    full = json.loads(output.read_text())
    assert compact['status'] == full['status'] == status
    assert full['criterion_sha256'] == contract.criterion_sha256
    assert len(json.dumps(compact).encode()) < 2048
    if case in ('target', 'mask'):
        assert full['first_difference']['coordinate'] == [0, 0, 1]
        assert full['first_difference']['provenance']['row'] == 105


def test_dtype_preserved_and_tampered_array_refused(tmp_path):
    batch, contract, path = reference(tmp_path)
    capture = tmp_path / 'capture.json'
    save_capture(batch, ['original-row'], capture)
    loaded, keys = load_capture(capture)
    assert loaded['inputs']['action'].dtype == np.float32
    assert loaded['inputs']['action_mask'].dtype == np.float64
    assert contract.check(loaded, keys)['status'] == 'pass'
    value = json.loads(capture.read_text())
    value['samples'][0]['action']['data'][0][0] += 0.1
    capture.write_text(json.dumps(value))
    with pytest.raises(InputError, match='content identity'):
        check_capture(path, contract.criterion_sha256, capture)


def test_pin_drift_does_not_rebuild_criterion(tmp_path, capsys):
    batch, contract, path = reference(tmp_path)
    capture = tmp_path / 'capture.json'
    save_capture(batch, ['original-row'], capture)
    original = path.read_bytes()
    output = tmp_path / 'report.json'
    assert main(['check', '--contract', str(path), '--criterion-sha256', '0' * 64,
                 '--capture', str(capture), '--output', str(output), '--json']) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'input_error' and 'retained pin' in result['message']
    assert not output.exists() and path.read_bytes() == original


@pytest.mark.parametrize('special', ['missing', 'fifo', 'symlink'])
def test_bad_capture_paths_refuse_promptly(tmp_path, capsys, special):
    _, contract, path = reference(tmp_path)
    capture = tmp_path / special
    if special == 'fifo':
        os.mkfifo(capture)
    elif special == 'symlink':
        capture.symlink_to(path)
    assert main(['check', '--contract', str(path), '--criterion-sha256', contract.criterion_sha256,
                 '--capture', str(capture), '--output', str(tmp_path / 'report'), '--json']) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'input_error'


def test_output_cannot_overwrite_contract_or_capture(tmp_path, capsys):
    batch, contract, path = reference(tmp_path)
    capture = tmp_path / 'capture.json'
    save_capture(batch, ['original-row'], capture)
    for output in (path, capture):
        before = output.read_bytes()
        assert main(['check', '--contract', str(path), '--criterion-sha256', contract.criterion_sha256,
                     '--capture', str(capture), '--output', str(output), '--json']) == 2
        assert json.loads(capsys.readouterr().out)['status'] == 'input_error'
        assert output.read_bytes() == before


def test_duplicate_capture_keys_are_refused(tmp_path):
    capture = tmp_path / 'duplicate.json'
    capture.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(InputError, match='Duplicate'):
        load_capture(capture)


def test_mixed_sample_dtype_cannot_be_silently_coerced(tmp_path):
    from policy_doctor.training_guard import _descriptor

    capture = tmp_path / 'mixed.json'
    capture.write_text(json.dumps({'schema_version': 1, 'kind': 'native_training_target_capture',
        'source_keys': ['a', 'b'], 'samples': [
            {'action': _descriptor(np.ones((2, 3), dtype=dtype)),
             'action_mask': _descriptor(np.ones((2, 3), dtype=np.float64))}
            for dtype in (np.float32, np.float64)]}))
    with pytest.raises(InputError, match='without coercion'):
        load_capture(capture)
