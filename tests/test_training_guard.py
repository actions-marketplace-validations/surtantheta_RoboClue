import copy
import json

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import GuardedCollator, TargetContract, TargetContractViolation


def batch():
    return {'inputs': {'action': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                       'action_mask': np.ones((2, 3, 4), dtype=bool)}, 'unassessed': object()}


def contract(value):
    return TargetContract.from_references([(value, ['episode0:0', 'episode0:1'],
        [{'path': 'original.parquet', 'physical_row': n, 'column': 'action'} for n in (0, 1)])],
        intent={'preserve_declared_relative_targets': True, 'scope': 'controlled reference bank'})


def test_fault_and_correction_use_one_immutable_criterion(tmp_path):
    healthy = batch()
    frozen = contract(healthy)
    pin = frozen.criterion_sha256
    assert frozen.check(healthy, ['episode0:0', 'episode0:1'])['status'] == 'pass'
    faulty = copy.deepcopy(healthy)
    faulty['inputs']['action'][1, 2, 3] += 5
    error = frozen.check(faulty, ['episode0:0', 'episode0:1'])
    assert error['status'] == 'error' and error['differing_target_values'] == 1
    assert error['first_difference']['coordinate'] == [1, 2, 3]
    assert error['first_difference']['expected'] == 23 and error['first_difference']['observed'] == 28
    assert error['first_difference']['provenance']['physical_row'] == 1
    path = tmp_path / 'criterion.json'
    frozen.save(path)
    restarted = TargetContract.load(path, expected_sha256=pin)
    assert restarted.check(healthy, ['episode0:0', 'episode0:1'])['status'] == 'pass'
    assert restarted.criterion_sha256 == error['criterion_sha256'] == pin
    with pytest.raises(InputError, match='new output'):
        frozen.save(path)


def test_reference_does_not_share_mutable_arrays_or_metadata():
    healthy = batch()
    frozen = contract(healthy)
    before = frozen.criterion_sha256
    healthy['inputs']['action'][0, 0, 0] = 100
    exposed = frozen.document
    exposed['intent']['scope'] = 'changed'
    assert frozen.criterion_sha256 == before
    assert frozen.check(healthy, ['episode0:0', 'episode0:1'])['status'] == 'error'


@pytest.mark.parametrize('fault', ['mask', 'dtype', 'shape'])
def test_mask_and_schema_changes_are_not_cleared(fault):
    healthy = batch()
    frozen = contract(healthy)
    bad = copy.deepcopy(healthy)
    if fault == 'mask':
        bad['inputs']['action_mask'][0, 0, 0] = False
    elif fault == 'dtype':
        bad['inputs']['action'] = bad['inputs']['action'].astype(np.float64)
    else:
        bad['inputs']['action'] = bad['inputs']['action'][:, :2]
        bad['inputs']['action_mask'] = bad['inputs']['action_mask'][:, :2]
    assert frozen.check(bad, ['episode0:0', 'episode0:1'])['status'] == 'error'


def test_target_values_and_mask_schema_are_reported_as_separate_findings():
    healthy = batch()
    frozen = contract(healthy)
    bad = copy.deepcopy(healthy)
    bad['inputs']['action'][0, 1, 2] += 5
    bad['inputs']['action_mask'] = bad['inputs']['action_mask'].astype(np.float64)
    report = frozen.check(bad, ['episode0:0', 'episode0:1'])
    assert report['status'] == 'error' and report['differing_target_values'] == 1
    assert report['first_difference']['coordinate'] == [0, 1, 2]
    target = [item for item in report['findings'] if item['field'] == 'inputs.action']
    mask = [item for item in report['findings'] if item['field'] == 'inputs.action_mask']
    assert target[0]['code'] == 'target_values_differ' and target[0]['differing_values'] == 1
    assert mask[0]['code'] == 'target_schema_mismatch'
    assert mask[0]['expected_dtype'] == 'bool' and mask[0]['observed_dtype'] == 'float64'
    assert report['mask_values_checked'] == 0


def test_findings_are_bounded_without_omitting_total_mismatches():
    healthy = batch()
    frozen = contract(healthy)
    many = {'inputs': {key: np.repeat(value[:1], 20, axis=0)
                       for key, value in healthy['inputs'].items()}}
    many['inputs']['action'] += 5
    report = frozen.check(many, ['episode0:0'] * 20)
    assert report['status'] == 'error' and report['finding_count'] == 20
    assert len(report['findings']) == 8 and report['findings_truncated']
    assert report['differing_target_values'] == 20 * 3 * 4


def test_uncovered_or_unestablished_inputs_stay_unknown():
    healthy = batch()
    frozen = contract(healthy)
    assert frozen.check(healthy, ['new:0', 'new:1'])['status'] == 'unknown'
    assert frozen.check({'inputs': {}}, ['episode0:0', 'episode0:1'])['status'] == 'unknown'
    bad = copy.deepcopy(healthy)
    bad['inputs']['action'][0, 0, 0] = np.nan
    assert frozen.check(bad, ['episode0:0', 'episode0:1'])['status'] == 'error'


def test_a_nonbinary_loss_mask_is_a_known_error():
    healthy = batch()
    frozen = contract(healthy)
    bad = copy.deepcopy(healthy)
    bad['inputs']['action_mask'] = bad['inputs']['action_mask'].astype(np.float32)
    bad['inputs']['action_mask'][0, 0, 0] = 0.5
    report = frozen.check(bad, ['episode0:0', 'episode0:1'])
    assert report['status'] == 'error' and report['code'] == 'invalid_target_batch'


def test_capacity_refusal_is_unassessed_and_still_blocks_delivery():
    one = {'inputs': {'action': np.ones((1, 50, 64), dtype=np.float32),
                      'action_mask': np.ones((1, 50, 64), dtype=bool)}}
    frozen = TargetContract.from_references([(one, ['A'], [{'source': 'declared-A'}])],
                                            intent={'scope': 'capacity control'})
    large = {'inputs': {key: np.repeat(value, 400, axis=0) for key, value in one['inputs'].items()}}
    wrapped = GuardedCollator(lambda _: large, frozen, source_keys=lambda _: ['A'] * 400)
    with pytest.raises(TargetContractViolation) as caught:
        wrapped([object()] * 400)
    assert caught.value.report['status'] == 'unknown'
    assert 'budget' in caught.value.report['message']
    assert wrapped.blocked_batches == 1 and wrapped.passed_batches == 0


def test_guard_returns_the_original_batch_and_blocks_before_delivery():
    healthy = batch()
    frozen = contract(healthy)
    features = [object(), object()]
    calls = []
    current = [healthy]

    def original(items):
        assert items is features
        calls.append(items)
        return current[0]

    wrapped = GuardedCollator(original, frozen, source_keys=lambda _: ['episode0:0', 'episode0:1'])
    assert wrapped(features) is healthy and calls == [features]
    current[0] = copy.deepcopy(healthy)
    current[0]['inputs']['action'][0, 0, 0] = 7
    delivered = 0
    with pytest.raises(TargetContractViolation) as caught:
        result = wrapped(features)
        delivered += 1
        assert result is current[0]
    assert delivered == 0 and wrapped.passed_batches == wrapped.blocked_batches == 1
    assert caught.value.report is wrapped.last_report


def test_mutating_the_saved_reference_cannot_replace_the_retained_pin(tmp_path):
    frozen = contract(batch())
    path = tmp_path / 'criterion.json'
    frozen.save(path)
    value = json.loads(path.read_text())
    value['intent']['scope'] = 'new objective'
    path.write_text(json.dumps(value))
    with pytest.raises(InputError, match='retained pin'):
        TargetContract.load(path, expected_sha256=frozen.criterion_sha256)


def test_collator_cannot_repair_pairing_by_mutating_aliased_source_keys():
    healthy = batch()
    frozen = contract(healthy)
    original_keys = ['episode0:0', 'episode0:1']
    reordered = copy.deepcopy(healthy)
    for field in ('action', 'action_mask'):
        reordered['inputs'][field] = reordered['inputs'][field][::-1].copy()

    def original(_):
        original_keys.reverse()
        return reordered

    wrapped = GuardedCollator(original, frozen, source_keys=lambda _: original_keys)
    with pytest.raises(TargetContractViolation) as caught:
        wrapped([object(), object()])
    assert original_keys == ['episode0:1', 'episode0:0']
    assert caught.value.report['first_difference']['source_key'] == 'episode0:0'
    assert wrapped.blocked_batches == 1 and wrapped.passed_batches == 0


def test_a_different_explicit_target_objective_can_have_its_own_valid_contract():
    other = batch()
    other['inputs']['action'] += 100
    declared = contract(other)
    assert declared.check(other, ['episode0:0', 'episode0:1'])['status'] == 'pass'
    assert declared.criterion_sha256 != contract(batch()).criterion_sha256
