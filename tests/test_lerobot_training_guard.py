"""Actual Torch policy hooks; upstream LeRobot/public-data proof is a separate gate."""
import gc
import json
import weakref

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.lerobot_training_guard import (
    LeRobotTrainingGuard,
    capture_lerobot_reference,
    lerobot_target_view,
)
from policy_doctor.training_guard import TargetContract, TargetContractViolation

torch = pytest.importorskip('torch')
DATASET = 'sha256:' + '1' * 64


def batch(value=0.25):
    return {'action': torch.full((2, 3, 4), value), 'index': torch.tensor([7, 19]),
            'action_is_pad': torch.tensor([[False, False, True], [False, False, False]]),
            'observation.state': torch.zeros(2, 4)}


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.calls = 0
        self.last_batch = None

    def forward(self, batch, reduction='mean'):
        self.calls += 1
        self.last_batch = batch
        loss = ((self.weight - batch['action']) ** 2).mean()
        return loss if reduction == 'mean' else loss.expand(2)


def reference(tmp_path, *, padding='action_is_pad'):
    criterion = capture_lerobot_reference([batch()], dataset_id=DATASET,
                                         intent={'recipe': 'declared intended preprocessing'}, padding=padding)
    path = tmp_path / 'reference.json'
    criterion.save(path)
    return path, criterion.criterion_sha256


def attach(policy, tmp_path, **kwargs):
    path, pin = reference(tmp_path)
    return LeRobotTrainingGuard(policy, contract=path, criterion_sha256=pin, dataset_id=DATASET, **kwargs)


def test_policy_arguments_identity_gradients_and_detachment(tmp_path):
    model, actual = Policy(), batch()
    with attach(model, tmp_path) as guard:
        loss = model(actual, reduction='none')
        loss.mean().backward()
        assert model.last_batch is actual and model.calls == 1
        assert model.weight.grad is not None and guard.checked == 1
        assert guard.reports[0]['samples_checked'] == 2
    assert guard.closed and not model._forward_pre_hooks
    assert not hasattr(model, '_roboclue_policy_guard')


def test_wrong_target_accepted_without_guard_blocked_then_corrected_same_pin(tmp_path):
    model = Policy()
    path, pin = reference(tmp_path)
    wrong = batch(0.75)
    assert torch.isfinite(model(wrong)) and model.calls == 1
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET) as guard:
        with pytest.raises(TargetContractViolation) as caught:
            model(wrong)
        assert caught.value.report['differing_target_values'] == 24
        assert model.calls == 1
        model(batch())
        assert model.calls == 2 and guard.checked == 1
    assert pin == guard.contract.criterion_sha256


def test_padding_change_is_error_not_normalization_guess(tmp_path):
    model = Policy()
    with attach(model, tmp_path):
        wrong = batch()
        wrong['action_is_pad'][0, 0] = True
        with pytest.raises(TargetContractViolation) as caught:
            model(wrong)
        assert caught.value.report['differing_mask_values'] == 4
        assert model.calls == 0


def test_shuffle_pairs_global_row_ids_instead_of_batch_order(tmp_path):
    model, original = Policy(), batch()
    original['action'][1] = 0.5
    criterion = capture_lerobot_reference([original], dataset_id=DATASET, intent={'recipe': 'two distinct rows'})
    path = tmp_path / 'distinct.json'
    criterion.save(path)
    shuffled = {k: value.flip(0) for k, value in original.items()}
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=criterion.criterion_sha256, dataset_id=DATASET):
        model(shuffled)
    assert model.calls == 1


@pytest.mark.parametrize('mutation,code', [
    (lambda b: b.pop('index'), 'unsupported_lerobot_policy_input'),
    (lambda b: b.update(index=torch.tensor([8, 20])), 'uncovered_source_sample'),
    (lambda b: b.update(index=torch.tensor([7., 19.])), 'unsupported_lerobot_policy_input'),
    (lambda b: b.update(action_is_pad=torch.ones(2, 3)), 'unsupported_lerobot_policy_input'),
    (lambda b: b.update(action=torch.ones(2, 4)), 'unsupported_lerobot_policy_input'),
])
def test_unsupported_or_uncovered_inputs_block_before_forward(tmp_path, mutation, code):
    model, actual = Policy(), batch()
    mutation(actual)
    with attach(model, tmp_path):
        with pytest.raises(TargetContractViolation) as caught:
            model(actual)
        assert caught.value.report['status'] == 'unknown'
        assert caught.value.report['code'] == code
    assert model.calls == 0


def test_keyword_batch_and_later_unassessed_scope(tmp_path):
    model = Policy()
    with attach(model, tmp_path) as guard:
        model(batch=batch())
        model(batch(9.0))
        assert guard.checked == guard.unassessed == 1
    assert 'later inputs unassessed' in guard.summary()['scope']


def test_input_modified_after_preprocessing_is_checked(tmp_path):
    model, actual = Policy(), batch()
    with attach(model, tmp_path):
        actual['action'].add_(1.0)
        with pytest.raises(TargetContractViolation):
            model(actual)
    assert model.calls == 0


def test_later_hook_cannot_silently_mutate_after_guard(tmp_path):
    model = Policy()
    with attach(model, tmp_path):
        handle = model.register_forward_pre_hook(lambda *_: None)
        try:
            with pytest.raises(TargetContractViolation) as caught:
                model(batch())
            assert 'later forward pre-hook' in caught.value.report['message']
        finally:
            handle.remove()
    assert model.calls == 0


def test_existing_hooks_run_first_and_keep_their_ownership(tmp_path):
    model = Policy()
    handle = model.register_forward_pre_hook(lambda *_: None)
    with attach(model, tmp_path):
        model(batch())
    assert list(model._forward_pre_hooks) == [handle.id]
    handle.remove()


def test_wrong_pin_or_dataset_refused_without_mutating_policy(tmp_path):
    model = Policy()
    path, pin = reference(tmp_path)
    with pytest.raises(InputError, match='retained pin'):
        LeRobotTrainingGuard(model, contract=path, criterion_sha256='0' * 64, dataset_id=DATASET)
    with pytest.raises(InputError, match='different dataset'):
        LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id='sha256:' + '2' * 64)
    assert not model._forward_pre_hooks


def test_uninitialized_distributed_execution_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(InputError, match='Initialize'):
        attach(Policy(), tmp_path)


def test_double_attachment_refused_and_closed_handle_releases_policy(tmp_path):
    model = Policy()
    reference_to_model = weakref.ref(model)
    path, pin = reference(tmp_path)
    guard = LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET)
    with pytest.raises(InputError, match='already has'):
        LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET)
    guard.close()
    del model
    gc.collect()
    assert reference_to_model() is None and guard.closed


def test_explicit_all_valid_intent_and_contradictory_pad():
    actual = batch()
    with pytest.raises(InputError, match='contradicts'):
        lerobot_target_view(actual, dataset_id=DATASET, padding='all_valid')
    actual.pop('action_is_pad')
    view, _ = lerobot_target_view(actual, dataset_id=DATASET, padding='all_valid')
    assert view['inputs']['action_mask'].all()
    np.testing.assert_array_equal(view['inputs']['action'], actual['action'].numpy())


def test_nonfinite_targets_never_pass(tmp_path):
    actual, model = batch(), Policy()
    actual['action'][0, 0, 0] = float('nan')
    with attach(model, tmp_path):
        with pytest.raises(TargetContractViolation) as caught:
            model(actual)
        assert caught.value.report['status'] == 'error'
    assert model.calls == 0


def with_cameras():
    return {**batch(), 'observation.images.front': torch.full((2, 3, 2, 2), 0.25),
            'observation.images.wrist': torch.full((2, 3, 2, 2), 0.75)}


def input_reference(tmp_path):
    fields = ('observation.state', 'observation.images.front', 'observation.images.wrist')
    declarations = {'units': {'observation.state': 'declared_test_unit'},
                    'camera_map': {'front': 'declared_front', 'wrist': 'declared_wrist'}}
    criterion = capture_lerobot_reference([with_cameras()], dataset_id=DATASET, input_fields=fields,
        declarations=declarations, intent={'recipe': 'deterministic input-preservation test, no physical semantics'})
    path = tmp_path / 'inputs.json'
    criterion.save(path)
    return path, criterion.criterion_sha256, declarations


def test_swapped_camera_inputs_block_with_other_camera_match(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    actual, model = with_cameras(), Policy()
    actual['observation.images.front'], actual['observation.images.wrist'] = (
        actual['observation.images.wrist'], actual['observation.images.front'])
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations) as guard:
        with pytest.raises(TargetContractViolation) as caught:
            model(actual)
        report = caught.value.report
        assert report['differing_target_values'] == 0
        assert report['code'] == 'model_input_witness_changed'
        assert report['first_difference']['field'] == 'observation.images.front'
        assert report['first_difference']['observed_matches_other_reference_fields'] == ['observation.images.wrist']
        model(with_cameras())
        assert guard.checked == 1
    assert model.calls == 1


def test_changed_declared_units_block_without_inferring_physical_truth(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    declarations['units']['observation.state'] = 'different_declared_unit'
    model = Policy()
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations):
        with pytest.raises(TargetContractViolation) as caught:
            model(with_cameras())
        assert caught.value.report['code'] == 'declared_model_input_contract_changed'
        assert 'physical truth is not inferred' in caught.value.report['message']
    assert model.calls == 0


def test_state_scaling_change_detected_even_when_action_targets_unchanged(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    actual, model = with_cameras(), Policy()
    actual['observation.state'] += 0.1
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations):
        with pytest.raises(TargetContractViolation) as caught:
            model(actual)
        assert caught.value.report['first_difference']['field'] == 'observation.state'
        assert caught.value.report['differing_target_values'] == 0
    assert model.calls == 0


def test_stochastic_references_require_separate_explicit_objective():
    first, different = with_cameras(), with_cameras()
    different['observation.images.front'] += 0.1
    with pytest.raises(InputError, match='stochastic augmentation'):
        capture_lerobot_reference([first, different], dataset_id=DATASET,
            input_fields=['observation.images.front'], intent={'recipe': 'badly declared deterministic images'})


@pytest.mark.parametrize('fields', ['observation.state', [1], [['bad']], ['x', 'x']])
def test_invalid_input_field_declarations_refused(fields):
    with pytest.raises(InputError, match='sixteen distinct'):
        capture_lerobot_reference([batch()], dataset_id=DATASET, input_fields=fields, intent={'recipe': 'test'})


def test_missing_required_camera_stays_unknown_and_blocked(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    actual, model = with_cameras(), Policy()
    del actual['observation.images.wrist']
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations):
        with pytest.raises(TargetContractViolation) as caught:
            model(actual)
        assert caught.value.report['status'] == 'unknown'
        assert 'field is missing' in caught.value.report['message']
    assert model.calls == 0


def test_conflicting_duplicate_row_cannot_hide_first_camera_change(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    actual, model = with_cameras(), Policy()
    actual['index'][1] = actual['index'][0]
    actual['action_is_pad'][1] = actual['action_is_pad'][0]
    actual['observation.images.front'][0] = 0.75
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations):
        with pytest.raises(TargetContractViolation) as caught:
            model(actual)
        assert caught.value.report['status'] == 'unknown'
        assert 'Duplicate source rows' in caught.value.report['message']
    assert model.calls == 0


def test_conflicting_duplicate_row_refused_while_freezing_inputs():
    actual = with_cameras()
    actual['index'][1] = actual['index'][0]
    actual['action_is_pad'][1] = actual['action_is_pad'][0]
    actual['observation.images.front'][0] = 0.75
    with pytest.raises(InputError, match='Duplicate source rows'):
        capture_lerobot_reference([actual], dataset_id=DATASET,
            input_fields=['observation.images.front'], intent={'recipe': 'ambiguous per-row images'})


def test_identical_repeated_source_row_is_not_a_false_error(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    actual, model = with_cameras(), Policy()
    actual['index'][1] = actual['index'][0]
    actual['action_is_pad'][1] = actual['action_is_pad'][0]
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations):
        model(actual)
    assert model.calls == 1


def test_mutating_summary_or_failure_report_cannot_rewrite_retained_witnesses(tmp_path):
    path, pin, declarations = input_reference(tmp_path)
    actual, model = with_cameras(), Policy()
    actual['observation.state'] += 0.1
    with LeRobotTrainingGuard(model, contract=path, criterion_sha256=pin, dataset_id=DATASET,
                              declarations=declarations) as guard:
        guard.summary()['declared_input_fields'].clear()
        with pytest.raises(TargetContractViolation) as first:
            model(actual)
        first.value.report['first_difference']['expected'].update(
            first.value.report['first_difference']['observed'])
        with pytest.raises(TargetContractViolation) as second:
            model(actual)
        assert second.value.report['code'] == 'model_input_witness_changed'
        assert len(guard.summary()['declared_input_fields']) == 3
    assert model.calls == 0


@pytest.mark.parametrize('malformation', ['row_list', 'missing_field', 'bad_descriptor', 'bad_shape', 'bad_dtype'])
def test_pinned_malformed_input_witnesses_refused_before_hook_attachment(tmp_path, malformation):
    path, _, declarations = input_reference(tmp_path)
    value = json.loads(path.read_text())
    inputs = value['intent']['roboclue_model_inputs']
    key = next(iter(inputs['references']))
    if malformation == 'row_list':
        inputs['references'][key] = []
    elif malformation == 'missing_field':
        inputs['references'][key].pop('observation.state')
    elif malformation == 'bad_descriptor':
        inputs['references'][key]['observation.state'] = None
    elif malformation == 'bad_shape':
        inputs['references'][key]['observation.state']['shape'] = [True]
    else:
        inputs['references'][key]['observation.state']['dtype'] = '|O'
    criterion = TargetContract(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
    malformed = tmp_path / 'malformed.json'
    criterion.save(malformed)
    model = Policy()
    with pytest.raises(InputError, match='Retained model-input'):
        LeRobotTrainingGuard(model, contract=malformed, criterion_sha256=criterion.criterion_sha256,
                              dataset_id=DATASET, declarations=declarations)
    assert not model._forward_pre_hooks and not hasattr(model, '_roboclue_policy_guard')
