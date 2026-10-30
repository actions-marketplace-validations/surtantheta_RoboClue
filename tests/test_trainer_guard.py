"""Generated-data tests of the startup hook; native training is a separate gate."""
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.trainer_guard import TrainerStartupGuard
from policy_doctor.training_guard import TargetContract, TargetContractViolation


def batch(value=0.25):
    return {'inputs': {'action': np.full((1, 2, 3), value, dtype=np.float32),
                       'action_mask': np.ones((1, 2, 3), dtype=np.float64)}, 'other': object()}


def contract():
    return TargetContract.from_references([(batch(), ['source-A'], [{'generated_unit_fixture': True}])],
                                         intent={'declared_generated_target': 0.25})


class HarnessTrainer:
    def __init__(self):
        self.args = SimpleNamespace(dataloader_num_workers=0, remove_unused_columns=False,
                                    world_size=1, eval_on_start=False)
        self.native_calls = self.loss_calls = 0
        self.next_batch = batch()
        self.last_loss_arguments = None
        self.data_collator = self.collator

    def collator(self, _features):
        self.native_calls += 1
        return self.next_batch

    def compute_loss(self, model, inputs, *args, **kwargs):
        self.loss_calls += 1
        self.last_loss_arguments = (model, inputs, args, kwargs)
        return 'original-loss-result'


def attach(trainer, **kwargs):
    return TrainerStartupGuard(trainer, contract(), source_keys=lambda rows: [row['key'] for row in rows], **kwargs)


def test_original_batch_loss_arguments_and_hooks_preserved():
    trainer = HarnessTrainer()
    original_collator, original_loss = trainer.data_collator, trainer.compute_loss
    with attach(trainer) as guard:
        actual = trainer.data_collator([{'key': 'source-A'}])
        assert actual is trainer.next_batch
        model = object()
        assert trainer.compute_loss(model, actual, True, num_items_in_batch=1) == 'original-loss-result'
        assert trainer.last_loss_arguments == (model, actual, (True,), {'num_items_in_batch': 1})
        assert guard.checked_collations == guard.checked_loss_inputs == 1
    assert trainer.data_collator is original_collator
    assert trainer.compute_loss.__func__ is original_loss.__func__
    assert trainer.compute_loss.__self__ is trainer
    assert 'compute_loss' not in trainer.__dict__
    assert guard.closed and not hasattr(trainer, '_roboclue_startup_guard')


def test_finite_native_error_never_reaches_loss():
    trainer = HarnessTrainer()
    trainer.next_batch = batch(0.75)
    with attach(trainer) as guard:
        with pytest.raises(TargetContractViolation) as caught:
            trainer.data_collator([{'key': 'source-A'}])
        assert caught.value.report['status'] == 'error'
        assert trainer.native_calls == 1 and trainer.loss_calls == 0
        assert guard.blocked_batches == 1


def test_post_collator_target_change_blocked_before_loss():
    trainer = HarnessTrainer()
    with attach(trainer) as guard:
        original = trainer.data_collator([{'key': 'source-A'}])
        altered = {'inputs': {**original['inputs'], 'action': original['inputs']['action'] + 0.125}}
        with pytest.raises(TargetContractViolation) as caught:
            trainer.compute_loss(object(), altered)
        assert caught.value.report['differing_target_values'] == 6
        assert trainer.loss_calls == 0 and guard.checked_loss_inputs == 0


def test_unpaired_loss_input_is_unknown_and_blocked():
    trainer = HarnessTrainer()
    with attach(trainer) as guard:
        with pytest.raises(TargetContractViolation) as caught:
            trainer.compute_loss(object(), trainer.next_batch)
        assert caught.value.report['status'] == 'unknown'
        assert caught.value.report['code'] == 'unpaired_trainer_batch'
        assert trainer.loss_calls == 0 and guard.blocked_batches == 1


def test_uncovered_source_blocks_native_accepted_batch():
    trainer = HarnessTrainer()
    with attach(trainer):
        with pytest.raises(TargetContractViolation) as caught:
            trainer.data_collator([{'key': 'source-B'}])
        assert caught.value.report['code'] == 'uncovered_source_sample'
        assert trainer.native_calls == 1 and trainer.loss_calls == 0


def test_later_batches_explicitly_unassessed_and_training_can_continue():
    trainer = HarnessTrainer()
    with attach(trainer) as guard:
        first = trainer.data_collator([{'key': 'source-A'}])
        trainer.compute_loss(object(), first)
        trainer.next_batch = batch(99.0)
        later = trainer.data_collator([{'key': 'uncovered-later-source'}])
        assert trainer.compute_loss(object(), later) == 'original-loss-result'
        summary = guard.summary()
        assert summary['checked_loss_inputs'] == 1 and summary['unassessed_loss_inputs'] == 1
        assert summary['unassessed_collations'] == 1 and 'later batches unassessed' in summary['scope']


def test_prefetched_later_batch_does_not_steal_first_source_pairing():
    trainer = HarnessTrainer()
    with attach(trainer) as guard:
        first = trainer.data_collator([{'key': 'source-A'}])
        trainer.next_batch = batch(0.75)
        second = trainer.data_collator([{'key': 'source-B'}])
        trainer.compute_loss(object(), first)
        trainer.compute_loss(object(), second)
        assert guard.checked_loss_inputs == 1 and guard.unassessed_loss_inputs == 1


def test_source_keys_frozen_before_native_collator_changes_them():
    trainer = HarnessTrainer()
    keys = ['source-A']
    original = trainer.data_collator

    def collator(rows):
        keys[0] = 'changed-by-native-collator'
        return original(rows)

    trainer.data_collator = collator
    with TrainerStartupGuard(trainer, contract(), source_keys=lambda _rows: keys) as guard:
        actual = trainer.data_collator([{}])
        trainer.compute_loss(object(), actual)
        assert guard.checked_loss_inputs == 1


@pytest.mark.parametrize('wrong', [False, True])
def test_scope_callback_called_once_for_pass_or_failure(wrong):
    trainer = HarnessTrainer()
    calls = []
    if wrong:
        trainer.next_batch = batch(0.5)
    guard = attach(trainer, on_scope_end=lambda: calls.append('finished'))
    if wrong:
        with pytest.raises(TargetContractViolation):
            trainer.data_collator([{'key': 'source-A'}])
    else:
        trainer.data_collator([{'key': 'source-A'}])
    guard.close()
    guard.close()
    assert calls == ['finished']


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('dataloader_num_workers', 1, 'dataloader_num_workers=0'),
    ('remove_unused_columns', True, 'remove_unused_columns=False'),
    ('world_size', 2, 'world_size=1'),
    ('eval_on_start', True, 'eval_on_start=False'),
])
def test_unsupported_trainer_options_refused_before_hook_mutation(field, value, message):
    trainer = HarnessTrainer()
    original = trainer.data_collator
    setattr(trainer.args, field, value)
    with pytest.raises(InputError, match=message):
        attach(trainer)
    assert trainer.data_collator is original


@pytest.mark.parametrize('count', [0, True, 129])
def test_invalid_scope_count_refused(count):
    with pytest.raises(InputError, match='batches must be an integer'):
        attach(HarnessTrainer(), batches=count)


def test_second_attachment_refused_without_disturbing_first():
    trainer = HarnessTrainer()
    with attach(trainer) as guard:
        with pytest.raises(InputError, match='already has'):
            attach(trainer)
        assert trainer._roboclue_startup_guard is guard


def test_detach_refuses_to_overwrite_an_external_hook_change():
    trainer = HarnessTrainer()
    guard = attach(trainer)
    own_collator = trainer.data_collator
    def replacement(rows):
        return rows
    trainer.data_collator = replacement
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        guard.close()
    assert trainer.data_collator is replacement
    trainer.data_collator = own_collator
    guard.close()


def test_retained_closed_handle_does_not_retain_trainer_or_model():
    def make_closed_handle():
        trainer = HarnessTrainer()
        reference = weakref.ref(trainer)
        guard = attach(trainer)
        guard.close()
        return guard, reference

    guard, reference = make_closed_handle()
    gc.collect()
    assert reference() is None
    assert guard.closed and guard.summary()['closed']


def test_raising_scope_callback_still_restores_hooks_and_releases_trainer():
    trainer = HarnessTrainer()
    original_collator = trainer.data_collator

    def fail_callback():
        raise RuntimeError('observer shutdown failed')

    guard = attach(trainer, on_scope_end=fail_callback)
    with pytest.raises(RuntimeError, match='observer shutdown failed'):
        guard.close()
    assert trainer.data_collator is original_collator
    assert 'compute_loss' not in trainer.__dict__
    assert not hasattr(trainer, '_roboclue_startup_guard')
    assert guard.closed and guard.trainer is guard.original_loss is None
    assert guard.close()['closed']


def test_context_cleanup_preserves_original_training_error_with_callback_note():
    trainer = HarnessTrainer()

    def fail_callback():
        raise RuntimeError('observer shutdown failed')

    original_error = ValueError('native training failed')
    with pytest.raises(ValueError, match='native training failed') as caught:
        with attach(trainer, on_scope_end=fail_callback) as guard:
            raise original_error
    assert caught.value is original_error
    assert original_error.__notes__ == ['RoboClue guard cleanup also failed: RuntimeError: observer shutdown failed']
    assert guard.closed and guard.trainer is None
    assert 'compute_loss' not in trainer.__dict__
