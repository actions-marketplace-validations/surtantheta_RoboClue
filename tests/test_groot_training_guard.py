"""Generated source-pairing controls; actual native training is qualified separately."""
import gc
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from policy_doctor import groot_training_guard as integration
from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import TargetContract, TargetContractViolation


class Dataset:
    def __init__(self, start=0):
        self.sharded_episodes = [[(0, [start, start + 1, start + 2])]]
        self.calls = 0
        self.value = 0.25

    def get_shard(self, _index):
        self.calls += 1
        self.result = [{'row': int(row), 'value': self.value} for _, rows in self.sharded_episodes[0] for row in rows]
        return self.result


class Trainer:
    def __init__(self, start=0):
        self.dataset = Dataset(start)
        processor = SimpleNamespace(use_relative_action=True, use_percentiles=False, use_mean_std=False,
                                    clip_outliers=False, max_action_dim=3, max_action_horizon=2,
                                    state_action_processor=SimpleNamespace(statistics={'arm': [0, 1]}))
        self.train_dataset = SimpleNamespace(datasets=[self.dataset], processor=processor)
        self.args = SimpleNamespace(dataloader_num_workers=0, remove_unused_columns=False, world_size=1, eval_on_start=False)
        self.data_collator = self.collate
        self.loss_calls = 0

    def collate(self, rows):
        return {'inputs': {'action': np.stack([np.full((2, 3), row['value'], dtype=np.float32) for row in rows]),
                           'action_mask': np.ones((len(rows), 2, 3), dtype=np.float64)}, 'original_marker': object()}

    def compute_loss(self, _model, inputs):
        self.loss_calls += 1
        return inputs['original_marker']

    def get_train_dataloader(self):
        rows = self.dataset.get_shard(0)
        # This control deliberately shuffles the original sample objects.
        return iter([self.data_collator([rows[2], rows[0]])])


@pytest.fixture
def source_controls(monkeypatch):
    monkeypatch.setattr(integration, '_native', lambda trainer: (trainer.train_dataset, integration._Files()))

    def describe(observer, _dataset, position, step):
        return {'source': {'generated_control': True, 'position': position, 'step': step},
                'processor': observer.processor, 'scope': 'Generated unit control only'}

    monkeypatch.setattr(integration.Gr00tSourceObserver, '_describe', describe)


def reference(tmp_path):
    trainer = Trainer()
    contract = integration.capture_gr00t_reference(trainer, output=tmp_path / 'contract.json',
                                                   intent={'generated_control': True})
    assert trainer.loss_calls == 0 and 'get_shard' not in trainer.dataset.__dict__
    return contract


def test_original_multi_sample_shard_and_shuffled_objects_preserved(source_controls):
    trainer = Trainer()
    observer = integration.Gr00tSourceObserver(trainer)
    result = trainer.dataset.get_shard(0)
    assert result is trainer.dataset.result and len(result) == 3
    records = observer.records([result[2], result[0]])
    assert records[0][0] is result[2] and records[1][0] is result[0]
    assert [record[2]['source']['step'] for record in records] == [2, 0]
    observer.stop_capture()
    assert not observer.samples
    assert trainer.dataset.get_shard(0) is trainer.dataset.result
    observer.close()
    assert 'get_shard' not in trainer.dataset.__dict__


def test_intended_reference_then_wrong_target_block_and_same_pin_correction(tmp_path, source_controls):
    contract = reference(tmp_path)
    fault = Trainer()
    fault.dataset.value = 0.75
    fault.train_dataset.processor.use_relative_action = False
    with pytest.raises(TargetContractViolation) as caught:
        with integration.Gr00tTrainingGuard(fault, contract=tmp_path / 'contract.json',
                                            criterion_sha256=contract.criterion_sha256):
            next(fault.get_train_dataloader())
    assert fault.loss_calls == 0 and caught.value.report['differing_target_values'] == 12
    assert {'option': 'use_relative_action', 'expected': True, 'observed': False} in caught.value.report['configuration_changes']
    assert 'get_shard' not in fault.dataset.__dict__
    corrected = Trainer()
    with integration.Gr00tTrainingGuard(corrected, contract=tmp_path / 'contract.json',
                                        criterion_sha256=contract.criterion_sha256) as guard:
        native = next(corrected.get_train_dataloader())
        assert corrected.compute_loss(None, native) is native['original_marker']
    assert guard.summary()['checked_loss_inputs'] == 1 and not guard.summary()['configuration_changes']
    assert TargetContract.load(tmp_path / 'contract.json', expected_sha256=contract.criterion_sha256) == contract


def test_unknown_source_blocks_instead_of_guessing(tmp_path, source_controls):
    contract = reference(tmp_path)
    trainer = Trainer(20)
    with pytest.raises(TargetContractViolation) as caught:
        with integration.Gr00tTrainingGuard(trainer, contract=tmp_path / 'contract.json',
                                            criterion_sha256=contract.criterion_sha256):
            next(trainer.get_train_dataloader())
    assert caught.value.report['status'] == 'unknown' and trainer.loss_calls == 0


def test_bad_pin_and_bad_options_do_not_leave_source_hooks(tmp_path, source_controls):
    contract = reference(tmp_path)
    trainer = Trainer()
    with pytest.raises(InputError):
        integration.Gr00tTrainingGuard(trainer, contract=tmp_path / 'contract.json', criterion_sha256='0' * 64)
    assert 'get_shard' not in trainer.dataset.__dict__
    trainer.args.dataloader_num_workers = 2
    with pytest.raises(InputError, match='dataloader_num_workers=0'):
        integration.Gr00tTrainingGuard(trainer, contract=tmp_path / 'contract.json',
                                       criterion_sha256=contract.criterion_sha256)
    assert 'get_shard' not in trainer.dataset.__dict__


def test_retained_closed_guard_does_not_retain_trainer(tmp_path, source_controls):
    contract = reference(tmp_path)
    trainer = Trainer()
    ref = weakref.ref(trainer)
    guard = integration.Gr00tTrainingGuard(trainer, contract=tmp_path / 'contract.json',
                                          criterion_sha256=contract.criterion_sha256)
    guard.close()
    del trainer
    gc.collect()
    assert ref() is None and guard.observer.closed


def test_explicit_source_bound_refuses_large_shard_before_native_read(source_controls):
    trainer = Trainer()
    observer = integration.Gr00tSourceObserver(trainer, max_samples=2)
    try:
        with pytest.raises(InputError, match='shard exceeds'):
            trainer.dataset.get_shard(0)
        assert trainer.dataset.calls == 0
    finally:
        observer.close()


def test_external_source_hook_change_not_clobbered(source_controls):
    trainer = Trainer()
    observer = integration.Gr00tSourceObserver(trainer)
    wrapper = trainer.dataset.get_shard
    trainer.dataset.get_shard = lambda _index: []
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        observer.close()
    trainer.dataset.get_shard = wrapper
    observer.close()


def test_file_source_fingerprint_changes_and_special_inputs(tmp_path):
    path = tmp_path / 'rows'
    path.write_bytes(b'original')
    files = integration._Files()
    before = files.fingerprint(path)
    assert files.fingerprint(path) == before and files.bytes_hashed == 8
    path.write_bytes(b'changed')
    with pytest.raises(InputError, match='changed'):
        files.verify()
    with pytest.raises(InputError, match='changed'):
        files.fingerprint(path)
    with pytest.raises(InputError, match='Cannot read'):
        integration._Files().fingerprint(tmp_path / 'missing')
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(InputError, match='regular'):
        integration._Files().fingerprint(link)


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='FIFO needs POSIX')
def test_fifo_source_refused_before_open(tmp_path):
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    with pytest.raises(InputError, match='regular'):
        integration._Files().fingerprint(fifo)


def test_native_module_absence_has_actionable_error():
    with pytest.raises(InputError, match='Load the supported GR00T'):
        integration.Gr00tSourceObserver(Trainer())


def test_reference_refuses_existing_destination_before_hooking(tmp_path, source_controls):
    path = tmp_path / 'existing'
    path.write_text('keep')
    trainer = Trainer()
    with pytest.raises(InputError, match='new file'):
        integration.capture_gr00t_reference(trainer, output=path, intent={'generated': True})
    assert path.read_text() == 'keep' and 'get_shard' not in trainer.dataset.__dict__


def test_two_dataset_hook_ownership_is_checked_before_any_restore(source_controls):
    trainer = Trainer()
    other = Dataset(10)
    trainer.train_dataset.datasets.append(other)
    observer = integration.Gr00tSourceObserver(trainer)
    first_wrapper, second_wrapper = trainer.dataset.get_shard, other.get_shard
    other.get_shard = lambda _index: []
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        observer.close()
    assert trainer.dataset.get_shard is first_wrapper
    other.get_shard = second_wrapper
    observer.close()
    assert 'get_shard' not in trainer.dataset.__dict__ and 'get_shard' not in other.__dict__


def test_reference_prefetch_failure_still_shuts_executor_and_does_not_publish(tmp_path, source_controls):
    trainer = Trainer()
    native_loader = trainer.get_train_dataloader
    shutdown = []

    def failed_prefetch(**_kwargs):
        raise RuntimeError('native prefetch failed')

    def loader():
        value = native_loader()
        trainer.train_dataset._cache_job = SimpleNamespace(cancel=lambda: False, result=failed_prefetch)
        trainer.train_dataset._executor = SimpleNamespace(shutdown=lambda **kwargs: shutdown.append(kwargs))
        return value

    trainer.get_train_dataloader = loader
    path = tmp_path / 'contract.json'
    with pytest.raises(RuntimeError, match='native prefetch failed'):
        integration.capture_gr00t_reference(trainer, output=path, intent={'generated': True})
    assert shutdown == [{'wait': True, 'cancel_futures': True}]
    assert not path.exists() and 'get_shard' not in trainer.dataset.__dict__
