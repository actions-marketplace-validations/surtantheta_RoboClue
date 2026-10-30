"""Generated adapter controls; native OpenPI execution is a separate gate."""
import gc
import weakref

import numpy as np
import pytest

from policy_doctor import openpi_training_guard as integration
from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import TargetContractViolation


class Transformed:
    def __init__(self, value=.25):
        self.value = value
        self.last = None
        self.collision = False

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.last = {'actions': np.full((2, 3), self.value + index, dtype=np.float32),
                     'state': np.full(3, index, dtype=np.float32)}
        if self.collision:
            self.last[integration.MARKER] = np.int64(99)
        return self.last


class Inner:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for indices in ([2, 0], [1, 2]):
            samples = [self.dataset[index] for index in indices]
            yield {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}


class Loader:
    def __init__(self, dataset):
        self._data_loader = Inner(dataset)
        self.native_observation_calls = 0
        self.alter_after_pairing = False
        self.last = None
        self.close_error = None

    def __iter__(self):
        try:
            for batch in self._data_loader:
                assert integration.MARKER not in batch
                self.native_observation_calls += 1
                actions = batch['actions'] + .5 if self.alter_after_pairing else batch['actions']
                self.last = (object(), actions)
                yield self.last
        finally:
            if self.close_error is not None:
                raise self.close_error


@pytest.fixture
def controls(tmp_path, monkeypatch):
    root = tmp_path / 'materialized'
    root.mkdir()
    (root / 'rows.json').write_text('{"generated_source_identity":true}')
    monkeypatch.setattr(integration, '_native', lambda loader: (loader._data_loader, loader._data_loader.dataset))

    def make(value=.25):
        source = integration.OpenpiSourceDataset(Transformed(value), dataset_root=root)
        return Loader(source)

    return make, root


def reference(tmp_path, make):
    path = tmp_path / 'contract.json'
    contract = integration.capture_openpi_reference(make(), output=path,
                                                     intent={'generated_control': 'intended target'})
    return path, contract


def test_original_random_access_index_survives_transform_repack_and_shuffle(controls):
    make, _root = controls
    dataset = make()._data_loader.dataset
    enveloped = dataset[2]
    assert integration.MARKER not in dataset.dataset.last
    assert enveloped['actions'] is dataset.dataset.last['actions']
    assert enveloped[integration.MARKER] == 2
    assert dataset.key(2).endswith(':index:2')
    assert dataset.provenance(2)['original_random_access_index'] == 2


@pytest.mark.parametrize('index', [-1, 3, True, .5])
def test_unsupported_indices_cannot_become_source_counter_identities(controls, index):
    dataset = controls[0]()._data_loader.dataset
    with pytest.raises(InputError, match='index'):
        dataset[index]


def test_reserved_source_metadata_collision_is_refused(controls):
    dataset = controls[0]()._data_loader.dataset
    dataset.dataset.collision = True
    with pytest.raises(InputError, match='collides'):
        dataset[0]


def test_reference_fault_and_correction_preserve_same_pin(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    fault = make(.75)
    original = fault._data_loader
    with pytest.raises(TargetContractViolation) as caught:
        with integration.OpenpiTrainingGuard(fault, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
            list(guard)
    assert caught.value.report['differing_target_values'] == 12
    assert fault.native_observation_calls == 0 and fault._data_loader is original
    corrected = make()
    with integration.OpenpiTrainingGuard(corrected, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
        first = next(iter(guard))
        assert first is corrected.last
    assert guard.summary()['checked_deliveries'] == 1
    assert guard.summary()['comparison_mask_scope'].startswith('RoboClue all-true')
    assert 'NOT a native' in guard.summary()['comparison_mask_scope']
    assert contract.document['references'][next(iter(contract.document['references']))]['action_mask']['dtype'] == '|b1'


def test_native_observation_conversion_is_rechecked_before_delivery(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    loader.alter_after_pairing = True
    with pytest.raises(TargetContractViolation) as caught:
        with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
            list(guard)
    assert caught.value.report['differing_target_values'] == 12
    assert loader.native_observation_calls == 1 and guard.checked_deliveries == 0


def test_changed_materialized_source_is_unknown_instead_of_repaired_by_counter(tmp_path, controls):
    make, root = controls
    path, contract = reference(tmp_path, make)
    (root / 'rows.json').write_text('{"different_materialized_input":true}')
    loader = make()
    with pytest.raises(TargetContractViolation) as caught:
        with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
            list(guard)
    assert caught.value.report['status'] == 'unknown'
    assert caught.value.report['code'] == 'uncovered_source_sample' and loader.native_observation_calls == 0


def test_later_batches_remain_unassessed_and_original_tuple_is_preserved(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
        outputs = list(guard)
        assert outputs[-1] is loader.last
        assert guard.checked_native_batches == guard.checked_deliveries == guard.unassessed_deliveries == 1


def test_existing_contract_output_is_refused_without_loader_changes(tmp_path, controls):
    make, _root = controls
    path, _contract = reference(tmp_path, make)
    loader = make()
    original = loader._data_loader
    with pytest.raises(InputError, match='new file'):
        integration.capture_openpi_reference(loader, output=path, intent={'generated': True})
    assert loader._data_loader is original


def test_input_change_after_wrapper_construction_is_refused(tmp_path, controls):
    make, root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    (root / 'rows.json').write_text('changed during observation')
    with pytest.raises(InputError, match='content changed'):
        with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
            list(guard)
    assert loader.native_observation_calls == 0


def test_common_held_startup_iterator_closes_on_context_exit_and_releases_loader(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    ref = weakref.ref(loader)
    with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
        iterator = iter(guard)
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    guard.close()
    del loader
    gc.collect()
    assert ref() is None and guard.closed


def test_early_break_iterator_is_owned_and_closed_by_guard(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    original = loader._data_loader
    with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
        iterator = iter(guard)
        for _batch in iterator:
            break
    assert loader._data_loader is original and not guard.iterator_active
    with pytest.raises(StopIteration):
        next(iterator)


def test_iterator_close_failure_restores_loader_and_preserves_body_error(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    original = loader._data_loader
    body_error, close_error = ValueError('original training failure'), OSError('private close detail')
    with pytest.raises(ValueError) as caught:
        with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
            iterator = iter(guard)
            next(iterator)
            loader.close_error = close_error
            raise body_error
    assert caught.value is body_error
    assert body_error.__notes__ == ['OpenPI guard cleanup also failed: OSError']
    assert loader._data_loader is original and guard.closed and guard.loader is None
    with pytest.raises(StopIteration):
        next(iterator)


def test_normal_context_iterator_close_failure_propagates_same_error(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    close_error = OSError('close failure')
    with pytest.raises(OSError) as caught:
        with integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256) as guard:
            iterator = iter(guard)
            next(iterator)
            loader.close_error = close_error
    assert caught.value is close_error and guard.closed


def test_external_loader_hook_is_not_clobbered(tmp_path, controls):
    make, _root = controls
    path, contract = reference(tmp_path, make)
    loader = make()
    guard = integration.OpenpiTrainingGuard(loader, contract=path, criterion_sha256=contract.criterion_sha256)
    wrapper = loader._data_loader
    replacement = object()
    loader._data_loader = replacement
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        guard.close()
    assert loader._data_loader is replacement
    loader._data_loader = wrapper
    guard.close()


def test_missing_native_module_has_actionable_error():
    with pytest.raises(InputError, match='supported original OpenPI'):
        integration._native(object())


@pytest.mark.parametrize('case', ['post_delivery', 'unpaired', 'reorder', 'duplicate', 'dtype', 'later_hook'])
def test_actual_torch_model_pre_hook_refuses_unsupported_or_changed_target(tmp_path, controls, case):
    torch = pytest.importorskip('torch', reason='Optional real Torch hook controls; native CPU matrix also qualifies these cases')
    make, _root = controls
    loader = make()
    path = tmp_path / 'model-contract.json'
    contract = integration.capture_openpi_reference(loader, output=path, intent={'generated': 'model-boundary control'}, batches=2)

    class Consumer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, observation, actions):
            self.calls += 1
            return actions.sum()

    model, later = Consumer(), None
    try:
        with pytest.raises(TargetContractViolation) as caught:
            with integration.OpenpiTrainingGuard(make(), contract=path, criterion_sha256=contract.criterion_sha256,
                                                  batches=2, model=model) as guard:
                if case == 'unpaired':
                    model(None, torch.zeros((2, 2, 3)))
                else:
                    iterator = iter(guard)
                    first = next(iterator)
                    actual = torch.as_tensor(first[1]).to(torch.float32)
                    if case == 'post_delivery':
                        actual = actual + .5
                    elif case == 'reorder':
                        actual = actual.flip(0)
                    elif case == 'dtype':
                        actual = actual.to(torch.float64)
                    elif case == 'duplicate':
                        next(iterator)
                        model(first[0], actual)
                    elif case == 'later_hook':
                        later = model.register_forward_pre_hook(lambda _model, arguments: arguments)
                    model(first[0], actual)
        assert model.calls == (1 if case == 'duplicate' else 0)
        assert caught.value.report['status'] == ('unknown' if case in ('unpaired', 'later_hook') else 'error')
        assert guard.closed and guard.model is None and not hasattr(model, '_roboclue_openpi_guard')
    finally:
        if later is not None:
            later.remove()
