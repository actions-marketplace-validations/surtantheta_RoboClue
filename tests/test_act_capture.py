"""Observer mechanics on generated arrays/objects, NOT genuine native ACT execution."""
import hashlib
import json
import sys
import weakref
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from policy_doctor import act_capture as capture
from policy_doctor.artifacts import InputError


class GeneratedPolicy:
    def __init__(self, *, batch=1, ensemble=False, chunk=3):
        from types import SimpleNamespace
        self.config = SimpleNamespace(n_action_steps=1 if ensemble else 2, chunk_size=chunk,
                                      temporal_ensemble_coeff=.5 if ensemble else None)
        self.batch = batch
        self.calls = 0
        if ensemble:
            self.temporal_ensembler = SimpleNamespace(
                ensembled_actions=None, ensembled_actions_count=None,
                ensemble_weights=np.exp(-.5 * np.arange(chunk)),
                ensemble_weights_cumsum=np.cumsum(np.exp(-.5 * np.arange(chunk))))
        self.reset()

    def reset(self):
        if self.config.temporal_ensemble_coeff is None:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        else:
            self.temporal_ensembler.ensembled_actions = None
            self.temporal_ensembler.ensembled_actions_count = None

    def predict_action_chunk(self, batch):
        self.calls += 1
        return np.zeros((self.batch, self.config.chunk_size, 2)) + self.calls

    def select_action(self, batch):
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            helper = self.temporal_ensembler
            helper.ensembled_actions = actions[:, 1:].copy()
            helper.ensembled_actions_count = np.ones((actions.shape[1] - 1, 1), dtype=np.int64)
            return actions[:, 0]
        if not self._action_queue:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(1, 0, 2))
        return self._action_queue.popleft()


@pytest.fixture
def mechanical(monkeypatch):
    """Bypass ONLY native-source gate; saved source bytes visibly remain generated."""
    sources = {'modeling_act.py': b'# generated mechanics, not native ACT\n',
               'configuration_act.py': b'# generated mechanics configuration\n',
               'LICENSE.txt': b'Generated test fixture\n'}
    definition = {'select': GeneratedPolicy.select_action,
                  'predict': GeneratedPolicy.predict_action_chunk,
                  'reset': GeneratedPolicy.reset, 'bridges': set(), 'sources': sources,
                  'source': {'repository': 'generated-control', 'revision': None,
                             'modeling_sha256': hashlib.sha256(sources['modeling_act.py']).hexdigest(),
                             'config_sha256': hashlib.sha256(sources['configuration_act.py']).hexdigest()}}
    monkeypatch.setattr(capture, '_native_definition', lambda policy: definition)
    return definition


def document(path):
    data = json.loads((path / 'selection.json').read_text())
    assert json.loads((path / 'READY.json').read_text())['manifest_sha256'] == hashlib.sha256((path / 'selection.json').read_bytes()).hexdigest()
    with np.load(path / 'arrays.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return data, arrays


def test_queue_predict_only_when_empty_and_actual_reset(tmp_path, mechanical):
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        first = p.select_action({})
        second = p.select_action({})
        p.reset()
        third = p.select_action({})
    data, arrays = document(tmp_path / 'capture')
    assert observation.receipt['published'] is True
    assert observation.receipt['status'] == 'captured'
    assert p.calls == 2
    assert [e['type'] for e in data['events']] == ['select', 'select', 'reset', 'select']
    assert [len(e['predictions']) for e in data['events']] == [1, 0, 0, 1]
    assert data['events'][0]['predictions'][0]['query_id'] == 0
    assert data['events'][3]['predictions'][0]['query_id'] == 1
    assert len(data['events'][0]['post_state']['queue']) == 1
    assert data['events'][1]['post_state']['queue'] == []
    assert data['events'][2]['selected'] is None
    for e, value in zip([data['events'][0], data['events'][1], data['events'][3]], [first, second, third], strict=True):
        np.testing.assert_array_equal(arrays[e['selected']], value)
    assert data['source']['repository'] == 'generated-control'


def test_copies_actual_values_before_caller_mutation(tmp_path, mechanical):
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
        value = p.select_action({})
        value[:] = 777
    data, arrays = document(tmp_path / 'capture')
    np.testing.assert_array_equal(arrays[data['events'][0]['selected']], np.ones((1, 2)))


@pytest.mark.parametrize('chunk', [1, 3])
def test_ensemble_zero_and_empty_retained_not_missing(tmp_path, mechanical, chunk):
    p = GeneratedPolicy(ensemble=True, chunk=chunk)
    p.calls = -1
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
        value = p.select_action({})
        p.reset()
    data, arrays = document(tmp_path / 'capture')
    event = data['events'][0]
    assert event['pre_state']['aggregate'] is None
    assert event['post_state']['aggregate'] is not None
    assert arrays[event['post_state']['aggregate']].shape == (1, chunk - 1, 2)
    np.testing.assert_array_equal(arrays[event['selected']], value)
    assert not arrays[event['selected']].any()
    assert data['events'][1]['post_state']['aggregate'] is None
    np.testing.assert_array_equal(arrays[event['pre_state']['weights']], p.temporal_ensembler.ensemble_weights)


def test_explicit_rows_preserve_full_return(tmp_path, mechanical):
    p = GeneratedPolicy(batch=4096)
    with capture.capture_act_selection(p, tmp_path / 'capture', env_indices=[4095, 0], allow_tensor_snapshot=True):
        value = p.select_action({})
    data, arrays = document(tmp_path / 'capture')
    assert value.shape == (4096, 2)
    assert arrays[data['events'][0]['selected']].shape == (2, 2)
    assert data['environment_selection'] == {'source_num_envs': 4096, 'env_indices': [4095, 0], 'saved_rows': 2, 'omitted_envs': 4094}


@pytest.mark.parametrize('kwargs', [dict(env_indices=[]), dict(env_indices=[True]), dict(env_indices=[0, 0]), dict(env_indices=[-1]), dict(env_indices=list(range(33))), dict(allow_tensor_snapshot=1), dict(domain='executed_action')])
def test_strict_arguments(tmp_path, mechanical, kwargs):
    with pytest.raises(InputError):
        capture.capture_act_selection(GeneratedPolicy(), tmp_path / 'capture', **kwargs)


def test_unknown_unselected_or_exception_publishes_nothing(tmp_path, mechanical):
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'none', allow_tensor_snapshot=True) as observation:
        pass
    assert observation.receipt['published'] is False
    sentinel = RuntimeError('same native exception')
    with pytest.raises(RuntimeError) as failure:
        with capture.capture_act_selection(p, tmp_path / 'exception', allow_tensor_snapshot=True) as observation:
            p.select_action({})
            raise sentinel
    assert failure.value is sentinel
    assert observation.receipt['published'] is False
    assert not (tmp_path / 'exception').exists()
    assert sys.getprofile() is None


def test_existing_and_replacement_profiler_preserved(tmp_path, mechanical):
    def other(*args):
        pass
    p = GeneratedPolicy()
    sys.setprofile(other)
    try:
        with pytest.raises(InputError):
            with capture.capture_act_selection(p, tmp_path / 'busy', allow_tensor_snapshot=True):
                p.select_action({})
        assert sys.getprofile() is other
    finally:
        sys.setprofile(None)
    with capture.capture_act_selection(p, tmp_path / 'replaced', allow_tensor_snapshot=True) as observation:
        p.select_action({})
        sys.setprofile(other)
    try:
        assert sys.getprofile() is other
        assert not observation.receipt['published']
    finally:
        sys.setprofile(None)


def test_nested_and_reused_context_refused(tmp_path, mechanical):
    p = GeneratedPolicy()
    context = capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True)
    with context:
        with pytest.raises(InputError):
            with capture.capture_act_selection(p, tmp_path / 'nested', allow_tensor_snapshot=True):
                pass
        p.select_action({})
    with pytest.raises(InputError):
        with context:
            pass


def test_existing_destination_and_symlink_parent_refused(tmp_path, mechanical):
    p = GeneratedPolicy()
    target = tmp_path / 'exists'
    target.mkdir()
    with pytest.raises(InputError):
        with capture.capture_act_selection(p, target, allow_tensor_snapshot=True):
            pass
    link = tmp_path / 'linked'
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(InputError):
        with capture.capture_act_selection(p, link / 'out', allow_tensor_snapshot=True):
            pass


def test_budget_refuses_before_host_copy(tmp_path, mechanical, monkeypatch):
    p = GeneratedPolicy()
    monkeypatch.setattr(capture, 'MAX_VALUES', 1)
    calls = []
    original = capture._host_copy
    monkeypatch.setattr(capture, '_host_copy', lambda *args: calls.append(args) or original(*args))
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        p.select_action({})
    assert not calls
    assert not observation.receipt['published']
    assert not (tmp_path / 'capture').exists()


def test_events_cap_and_source_drift_no_partial_capture(tmp_path, mechanical, monkeypatch):
    p = GeneratedPolicy()
    monkeypatch.setattr(capture, 'MAX_EVENTS', 1)
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        p.select_action({})
        p.select_action({})
    assert not observation.receipt['published']
    assert not (tmp_path / 'capture').exists()


def test_release_policy_reference(tmp_path, mechanical):
    p = GeneratedPolicy()
    reference = weakref.ref(p)
    context = capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True)
    with context:
        p.select_action({})
    del p
    import gc
    gc.collect()
    assert reference() is None


def test_generated_object_cannot_claim_native_source(tmp_path):
    with pytest.raises(InputError):
        with capture.capture_act_selection(GeneratedPolicy(), tmp_path / 'capture', allow_tensor_snapshot=True):
            pass


def test_decorated_predict_ancestry_is_observed(tmp_path, mechanical, monkeypatch):
    from functools import wraps
    original = GeneratedPolicy.predict_action_chunk

    def decorate(func):
        @wraps(func)
        def bridge(*args, **kwargs):
            return func(*args, **kwargs)
        return bridge

    wrapped = decorate(original)
    mechanical['bridges'] = {wrapped.__code__}
    monkeypatch.setattr(GeneratedPolicy, 'predict_action_chunk', wrapped)
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
        p.select_action({})
    data, _ = document(tmp_path / 'capture')
    assert len(data['events'][0]['predictions']) == 1


def test_other_policy_and_direct_predictions_are_not_selected_queries(tmp_path, mechanical):
    p, other = GeneratedPolicy(), GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
        other.select_action({})
        p.predict_action_chunk({})
        value = p.select_action({})
    data, arrays = document(tmp_path / 'capture')
    assert len(data['events']) == 1
    assert data['events'][0]['predictions'][0]['query_id'] == 0
    np.testing.assert_array_equal(arrays[data['events'][0]['selected']], value)


def test_caught_native_error_remains_same_and_discards_capture(tmp_path, mechanical, monkeypatch):
    sentinel = RuntimeError('native prediction failure')

    def fail(self, batch):
        raise sentinel

    mechanical['predict'] = fail
    monkeypatch.setattr(GeneratedPolicy, 'predict_action_chunk', fail)
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        with pytest.raises(RuntimeError) as error:
            p.select_action({})
        assert error.value is sentinel
    assert not observation.receipt['published']
    assert sys.getprofile() is None
    assert not (tmp_path / 'capture').exists()


def test_caught_reset_exception_is_not_a_normal_none_return(tmp_path, mechanical, monkeypatch):
    p = GeneratedPolicy()
    sentinel = RuntimeError('native reset failure')

    def fail(self):
        raise sentinel

    mechanical['reset'] = fail
    monkeypatch.setattr(GeneratedPolicy, 'reset', fail)
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        with pytest.raises(RuntimeError) as error:
            p.reset()
        assert error.value is sentinel
    assert not observation.receipt['published']


def test_source_drift_refuses_publication(tmp_path, mechanical):
    source = tmp_path / 'original.py'
    source.write_bytes(mechanical['sources']['modeling_act.py'])
    mechanical['paths'] = {'modeling_act.py': source}
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        p.select_action({})
        source.write_bytes(b'# source changed\n')
    assert not observation.receipt['published']
    assert 'source changed' in observation.receipt['reason']
    assert not (tmp_path / 'capture').exists()


def test_publication_race_preserves_existing_directory(tmp_path, mechanical):
    p = GeneratedPolicy()
    output = tmp_path / 'capture'
    with capture.capture_act_selection(p, output, allow_tensor_snapshot=True) as observation:
        p.select_action({})
        output.mkdir(mode=0o700)
        (output / 'mine').write_bytes(b'keep')
    assert not observation.receipt['published']
    assert (output / 'mine').read_bytes() == b'keep'
    assert sorted(p.name for p in output.iterdir()) == ['mine']


def test_large_default_batch_refused_without_host_copy(tmp_path, mechanical, monkeypatch):
    p = GeneratedPolicy(batch=33)
    copies = []
    monkeypatch.setattr(capture, '_host_copy', lambda *args: copies.append(args))
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        value = p.select_action({})
    assert value.shape == (33, 2)
    assert not copies
    assert not observation.receipt['published']
    assert 'env_indices' in observation.receipt['reason']


def test_per_event_settings_are_copied_and_reset_only_has_unknown_rows(tmp_path, mechanical):
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
        p.reset()
        p.config.n_action_steps = 1
        p.reset()
    data, arrays = document(tmp_path / 'capture')
    assert arrays == {}
    assert [e['settings']['n_action_steps'] for e in data['events']] == [2, 1]
    assert data['environment_selection']['source_num_envs'] is None
    assert data['environment_selection']['env_indices'] is None


def test_manifest_and_array_limit_refuse_no_partial_output(tmp_path, mechanical, monkeypatch):
    p = GeneratedPolicy()
    monkeypatch.setattr(capture, 'MAX_ARRAYS', 1)
    with capture.capture_act_selection(p, tmp_path / 'arrays', allow_tensor_snapshot=True) as observation:
        p.select_action({})
    assert not observation.receipt['published']
    monkeypatch.setattr(capture, 'MAX_ARRAYS', 4096)
    monkeypatch.setattr(capture, 'MAX_MANIFEST', 1)
    p.reset()
    with capture.capture_act_selection(p, tmp_path / 'manifest', allow_tensor_snapshot=True) as observation:
        p.select_action({})
    assert not observation.receipt['published']
    assert not (tmp_path / 'arrays').exists()
    assert not (tmp_path / 'manifest').exists()


def test_tensor_optin_refusal_precedes_native_call(tmp_path, mechanical):
    p = GeneratedPolicy()
    with pytest.raises(InputError, match='allow_tensor_snapshot'):
        with capture.capture_act_selection(p, tmp_path / 'capture'):
            p.select_action({})
    assert p.calls == 0
    assert sys.getprofile() is None


def test_real_cpu_tensor_rows_selected_before_numpy_copy(tmp_path, mechanical, monkeypatch):
    torch = pytest.importorskip('torch')
    context = capture.capture_act_selection(GeneratedPolicy(), tmp_path / 'capture', env_indices=[4095, 0], allow_tensor_snapshot=True)
    context._definition = mechanical
    seen = []
    original = torch.Tensor.numpy

    def numpy_array(self, *args, **kwargs):
        seen.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'numpy', numpy_array)
    data = torch.arange(4096 * 3 * 2, dtype=torch.float32).reshape(4096, 3, 2)
    reference = data.clone()
    key = context._snapshot(data, 'selected_rows', rank=3, batch=True)
    assert seen == [(2, 3, 2)]
    np.testing.assert_array_equal(context._arrays[key], original(data[[4095, 0]]))
    assert torch.equal(data, reference)


def test_meta_tensor_refused_before_copy(tmp_path, mechanical, monkeypatch):
    torch = pytest.importorskip('torch')
    context = capture.capture_act_selection(GeneratedPolicy(), tmp_path / 'capture', allow_tensor_snapshot=True)
    context._definition = mechanical
    copies = []
    monkeypatch.setattr(capture, '_host_copy', lambda *args: copies.append(args))
    with pytest.raises(InputError, match='CPU'):
        context._snapshot(torch.zeros((1, 2), device='meta'), 'unsupported', rank=2, batch=True)
    assert not copies


def test_queue_length_is_bounded_before_any_copy(tmp_path, mechanical, monkeypatch):
    p = GeneratedPolicy()
    p.config.n_action_steps = p.config.chunk_size = 256
    p._action_queue = deque([np.zeros((1, 2)) for _ in range(257)], maxlen=300)
    copies = []
    monkeypatch.setattr(capture, '_host_copy', lambda *args: copies.append(args))
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        value = p.select_action({})
    assert value.shape == (1, 2)
    assert not copies
    assert not observation.receipt['published']


def test_profiler_disable_restore_gap_is_not_complete_coverage(tmp_path, mechanical):
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        p.select_action({})
        saved = sys.getprofile()
        sys.setprofile(None)
        p.reset()
        sys.setprofile(saved)
        p.select_action({})
    assert not observation.receipt['published']
    assert not (tmp_path / 'capture').exists()
    assert sys.getprofile() is None


def test_saved_coverage_is_conditional_not_state_mutation_attestation(tmp_path, mechanical):
    p = GeneratedPolicy()
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
        p.select_action({})
    data, _ = document(tmp_path / 'capture')
    assert data['coverage'] == {'target': 'one_policy_on_owner_thread',
                                'complete_target_call_history': True,
                                'external_state_mutations': 'untracked'}


def test_chunk_above_reader_limit_refused_before_native_call(tmp_path, mechanical):
    p = GeneratedPolicy(chunk=257)
    with pytest.raises(InputError, match='chunk'):
        with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True):
            p.select_action({})
    assert p.calls == 0


@pytest.mark.parametrize('rank,integer,batch,shape', [(3, False, True, (1, 257, 2)),
                                                     (1, False, False, (257,)),
                                                     (2, True, False, (257, 1))])
def test_snapshot_state_dimensions_match_reader_cap(tmp_path, mechanical, monkeypatch, rank, integer, batch, shape):
    context = capture.capture_act_selection(GeneratedPolicy(), tmp_path / 'capture', allow_tensor_snapshot=True)
    context._definition = mechanical
    copied = []
    monkeypatch.setattr(capture, '_host_copy', lambda *args: copied.append(args))
    with pytest.raises(InputError):
        context._snapshot(np.zeros(shape, dtype=np.int64 if integer else np.float32),
                          'oversized', rank=rank, integer=integer, batch=batch)
    assert not copied


def test_extended_float_refused_before_copy(tmp_path, mechanical, monkeypatch):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip('Platform has no extended floating dtype')
    context = capture.capture_act_selection(GeneratedPolicy(), tmp_path / 'capture', allow_tensor_snapshot=True)
    context._definition = mechanical
    copied = []
    monkeypatch.setattr(capture, '_host_copy', lambda *args: copied.append(args))
    with pytest.raises(InputError, match='dtype'):
        context._snapshot(np.zeros((1, 2), dtype=np.longdouble), 'extended', rank=2, batch=True)
    assert not copied


def test_ensemble_identity_guard_rejects_custom_helper_and_replaced_methods():
    """Generated source-identity mechanics only; no pinned-native execution claim."""
    from types import MethodType, SimpleNamespace

    class ACTTemporalEnsembler:
        def update(self, actions):
            return actions

        def reset(self):
            pass

    for name in ('update', 'reset'):
        function = getattr(ACTTemporalEnsembler, name)
        function.__module__ = 'lerobot.policies.act.modeling_act'
        function.__qualname__ = 'ACTTemporalEnsembler.' + name
    module = SimpleNamespace(ACTTemporalEnsembler=ACTTemporalEnsembler,
                             __name__='lerobot.policies.act.modeling_act')
    helper = ACTTemporalEnsembler()
    policy = SimpleNamespace(temporal_ensembler=helper)
    path = Path(__file__)
    capture._ensemble_definition(policy, module, path)
    policy.temporal_ensembler = SimpleNamespace(update=helper.update, reset=helper.reset)
    with pytest.raises(InputError, match='exact original'):
        capture._ensemble_definition(policy, module, path)
    policy.temporal_ensembler = helper
    helper.update = MethodType(lambda self, actions: actions, helper)
    with pytest.raises(InputError, match='Replaced'):
        capture._ensemble_definition(policy, module, path)
    del helper.update
    helper.reset = lambda: None
    with pytest.raises(InputError, match='Replaced'):
        capture._ensemble_definition(policy, module, path)


class GeneratedTensorHelper:
    """Generated Torch mechanism, not original native ACT implementation."""
    def __init__(self, chunk):
        import torch
        self.ensembled_actions = self.ensembled_actions_count = None
        self.ensemble_weights = torch.ones(chunk)
        self.ensemble_weights_cumsum = self.ensemble_weights.cumsum(0)

    def update(self, actions):
        import torch
        self.ensembled_actions = actions[:, 1:].clone()
        self.ensembled_actions_count = torch.ones((actions.shape[1] - 1, 1), dtype=torch.int64)
        return actions[:, 0].clone()


class GeneratedTensorPolicy(GeneratedPolicy):
    def __init__(self, *, ensemble=False):
        super().__init__(ensemble=False)
        if ensemble:
            self.config.n_action_steps = 1
            self.config.temporal_ensemble_coeff = 0.
            self.temporal_ensembler = GeneratedTensorHelper(self.config.chunk_size)
        self.reset()

    def predict_action_chunk(self, batch):
        import torch
        self.calls += 1
        return torch.zeros((1, self.config.chunk_size, 2))

    def select_action(self, batch):
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)
        if not self._action_queue:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


@pytest.fixture
def tensor_mechanical(mechanical):
    pytest.importorskip('torch')
    mechanical.update(select=GeneratedTensorPolicy.select_action,
                      predict=GeneratedTensorPolicy.predict_action_chunk,
                      update=GeneratedTensorHelper.update, ensemble_class=GeneratedTensorHelper)
    return mechanical


def test_profiled_tensor_queue_fresh_and_midrun_bindings(tmp_path, tensor_mechanical):
    p = GeneratedTensorPolicy()
    with capture.capture_act_selection(p, tmp_path / 'begin', allow_tensor_snapshot=True) as observation:
        fresh = p.select_action({})
    assert observation.receipt['published']
    data, _ = document(tmp_path / 'begin')
    witness = data['events'][0]['local_binding']
    assert witness['status'] == 'pass'
    assert witness['basis'] == 'query_view_mapping'
    assert witness['query_offset'] == 0
    assert witness['query']['query_id'] == 0
    assert witness['history'] == 'unknown'
    assert data['binding_scope']['off_path_update_calls'] == 0
    actual_head = p._action_queue[0]
    with capture.capture_act_selection(p, tmp_path / 'midrun', allow_tensor_snapshot=True):
        retained = p.select_action({})
    assert retained is actual_head
    assert fresh is not retained
    data, _ = document(tmp_path / 'midrun')
    witness = data['events'][0]['local_binding']
    assert witness['status'] == 'pass'
    assert witness['basis'] == 'pre_queue_head_identity'
    assert witness['query'] is None
    assert witness['selected_token'] == witness['queue_head']['token']
    assert witness['history'] == 'unknown'


def test_profiled_tensor_update_argument_and_return_bindings(tmp_path, tensor_mechanical):
    p = GeneratedTensorPolicy(ensemble=True)
    with capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True) as observation:
        result = p.select_action({})
    assert observation.receipt['published']
    data, arrays = document(tmp_path / 'capture')
    witness = data['events'][0]['local_binding']
    assert witness['status'] == 'pass'
    assert witness['basis'] == 'ensemble_argument_return_identity'
    assert witness['relations']['actions_is_prediction'] is True
    assert witness['relations']['selected_is_update_return'] is True
    assert witness['update']['call_count'] == 1
    assert witness['selected_token'] == witness['update']['return_token']
    assert witness['query']['token'] == witness['update']['actions_token']
    np.testing.assert_array_equal(arrays[witness['selected_array']], result.numpy())


def test_profiled_offpath_target_update_unknown_but_other_instance_ignored(tmp_path, tensor_mechanical):
    import torch
    p = GeneratedTensorPolicy(ensemble=True)
    other = GeneratedTensorHelper(p.config.chunk_size)
    with capture.capture_act_selection(p, tmp_path / 'other', allow_tensor_snapshot=True):
        other.update(torch.zeros((1, 3, 2)))
        p.select_action({})
    data, _ = document(tmp_path / 'other')
    assert data['binding_scope']['off_path_update_calls'] == 0
    assert data['events'][0]['local_binding']['status'] == 'pass'
    with capture.capture_act_selection(p, tmp_path / 'offpath', allow_tensor_snapshot=True) as observation:
        p.select_action({})
        p.temporal_ensembler.update(torch.zeros((1, 3, 2)))
    assert observation.receipt['published']
    data, _ = document(tmp_path / 'offpath')
    assert data['binding_scope']['off_path_update_calls'] == 1
    assert data['events'][0]['local_binding']['status'] == 'unknown'


def test_profiled_bindings_release_policy_helper_and_prediction_refs(tmp_path, tensor_mechanical):
    p = GeneratedTensorPolicy(ensemble=True)
    policy_ref, helper_ref = weakref.ref(p), weakref.ref(p.temporal_ensembler)
    context = capture.capture_act_selection(p, tmp_path / 'capture', allow_tensor_snapshot=True)
    with context:
        result = p.select_action({})
    assert context.receipt['published']
    del p, result
    import gc
    gc.collect()
    assert policy_ref() is None and helper_ref() is None
    assert context._bindings._active is None
    assert context._bindings.tokens._refs == {}
