"""Generated Python-call mechanics, not native processor or robot execution."""
import hashlib
import json
import sys
from collections import deque

import numpy as np
import pytest

from policy_doctor import command_capture as cc
from policy_doctor.artifacts import InputError


class Policy:
    def __init__(self):
        self.queue = deque([np.array([[1., 2.]], dtype=np.float32)])
    def select_action(self, observation):
        return self.queue.popleft()


class Unnormalize:
    def __init__(self):
        self.norm_map = {'ACTION': 'MEAN_STD'}
        self.eps = 1e-8
        self._tensor_stats = {'action': {'mean': np.array([3., 5.], np.float32), 'std': np.array([2., 4.], np.float32)}}
    def _apply_transform(self, tensor, key, feature_type, inverse=False):
        stats = self._tensor_stats[key]
        mean, std = stats['mean'], stats['std']
        if getattr(self, 'replace_stats', False):
            self._tensor_stats = {'action': {'mean': np.array([100., 100.], np.float32),
                                            'std': np.array([1., 1.], np.float32)}}
        return tensor * std + mean
    def __call__(self, transition):
        return {'action': self._apply_transform(transition['action'], 'action', 'ACTION', inverse=True)}


class Device:
    device = 'cpu'
    float_dtype = None
    def __call__(self, transition):
        return transition.copy()


class Pipeline:
    def __init__(self):
        self.steps = [Unnormalize(), Device()]
        self.before_step_hooks = []
        self.after_step_hooks = []
    def __call__(self, data):
        transition = {'action': data}
        for step in self.steps:
            transition = step(transition)
        return transition['action']


class Engine:
    def __init__(self):
        self._policy, self._postprocessor = Policy(), Pipeline()
        self.allowed = True
        self.acks = 0
    def get_action(self, observation):
        action = self._policy.select_action(observation)
        action = self._postprocessor(action)
        return action.squeeze(0)
    def record_dispatch(self):
        self.acks += 1


class Interpolator:
    def __init__(self):
        self.multiplier = 1
        self._prev, self._buffer, self._idx, self._emitted_policy_action = None, [], 0, False
    def add(self, action):
        self._prev = action.copy()
        self._buffer = [action.copy()]
        self._idx = 0
    def get(self):
        action = self._buffer[self._idx]
        self._idx += 1
        self._emitted_policy_action = True
        return action


def tick(engine, interpolator, observation=None, *, send):
    if not engine.allowed:
        return None
    action = engine.get_action(observation)
    interpolator.add(action)
    target = interpolator.get()
    send(target)
    engine.record_dispatch()
    return target


@pytest.fixture
def mechanical(monkeypatch):
    def definition(engine, interpolator, consumer):
        sources = {'generated.py': b'generated mechanics only'}
        return {'functions': {'tick': tick, 'engine': Engine.get_action, 'select': Policy.select_action,
                             'pipeline': Pipeline.__call__, 'step0': Unnormalize.__call__,
                             'step1': Device.__call__, 'transform': Unnormalize._apply_transform,
                             'add': Interpolator.add, 'get': Interpolator.get,
                             'ack': Engine.record_dispatch, 'consumer': consumer.__func__ if hasattr(consumer, '__func__') else consumer},
                'sources': sources, 'paths': {}, 'source': {'repository': 'generated-control', 'revision': 'test'},
                'consumer_argument': 'action', 'consumer_owner': getattr(consumer, '__self__', None),
                'consumer_owner_argument': consumer.__func__.__code__.co_varnames[0] if hasattr(consumer, '__func__') else None,
                'pipeline_supported': True, 'converters_supported': True}
    monkeypatch.setattr(cc, '_native_definition', definition)


def load(path):
    data = json.loads((path / 'capture.json').read_text())
    assert json.loads((path / 'READY.json').read_text())['manifest_sha256'] == hashlib.sha256((path / 'capture.json').read_bytes()).hexdigest()
    with np.load(path / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return data, arrays


def context(tmp_path, consumer, **kwargs):
    engine, interpolation = Engine(), Interpolator()
    observer = cc.capture_command_path(engine, interpolation, consumer, tmp_path / 'capture',
                                      action_names=['left.pos', 'right.pos'], consumer_namespace='generated-recipient', **kwargs)
    return engine, interpolation, observer


def test_actual_call_chain_and_cloned_interpolation(mechanical, tmp_path):
    received = []
    def consumer(action):
        received.append(action.copy())
        return action / 2
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        result = tick(engine, interpolation, {}, send=consumer)
    assert observer.receipt['published']
    data, arrays = load(tmp_path / 'capture')
    row = data['ticks'][0]
    np.testing.assert_array_equal(result, [5., 13.])
    np.testing.assert_array_equal(arrays[row['selected']], [[1., 2.]])
    np.testing.assert_array_equal(arrays[row['requested']], result)
    np.testing.assert_array_equal(arrays[row['returned']], result / 2)
    assert row['outcome'] == 'sent' and row['consumer_completed'] is True and row['ack_count'] == 1
    assert row['bindings']['relations']['selected_to_post_input'] is True
    assert row['bindings']['relations']['engine_to_add_input'] is True
    assert row['bindings']['relations']['get_return_to_consumer'] is True
    assert row['bindings']['tokens']['engine_output'] != row['bindings']['tokens']['get_output']
    assert row['postprocessor']['steps'][0]['transform']['stats']['mean'] in arrays
    assert row['postprocessor']['steps'][0]['transform']['inverse'] is True
    assert engine.acks == 1 and sys.getprofile() is None
    assert observer._engine is None and observer._frames == {} and not observer._arrays


def test_consumer_none_return_is_success(mechanical, tmp_path):
    def consumer(action):
        return None
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        tick(engine, interpolation, {}, send=consumer)
    row = load(tmp_path / 'capture')[0]['ticks'][0]
    assert row['outcome'] == 'sent' and row['consumer_completed'] is True
    assert row['returned'] is None and row['ack_count'] == 1


def test_caught_consumer_exception_is_not_send_success(mechanical, tmp_path):
    failure = RuntimeError('original failure')
    def consumer(action):
        raise failure
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        with pytest.raises(RuntimeError) as exc:
            tick(engine, interpolation, {}, send=consumer)
        assert exc.value is failure
    assert engine.acks == 0
    row = load(tmp_path / 'capture')[0]['ticks'][0]
    assert row['outcome'] == 'error' and not row['consumer_completed'] and row['ack_count'] == 0


def test_uncaught_exception_no_partial_publication(mechanical, tmp_path):
    def consumer(action):
        raise ValueError('preserved')
    engine, interpolation, observer = context(tmp_path, consumer)
    with pytest.raises(ValueError, match='preserved'), observer:
        tick(engine, interpolation, {}, send=consumer)
    assert not observer.receipt['published'] and not (tmp_path / 'capture').exists()
    assert sys.getprofile() is None


def test_denied_tick_does_not_invent_consumer(mechanical, tmp_path):
    def consumer(action):
        raise AssertionError('must not run')
    engine, interpolation, observer = context(tmp_path, consumer)
    engine.allowed = False
    with observer:
        assert tick(engine, interpolation, {}, send=consumer) is None
    row = load(tmp_path / 'capture')[0]['ticks'][0]
    assert row['outcome'] == 'not_sent' and row['requested'] is None and row['ack_count'] == 0


def test_equal_clone_in_alternative_callback_not_associated(mechanical, tmp_path):
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        tick(engine, interpolation, {}, send=lambda action: consumer(action.copy()))
    assert not observer.receipt['published']


@pytest.mark.parametrize('names,namespace', [([], 'x'), (['a', 'a'], 'x'), ([True], 'x'), (['a'], ''), (['a'], True)])
def test_strict_explicit_contract(tmp_path, names, namespace):
    with pytest.raises(InputError):
        cc.capture_command_path(None, None, lambda action: action, tmp_path / 'capture',
                                action_names=names, consumer_namespace=namespace)


def test_existing_profiler_preserved(mechanical, tmp_path):
    def consumer(action):
        return action
    _, _, observer = context(tmp_path, consumer)
    def hook(*args):
        return None
    sys.setprofile(hook)
    try:
        with pytest.raises(InputError, match='profiler'):
            with observer:
                pass
        assert sys.getprofile() is hook
    finally:
        sys.setprofile(None)


def test_profiler_disable_restore_gap_refuses(mechanical, tmp_path):
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        hook = sys.getprofile()
        sys.setprofile(None)
        sys.setprofile(hook)
        tick(engine, interpolation, {}, send=consumer)
    assert not observer.receipt['published'] and sys.getprofile() is None


def test_source_mismatch_refuses_before_instrumentation(tmp_path):
    with pytest.raises(InputError, match='native|supported'):
        with cc.capture_command_path(Engine(), Interpolator(), lambda action: action, tmp_path / 'capture',
                                     action_names=['a', 'b'], consumer_namespace='test'):
            pass
    assert sys.getprofile() is None


def test_bound_method_consumer_and_snapshot_before_mutation(mechanical, tmp_path):
    class Consumer:
        def send(this, action):
            return action.copy()
    recipient = Consumer()
    engine, interpolation, observer = context(tmp_path, recipient.send)
    with observer:
        value = tick(engine, interpolation, {}, send=recipient.send)
        value[:] = 999
    row, arrays = load(tmp_path / 'capture')
    np.testing.assert_array_equal(arrays[row['ticks'][0]['requested']], [5., 13.])
    assert observer._consumer is None


@pytest.mark.parametrize('value', [np.ones((2, 2), np.float32), np.ones((1, 2), np.float64),
                                 np.ones((1, 3), np.float32), np.array([[np.nan, 0.]], np.float32)])
def test_unsupported_payload_does_not_alter_native_return(mechanical, tmp_path, value):
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    engine._policy.queue = deque([value])
    with observer:
        try:
            result = tick(engine, interpolation, {}, send=consumer)
        except ValueError:
            # Original generated engine squeeze can independently reject batch2.
            pass
        else:
            assert result.shape == value.squeeze(0).shape
    assert not observer.receipt['published'] and not (tmp_path / 'capture').exists()
    assert sys.getprofile() is None


def test_tick_budget_is_nonintrusive(mechanical, tmp_path, monkeypatch):
    monkeypatch.setattr(cc, 'MAX_TICKS', 1)
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        first = tick(engine, interpolation, {}, send=consumer)
        engine._policy.queue.append(np.ones((1, 2), np.float32))
        second = tick(engine, interpolation, {}, send=consumer)
    assert first is not None and second is not None and engine.acks == 2
    assert not observer.receipt['published']


def test_value_budget_checked_before_copy(mechanical, tmp_path, monkeypatch):
    monkeypatch.setattr(cc, 'MAX_VALUES', 1)
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    with observer:
        result = tick(engine, interpolation, {}, send=consumer)
    assert result is not None and not observer.receipt['published']
    assert 'bounds' in observer.receipt['reason']


def test_destination_no_overwrite_or_symlink(mechanical, tmp_path):
    def consumer(action):
        return action
    destination = tmp_path / 'capture'
    destination.mkdir()
    (destination / 'original').write_text('preserve')
    _, _, observer = context(tmp_path, consumer)
    with pytest.raises(InputError, match='exists'):
        with observer:
            pass
    assert (destination / 'original').read_text() == 'preserve'
    link = tmp_path / 'alias'
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        with cc.capture_command_path(Engine(), Interpolator(), consumer, link / 'new',
                                     action_names=['a', 'b'], consumer_namespace='test'):
            pass


def test_replacement_profiler_preserved(mechanical, tmp_path):
    def consumer(action):
        return action
    def replacement(*args):
        return None
    _, _, observer = context(tmp_path, consumer)
    try:
        with observer:
            sys.setprofile(replacement)
        assert sys.getprofile() is replacement
        assert not observer.receipt['published']
    finally:
        sys.setprofile(None)


def test_inference_tensor_view_witness_and_equal_clone(tmp_path):
    torch = pytest.importorskip('torch')
    observer = cc.capture_command_path(None, None, lambda action: action, tmp_path / 'unused',
                                      action_names=['a', 'b'], consumer_namespace='generated')
    with torch.inference_mode():
        post = torch.tensor([[1., 2.]], dtype=torch.float32)
        engine = post.squeeze(0).cpu()
        witness = observer._view_pair(post, engine)
        assert witness['same_live_storage'] is True
        assert witness['post']['base_token'] == witness['engine']['base_token']
        assert witness['post']['shape'] == [1, 2] and witness['engine']['shape'] == [2]
        assert witness['engine']['stride'] == [witness['post']['stride'][1]]
        assert observer._view_pair(post, engine.clone())['same_live_storage'] is False
    observer._clear()
    assert not observer._tokens._refs


@pytest.mark.parametrize('change', ['pipeline', 'step', 'converter', 'order'])
def test_runtime_instance_converter_or_order_drift_refuses(mechanical, tmp_path, change):
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    engine._ordered_action_keys = ['left.pos', 'right.pos']
    engine._postprocessor.to_transition = lambda value: value
    engine._postprocessor.to_output = lambda value: value
    with observer:
        if change == 'pipeline':
            engine._postprocessor = Pipeline()
        elif change == 'step':
            engine._postprocessor.steps[0] = Unnormalize()
        elif change == 'converter':
            engine._postprocessor.to_output = lambda value: value.copy()
        else:
            engine._ordered_action_keys.reverse()
        tick(engine, interpolation, {}, send=consumer)
    assert not observer.receipt['published']
    assert not (tmp_path / 'capture').exists()


def test_used_statistics_come_from_actual_transform_locals(mechanical, tmp_path):
    def consumer(action):
        return action
    engine, interpolation, observer = context(tmp_path, consumer)
    engine._postprocessor.steps[0].replace_stats = True
    with observer:
        tick(engine, interpolation, {}, send=consumer)
    data, arrays = load(tmp_path / 'capture')
    transform = data['ticks'][0]['postprocessor']['steps'][0]['transform']
    np.testing.assert_array_equal(arrays[transform['stats']['mean']], [3., 5.])
    np.testing.assert_array_equal(arrays[transform['stats']['std']], [2., 4.])
