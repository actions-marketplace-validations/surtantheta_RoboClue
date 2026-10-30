"""Local live-object witnesses: generated controls, never native/policy provenance."""
import gc
import weakref

import pytest

from policy_doctor.act_bindings import BindingTracker, OpaqueTokens


def event(mode='queue', queue=False, steps=2, maxlen=2):
    return {'event_id': 0, 'type': 'select', 'settings': {'n_action_steps': steps, 'chunk_size': 4,
            'temporal_ensemble_coeff': None if mode == 'queue' else .5},
            'pre_state': {'mode': mode, 'queue': ['pre_head'] if queue else [], 'queue_maxlen': maxlen},
            'predictions': [], 'selected': 'selected'}


def test_token_expiry_never_reuses_identity_generation(monkeypatch):
    class Value:
        pass
    registry = OpaqueTokens()
    monkeypatch.setattr(registry, '_identity', lambda value: 123)
    first = Value()
    old = registry.token(first)
    assert registry.matches(old, first)
    ref = weakref.ref(first)
    del first
    gc.collect()
    assert ref() is None
    assert not registry.live(old)
    second = Value()
    new = registry.token(second)
    assert new != old
    assert not registry.matches(old, second)
    registry.clear()
    assert not registry.live(new)


def test_token_cap_does_not_retain_objects():
    class Value:
        pass
    registry = OpaqueTokens(max_tokens=1)
    a, b = Value(), Value()
    assert registry.token(a) == 'o000000'
    assert registry.token(b) is None


def test_actual_head_identity_does_not_guess_earlier_query():
    class Value:
        pass
    value = Value()
    tracker = BindingTracker()
    e = event(queue=True)
    tracker.start(e, queue_head=value)
    result = tracker.finish(e, value)
    assert result['status'] == 'pass'
    assert result['basis'] == 'pre_queue_head_identity'
    assert result['selected_token'] == result['queue_head']['token']
    assert result['query'] is None
    assert result['history'] == 'unknown'


def test_equal_head_clone_is_not_same_object():
    torch = pytest.importorskip('torch')
    head = torch.zeros((1, 2))
    tracker = BindingTracker()
    e = event(queue=True)
    tracker.start(e, queue_head=head)
    result = tracker.finish(e, head.clone())
    assert result['status'] == 'unknown'
    assert result['relations']['selected_is_pre_head'] is False


def test_fresh_query_exact_view_with_native_full_batch():
    torch = pytest.importorskip('torch')
    prediction = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    tracker = BindingTracker()
    e = event()
    query = {'query_id': 0, 'array': 'prediction'}
    tracker.start(e)
    tracker.prediction(prediction, query)
    result = tracker.finish(e, prediction[:, 0])
    assert result['status'] == 'pass'
    assert result['basis'] == 'query_view_mapping'
    assert result['query_offset'] == 0
    assert result['query_view']['shape'] == [2, 4, 3]
    assert result['selected_view']['shape'] == [2, 3]
    assert result['query_view']['base_token'] == result['selected_view']['base_token']
    assert result['history'] == 'unknown'


@pytest.mark.parametrize('kind', ['clone', 'wrong_offset', 'broadcast', 'overlap'])
def test_values_or_shared_base_without_unique_expected_mapping_are_unknown(kind):
    torch = pytest.importorskip('torch')
    root = torch.zeros(32)
    prediction = root[:8].reshape(1, 4, 2)
    if kind == 'clone':
        selected = prediction[:, 0].clone()
    elif kind == 'wrong_offset':
        selected = prediction[:, 1]
    elif kind == 'broadcast':
        prediction = root[:2].reshape(1, 1, 2).expand(1, 4, 2)
        selected = prediction[:, 0]
    else:
        prediction = root.as_strided((1, 4, 2), (8, 1, 1))
        selected = prediction[:, 0]
    tracker = BindingTracker()
    e = event()
    tracker.start(e)
    tracker.prediction(prediction, {'query_id': 0, 'array': 'prediction'})
    assert tracker.finish(e, selected)['status'] == 'unknown'


def test_live_base_expiry_does_not_join_equal_values():
    torch = pytest.importorskip('torch')
    tracker = BindingTracker()
    e = event()
    tracker.start(e)
    prediction = torch.zeros((1, 4, 2))
    root_ref = weakref.ref(prediction)
    tracker.prediction(prediction, {'query_id': 0, 'array': 'prediction'})
    del prediction
    gc.collect()
    assert root_ref() is None
    assert tracker.finish(e, torch.zeros((1, 2)))['status'] == 'unknown'


def test_maxlen_truncation_uses_actual_first_retained_offset():
    torch = pytest.importorskip('torch')
    prediction = torch.zeros((1, 4, 2))
    tracker = BindingTracker()
    e = event(steps=4, maxlen=2)
    tracker.start(e)
    tracker.prediction(prediction, {'query_id': 3, 'array': 'prediction'})
    result = tracker.finish(e, prediction[:, 2])
    assert result['status'] == 'pass'
    assert result['query_offset'] == 2


def test_noncontiguous_unique_compatible_view_supported():
    torch = pytest.importorskip('torch')
    prediction = torch.arange(24., dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    tracker = BindingTracker()
    e = event()
    tracker.start(e)
    tracker.prediction(prediction, {'query_id': 0, 'array': 'prediction'})
    assert tracker.finish(e, prediction[:, 0])['status'] == 'pass'


def test_ensemble_actual_argument_and_return_identity():
    torch = pytest.importorskip('torch')
    class Helper:
        pass
    helper = Helper()
    prediction = torch.zeros((1, 4, 2))
    result = torch.ones((1, 2))
    tracker = BindingTracker()
    e = event(mode='ensemble')
    tracker.start(e, helper=helper)
    tracker.prediction(prediction, {'query_id': 0, 'array': 'prediction'})
    tracker.update_start(helper, prediction, inside_select=True)
    tracker.update_return(result)
    witness = tracker.finish(e, result)
    assert witness['status'] == 'pass'
    assert witness['basis'] == 'ensemble_argument_return_identity'
    assert witness['update']['actions_token'] == witness['update']['prediction_token']
    assert witness['update']['return_token'] == witness['selected_token']
    assert witness['history'] == 'unknown'


@pytest.mark.parametrize('which', ['argument', 'return', 'helper', 'repeated'])
def test_equal_ensemble_clones_or_other_instance_or_repeat_not_qualified(which):
    torch = pytest.importorskip('torch')
    class Helper:
        pass
    helper, other = Helper(), Helper()
    prediction = torch.zeros((1, 4, 2))
    result = torch.zeros((1, 2))
    tracker = BindingTracker()
    e = event(mode='ensemble')
    tracker.start(e, helper=helper)
    tracker.prediction(prediction, {'query_id': 0, 'array': 'prediction'})
    tracker.update_start(other if which == 'helper' else helper,
                         prediction.clone() if which == 'argument' else prediction, inside_select=True)
    tracker.update_return(result)
    if which == 'repeated':
        tracker.update_start(helper, prediction, inside_select=True)
        tracker.update_return(result)
    assert tracker.finish(e, result.clone() if which == 'return' else result)['status'] == 'unknown'


def test_offpath_update_invalidates_qualified_local_results():
    class Value:
        pass
    value, helper = Value(), Value()
    tracker = BindingTracker()
    e = event(queue=True)
    tracker.start(e, queue_head=value)
    e['local_binding'] = tracker.finish(e, value)
    assert e['local_binding']['status'] == 'pass'
    tracker.update_start(helper, value, inside_select=False)
    tracker.finalize([e])
    assert e['local_binding']['status'] == 'unknown'
    assert tracker.scope()['off_path_update_calls'] == 1


def test_tracker_keeps_only_weak_references():
    torch = pytest.importorskip('torch')
    tracker = BindingTracker()
    e = event()
    tracker.start(e)
    prediction = torch.zeros((1, 4, 2))
    reference = weakref.ref(prediction)
    tracker.prediction(prediction, {'query_id': 0, 'array': 'prediction'})
    del prediction
    gc.collect()
    assert reference() is None
    tracker.clear()
