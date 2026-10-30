"""Generated saved-state controls for the explicit consumer cadence requirement."""
import hashlib
import json

import numpy as np
import pytest
from test_selection import capture as capture_fixture
from test_selection import rewrite_manifest

from policy_doctor import cadence, selection
from policy_doctor.artifacts import InputError


@pytest.fixture(name="capture")
def saved_packet(tmp_path, monkeypatch):
    return capture_fixture.__wrapped__(tmp_path, monkeypatch)


def packet(capture, sequence, *, name='capture', steps=4, shift=0.):
    arrays, events, queue = {}, [], []
    query_id = 0
    for index, operation in enumerate(sequence):
        def add(key, value, index=index):
            ref = f'e{index}_{key}'
            arrays[ref] = np.asarray(value, dtype=np.float64)
            return ref
        settings = {'n_action_steps': steps, 'chunk_size': steps, 'temporal_ensemble_coeff': None}
        def state(refs):
            return {'mode': 'queue', 'queue_maxlen': steps, 'queue': list(refs)}
        if operation == 'R':
            events.append({'event_id': index, 'type': 'reset', 'completion': 'normal',
                           'settings': settings, 'pre_state': state(queue), 'post_state': state([]),
                           'selected': None, 'predictions': []})
            queue = []
            continue
        predictions = []
        if operation == 'Q':
            # A generated external clear can start another query; call coverage is not state provenance.
            queue = []
            chunk = (np.arange(steps * 2).reshape(1, steps, 2) + shift + query_id * 100.)
            query = add('query', chunk)
            queue = [add(f'row{k}', chunk[:, k]) for k in range(steps)]
            predictions = [{'query_id': query_id, 'array': query}]
            query_id += 1
            pre = []
        else:
            # Explicit saved mid-run state, not an invented earlier query.
            if not queue:
                queue = [add('retained', [[shift + index, shift - index]])]
            pre = list(queue)
        selected, queue = queue[0], queue[1:]
        events.append({'event_id': index, 'type': 'select', 'completion': 'normal',
                       'settings': settings, 'pre_state': state(pre), 'post_state': state(queue),
                       'selected': selected, 'predictions': predictions})
    return capture(events, arrays, name)


def test_known_first_violation_has_original_native_evidence(capture):
    root = packet(capture, 'QSS')
    report = cadence.audit_query_cadence(root, max_query_gap=2)
    assert report['status'] == report['cadence_criterion']['status'] == 'error'
    first = report['cadence_criterion']['first_violation']
    assert (first['event_id'], first['selection_ordinal'], first['gap'], first['gap_lower_bound']) == (2, 2, 3, 3)
    assert (first['query_id'], first['query_event_id']) == (0, 0)
    manifest_hash = hashlib.sha256((root / 'selection.json').read_bytes()).hexdigest()
    assert any(e['sha256'] == manifest_hash and e['pointer'] == '/events/2/predictions' for e in first['evidence'])
    assert report['numeric_replay_status'] == 'pass'
    assert report['query_lineage_status'] == 'unknown'
    assert report['downstream_association']['status'] == 'unknown'


def test_supported_old_capture_passes_without_local_binding(capture):
    report = cadence.audit_query_cadence(packet(capture, 'QSQ', steps=2), max_query_gap=2)
    assert report['status'] == report['cadence_criterion']['status'] == 'pass'
    assert report['consumer_requirement'] == {'max_query_gap': 2, 'domain': 'policy_output', 'unit': 'selection_calls'}
    assert [e.get('gap') for e in report['events']] == [1, 2, 1]
    assert report['query_lineage_status'] == 'unknown'


@pytest.mark.parametrize('sequence,limit,status,event,gap,lower', [
    ('S', 2, 'unknown', None, None, None),
    ('SS', 2, 'error', 1, None, 3),
    ('S', 1, 'error', 0, None, 2),
    ('SQ', 2, 'unknown', None, None, None),
    ('SQSS', 2, 'error', 3, 3, 3),
    ('QSRS', 2, 'error', 3, 3, 3),
    ('SRS', 2, 'error', 2, None, 3),
    ('RR', 1, 'unknown', None, None, None),
    ('QRQ', 1, 'pass', None, None, None),
])
def test_capture_wide_counts_and_prefix_bounds(capture, sequence, limit, status, event, gap, lower):
    report = cadence.audit_query_cadence(packet(capture, sequence), max_query_gap=limit)
    assert report['cadence_criterion']['status'] == status
    first = report['cadence_criterion']['first_violation']
    if event is None:
        assert first is None
    else:
        assert (first['event_id'], first['gap'], first['gap_lower_bound']) == (event, gap, lower)
    assert report['cadence_criterion']['reset_count'] == sequence.count('R')


@pytest.mark.parametrize('coverage', [None, {}, {'target': 'other', 'complete_target_call_history': True},
    {'target': 'one_policy_on_owner_thread', 'complete_target_call_history': False},
    {'target': 'one_policy_on_owner_thread', 'complete_target_call_history': 1}])
def test_missing_or_incomplete_coverage_cannot_certify_fault(capture, coverage):
    root = packet(capture, 'QSS')
    rewrite_manifest(root, lambda m: m.update(coverage=coverage))
    report = cadence.audit_query_cadence(root, max_query_gap=1)
    assert report['cadence_criterion']['status'] == 'unknown'
    assert report['cadence_criterion']['first_violation'] is None
    assert report['coverage_supported'] is False


def test_unsupported_source_is_not_a_consumer_fault(capture, monkeypatch):
    root = packet(capture, 'QSS')
    monkeypatch.setattr(selection, 'SUPPORTED_SOURCE', {**selection.SUPPORTED_SOURCE, 'revision': 'unsupported'})
    report = cadence.audit_query_cadence(root, max_query_gap=1)
    assert report['cadence_criterion']['status'] == report['numeric_replay_status'] == 'unknown'
    assert report['cadence_criterion']['first_violation'] is None


def test_numeric_error_is_separate_from_cadence_error(capture):
    root = packet(capture, 'QSS')
    def corrupt(m):
        # Another valid-shaped native array differs from the predicted selected value.
        m['events'][0]['selected'] = m['events'][0]['post_state']['queue'][0]
    rewrite_manifest(root, corrupt)
    report = cadence.audit_query_cadence(root, max_query_gap=1)
    assert report['numeric_replay_status'] == report['status'] == 'error'
    assert report['cadence_criterion']['status'] == 'unknown'
    assert report['cadence_criterion']['first_violation'] is None


@pytest.mark.parametrize('value', [True, False, 0, -1, 1., '1', None, 1_000_001, 10**100])
def test_strict_requirement_rejected_before_intake(value):
    with pytest.raises(InputError, match='max_query_gap'):
        cadence.audit_query_cadence('/does/not/exist', max_query_gap=value)


def test_changed_values_do_not_change_requirement(capture):
    left = cadence.audit_query_cadence(packet(capture, 'QQ', name='a'), max_query_gap=1)
    right = cadence.audit_query_cadence(packet(capture, 'QQ', name='b', shift=91.), max_query_gap=1)
    assert left['cadence_criterion'] == right['cadence_criterion']
    assert left['status'] == 'pass'


def test_malformed_query_count_refused(capture):
    root = packet(capture, 'Q')
    rewrite_manifest(root, lambda m: m['events'][0]['predictions'].append(m['events'][0]['predictions'][0]))
    with pytest.raises(InputError, match='at most one query'):
        cadence.audit_query_cadence(root, max_query_gap=1)


def test_original_hashes_unchanged_and_single_load(capture, monkeypatch):
    root = packet(capture, 'QS')
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    original, calls = cadence._load, []
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(cadence, '_load', counted)
    assert cadence.audit_query_cadence(root, max_query_gap=2)['status'] == 'pass'
    assert len(calls) == 1
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_final_recheck_refuses_valid_recommit_during_analysis(capture, monkeypatch):
    root = packet(capture, 'Q')
    original = cadence._analyze
    def mutate(*args):
        result = original(*args)
        rewrite_manifest(root, lambda m: m.update(extra='changed during audit'))
        return result
    monkeypatch.setattr(cadence, '_analyze', mutate)
    with pytest.raises(InputError, match='changed'):
        cadence.audit_query_cadence(root, max_query_gap=1)


def test_tampered_payload_refused(capture):
    root = packet(capture, 'Q')
    (root / 'arrays.npz').write_bytes(b'changed')
    with pytest.raises(InputError, match='hash'):
        cadence.audit_query_cadence(root, max_query_gap=1)


def test_largest_event_report_stays_compact(capture):
    report = cadence.audit_query_cadence(packet(capture, 'Q' * 256, steps=1), max_query_gap=1)
    assert report['status'] == 'pass'
    assert len(report['events']) == 256
    assert len(json.dumps(report).encode()) <= 192 * 1024


def test_ensemble_chunk_one_zero_values_query_each_time(capture):
    arrays = {'query': np.zeros((1, 1, 2)), 'selected': np.zeros((1, 2)),
              'empty': np.zeros((1, 0, 2)), 'counts': np.zeros((0, 1), dtype=np.int64),
              'weights': np.ones(1)}
    events = []
    for i in range(3):
        pre = {'mode': 'ensemble', 'aggregate': None if i == 0 else 'empty',
               'counts': None if i == 0 else 'counts', 'weights': 'weights', 'weights_cumsum': 'weights'}
        post = {**pre, 'aggregate': 'empty', 'counts': 'counts'}
        events.append({'event_id': i, 'type': 'select', 'completion': 'normal',
                       'settings': {'n_action_steps': 1, 'chunk_size': 1, 'temporal_ensemble_coeff': 0.},
                       'pre_state': pre, 'post_state': post, 'selected': 'selected',
                       'predictions': [{'query_id': i, 'array': 'query'}]})
    root = capture(events, arrays)
    report = cadence.audit_query_cadence(root, max_query_gap=1)
    assert report['status'] == report['numeric_replay_status'] == 'pass'
    rewrite_manifest(root, lambda m: m['events'][0].update(predictions=[]))
    # Repair later IDs so this is absent evidence, not malformed sequential numbering.
    rewrite_manifest(root, lambda m: [e['predictions'][0].update(query_id=i) for i, e in enumerate(m['events'][1:])])
    report = cadence.audit_query_cadence(root, max_query_gap=1)
    assert report['numeric_replay_status'] == report['cadence_criterion']['status'] == 'unknown'
    assert report['cadence_criterion']['first_violation'] is None


@pytest.mark.parametrize('atol,rtol', [(True, 0.), (-1., 0.), (0., float('nan')), (0., '1')])
def test_invalid_tolerances_refused_before_intake(atol, rtol):
    with pytest.raises(InputError):
        cadence.audit_query_cadence('/does/not/exist', max_query_gap=1, atol=atol, rtol=rtol)
