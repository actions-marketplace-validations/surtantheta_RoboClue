"""Controlled saved selection state; genuine native runs have separate receipts."""
import hashlib
import io
import json

import numpy as np
import pytest

from policy_doctor import selection
from policy_doctor.artifacts import InputError
from policy_doctor.rollouts import _npz_preflight


def test_queue_math_uses_retained_state_without_inventing_origin():
    selected, post = selection._queue_replay([np.array([[9., 3.]])], None, 2)
    np.testing.assert_array_equal(selected, [[9., 3.]])
    assert post == []


def test_queue_query_is_limited_and_transposed():
    chunk = np.array([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]])
    selected, post = selection._queue_replay([], chunk, 2)
    np.testing.assert_array_equal(selected, chunk[:, 0])
    assert len(post) == 1
    np.testing.assert_array_equal(post[0], chunk[:, 1])


def test_ensemble_uses_actual_older_weight_and_does_not_mutate():
    actions = np.array([[[0., 0.], [7., 8.]]], dtype=np.float32)
    prior = np.array([[[3., 6.]]], dtype=np.float32)
    counts = np.array([[1]], dtype=np.int64)
    weights = np.array([1., np.exp(-.5)], dtype=np.float32)
    cumsum = weights.cumsum()
    original = prior.copy()
    selected, post, post_counts = selection._ensemble_replay(prior, counts, actions, weights, cumsum)
    np.testing.assert_allclose(selected, [[3 / cumsum[1], 6 / cumsum[1]]], rtol=1e-6)
    np.testing.assert_array_equal(post, actions[:, 1:])
    np.testing.assert_array_equal(post_counts, [[1]])
    np.testing.assert_array_equal(prior, original)


def test_ensemble_chunk_one_empty_state_is_healthy():
    current = np.zeros((2, 1, 3), dtype=np.float32)
    selected, post, count = selection._ensemble_replay(None, None, current, np.ones(1), np.ones(1))
    np.testing.assert_array_equal(selected, np.zeros((2, 3)))
    assert post.shape == (2, 0, 3) and count.shape == (0, 1)
    selected2, post2, count2 = selection._ensemble_replay(post, count, current, np.ones(1), np.ones(1))
    np.testing.assert_array_equal(selected2, selected)
    assert post2.shape == post.shape and count2.shape == count.shape


def test_npz_event_member_cap_keeps_existing_default():
    raw = io.BytesIO()
    np.savez(raw, **{f'x{i}': np.zeros(1) for i in range(33)})
    with pytest.raises(InputError, match='32 array members'):
        _npz_preflight(raw.getvalue(), 100000)
    _npz_preflight(raw.getvalue(), 100000, max_streams=4096)
    with pytest.raises(InputError, match='integer'):
        _npz_preflight(raw.getvalue(), 100000, max_streams=True)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    def write(events, arrays, name='capture'):
        root = tmp_path / name
        root.mkdir()
        (root / 'sources').mkdir()
        sources = {'modeling_act.py': b'controlled model test fixture',
                   'configuration_act.py': b'controlled config test fixture', 'LICENSE.txt': b'test only'}
        hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()}
        identity = {'repository': 'controlled-test', 'revision': 'test',
                    'modeling_sha256': hashes['modeling_act.py'], 'config_sha256': hashes['configuration_act.py']}
        monkeypatch.setattr(selection, 'SUPPORTED_SOURCE', identity)
        files = []
        for key, value in sources.items():
            (root / 'sources' / key).write_bytes(value)
            files.append({'path': 'sources/' + key, 'sha256': hashes[key]})
        np.savez(root / 'arrays.npz', **arrays)
        files.append({'path': 'arrays.npz', 'sha256': hashlib.sha256((root / 'arrays.npz').read_bytes()).hexdigest()})
        source = {**identity, 'files': {key: {'sha256': value} for key, value in hashes.items()}}
        data = {'schema_version': 1, 'kind': 'act_selection_capture', 'domain': 'policy_output',
                'source': source, 'environment_selection': {'source_num_envs': 1, 'env_indices': [0], 'saved_rows': 1, 'omitted_envs': 0},
                'coverage': {'target': 'one_policy_on_owner_thread', 'complete_target_call_history': True, 'external_state_mutations': 'untracked'}, 'events': events, 'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in arrays.items()}, 'files': files}
        raw = json.dumps(data).encode()
        (root / 'selection.json').write_bytes(raw)
        (root / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))
        return root
    return write


def event(index, pre, selected, post, predictions=()):
    return {'event_id': index, 'type': 'select', 'completion': 'normal',
            'settings': {'n_action_steps': 2, 'chunk_size': 3, 'temporal_ensemble_coeff': None},
            'pre_state': {'mode': 'queue', 'queue_maxlen': 2, 'queue': pre}, 'post_state': {'mode': 'queue', 'queue_maxlen': 2, 'queue': post},
            'selected': selected, 'predictions': list(predictions)}


def test_complete_queue_explanation_and_first_contradiction(capture):
    arrays = {'chunk': np.array([[[1., 2.], [3., 4.], [5., 6.]]]), 's0': np.array([[1., 2.]]), 's1': np.array([[3., 4.]]), 'bad': np.array([[30., 40.]])}
    events = [event(0, [], 's0', ['s1'], [{'query_id': 0, 'array': 'chunk'}]), event(1, ['s1'], 'bad', [])]
    report = selection.audit_selection_capture(capture(events, arrays))
    assert report['status'] == 'error'
    assert report['events'][0]['numeric_replay']['status'] == 'pass'
    assert report['events'][1]['numeric_replay']['status'] == 'error'
    assert report['events'][1]['origin']['status'] == 'unknown'
    assert report['events'][1]['arithmetic_contributors']['expected_contributors'] == [{'query_id': 0, 'query_event_id': 0, 'offset': 1, 'weight': 1.0}]
    assert report['first_contradiction']['event_id'] == 1
    assert report['downstream_association']['status'] == 'unknown'


def test_midrun_numeric_pass_does_not_invent_query(capture):
    report = selection.audit_selection_capture(capture([event(0, ['s'], 's', [])], {'s': np.zeros((1, 2))}))
    assert report['events'][0]['numeric_replay']['status'] == 'pass'
    assert report['events'][0]['origin']['status'] == 'unknown'
    assert report['status'] == 'unknown'


def test_supported_comparison_reports_lawful_difference_without_fault(capture):
    left = capture([event(0, ['s'], 's', [])], {'s': np.zeros((1, 2))}, 'left')
    right = capture([event(0, ['s'], 's', [])], {'s': np.ones((1, 2))}, 'right')
    report = selection.compare_selection_captures(left, right)
    assert report['first_selected_difference']['selection_ordinal'] == 0
    assert report['difference_is_fault'] is False
    assert report['status'] == 'unknown'  # lineage absent on both sides
    assert report['alignment_basis'] == 'Captured selection-call ordinal only; no episode/time alignment asserted.'


def test_source_drift_is_unknown_and_tampered_payload_refused(capture, monkeypatch):
    root = capture([event(0, ['s'], 's', [])], {'s': np.zeros((1, 2))})
    monkeypatch.setattr(selection, 'SUPPORTED_SOURCE', {**selection.SUPPORTED_SOURCE, 'modeling_sha256': '0' * 64})
    assert selection.audit_selection_capture(root)['source_supported'] is False
    assert selection.audit_selection_capture(root)['status'] == 'unknown'
    (root / 'arrays.npz').write_bytes(b'changed')
    with pytest.raises(InputError, match='hash'):
        selection.audit_selection_capture(root)


def test_existing_capture_routes_and_cli_expose_same_explanation(capture, capsys):
    from policy_doctor.capture import audit_capture
    from policy_doctor.cli import main
    from policy_doctor.runtime_diff import compare_runtime_captures
    chunk = np.array([[[1., 2.], [3., 4.], [5., 6.]]])
    root = capture([event(0, [], 's0', ['s1'], [{'query_id': 0, 'array': 'chunk'}])],
                   {'chunk': chunk, 's0': chunk[:, 0], 's1': chunk[:, 1]})
    assert audit_capture(str(root))['status'] == 'unknown'
    assert compare_runtime_captures(str(root), str(root))['first_selected_difference'] is None
    assert main(['audit-capture', str(root), '--json']) == 3
    report = json.loads(capsys.readouterr().out)
    assert report['events'][0]['arithmetic_contributors']['contributors'][0]['query_id'] == 0
    assert main(['audit-capture', str(root), '--markdown']) == 3
    assert 'Selection origins' in capsys.readouterr().out
    with pytest.raises(InputError, match='layout criteria'):
        audit_capture(str(root), expected_names=['joint'])


def rewrite_manifest(root, mutate):
    path = root / 'selection.json'
    data = json.loads(path.read_text())
    mutate(data)
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    (root / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


@pytest.mark.parametrize('bad', [{}, {'source_num_envs': 1, 'env_indices': [0], 'saved_rows': True, 'omitted_envs': False},
                                 {'source_num_envs': 64, 'env_indices': list(range(64)), 'saved_rows': 64, 'omitted_envs': 0}])
def test_row_selection_is_explicit_bounded_and_strict(capture, bad):
    root = capture([event(0, ['s'], 's', [])], {'s': np.zeros((1, 2))})
    rewrite_manifest(root, lambda data: data.update(environment_selection=bad))
    with pytest.raises(InputError, match='selection'):
        selection.audit_selection_capture(root)


def test_missing_or_gapped_coverage_keeps_arithmetic_history_unknown(capture):
    chunk = np.ones((1, 3, 2))
    root = capture([event(0, [], 's', ['s'], [{'query_id': 0, 'array': 'chunk'}])], {'chunk': chunk, 's': chunk[:, 0]})
    assert selection.audit_selection_capture(root)['events'][0]['arithmetic_contributors']['status'] == 'pass'
    rewrite_manifest(root, lambda data: data.pop('coverage'))
    report = selection.audit_selection_capture(root)
    assert report['events'][0]['numeric_replay']['status'] == 'pass'
    assert report['events'][0]['arithmetic_contributors']['status'] == 'unknown'
    assert report['events'][0]['origin']['status'] == 'unknown'


def test_actual_deque_maxlen_controls_retention_after_config_change():
    query = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    selected, post = selection._queue_replay([], query, steps=4, maxlen=2)
    np.testing.assert_array_equal(selected, query[:, 2])
    np.testing.assert_array_equal(post[0], query[:, 3])


def test_reset_only_capture_does_not_certify_an_action(capture):
    reset = {'event_id': 0, 'type': 'reset', 'completion': 'normal',
             'settings': {'n_action_steps': 2, 'chunk_size': 3, 'temporal_ensemble_coeff': None},
             'pre_state': {'mode': 'queue', 'queue_maxlen': 2, 'queue': []},
             'post_state': {'mode': 'queue', 'queue_maxlen': 2, 'queue': []},
             'predictions': [], 'selected': None}
    root = capture([reset], {})
    rewrite_manifest(root, lambda data: data.update(environment_selection={key: None for key in data['environment_selection']}))
    report = selection.audit_selection_capture(root)
    assert report['status'] == 'unknown' and report['selection_count'] == 0
    assert report['events'][0]['numeric_replay']['status'] == 'pass'


@pytest.mark.parametrize('weights,coefficient', [(np.array([1., .5], dtype=np.float32), 0.),
                                               (np.array([1., 0.], dtype=np.float32), 1000.)])
def test_actual_weights_override_declared_formula_without_false_fault(capture, weights, coefficient):
    arrays = {'a': np.array([[[6., 12.]]], dtype=np.float32), 'count': np.array([[1]], dtype=np.int64),
              'chunk': np.zeros((1, 2, 2), dtype=np.float32), 'w': weights, 'c': weights.cumsum(),
              'post': np.zeros((1, 1, 2), dtype=np.float32)}
    arrays['s'] = arrays['a'][:, 0] / arrays['c'][1]
    pre = {'mode': 'ensemble', 'aggregate': 'a', 'counts': 'count', 'weights': 'w', 'weights_cumsum': 'c'}
    post = dict(pre, aggregate='post')
    e = {'event_id': 0, 'type': 'select', 'completion': 'normal', 'pre_state': pre, 'post_state': post,
         'settings': {'chunk_size': 2, 'n_action_steps': 1, 'temporal_ensemble_coeff': coefficient},
         'predictions': [{'query_id': 0, 'array': 'chunk'}], 'selected': 's'}
    report = selection.audit_selection_capture(capture([e], arrays))
    assert report['events'][0]['numeric_replay']['status'] == 'pass'
    assert report['status'] == 'unknown'  # no original lineage supplied
    if coefficient == 0:
        assert report['events'][0]['numeric_replay']['declared_coefficient_matches_stored_weights'] is False


def test_changed_deque_identity_setting_breaks_conditional_history(capture):
    arrays = {'chunk': np.zeros((1, 3, 2)), 's': np.zeros((1, 2))}
    first = event(0, [], 's', ['s'], [{'query_id': 0, 'array': 'chunk'}])
    second = event(1, ['s'], 's', [])
    second['pre_state']['queue_maxlen'] = 1
    second['post_state']['queue_maxlen'] = 1
    report = selection.audit_selection_capture(capture([first, second], arrays))
    assert report['events'][1]['numeric_replay']['status'] == 'pass'
    assert report['events'][1]['arithmetic_contributors']['status'] == 'unknown'
    assert report['events'][1]['origin']['status'] == 'unknown'


def test_comparison_binds_audit_and_reloaded_bytes(capture, monkeypatch):
    left = capture([event(0, ['s'], 's', [])], {'s': np.zeros((1, 2))}, 'left')
    right = capture([event(0, ['s'], 's', [])], {'s': np.zeros((1, 2))}, 'right')
    actual_load = selection._load
    calls = 0
    def load(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            # A validly re-committed packet replaces a prior audited snapshot.
            np.savez(right / 'arrays.npz', s=np.ones((1, 2)))
            def rebind(data):
                for entry in data['files']:
                    if entry['path'] == 'arrays.npz':
                        entry['sha256'] = hashlib.sha256((right / 'arrays.npz').read_bytes()).hexdigest()
            rewrite_manifest(right, rebind)
        return actual_load(path)
    monkeypatch.setattr(selection, '_load', load)
    with pytest.raises(InputError, match='between audit and comparison'):
        selection.compare_selection_captures(left, right)
