"""Two generated grouped native schemas; no simulator or policy run."""
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.evaluation_cohort import audit_evaluation_cohort, validate_parameters

TABLE = {'groups': '/tasks', 'group_id': '/task', 'records': '/episodes',
         'record_id': '/episode_id', 'outcome': '/metrics/success'}
OTHER = {'groups': '/runs', 'group_id': '/label', 'records': '/trials',
         'record_id': '/id', 'outcome': '/ok'}


def document():
    return {'config': {'seed': 7, 'interface': 'allowed'}, 'tasks': [
        {'task': 'A', 'episodes': [{'episode_id': 0, 'metrics': {'success': True}},
                                  {'episode_id': 1, 'metrics': {'success': False}}]},
        {'task': 'B', 'episodes': [{'episode_id': '0', 'metrics': {'success': True}}]}]}


def write(tmp_path, name, value=None):
    root = tmp_path / name
    root.mkdir()
    (root / 'result.json').write_text(json.dumps(document() if value is None else value))
    return root


def audit(candidate, reference=None, equal_fields=None, table=None, **kwargs):
    return audit_evaluation_cohort(candidate, reference=reference, table=table or TABLE,
                                   equal_fields=['/config/seed'] if equal_fields is None else equal_fields,
                                   **kwargs)


def test_allowed_outcome_interface_changes_and_row_reordering(tmp_path):
    ref = write(tmp_path, 'ref')
    data = document()
    data['config']['interface'] = 'new intended interface'
    data['tasks'].reverse()
    data['tasks'][1]['episodes'].reverse()
    data['tasks'][1]['episodes'][0]['metrics']['success'] = True
    candidate = write(tmp_path, 'candidate', data)
    report = audit(candidate, ref)
    assert report['kind'] == 'evaluation_cohort_audit'
    assert report['status'] == 'pass'
    assert report['cohort_comparison']['status'] == 'pass'
    assert report['candidate']['total_records'] == 3
    assert report['candidate']['true_count'] == 3
    assert report['reference']['false_count'] == 1
    assert report['source_intake']['document_selected']
    assert 'physical' in ' '.join(report['unverified'])


def test_second_schema_preserves_semantic_names_and_typed_ids(tmp_path):
    data = {'runs': [{'label': ' leading and trailing ', 'trials': [
        {'id': 1, 'ok': True}, {'id': '1', 'ok': False}]}]}
    root = write(tmp_path, 'source', data)
    report = audit(root, root, table=OTHER, equal_fields=[])
    assert report['status'] == 'pass'
    assert report['candidate']['unique_records'] == 2
    assert report['candidate']['groups'][0]['group_id'] == ' leading and trailing '


def test_known_mismatch_and_missing_context_are_distinct(tmp_path):
    ref = write(tmp_path, 'ref')
    data = document()
    data['config']['seed'] = 9
    candidate = write(tmp_path, 'candidate', data)
    report = audit(candidate, ref, equal_fields=['/config/seed', '/missing'])
    assert report['status'] == 'error'
    assert [field['status'] for field in report['fields']] == ['error', 'unknown']
    witness = report['fields'][0]
    assert witness['candidate']['pointer'] == '/config/seed'
    assert witness['candidate']['sha256']
    assert witness['candidate']['value'] == 9
    assert witness['reference']['value'] == 7


def test_exact_typed_missing_extra_record_sets(tmp_path):
    ref = write(tmp_path, 'ref')
    data = document()
    data['tasks'][0]['episodes'][0]['episode_id'] = '0'
    candidate = write(tmp_path, 'candidate', data)
    result = audit(candidate, ref)['cohort_comparison']
    assert result['status'] == 'error'
    assert result['missing_count'] == result['extra_count'] == 1
    assert result['missing'][0]['record_id_type'] == 'int'
    assert result['extra'][0]['record_id_type'] == 'str'


@pytest.mark.parametrize('fault', ['missing_id', 'bool_id', 'missing_outcome', 'int_outcome', 'missing_records'])
def test_incomplete_native_rows_stay_unknown_without_rate(tmp_path, fault):
    data = document()
    row = data['tasks'][0]['episodes'][0]
    if fault == 'missing_id':
        del row['episode_id']
    elif fault == 'bool_id':
        row['episode_id'] = True
    elif fault == 'missing_outcome':
        del row['metrics']['success']
    elif fault == 'int_outcome':
        row['metrics']['success'] = 1
    else:
        del data['tasks'][0]['episodes']
    root = write(tmp_path, 'source', data)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert report['candidate']['rate'] is None


def test_duplicate_records_are_proven_error_without_rate(tmp_path):
    data = document()
    data['tasks'][0]['episodes'].append(data['tasks'][0]['episodes'][0].copy())
    root = write(tmp_path, 'source', data)
    report = audit(root, root)
    assert report['status'] == 'error'
    assert report['candidate']['duplicate_records'] == 1
    assert report['candidate']['rate'] is None


def test_same_drop_both_is_only_observed_set_pass(tmp_path):
    data = document()
    data['tasks'][0]['episodes'].pop()
    data['tasks'][0]['num_episodes'] = 200
    root = write(tmp_path, 'source', data)
    report = audit(root, root)
    assert report['status'] == 'pass'
    assert report['candidate']['total_records'] == 2
    assert 'declared cohort completeness' in report['unverified']


def test_missing_reference_never_implicitly_self_compares(tmp_path):
    root = write(tmp_path, 'source')
    report = audit(root, equal_fields=[])
    assert report['status'] == 'unknown'
    assert report['source_intake']['status'] == 'pass'
    assert report['candidate']['status'] == 'pass'
    assert audit(root, tmp_path / 'absent')['status'] == 'unknown'


def test_document_selection_and_ambiguity(tmp_path):
    root = write(tmp_path, 'source')
    (root / 'config.json').write_text('{"not_a_table":true}')
    assert audit(root, root)['status'] == 'pass'
    (root / 'second.json').write_text(json.dumps(document()))
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert not report['source_intake']['document_selected']
    assert audit(root, root, document_pattern='result.json')['status'] == 'pass'


@pytest.mark.parametrize('raw', ['{"tasks":[],"tasks":[]}', '{bad}', '{"x":NaN}', '{"x":1e999}'])
def test_malformed_matching_json_cannot_be_ignored(tmp_path, raw):
    root = write(tmp_path, 'source')
    (root / 'malformed.json').write_text(raw)
    assert audit(root, root)['source_intake']['status'] == 'unknown'


def test_callable_text_in_selected_context_is_inert_unknown(tmp_path):
    marker = tmp_path / 'executed'
    data = document()
    data['config']['code'] = f"functools.partial(__import__('pathlib').Path({str(marker)!r}).touch)"
    root = write(tmp_path, 'source', data)
    assert audit(root, root, equal_fields=['/config/code'])['status'] == 'unknown'
    assert not marker.exists()
    assert audit(root, root)['status'] == 'pass'


@pytest.mark.parametrize('params', [
    {}, {'table': TABLE}, {'table': TABLE, 'equal_fields': '/config'},
    {'table': dict(TABLE, unknown='/x'), 'equal_fields': []},
    {'table': dict(TABLE, record_id=True), 'equal_fields': []},
    {'table': TABLE, 'equal_fields': ['/bad~2']},
    {'table': TABLE, 'equal_fields': ['/x', '/x']},
    {'table': TABLE, 'equal_fields': [], 'document_pattern': '../*.json'},
    {'table': TABLE, 'equal_fields': [], 'reference': 'other'},
])
def test_strict_declarative_parameters(params):
    with pytest.raises(InputError):
        validate_parameters(params)


def test_adjacent_large_ints_are_not_float_normalized(tmp_path):
    data = document()
    data['tasks'][0]['episodes'][0]['episode_id'] = 2**53
    data['tasks'][0]['episodes'][1]['episode_id'] = 2**53 + 1
    root = write(tmp_path, 'source', data)
    result = audit(root, root)
    assert result['status'] == 'pass'
    assert result['candidate']['unique_records'] == 3
    data['tasks'][0]['episodes'][0]['episode_id'] = 2**53 + 2
    changed = write(tmp_path, 'changed', data)
    difference = audit(changed, root)['cohort_comparison']
    assert difference['missing'][0]['record_id'] == 2**53
    assert difference['extra'][0]['record_id'] == 2**53 + 2


@pytest.mark.parametrize('value', [1.0, True, None, '', [], {}])
def test_invalid_ids_never_allow_exact_inventory_inference(tmp_path, value):
    ref = write(tmp_path, 'ref')
    data = document()
    data['tasks'][0]['episodes'][0]['episode_id'] = value
    root = write(tmp_path, 'source', data)
    result = audit(root, ref)
    assert result['status'] == 'unknown'
    assert result['cohort_comparison']['status'] == 'unknown'
    assert result['cohort_comparison']['missing_count'] == 0
    assert result['candidate']['rate'] is None


@pytest.mark.parametrize('groups', [[], [{'task': 'A', 'episodes': []}]])
def test_empty_table_is_unknown_not_zero_success(tmp_path, groups):
    root = write(tmp_path, 'empty', dict(document(), tasks=groups))
    result = audit(root, root)
    assert result['status'] == 'unknown'
    assert result['candidate']['rate'] is None
    assert result['cohort_comparison']['status'] == 'unknown'
    assert set(result['completeness'].values()) == {'unknown'}


def test_duplicate_group_is_error_even_without_duplicate_records(tmp_path):
    data = document()
    data['tasks'][1]['task'] = 'A'
    root = write(tmp_path, 'source', data)
    result = audit(root, root)
    assert result['status'] == 'error'
    assert result['candidate']['duplicate_groups'] == 1
    assert result['candidate']['rate'] is None
    assert result['cohort_comparison']['status'] == 'unknown'


def test_rfc6901_native_names_and_array_indices(tmp_path):
    data = {'a/b': {'~tasks': [{'task': 'é A', 'episodes': [
        {'episode_id': 1, 'metrics': {'success': True}}]}]}, 'config': [7]}
    root = write(tmp_path, 'source', data)
    table = dict(TABLE, groups='/a~1b/~0tasks')
    assert audit(root, root, table=table, equal_fields=['/config/0'])['status'] == 'pass'
    assert audit(root, root, table=table, equal_fields=['/config/00'])['status'] == 'unknown'


def test_symlinks_and_nonflat_sources_stay_unknown(tmp_path):
    source = write(tmp_path, 'source')
    linked = tmp_path / 'linked'
    linked.symlink_to(source, target_is_directory=True)
    assert audit(linked, source)['source_intake']['status'] == 'unknown'
    (source / 'alias.json').symlink_to(source / 'result.json')
    assert audit(source, source)['source_intake']['status'] == 'unknown'
    (source / 'alias.json').unlink()
    (source / 'subfolder').mkdir()
    assert audit(source, source)['source_intake']['status'] == 'unknown'


def test_intake_budgets_and_depth_preflight(tmp_path, monkeypatch):
    from policy_doctor import evaluation_cohort as module
    root = write(tmp_path, 'source')
    monkeypatch.setattr(module, 'MAX_BYTES', 10)
    assert audit(root, root)['source_intake']['status'] == 'unknown'
    monkeypatch.setattr(module, 'MAX_BYTES', 2 * 1024**2)
    # Keep depth intake independent of mutating a previously enumerated directory.
    deep_root = write(tmp_path, 'deep-source')
    (deep_root / 'deep.json').write_text('[' * 33 + '0' + ']' * 33)
    assert {path.name for path in deep_root.iterdir()} == {'result.json', 'deep.json'}
    called = False
    original = module.json.loads

    def guarded(text, **kwargs):
        nonlocal called
        if text.startswith('['):
            called = True
            raise AssertionError('Recursive decoder entered before depth refusal')
        return original(text, **kwargs)

    monkeypatch.setattr(module.json, 'loads', guarded)
    assert audit(deep_root, deep_root)['source_intake']['status'] == 'unknown'
    assert not called


def test_file_and_directory_budgets(tmp_path, monkeypatch):
    from policy_doctor import evaluation_cohort as module
    root = write(tmp_path, 'source')
    (root / 'unrelated.json').write_text('{}')
    monkeypatch.setattr(module, 'MAX_FILES', 1)
    assert audit(root, root)['source_intake']['status'] == 'unknown'
    monkeypatch.setattr(module, 'MAX_FILES', 32)
    monkeypatch.setattr(module, 'MAX_ENTRIES', 1)
    assert audit(root, root)['source_intake']['status'] == 'unknown'


def test_source_drift_rejected_even_when_bytes_restored(tmp_path, monkeypatch):
    from policy_doctor import evaluation_cohort as module
    root = write(tmp_path, 'source')
    path = root / 'result.json'
    original_bytes = path.read_bytes()
    original = module._fields

    def mutate(*args):
        # Replacing the file makes the identity change observable even on a
        # filesystem that coalesces rapid same-inode timestamp updates.
        replacement = path.with_name('replacement.json')
        replacement.write_bytes(original_bytes)
        path.write_bytes(b'{}')
        replacement.replace(path)
        return original(*args)

    monkeypatch.setattr(module, '_fields', mutate)
    with pytest.raises(InputError, match='Stable original evidence'):
        audit(root, root)
    assert path.read_bytes() == original_bytes


def test_response_budget_fallback_retains_counts_and_is_unknown(tmp_path, monkeypatch):
    from policy_doctor import evaluation_cohort as module
    root = write(tmp_path, 'source')
    # Force fallback without assuming a particular temporary-directory length.
    full_size = len(module._canonical(audit(root, root)))
    monkeypatch.setattr(module, 'MAX_RESPONSE', full_size - 1)
    result = audit(root, root)
    assert result['status'] == 'unknown'
    assert result['response_budget_exceeded']
    assert result['candidate']['total_records'] == 3
    assert result['candidate']['groups_omitted'] == 2
    assert result['finding_count'] == 1
    assert result['finding_status_counts']['unknown'] == 1
    assert len(module._canonical(result)) <= full_size - 1


def test_many_missing_keys_have_exact_counts_and_bounded_original_witnesses(tmp_path):
    data = document()
    data['tasks'] = [{'task': 'A', 'episodes': [
        {'episode_id': i, 'metrics': {'success': True}} for i in range(100)]}]
    ref = write(tmp_path, 'ref', data)
    data['tasks'][0]['episodes'] = [
        {'episode_id': i, 'metrics': {'success': False}} for i in range(100, 200)]
    root = write(tmp_path, 'source', data)
    original_bytes = (root / 'result.json').read_bytes()
    result = audit(root, ref)
    comparison = result['cohort_comparison']
    assert comparison['missing_count'] == comparison['extra_count'] == 100
    assert len(comparison['missing']) == len(comparison['extra']) == 16
    assert comparison['extra'][0]['evidence']['pointer'].startswith('/tasks/0/episodes/')
    assert len(json.dumps(result).encode()) <= 128 * 1024
    assert (root / 'result.json').read_bytes() == original_bytes


def test_large_context_preview_is_bounded_and_original_hashed(tmp_path):
    data = document()
    data['config']['large'] = ['x' * 1000] * 100
    root = write(tmp_path, 'source', data)
    result = audit(root, root, equal_fields=['/config/large'])
    assert result['status'] == 'pass'
    field = result['fields'][0]['candidate']
    assert field['value_bytes'] > 100000
    assert field['preview_truncated']
    assert len(field['preview']) == 256
    assert len(field['sha256']) == len(field['canonical_sha256']) == 64
    assert 'value' not in field


def test_native_groups_and_records_must_be_objects(tmp_path):
    array_groups = write(tmp_path, 'array_groups', {'tasks': [['A', [[1, True]]]]})
    selectors = {'groups': '/tasks', 'group_id': '/0', 'records': '/1',
                 'record_id': '/0', 'outcome': '/1'}
    result = audit(array_groups, array_groups, table=selectors, equal_fields=[])
    assert result['status'] == 'unknown'
    assert result['candidate']['rate'] is None
    array_records = write(tmp_path, 'array_records', {'tasks': [{'task': 'A', 'episodes': [[1, True]]}]})
    result = audit(array_records, array_records, table=dict(TABLE, record_id='/0', outcome='/1'),
                   equal_fields=[])
    assert result['status'] == 'unknown'
    assert result['candidate']['rate'] is None
