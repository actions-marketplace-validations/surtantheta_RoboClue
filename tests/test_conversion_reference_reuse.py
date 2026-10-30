"""Generated source/export controls for fixed-criterion reuse; no original data."""
import copy
import json
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pytest
from test_conversion_fidelity_reader_proposal import Generated
from test_conversion_preflight_artifacts import corrupt_action

from policy_doctor import conversion_fidelity as engine
from policy_doctor.artifacts import InputError


def save(generated, report, name='reference.json'):
    path = generated.root / name
    path.write_text(json.dumps({'schema_version': 1, 'kind': 'dataset_preflight',
        'components': {'conversion_fidelity': report}}, allow_nan=False))
    return path


def restored_check(generated, reference, **kwargs):
    restored = engine.resolve_conversion_reference(str(reference), read_roots=[str(generated.root)])
    return engine.check_conversion_fidelity(str(generated.export), **restored['selectors'],
        expected_criterion=restored['expected_criterion'], reference_binding=restored['reference_binding'],
        read_roots=[str(generated.root)], **kwargs)


@pytest.mark.parametrize('value', [True, {}, '', 'A'*64, 'g'*64, 'a'*63, '\ud800'])
def test_digest_is_strict_before_reference_read(monkeypatch, value):
    monkeypatch.setattr(engine, '_read_small', lambda *a, **k: pytest.fail('read before argument refusal'))
    with pytest.raises(InputError):
        engine.resolve_conversion_reference('/reference.json', value)


@pytest.mark.parametrize('value', ['~/file', '/tmp/../file', '/tmp/\ud800', '/tmp/\x00file'])
def test_reference_path_refused_before_read(monkeypatch, value):
    monkeypatch.setattr(engine, '_read_small', lambda *a, **k: pytest.fail('invalid path read'))
    with pytest.raises(InputError):
        engine.resolve_conversion_reference(value)


@pytest.mark.parametrize('change', ['missing_action', 'width42', 'int32', 'null', 'truncated'])
def test_wrong_candidate_still_establishes_original_criterion(tmp_path, change):
    generated = Generated(tmp_path)
    healthy = generated.check()
    table = generated.table(0)
    if change == 'missing_action':
        table = table.drop(['action'])
    elif change == 'truncated':
        table = table.slice(0, len(table)-1)
    else:
        values = table['action'].to_pylist()
        if change == 'width42':
            values = [row[:-1] for row in values]
        elif change == 'null':
            values[0][0] = None
        dtype = pa.int32() if change == 'int32' else pa.float32()
        if change == 'int32':
            values = [[int(value) for value in row] for row in values]
        table = table.set_column(table.schema.get_field_index('action'), 'action',
                                 pa.array(values, type=pa.list_(dtype)))
    generated.write(0, table)
    wrong = generated.check()
    assert wrong['status'] == 'error', wrong
    assert not wrong['coverage']['whole_numeric_source_to_export']
    assert wrong['conversion_reuse']['criterion_basis_complete']
    assert wrong['criterion_sha256'] == healthy['criterion_sha256']
    reference = save(generated, wrong)
    generated.restore()
    corrected = restored_check(generated, reference)
    assert corrected['status'] == corrected['numeric_association_status'] == 'pass', corrected
    assert corrected['criterion_reuse']['matches'] is True
    assert corrected['criterion_reuse']['reference_original_status'] == 'error'
    assert corrected['criterion_reuse']['original_report_reproduced'] is False


@pytest.mark.parametrize('change', ['missing_aux', 'nonfinite_aux', 'nonfinite_joint'])
def test_incomplete_original_cannot_qualify_even_resealed_intent(tmp_path, change):
    generated = Generated(tmp_path)
    with h5py.File(generated.original, 'r+') as original:
        if change == 'missing_aux':
            del original['data/demo_z/action/navigate_cmd']
        elif change == 'nonfinite_aux':
            original['data/demo_z/action/base_height_cmd'][0, 0] = np.nan
        else:
            original['data/demo_z/processed_actions'][0, 0] = np.nan
    report = generated.check()
    assert not report['conversion_reuse']['criterion_basis_complete']
    path = save(generated, report)
    with pytest.raises(InputError, match='incomplete'):
        engine.resolve_conversion_reference(str(path), read_roots=[str(tmp_path)])
    # A caller can author a self-consistent intent report. It is not history.
    # Actual source validation still prevents that claim from certifying a run.
    report['conversion_reuse']['criterion_basis_complete'] = True
    path = save(generated, report, 'forged-intent.json')
    result = restored_check(generated, path)
    assert result['criterion_reuse']['status'] == 'unknown'
    assert result['criterion_reuse']['matches'] is None
    assert result['numeric_association_status'] == 'unknown'
    assert not result['criterion_reuse']['candidate_evaluated_under_expected_criterion']


def test_candidate_only_relocation_and_external_pin(tmp_path):
    generated = Generated(tmp_path)
    before = generated.check()
    reference = save(generated, before)
    target = tmp_path/'other-export'
    shutil.copytree(generated.export, target)
    restored = engine.resolve_conversion_reference(str(reference), before['criterion_sha256'],
                                                    read_roots=[str(tmp_path)])
    after = engine.check_conversion_fidelity(str(target), **restored['selectors'],
        expected_criterion=restored['expected_criterion'], reference_binding=restored['reference_binding'],
        read_roots=[str(tmp_path)])
    assert after['criterion_sha256'] == before['criterion_sha256']
    assert after['criterion_reuse']['matches'] is True
    with pytest.raises(InputError, match='conflicts'):
        engine.resolve_conversion_reference(str(reference), '0'*64, read_roots=[str(tmp_path)])


def test_changed_original_preserves_current_error_but_not_expected_pass(tmp_path):
    generated = Generated(tmp_path)
    before = generated.check()
    reference = save(generated, before)
    with h5py.File(generated.original, 'r+') as original:
        original['data/demo_z/processed_actions'][0, 0] += np.float32(1)
    after = restored_check(generated, reference)
    assert after['status'] == 'error' and after['numeric_association_status'] == 'unknown'
    assert after['criterion_reuse']['matches'] is False
    assert after['current_criterion_observations']['numeric_association_status'] == 'error'
    assert any(f['code'] == 'conversion.numeric_joint_association' for f in after['findings'])
    assert after['findings'][0]['code'] == 'conversion.criterion_drift'
    assert any(field.startswith('/original_sha256') for field in after['criterion_reuse']['drift_fields'])
    with pytest.raises(InputError, match='replacement baseline'):
        engine.resolve_conversion_reference(str(save(generated, after, 'refused.json')),
                                            read_roots=[str(tmp_path)])


def test_equal_digest_without_complete_source_budget_is_not_match(tmp_path):
    generated = Generated(tmp_path)
    before = generated.check()
    reference = save(generated, before)
    after = restored_check(generated, reference, max_rows=8)
    assert after['criterion_sha256'] == before['criterion_sha256']
    assert after['criterion_reuse']['matches'] is None
    assert after['criterion_reuse']['status'] == after['numeric_association_status'] == 'unknown'


@pytest.mark.parametrize('field', ['dependencies', 'python', 'evaluator'])
def test_version_dependency_frontend_drift_is_enforced(tmp_path, field):
    generated = Generated(tmp_path)
    report = generated.check()
    assert {'preflight.py', 'cli.py', 'server.py', '__init__.py'} <= set(report['criterion']['evaluator'])
    if field == 'dependencies':
        report['criterion'][field]['numpy'] += '.generated-other'
    elif field == 'python':
        report['criterion'][field] = '0.0.0-generated'
    else:
        report['criterion'][field]['server.py'] = '0'*64
    digest = engine._criterion_digest(report['criterion'])
    report['criterion_sha256'] = report['conversion_reuse']['criterion_sha256'] = digest
    after = restored_check(generated, save(generated, report))
    assert after['criterion_reuse']['matches'] is False
    assert after['status'] == 'error' and after['numeric_association_status'] == 'unknown'
    assert after['current_criterion_observations']['numeric_association_status'] == 'pass'


def test_old_or_concise_report_cannot_invent_explicit_cwd(tmp_path):
    generated = Generated(tmp_path)
    original = generated.check()
    for key in ['conversion_reuse', 'criterion']:
        report = copy.deepcopy(original)
        report.pop(key)
        with pytest.raises(InputError):
            engine.resolve_conversion_reference(str(save(generated, report)), read_roots=[str(tmp_path)])
    original['conversion_reuse']['inputs']['converter_directory'] = None
    with pytest.raises(InputError, match='explicit converter_directory'):
        engine.resolve_conversion_reference(str(save(generated, original)), read_roots=[str(tmp_path)])


def test_reference_authorization_and_changed_file_precede_worker(tmp_path, monkeypatch):
    generated = Generated(tmp_path)
    original = generated.check()
    reference = save(generated, original)
    with pytest.raises(InputError, match='omit explicit'):
        engine.resolve_conversion_reference(str(reference), original_recording=str(generated.original))
    restored = engine.resolve_conversion_reference(str(reference), read_roots=[str(tmp_path)])
    reference.write_text(reference.read_text()+'\n')
    monkeypatch.setattr(engine.subprocess, 'run', lambda *a, **k: pytest.fail('worker ran after reference changed'))
    with pytest.raises(InputError, match='changed'):
        engine.check_conversion_fidelity(str(generated.export), **restored['selectors'],
            expected_criterion=restored['expected_criterion'], reference_binding=restored['reference_binding'],
            read_roots=[str(tmp_path)])
    original['conversion_reuse']['inputs']['original_recording'] = '/outside/original.hdf5'
    with pytest.raises(InputError, match='outside authorized'):
        engine.resolve_conversion_reference(str(save(generated, original)), read_roots=[str(tmp_path)])


@pytest.mark.parametrize('raw', [b'{"kind":1,"kind":2}', b'{"x":NaN}',
                                b'{"x":'+b'['*34+b'0'+b']'*34+b'}'])
def test_malformed_json_is_controlled(tmp_path, raw):
    reference = tmp_path/'bad.json'
    reference.write_bytes(raw)
    with pytest.raises(InputError):
        engine.resolve_conversion_reference(str(reference), read_roots=[str(tmp_path)])


def test_forged_internal_context_and_prepared_mismatch_refused(tmp_path):
    generated = Generated(tmp_path)
    with pytest.raises(InputError, match='internal saved-report'):
        generated.check(expected_criterion='0'*64, reference_binding={})
    prepared = engine.prepare_conversion_inputs(str(generated.export), **generated.kwargs())
    with pytest.raises(InputError, match='Prepared'):
        generated.check(expected_criterion='0'*64, _prepared=prepared)


def test_digest_only_assertion_preserves_verified_error(tmp_path):
    generated = Generated(tmp_path)
    corrupt_action(generated)
    result = generated.check(expected_criterion='0'*64)
    assert result['criterion_reuse']['status'] == 'error'
    assert result['numeric_association_status'] == 'unknown'
    assert result['current_criterion_observations']['numeric_association_status'] == 'error'
    assert any(f['code'] == 'conversion.numeric_joint_association' for f in result['findings'])


@pytest.mark.parametrize('field', ['scalar_policy', 'numeric_semantic_profile'])
def test_self_consistent_unbounded_descriptor_keys_are_refused(tmp_path, field):
    generated = Generated(tmp_path)
    report = generated.check()
    report['criterion'][field]['generated-long-key-' + 'x'*8192] = '0'*64
    digest = engine._criterion_digest(report['criterion'])
    report['criterion_sha256'] = report['conversion_reuse']['criterion_sha256'] = digest
    with pytest.raises(InputError):
        engine.resolve_conversion_reference(str(save(generated, report)), read_roots=[str(tmp_path)])


@pytest.mark.parametrize('status', [{}, [], True])
def test_malformed_historical_status_is_controlled(tmp_path, status):
    generated = Generated(tmp_path)
    report = generated.check()
    report['status'] = status
    with pytest.raises(InputError):
        engine.resolve_conversion_reference(str(save(generated, report)), read_roots=[str(tmp_path)])


@pytest.mark.parametrize('key,value', [('integer_fps_min', True), ('task_index_min', False), ('integer_fps_max', 1000.0)])
def test_drift_field_paths_are_bounded_and_policy_values_are_typed(tmp_path, key, value):
    path = 'x'*8192
    fields, omitted = engine._drift_fields({path: 1}, {path: 2})
    assert len(fields) == 1 and len(fields[0]) <= 256 and '#sha256:' in fields[0]
    assert omitted == 0
    generated = Generated(tmp_path)
    report = generated.check()
    report['criterion']['scalar_policy'][key] = value
    digest = engine._criterion_digest(report['criterion'])
    report['criterion_sha256'] = report['conversion_reuse']['criterion_sha256'] = digest
    with pytest.raises(InputError):
        engine.resolve_conversion_reference(str(save(generated, report)), read_roots=[str(tmp_path)])


def test_large_full_report_generates_complete_small_view_for_later_export(tmp_path):
    generated = Generated(tmp_path)
    baseline = generated.check()
    full = {'schema_version': 1, 'kind': 'dataset_preflight', 'status': 'unknown',
            'components': {'conversion_fidelity': baseline},
            'findings': [{'message': 'generated unrelated diagnostic '*12000}]}
    assert len(json.dumps(full).encode()) > 192*1024
    view = engine.conversion_reference_view(full)
    assert view['reference_available'] and view['kind'] == 'conversion_preflight_reference'
    assert view['status'] == 'unknown'  # The omitted preflight axes were not cleared.
    assert len(json.dumps(view).encode()) <= 192*1024
    assert 'findings' not in view and 'auxiliary_episode_records' not in view['components']['conversion_fidelity']
    assert view['components']['conversion_fidelity']['criterion'] == baseline['criterion']
    reference = tmp_path/'view.json'
    reference.write_text(json.dumps(view))
    corrupt_action(generated)
    wrong = restored_check(generated, reference)
    assert wrong['criterion_reuse']['matches'] is True and wrong['numeric_association_status'] == 'error'
    generated.restore()
    corrected = restored_check(generated, reference)
    assert corrected['criterion_reuse']['matches'] is True and corrected['numeric_association_status'] == 'pass'


def test_view_keeps_original_error_and_refuses_descriptor_truncation_or_drift(tmp_path):
    generated = Generated(tmp_path)
    corrupt_action(generated)
    original = generated.check()
    full = {'schema_version': 1, 'kind': 'dataset_preflight', 'status': 'error',
            'components': {'conversion_fidelity': original}}
    view = engine.conversion_reference_view(full)
    assert view['reference_available'] and view['status'] == 'error'
    assert view['components']['conversion_fidelity']['status'] == 'error'
    unavailable = engine.conversion_reference_view(full, max_bytes=1024)
    assert unavailable['status'] == 'error' and unavailable['reference_status'] == 'unknown' and unavailable['reference_available'] is False
    assert unavailable['source_report_status'] == unavailable['source_conversion_status'] == 'error'
    assert unavailable['source_error_summary']['code'].startswith('conversion.')
    assert len(json.dumps(unavailable).encode()) <= 1024
    assert 'components' not in unavailable
    reference = tmp_path/'unavailable.json'
    reference.write_text(json.dumps(unavailable))
    with pytest.raises(InputError):
        engine.resolve_conversion_reference(str(reference), read_roots=[str(tmp_path)])
    refused = generated.check(expected_criterion='0'*64)
    full['components']['conversion_fidelity'] = refused
    blocked_view = engine.conversion_reference_view(full)
    assert blocked_view['status'] == 'error' and blocked_view['reference_status'] == 'unknown' and not blocked_view['reference_available']
    assert 'replacement baseline' in blocked_view['reason']


@pytest.mark.parametrize('budget', [True, 0, 1023, 192*1024+1, '192000'])
def test_view_budget_is_strict_without_numerical_intake(budget):
    with pytest.raises(InputError):
        engine.conversion_reference_view({}, max_bytes=budget)


def test_view_cannot_promote_incomplete_source(tmp_path):
    generated = Generated(tmp_path)
    with h5py.File(generated.original, 'r+') as original:
        del original['data/demo_z/action/base_height_cmd']
    unknown = generated.check()
    view = engine.conversion_reference_view({'schema_version': 1, 'kind': 'dataset_preflight',
        'status': 'unknown', 'components': {'conversion_fidelity': unknown}})
    assert view['status'] == 'unknown' and not view['reference_available']


def test_unavailable_view_keeps_error_and_byte_bound_with_unicode_preview(tmp_path):
    generated = Generated(tmp_path)
    corrupt_action(generated)
    original = generated.check()
    original['findings'][0]['message'] = '\U0001f916'*500
    original['findings'][0]['remedy'] = '\U0001f916'*500
    view = engine.conversion_reference_view({'schema_version': 1, 'kind': 'dataset_preflight',
        'status': 'error', 'components': {'conversion_fidelity': original}}, max_bytes=1024)
    assert view['status'] == 'error' and view['reference_status'] == 'unknown'
    assert view['source_error_summary']['code'].startswith('conversion.')
    assert len(json.dumps(view).encode()) <= 1024


def test_known_original_hash_drift_survives_unsupported_source_subset(tmp_path):
    generated = Generated(tmp_path)
    before = generated.check()
    reference = save(generated, before)
    with h5py.File(generated.original, 'r+') as original:
        del original['data/demo_z/action/base_height_cmd']
    result = restored_check(generated, reference)
    assert not result['conversion_reuse']['criterion_basis_complete']
    assert result['criterion_reuse']['status'] == 'error'
    assert result['criterion_reuse']['matches'] is False
    assert result['numeric_association_status'] == 'unknown'
