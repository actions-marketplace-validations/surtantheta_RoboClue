"""Recorded settings must remain literal, bounded, and distinct from live context."""

import hashlib
import importlib
import json
import os
import stat
import sys
from contextlib import contextmanager

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_context import audit_native_run_context, validate_parameters


def make_run(tmp_path, name, document=None):
    root = tmp_path / name
    root.mkdir()
    (root / 'env_cfg.json').write_text(json.dumps(document if document is not None else {
        'instruction': 'Put the banana in the bowl', 'actions': {'body': {'scale': 1.0}},
        'terminations': {'success': {'threshold': 0.1}}, 'decimation': 8, 'num_envs': 1,
        'sim': {'dt': 1 / 120, 'device': 'cuda:0'}, 'scene': {'num_envs': 1},
    }))
    return root


def audit(candidate, reference=None, fields=None):
    return audit_native_run_context(candidate, reference=reference,
                                    equal_fields=fields or ['/instruction', '/actions/body/scale'])


def test_consequential_literal_drift_has_before_after_evidence(tmp_path):
    reference = make_run(tmp_path, 'reference')
    candidate = make_run(tmp_path, 'candidate')
    path = candidate / 'env_cfg.json'
    value = json.loads(path.read_text())
    value['actions']['body']['scale'] = 0.5
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    report = audit(candidate, reference)
    assert report['status'] == report['contract_status'] == 'error'
    witness = next(field for field in report['fields'] if field['status'] == 'error')
    assert witness['pointer'] == '/actions/body/scale'
    assert witness['reference']['value'] == 1.0 and witness['candidate']['value'] == 0.5
    assert witness['candidate']['sha256'] == hashlib.sha256(before).hexdigest()
    assert witness['candidate']['pointer'] == '/actions/body/scale'
    assert path.read_bytes() == before
    assert report['findings'][0]['code'] == 'recorded_context_mismatch'
    json.dumps(report, allow_nan=False)


def test_self_and_reference_pass_only_selected_settings(tmp_path):
    reference = make_run(tmp_path, 'reference')
    candidate = make_run(tmp_path, 'candidate')
    path = candidate / 'env_cfg.json'
    value = json.loads(path.read_text())
    value['sim']['device'] = 'cuda:7'
    value['recorders'] = {'output': '/different/legal/location'}
    path.write_text(json.dumps(value))
    assert audit(reference, reference)['status'] == 'pass'
    report = audit(candidate, reference)
    assert report['status'] == 'pass' and not report['findings']
    assert 'effective runtime configuration' in report['unverified']
    assert 'cross-run episode/cohort association' in report['unverified']


def test_missing_reference_and_missing_fields_are_unknown_with_advice(tmp_path):
    candidate = make_run(tmp_path, 'candidate')
    report = audit(candidate)
    assert report['status'] == 'unknown'
    assert any(item['code'] == 'reference_missing' for item in report['findings'])
    assert 'Supply' in report['next_action']
    report = audit(candidate, candidate, ['/sim/missing'])
    assert report['status'] == 'unknown'
    assert report['fields'][0]['candidate']['present'] is False


def test_known_mismatch_dominates_missing_field_and_callable_unknown(tmp_path):
    reference = make_run(tmp_path, 'reference', {'instruction': 'old', 'actions': '<function same>'})
    candidate = make_run(tmp_path, 'candidate', {'instruction': 'new', 'actions': '<function same>'})
    report = audit(candidate, reference, ['/instruction', '/actions', '/sim/missing'])
    assert report['status'] == 'error'
    assert [field['status'] for field in report['fields']] == ['error', 'unknown', 'unknown']


@pytest.mark.parametrize('representation', [
    'functools.partial(some_callable, threshold=0.1)', '<function predicate>',
    '<function predicate at 0x123ab>', '<bound method Condition.check>',
    {'safe': 1, 'condition': ['functools.partial(<function f>)']},
])
def test_callable_representations_are_inert_and_unknown_even_when_equal(tmp_path, representation):
    root = make_run(tmp_path, 'candidate', {'terminations': representation})
    report = audit(root, root, ['/terminations'])
    assert report['status'] == 'unknown'
    assert 'inert text' in report['fields'][0]['reason']


def test_callable_text_never_executes_and_unselected_callable_does_not_poison_literals(tmp_path):
    marker = tmp_path / 'executed'
    code = f"functools.partial(__import__('pathlib').Path({str(marker)!r}).touch)"
    root = make_run(tmp_path, 'candidate', {'instruction': 'same', 'terminations': {'condition': code}})
    assert audit(root, root, ['/instruction'])['status'] == 'pass'
    assert audit(root, root, ['/terminations'])['status'] == 'unknown'
    assert not marker.exists()


def test_strict_pointer_escaping_arrays_and_exact_json_type_equality(tmp_path):
    reference = make_run(tmp_path, 'reference', {'actions': {'a/b~c': [1]}})
    candidate = make_run(tmp_path, 'candidate', {'actions': {'a/b~c': [True]}})
    assert audit(candidate, reference, ['/actions/a~1b~0c/0'])['status'] == 'error'
    assert audit(reference, reference, ['/actions/a~1b~0c/00'])['status'] == 'unknown'


@pytest.mark.parametrize('parameters', [
    None, [], {}, {'equal_fields': []}, {'equal_fields': '/sim'}, {'equal_fields': [True]},
    {'equal_fields': ['sim']}, {'equal_fields': ['/']}, {'equal_fields': ['/policy']},
    {'equal_fields': ['/simulator']}, {'equal_fields': ['/sim/~2']}, {'equal_fields': ['/sim/~']},
    {'equal_fields': ['/sim/\n']}, {'equal_fields': ['/sim', '/sim']},
    {'equal_fields': ['/sim'], 'reference': 'other'}, {'equal_fields': ['/sim'] * 65},
])
def test_parameter_validation_rejects_ambiguous_or_unsupported_contracts(parameters):
    with pytest.raises(InputError):
        validate_parameters(parameters)


def test_parameter_validation_returns_an_independent_list():
    fields = ['/instruction', '/actions', '/terminations', '/sim', '/decimation', '/scene', '/num_envs']
    normalized = validate_parameters({'equal_fields': fields})
    assert normalized == {'equal_fields': fields}
    assert normalized['equal_fields'] is not fields


@pytest.mark.parametrize('text', ['[]', '{broken', '{"sim":1,"sim":2}', '{"sim":NaN}', '{"sim":1e999}'])
def test_malformed_recorded_config_never_passes(tmp_path, text):
    root = make_run(tmp_path, 'candidate')
    (root / 'env_cfg.json').write_text(text)
    assert audit(root, root)['status'] == 'unknown'


@pytest.mark.parametrize('budget', ['MAX_JSON_BYTES', 'MAX_JSON_DEPTH', 'MAX_JSON_ENTRIES', 'MAX_FILES',
                                    'MAX_FILE_BYTES', 'MAX_TOTAL_BYTES', 'MAX_DISCOVERY_ENTRIES'])
def test_input_budgets_fail_closed(tmp_path, monkeypatch, budget):
    root = make_run(tmp_path, 'candidate')
    module = importlib.import_module('policy_doctor.native_context')
    monkeypatch.setattr(module, budget, 0)
    assert audit(root, root)['status'] == 'unknown'


@pytest.mark.parametrize('link', ['root', 'parent', 'config'])
def test_filesystem_links_are_refused(tmp_path, link):
    root = make_run(tmp_path, 'candidate')
    if link == 'root':
        path = tmp_path / 'linked'
        path.symlink_to(root, target_is_directory=True)
        root = path
    elif link == 'parent':
        path = tmp_path / 'parent'
        path.symlink_to(tmp_path, target_is_directory=True)
        root = path / 'candidate'
    elif link == 'config':
        source = root / 'env_cfg.json'
        source.rename(tmp_path / 'outside.json')
        source.symlink_to(tmp_path / 'outside.json')
    assert audit(root, root)['status'] == 'unknown'


def test_filesystem_hardlink_staging_preserves_identity_and_is_supported(tmp_path):
    root = make_run(tmp_path, 'candidate')
    os.link(root / 'env_cfg.json', tmp_path / 'verified-original.json')
    assert audit(root, root)['status'] == 'pass'


def add_hdf5(root, *, multi=False):
    h5py = pytest.importorskip('h5py')
    with h5py.File(root / 'run_0.hdf5', 'w') as handle:
        data = handle.create_group('data')
        data.attrs['isaaclab_version'] = '2.2.0'
        for i in range(2 if multi else 1):
            demo = data.create_group(f'demo_{i}')
            demo.attrs['success'] = i == 0
            demo.attrs['num_samples'] = 648
            demo.create_dataset('actions', data=[[1.0, 2.0]])
    return h5py


def test_metadata_only_outcomes_are_distinct_within_file_episode_evidence(tmp_path, monkeypatch):
    root = make_run(tmp_path, 'candidate')
    h5py = add_hdf5(root, multi=True)
    (root / 'log_0.json').write_text(json.dumps([{}] * 530))
    def no_dataset_reads(*args, **kwargs):
        raise AssertionError('Dataset values must never be read')
    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_dataset_reads)
    # Exercise the exact worker body under the read-denying witness too; the
    # public audit now runs that body in a separate isolated process.
    raw = importlib.import_module('policy_doctor.native_context')._hdf5_metadata(root / 'run_0.hdf5')
    assert raw['dataset_value_reads'] == 0
    report = audit(root, root)
    assert report['status'] == 'pass'
    outcomes = report['recorded_outcomes']['candidate']
    assert [item['episode'] for item in outcomes] == ['demo_0', 'demo_1']
    assert [item['attrs']['success'] for item in outcomes] == [True, False]
    assert all(item['cross_run_episode_association'] == 'unknown' for item in outcomes)
    assert all(item['episode_association'] == 'within_file_only' for item in outcomes)
    assert all(item['sha256'] for item in outcomes)
    meta = report['native_metadata']['candidate'][0]
    assert meta['dataset_value_reads'] == 0
    assert any(attr['name'] == 'isaaclab_version' for item in meta['provenance']
               for attr in item['undecoded_attrs'])
    assert any(item['name'] == 'isaaclab_version' for item in report['unverified_attributes']['candidate'])
    assert not report['findings']  # No 530-vs-648 fault and no claimed cohort alignment.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('kind', ['soft', 'external_link', 'hard_alias', 'cycle', 'external_storage', 'virtual'])
def test_hdf5_links_and_nonlocal_storage_are_refused(tmp_path, kind):
    root = make_run(tmp_path, 'candidate')
    h5py = add_hdf5(root)
    with h5py.File(root / 'run_0.hdf5', 'a') as handle:
        if kind == 'soft':
            handle['link'] = h5py.SoftLink('/data')
        elif kind == 'external_link':
            handle['link'] = h5py.ExternalLink('other.hdf5', '/data')
        elif kind == 'hard_alias':
            handle['link'] = handle['data']
        elif kind == 'cycle':
            handle['data']['cycle'] = handle['data']
        elif kind == 'external_storage':
            handle.create_dataset('external', (1,), dtype='f4', external=[('outside.bin', 0, 4)])
        else:
            layout = h5py.VirtualLayout(shape=(1,), dtype='f4')
            layout[:] = h5py.VirtualSource('other.hdf5', 'data', shape=(1,))
            handle.create_virtual_dataset('virtual', layout)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert not report['recorded_outcomes']['candidate']


@pytest.mark.parametrize('budget', ['MAX_OBJECTS', 'MAX_HDF5_DEPTH', 'MAX_ATTRS', 'MAX_ATTR_BYTES',
                                    'MAX_TOTAL_ATTRS', 'MAX_TOTAL_ATTR_BYTES'])
def test_hdf5_metadata_bounds(tmp_path, monkeypatch, budget):
    root = make_run(tmp_path, 'candidate')
    add_hdf5(root)
    module = importlib.import_module('policy_doctor.native_context')
    monkeypatch.setattr(module, budget, 0)
    # Lower the constants in the worker algorithm itself; parent monkeypatches
    # intentionally cannot alter the isolated worker's production constants.
    with pytest.raises(ValueError):
        module._hdf5_metadata(root / 'run_0.hdf5')


def test_optional_h5py_failure_is_unknown_without_importing_simulator(tmp_path, monkeypatch):
    root = make_run(tmp_path, 'candidate')
    (root / 'run_0.hdf5').write_bytes(b'not hdf5')
    monkeypatch.setitem(sys.modules, 'h5py', None)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert any(item['code'] == 'native_source_unknown' for item in report['findings'])


def test_nested_native_task_folders_keep_file_association(tmp_path):
    reference = tmp_path / 'reference'
    candidate = tmp_path / 'candidate'
    reference.mkdir()
    candidate.mkdir()
    for parent in (reference, candidate):
        make_run(parent, 'task_a', {'instruction': 'a'})
        make_run(parent, 'task_b', {'instruction': 'b'})
    (candidate / 'task_b' / 'env_cfg.json').write_text('{"instruction":"changed"}')
    report = audit(candidate, reference, ['/instruction'])
    assert report['status'] == 'error'
    mismatch = [item for item in report['fields'] if item['status'] == 'error']
    assert len(mismatch) == 1 and mismatch[0]['config'] == 'task_b/env_cfg.json'


def test_changed_native_source_invalidates_its_field_evidence(tmp_path, monkeypatch):
    root = make_run(tmp_path, 'candidate')
    module = importlib.import_module('policy_doctor.native_context')
    original_hash = module._hash
    calls = 0
    def changing_hash(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / 'env_cfg.json').write_text('{"instruction":"changed while reading"}')
        return original_hash(path)
    monkeypatch.setattr(module, '_hash', changing_hash)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert any(item['code'] == 'native_source_changed' for item in report['findings'])


def test_large_selected_subtree_is_compared_exactly_with_bounded_evidence(tmp_path):
    reference = make_run(tmp_path, 'reference', {'actions': {'padding': 'x' * 10_000, 'scale': 1}})
    candidate = make_run(tmp_path, 'candidate', {'actions': {'padding': 'x' * 10_000, 'scale': 2}})
    report = audit(candidate, reference, ['/actions'])
    assert report['status'] == 'error'
    assert report['fields'][0]['candidate']['value_truncated']
    assert 'value' not in report['fields'][0]['candidate']
    assert report['fields'][0]['candidate']['canonical_sha256'] != report['fields'][0]['reference']['canonical_sha256']


@pytest.mark.parametrize('directory', [None, 42, [], {}])
def test_invalid_directory_input_returns_unknown_without_traceback(tmp_path, directory):
    reference = make_run(tmp_path, 'reference')
    assert audit(directory, reference)['status'] == 'unknown'


def test_missing_reference_evidence_never_invents_a_source_hash(tmp_path):
    from policy_doctor.reports import markdown_report

    root = make_run(tmp_path, 'candidate')
    report = audit(root)
    assert report['fields'][0]['reference']['present'] is False
    for finding in report['findings']:
        for evidence in finding['evidence']:
            assert evidence['path'] and evidence['sha256']
    rendered = markdown_report(report)
    assert 'unknown' in rendered and 'Supply a reference native run folder' in rendered


def test_same_bytes_with_replaced_inode_invalidate_a_changed_snapshot(tmp_path, monkeypatch):
    root = make_run(tmp_path, 'candidate')
    path = root / 'env_cfg.json'
    module = importlib.import_module('policy_doctor.native_context')
    original_read = module._read_config
    def changing_read(source):
        value = original_read(source)
        original_inode = path.stat().st_ino
        replacement = path.with_name('replacement.json')
        replacement.write_bytes(path.read_bytes())
        # A distinct inode makes the observed change deterministic on coarse timestamp filesystems.
        os.replace(replacement, path)
        assert path.stat().st_ino != original_inode
        return value
    monkeypatch.setattr(module, '_read_config', changing_read)
    assert audit(root, root)['status'] == 'unknown'


def test_added_native_sidecar_invalidates_directory_snapshot(tmp_path, monkeypatch):
    root = make_run(tmp_path, 'candidate')
    module = importlib.import_module('policy_doctor.native_context')
    original_read = module._read_config
    def adding_read(source):
        value = original_read(source)
        (root / 'log_0.json').write_text('[]')
        return value
    monkeypatch.setattr(module, '_read_config', adding_read)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert any(item['code'] == 'native_inventory_changed' for item in report['findings'])


@pytest.mark.parametrize('nested', [False, True])
def test_directory_change_invalidates_a_stale_native_listing(tmp_path, monkeypatch, nested):
    root = make_run(tmp_path, 'candidate')
    destination = root / 'observations' if nested else root
    destination.mkdir(exist_ok=True)
    module = importlib.import_module('policy_doctor.native_context')
    original_read, original_scandir = module._read_config, os.scandir

    @contextmanager
    def stale_scandir(directory):
        with original_scandir(directory) as entries:
            yield (item for item in entries if item.name != 'log_0.json')

    def adding_read(source):
        value = original_read(source)
        before = destination.stat()
        (destination / 'log_0.json').write_text('[]')
        # Preserve an independent directory-state witness even when enumeration
        # is stale or the filesystem has coarse metadata timestamps.
        os.utime(destination, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return value

    monkeypatch.setattr(module.os, 'scandir', stale_scandir)
    monkeypatch.setattr(module, '_read_config', adding_read)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert any(item['code'] == 'native_inventory_changed' for item in report['findings'])
    assert (destination / 'log_0.json').exists()


def test_preexisting_native_sidecar_keeps_a_stable_snapshot_healthy(tmp_path):
    root = make_run(tmp_path, 'candidate')
    nested = root / 'observations'
    nested.mkdir()
    (nested / 'log_0.json').write_text('[]')
    report = audit(root, root)
    assert report['status'] == 'pass' and not report['findings']


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('operation', ['add', 'delete', 'replace', 'rewrite'])
def test_native_events_reject_changes_with_stale_listing_and_directory_metadata(
    tmp_path, monkeypatch, nested, operation,
):
    root = make_run(tmp_path, 'candidate')
    destination = root / 'observations' if nested else root
    destination.mkdir(exist_ok=True)
    sidecar = destination / 'log_0.json'
    if operation != 'add':
        sidecar.write_text('[]')
    module = importlib.import_module('policy_doctor.native_context')
    original_read, original_scandir, original_signature = module._read_config, os.scandir, module._signature

    @contextmanager
    def stale_scandir(directory):
        with original_scandir(directory) as entries:
            yield (item for item in entries if item.name != 'log_0.json')

    def stale_directory_signature(info):
        return (0,) * 6 if stat.S_ISDIR(info.st_mode) else original_signature(info)

    changed = False

    def changing_read(source):
        nonlocal changed
        value = original_read(source)
        if not changed:
            changed = True
            if operation == 'delete':
                sidecar.unlink()
            elif operation == 'replace':
                temporary = destination / 'replacement'
                temporary.write_text('[]')
                temporary.replace(sidecar)
            else:
                sidecar.write_text('[]')
        return value

    monkeypatch.setattr(module.os, 'scandir', stale_scandir)
    monkeypatch.setattr(module, '_signature', stale_directory_signature)
    monkeypatch.setattr(module, '_read_config', changing_read)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert any(item['code'] == 'native_inventory_changed' for item in report['findings'])


def test_unavailable_native_event_monitor_is_unknown_without_a_pass(tmp_path, monkeypatch):
    root = make_run(tmp_path, 'candidate')
    module = importlib.import_module('policy_doctor.native_context')

    def unavailable():
        raise OSError('monitor unavailable')

    monkeypatch.setattr(module, '_DirectoryEvents', unavailable)
    report = audit(root, root)
    assert report['status'] == 'unknown'
    assert not report['fields']
    assert any(item['code'] == 'native_intake_unknown' for item in report['findings'])


def test_native_intake_closes_event_descriptors_on_success_and_refusal(tmp_path):
    root = make_run(tmp_path, 'candidate')
    before = set(os.listdir('/proc/self/fd'))
    for _ in range(10):
        assert audit(root, root)['status'] == 'pass'
        assert audit(root / 'missing', root)['status'] == 'unknown'
    assert set(os.listdir('/proc/self/fd')) == before
