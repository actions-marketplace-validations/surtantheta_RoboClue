"""Shared incident lifecycle: native fault, exact correction and retained controls."""
import hashlib
import importlib
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError


def module():
    return importlib.import_module('policy_doctor.incidents')


def dataset(tmp_path, name, bad=True):
    root = tmp_path / name
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'total_frames': 60, 'total_episodes': 2,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {'action': {'dtype': 'float32', 'shape': [2]}, 'state': {'dtype': 'float32', 'shape': [2]}}}
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    (root / 'meta' / 'stats.json').write_text('{"retained":true}\n')
    rows = [{'index': i, 'episode_index': i // 30, 'frame_index': i % 30,
             'action': [float(i), -float(i)], 'state': [1000. + i, 0.]} for i in range(60)]
    meta = [{'episode_index': e, 'length': 30, 'dataset_from_index': e * 30,
             'dataset_to_index': 33 if bad and e == 0 else (e + 1) * 30,
             'data/chunk_index': 0, 'data/file_index': 0, 'tasks': ['retain']} for e in range(2)]
    pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / 'file-000.parquet')
    return root


def trajectory(tmp_path, name, *, bad=True, suffix='.json', initial=0):
    data = {'joint_trajectory': {'joint_names': ['shoulder'], 'points': [
        {'positions': [0.0], 'velocities': [4.0 if bad else 0.0], 'time_from_start': {'sec': initial, 'nanosec': 0}},
        {'positions': [1.0], 'velocities': [0.0], 'time_from_start': {'sec': initial + 1, 'nanosec': 0}}]}}
    path = tmp_path / (name + suffix)
    if suffix == '.json':
        path.write_text(json.dumps(data))
    else:
        yaml = pytest.importorskip('yaml')
        path.write_text('# Original native wrapper and comments.\n' + yaml.safe_dump(data))
    return path


def window_parameters():
    return {'episode_index': 0, 'anchor_index': 29, 'delta_indices': [0, 1, 2, 3, 4]}


def trajectory_parameters():
    return {'sample_times': [.75], 'position_limits': {'shoulder': [0., 1.]}}


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def reseal(root, mutate):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    (root / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


def test_windows_exact_copy_repair_and_relocated_shared_case(tmp_path):
    native = dataset(tmp_path, 'bad')
    before = files(native)
    case = tmp_path / 'case'
    result = module().export_incident_case('episode_windows', str(native), str(case), window_parameters(),
                                          apply_metadata_corrections=True)
    assert result['status'] == 'pass' and result['before_status'] == 'error' and result['after_status'] == 'pass'
    assert result['change_verification']['logical_cells_changed'] == 1
    assert result['change_verification']['non_target_files_unchanged']
    assert result['checks']['before']['query']['declared']['cross_episode_witness_count'] == 3
    assert result['checks']['after']['query']['declared']['cross_episode_witness_count'] == 0
    assert files(native) == before
    assert (case / 'before' / 'meta' / 'stats.json').read_bytes() == (case / 'after' / 'meta' / 'stats.json').read_bytes()
    shutil.rmtree(native)
    moved = tmp_path / 'colleague'
    case.rename(moved)
    snapshot = files(moved)
    replay = module().reproduce_incident_case(str(moved))
    assert replay['status'] == 'pass' and replay['reproduction_matches'] and replay['lifecycle_verified']
    assert replay['checks']['before']['status'] == 'error' and replay['checks']['after']['status'] == 'pass'
    assert files(moved) == snapshot


@pytest.mark.parametrize('suffix', ['.json', '.yaml'])
def test_ros_supplied_correction_and_healthy_unknown_controls_share_same_runner(tmp_path, suffix):
    bad = trajectory(tmp_path, 'bad', suffix=suffix)
    good = trajectory(tmp_path, 'good', bad=False, suffix=suffix)
    unknown = trajectory(tmp_path, 'unknown', initial=2, suffix=suffix)
    controls = [{'name': 'healthy', 'source': str(good), 'parameters': trajectory_parameters(), 'expected_status': 'pass'},
                {'name': 'initial_state_unknown', 'source': str(unknown), 'parameters': trajectory_parameters(), 'expected_status': 'unknown'}]
    case = tmp_path / 'case'
    result = module().export_incident_case('trajectory_sampling', str(bad), str(case), trajectory_parameters(),
                                          corrected=str(good), controls=controls)
    assert result['status'] == 'pass' and result['before_status'] == 'error' and result['after_status'] == 'pass'
    assert (case / 'before' / ('trajectory' + suffix)).read_bytes() == bad.read_bytes()
    assert [c['check']['status'] for c in result['checks']['controls']] == ['pass', 'unknown']
    assert module().reproduce_incident_case(str(case))['status'] == 'pass'


def test_supplied_windows_correction_is_validated_without_rewriting_it(tmp_path):
    before = dataset(tmp_path, 'before')
    after = dataset(tmp_path, 'after', bad=False)
    actual_after = files(after)
    result = module().export_incident_case('episode_windows', str(before), str(tmp_path / 'case'), window_parameters(), corrected=str(after))
    assert result['change_verification']['logical_cells_changed'] == 1
    assert files(after) == actual_after


@pytest.mark.parametrize('mutation', ['healthy_before', 'unknown_before', 'bad_after', 'missing_correction', 'both_corrections'])
def test_incomplete_error_to_pass_lifecycle_is_refused(tmp_path, mutation):
    before = dataset(tmp_path, 'before', bad=mutation != 'healthy_before')
    after = dataset(tmp_path, 'after', bad=mutation == 'bad_after')
    kwargs = {'corrected': str(after)}
    parameters = window_parameters()
    if mutation == 'unknown_before':
        parameters['max_rows'] = 20
    elif mutation == 'missing_correction':
        kwargs = {}
    elif mutation == 'both_corrections':
        kwargs['apply_metadata_corrections'] = True
    with pytest.raises(InputError):
        module().export_incident_case('episode_windows', str(before), str(tmp_path / 'case'), parameters, **kwargs)
    assert not (tmp_path / 'case').exists()


@pytest.mark.parametrize('mutation', ['data', 'other_cell', 'schema', 'extra_file'])
def test_windows_correction_cannot_hide_fault_by_changing_other_evidence(tmp_path, mutation):
    before, after = dataset(tmp_path, 'before'), dataset(tmp_path, 'after', bad=False)
    if mutation == 'data':
        path = after / 'data' / 'chunk-000' / 'file-000.parquet'
        rows = pq.read_table(path).to_pylist()
        rows[0]['action'] = [999., 0.]
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif mutation in ('other_cell', 'schema'):
        path = after / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
        table = pq.read_table(path)
        if mutation == 'other_cell':
            rows = table.to_pylist()
            rows[1]['tasks'] = ['changed']
            table = pa.Table.from_pylist(rows, schema=table.schema)
        else:
            table = table.replace_schema_metadata({b'changed': b'yes'})
        pq.write_table(table, path)
    else:
        (after / 'extra.txt').write_text('extra')
    with pytest.raises(InputError):
        module().export_incident_case('episode_windows', str(before), str(tmp_path / 'case'), window_parameters(), corrected=str(after))


@pytest.mark.parametrize('kwargs', [{'apply_metadata_corrections': 1}, {'check': 'run_shell'},
    {'parameters': {'command': 'do not execute'}}, {'controls': [{'name': 'x'}]}])
def test_strict_declarative_inputs_refuse(tmp_path, kwargs):
    arguments = dict(check='episode_windows', source=str(dataset(tmp_path, 'bad')), destination=str(tmp_path / 'case'),
                     parameters=window_parameters(), apply_metadata_corrections=True)
    arguments.update(kwargs)
    with pytest.raises(InputError):
        module().export_incident_case(**arguments)


def test_nested_destination_and_existing_destination_are_refused(tmp_path):
    source = dataset(tmp_path, 'bad')
    for destination in (source / 'nested-case', source):
        with pytest.raises(InputError):
            module().export_incident_case('episode_windows', str(source), str(destination), window_parameters(), apply_metadata_corrections=True)
    assert not (source / 'nested-case').exists()


def test_source_symlink_escape_is_refused_before_copy(tmp_path):
    source = dataset(tmp_path, 'bad')
    outside = tmp_path / 'outside'
    outside.write_text('private')
    (source / 'escape').symlink_to(outside)
    with pytest.raises(InputError, match='symlink|root|escape'):
        module().export_incident_case('episode_windows', str(source), str(tmp_path / 'case'), window_parameters(), apply_metadata_corrections=True)


@pytest.mark.parametrize('mutation', ['native', 'expected', 'command', 'verifier'])
def test_shared_runner_detects_tamper_and_verifier_drift(tmp_path, mutation):
    root = tmp_path / 'case'
    module().export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(root), window_parameters(), apply_metadata_corrections=True)
    if mutation == 'native':
        path = root / 'before' / 'meta' / 'info.json'
        path.write_bytes(path.read_bytes() + b' ')
        with pytest.raises(InputError, match='hash'):
            module().reproduce_incident_case(str(root))
    elif mutation == 'command':
        reseal(root, lambda m: m.update(command='do not execute'))
        with pytest.raises(InputError):
            module().reproduce_incident_case(str(root))
    elif mutation == 'verifier':
        reseal(root, lambda m: m['verifier'].update(engine_version='old'))
        result = module().reproduce_incident_case(str(root))
        assert result['status'] == 'unknown' and not result['verifier_matches']
    else:
        path = root / 'expected.json'
        expected = json.loads(path.read_text())
        expected['before']['status'] = 'pass'
        raw = json.dumps(expected).encode()
        path.write_bytes(raw)
        reseal(root, lambda m: m['files'].update({'expected.json': hashlib.sha256(raw).hexdigest()}))
        result = module().reproduce_incident_case(str(root))
        assert result['status'] == 'error' and not result['reproduction_matches']


def test_controls_cannot_be_silently_relabeled(tmp_path):
    before, after = trajectory(tmp_path, 'bad'), trajectory(tmp_path, 'good', bad=False)
    unknown = trajectory(tmp_path, 'unknown', initial=2)
    controls = [{'name': 'unknown', 'source': str(unknown), 'parameters': trajectory_parameters(), 'expected_status': 'pass'}]
    with pytest.raises(InputError):
        module().export_incident_case('trajectory_sampling', str(before), str(tmp_path / 'case'), trajectory_parameters(), corrected=str(after), controls=controls)


def test_truncated_metadata_patches_refuse_application(tmp_path, monkeypatch):
    subject = module()
    native = subject.audit_episode_windows

    def truncated(*args, **kwargs):
        result = native(*args, **kwargs)
        result['metadata_patches_truncated'] = True
        return result

    monkeypatch.setattr(subject, 'audit_episode_windows', truncated)
    with pytest.raises(InputError, match='truncat|complete'):
        subject.export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(tmp_path / 'case'), window_parameters(), apply_metadata_corrections=True)


def test_human_control_labels_do_not_choose_storage_paths(tmp_path):
    bad, good = trajectory(tmp_path, 'bad'), trajectory(tmp_path, 'good', bad=False)
    unknown = trajectory(tmp_path, 'unknown', initial=2)
    labels = ['healthy reference', 'initial state absent', '../label is data']
    controls = [{'name': label, 'source': str(unknown if i == 1 else good), 'parameters': trajectory_parameters(),
                 'expected_status': 'unknown' if i == 1 else 'pass'} for i, label in enumerate(labels)]
    root = tmp_path / 'case'
    result = module().export_incident_case('trajectory_sampling', str(bad), str(root), trajectory_parameters(),
                                          corrected=str(good), controls=controls)
    assert [c['name'] for c in result['checks']['controls']] == labels
    assert {p.name for p in (root / 'controls').iterdir()} == {'0', '1', '2'}
    assert module().reproduce_incident_case(str(root))['status'] == 'pass'


def test_native_info_json_alias_uses_the_effective_dataset_root(tmp_path):
    source = dataset(tmp_path, 'bad')
    root = tmp_path / 'case'
    result = module().export_incident_case('episode_windows', str(source / 'meta' / 'info.json'), str(root),
                                          window_parameters(), apply_metadata_corrections=True)
    assert result['after_status'] == 'pass'
    assert (root / 'before' / 'data' / 'chunk-000' / 'file-000.parquet').exists()


def test_info_alias_with_symlinked_metadata_directory_refuses(tmp_path):
    source = dataset(tmp_path, 'bad')
    outside = tmp_path / 'outside-meta'
    (source / 'meta').rename(outside)
    (source / 'meta').symlink_to(outside, target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        module().export_incident_case('episode_windows', str(source / 'meta' / 'info.json'), str(tmp_path / 'case'),
                                      window_parameters(), apply_metadata_corrections=True)


def test_complete_tree_budget_refuses_without_partial_publication_or_original_changes(tmp_path, monkeypatch):
    subject = module()
    source = dataset(tmp_path, 'bad')
    original = files(source)
    payload_size = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    monkeypatch.setattr(subject, 'MAX_TOTAL_BYTES', payload_size + 100)
    with pytest.raises(InputError, match='caps|budget'):
        subject.export_incident_case('episode_windows', str(source), str(tmp_path / 'case'), window_parameters(), apply_metadata_corrections=True)
    assert not (tmp_path / 'case').exists() and not list(tmp_path.glob('.policy-doctor-incident-*'))
    assert files(source) == original


def test_source_change_during_copy_refuses_publication(tmp_path, monkeypatch):
    subject = module()
    source, corrected = trajectory(tmp_path, 'bad'), trajectory(tmp_path, 'good', bad=False)
    original_hash = subject._hash
    changed = False

    def changing(path):
        nonlocal changed
        result = original_hash(path)
        if path == source and not changed:
            changed = True
            source.write_bytes(source.read_bytes() + b' ')
        return result

    monkeypatch.setattr(subject, '_hash', changing)
    with pytest.raises(InputError, match='changed while copying'):
        subject.export_incident_case('trajectory_sampling', str(source), str(tmp_path / 'case'), trajectory_parameters(), corrected=str(corrected))
    assert not (tmp_path / 'case').exists()


def test_changes_during_diagnostic_evaluation_refuse_publication(tmp_path, monkeypatch):
    subject = module()
    source = dataset(tmp_path, 'bad')
    original, changed = subject._run, False

    def changing(check, native, parameters):
        nonlocal changed
        result = original(check, native, parameters)
        if native.name == 'before' and (native.parent / 'after').exists() and not changed:
            changed = True
            for role in ('before', 'after'):
                path = native.parent / role / 'meta' / 'stats.json'
                path.write_bytes(path.read_bytes() + b' ')
        return result

    monkeypatch.setattr(subject, '_run', changing)
    with pytest.raises(InputError, match='changed during diagnostic'):
        subject.export_incident_case('episode_windows', str(source), str(tmp_path / 'case'), window_parameters(), apply_metadata_corrections=True)
    assert not (tmp_path / 'case').exists()


@pytest.mark.parametrize('field', ['observed', 'sha256'])
def test_metadata_application_rechecks_fresh_original_hash_and_cell(tmp_path, monkeypatch, field):
    subject = module()
    original = subject.audit_episode_windows

    def stale(*args, **kwargs):
        report = original(*args, **kwargs)
        if report['metadata_patches']:
            report['metadata_patches'][0][field] = 34 if field == 'observed' else '0' * 64
        return report

    monkeypatch.setattr(subject, 'audit_episode_windows', stale)
    with pytest.raises(InputError, match='stale'):
        subject.export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(tmp_path / 'case'), window_parameters(), apply_metadata_corrections=True)


@pytest.mark.parametrize('names', [['x'] * 5, ['duplicate', 'duplicate'], [''], ['new\nline'], [' padded ']])
def test_control_count_duplicate_and_nonprintable_labels_refuse(tmp_path, names):
    before, after = trajectory(tmp_path, 'bad'), trajectory(tmp_path, 'good', bad=False)
    controls = [{'name': name, 'source': str(after), 'parameters': trajectory_parameters(), 'expected_status': 'pass'} for name in names]
    with pytest.raises(InputError):
        module().export_incident_case('trajectory_sampling', str(before), str(tmp_path / 'case'), trajectory_parameters(), corrected=str(after), controls=controls)


@pytest.mark.parametrize('change', ['check_type', 'after_parameters', 'traversal', 'unused_payload'])
def test_shared_manifest_cannot_add_executable_or_undeclared_semantics(tmp_path, change):
    subject = module()
    root = tmp_path / 'case'
    subject.export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(root), window_parameters(), apply_metadata_corrections=True)

    def mutate(manifest):
        if change == 'check_type':
            manifest['check'] = {'command': 'do not execute'}
        elif change == 'after_parameters':
            manifest['after_parameters'] = {'bounds': 'changed'}
        elif change == 'traversal':
            manifest['before'] = '../outside'
        else:
            path = root / 'unused.py'
            path.write_text('do not execute')
            manifest['files']['unused.py'] = hashlib.sha256(path.read_bytes()).hexdigest()

    reseal(root, mutate)
    with pytest.raises(InputError):
        subject.reproduce_incident_case(str(root))


def test_case_mutation_during_reproduction_refuses_success(tmp_path, monkeypatch):
    subject = module()
    root = tmp_path / 'case'
    subject.export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(root), window_parameters(), apply_metadata_corrections=True)
    original, changed = subject._run, False

    def changing(check, native, parameters):
        nonlocal changed
        result = original(check, native, parameters)
        if not changed:
            changed = True
            (root / 'README.md').write_bytes((root / 'README.md').read_bytes() + b' ')
        return result

    monkeypatch.setattr(subject, '_run', changing)
    with pytest.raises(InputError, match='changed'):
        subject.reproduce_incident_case(str(root))


def test_verifier_drift_takes_unknown_precedence_over_numeric_disagreement(tmp_path):
    root = tmp_path / 'case'
    subject = module()
    subject.export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(root), window_parameters(), apply_metadata_corrections=True)
    expected_path = root / 'expected.json'
    expected = json.loads(expected_path.read_text())
    expected['before']['status'] = 'pass'
    raw = json.dumps(expected).encode()
    expected_path.write_bytes(raw)

    def mutate(manifest):
        manifest['files']['expected.json'] = hashlib.sha256(raw).hexdigest()
        manifest['verifier']['engine_version'] = 'old'

    reseal(root, mutate)
    result = subject.reproduce_incident_case(str(root))
    assert result['status'] == 'unknown' and not result['verifier_matches'] and not result['reproduction_matches']


@pytest.mark.parametrize('regime', ['episode_windows', 'trajectory_sampling'])
def test_retained_case_checks_later_healthy_faulty_and_unknown_artifacts(tmp_path, regime):
    subject = module()
    root = tmp_path / 'case'
    if regime == 'episode_windows':
        bad, good = dataset(tmp_path, 'bad'), dataset(tmp_path, 'good', bad=False)
        unknown = dataset(tmp_path, 'unknown', bad=False)
        (unknown / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet').unlink()
        subject.export_incident_case(regime, str(bad), str(root), window_parameters(), apply_metadata_corrections=True)
    else:
        bad, good = trajectory(tmp_path, 'bad'), trajectory(tmp_path, 'good', bad=False)
        unknown = trajectory(tmp_path, 'unknown', bad=False, initial=2)
        subject.export_incident_case(regime, str(bad), str(root), trajectory_parameters(), corrected=str(good))
    for candidate, expected in [(good, 'pass'), (bad, 'error'), (unknown, 'unknown')]:
        case_before = files(root)
        candidate_before = files(candidate) if candidate.is_dir() else candidate.read_bytes()
        result = subject.check_incident_candidate(str(root), str(candidate))
        assert result['kind'] == 'incident_candidate_check' and result['status'] == expected
        assert result['case_status'] == 'pass' and result['case_verified'] and result['candidate_evaluated']
        assert result['expected_status'] == 'pass' and result['candidate']['status'] == expected
        assert files(root) == case_before
        assert (files(candidate) if candidate.is_dir() else candidate.read_bytes()) == candidate_before


@pytest.mark.parametrize('failure', ['drift', 'reproduction'])
def test_unverified_case_never_evaluates_candidate(tmp_path, monkeypatch, failure):
    subject = module()
    root = tmp_path / 'case'
    subject.export_incident_case('episode_windows', str(dataset(tmp_path, 'bad')), str(root), window_parameters(), apply_metadata_corrections=True)
    candidate = dataset(tmp_path, 'later', bad=False)
    if failure == 'drift':
        reseal(root, lambda m: m['verifier'].update(engine_version='old'))
    else:
        path = root / 'expected.json'
        expected = json.loads(path.read_text())
        expected['before']['status'] = 'pass'
        raw = json.dumps(expected).encode()
        path.write_bytes(raw)
        reseal(root, lambda m: m['files'].update({'expected.json': hashlib.sha256(raw).hexdigest()}))
    original = subject._run

    def guard(check, source, parameters):
        assert source != candidate, 'An unverified case must not evaluate the later candidate'
        return original(check, source, parameters)

    monkeypatch.setattr(subject, '_run', guard)
    result = subject.check_incident_candidate(str(root), str(candidate))
    assert result['status'] in ('unknown', 'error') and not result['candidate_evaluated']
    assert result['candidate'] is None and not result['case_verified']


def test_invalid_case_refuses_before_candidate_access(tmp_path, monkeypatch):
    subject = module()
    source = dataset(tmp_path, 'bad')
    root = tmp_path / 'case'
    subject.export_incident_case('episode_windows', str(source), str(root), window_parameters(), apply_metadata_corrections=True)
    (root / 'READY.json').unlink()
    candidate = tmp_path / 'not-accessed'
    with pytest.raises(InputError):
        subject.check_incident_candidate(str(root), str(candidate))


def test_candidate_parameters_cannot_be_changed_to_hide_the_fault(tmp_path):
    subject = module()
    bad, good = trajectory(tmp_path, 'bad'), trajectory(tmp_path, 'good', bad=False)
    root = tmp_path / 'case'
    subject.export_incident_case('trajectory_sampling', str(bad), str(root), trajectory_parameters(), corrected=str(good))
    result = subject.check_incident_candidate(str(root), str(bad))
    assert result['parameters'] == trajectory_parameters() and result['status'] == 'error'


def test_corrected_windows_with_an_advisory_warning_still_export(tmp_path):
    # Validator round 3: once warnings reach the window status, a corrected dataset holding a single-frame
    # episode (as lerobot/svla_so100_stacking does) was refused as a corrected copy and as a control.
    root = tmp_path / 'bad'
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    lengths = (30, 1, 30)
    info = {'codebase_version': 'v3.0', 'total_frames': sum(lengths), 'total_episodes': 3,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {'action': {'dtype': 'float32', 'shape': [2]}}}
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    rows, meta, start = [], [], 0
    for ep, length in enumerate(lengths):
        rows += [{'index': start + i, 'episode_index': ep, 'frame_index': i, 'action': [float(start + i), 0.]} for i in range(length)]
        meta.append({'episode_index': ep, 'length': length, 'dataset_from_index': start,
                     'dataset_to_index': 29 if ep == 0 else start + length, 'data/chunk_index': 0, 'data/file_index': 0})
        start += length
    pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / 'file-000.parquet')
    # Whole-dataset parameters ({}), exactly as the validator ran it, so the episode consistency pass runs.
    result = module().export_incident_case('episode_windows', str(root), str(tmp_path / 'case'), {},
                                          apply_metadata_corrections=True)
    assert result['status'] == 'pass' and result['before_status'] == 'error' and result['after_status'] == 'warning'


def test_check_case_expects_the_retained_corrected_status(tmp_path):
    # Validator round 4: check-case hard-coded expected_status 'pass' for cases corrected to 'warning'.
    root = tmp_path / 'bad'
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    lengths = (30, 1, 30)
    (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'total_frames': sum(lengths),
        'total_episodes': 3, 'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'features': {'action': {'dtype': 'float32', 'shape': [2]}}}))
    rows, meta, start = [], [], 0
    for ep, length in enumerate(lengths):
        rows += [{'index': start + i, 'episode_index': ep, 'frame_index': i, 'action': [float(start + i), 0.]} for i in range(length)]
        meta.append({'episode_index': ep, 'length': length, 'dataset_from_index': start,
                     'dataset_to_index': 29 if ep == 0 else start + length, 'data/chunk_index': 0, 'data/file_index': 0})
        start += length
    pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / 'file-000.parquet')
    case = tmp_path / 'case'
    module().export_incident_case('episode_windows', str(root), str(case), {}, apply_metadata_corrections=True)
    candidate = tmp_path / 'candidate'
    shutil.copytree(case / 'after', candidate)
    result = module().check_incident_candidate(str(case), str(candidate))
    assert result['status'] == 'warning' and result['expected_status'] == 'warning'
