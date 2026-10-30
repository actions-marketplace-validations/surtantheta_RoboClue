"""Whole-result retained replay and later native candidate acceptance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from policy_doctor import retained
from policy_doctor.artifacts import InputError


def recording(path, *, value=1., missing=False):
    arrays = {'index': np.arange(3, dtype=np.int64), 'action': np.array([[value], [2.], [3.]])}
    if missing:
        arrays.pop('action')
    np.savez(path, **arrays)
    return path


def frozen(tmp_path, *, value=1., parameters=None):
    reference = recording(tmp_path / 'reference.npz')
    source = recording(tmp_path / 'source.npz', value=value)
    directory = tmp_path / 'retained'
    parameters = {'alignment_key': 'index'} if parameters is None else parameters
    result = retained.retain_check('recording_equivalence', source, directory, reference=reference, parameters=parameters)
    return directory, source, reference, result


def resign(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in manifest['files']:
        manifest['files'][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    raw = retained._encode(manifest)
    (directory / 'manifest.json').write_bytes(raw)
    (directory / 'READY.json').write_bytes(retained._encode({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


@pytest.mark.parametrize(('value', 'parameters', 'status'), [(1., {'alignment_key': 'index'}, 'pass'),
                                                          (9., {'alignment_key': 'index'}, 'different'),
                                                          (1., {}, 'unknown')])
def test_retention_replays_truth_after_relocation(tmp_path, value, parameters, status):
    directory, source, reference, result = frozen(tmp_path, value=value, parameters=parameters)
    originals = {p: p.read_bytes() for p in (source, reference)}
    assert result['status'] == 'pass' and result['original_status'] == status
    assert result['check_result']['status'] == status
    moved = tmp_path / 'colleague'
    directory.rename(moved)
    for path, raw in originals.items():
        assert path.read_bytes() == raw
        path.unlink()
    replay = retained.reproduce_retained_check(moved)
    assert replay['status'] == 'pass'
    assert replay['original_status'] == status
    assert replay['reproduction_matches'] and replay['verifier_matches']
    assert replay['current_report_sha256'] == replay['recorded_report_sha256']


@pytest.mark.parametrize(('value', 'missing', 'status'), [(1., False, 'pass'), (9., False, 'different'), (1., True, 'unknown')])
def test_later_candidate_same_frozen_reference(tmp_path, value, missing, status):
    directory, _, reference, _ = frozen(tmp_path, value=8.)
    later = recording(tmp_path / 'later.npz', value=value, missing=missing)
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    raw = later.read_bytes()
    result = retained.check_retained_candidate(directory, later)
    assert result['candidate_evaluated'] and result['candidate_status'] == status
    assert result['status'] == status
    assert all(p.read_bytes() == data for p, data in before.items())
    assert later.read_bytes() == raw
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == json.loads((directory / 'provenance.json').read_text())['reference']['original_sha256']


def test_full_report_changes_same_status_counts_detected(tmp_path, monkeypatch):
    directory, _, _, _ = frozen(tmp_path)
    original = retained.compare_rollouts
    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        report['alignment']['reason'] = 'Changed finding with unchanged verdict and stream counts'
        return report
    monkeypatch.setattr(retained, 'compare_rollouts', changed)
    result = retained.reproduce_retained_check(directory)
    assert result['status'] == 'error' and not result['reproduction_matches']
    later = recording(tmp_path / 'later.npz')
    result = retained.check_retained_candidate(directory, later)
    assert result['status'] == 'unknown' and not result['candidate_evaluated']


@pytest.mark.parametrize('parameters', [{'atol': True}, {'rtol': float('inf')}, {'max_rows': 0},
                                      {'streams': ['action'], 'alignment_key': 'index'}, {'wat': 1},
                                      {'candidate_episode': True}, {'streams': []}, {'streams': ['index', 'index']}])
def test_invalid_parameters_never_publish(tmp_path, parameters):
    reference = recording(tmp_path / 'reference.npz')
    source = recording(tmp_path / 'source.npz')
    directory = tmp_path / 'case'
    with pytest.raises(InputError):
        retained.retain_check('recording_equivalence', source, directory, reference=reference, parameters=parameters)
    assert not directory.exists()


@pytest.mark.parametrize('mode', ['payload', 'expected', 'ready', 'extra', 'empty_directory', 'symlink', 'manifest_keys'])
def test_tamper_refused(tmp_path, mode):
    directory, _, _, _ = frozen(tmp_path)
    if mode == 'payload':
        (directory / 'candidate' / 'recording.npz').write_bytes(b'changed')
    elif mode == 'expected':
        (directory / 'expected.json').write_text('{}')
    elif mode == 'ready':
        (directory / 'READY.json').write_text('{}')
    elif mode == 'extra':
        (directory / 'extra').write_text('unlisted')
    elif mode == 'empty_directory':
        (directory / 'extra').mkdir()
    elif mode == 'symlink':
        (directory / 'extra').symlink_to(directory / 'expected.json')
    else:
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['extra'] = True
        (directory / 'manifest.json').write_bytes(retained._encode(manifest))
        resign(directory)
    with pytest.raises(InputError):
        retained.reproduce_retained_check(directory)


def test_verifier_drift_blocks_replay_and_later_before_intake(tmp_path, monkeypatch):
    directory, _, _, _ = frozen(tmp_path)
    original = retained._fingerprint
    monkeypatch.setattr(retained, '_fingerprint', lambda manifest: original(manifest) | {'engine_version': 'different'})
    result = retained.reproduce_retained_check(directory)
    assert result['status'] == 'unknown' and not result['verifier_matches']
    result = retained.check_retained_candidate(directory, tmp_path / 'does-not-exist.npz')
    assert result['status'] == 'unknown' and not result['candidate_evaluated']


def test_overwrite_and_nested_destination_refused(tmp_path):
    directory, source, reference, _ = frozen(tmp_path)
    marker = (directory / 'READY.json').read_bytes()
    with pytest.raises(InputError):
        retained.retain_check('recording_equivalence', source, directory, reference=reference)
    assert (directory / 'READY.json').read_bytes() == marker
    root = tmp_path / 'source-tree'
    root.mkdir()
    with pytest.raises(InputError):
        retained.retain_check('policy_metadata', root, root / 'nested')
    assert not (root / 'nested').exists()


def test_racing_destination_is_preserved(tmp_path, monkeypatch):
    source, reference = recording(tmp_path / 'source.npz'), recording(tmp_path / 'reference.npz')
    directory = tmp_path / 'case'
    publish = retained.publish_capture
    def race(staging, target):
        target.mkdir()
        (target / 'other-owner').write_text('preserve')
        publish(staging, target)
    monkeypatch.setattr(retained, 'publish_capture', race)
    with pytest.raises(InputError):
        retained.retain_check('recording_equivalence', source, directory, reference=reference)
    assert list(p.name for p in directory.iterdir()) == ['other-owner']


def test_source_mutation_during_copy_refused(tmp_path, monkeypatch):
    source, reference = recording(tmp_path / 'source.npz'), recording(tmp_path / 'reference.npz')
    copy = retained.shutil.copyfile
    def changed(src, dest):
        result = copy(src, dest)
        if Path(src) == source:
            source.write_bytes(source.read_bytes() + b'changed')
        return result
    monkeypatch.setattr(retained.shutil, 'copyfile', changed)
    with pytest.raises(InputError):
        retained.retain_check('recording_equivalence', source, tmp_path / 'case', reference=reference)
    assert not (tmp_path / 'case').exists()


def test_replay_mutation_refused(tmp_path, monkeypatch):
    directory, _, _, _ = frozen(tmp_path)
    original = retained.compare_rollouts
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (directory / 'expected.json').write_text('{}')
        return result
    monkeypatch.setattr(retained, 'compare_rollouts', changed)
    with pytest.raises(InputError):
        retained.reproduce_retained_check(directory)


@pytest.mark.parametrize('suffix', ['.csv', '.hdf5', '.parquet'])
def test_native_formats_preserve_bytes_and_episode_selection(tmp_path, suffix):
    def write(path, value):
        if suffix == '.csv':
            path.write_text(f'index,action\n0,{value}\n1,2\n2,3\n')
        elif suffix == '.hdf5':
            h5py = pytest.importorskip('h5py')
            with h5py.File(path, 'w') as file:
                for i in range(2):
                    demo = file.create_group(f'data/demo_{i}')
                    demo['index'] = np.arange(3)
                    demo['actions'] = np.array([[value if i == 1 else 999.], [2.], [3.]])
        else:
            pa = pytest.importorskip('pyarrow')
            pq = pytest.importorskip('pyarrow.parquet')
            pq.write_table(pa.table({'index': [0, 1, 2, 3, 4, 5], 'episode_index': [0, 0, 0, 1, 1, 1],
                                     'action': [[999.], [2.], [3.], [value], [2.], [3.]]}), path)
        return path
    reference, source = write(tmp_path / ('reference' + suffix), 1.), write(tmp_path / ('source' + suffix), 7.)
    parameters = {'alignment_key': 'index'}
    if suffix != '.csv':
        parameters |= {'reference_episode': 1, 'candidate_episode': 1}
    raw = {p: p.read_bytes() for p in (reference, source)}
    directory = tmp_path / 'native'
    result = retained.retain_check('recording_equivalence', source, directory, reference=reference, parameters=parameters)
    assert result['original_status'] == 'different'
    assert all(p.read_bytes() == b for p, b in raw.items())
    assert (directory / 'candidate' / ('recording' + suffix)).read_bytes() == raw[source]
    assert retained.reproduce_retained_check(directory)['status'] == 'pass'
    later = write(tmp_path / ('later' + suffix), 1.)
    assert retained.check_retained_candidate(directory, later)['status'] == 'pass'


def test_explicit_missing_stream_selection_is_not_pass(tmp_path):
    # Preserve the native auditor's stronger refusal for an explicitly requested
    # absent NPZ stream; do not manufacture a replayable compatibility verdict.
    with pytest.raises(InputError, match='streams are missing'):
        frozen(tmp_path, parameters={'alignment_key': 'index', 'streams': ['index', 'action', 'absent']})
    assert not (tmp_path / 'retained').exists()


@pytest.mark.parametrize(('dataset_name', 'scope', 'status'), [('corrected_dataset', 'metadata', 'pass'),
                                                            ('dataset', 'metadata', 'error'),
                                                            ('corrected_dataset', 'deployment', 'unknown')])
def test_metadata_pair_relocation_and_later_frozen_role(tmp_path, dataset_name, scope, status):
    from policy_doctor.demo import create_policy_demo
    pytest.importorskip('safetensors')
    demo = create_policy_demo(tmp_path / 'demo')
    checkpoint, dataset = Path(demo['checkpoint']), Path(demo[dataset_name])
    original = {p: p.read_bytes() for root in (checkpoint, dataset) for p in root.rglob('*') if p.is_file()}
    directory = tmp_path / 'retained'
    result = retained.retain_check('policy_metadata', checkpoint, directory, dataset=dataset, parameters={'scope': scope})
    assert result['original_status'] == status
    assert result['candidate_role'] == 'dataset'
    assert all(p.read_bytes() == b for p, b in original.items())
    moved = tmp_path / 'colleague'
    directory.rename(moved)
    assert retained.reproduce_retained_check(moved)['status'] == 'pass'
    later = retained.check_retained_candidate(moved, Path(demo['corrected_dataset']))
    assert later['candidate_status'] == ('pass' if scope == 'metadata' else 'unknown')
    assert later['candidate_role'] == 'dataset'
    broken = retained.check_retained_candidate(moved, Path(demo['dataset']))
    assert broken['status'] == 'error'


def test_metadata_checkpoint_only_and_explicit_replaceable_role(tmp_path):
    from policy_doctor.demo import create_policy_demo
    pytest.importorskip('safetensors')
    demo = create_policy_demo(tmp_path / 'demo')
    checkpoint = Path(demo['checkpoint'])
    result = retained.retain_check('policy_metadata', checkpoint, tmp_path / 'single')
    assert result['candidate_role'] == 'checkpoint'
    assert retained.reproduce_retained_check(tmp_path / 'single')['status'] == 'pass'
    pair = tmp_path / 'pair'
    retained.retain_check('policy_metadata', checkpoint, pair, dataset=demo['corrected_dataset'], candidate_role='checkpoint')
    assert retained.check_retained_candidate(pair, checkpoint)['candidate_status'] == 'pass'


def test_metadata_absent_normalization_state_never_becomes_pass(tmp_path):
    from policy_doctor.demo import create_policy_demo
    pytest.importorskip('safetensors')
    demo = create_policy_demo(tmp_path / 'demo')
    checkpoint = Path(demo['checkpoint'])
    (checkpoint / 'pre.safetensors').unlink()
    (checkpoint / 'post.safetensors').unlink()
    directory = tmp_path / 'unknown'
    result = retained.retain_check('policy_metadata', checkpoint, directory, dataset=demo['corrected_dataset'])
    assert result['original_status'] == 'error'
    expected = json.loads((directory / 'expected.json').read_text())
    assert expected['status'] != 'pass'
    assert retained.reproduce_retained_check(directory)['status'] == 'pass'


@pytest.mark.parametrize('role', ['', False, 0, [], 'reference'])
def test_invalid_replaceable_role_refused(tmp_path, role):
    source, reference = recording(tmp_path / 'source.npz'), recording(tmp_path / 'reference.npz')
    with pytest.raises(InputError):
        retained.retain_check('recording_equivalence', source, tmp_path / 'case', reference=reference, candidate_role=role)
    assert not (tmp_path / 'case').exists()


def test_same_bytes_source_replacement_during_copy_refused(tmp_path, monkeypatch):
    source, reference = recording(tmp_path / 'source.npz'), recording(tmp_path / 'reference.npz')
    copy = retained.shutil.copyfile
    def replaced(src, dest):
        result = copy(src, dest)
        if Path(src) == source:
            replacement = tmp_path / 'replacement.npz'
            replacement.write_bytes(source.read_bytes())
            replacement.replace(source)
        return result
    monkeypatch.setattr(retained.shutil, 'copyfile', replaced)
    with pytest.raises(InputError):
        retained.retain_check('recording_equivalence', source, tmp_path / 'case', reference=reference)
    assert not (tmp_path / 'case').exists()


def test_unknown_metadata_without_processor_evidence_retained(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({'type': 'act', 'input_features': {},
        'output_features': {'action': {'type': 'ACTION', 'shape': [2]}}}))
    result = retained.retain_check('policy_metadata', checkpoint, tmp_path / 'case')
    assert result['original_status'] == 'unknown'
    replay = retained.reproduce_retained_check(tmp_path / 'case')
    assert replay['status'] == 'pass' and replay['original_status'] == 'unknown'
    assert replay['check_result']['counts']['unknown'] > 0


@pytest.mark.parametrize(('changed_dependency', 'status'), [('h5py', 'pass'), ('pyarrow', 'pass'),
                                                          ('PyYAML', 'pass'), ('numpy', 'unknown'), ('safetensors', 'unknown')])
def test_metadata_fingerprint_only_actual_dependency_versions(tmp_path, monkeypatch, changed_dependency, status):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({'type': 'act', 'input_features': {},
        'output_features': {'action': {'type': 'ACTION', 'shape': [2]}}}))
    retained.retain_check('policy_metadata', checkpoint, tmp_path / 'case')
    original = retained.importlib.metadata.version
    def changed(name):
        return 'changed-version' if name == changed_dependency else original(name)
    monkeypatch.setattr(retained.importlib.metadata, 'version', changed)
    replay = retained.reproduce_retained_check(tmp_path / 'case')
    assert replay['status'] == status
    assert replay['verifier_matches'] == (status == 'pass')


def test_numeric_preview_keeps_first_witness_and_full_original_digest(tmp_path):
    directory, _, _, result = frozen(tmp_path, value=9.)
    expected = json.loads((directory / 'expected.json').read_text())
    summary = result['check_result']
    assert summary['first_observed_difference'] == expected['first_observed_difference']
    assert not summary['first_observed_difference_omitted']
    assert summary['first_observed_difference']['evidence'][0]['sha256']
    assert summary['counts'] == {'different': 1, 'equal': 0, 'incompatible': 0}
    assert len(retained._encode(result)) <= 16384


def test_new_candidate_format_cannot_introduce_unrecorded_dependency(tmp_path):
    directory, _, _, _ = frozen(tmp_path)
    with pytest.raises(InputError, match='unrecorded verifier dependency'):
        retained.check_retained_candidate(directory, tmp_path / 'later.hdf5')


def test_npz_fingerprint_ignores_unused_safetensors_and_hdf5_versions(tmp_path, monkeypatch):
    directory, _, _, _ = frozen(tmp_path)
    original = retained.importlib.metadata.version
    monkeypatch.setattr(retained.importlib.metadata, 'version', lambda name: 'changed' if name in ('h5py', 'safetensors', 'pyarrow') else original(name))
    assert retained.reproduce_retained_check(directory)['status'] == 'pass'


def test_same_csv_file_can_supply_both_original_roles(tmp_path):
    source = tmp_path / 'recording.csv'
    source.write_text('index,action\n0,1\n1,2\n2,3\n')
    raw = source.read_bytes()
    directory = tmp_path / 'self-compare'
    result = retained.retain_check('recording_equivalence', source, directory, reference=source,
                                   parameters={'alignment_key': 'index'})
    assert result['original_status'] == 'pass'
    expected = json.loads((directory / 'expected.json').read_text())
    assert [a['path'] for a in expected['artifacts']] == ['reference', 'candidate']
    assert expected['artifacts'][0]['sha256'] == expected['artifacts'][1]['sha256']
    source.unlink()
    directory.rename(tmp_path / 'moved')
    assert retained.reproduce_retained_check(tmp_path / 'moved')['status'] == 'pass'
    source.write_bytes(raw)
    assert retained.check_retained_candidate(tmp_path / 'moved', source)['status'] == 'pass'


def test_same_hdf5_file_keeps_different_episode_roles_and_witnesses(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path / 'recording.hdf5'
    with h5py.File(source, 'w') as file:
        for episode, value in [('demo_0', 1.), ('demo_1', 7.)]:
            demo = file.create_group('data/' + episode)
            demo['index'] = np.arange(3)
            demo['actions'] = np.array([[value], [2.], [3.]])
    raw = source.read_bytes()
    directory = tmp_path / 'two-episodes'
    result = retained.retain_check('recording_equivalence', source, directory, reference=source,
                                   parameters={'alignment_key': 'index', 'reference_episode': 0, 'candidate_episode': 1})
    assert result['original_status'] == 'different'
    expected = json.loads((directory / 'expected.json').read_text())
    witness = expected['first_observed_difference']['evidence']
    assert [w['path'] for w in witness] == ['reference', 'candidate']
    assert [w['episode'] for w in witness] == ['demo_0', 'demo_1']
    assert [w['value'] for w in witness] == [1., 7.]
    assert source.read_bytes() == raw
    assert retained.reproduce_retained_check(directory)['status'] == 'pass'
    later = retained.check_retained_candidate(directory, source)
    assert later['status'] == 'different'
    assert later['check_result']['first_observed_difference'] == expected['first_observed_difference']


def test_checkpoint_and_dataset_can_share_native_runpack_root(tmp_path):
    from policy_doctor.demo import create_policy_demo
    pytest.importorskip('safetensors')
    demo = create_policy_demo(tmp_path / 'demo')
    checkpoint = Path(demo['checkpoint'])
    retained.shutil.copytree(Path(demo['corrected_dataset']) / 'meta', checkpoint / 'meta')
    originals = {p: p.read_bytes() for p in checkpoint.rglob('*') if p.is_file()}
    directory = tmp_path / 'runpack'
    result = retained.retain_check('policy_metadata', checkpoint, directory, dataset=checkpoint)
    assert result['original_status'] == 'pass'
    expected = json.loads((directory / 'expected.json').read_text())
    paths = {a['path'] for a in expected['artifacts']}
    assert 'checkpoint/config.json' in paths
    assert 'dataset/meta/info.json' in paths
    assert 'checkpoint/meta/info.json' not in paths
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    retained.shutil.rmtree(checkpoint)
    directory.rename(tmp_path / 'moved')
    assert retained.reproduce_retained_check(tmp_path / 'moved')['status'] == 'pass'
    assert retained.check_retained_candidate(tmp_path / 'moved', demo['corrected_dataset'])['status'] == 'pass'
