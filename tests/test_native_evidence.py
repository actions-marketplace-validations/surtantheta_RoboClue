"""Generated native-format controls, not simulator recordings."""
import json

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_evidence import fingerprint_native_context, snapshot_native_context

h5py = pytest.importorskip('h5py')


def fixture(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'env_cfg.json').write_bytes(b'{"instruction":"original", "decimation":8}\n')
    (root / 'log_0.json').write_bytes(b'[{}]\n')
    (root / 'LICENSE.txt').write_bytes(b'original license\n')
    with h5py.File(root / 'data.hdf5', 'w') as f:
        data = f.create_group('data')
        data.attrs['policy'] = 'native UTF8 π'
        demo = data.create_group('demo_0')
        demo.attrs['success'] = True
        demo.attrs['num_samples'] = 4
        demo.create_dataset('actions', data=np.arange(8, dtype='<f4').reshape(4, 2), compression='gzip')
        demo.create_dataset('obs/image', shape=(4, 8, 8, 3), dtype='uint8')
    return root


def test_native_subset_preserves_values_attrs_bytes_and_omissions(tmp_path):
    src = fixture(tmp_path)
    original = fingerprint_native_context(src)
    dest = tmp_path / 'subset'
    assert snapshot_native_context(src, dest) == dest
    assert fingerprint_native_context(src) == original
    assert (src / 'env_cfg.json').read_bytes() == (dest / 'env_cfg.json').read_bytes()
    assert (src / 'LICENSE.txt').read_bytes() == (dest / 'LICENSE.txt').read_bytes()
    with h5py.File(dest / 'data.hdf5', 'r') as f:
        assert f['data'].attrs['policy'] == 'native UTF8 π'
        assert f['data/demo_0'].attrs['success']
        np.testing.assert_array_equal(f['data/demo_0/actions'][:], np.arange(8, dtype='<f4').reshape(4, 2))
        assert 'image' not in f['data/demo_0/obs']
    receipt = json.loads((dest / 'evidence_selection.json').read_text())
    assert receipt['original_bytes_preserved'] is False
    assert receipt['original_origin'] == 'author_assertion_unless_original_available'
    assert receipt['hdf5']['selected'][0]['row_range'] == [0, 4]
    assert receipt['hdf5']['omitted'][0]['reason'] == 'media_or_rank_above_two'
    assert receipt['licenses']['present'] == ['LICENSE.txt']
    assert str(src) not in json.dumps(receipt)


@pytest.mark.parametrize('kind', ['soft', 'external', 'alias', 'reference_attribute'])
def test_unsafe_hdf5_refuses_without_partial_destination(tmp_path, kind):
    src = fixture(tmp_path)
    with h5py.File(src / 'data.hdf5', 'a') as f:
        if kind == 'soft':
            f['bad'] = h5py.SoftLink('/data')
        elif kind == 'external':
            f['bad'] = h5py.ExternalLink('outside.hdf5', '/data')
        elif kind == 'alias':
            f['duplicate'] = f['data']
        else:
            f.attrs['ref'] = f['data'].ref
    dest = tmp_path / 'subset'
    with pytest.raises(InputError):
        snapshot_native_context(src, dest)
    assert not dest.exists()


def test_no_overwrite_nested_or_escaping_input(tmp_path):
    src = fixture(tmp_path)
    dest = tmp_path / 'existing'
    dest.mkdir()
    with pytest.raises(InputError):
        snapshot_native_context(src, dest)
    assert dest.is_dir()
    with pytest.raises(InputError):
        snapshot_native_context(src, src / 'nested')
    (src / 'escape').symlink_to(tmp_path)
    with pytest.raises(InputError):
        fingerprint_native_context(src)


def test_unknown_files_are_hashed_and_explicitly_omitted(tmp_path):
    src = fixture(tmp_path)
    (src / 'notes.txt').write_text('not copied')
    dest = snapshot_native_context(src, tmp_path / 'subset')
    receipt = json.loads((dest / 'evidence_selection.json').read_text())
    assert not (dest / 'notes.txt').exists()
    assert receipt['omitted_files'][0]['path'] == 'notes.txt'
    assert receipt['omitted_files'][0]['sha256']


def test_media_value_payload_is_never_accessed(tmp_path, monkeypatch):
    from policy_doctor.native_evidence import _process_hdf5
    src = fixture(tmp_path)
    original = h5py.Dataset.__getitem__
    accessed = []

    def witness(self, item):
        accessed.append(self.name)
        if '/image' in self.name:
            raise AssertionError('image values accessed')
        return original(self, item)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', witness)
    result = _process_hdf5(src / 'data.hdf5', tmp_path / 'selected.hdf5')
    assert accessed == ['/data/demo_0/actions']
    assert result['omitted'][0]['path'].endswith('/image')


def test_metadata_only_never_reads_any_dataset_values(tmp_path, monkeypatch):
    from policy_doctor.native_evidence import _process_hdf5
    src = fixture(tmp_path)

    def forbidden(*args):
        raise AssertionError('metadata inspection accessed dataset values')

    monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
    result = _process_hdf5(src / 'data.hdf5', None)
    assert result['data_attrs']['policy'] == 'native UTF8 π'
    assert result['episodes'][0]['attrs']['num_samples'] == 4


def test_value_budget_omits_whole_dataset_without_trimming(tmp_path, monkeypatch):
    from policy_doctor import native_evidence
    src = fixture(tmp_path)
    monkeypatch.setattr(native_evidence, 'MAX_SELECTED_BYTES', 20)
    # Chunk storage is also preflighted; use a contiguous numeric source here.
    with h5py.File(src / 'data.hdf5', 'a') as f:
        del f['data/demo_0/actions']
        f.create_dataset('data/demo_0/actions', data=np.arange(8, dtype='float32').reshape(4, 2))
    result = native_evidence._process_hdf5(src / 'data.hdf5', tmp_path / 'selected.hdf5')
    assert not result['selected']
    assert any(x['reason'] == 'selected_decoded_byte_budget' for x in result['omitted'])


def test_oversized_vlen_attribute_refuses_in_bounded_worker(tmp_path):
    src = fixture(tmp_path)
    with h5py.File(src / 'data.hdf5', 'a') as f:
        f.attrs['huge'] = 'x' * (2 * 1024**2 + 1)
    dest = tmp_path / 'subset'
    with pytest.raises(InputError, match='budget'):
        snapshot_native_context(src, dest)
    assert not dest.exists()


def test_source_change_during_selection_is_refused(tmp_path, monkeypatch):
    from policy_doctor import native_evidence
    src = fixture(tmp_path)
    original = native_evidence._isolated_hdf5

    def mutate(*args):
        result = original(*args)
        (src / 'env_cfg.json').write_text('{"changed":true}')
        return result

    monkeypatch.setattr(native_evidence, '_isolated_hdf5', mutate)
    dest = tmp_path / 'subset'
    with pytest.raises(InputError, match='changed'):
        snapshot_native_context(src, dest)
    assert not dest.exists()


def test_missing_optional_dependency_refuses_without_publication(tmp_path, monkeypatch):
    from policy_doctor import native_evidence
    src = fixture(tmp_path)

    def unavailable(*args):
        raise InputError('Install policy-deployment-doctor[dataset] for HDF5 support')

    monkeypatch.setattr(native_evidence, '_isolated_hdf5', unavailable)
    with pytest.raises(InputError):
        snapshot_native_context(src, tmp_path / 'subset')
    assert not (tmp_path / 'subset').exists()


def test_source_change_and_restoration_still_refuses(tmp_path, monkeypatch):
    from policy_doctor import native_evidence
    src = fixture(tmp_path)
    original = native_evidence._isolated_hdf5

    def change_then_restore(*args):
        result = original(*args)
        config = src / 'env_cfg.json'
        data = config.read_bytes()
        config.write_bytes(b'changed')
        config.write_bytes(data)
        return result

    monkeypatch.setattr(native_evidence, '_isolated_hdf5', change_then_restore)
    with pytest.raises(InputError, match='changed'):
        snapshot_native_context(src, tmp_path / 'subset')
    assert not (tmp_path / 'subset').exists()


def test_isolated_worker_does_not_import_artifact_modules(tmp_path, monkeypatch):
    src = fixture(tmp_path)
    marker = tmp_path / 'EXECUTED'
    (tmp_path / 'h5py.py').write_text(f'open({str(marker)!r},"w").write("executed")\n')
    (tmp_path / 'numpy.py').write_text(f'open({str(marker)!r},"w").write("executed")\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    snapshot_native_context(src, tmp_path / 'subset')
    assert not marker.exists()


def test_resource_budgets_and_ambiguous_inputs_refuse(tmp_path, monkeypatch):
    from policy_doctor import native_evidence
    src = fixture(tmp_path)
    monkeypatch.setattr(native_evidence, 'MAX_FILES', 2)
    with pytest.raises(InputError, match='64 entries'):
        fingerprint_native_context(src)
    monkeypatch.setattr(native_evidence, 'MAX_FILES', 64)
    (src / 'run_0.hdf5').write_bytes(b'not opened')
    with pytest.raises(InputError, match='exactly one'):
        snapshot_native_context(src, tmp_path / 'subset')
    assert not (tmp_path / 'subset').exists()


@pytest.mark.parametrize('kind', ['virtual', 'external_storage', 'nonbuiltin_filter'])
def test_hdf5_storage_and_filter_guards(tmp_path, kind):
    src = fixture(tmp_path)
    with h5py.File(src / 'data.hdf5', 'a') as f:
        if kind == 'virtual':
            layout = h5py.VirtualLayout(shape=(4,), dtype='float32')
            layout[:] = h5py.VirtualSource('outside.hdf5', '/data', shape=(4,))
            f.create_virtual_dataset('unsupported', layout)
        elif kind == 'external_storage':
            f.create_dataset('unsupported', shape=(4,), dtype='float32', external=[('outside.bin', 0, 16)])
        else:
            f.create_dataset('unsupported', data=np.arange(4, dtype='float32'), compression='lzf')
    with pytest.raises(InputError):
        snapshot_native_context(src, tmp_path / 'subset')
    assert not (tmp_path / 'subset').exists()


def test_unsupported_string_dataset_is_inventoried_without_value_read(tmp_path, monkeypatch):
    from policy_doctor.native_evidence import _process_hdf5
    src = fixture(tmp_path)
    with h5py.File(src / 'data.hdf5', 'a') as f:
        f.create_dataset('labels', data=['abc'], dtype=h5py.string_dtype())
    original = h5py.Dataset.__getitem__

    def witness(self, key):
        if self.name == '/labels':
            raise AssertionError('unsupported string value read')
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', witness)
    result = _process_hdf5(src / 'data.hdf5', tmp_path / 'selected.hdf5')
    assert any(x['path'] == '/labels' and x['reason'] == 'unsupported_nonnumeric_dtype'
               for x in result['omitted'])


def test_optional_source_provenance_is_preserved_as_untrusted_bytes(tmp_path):
    src = fixture(tmp_path)
    provenance = b'{"origin":"caller claim", "run":"never execute"}\n'
    (src / 'source_provenance.json').write_bytes(provenance)
    dest = snapshot_native_context(src, tmp_path / 'subset')
    assert (dest / 'source_provenance.json').read_bytes() == provenance
    receipt = json.loads((dest / 'evidence_selection.json').read_text())
    assert receipt['original_origin'] == 'author_assertion_unless_original_available'


def test_sidecar_copy_never_uses_unbounded_read(tmp_path, monkeypatch):
    from pathlib import Path
    src = fixture(tmp_path)
    original_open = Path.open
    reads = []

    class Witness:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            reads.append(size)
            assert 0 <= size <= 2 * 1024**2 + 1, 'unbounded sidecar allocation'
            return self.stream.read(size)

    def guarded(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == src / 'env_cfg.json' and args and args[0] == 'rb':
            return Witness(stream)
        return stream

    monkeypatch.setattr(Path, 'open', guarded)
    snapshot_native_context(src, tmp_path / 'subset')
    assert 2 * 1024**2 + 1 in reads


def test_context_metadata_uses_isolated_worker_and_preserves_shape(tmp_path, monkeypatch):
    from policy_doctor import native_context, native_evidence
    src = fixture(tmp_path)
    marker = tmp_path / 'EXECUTED'
    (tmp_path / 'h5py.py').write_text(f'open({str(marker)!r},"w").write("executed")\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))

    def forbidden(*args):
        raise AssertionError('context metadata executed in parent')

    monkeypatch.setattr(native_context, '_hdf5_metadata', forbidden)
    result = native_evidence.read_context_metadata_hdf5(src / 'data.hdf5')
    assert set(result) == {'episodes', 'provenance', 'object_count', 'dataset_value_reads'}
    assert result['dataset_value_reads'] == 0
    assert result['episodes'][0]['attrs']['num_samples'] == 4
    assert not marker.exists()


def test_context_worker_refuses_links_and_untrusted_modes(tmp_path):
    from policy_doctor import native_evidence
    src = fixture(tmp_path)
    with h5py.File(src / 'data.hdf5', 'a') as f:
        f['escape'] = h5py.SoftLink('/data')
    with pytest.raises(InputError):
        native_evidence.read_context_metadata_hdf5(src / 'data.hdf5')
    with pytest.raises(InputError, match='mode'):
        native_evidence._worker(str(src / 'data.hdf5'), '', str(tmp_path / 'report'), 'artifact_module')


def test_sidecar_growth_hits_bounded_read_and_cleans_destination(tmp_path, monkeypatch):
    from pathlib import Path
    src = fixture(tmp_path)
    original_open = Path.open
    seen = []

    class GrowingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            assert size >= 0
            if size == 2 * 1024**2 + 1:
                seen.append(size)
                return b'x' * size
            return self.stream.read(size)

    def guarded(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == src / 'env_cfg.json' and args and args[0] == 'rb':
            return GrowingReader(stream)
        return stream

    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(InputError, match='grew'):
        snapshot_native_context(src, tmp_path / 'subset')
    assert seen == [2 * 1024**2 + 1]
    assert not (tmp_path / 'subset').exists()
