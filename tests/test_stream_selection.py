"""Native NPZ selection must avoid decoding excluded values, not just hide them."""
import hashlib
import io
import struct
import zipfile

import numpy as np
import pytest

from policy_doctor import rollouts
from policy_doctor.artifacts import InputError


def _native(path, *, shift=False, extra=None):
    arrays = {'sample_id': np.arange(6), 'actions': np.zeros((6, 2)),
              'rgb': np.zeros((6, 1, 3, 4, 4), dtype=np.uint8)}
    if shift:
        arrays['actions'][3, 1] = 0.25
    arrays.update(extra or {})
    np.savez_compressed(path, **arrays)
    return path


def _npy(array):
    data = io.BytesIO()
    np.save(data, array)
    return data.getvalue()


def _archive(path, entries):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return path


def test_native_selected_self_preserves_bytes_and_names(tmp_path):
    path = _native(tmp_path / 'native.npz')
    before = path.read_bytes()
    report = rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'],
                                      alignment_key='sample_id')
    assert report['status'] == 'pass' and report['comparison_complete']
    assert [row['key'] for row in report['streams']] == ['actions']
    assert report['alignment']['sample_phase'] == 'unknown'
    assert report['causal_conclusion'] == 'unknown'
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    artifact = report['artifacts'][0]
    assert artifact['sha256'] == hashlib.sha256(before).hexdigest()
    selection = artifact['stream_selection']
    assert selection['selected'] == ['sample_id', 'actions']
    assert selection['excluded'] == ['rgb']
    assert selection['inventory']['rgb']['shape'] == [6, 1, 3, 4, 4]
    assert selection['inventory']['rgb']['dtype'] == 'uint8'
    assert not selection['inventory']['rgb']['values_decoded']
    assert 'not' in selection['scope'].lower()


def test_generated_shift_reports_native_coordinate_and_original_hash(tmp_path):
    reference = _native(tmp_path / 'reference.npz')
    candidate = _native(tmp_path / 'generated-shift.npz', shift=True)
    report = rollouts.compare_rollouts(reference, candidate, streams=['sample_id', 'actions'],
                                      alignment_key='sample_id')
    assert report['status'] == 'different'
    first = report['first_observed_difference']
    assert first['stream'] == 'actions' and first['row'] == 3 and first['coordinate'] == [1]
    assert first['sample_id'] == 3
    assert first['evidence'][1]['sha256'] == hashlib.sha256(candidate.read_bytes()).hexdigest()


def test_value_decoder_access_witness_excludes_rgb(tmp_path, monkeypatch):
    path = _native(tmp_path / 'native.npz')
    decoded = []
    original = np.lib.format.read_array

    def witness(stream, *args, **kwargs):
        decoded.append(stream.name)
        assert stream.name != 'rgb.npy', 'Excluded RGB value decoder was invoked'
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, 'read_array', witness)
    report = rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'],
                                      alignment_key='sample_id')
    assert report['status'] == 'pass'
    assert decoded == ['sample_id.npy', 'actions.npy'] * 2


def test_inventory_reads_headers_without_any_value_decoder(tmp_path, monkeypatch):
    path = _native(tmp_path / 'native.npz')

    def forbidden(*args, **kwargs):
        pytest.fail('Inventory must not invoke the array value decoder')

    monkeypatch.setattr(np.lib.format, 'read_array', forbidden)
    result = rollouts.inspect_rollout(path, streams=['actions'])
    assert list(result['streams']) == ['actions']
    assert not result['values_validated']
    assert result['artifact']['stream_selection']['excluded'] == ['sample_id', 'rgb']


def test_default_still_refuses_rank_five_media(tmp_path):
    path = _native(tmp_path / 'native.npz')
    with pytest.raises(InputError, match='rank'):
        rollouts.inspect_rollout(path)
    with pytest.raises(InputError, match='rank'):
        rollouts.compare_rollouts(path, path)


@pytest.mark.parametrize('selection', [[], ['actions', 'actions'], True, 'actions', [True], [1], [''],
                                       ['actions', None], ('actions',), ['absent'], ['rgb'], ['\ud800']])
def test_invalid_or_unsupported_selection_refuses(tmp_path, selection):
    path = _native(tmp_path / 'native.npz')
    with pytest.raises(InputError):
        rollouts.compare_rollouts(path, path, streams=selection)


def test_alignment_must_be_explicitly_selected(tmp_path):
    path = _native(tmp_path / 'native.npz')
    with pytest.raises(InputError, match='alignment'):
        rollouts.compare_rollouts(path, path, streams=['actions'], alignment_key='sample_id')


def test_without_alignment_selection_remains_unknown(tmp_path):
    path = _native(tmp_path / 'native.npz')
    result = rollouts.compare_rollouts(path, path, streams=['actions'])
    assert result['status'] == 'unknown' and not result['comparison_complete']


@pytest.mark.parametrize('extension', ['csv', 'parquet', 'h5', 'hdf5'])
def test_other_formats_refuse_selection_before_decode(tmp_path, extension):
    path = tmp_path / f'other.{extension}'
    path.write_bytes(b'not a valid recording')
    with pytest.raises(InputError, match='NPZ'):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_excluded_nonfinite_values_are_uninspected(tmp_path):
    path = _native(tmp_path / 'native.npz', extra={'uninspected': np.array([np.nan])})
    report = rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'],
                                      alignment_key='sample_id')
    assert report['status'] == 'pass'
    item = report['artifacts'][0]['stream_selection']['inventory']['uninspected']
    assert not item['values_decoded'] and 'not decoded' in item['reason']
    with pytest.raises(InputError, match='Non-finite'):
        rollouts.compare_rollouts(path, path, streams=['uninspected'])


def test_object_data_refused_even_when_excluded_without_execution(tmp_path):
    path = _native(tmp_path / 'native.npz', extra={'object': np.array([{'never': 'execute'}], dtype=object)})
    with pytest.raises(InputError, match='object|pickle'):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_safe_unsupported_dtype_can_be_excluded_but_not_selected(tmp_path):
    path = _native(tmp_path / 'native.npz', extra={'label': np.array(['native'])})
    report = rollouts.inspect_rollout(path, streams=['actions'])
    item = report['artifact']['stream_selection']['inventory']['label']
    assert not item['supported_numeric'] and not item['values_decoded']
    with pytest.raises(InputError, match='numeric'):
        rollouts.inspect_rollout(path, streams=['label'])


@pytest.mark.parametrize('member', ['../rgb.npy', '/rgb.npy', 'C:/rgb.npy', 'bad\\rgb.npy', '.npy', 'rgb.txt'])
def test_unsafe_excluded_member_paths_refuse(tmp_path, member):
    path = _archive(tmp_path / 'unsafe.npz', [('actions.npy', _npy(np.ones((2, 1)))),
                                             (member, _npy(np.ones(2)))])
    with pytest.raises(InputError):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_duplicate_excluded_member_refuses(tmp_path):
    with pytest.warns(UserWarning, match='Duplicate name'):
        path = _archive(tmp_path / 'duplicate.npz', [('actions.npy', _npy(np.ones((2, 1)))),
                                                  ('rgb.npy', _npy(np.zeros(2))),
                                                  ('rgb.npy', _npy(np.zeros(2)))])
    with pytest.raises(InputError, match='duplicate'):
        rollouts.inspect_rollout(path, streams=['actions'])


@pytest.mark.parametrize('payload', [b'not-npy', _npy(np.zeros(2))[:-1], _npy(np.zeros(2)) + b'extra'])
def test_malformed_excluded_header_or_payload_length_refuses(tmp_path, payload):
    path = _archive(tmp_path / 'bad.npz', [('actions.npy', _npy(np.ones((2, 1)))), ('rgb.npy', payload)])
    with pytest.raises(InputError):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_selected_values_budget_not_charged_for_unselected_media(tmp_path, monkeypatch):
    path = _native(tmp_path / 'native.npz', extra={'other_numeric': np.zeros(21)})
    monkeypatch.setattr(rollouts, 'MAX_VALUES', 20)
    assert rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'],
                                    alignment_key='sample_id')['status'] == 'pass'
    with pytest.raises(InputError, match='budget'):
        rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions', 'other_numeric'])


def test_expanded_inventory_budget_refuses_compressed_bomb(tmp_path, monkeypatch):
    path = _native(tmp_path / 'native.npz')
    monkeypatch.setattr(rollouts, 'MAX_NPZ_INVENTORY_BYTES', 100)
    with pytest.raises(InputError, match='expanded|inventory'):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_default_expanded_limit_unchanged_selection_is_explicit(tmp_path, monkeypatch):
    path = tmp_path / 'numeric.npz'
    np.savez_compressed(path, sample_id=np.arange(6), actions=np.zeros((6, 2)),
                        other=np.zeros(1000, dtype=np.uint8))
    monkeypatch.setattr(rollouts, 'MAX_BYTES', path.stat().st_size + 1)
    with pytest.raises(InputError, match='expanded'):
        rollouts.inspect_rollout(path)
    assert list(rollouts.inspect_rollout(path, streams=['actions'])['streams']) == ['actions']


def test_selected_payload_bytes_budget_refuses(tmp_path, monkeypatch):
    path = tmp_path / 'numeric.npz'
    np.savez_compressed(path, actions=np.zeros((1000, 2)))
    monkeypatch.setattr(rollouts, 'MAX_BYTES', path.stat().st_size + 1)
    with pytest.raises(InputError, match='budget'):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_selection_snapshot_not_mutated_by_caller(tmp_path):
    path = _native(tmp_path / 'native.npz')
    selection = ['actions']
    report = rollouts.inspect_rollout(path, streams=selection)
    selection.append('rgb')
    assert report['artifact']['stream_selection']['selected'] == ['actions']


def test_exact_native_keys_do_not_trigger_np_load_filename_alias(tmp_path):
    path = _archive(tmp_path / 'ambiguous-to-np-load.npz', [
        ('sample_id.npy', _npy(np.arange(2))),
        ('foo.npy', _npy(np.ones((2, 1)))),
        ('foo.npy.npy', _npy(np.full((2, 1), 7.)))])
    report = rollouts.compare_rollouts(path, path, streams=['sample_id', 'foo', 'foo.npy'],
                                      alignment_key='sample_id')
    assert report['status'] == 'pass'
    assert {row['key'] for row in report['streams']} == {'foo', 'foo.npy'}
    _, arrays, _ = rollouts._recording(path, True, streams=['foo', 'foo.npy'])
    assert arrays['foo'].tolist() == [[1.], [1.]]
    assert arrays['foo.npy'].tolist() == [[7.], [7.]]


def test_structured_object_excluded_also_refuses(tmp_path):
    nested = np.zeros(1, dtype=[('hidden', object)])
    path = _native(tmp_path / 'object.npz', extra={'nested': nested})
    with pytest.raises(InputError, match='object|pickle'):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_unselected_header_does_not_request_payload_bytes(tmp_path, monkeypatch):
    path = _native(tmp_path / 'native.npz')
    with zipfile.ZipFile(path) as archive:
        original_rgb = archive.read('rgb.npy')
    header = io.BytesIO(original_rgb)
    np.lib.format.read_magic(header)
    np.lib.format.read_array_header_1_0(header)
    payload_start = header.tell()
    requests = []
    original_open = zipfile.ZipFile.open

    def witnessed_open(archive, name, *args, **kwargs):
        stream = original_open(archive, name, *args, **kwargs)
        if stream.name == 'rgb.npy':
            original_read = stream.read

            def read(size=-1):
                assert 0 <= size <= payload_start - stream.tell(), 'RGB payload bytes requested'
                requests.append(size)
                return original_read(size)

            stream.read = read
        return stream

    monkeypatch.setattr(zipfile.ZipFile, 'open', witnessed_open)
    assert rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'],
                                    alignment_key='sample_id')['status'] == 'pass'
    assert sum(requests) == payload_start * 2


def test_oversized_unselected_header_refuses(tmp_path):
    payload = io.BytesIO()
    np.lib.format.write_array_header_2_0(payload, {
        'descr': [('field_' + 'x' * 100, '<f8')] * 200,
        'fortran_order': False, 'shape': (0,)})
    path = _archive(tmp_path / 'header.npz', [('actions.npy', _npy(np.ones((2, 1)))),
                                            ('excluded.npy', payload.getvalue())])
    with pytest.raises(InputError, match='header|Header'):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_too_many_unselected_members_refuses(tmp_path):
    path = _archive(tmp_path / 'many.npz', [(f'array{i}.npy', _npy(np.ones(1))) for i in range(33)])
    with pytest.raises(InputError, match='32'):
        rollouts.inspect_rollout(path, streams=['array0'])


def test_excluded_complex_and_string_values_are_only_header_inventory(tmp_path):
    path = _native(tmp_path / 'unsupported.npz', extra={'complex': np.array([complex(np.nan, 1)]),
                                                      'string': np.array(['untouched'])})
    inventory = rollouts.inspect_rollout(path, streams=['actions'])['artifact']['stream_selection']['inventory']
    for key in ['complex', 'string']:
        assert not inventory[key]['supported_numeric'] and not inventory[key]['values_decoded']


def test_fortran_order_native_selected_values_remain_correct(tmp_path):
    path = tmp_path / 'fortran.npz'
    np.savez(path, sample_id=np.arange(6), actions=np.asfortranarray(np.arange(12).reshape(6, 2)))
    assert rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'],
                                    alignment_key='sample_id')['status'] == 'pass'


@pytest.mark.parametrize('selection', [None, ['actions']])
@pytest.mark.parametrize('lie', [False, True])
def test_member_preflight_refuses_before_zipfile_allocation(tmp_path, monkeypatch, selection, lie):
    path = _archive(tmp_path / 'many.npz', [('actions.npy', _npy(np.ones(2)))] + [
        (f'excluded{i}.npy', _npy(np.zeros(1))) for i in range(40)])
    if lie:
        raw = bytearray(path.read_bytes())
        end = raw.rfind(b'PK\x05\x06')
        struct.pack_into('<HH', raw, end + 8, 1, 1)
        path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        pytest.fail('Oversized actual central directory reached ZipFile/ZipInfo allocation')

    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    monkeypatch.setattr(zipfile, 'ZipInfo', forbidden)
    with pytest.raises(InputError, match='32|directory'):
        rollouts.inspect_rollout(path, streams=selection)


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('selection', [None, ['sample_id', 'actions']])
def test_ordinary_numpy_multi_member_archive_preserved(tmp_path, compressed, selection):
    path = tmp_path / 'ordinary.npz'
    writer = np.savez_compressed if compressed else np.savez
    writer(path, sample_id=np.arange(3), actions=np.ones((3, 2)))
    assert rollouts.compare_rollouts(path, path, streams=selection, alignment_key='sample_id')['status'] == 'pass'


@pytest.mark.parametrize('selection', [None, ['actions']])
def test_zip64_local_headers_supported_with_bounded_standard_central_directory(tmp_path, selection):
    path = tmp_path / 'local-zip64.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        with archive.open('actions.npy', 'w', force_zip64=True) as stream:
            stream.write(_npy(np.ones((3, 2))))
    assert struct.unpack_from('<I', path.read_bytes(), 18)[0] == 0xffffffff
    assert rollouts.inspect_rollout(path, streams=selection)['streams']['actions']['shape'] == [3, 2]


@pytest.mark.parametrize('kind', ['disk', 'zip64', 'wrong_count', 'wrong_directory_size'])
def test_invalid_directory_refuses_before_zipfile_allocation(tmp_path, monkeypatch, kind):
    path = _archive(tmp_path / 'invalid.npz', [('actions.npy', _npy(np.ones(2)))])
    raw = bytearray(path.read_bytes())
    end = raw.rfind(b'PK\x05\x06')
    if kind == 'disk':
        struct.pack_into('<H', raw, end + 4, 1)
    elif kind == 'zip64':
        struct.pack_into('<HH', raw, end + 8, 0xffff, 0xffff)
    elif kind == 'wrong_count':
        struct.pack_into('<HH', raw, end + 8, 2, 2)
    else:
        size = struct.unpack_from('<I', raw, end + 12)[0]
        struct.pack_into('<I', raw, end + 12, size + 1)
    path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid central directory reached ZipFile allocation')

    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    with pytest.raises(InputError):
        rollouts.inspect_rollout(path, streams=['actions'])


def test_npz_named_symlink_to_csv_cannot_silently_ignore_selection(tmp_path):
    csv_path = tmp_path / 'actual.csv'
    csv_path.write_text('sample_id,actions\n0,1\n1,1\n')
    alias = tmp_path / 'alias.npz'
    alias.symlink_to(csv_path)
    with pytest.raises(InputError, match='NPZ'):
        rollouts.inspect_rollout(alias, streams=['absent'])
    with pytest.raises(InputError, match='NPZ'):
        rollouts.compare_rollouts(alias, alias, streams=['sample_id', 'actions'], alignment_key='sample_id')


def test_selected_coverage_cannot_claim_whole_recording_equivalence(tmp_path):
    path = _native(tmp_path / 'native.npz')
    result = rollouts.compare_rollouts(path, path, streams=['sample_id', 'actions'], alignment_key='sample_id')
    assert result['comparison_complete']
    assert 'not' in result['coverage_scope'] and 'whole-recording equivalence' in result['coverage_scope']


@pytest.mark.parametrize('selection', [None, ['actions']])
def test_bounded_archive_and_member_comments_and_ordinary_extra_preserved(tmp_path, selection):
    path = tmp_path / 'comments.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.comment = b'ordinary archive comment'
        member = zipfile.ZipInfo('actions.npy')
        member.comment = b'ordinary member comment'
        member.extra = struct.pack('<HH', 0xbeef, 3) + b'abc'
        archive.writestr(member, _npy(np.ones((3, 2))))
    assert rollouts.inspect_rollout(path, streams=selection)['streams']['actions']['shape'] == [3, 2]


@pytest.mark.parametrize('selection', [None, ['actions']])
def test_trailing_bytes_intentionally_refused_before_zipfile_allocation(tmp_path, monkeypatch, selection):
    path = _archive(tmp_path / 'trailing.npz', [('actions.npy', _npy(np.ones(2)))])
    path.write_bytes(path.read_bytes() + b'trailing data')

    def forbidden(*args, **kwargs):
        pytest.fail('Noncanonical trailing bytes reached ZipFile allocation')

    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    with pytest.raises(InputError, match='trailing'):
        rollouts.inspect_rollout(path, streams=selection)


@pytest.mark.parametrize('selection', [None, ['actions']])
@pytest.mark.parametrize('kind', ['later_eocd', 'zip64_locator'])
def test_preflight_cannot_validate_a_different_directory_than_native_zip(tmp_path, monkeypatch, selection, kind):
    path = _archive(tmp_path / 'alternate-directory.npz', [('actions.npy', _npy(np.ones(2)))])
    base = path.read_bytes()
    end = base.rfind(b'PK\x05\x06')
    directory_offset = struct.unpack_from('<I', base, end + 16)[0]
    ordinary_directory = base[directory_offset:end]
    hidden_directory = b''
    for i in range(41):
        name = f'hidden{i}.npy'.encode()
        record = bytearray(ordinary_directory[:46])
        struct.pack_into('<HHH', record, 28, len(name), 0, 0)
        hidden_directory += record + name
    if kind == 'later_eocd':
        # This later signature has an invalid comment length. Python still picks it;
        # preflight must never fall back to the valid earlier outer end record.
        alternate_end = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 41, 41,
                                    len(hidden_directory), 0, 1)
        raw = bytearray(base)
        struct.pack_into('<H', raw, end + 20, len(hidden_directory) + len(alternate_end))
        raw.extend(hidden_directory + alternate_end)
    else:
        # Ordinary EOCD fields have no ZIP64 sentinels. The locator overrides them
        # with 41 hidden central records in an otherwise ordinary member comment.
        zip64_end = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45, 0, 0,
                                41, 41, len(hidden_directory), 0)
        locator = struct.pack('<4sLQL', b'PK\x06\x07', 0, 0, 1)
        payload = hidden_directory + zip64_end + locator
        outer = bytearray(ordinary_directory)
        struct.pack_into('<H', outer, 32, len(payload))
        ordinary_end = bytearray(base[end:])
        struct.pack_into('<I', ordinary_end, 12, len(outer) + len(payload))
        raw = base[:directory_offset] + outer + payload + ordinary_end
    path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        pytest.fail('Native ZIP parser reached a directory different from the preflighted one')

    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    monkeypatch.setattr(zipfile, 'ZipInfo', forbidden)
    with pytest.raises(InputError):
        rollouts.inspect_rollout(path, streams=selection)
