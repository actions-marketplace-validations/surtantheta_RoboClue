"""Refuse real encrypted metadata before uncontained native column inspection."""
import io
import struct

import pytest

from policy_doctor.native_parquet_guard import (
    UnsupportedParquetMetadata,
    require_unencrypted_parquet,
)


def frame(footer):
    return b'PAR1' + footer + struct.pack('<I', len(footer)) + b'PAR1'


@pytest.mark.parametrize('encoding', [b'', b'\x1e\x00', b'\x18\x80', b'\x18\xff' * 20,
                                      b'\x1c' * 40 + b'\x00' * 41])
def test_unresolved_wire_values_refuse_without_arrow_or_large_allocations(encoding):
    with pytest.raises(UnsupportedParquetMetadata):
        require_unencrypted_parquet(io.BytesIO(frame(encoding)))


@pytest.mark.parametrize('footer', [b'\x8c\x00\x00', b'\x98\x00\x00',
    b'\x49\x1c\x19\x1c\x8c\x00\x00\x00\x00',
    b'\x49\x1c\x19\x1c\x98\x00\x00\x00\x00'])
def test_file_and_nested_column_crypto_fields_are_refused(footer):
    with pytest.raises(UnsupportedParquetMetadata, match='Encrypted'):
        require_unencrypted_parquet(io.BytesIO(frame(footer)))


def test_real_plain_parquet_with_crypto_marker_bytes_in_user_metadata_is_allowed(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    path = tmp_path / 'plain.parquet'
    table = pa.table({'index': [0, 1], 'flag': [True, False], 'value': [[1., 2.], [3., 4.]]})
    table = table.replace_schema_metadata({b'opaque': b'\x8c\x00\x98\x00encrypted_column_metadata'})
    pq.write_table(table, path)
    require_unencrypted_parquet(path)
    stream = io.BytesIO(path.read_bytes())
    stream.seek(5)
    require_unencrypted_parquet(stream)
    assert stream.tell() == 5
    assert pq.read_table(path).equals(table)


def test_budget_refusal_precedes_footer_decode_and_preserves_position():
    stream = io.BytesIO(frame(b'\x00' * 100))
    stream.seek(3)
    with pytest.raises(UnsupportedParquetMetadata):
        require_unencrypted_parquet(stream, max_footer=20)
    assert stream.tell() == 3


def test_every_prefix_of_real_footer_refuses_without_reading_past_bounds(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    path = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'index': list(range(10))}), path)
    raw = path.read_bytes()
    length = int.from_bytes(raw[-8:-4], 'little')
    footer = raw[-8 - length:-8]
    for end in range(len(footer)):
        with pytest.raises(UnsupportedParquetMetadata):
            require_unencrypted_parquet(io.BytesIO(frame(footer[:end])))
    require_unencrypted_parquet(io.BytesIO(raw))
