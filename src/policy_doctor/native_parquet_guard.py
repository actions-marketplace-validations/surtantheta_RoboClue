"""Fixed, bounded Parquet preflight; usable without importing policy_doctor."""

import json
import os
import stat
import sys
from pathlib import Path

MAX_FILE = 128 * 1024**2
MAX_FOOTER = 2 * 1024**2
MAX_DECODED = 32 * 1024**2
MAX_FILES = 129
MAX_ROWS = 100_000
MAX_COLUMNS = 256
MAX_ROW_GROUPS = 128
MAX_DEPTH = 6
MAX_REQUEST = 128 * 1024


class UnsupportedParquetMetadata(ValueError):
    """A bounded metadata inspection cannot authorize native Arrow access."""


class _CompactFooter:
    """Skip Compact-Thrift values without materializing metadata or calling Arrow.

    Apache parquet-format@8c91fac2: FileMetaData fields8/9 and ColumnChunk
    fields8/9 identify encryption, including files with ordinary PAR1 magic.
    Unsupported encodings/budgets are refusals, never corruption diagnoses.
    """

    def __init__(self, raw):
        self.raw, self.offset, self.items = raw, 0, 0

    def take(self, size):
        if size < 0 or self.offset + size > len(self.raw):
            raise UnsupportedParquetMetadata('Truncated or unsupported Compact-Thrift footer')
        start = self.offset
        self.offset += size
        return self.raw[start:self.offset]

    def byte(self):
        return self.take(1)[0]

    def integer(self, bits=32):
        value = 0
        for shift in range(0, bits, 7):
            byte = self.byte()
            value |= (byte & 127) << shift
            if not byte & 128:
                if value >= 1 << bits:
                    break
                return value
        raise UnsupportedParquetMetadata('Compact-Thrift integer exceeds its bound')

    def step(self, depth):
        self.items += 1
        if self.items > 1_000_000 or depth > 32:
            raise UnsupportedParquetMetadata('Compact-Thrift metadata inspection budget exceeded')

    def structure(self, depth=0, context=None):
        previous, seen = 0, set()
        while True:
            self.step(depth)
            header = self.byte()
            if header == 0:
                return
            kind, delta = header & 15, header >> 4
            if delta:
                field = previous + delta
            else:
                zigzag = self.integer(16)
                field = (zigzag >> 1) ^ -(zigzag & 1)
            if not 1 <= field <= 32767 or field in seen:
                raise UnsupportedParquetMetadata('Ambiguous Compact-Thrift field identity')
            seen.add(field)
            previous = field
            if context in ('file', 'column') and field in (8, 9):
                raise UnsupportedParquetMetadata('Encrypted Parquet metadata is unsupported; no decryption is attempted')
            child = 'group' if context == 'file' and field == 4 else (
                'column' if context == 'group' and field == 1 else None)
            if child and kind != 9:
                raise UnsupportedParquetMetadata('Unsupported Parquet metadata container')
            self.value(kind, depth + 1, child, field_boolean=True)

    def value(self, kind, depth, context=None, *, field_boolean=False):
        self.step(depth)
        if kind in (1, 2):
            if not field_boolean and self.byte() not in (1, 2):
                raise UnsupportedParquetMetadata('Unsupported Compact-Thrift boolean')
        elif kind == 3:
            self.take(1)
        elif kind in (4, 5, 6):
            self.integer({4: 16, 5: 32, 6: 64}[kind])
        elif kind == 7:
            self.take(8)
        elif kind == 8:
            self.take(self.integer())
        elif kind in (9, 10):
            header = self.byte()
            count, element = header >> 4, header & 15
            if count == 15:
                count = self.integer()
            if count > 65536 or context and element != 12:
                raise UnsupportedParquetMetadata('Compact-Thrift collection exceeds supported bounds')
            for _ in range(count):
                self.value(element, depth + 1, context)
        elif kind == 11:
            count = self.integer()
            if count > 65536:
                raise UnsupportedParquetMetadata('Compact-Thrift map exceeds supported bounds')
            if count:
                pair = self.byte()
                for _ in range(count):
                    self.value(pair >> 4, depth + 1)
                    self.value(pair & 15, depth + 1)
        elif kind == 12:
            self.structure(depth + 1, context)
        elif kind == 13:
            self.take(16)
        else:
            raise UnsupportedParquetMetadata('Unsupported Compact-Thrift value encoding')


def require_unencrypted_parquet(source, *, max_footer=16 * 1024**2):
    """Refuse unsupported/encrypted footer metadata before native column access.

    Caller retains file authorization and its own tighter allocation budgets.
    Stream position is preserved. A successful scan establishes no row validity.
    """
    if type(max_footer) is not int or not 1 <= max_footer <= 16 * 1024**2:
        raise UnsupportedParquetMetadata('Unsupported footer inspection allowance')

    def inspect(stream):
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
        if stream.read(4) != b'PAR1' or size < 12:
            raise UnsupportedParquetMetadata('Only bounded ordinary plaintext Parquet is assessed')
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        length = int.from_bytes(trailer[:4], 'little')
        if trailer[4:] != b'PAR1' or not 0 < length <= min(max_footer, size - 12):
            raise UnsupportedParquetMetadata('Parquet footer exceeds supported framing or inspection bounds')
        stream.seek(-8 - length, os.SEEK_END)
        footer = _CompactFooter(stream.read(length))
        footer.structure(context='file')
        if footer.offset != length:
            raise UnsupportedParquetMetadata('Trailing or unsupported Parquet footer metadata')

    if hasattr(source, 'read') and hasattr(source, 'seek'):
        position = source.tell()
        try:
            inspect(source)
        finally:
            source.seek(position)
    else:
        with open(source, 'rb') as stream:
            inspect(stream)


def _open_regular(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError('Parquet paths must not contain symlinks')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    handle = os.fdopen(descriptor, 'rb')
    before = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE or before.st_size < 12:
        handle.close()
        raise ValueError('Parquet file exceeds regular-file byte bounds')
    return path, handle, before


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _schema_leaves(pa, schema):
    """Classify physical leaves; no Arrow/Python values are materialized."""
    leaves = []

    def visit(dtype, depth):
        if depth > MAX_DEPTH:
            raise ValueError('Parquet schema nesting exceeds supported depth')
        if pa.types.is_dictionary(dtype):
            visit(dtype.value_type, depth + 1)
        elif pa.types.is_struct(dtype):
            if not 1 <= dtype.num_fields <= MAX_COLUMNS:
                raise ValueError('Parquet struct exceeds supported field bounds')
            for field in dtype:
                visit(field.type, depth + 1)
        elif pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
            if pa.types.is_fixed_size_list(dtype) and not 1 <= dtype.list_size <= 4096:
                raise ValueError('Parquet fixed vector width exceeds supported bounds')
            visit(dtype.value_type, depth + 1)
        elif pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype):
            leaves.append(('numeric', depth))
        elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            leaves.append(('string', depth))
        else:
            raise ValueError('Unsupported Parquet schema type: ' + str(dtype))
        if len(leaves) > MAX_COLUMNS:
            raise ValueError('Parquet schema exceeds supported leaf count')

    if not 1 <= len(schema) <= MAX_COLUMNS:
        raise ValueError('Parquet schema exceeds supported column bounds')
    for field in schema:
        visit(field.type, 1)
    return leaves


def guard_file(path):
    """Bound metadata and conservative decoded allocation before a value read."""
    path, handle, before = _open_regular(path)
    with handle:
        if handle.read(4) != b'PAR1':
            raise ValueError('Only ordinary unencrypted Parquet is supported')
        handle.seek(-8, os.SEEK_END)
        trailer = handle.read(8)
        footer_size = int.from_bytes(trailer[:4], 'little')
        if trailer[4:] != b'PAR1' or not 1 <= footer_size <= min(MAX_FOOTER, before.st_size - 12):
            raise ValueError('Parquet footer exceeds bounded metadata size')
        require_unencrypted_parquet(handle, max_footer=MAX_FOOTER)
        import pyarrow as pa
        import pyarrow.parquet as pq

        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        parquet = pq.ParquetFile(handle, thrift_string_size_limit=MAX_FOOTER,
                                 thrift_container_size_limit=100_000)
        metadata = parquet.metadata
        if (not 0 <= metadata.num_rows <= MAX_ROWS or not 1 <= metadata.num_columns <= MAX_COLUMNS
                or not 0 <= metadata.num_row_groups <= MAX_ROW_GROUPS):
            raise ValueError('Parquet dimensions exceed supported bounds')
        leaves = _schema_leaves(pa, parquet.schema_arrow)
        if len(leaves) != metadata.num_columns:
            raise ValueError('Parquet logical and physical leaf counts differ')
        bound = 0
        values = 0
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            if group.num_columns != len(leaves):
                raise ValueError('Parquet row-group schema differs')
            for column_index, (kind, depth) in enumerate(leaves):
                column = group.column(column_index)
                count, encoded = column.num_values, column.total_uncompressed_size
                if type(count) is not int or type(encoded) is not int or count < 0 or encoded < 0:
                    raise ValueError('Invalid Parquet allocation metadata')
                # Numeric/object/offset overhead and RLE/dictionary expansion are
                # charged independently of the physical page sizes. One string
                # may repeat count times; statistics.max is never a length cap.
                allocation = encoded + count * 64 * depth
                if kind == 'string':
                    allocation += encoded * count
                bound += allocation
                values += count
                if bound > MAX_DECODED:
                    raise ValueError('Parquet conservative decoded allocation exceeds 32 MiB')
        if (_identity(os.fstat(handle.fileno())) != _identity(before)
                or _identity(path.stat(follow_symlinks=False)) != _identity(before)):
            raise ValueError('Parquet source changed during preflight')
        return {'path': str(path), 'size_bytes': before.st_size, 'footer_bytes': footer_size,
                'rows': metadata.num_rows, 'row_groups': metadata.num_row_groups,
                'leaf_columns': metadata.num_columns, 'leaf_values': values,
                'decoded_bound_bytes': bound}


def run(request):
    if (type(request) is not dict or set(request) - {'operation', 'paths', 'output'}
            or request.get('operation') != 'guard_parquet'):
        raise ValueError('Unsupported fixed Parquet guard request')
    paths = request.get('paths')
    if (type(paths) is not list or not 1 <= len(paths) <= MAX_FILES
            or any(type(path) is not str or not path or len(path) > 4096 for path in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError('Parquet guard requires bounded unique paths')
    files, total = [], 0
    for path in paths:
        result = guard_file(path)
        total += result['decoded_bound_bytes']
        if total > MAX_DECODED:
            raise ValueError('Aggregate Parquet decoded allocation exceeds 32 MiB')
        files.append(result)
    return {'operation': 'guard_parquet', 'execution_status': 'complete',
            'decoded_bound_bytes': total, 'files': files}


def main():
    request = None
    output = None
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(raw) > MAX_REQUEST:
            raise ValueError('Parquet guard request exceeds byte bound')
        request = json.loads(raw)
        if type(request) is not dict or type(request.get('output')) is not str:
            raise ValueError('Parquet guard requires an owned output directory')
        candidate = Path(request['output']).absolute()
        if (any(parent.is_symlink() for parent in (candidate, *candidate.parents))
                or (candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())))):
            raise ValueError('Parquet guard output must be an empty regular directory')
        candidate.mkdir(parents=True, exist_ok=True)
        output = candidate
        result = run(request)
    except Exception as error:
        result = {'operation': 'guard_parquet', 'execution_status': 'error',
                  'error': {'type': type(error).__name__, 'message': str(error)[:2000]}}
    if output is not None:
        (output / 'result.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print(json.dumps({'execution_status': result['execution_status']}))
    return 0 if result['execution_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
