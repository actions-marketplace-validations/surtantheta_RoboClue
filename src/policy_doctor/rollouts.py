"""Bounded offline numeric recording comparison. No interpolation or causal inference."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import struct
import zipfile
from pathlib import Path

import numpy as np

from .artifacts import InputError

MAX_BYTES = 64 * 1024 * 1024
MAX_VALUES = 8_000_000
MAX_STREAMS = 32
MAX_NPZ_INVENTORY_BYTES = 256 * 1024 * 1024


def _stream_selection(streams):
    if streams is None:
        return None
    if (type(streams) is not list or not 1 <= len(streams) <= MAX_STREAMS or
            any(type(key) is not str or not key or not key.isprintable() or
                len(key.encode('utf-8')) > 256 for key in streams) or len(set(streams)) != len(streams)):
        raise InputError('streams must be 1–32 unique nonempty string native NPZ keys (at most 256 UTF-8 bytes each)')
    return list(streams)


def _source(path: str) -> tuple[Path, bytes, dict]:
    try:
        source = Path(path).expanduser().resolve()
        limit = 256 * 1024 * 1024 if source.suffix.lower() in ('.h5', '.hdf5') else MAX_BYTES
        if not source.exists():
            raise InputError(f'Recording input does not exist: {str(path)!r}. Check the recording path and rerun.')
        if not source.is_file() or source.stat().st_size > limit:
            raise InputError(f'Recording must be a regular file no larger than {limit} bytes: {str(path)!r}')
        with source.open('rb') as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise InputError(f'Recording exceeds byte limit: {str(path)!r}')
    except (OSError, RuntimeError) as exc:
        raise InputError(f'Cannot read recording {str(path)!r}: {exc}') from exc
    return source, raw, {'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def _npz_preflight(raw: bytes, expanded_limit: int, *, max_streams: int = MAX_STREAMS) -> None:
    """Bound actual central records before ZipFile allocates one ZipInfo per record.

    Standard central directories are sufficient for bounded NPZ files. ZIP64
    local headers (used by NumPy) are supported; ZIP64 central/multi-disk,
    ambiguous end signatures and trailing bytes are deliberately refused.
    """
    if type(max_streams) is not int or not 1 <= max_streams <= 4096:
        raise InputError('NPZ member cap must be an integer between 1 and 4096')
    end = raw.rfind(b'PK\x05\x06', max(0, len(raw) - 65557))
    # Match ZipFile's LAST-signature choice; falling back to an earlier valid
    # record would allow it to allocate from a different directory afterwards.
    if end < 0 or end + 22 > len(raw):
        raise InputError('NPZ requires a bounded standard ZIP end record with no trailing bytes')
    fields = struct.unpack_from('<4s4H2IH', raw, end)
    if end + 22 + fields[7] != len(raw):
        raise InputError('NPZ ambiguous ZIP end signature or trailing bytes are refused')
    # _EndRecData64 honors this locator even without ordinary EOCD sentinels.
    # Refuse it before the native parser can override our count/size/offset.
    if end >= 20 and raw[end - 20:end - 16] == b'PK\x06\x07':
        raise InputError('NPZ ZIP64 central directory locator is unsupported')
    _, disk, directory_disk, disk_count, count, directory_size, directory_offset, _ = fields
    if disk or directory_disk or disk_count != count:
        raise InputError('NPZ multi-disk or inconsistent central directories are refused')
    if count == 0xffff or directory_size == 0xffffffff or directory_offset == 0xffffffff:
        raise InputError('NPZ ZIP64 central directory is unsupported within the bounded archive limit')
    start = end - directory_size
    prefix = start - directory_offset
    if start < 0 or prefix < 0:
        raise InputError('NPZ central directory bounds are invalid')
    position, actual_count, expanded = start, 0, 0
    while position < end:
        if position + 46 > end or raw[position:position + 4] != b'PK\x01\x02':
            raise InputError('NPZ central directory records are malformed or unsupported')
        record = struct.unpack_from('<4s6H3I5H2I', raw, position)
        actual_count += 1
        if actual_count > max_streams:
            raise InputError(f'NPZ actual central directory exceeds {max_streams} array members')
        name_size, extra_size, comment_size = record[10:13]
        next_position = position + 46 + name_size + extra_size + comment_size
        if not name_size or next_position > end:
            raise InputError('NPZ central directory member lengths are invalid')
        if record[13] or any(record[index] == 0xffffffff for index in (8, 9, 16)):
            raise InputError('NPZ ZIP64 central or multi-disk members are unsupported')
        if record[16] + prefix + 30 > start:
            raise InputError('NPZ central directory local-header offset is invalid')
        extra, extra_end = position + 46 + name_size, position + 46 + name_size + extra_size
        while extra < extra_end:
            if extra + 4 > extra_end:
                raise InputError('NPZ central directory extra fields are malformed')
            tag, size = struct.unpack_from('<HH', raw, extra)
            if tag == 1:
                raise InputError('NPZ ZIP64 central directory extra fields are unsupported')
            extra += 4 + size
            if extra > extra_end:
                raise InputError('NPZ central directory extra fields are malformed')
        expanded += record[9]
        if expanded > expanded_limit:
            raise InputError('NPZ excessive expanded inventory size')
        position = next_position
    if not actual_count or actual_count != count or position != end:
        raise InputError('NPZ actual central directory count or size disagrees with its end record')


def _npz(raw: bytes, values: bool) -> tuple[dict, dict]:
    specs, arrays, total = {}, {}, 0
    try:
        _npz_preflight(raw, MAX_BYTES)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_STREAMS:
                raise InputError('NPZ requires 1–32 numeric array members')
            if len({e.filename for e in entries}) != len(entries) or sum(e.file_size for e in entries) > MAX_BYTES:
                raise InputError('NPZ duplicate members or excessive expanded size')
            for entry in entries:
                name = entry.filename
                if not name.endswith('.npy') or name.startswith('/') or '..' in Path(name).parts:
                    raise InputError('NPZ members must be relative .npy arrays')
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise InputError('Unsupported NPY header version')
                    shape, _, dtype = reader(stream, max_header_size=10000)
                    size = math.prod(shape)
                    total += size
                    if dtype.kind not in 'biuf' or dtype.itemsize > 8 or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape):
                        raise InputError('Recording arrays must be real numeric/bool, rank <=4; pickle/object data is refused')
                    if total > MAX_VALUES or size * dtype.itemsize != entry.file_size - stream.tell():
                        raise InputError('Recording array budget or declared payload size is invalid')
                key = name[:-4]
                specs[key] = {'shape': list(shape), 'dtype': str(dtype)}
            if values:
                with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=10000) as loaded:
                    for key in specs:
                        array = loaded[key]
                        if not np.isfinite(array).all():
                            raise InputError(f'Non-finite values in recording stream {key}')
                        arrays[key] = array
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, OverflowError, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f'Invalid numeric NPZ: {exc}') from exc
    return specs, arrays


def _npz_selected(raw: bytes, values: bool, streams: list[str]) -> tuple[dict, dict, dict]:
    """Inventory every header, decode only exact requested members; never use pickle."""
    specs, arrays, inventory, members = {}, {}, {}, {}
    selected_values = selected_bytes = 0
    try:
        _npz_preflight(raw, MAX_NPZ_INVENTORY_BYTES)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_STREAMS:
                raise InputError('NPZ requires 1–32 array members')
            if len({entry.filename for entry in entries}) != len(entries):
                raise InputError('NPZ duplicate members are refused')
            if sum(entry.file_size for entry in entries) > MAX_NPZ_INVENTORY_BYTES:
                raise InputError('NPZ expanded inventory exceeds 256 MiB; provide a smaller native packet')
            for entry in entries:
                name = entry.filename
                parts = name.split('/')
                if (entry.orig_filename != name or not name.endswith('.npy') or name.startswith('/') or
                        '\\' in name or ':' in name or
                        any(part in ('', '.', '..') for part in parts) or
                        not name[:-4] or len(name[:-4].encode('utf-8')) > 256 or not name.isprintable()):
                    raise InputError('NPZ members must be unambiguous bounded relative .npy array names')
                if entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise InputError('NPZ encrypted or unsupported-compression members are refused')
                key = name[:-4]
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise InputError('Unsupported NPY header version')
                    shape, _, dtype = reader(stream, max_header_size=10000)
                    if (len(shape) > 32 or any(type(n) is not int or not 0 <= n <= 2**63 - 1 for n in shape)):
                        raise InputError('NPZ inventory shape is invalid or exceeds bounded rank 32')
                    if dtype.hasobject:
                        raise InputError('NPZ pickle/object data is refused even in excluded streams')
                    if len(str(dtype)) > 512:
                        raise InputError('NPZ dtype inventory exceeds the bounded 512-character description')
                    size = math.prod(shape)
                    payload_bytes = size * dtype.itemsize
                    if payload_bytes != entry.file_size - stream.tell():
                        raise InputError('NPZ declared payload size is invalid')
                supported = dtype.kind in 'biuf' and 0 < dtype.itemsize <= 8 and len(shape) <= 4
                selected = key in streams
                if selected:
                    if not supported:
                        raise InputError(f'Selected stream {key} must be real numeric/bool, rank <=4, item size <=8')
                    selected_values += size
                    selected_bytes += payload_bytes
                    if selected_values > MAX_VALUES or selected_bytes > MAX_BYTES:
                        raise InputError('Selected NPZ numeric array budget exceeded (8M values / 64 MiB)')
                    specs[key] = {'shape': list(shape), 'dtype': str(dtype)}
                inventory[key] = {'shape': list(shape), 'dtype': str(dtype), 'selected': selected,
                                  'supported_numeric': supported, 'payload_bytes': payload_bytes,
                                  'values_decoded': False,
                                  'reason': 'Selected numeric stream; header only.' if selected else
                                            'Explicitly excluded; values not decoded or validated.'}
                members[key] = entry
            missing = [key for key in streams if key not in members]
            if missing:
                raise InputError(f'Selected native NPZ streams are missing: {missing}')
            # Exact ZipInfo lookup avoids NpzFile's raw filename/key ambiguity (foo vs foo.npy).
            if values:
                for key in streams:
                    with archive.open(members[key]) as stream:
                        array = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=10000)
                    if list(array.shape) != specs[key]['shape'] or str(array.dtype) != specs[key]['dtype']:
                        raise InputError('Decoded NPZ array does not match its inventoried header')
                    if not np.isfinite(array).all():
                        raise InputError(f'Non-finite values in recording stream {key}')
                    arrays[key] = array
                    inventory[key]['values_decoded'] = True
                    inventory[key]['reason'] = 'Selected numeric values decoded and checked for finiteness.'
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, OverflowError,
            RuntimeError, NotImplementedError, UnicodeError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f'Invalid selected-stream NPZ: {exc}') from exc
    return specs, arrays, {
        'selected': list(streams), 'excluded': [key for key in inventory if key not in streams],
        'inventory': inventory,
        'limits': {'archive_bytes': MAX_BYTES, 'expanded_inventory_bytes': MAX_NPZ_INVENTORY_BYTES,
                   'selected_payload_bytes': MAX_BYTES, 'selected_values': MAX_VALUES,
                   'members': MAX_STREAMS, 'header_bytes_per_member': 10000},
        'scope': 'Explicit selected numeric streams only, not whole-file equivalence. '
                 'Excluded member headers and declared byte lengths are inspected; excluded values are not decoded '
                 'or validated. Original archive bytes are hashed; no streams, IDs, units or phase are inferred.'}


def _csv(raw: bytes) -> tuple[dict, dict]:
    try:
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')))
        names = next(reader)
        if not names or len(names) > MAX_STREAMS or len(set(names)) != len(names) or any(not n for n in names):
            raise InputError('CSV requires 1–32 unique nonempty numeric column names')
        columns = [[] for _ in names]
        for index, row in enumerate(reader):
            if index >= 100000 or len(row) != len(names):
                raise InputError('CSV row limit exceeded or inconsistent columns')
            for column, value in zip(columns, row, strict=True):
                try:
                    number = int(value)
                except ValueError:
                    number = float(value)
                column.append(number)
        arrays = {}
        for name, column in zip(names, columns, strict=True):
            integers = [value for value in column if type(value) is int]
            if integers:
                low, high = min(integers), max(integers)
                if len(integers) == len(column):
                    if low < -(2**63) or high >= 2**64 or (low < 0 and high >= 2**63):
                        raise InputError('CSV integer column has no exact supported 64-bit representation')
                    dtype = np.uint64 if high >= 2**63 else np.int64
                else:
                    if low < -(2**53) or high > 2**53:
                        raise InputError('CSV mixed integer/float column would lose integer precision')
                    dtype = np.float64
            else:
                dtype = np.float64
            arrays[name] = np.asarray(column, dtype=dtype)
        if any(a.dtype.kind not in 'biuf' or not np.isfinite(a).all() for a in arrays.values()):
            raise InputError('CSV contains unsupported or non-finite values')
        return {key: {'shape': list(a.shape), 'dtype': str(a.dtype)} for key, a in arrays.items()}, arrays
    except (ValueError, UnicodeError, csv.Error, StopIteration, OverflowError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f'Invalid numeric CSV: {exc}') from exc


def _parquet(raw: bytes, episode) -> tuple[dict, dict, int, list]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputError('Install policy-deployment-doctor[dataset] for native LeRobot Parquet') from exc
    try:
        from .native_parquet_guard import require_unencrypted_parquet

        require_unencrypted_parquet(io.BytesIO(raw))
        file = pq.ParquetFile(io.BytesIO(raw))
        if file.metadata.num_rows > 1_000_000:
            raise InputError('Parquet episode selection scan exceeds one million rows')
        columns = []
        for field in file.schema_arrow:
            scalar = field.type.value_type if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type) else field.type
            if pa.types.is_integer(scalar) or pa.types.is_floating(scalar) or pa.types.is_boolean(scalar):
                columns.append(field.name)
        if 'episode_index' not in columns or 'action' not in columns:
            raise InputError('Native LeRobot Parquet requires numeric action and episode_index; use CSV/NPZ for generic recordings')
        if len(columns) > MAX_STREAMS:
            raise InputError('Parquet has more than 32 numeric streams')
        if isinstance(episode, str) and episode.isdecimal():
            episode = int(episode)
        if episode is not None and type(episode) is not int:
            raise InputError('LeRobot episode must be an integer index')
        leaves = [file.metadata.row_group(i).column(j)
                  for i in range(file.metadata.num_row_groups)
                  for j in range(file.metadata.num_columns)
                  if any(file.metadata.row_group(i).column(j).path_in_schema == name or
                         file.metadata.row_group(i).column(j).path_in_schema.startswith(name + '.') for name in columns)]
        if (sum(leaf.total_uncompressed_size for leaf in leaves) > MAX_BYTES or
                sum(leaf.num_values for leaf in leaves) > MAX_VALUES):
            raise InputError('Parquet numeric columns exceed expanded byte/value scan budget')
        selected, observed, total = [], set(), 0
        for batch in file.iter_batches(batch_size=1024, columns=columns):
            ids = batch.column('episode_index')
            if not pa.types.is_integer(ids.type) or ids.null_count:
                raise InputError('Invalid native episode_index')
            if episode is None:
                observed.update(ids.unique().to_pylist())
                if len(observed) > 1:
                    raise InputError('Multiple native episodes; supply reference_episode/candidate_episode')
            else:
                batch = batch.filter(pc.equal(ids, episode))
            total += sum(len(column.values) if pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type)
                         else len(column) for column in batch.columns)
            if total > MAX_VALUES or len(selected) + batch.num_rows > 100000:
                raise InputError('Selected episode exceeds numeric value/row budget')
            selected.extend(batch.to_pylist())
        if not selected:
            raise InputError('Requested episode not present in this Parquet file')
        arrays = {key: np.asarray([row[key] for row in selected]) for key in columns}
        if sum(a.size for a in arrays.values()) > MAX_VALUES or any(a.dtype.kind not in 'biuf' or not np.isfinite(a).all() for a in arrays.values()):
            raise InputError('Native episode exceeds numeric budget or contains invalid values')
        return ({k: {'shape': list(a.shape), 'dtype': str(a.dtype)} for k, a in arrays.items()}, arrays,
                selected[0]['episode_index'], [n for n in file.schema_arrow.names if n not in columns])
    except (ValueError, OSError, TypeError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f'Invalid native Parquet recording: {exc}') from exc


def _hdf5(raw: bytes, values: bool, episode) -> tuple[dict, dict, str, list]:
    try:
        import h5py
    except ImportError as exc:
        raise InputError('Install policy-deployment-doctor[dataset] for robomimic HDF5') from exc
    def member(group, name):
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise InputError('HDF5 soft/external links are refused')
        return group[name]
    specs, arrays, excluded, total = {}, {}, [], 0
    try:
        with h5py.File(io.BytesIO(raw), 'r') as file:
            data = member(file, 'data')
            if not isinstance(data, h5py.Group):
                raise InputError('robomimic data must be a group')
            if episode is None:
                names = []
                for name in data:
                    names.append(name)
                    if len(names) > 1:
                        raise InputError('Multiple native episodes; select a demo_N episode explicitly')
                if not names:
                    raise InputError('No native episodes')
                episode = names[0]
            if type(episode) is int:
                episode = f'demo_{episode}'
            if not isinstance(episode, str) or not episode.startswith('demo_') or '/' in episode or episode not in data:
                raise InputError('Invalid or absent robomimic episode selector')
            selected = member(data, episode)
            if not isinstance(selected, h5py.Group):
                raise InputError('Native episode must be a group')
            for prefix in ('', 'obs', 'next_obs'):
                if prefix and prefix not in selected:
                    continue
                group = selected if not prefix else member(selected, prefix)
                if not isinstance(group, h5py.Group):
                    raise InputError('Native observation container must be a group')
                seen = 0
                for name in group:
                    seen += 1
                    if seen > 128:
                        raise InputError('HDF5 episode member budget exceeded')
                    item = member(group, name)
                    if isinstance(item, h5py.Group):
                        continue
                    key = f'{prefix}/{name}' if prefix else name
                    if item.is_virtual or item.external:
                        raise InputError('HDF5 virtual or external-storage datasets are refused')
                    if item.ndim not in (1, 2) or item.dtype.kind not in 'biuf' or item.dtype.itemsize > 8:
                        excluded.append(key)
                        continue  # Preserve obs/next_obs; do not decode image arrays.
                    total += math.prod(item.shape)
                    if total > MAX_VALUES or len(specs) >= MAX_STREAMS or item.shape[0] > 100000:
                        raise InputError('HDF5 numeric episode budget exceeded')
                    creation = item.id.get_create_plist()
                    if any(creation.get_filter(i)[0] not in (1, 2, 3, 4, 5, 6, 32000) for i in range(creation.get_nfilters())):
                        raise InputError('Unsupported HDF5 filter; dynamic plugins are not used')
                    specs[key] = {'shape': list(item.shape), 'dtype': str(item.dtype)}
                    if values:
                        arrays[key] = item[...]
                        if not np.isfinite(arrays[key]).all():
                            raise InputError('Non-finite native HDF5 stream')
            if 'actions' not in specs:
                raise InputError('robomimic episode has no numeric actions dataset')
        return specs, arrays, episode, excluded
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f'Invalid native HDF5: {exc}') from exc


def _recording(path: str, values: bool, episode=None, streams=None) -> tuple[dict, dict, dict]:
    streams = _stream_selection(streams)
    source, raw, artifact = _source(path)
    if streams is not None and source.suffix.lower() != '.npz':
        raise InputError('Explicit stream selection is supported for NPZ recordings only')
    if source.suffix.lower() == '.npz':
        if streams is None:
            specs, arrays = _npz(raw, values)
        else:
            specs, arrays, artifact['stream_selection'] = _npz_selected(raw, values, streams)
    elif source.suffix.lower() == '.csv':
        specs, arrays = _csv(raw)
    elif source.suffix.lower() == '.parquet':
        specs, arrays, artifact['episode'], artifact['excluded_streams'] = _parquet(raw, episode)
    elif source.suffix.lower() in ('.h5', '.hdf5'):
        specs, arrays, artifact['episode'], artifact['excluded_streams'] = _hdf5(raw, values, episode)
    else:
        raise InputError('Supported recordings: LeRobot .parquet, robomimic .hdf5/.h5, numeric .npz/.csv')
    if episode is not None and source.suffix.lower() in ('.csv', '.npz'):
        raise InputError('Episode selectors apply to native Parquet/HDF5 recordings only')
    return specs, arrays, artifact


def inspect_rollout(path: str, episode=None, *, streams: list[str] | None = None) -> dict:
    specs, _, artifact = _recording(path, False, episode, streams)
    return {'streams': specs, 'artifact': artifact, 'values_validated': Path(path).suffix.lower() in ('.csv', '.parquet'),
            'scope': 'Array inventory; names do not establish units, action phase, timestamps or controller state.'}


def _integer_equal(left, right):
    """Avoid NumPy 1.x int64/uint64 promotion to inexact float64."""
    if left.dtype.kind == 'i' and right.dtype.kind == 'u':
        return (left >= 0) & (left.astype(np.uint64) == right.astype(np.uint64))
    if left.dtype.kind == 'u' and right.dtype.kind == 'i':
        return (right >= 0) & (left.astype(np.uint64) == right.astype(np.uint64))
    return left == right


def _index_equal(left, right):
    if left.dtype.kind in 'iu' and right.dtype.kind in 'iu':
        return _integer_equal(left, right).all()
    # Float IDs cannot authenticate large integer identity across dtype conversion.
    if any(np.any(v < -(2**53)) or np.any(v > 2**53) for v in (left, right)):
        return False
    return np.equal(left, right).all()


def compare_rollouts(reference: str, candidate: str, *, alignment_key: str | None = None,
                     atol: float = 1e-6, rtol: float = 1e-5, max_rows: int = 100000,
                     reference_episode: int | str | None = None, candidate_episode: int | str | None = None,
                     streams: list[str] | None = None) -> dict:
    if (type(max_rows) is not int or not 1 <= max_rows <= 100000 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (atol, rtol))):
        raise InputError('Require max_rows in [1,100000] and finite nonnegative tolerances')
    if alignment_key is not None and (not isinstance(alignment_key, str) or not alignment_key):
        raise InputError('alignment_key must be a nonempty stream name')
    streams = _stream_selection(streams)
    if streams is not None and alignment_key is not None and alignment_key not in streams:
        raise InputError('The alignment_key must be explicitly included in streams')
    a_spec, a, a_artifact = _recording(reference, True, reference_episode, streams)
    b_spec, b, b_artifact = _recording(candidate, True, candidate_episode, streams)
    common = sorted(set(a) & set(b))
    missing = {'reference_only': sorted(set(a) - set(b)), 'candidate_only': sorted(set(b) - set(a))}
    aligned = False
    alignment_reason = 'Row-position comparison only; no shared sequential integer index was supplied.'
    if alignment_key:
        left, right = a.get(alignment_key), b.get(alignment_key)
        if left is not None and right is not None and left.ndim == right.ndim == 1 and len(left) == len(right) and len(left):
            numeric_index = all(v.dtype.kind in 'iu' or (v.dtype.kind == 'f' and np.equal(v, np.floor(v)).all()) for v in (left, right))
            sequential = numeric_index and np.equal(np.diff(left), 1).all() and np.equal(np.diff(right), 1).all() and (left[1:] > left[:-1]).all() and (right[1:] > right[:-1]).all()
            aligned = bool(sequential and _index_equal(left, right))
        alignment_reason = 'Matching consecutive integer sample IDs.' if aligned else 'Supplied index is missing, unequal, empty, gapped or not sequential integer IDs.'
    complete = bool(aligned and common and not any(missing.values()))
    results, earliest, context_first = [], None, None
    native_parquet = all(Path(p).suffix.lower() == '.parquet' for p in (reference, candidate))
    for key in common:
        if key == alignment_key:
            continue
        left, right = a[key], b[key]
        context = native_parquet and key in ('episode_index', 'index', 'task_index', 'frame_index')
        stream = {'key': key, 'role': 'native_bookkeeping' if context else 'recorded_numeric_signal',
                  'reference_shape': a_spec[key]['shape'], 'candidate_shape': b_spec[key]['shape']}
        if left.ndim < 1 or right.ndim < 1 or left.shape[1:] != right.shape[1:] or not len(left) or not len(right):
            complete = False
            results.append(dict(stream, status='incompatible', equal=None, rows_compared=0))
            continue
        rows = min(len(left), len(right), max_rows)
        coverage = rows == len(left) == len(right)
        if aligned:
            coverage = coverage and len(left) == len(a[alignment_key])
        complete &= coverage
        x, y = left[:rows], right[:rows]
        # Integer/bool streams compare exactly, including values above 2**53.
        discrete = x.dtype.kind in 'biu' and y.dtype.kind in 'biu'
        if not discrete and x.dtype.kind != y.dtype.kind:
            for v in (x, y):
                if v.dtype.kind in 'iu' and (np.any(v > 2**53) or (v.dtype.kind == 'i' and np.any(v < -(2**53)))):
                    raise InputError('Mixed integer/float stream would lose integer precision')
        equal = _integer_equal(x, y) if discrete else np.isclose(x, y, atol=atol, rtol=rtol, equal_nan=False)
        mask = ~np.asarray(equal)
        changed = int(np.count_nonzero(mask))
        first = None
        if changed:
            coordinate = tuple(int(i) for i in np.unravel_index(int(np.argmax(mask)), mask.shape))
            row = coordinate[0]
            def evidence(artifact, value, key=key, coordinate=coordinate):
                return {'path': artifact['path'], 'sha256': artifact['sha256'], 'array': key,
                        'episode': artifact.get('episode'),
                        'index': list(coordinate), 'value': value.item()}
            first = {'row': row, 'coordinate': list(coordinate[1:]),
                     'sample_id': a[alignment_key][row].item() if aligned and row < len(a[alignment_key]) else None,
                     'reference_value': x[coordinate].item(), 'candidate_value': y[coordinate].item(),
                     'evidence': [evidence(a_artifact, x[coordinate]), evidence(b_artifact, y[coordinate])]}
            if context and (context_first is None or row < context_first['row']):
                context_first = dict(first, stream=key)
            if not context and (earliest is None or row < earliest['row']):
                earliest = dict(first, stream=key)
        if discrete:
            maximum = None
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                maximum = float(np.max(delta))
            if not math.isfinite(maximum):
                maximum = None
        results.append(dict(stream, status='different' if first else 'equal', equal=not bool(changed),
                            rows_compared=rows, complete=coverage, first_difference=first,
                            different_values=changed, max_absolute_delta=maximum,
                            units='unknown', comparison='exact integer/bool' if discrete else 'per-coordinate numeric tolerance'))
    if not results:
        complete = False
    different = any(s.get('status') == 'different' and s['role'] != 'native_bookkeeping' for s in results)
    return {'schema_version': 1, 'status': 'unknown' if not complete else 'different' if different else 'pass',
            'scope': 'Observed numeric recorded-array differences only. Native episode keys preserved. Images/video, interpolation, units, physical fault and causation are not established.',
            'comparison_complete': bool(complete), 'observed_difference': different,
            'coverage_scope': ('Only explicitly selected native NPZ numeric streams and compared row ranges. '
                               'comparison_complete=True does not establish whole-recording equivalence; '
                               'excluded values are uninspected.' if streams is not None else
                               'Selected numeric streams in the supplied files/episode selection, not dataset-wide episode completeness or video integrity.'),
            'first_observed_difference': earliest, 'streams': results, 'unmatched_streams': missing,
            'bookkeeping_difference': context_first is not None, 'first_bookkeeping_difference': context_first,
            'bookkeeping_scope': 'Native LeRobot episode_index, global index, task_index and frame_index are compared and disclosed separately from numeric signals. Different task IDs still require task-semantic review.',
            'alignment': {'key': alignment_key, 'status': 'verified_sample_ids' if aligned else 'unknown',
                          'reason': alignment_reason, 'sample_phase': 'unknown',
                          'limitation': 'Matching IDs do not establish a common physical initial state, clock, action phase or intervention.'},
            'causal_conclusion': 'unknown',
            'unverified_state': ['controller memory', 'policy RNG/history', 'action queue', 'sensor timestamp/phase', 'units and dimension order'],
            'artifacts': [a_artifact, b_artifact], 'tolerances': {'atol': atol, 'rtol': rtol, 'max_rows': max_rows}}
