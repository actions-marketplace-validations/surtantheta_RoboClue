"""Fixed, resource-limited Parquet intake; never imports artifact code."""
import hashlib
import json
import math
import sys
from pathlib import Path


def run(request):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(request['path'])
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows > 100_000 or parquet.metadata.num_columns > 256:
        raise ValueError('Native row intake exceeds bounded Parquet dimensions')
    columns = request['columns']
    names = parquet.schema_arrow.names
    identity = ['episode_index', 'index', 'timestamp']
    has_frame = 'frame_index' in names
    if has_frame:
        identity.append('frame_index')
    if not set(identity + columns) <= set(names):
        raise ValueError('Original native identity/state columns are missing')
    for name in identity:
        dtype = parquet.schema_arrow.field(name).type
        if not (pa.types.is_integer(dtype) or (name == 'timestamp' and pa.types.is_floating(dtype))):
            raise ValueError('Native identity columns require numeric scalar schema')
    for name in columns:
        dtype = parquet.schema_arrow.field(name).type
        if not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype)):
            raise ValueError('Native state requires a numeric vector schema')
        if not (pa.types.is_integer(dtype.value_type) or pa.types.is_floating(dtype.value_type)):
            raise ValueError('Native state requires a numeric vector schema')
    wanted = set(identity + columns)
    expanded = sum(parquet.metadata.row_group(g).column(c).total_uncompressed_size
                   for g in range(parquet.metadata.num_row_groups)
                   for c in range(parquet.metadata.row_group(g).num_columns)
                   if parquet.metadata.row_group(g).column(c).path_in_schema.split('.list.')[0] in wanted)
    if expanded > 32 * 1024**2:
        raise ValueError('Selected native Parquet columns exceed 32 MiB decoded budget')
    # Encoded page sizes do not bound dictionary/RLE expansion or Python objects.
    # Account for every selected leaf value before Arrow reads any column values.
    values = sum(parquet.metadata.row_group(g).column(c).num_values
                 for g in range(parquet.metadata.num_row_groups)
                 for c in range(parquet.metadata.row_group(g).num_columns)
                 if parquet.metadata.row_group(g).column(c).path_in_schema.split('.list.')[0] in wanted)
    if values > (32 * 1024**2) // 64:
        raise ValueError('Selected native Parquet values exceed conservative decoded allocation budget')
    rows = parquet.read(columns=sorted(wanted)).to_pylist()
    row_mode = all('physical_row' in sample for sample in request['samples'])
    if row_mode and has_frame:
        raise ValueError('Original frame_index is available; select it explicitly instead of a physical row')
    if not has_frame and not row_mode:
        raise ValueError('Original file has no native frame_index; explicitly select physical_row instead')
    if row_mode:
        # Official GR1 v2 files omit frame_index. Explicitly select an ordinal
        # within one complete episode file, without claiming a native frame ID.
        episodes = {sample['episode_index'] for sample in request['samples']}
        if len(episodes) != 1 or any(type(row['episode_index']) is not int or row['episode_index'] not in episodes for row in rows):
            raise ValueError('Frame-less selection requires an original single-episode file')
        indexes = [row['index'] for row in rows]
        if any(type(v) is not int or v < 0 for v in indexes) or len(set(indexes)) != len(indexes):
            raise ValueError('Frame-less original index identity must be unique')
    selected = []
    for sample in request['samples']:
        matches = [(i, row) for i, row in enumerate(rows) if row['episode_index'] == sample['episode_index']
                   and (i == sample['physical_row'] if row_mode else row['frame_index'] == sample['frame_index'])]
        if len(matches) != 1:
            raise ValueError('Need exactly one original native sample identity')
        position, row = matches[0]
        if any(type(row[k]) is not int or row[k] < 0 for k in identity if k != 'timestamp'):
            raise ValueError('Native sample IDs must be original nonnegative integers')
        if type(row['timestamp']) not in (int, float) or not math.isfinite(row['timestamp']):
            raise ValueError('Native sample timestamp must be finite')
        values = {key: row[key] for key in columns}
        for value in values.values():
            if not isinstance(value, list) or not 1 <= len(value) <= 4096 or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in value):
                raise ValueError('Native state must be a bounded finite numeric vector')
        selected.append({'identity': {**{key: row[key] for key in identity}, 'frame_index': row.get('frame_index'), 'physical_row': position},
                         'selection_basis': 'single_episode_physical_row' if row_mode else 'original_frame_index',
                         'columns': values, 'source': {'path': request['relative_path'],
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    return {'raw_samples': selected}


if __name__ == '__main__':
    payload = json.load(sys.stdin)
    result = run(payload)
    (Path(payload['output']) / 'result.json').write_text(json.dumps(result, allow_nan=False))
