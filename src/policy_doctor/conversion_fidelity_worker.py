"""Trusted bounded local numeric reader; never import supplied converter code."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import stat
import sys
from pathlib import Path

from .artifacts import InputError
from .conversion_fidelity import MAX_ORIGINAL, MAX_REPORT, _read_small, _regular, _stamp, _unknown
from .conversion_numeric import (
    MAX_VALUES_BYTES,
    SOURCE_HASHES,
    EpisodeProjection,
    NumericCriterion,
    Selection,
    _digest,
    evaluate_numeric,
)
from .conversion_profile_data import AUXILIARY_FIELDS, JOINT_NAMES
from .native_intake import _directory_fd

MAX_PARQUET = 16 * 1024 ** 2
MAX_PARQUET_TOTAL = 256 * 1024 ** 2
_AUDIT_DENIALS = {}
_PARTIAL = None  # One request per isolated worker; retain known findings on refusal.


def _keep_finding(findings, finding):
    """Bound display while retaining first ERROR witnesses across field/role codes."""
    primary, repeated, unknown, seen = [], [], [], set()
    for item in [*findings, finding]:
        if item['status'] == 'error':
            key = (item.get('code'), item.get('column'), item.get('role'))
            if key in seen:
                repeated.append(item)
            else:
                seen.add(key)
                primary.append(item)
        else:
            unknown.append(item)
    findings[:] = (primary + repeated + unknown)[:64]


def _json(raw):
    def pairs(items):
        value = {}
        for key, part in items:
            if key in value:
                raise InputError('Duplicate conversion metadata key')
            value[key] = part
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(InputError('Nonfinite metadata')))


def _data_files(root, maximum):
    result, entries = set(), 0
    def walk(relative, depth):
        nonlocal entries
        if depth > 3:
            raise InputError('Converted numeric tree nesting exceeds qualified layout')
        fd = _directory_fd(Path(root) / relative)
        try:
            with os.scandir(fd) as children:
                for child in children:
                    entries += 1
                    if entries > maximum:
                        raise InputError('Converted numeric tree exceeds file/membership bound')
                    mode = child.stat(follow_symlinks=False).st_mode
                    name = (Path(relative) / child.name).as_posix()
                    if stat.S_ISDIR(mode):
                        walk(name, depth + 1)
                    elif stat.S_ISREG(mode) and child.name.endswith('.parquet'):
                        result.add(name)
                    else:
                        raise InputError('Converted numeric tree has an unsupported member')
        finally:
            os.close(fd)
    walk('data', 1)
    return result


def read_conversion(plan, temporary):
    import numpy as np

    artifacts, findings = [], []
    counts = {'error': 0, 'unknown': 0}
    membership_errors = 0
    numeric_contract_errors = 0
    held_value_bytes = 0
    auxiliary_pairs = []
    auxiliary = {column: {'width': width, 'source_field': config_field, 'profile_role': role,
                         'expected_rows': 0, 'compared_rows': 0, 'error_count': 0, 'unknown_count': 0,
                         'source_dtypes': [], 'export_dtypes': []}
                 for config_field, column, width, role in AUXILIARY_FIELDS}
    def aux_add(column, status, code, message, **extra):
        auxiliary[column][status + '_count'] += 1
        add(status, code, message, column=column, **extra)
    def reserve_values(size):
        nonlocal held_value_bytes
        if held_value_bytes + size > MAX_VALUES_BYTES:
            return False
        held_value_bytes += size
        return True
    global _PARTIAL
    _PARTIAL = {'artifacts': artifacts, 'findings': findings, 'counts': counts}
    def add(status, code, message, **extra):
        nonlocal membership_errors, numeric_contract_errors
        if status == 'error' and code in {'conversion.export_vector_width', 'conversion.null_export',
                'conversion.missing_export_columns', 'conversion.export_vector_schema', 'conversion.dtype_changed'}:
            numeric_contract_errors += 1
        if status == 'error' and code in {'conversion.duplicate_episode_id', 'conversion.episode_membership',
                'conversion.file_membership', 'conversion.nominal_truncation', 'conversion.row_time_identity',
                'conversion.export_extent', 'conversion.export_vector_width', 'conversion.auxiliary_export_extent',
                'conversion.auxiliary_export_width', 'conversion.auxiliary_missing_export'}:
            membership_errors += 1
        counts[status] += 1
        _keep_finding(findings, {'status': status, 'code': code, 'message': message,
                             'remedy': 'Inspect the named original member and exported row; rerun unchanged criteria on a separately corrected export.', **extra})
    for expected in plan['artifacts']:
        _, observed = _read_small(expected['path'])
        if observed != expected:
            return _unknown('Authorized converter/config/reference bytes changed before numeric intake', artifacts)
        artifacts.append(observed)
    root = Path(plan['paths']['converted_directory'])
    config = plan['config']
    for field in auxiliary.values():
        field['source_dataset'] = config[field['source_field']]
    def metadata(name, jsonl=False):
        raw, artifact = _read_small(str(root / 'meta' / name), 2 * 1024 ** 2)
        artifact['kind'] = 'converted_metadata'
        artifacts.append(artifact)
        lines = raw.splitlines()
        if jsonl and len(lines) > 1024:
            raise InputError('Converted metadata record bound exceeded')
        return [_json(line) for line in lines if line.strip()] if jsonl else _json(raw)
    info = metadata(config['info_fname'])
    episodes = metadata(config['episodes_fname'], True)
    tasks = metadata(config['tasks_fname'], True)
    if type(info) is not dict or type(episodes) is not list or len(episodes) > 256:
        raise InputError('Converted metadata is outside qualified shape bounds')
    for field in ['data_path', 'fps']:
        if info.get(field) != config[field] or type(info.get(field)) is not type(config[field]):
            add('error', 'conversion.metadata_profile', 'Converted metadata changes the declared ' + field)
    expected_task = config['task_index']
    if (any(type(t) is not dict or type(t.get('task_index')) is not int for t in tasks)
            or len({t['task_index'] for t in tasks}) != len(tasks)
            or not any(t['task_index'] == expected_task and t.get('task') == config['language_instruction'] for t in tasks)):
        add('error', 'conversion.task_metadata', 'Task metadata does not uniquely retain the declared task identity/text')
    episode_map = {}
    duplicate_ids = set()
    for position, item in enumerate(episodes):
        if (type(item) is not dict or type(item.get('episode_index')) is not int
                or item['episode_index'] < 0 or type(item.get('length')) is not int or item['length'] < 0):
            raise InputError('Converted episode metadata requires strict original IDs and lengths')
        eid = item['episode_index']
        if eid in episode_map:
            duplicate_ids.add(eid)
            add('error', 'conversion.duplicate_episode_id', 'Converted metadata repeats native episode ID ' + str(eid))
        episode_map[eid] = (position, item)
    data_files = _data_files(root, plan['max_files'])
    if len(artifacts) + len(data_files) + 1 > plan['max_files']:
        raise InputError('Conversion file bound cannot cover all original/export numeric members')
    source_path = plan['paths']['original_recording']
    fd, before = _regular(source_path)
    stream = os.fdopen(fd, 'rb')
    try:
        if not 0 < before.st_size <= MAX_ORIGINAL:
            raise InputError('Original recording exceeds the dedicated12GiB regular-file bound')
        digest, read = hashlib.sha256(), 0
        while block := stream.read(1024 ** 2):
            read += len(block)
            if read > MAX_ORIGINAL:
                raise InputError('Original recording grew beyond its dedicated hash budget')
            digest.update(block)
        if read != before.st_size or _stamp(os.fstat(fd)) != _stamp(before):
            raise InputError('Original recording changed during complete hashing')
        original = {'path': source_path, 'kind': 'original_hdf5', 'sha256': digest.hexdigest(),
                    'size_bytes': read, 'stamp': _stamp(before),
                    'origin_scope': 'Caller-selected local content snapshot; not authenticated origin or historical producing execution'}
        artifacts.append(original)
        stream.seek(0)
        # h5py is imported ONLY after complete original hashing and FD identity binding.
        import h5py
        addresses = set()
        def member(group, name):
            link = group.get(name, getlink=True)
            if link is None:
                raise InputError('Source HDF5 member is absent: ' + name)
            if not isinstance(link, h5py.HardLink):
                raise InputError('Source HDF5 soft/external links are unsupported')
            obj = group[name]
            identity = h5py.h5o.get_info(obj.id)
            if identity.rc != 1 or identity.addr in addresses:
                raise InputError('Source HDF5 hard-link aliases are unsupported')
            addresses.add(identity.addr)
            return obj
        def numeric(dataset, width=43):
            if (not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external
                    or dataset.ndim != 2 or dataset.shape[1] != width
                    or not 2 <= dataset.shape[0] <= 4096 or dataset.dtype not in (np.dtype('float32'), np.dtype('float64'))
                    or not dataset.dtype.isnative):
                raise InputError('Source numeric member requires bounded native float32/float64 T-by' + str(width))
            props = dataset.id.get_create_plist()
            try:
                if any(props.get_filter(i)[0] not in {1, 2, 3, 4, 5, 6} for i in range(props.get_nfilters())):
                    raise InputError('Source HDF5 nonbuiltin filters are unsupported')
            finally:
                props.close()
            if dataset.chunks and np.prod(dataset.chunks) * dataset.dtype.itemsize > 2 * 1024 ** 2:
                raise InputError('Source HDF5 chunk allocation exceeds bound')
            return dataset
        with h5py.File(stream, 'r') as file:
            data = member(file, 'data')
            if not isinstance(data, h5py.Group) or len(data) > 256:
                raise InputError('Original source requires at most256 direct native data groups')
            headers, omissions, unmodeled = [], [], []
            source_keys = list(data.keys())  # Exact native enumeration, never sorted/reindexed.
            global_start, declared_rows, projected_bytes = 0, 0, 0
            for ordinal, key in enumerate(source_keys):
                if not key or key in {'.', '..'} or '/' in key or len(key) > 256 or any(ord(c) < 32 for c in key):
                    raise InputError('Source native group locator is unsupported')
                group = member(data, key)
                if not isinstance(group, h5py.Group) or len(group) > 128:
                    raise InputError('Source trajectory group exceeds bounded schema')
                keys = set(group.keys())
                if keys == {'initial_state'}:
                    omissions.append({'source_key': key, 'native_episode_index': ordinal, 'reason': 'initial_state_only'})
                    continue
                try:
                    obs = member(group, 'obs')
                    if not isinstance(obs, h5py.Group) or len(obs) > 128:
                        raise InputError('Unsupported source observation group')
                    state = numeric(member(obs, config['state_name_sim']))
                    action = numeric(member(group, config['action_name_sim']))
                    if state.shape != action.shape:
                        raise InputError('Source state/action lengths differ')
                except (InputError, KeyError) as error:
                    unmodeled.append({'source_key': key, 'native_episode_index': ordinal})
                    add('unknown', 'conversion.source_group_unsupported', key + ': ' + str(error))
                    continue
                length = action.shape[0] - 1
                projected_bytes += (state.nbytes + action.nbytes) * 2
                aux_members = {}
                try:
                    commands = member(group, 'action')
                    if not isinstance(commands, h5py.Group) or len(commands) > 128:
                        raise InputError('Unsupported source teleoperation group')
                except (InputError, KeyError) as error:
                    commands = None
                    command_reason = str(error)
                for field, column, width, _ in AUXILIARY_FIELDS:
                    try:
                        if commands is None:
                            raise InputError(command_reason)
                        item = numeric(member(commands, config[field]), width)
                        if item.shape[0] != action.shape[0]:
                            raise InputError('Source auxiliary and joint lengths differ')
                        aux_members[column] = item
                        projected_bytes += item.nbytes * 2
                    except (InputError, KeyError) as error:
                        aux_add(column, 'unknown', 'conversion.auxiliary_source_unsupported',
                                key + ': source ' + column + ' unavailable: ' + str(error))
                headers.append({'key': key, 'id': ordinal, 'length': length, 'state': state, 'action': action,
                                'global_start': global_start, 'auxiliary': aux_members})
                global_start += length
                declared_rows += length
            for field in auxiliary.values():
                field['expected_rows'] = declared_rows
            # Criterion eligibility is about the complete ORIGINAL, never the
            # candidate's success/coverage. Validate source values independently
            # even when an export column/file is missing or has the wrong shape.
            source_basis_complete = bool(headers) and not unmodeled and (
                declared_rows <= plan['max_rows'] and projected_bytes <= MAX_VALUES_BYTES)
            source_basis_reasons = []
            if source_basis_complete:
                for header in headers:
                    if len(header['auxiliary']) != len(AUXILIARY_FIELDS):
                        source_basis_complete = False
                        source_basis_reasons.append('Source auxiliary fields are incomplete')
                    for role in ['state', 'action']:
                        if not np.isfinite(header[role][...]).all():
                            source_basis_complete = False
                            source_basis_reasons.append('Source joint values are nonfinite')
                    for member in header['auxiliary'].values():
                        if not np.isfinite(member[:-1]).all():
                            source_basis_complete = False
                            source_basis_reasons.append('Emitted source auxiliary values are nonfinite')
            else:
                source_basis_reasons.append('Complete supported source selection/value budgets are unestablished')
            wanted_ids = {h['id'] for h in headers}
            unmodeled_ids = {part['native_episode_index'] for part in unmodeled}
            if wanted_ids - set(episode_map) or set(episode_map) - wanted_ids - unmodeled_ids:
                add('error', 'conversion.episode_membership',
                    'Converted episode IDs do not match all supported original numeric trajectory ordinals',
                    missing_ids=sorted(wanted_ids - set(episode_map)), extra_ids=sorted(set(episode_map) - wanted_ids - unmodeled_ids))
            def relative(eid):
                return config['data_path'].format(episode_chunk=eid // config['chunks_size'], episode_index=eid)
            wanted_files = {relative(eid) for eid in wanted_ids}
            uncertain_files = {relative(eid) for eid in unmodeled_ids}
            if wanted_files - data_files or data_files - wanted_files - uncertain_files:
                add('error', 'conversion.file_membership', 'Converted Parquet membership differs from original numeric trajectory ordinals',
                    missing_files=sorted(wanted_files - data_files)[:16], extra_files=sorted(data_files - wanted_files - uncertain_files)[:16])
            if declared_rows > plan['max_rows'] or projected_bytes > MAX_VALUES_BYTES:
                add('unknown', 'conversion.numeric_budget', 'Complete selected source/export numeric rows exceed explicit row/value budgets')
                projections = []
            else:
                import pyarrow as pa
                import pyarrow.parquet as pq

                from .native_parquet_guard import guard_file
                pa.set_cpu_count(1)
                pa.set_io_thread_count(1)
                projections, parquet_bytes = [], 0
                for header in headers:
                    eid, length, key = header['id'], header['length'], header['key']
                    path = str(root / relative(eid))
                    if eid not in episode_map or eid in duplicate_ids or relative(eid) not in data_files:
                        continue
                    position, record = episode_map[eid]
                    if record['length'] != length:
                        add('error', 'conversion.nominal_truncation',
                            key + ': native metadata length would select/truncate a different source extent',
                            native_episode_index=eid, metadata_position=position, expected_rows=length, nominal_rows=record['length'])
                    if type(record.get('tasks')) is not list or config['language_instruction'] not in record['tasks']:
                        add('error', 'conversion.episode_task', key + ': episode task text differs from declared conversion task')
                    try:
                        raw, artifact = _read_small(path, MAX_PARQUET)
                        artifact['kind'] = 'converted_parquet'
                        artifacts.append(artifact)
                        parquet_bytes += len(raw)
                        if parquet_bytes > MAX_PARQUET_TOTAL:
                            raise InputError('Converted Parquet aggregate raw-byte budget exceeded')
                        copied = Path(temporary) / ('episode_' + str(eid) + '.parquet')
                        copied.write_bytes(raw)
                        guard_file(copied)
                        parquet = pq.ParquetFile(io.BytesIO(raw), thrift_string_size_limit=2 * 1024 ** 2,
                                                 thrift_container_size_limit=100000)
                        if parquet.metadata.num_rows > 4096 or len(set(parquet.schema_arrow.names)) != len(parquet.schema_arrow.names):
                            raise InputError('Converted Parquet row/schema bounds exceeded')
                        required = ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', config['task_description_lerobot']]
                        schema = parquet.schema_arrow
                        total_rows = parquet.metadata.num_rows
                        for field, column, width, _ in AUXILIARY_FIELDS:
                            if column not in schema.names:
                                aux_add(column, 'error', 'conversion.auxiliary_missing_export', key + ': readable export lacks ' + column, export_file=path)
                                continue
                            dtype = schema.field(column).type
                            if not (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)) or dtype.value_type not in (pa.float32(), pa.float64()):
                                aux_add(column, 'error', 'conversion.auxiliary_export_dtype', key + ': ' + column + ' changes native floating vector representation', export_file=path)
                                continue
                            source_member = header['auxiliary'].get(column)
                            requested_bytes = total_rows * width * (4 if dtype.value_type == pa.float32() else 8)
                            if source_member is not None:
                                requested_bytes += source_member.nbytes
                            if not reserve_values(requested_bytes):
                                aux_add(column, 'unknown', 'conversion.auxiliary_value_budget', key + ': auxiliary values exceed aggregate64MiB bound')
                                continue
                            values_list = parquet.read(columns=[column])[column].to_pylist()
                            if any(type(row) is not list or len(row) != width for row in values_list):
                                aux_add(column, 'error', 'conversion.auxiliary_export_width', key + ': ' + column + ' must retain width' + str(width), export_file=path)
                                continue
                            if any(value is None for row in values_list for value in row):
                                aux_add(column, 'error', 'conversion.auxiliary_export_null', key + ': ' + column + ' contains a missing value', export_file=path)
                                continue
                            values_array = np.asarray(values_list, dtype='float32' if dtype.value_type == pa.float32() else 'float64').reshape(total_rows, width)
                            if source_member is None:
                                if not np.isfinite(values_array).all():
                                    aux_add(column, 'error', 'conversion.auxiliary_export_nonfinite', key + ': ' + column + ' contains nonfinite exported values', export_file=path)
                                continue
                            auxiliary_pairs.append((header, field, column, width, source_member[...], values_array, path, artifact['sha256']))
                        missing = sorted(set(required) - set(schema.names))
                        if missing:
                            add('error', 'conversion.missing_export_columns', key + ': readable converted schema lacks required columns ' + ', '.join(missing), export_file=path)
                            continue
                        unsupported_columns = False
                        for name in ['observation.state', 'action']:
                            dtype = schema.field(name).type
                            if not (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)):
                                add('error', 'conversion.export_vector_schema', key + ': ' + name + ' must retain the declared numeric vector representation', export_file=path, column=name)
                                unsupported_columns = True
                            elif dtype.value_type not in (pa.float32(), pa.float64()):
                                add('error', 'conversion.dtype_changed', key + ': ' + name + ' changes native floating source dtype to ' + str(dtype.value_type), export_file=path, column=name)
                                unsupported_columns = True
                        if unsupported_columns:
                            continue
                        if schema.field('timestamp').type != pa.float64() or any(schema.field(n).type != pa.int64() for n in required[3:]):
                            add('error', 'conversion.row_time_identity', key + ': converted native time/row identifiers change required float64/int64 types', export_file=path)
                            continue
                        table = parquet.read(columns=required)
                        if any(table[name].null_count for name in required):
                            add('error', 'conversion.null_export', key + ': selected converted columns contain null values')
                            continue
                        total = len(table)
                        if total != length:
                            add('error', 'conversion.export_extent', key + ': actual converted row count differs from native terminal-trimmed source extent')
                        if total == 0:
                            add('unknown', 'conversion.export_empty', key + ': no converted numeric rows are available to compare')
                            continue
                        for name, expected in [('frame_index', np.arange(length, dtype=np.int64)),
                            ('episode_index', np.full(length, eid, dtype=np.int64)),
                            ('index', np.arange(length, dtype=np.int64) + header['global_start']),
                            ('task_index', np.full(length, expected_task, dtype=np.int64)),
                            (config['task_description_lerobot'], np.full(length, expected_task, dtype=np.int64)),
                            ('timestamp', np.arange(length, dtype=np.float64) * (1.0 / config['fps']))]:
                            values = table[name].to_numpy()
                            if name == 'index' and unmodeled:
                                add('unknown', 'conversion.global_index_unsupported', key + ': unsupported earlier source groups prevent complete global-index attribution')
                            elif not np.array_equal(values, expected):
                                add('error', 'conversion.row_time_identity', key + ': converted ' + name + ' differs from declared original association',
                                    native_episode_index=eid, export_file=path, column=name)
                        def values(name, table=table, schema=schema, total=total, key=key, path=path):
                            lists = table[name].to_pylist()
                            for row_index, row in enumerate(lists):
                                if type(row) is not list or len(row) != 43:
                                    add('error', 'conversion.export_vector_width', key + ': converted ' + name + ' row ' + str(row_index) + ' does not retain43 source joint elements', export_file=path, export_row=row_index, column=name)
                                    return None
                                if any(value is None for value in row):
                                    add('error', 'conversion.null_export', key + ': converted ' + name + ' row ' + str(row_index) + ' contains a missing numeric joint value', export_file=path, export_row=row_index, column=name)
                                    return None
                            dtype = 'float32' if schema.field(name).type.value_type == pa.float32() else 'float64'
                            return np.asarray(lists, dtype=dtype).reshape(total, 43)
                        if not reserve_values(sum(header[role].nbytes + total * 43 * (4 if schema.field(name).type.value_type == pa.float32() else 8)
                                for role, name in [('state', 'observation.state'), ('action', 'action')])):
                            add('unknown', 'conversion.numeric_budget', key + ': actual joint projection allocation exceeds aggregate64MiB value bound')
                            continue
                        exported_state, exported_action = values('observation.state'), values('action')
                        if exported_state is None or exported_action is None:
                            continue
                        projections.append(EpisodeProjection(key, eid, header['state'][...], header['action'][...],
                                                             exported_state, exported_action))
                    except Exception as error:
                        add('unknown', 'conversion.export_read_unsupported', key + ': numeric export/member could not be completely read: ' + str(error)[:300], export_file=path)
            if _stamp(os.fstat(fd)) != _stamp(before):
                raise InputError('Original recording changed before numeric adjudication; hashed source values cannot be attributed')
            auxiliary_records = []
            for header, source_field, column, width, source_values, exported_values, export_file, export_sha in auxiliary_pairs:
                key, eid, length = header['key'], header['id'], header['length']
                expected = source_values[:-1]
                source_finite = bool(np.isfinite(expected).all())
                export_finite = bool(np.isfinite(exported_values).all())
                detail = auxiliary[column]
                for kind, matrix in [('source', expected), ('export', exported_values)]:
                    name = str(matrix.dtype)
                    if name not in detail[kind + '_dtypes']:
                        detail[kind + '_dtypes'].append(name)
                auxiliary_records.append({'source_key': key, 'native_episode_index': eid, 'column': column,
                    'width': width, 'source_dtype': str(expected.dtype), 'export_dtype': str(exported_values.dtype),
                    'expected_shape': list(expected.shape), 'observed_shape': list(exported_values.shape),
                    'expected_sha256': hashlib.sha256(expected.tobytes(order='C')).hexdigest(),
                    'observed_sha256': hashlib.sha256(exported_values.tobytes(order='C')).hexdigest()})
                if len(exported_values) != length:
                    aux_add(column, 'error', 'conversion.auxiliary_export_extent', key + ': ' + column + ' changes the terminal-trimmed extent', export_file=export_file)
                if expected.dtype != exported_values.dtype:
                    aux_add(column, 'error', 'conversion.auxiliary_dtype_changed', key + ': ' + column + ' changes source dtype', export_file=export_file)
                if not source_finite:
                    aux_add(column, 'unknown', 'conversion.auxiliary_source_nonfinite', key + ': emitted source ' + column + ' contains nonfinite values')
                if not export_finite:
                    aux_add(column, 'error', 'conversion.auxiliary_export_nonfinite', key + ': ' + column + ' contains nonfinite exported values', export_file=export_file)
                if not source_finite or not export_finite:
                    continue
                compared = min(length, len(exported_values))
                detail['compared_rows'] += compared
                for frame in range(compared):
                    different = np.flatnonzero(expected[frame] != exported_values[frame])
                    if len(different):
                        coordinate = int(different[0])
                        aux_add(column, 'error', 'conversion.auxiliary_numeric_association',
                            f'{key} source frame {frame} {config[source_field]}[{coordinate}]={float(expected[frame, coordinate])} -> {relative(eid)} row {frame} {column}[{coordinate}]={float(exported_values[frame, coordinate])}',
                            source_key=key, native_episode_index=eid, source_frame=frame, export_row=frame, coordinate=coordinate,
                            evidence=[{'path': source_path, 'sha256': original['sha256'], 'format': 'hdf5_dataset_index',
                                'pointer': '/data/' + key + '/action/' + config[source_field] + f'/{frame}/{coordinate}'},
                                {'path': export_file, 'sha256': export_sha, 'format': 'parquet_row_value', 'pointer': f'/rows/{frame}/{column}/{coordinate}'}])
            for field in auxiliary.values():
                field['status'] = ('error' if field['error_count'] else 'unknown' if field['unknown_count'] or unmodeled or not field['expected_rows'] or field['compared_rows'] != field['expected_rows'] else 'pass')
            auxiliary_status = ('error' if any(field['status'] == 'error' for field in auxiliary.values())
                else 'unknown' if any(field['status'] == 'unknown' for field in auxiliary.values()) else 'pass')
            selections = tuple(Selection(h['key'], h['id'], tuple(range(h['length']))) for h in headers)
            if selections:
                closure = tuple(sorted(SOURCE_HASHES.items()))
                # Template hash labels reviewed semantic profile, NOT actual config provenance.
                criterion = NumericCriterion(selections, closure, _digest(selections, closure))
                core = evaluate_numeric(criterion, projections)
                for status in counts:
                    counts[status] += core['counts'][status]
                by_path = {a['path']: a for a in artifacts}
                for found in core['findings']:
                    found = dict(found)
                    eid, key = found['native_episode_index'], found['source_key']
                    if found['code'] == 'conversion.numeric_joint_association':
                        frame, source_column, column = found['source_frame'], found['source_column'], found['export_column']
                        joint = JOINT_NAMES[source_column]
                        member_path = '/data/' + key + ('/obs/' + config['state_name_sim'] if found['role'] == 'state' else '/' + config['action_name_sim'])
                        exported = str(root / relative(eid))
                        found['message'] = (f'{key} source frame {frame} joint {joint} column {source_column}={found["expected"]} '
                                            f'-> {relative(eid)} row {frame} {found["role"]}[{column}]={found["observed"]}')
                        found['evidence'] = [
                            {'path': source_path, 'sha256': original['sha256'], 'format': 'hdf5_dataset_index',
                             'pointer': member_path + f'/{frame}/{source_column}'},
                            {'path': exported, 'sha256': by_path[exported]['sha256'], 'format': 'parquet_row_value',
                             'pointer': f'/rows/{frame}/' + ('action' if found['role'] == 'action' else 'observation.state') + f'/{column}'}]
                    else:
                        found['message'] = key + ': ' + found['code'].removeprefix('conversion.').replace('_', ' ')
                    found['remedy'] = 'Inspect the cited original member and converted row; preserve the original and rerun on a separate export.'
                    _keep_finding(findings, found)
                compared = core['coverage']['compared_selected_rows']
                numeric = 'error' if numeric_contract_errors else core['numeric_association_status']
            else:
                core, compared, numeric = None, 0, 'error' if numeric_contract_errors else 'unknown'
                add('unknown', 'conversion.no_supported_numeric_groups', 'No supported original numeric trajectories were available')
            if _stamp(os.fstat(fd)) != _stamp(before):
                raise InputError('Original recording changed during HDF5 member extraction')
        # Recheck original path identity and every metadata/source/export readset stamp.
        fd2, after = _regular(source_path)
        os.close(fd2)
        if _stamp(after) != _stamp(before):
            raise InputError('Original selected path changed after member extraction')
        for artifact in artifacts:
            descriptor, observed = _regular(artifact['path'])
            os.close(descriptor)
            if _stamp(observed) != artifact['stamp']:
                raise InputError('Conversion readset changed before completion')
        if _data_files(root, plan['max_files']) != data_files:
            add('unknown', 'conversion.export_membership_changed',
                'Converted numeric file membership changed while comparing; complete export coverage is unestablished')
        dependencies = {}
        for name in ['numpy', 'h5py', 'pyarrow', 'PyYAML']:
            dependencies[name] = importlib.metadata.version(name)
        evaluator = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ['conversion_numeric.py', 'conversion_fidelity.py', 'conversion_fidelity_worker.py', 'conversion_profile_data.py', 'native_intake.py', 'native_parquet_guard.py', 'artifacts.py',
                                  'preflight.py', 'cli.py', 'server.py', '__init__.py']}
        criterion_payload = {'schema': 'arena43_with_teleop_numeric_v2', 'original_sha256': original['sha256'],
            'original_size': original['size_bytes'], 'actual_converter_inputs': [
                {key: a[key] for key in ['path', 'sha256', 'size_bytes']} for a in plan['artifacts']],
            'actual_config': config, 'resolved_references': plan['references'],
            'source_selection': [(h['key'], h['id'], h['length']) for h in headers],
            'selection_rule': 'all_supported_original_numeric_rows_except_terminal_v2',
            'required_auxiliary_fields': AUXILIARY_FIELDS, 'scalar_policy': {'instruction_max_UTF8_bytes':4096, 'task_index_min':0, 'task_index_max':2**63-1, 'integer_fps_min':1, 'integer_fps_max':1000},
            'numeric_semantic_profile': dict(sorted(SOURCE_HASHES.items())),
            'semantic_template_hash_scope': 'Reviewed profile identity only; not a read actual config file',
            'evaluator': evaluator, 'dependencies': dependencies, 'python': sys.version.split()[0]}
        complete = (not counts['error'] and not counts['unknown'] and not unmodeled
                    and bool(headers) and compared == declared_rows and auxiliary_status == 'pass')
        status = 'error' if counts['error'] else 'unknown' if not complete else 'pass'
        criterion_sha = hashlib.sha256(json.dumps(criterion_payload, sort_keys=True).encode()).hexdigest()
        reuse_inputs = {key: plan['paths'][key] for key in ['original_recording', 'converter_source', 'converter_config']}
        reuse_inputs['converter_directory'] = plan['converter_directory']
        return {'schema_version': 1, 'kind': 'conversion_fidelity', 'status': status,
                'requested_scope': 'state_action_and_teleop_numeric_conversion',
                'qualified_profile': 'Pinned Arena G1 43-joint remapping plus three configured teleop fields; actual bounded instruction/taskID/integerfps declaration',
                'joint_numeric_association_status': numeric, 'auxiliary_numeric_association_status': auxiliary_status,
                'numeric_association_status': 'error' if 'error' in (numeric, auxiliary_status) else 'unknown' if 'unknown' in (numeric, auxiliary_status) else 'pass',
                'source_membership': {'status': 'error' if membership_errors else 'pass' if bool(headers) and not unmodeled and len(projections) == len(headers) and compared == declared_rows and all(field['compared_rows'] == declared_rows for field in auxiliary.values()) else 'unknown',
                    'scope': 'Complete original hash/same FD and selected native numeric member identities; local snapshot only'},
                'native_consumption': {'status': 'unknown', 'requested': False, 'reason': 'Native GR00T consumer was not executed'},
                'pixel_fidelity': {'status': 'unknown', 'reason': 'No source or decoded image arrays were read'},
                'criterion_sha256': criterion_sha,
                'conversion_reuse': {'schema': 'conversion_preflight_reuse_v1',
                    'criterion_schema': criterion_payload['schema'], 'criterion_sha256': criterion_sha,
                    'criterion_basis_complete': source_basis_complete, 'inputs': reuse_inputs,
                    'requested_scope': 'state_action_and_teleop_numeric_conversion'},
                'criterion_basis_incomplete_reasons': sorted(set(source_basis_reasons)),
                'criterion': criterion_payload, 'auxiliary_fields': auxiliary, 'auxiliary_episode_records': auxiliary_records, 'findings': findings, 'counts': counts,
                'findings_total': sum(counts.values()), 'counting_scope': 'Findings and field-row mismatch observations; a global row can contribute multiple findings', 'findings_truncated': sum(counts.values()) > len(findings),
                'artifacts': artifacts, 'coverage': {'whole_numeric_source_to_export': complete,
                    'whole_dataset_verified': False, 'original_source_groups': len(source_keys),
                    'numeric_eligible_groups': len(headers), 'initial_state_only_omissions': omissions,
                    'unsupported_source_groups': unmodeled, 'metadata_declared_episodes': info.get('total_episodes'),
                    'metadata_episode_records': len(episodes), 'expected_numeric_rows': declared_rows,
                    'compared_numeric_rows': compared, 'compared_auxiliary_rows': {column: field['compared_rows'] for column, field in auxiliary.items()},
                    'requested_base_height_navigation_dimensions':4, 'requested_additional_torso_dimensions':3,
                    'auxiliary_coordinate_count_scope': 'Declared requested widths; actual verified coverage is per-field',
                    'covered_auxiliary_command_dimensions': sum(field['width'] for field in auxiliary.values() if field['status'] == 'pass' and field['profile_role'] == 'native_action_input'),
                    'native_profile_scope': 'Static role annotation for reviewed N1.7 G1 profile only; no actual consumer/profile execution or model output shape claim',
                    'native_episode_ids': sorted(wanted_ids),
                    'original_hash_bytes': original['size_bytes'], 'max_original_hash_bytes': MAX_ORIGINAL,
                    'max_rows': plan['max_rows'], 'max_files': plan['max_files']},
                'scope': 'Actual caller-selected local source-to-export numeric membership/value comparison only. No actual converter execution, historical producing provenance, native consumer, pixels, normalization, model or physical claims.'}
    finally:
        stream.close()


def main(output):
    def audit(event, args):
        if (event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}
                or event.startswith(('os.exec', 'os.posix_spawn', 'os.spawn'))):
            _AUDIT_DENIALS[event] = _AUDIT_DENIALS.get(event, 0) + 1
            # OSError permits stdlib platform.processor's normal unavailable-name
            # fallback; the denied uname process is never executed.
            raise PermissionError('Conversion worker forbids network and external execution')
    sys.addaudithook(audit)
    request = sys.stdin.buffer.read(512 * 1024 + 1)
    try:
        if (not sys.flags.isolated or not sys.dont_write_bytecode or os.environ.get('PYTHONPATH')
                or os.environ.get('HDF5_PLUGIN_PATH') != '' or os.environ.get('HDF5_PLUGIN_PRELOAD') != '::'
                or Path.cwd() != Path(output).parent):
            raise InputError('Conversion worker requires isolated product launch/plugin/cwd boundary')
        if len(request) > 512 * 1024:
            raise InputError('Conversion request exceeds bound')
        plan = _json(request)
        result = read_conversion(plan, Path(output).parent)
    except Exception as error:
        result = _unknown('Bounded source/export numeric reader refused: ' + str(error)[:1000])
        if _PARTIAL is not None:
            partial = _PARTIAL
            result['artifacts'] = partial['artifacts']
            result['findings'] = (partial['findings'] + result['findings'])[:64]
            result['counts'] = dict(partial['counts'])
            result['counts']['unknown'] += 1
            result['findings_total'] = sum(result['counts'].values())
            result['findings_truncated'] = result['findings_total'] > len(result['findings'])
            if result['counts']['error']:
                result['status'] = 'error'
    result['worker'] = {'isolated': bool(sys.flags.isolated), 'bytecode_disabled': sys.dont_write_bytecode,
                        'pythonpath_empty': not os.environ.get('PYTHONPATH'),
                        'cwd': str(Path.cwd()), 'hdf5_plugins_disabled': os.environ.get('HDF5_PLUGIN_PRELOAD') == '::',
                        'torch_imported': 'torch' in sys.modules, 'audit_denials': dict(_AUDIT_DENIALS),
                        'audit_scope': 'Python audit events, not an operating-system sandbox'}
    raw = json.dumps(result, allow_nan=False).encode()
    if len(raw) > MAX_REPORT:
        shortened = _unknown('Conversion result exceeds complete-report byte bound')
        if result.get('counts', {}).get('error', 0):
            shortened['status'] = 'error'
        shortened['counts'] = result.get('counts', {'error': 0, 'unknown': 1})
        shortened['findings_total'] = sum(shortened['counts'].values())
        shortened['findings_truncated'] = True
        raw = json.dumps(shortened).encode()
    Path(output).write_bytes(raw)
