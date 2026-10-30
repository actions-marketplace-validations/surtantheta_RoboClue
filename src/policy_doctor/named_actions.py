"""Bounded offline named-action boundary checks; no native or driver execution."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import PurePosixPath

from .artifacts import InputError
from .layouts import _read, _safe_root, _tolerances
from .named_rules import (
    MAX_BYTES,
    MAX_DIM,
    MAX_MANIFEST,
    MAX_TICKS,
    RECORDING_SOURCE_HASHES,
    REPOSITORY,
    REVISION,
    SOURCE_HASHES,
)
from .selection import _digest, _parse

RELATIONS = ('interp_to_canonical', 'canonical_to_processor_input', 'processor_return_to_wrapper_input',
             'wrapper_input_to_driver_input', 'driver_return_to_wrapper_return', 'canonical_return_is_processor_input')
STAGES = ('canonical_at_processor', 'canonical_at_return', 'robot_processor_output',
          'wrapper_request', 'driver_request', 'driver_return', 'wrapper_return')
SCOPE = ('Observed pinned native named-action software boundary only. Canonical input, processor output, '
         'driver request/return and native function return are separate snapshots. Legal limiting and '
         'in-place changes are observations; only explicit consumer criteria make them incompatible. '
         'Earlier policy selection/processing, authenticated runtime origin, physical motion '
         'robot processor transform semantics and calibrated joint semantics remain unknown. '
         'Numeric replay covers pinned interpolation-to-name mapping and unchanged boundary snapshots only; '
         'it does not replay arbitrary robot processors or driver decorators. Native recording is qualified separately '
         'only when observed same-writer lifecycle and committed native rows agree. Declared order is not physical mapping proof.')


def _text(value):
    return type(value) is str and 1 <= len(value.encode('utf8')) <= 128 and value.isprintable() and value == value.strip()


def _names(value):
    if (type(value) is not list or not 1 <= len(value) <= MAX_DIM
            or any(not _text(v) for v in value) or len(set(value)) != len(value)):
        raise InputError('Named actions require distinct bounded declared action names')
    return list(value)


DATASET_FILES = ('data/chunk-000/file-000.parquet', 'meta/episodes/chunk-000/file-000.parquet',
                 'meta/info.json', 'meta/stats.json', 'meta/tasks.parquet')
RECORDING_RELATIONS = ('canonical_to_build_frame', 'build_to_facade', 'facade_to_writer', 'writer_to_buffer')


def _expected(recording=False, dataset=False):
    sources = {**SOURCE_HASHES, **(RECORDING_SOURCE_HASHES if recording else {})}
    return ({'capture.json', 'READY.json'} | {'sources/' + name for name in (*sources, 'driver_source.py')}
            | ({'dataset/' + name for name in DATASET_FILES} if dataset else set()))


def _tree(root, recording=False, dataset=False):
    expected, found, queue, entries = _expected(recording, dataset), set(), [root], 0
    directories = {str(PurePosixPath(path).parent) for path in expected}
    directories |= {str(parent) for path in expected for parent in PurePosixPath(path).parents}
    while queue:
        for path in queue.pop().iterdir():
            entries += 1
            relative = str(path.relative_to(root))
            if entries > 64 or path.is_symlink():
                raise InputError('Named capture symlink/entry budget refused')
            if path.is_dir() and relative in directories:
                queue.append(path)
            elif path.is_file() and relative in expected:
                found.add(relative)
            else:
                raise InputError('Unexpected named capture payload')
    if found != expected:
        raise InputError('Named capture requires the complete native source/dataset inventory')
    return found


def _load(capture):
    root = _safe_root(capture)
    raw, ready_raw = _read(root / 'capture.json', MAX_MANIFEST), _read(root / 'READY.json', 4096)
    data, ready = _parse(raw), _parse(ready_raw)
    if (set(ready) != {'schema_version', 'manifest_sha256'} or type(ready['schema_version']) is not int
            or ready['schema_version'] != 1 or ready['manifest_sha256'] != _digest(raw)):
        raise InputError('Named capture commit marker/hash mismatch')
    if (type(data.get('schema_version')) is not int or data['schema_version'] not in (1, 2)
            or data.get('kind') != 'named_action_capture' or data.get('domain') != 'native_named_action_software'
            or type(data.get('ticks')) is not list or not 1 <= len(data['ticks']) <= MAX_TICKS):
        raise InputError('Unsupported named-action schema/domain/tick count')
    recording = data['schema_version'] == 2
    if recording and (type(data.get('recording')) is not dict or type(data['recording'].get('dataset_files')) is not list):
        raise InputError('Schema2 requires explicit recording/dataset inventory')
    dataset = recording and bool(data['recording']['dataset_files'])
    _tree(root, recording, dataset)
    _names(data.get('action_names'))
    _names(data.get('engine_action_names'))
    if not _text(data.get('driver_namespace')):
        raise InputError('Named driver namespace must be explicit bounded text')
    sources = {**SOURCE_HASHES, **(RECORDING_SOURCE_HASHES if recording else {})}
    expected = _expected(recording, dataset) - {'capture.json', 'READY.json'}
    entries = data.get('files')
    if type(entries) is not list or len(entries) != len(expected):
        raise InputError('Named payload inventory is incomplete')
    hashes = {'capture.json': _digest(raw), 'READY.json': _digest(ready_raw)}
    total = len(raw) + len(ready_raw)
    for entry in entries:
        if (type(entry) is not dict or set(entry) != {'path', 'sha256'} or type(entry['path']) is not str
                or entry['path'] not in expected or entry['path'] in hashes or type(entry['sha256']) is not str
                or re.fullmatch('[a-f0-9]{64}', entry['sha256']) is None):
            raise InputError('Invalid named payload locator/hash')
        payload = _read(root / entry['path'], 512 * 1024 if entry['path'].startswith('sources/') else 8 * 1024 * 1024)
        total += len(payload)
        if total > MAX_BYTES or _digest(payload) != entry['sha256']:
            raise InputError('Named payload hash mismatch or byte budget exceeded')
        hashes[entry['path']] = _digest(payload)
    source = data.get('source')
    if (type(source) is not dict or source.get('files') !=
            {name: {'sha256': hashes['sources/' + name]} for name in (*sources, 'driver_source.py')}):
        raise InputError('Named source identity and native byte inventory disagree')
    supported = (source.get('repository') == REPOSITORY and source.get('revision') == REVISION
                 and all(hashes['sources/' + name] == digest for name, digest in sources.items()))
    if recording:
        rec = data.get('recording')
        expected_dataset = [{'path': 'dataset/' + path, 'sha256': hashes['dataset/' + path]} for path in DATASET_FILES] if dataset else []
        if (type(rec) is not dict or type(rec.get('dataset_files')) is not list
                or len(rec['dataset_files']) != len(expected_dataset)
                or any(type(item) is not dict or set(item) != {'path', 'sha256'} for item in rec['dataset_files'])
                or sorted(rec['dataset_files'], key=lambda item: str(item.get('path'))) != sorted(expected_dataset, key=lambda item: item['path'])):
            raise InputError('Recording dataset inventory must match original committed native files')
    return root, data, hashes, supported


def _recheck(root, hashes):
    if _tree(root, 'sources/episodic.py' in hashes, 'dataset/meta/info.json' in hashes) != set(hashes):
        raise InputError('Named evidence inventory changed during audit')
    for name, digest in hashes.items():
        if _digest(_read(root / name, MAX_BYTES)) != digest:
            raise InputError('Named evidence changed during audit')


def _number(value):
    return (type(value) in (int, float) and (type(value) is not int or abs(value) <= 2**53)
            and math.isfinite(value))


def _action(value):
    if value is None:
        return None
    if (type(value) is not dict or not 1 <= len(value) <= MAX_DIM
            or any(not _text(k) or not _number(v) for k, v in value.items())):
        raise InputError('Named action snapshots require bounded finite real scalar channels, excluding Boolean')
    return value


def _equal(left, right, atol, rtol):
    if left is None or right is None or set(left) != set(right):
        return None
    return all(math.isclose(left[k], right[k], abs_tol=atol, rel_tol=rtol) for k in left)


def _analyze(data, supported, atol, rtol):
    coverage = data.get('coverage')
    covered = (type(coverage) is dict and coverage.get('target') == 'one_native_send_next_action_on_owner_thread'
               and coverage.get('complete_target_call_history') is True and coverage.get('external_state_mutations') == 'untracked')
    events = []
    for ordinal, tick in enumerate(data['ticks']):
        if (type(tick) is not dict or type(tick.get('tick_id')) is not int or tick['tick_id'] != ordinal
                or tick.get('completion') not in ('normal', 'error') or tick.get('outcome') not in ('sent', 'not_sent', 'error')
                or any(type(tick.get(key)) is not bool for key in ('driver_called', 'driver_completed', 'wrapper_completed', 'emitted_policy_action'))
                or type(tick.get('ack_count')) is not int or tick['ack_count'] not in (0, 1)):
            raise InputError('Malformed named-action tick identity/completion/acknowledgement')
        if tick['outcome'] == 'sent' and (tick['completion'] != 'normal' or tick['ack_count'] != 1
                or not all(tick[key] for key in ('driver_called', 'driver_completed', 'wrapper_completed'))):
            raise InputError('Named sent tick requires normal driver/wrapper completion and native acknowledgement')
        if tick['outcome'] != 'sent' and tick['ack_count'] != 0:
            raise InputError('Unsent named tick cannot claim native acknowledgement')
        if tick['driver_completed'] and not tick['driver_called']:
            raise InputError('A driver return requires an observed call')
        if tick['wrapper_completed'] and not tick['driver_completed']:
            raise InputError('A wrapper return requires normal driver completion')
        roles = {key: _action(tick.get(key)) for key in STAGES}
        interpolated = tick.get('interpolated')
        if interpolated is not None and (type(interpolated) is not list or len(interpolated) != len(data['action_names'])
                                       or any(not _number(v) for v in interpolated)):
            raise InputError('Interpolated values must match the bounded actual declared action order')
        relations = tick.get('relations')
        if (type(relations) is not dict or set(relations) != set(RELATIONS)
                or any(v is not None and type(v) is not bool for v in relations.values())):
            raise InputError('Named live relations require exact nullable Boolean fields')
        checks = []
        mapping = None
        if interpolated is not None and roles['canonical_at_processor'] is not None and relations['interp_to_canonical'] is True:
            mapping = _equal(dict(zip(data['action_names'], interpolated, strict=True)), roles['canonical_at_processor'], atol, rtol)
            checks.append(mapping)
        for relation, left, right in (
                ('processor_return_to_wrapper_input', 'robot_processor_output', 'wrapper_request'),
                ('wrapper_input_to_driver_input', 'wrapper_request', 'driver_request'),
                ('driver_return_to_wrapper_return', 'driver_return', 'wrapper_return')):
            if relations[relation] is True:
                checks.append(_equal(roles[left], roles[right], atol, rtol))
        numeric = ('unknown' if not supported else 'error' if mapping is False else
                   'pass' if checks and all(v is True for v in checks) else 'unknown')
        complete = (supported and covered and tick['outcome'] == 'sent' and tick['ack_count'] == 1
                    and all(v is True for v in relations.values()) and interpolated is not None
                    and all(value is not None for value in roles.values()) and numeric == 'pass')
        driver_equal = _equal(roles['driver_request'], roles['driver_return'], atol, rtol)
        canonical_equal = _equal(roles['canonical_at_processor'], roles['driver_return'], atol, rtol)
        mutation = (roles['canonical_at_processor'] is not None and roles['canonical_at_return'] is not None
                    and (set(roles['canonical_at_processor']) != set(roles['canonical_at_return'])
                         or _equal(roles['canonical_at_processor'], roles['canonical_at_return'], atol, rtol) is False))
        events.append({'tick_id': ordinal, 'completion': tick['completion'], 'outcome': tick['outcome'],
                       'binding_status': 'pass' if complete else 'unknown', 'numeric_replay_status': numeric,
                       'driver_input_matches_return': driver_equal, 'canonical_matches_driver_return': canonical_equal,
                       'canonical_changed_during_call': mutation, 'driver_completed': tick['driver_completed'],
                       'emitted_policy_action': tick['emitted_policy_action']})
    return events, covered


class _RecordingUnknown(Exception):
    pass


def _strict_index(value):
    return type(value) is int and 0 <= value <= MAX_TICKS


def _vector(value, width):
    if type(value) is not list or len(value) != width or any(not _number(v) for v in value):
        raise InputError('Recording action arrays require bounded finite numeric values')
    return value


def _parquet_recording(root, relative, columns, hashes, *, rows, width=None):
    """Validate physical and expanded bounds before decoding selected columns."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = _read(root / relative, 8 * 1024 * 1024)
    if _digest(raw) != hashes[relative]:
        raise InputError('Native Parquet bytes changed before parsing')
    if (len(raw) < 12 or raw[:4] != b'PAR1' or raw[-4:] != b'PAR1'
            or not 0 < int.from_bytes(raw[-8:-4], 'little') <= 1024 * 1024):
        raise _RecordingUnknown('Unsupported/bounded native Parquet footer')
    from .native_parquet_guard import UnsupportedParquetMetadata, require_unencrypted_parquet

    try:
        require_unencrypted_parquet(io.BytesIO(raw), max_footer=1024 * 1024)
    except UnsupportedParquetMetadata as exc:
        raise _RecordingUnknown(str(exc)) from exc
    pf = pq.ParquetFile(io.BytesIO(raw))
    meta = pf.metadata
    if (meta.num_rows != rows or meta.num_columns > 128 or meta.num_row_groups > MAX_TICKS
            or any(column not in pf.schema_arrow.names for column in columns)):
        raise _RecordingUnknown('Native Parquet rows/columns do not establish complete recording coverage')
    expanded = cells = 0
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        for j in range(group.num_columns):
            column = group.column(j)
            if column.total_uncompressed_size < 0 or column.num_values < 0:
                raise _RecordingUnknown('Invalid native Parquet expanded byte/cell metadata')
            expanded += column.total_uncompressed_size
            cells += column.num_values
    if expanded > 8 * 1024 * 1024 or cells > 1000000:
        raise _RecordingUnknown('Native Parquet expanded byte/cell budget exceeded')
    for name in columns:
        dtype = pf.schema_arrow.field(name).type
        if name == 'action':
            valid = pa.types.is_float32(dtype) if width == 1 else (
                (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)) and pa.types.is_float32(dtype.value_type))
            if not valid:
                raise _RecordingUnknown('Native recorded action must use declared float32 scalar/vector storage')
        elif not pa.types.is_integer(dtype):
            raise _RecordingUnknown('Native recorded row identities must have integer Arrow types')
    result = []
    for batch in pf.iter_batches(batch_size=64, columns=list(columns)):
        result.extend(batch.to_pylist())
        if len(result) > MAX_TICKS:
            raise _RecordingUnknown('Native recording row budget exceeded')
    return result


def _recording(root, data, hashes, supported, events, atol, rtol):
    result = {'status': 'unknown', 'buffer_binding_status': 'unknown', 'persistence_status': 'unknown',
              'rows_checked': 0, 'driver_difference_events': 0, 'rows': [], 'first_difference': None, 'driver_comparison_complete': False,
              'reason': 'No observed native recording lifecycle',
              'scope': 'Observed fresh numeric native writer rows only. Recording canonical values does not prove applied or physical action; equality to driver return is an optional caller criterion.'}
    if data['schema_version'] != 2:
        return result
    rec = data['recording']
    rows = rec.get('rows')
    lifecycle = rec.get('lifecycle')
    coverage = rec.get('coverage')
    if (type(rows) is not list or len(rows) > MAX_TICKS or type(lifecycle) is not dict or type(coverage) is not dict):
        raise InputError('Malformed recording rows/lifecycle/coverage')
    lifecycle_fields = ('loop_completed', 'facade_save_completed', 'writer_save_completed',
                        'facade_finalize_completed', 'writer_finalize_completed')
    if (any(type(lifecycle.get(name)) is not bool for name in lifecycle_fields)
            or any(type(lifecycle.get(name)) is not int or not 0 <= lifecycle[name] <= MAX_TICKS
                   for name in ('save_episode_count', 'finalize_count'))
            or ('save_buffer_continuity' in lifecycle and type(lifecycle['save_buffer_continuity']) is not bool)):
        raise InputError('Recording lifecycle requires exact Boolean/completion counters')
    local = supported and coverage.get('target') == 'same_native_episodic_writer' and coverage.get('complete_target_call_history') is True
    joined, tick_ids = [], set()
    for ordinal, row in enumerate(rows):
        if (type(row) is not dict or any(not _strict_index(row.get(key)) for key in
                ('tick_id', 'episode_index', 'frame_index', 'index', 'buffer_size_before', 'buffer_size_after'))
                or any(type(row.get(key)) is not bool for key in ('facade_completed', 'writer_completed'))):
            raise InputError('Recording rows require exact bounded identity/completion fields')
        names = _names(row.get('action_names'))
        canonical = _action(row.get('canonical_at_frame'))
        vectors = {key: _vector(row.get(key), len(names)) for key in ('frame_action', 'action', 'buffer_action')}
        relations = row.get('relations')
        if (type(relations) is not dict or set(relations) != set(RECORDING_RELATIONS)
                or any(type(value) is not bool for value in relations.values())):
            raise InputError('Recording relations require exact Boolean fields')
        tick = data['ticks'][row['tick_id']] if row['tick_id'] < len(data['ticks']) else None
        valid = (row['tick_id'] not in tick_ids and row['episode_index'] == 0 and row['frame_index'] == row['index'] == ordinal
                 and row['buffer_size_before'] == ordinal and row['buffer_size_after'] == ordinal + 1
                 and row.get('source_role') == 'canonical_return' and row['facade_completed'] and row['writer_completed']
                 and all(relations.values()) and canonical is not None and tick is not None)
        tick_ids.add(row['tick_id'])
        if tick is not None:
            valid = (valid and tick['outcome'] == 'sent' and tick['emitted_policy_action']
                     and events[row['tick_id']]['binding_status'] == 'pass'
                     and _equal(canonical, tick.get('canonical_at_return'), atol, rtol) is True)
        expected = dict(zip(names, vectors['frame_action'], strict=True))
        valid = (valid and _equal(canonical, expected, atol, rtol) is True
                 and all(all(math.isclose(x, y, abs_tol=atol, rel_tol=rtol) for x, y in zip(vectors['frame_action'], values, strict=True))
                         for values in vectors.values()))
        local = local and valid
        joined.append((row, names, vectors))
    expected_tick_ids = {tick['tick_id'] for tick in data['ticks'] if tick['outcome'] == 'sent' and tick['emitted_policy_action']}
    local = local and bool(rows) and tick_ids == expected_tick_ids
    result['buffer_binding_status'] = 'pass' if local else 'unknown'
    if not local:
        result['reason'] = 'Observed buffer/canonical/tick identity joins are incomplete or inconsistent'
        return result
    if (not all(lifecycle[name] for name in lifecycle_fields) or lifecycle.get('save_buffer_continuity') is not True
            or lifecycle['save_episode_count'] != 1 or lifecycle['finalize_count'] != 1 or not rec['dataset_files']):
        result['reason'] = 'Native same-buffer save/finalize lifecycle or dataset copy is incomplete'
        return result
    try:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise _RecordingUnknown('Install policy-deployment-doctor[dataset] to verify persisted native Parquet rows') from None
        parsed = {}
        for relative in ('dataset/meta/info.json', 'dataset/meta/stats.json'):
            raw = _read(root / relative, MAX_MANIFEST)
            if _digest(raw) != hashes[relative]:
                raise InputError('Native recording JSON changed before parsing')
            parsed[relative] = _parse(raw)
        info = parsed['dataset/meta/info.json']
        feature = info.get('features', {}).get('action') if type(info.get('features')) is dict else None
        if (not str(info.get('codebase_version', '')).startswith('v3.') or info.get('storage_format', 'parquet') != 'parquet'
                or info.get('data_path') != 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
                or type(info.get('total_frames')) is not int or info['total_frames'] != len(rows)
                or type(info.get('total_episodes')) is not int or info['total_episodes'] != 1
                or type(info.get('total_tasks')) is not int or info['total_tasks'] != 1
                or type(feature) is not dict or feature.get('dtype') != 'float32'
                or any(type(ft) is not dict or ft.get('dtype') in ('image', 'video') for ft in info.get('features', {}).values())):
            raise _RecordingUnknown('Unsupported/incomplete native single-episode numeric metadata')
        names = _names(feature.get('names'))
        if feature.get('shape') != [len(names)] or any(type(v) is not int for v in feature['shape']):
            raise _RecordingUnknown('Native action feature width/names are inconsistent')
        native_rows = _parquet_recording(root, 'dataset/' + DATASET_FILES[0],
            ('index', 'episode_index', 'frame_index', 'action'), hashes, rows=len(rows), width=len(names))
        metadata = _parquet_recording(root, 'dataset/' + DATASET_FILES[1],
            ('episode_index', 'dataset_from_index', 'dataset_to_index', 'length', 'data/chunk_index', 'data/file_index'), hashes, rows=1)
        if metadata != [{'episode_index': 0, 'dataset_from_index': 0, 'dataset_to_index': len(rows),
                         'length': len(rows), 'data/chunk_index': 0, 'data/file_index': 0}]:
            raise _RecordingUnknown('Native episode metadata does not cover the fresh complete row span')
        _parquet_recording(root, 'dataset/meta/tasks.parquet', (), hashes, rows=1)
        comparable = True
        for ordinal, (observed, (row, captured_names, vectors)) in enumerate(zip(native_rows, joined, strict=True)):
            values = [observed['action']] if len(names) == 1 else observed['action']
            _vector(values, len(names))
            if (any(type(observed[key]) is not int for key in ('index', 'episode_index', 'frame_index'))
                    or observed['index'] != ordinal or observed['episode_index'] != 0 or observed['frame_index'] != ordinal
                    or names != captured_names or not all(math.isclose(x, y, abs_tol=atol, rel_tol=rtol)
                        for x, y in zip(values, vectors['buffer_action'], strict=True))):
                raise _RecordingUnknown('Persisted physical rows do not match observed writer buffer identities/values')
            recorded = dict(zip(names, values, strict=True))
            driver = data['ticks'][row['tick_id']]['driver_return']
            equal = _equal(recorded, driver, atol, rtol)
            comparable = comparable and equal is not None
            evidence = [{'path': str(root / 'dataset' / DATASET_FILES[0]), 'sha256': hashes['dataset/' + DATASET_FILES[0]],
                         'row': ordinal, 'column': 'action', 'row_identity': 'zero-based physical Parquet row'},
                        {'path': str(root / 'capture.json'), 'sha256': hashes['capture.json'], 'pointer': f"/ticks/{row['tick_id']}/driver_return"}]
            item = {'tick_id': row['tick_id'], 'episode_index': 0, 'frame_index': ordinal, 'index': ordinal,
                    'recorded_matches_driver_return': equal, 'evidence': evidence}
            result['rows_checked'] += 1
            if equal is False:
                result['driver_difference_events'] += 1
                if result['first_difference'] is None:
                    result['first_difference'] = {**item, 'recorded': recorded, 'driver_return': driver}
            if len(result['rows']) < 12:
                result['rows'].append(item)
        result.update(status='pass', persistence_status='pass', driver_comparison_complete=comparable, reason=None)
    except _RecordingUnknown as exc:
        result['reason'] = str(exc)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise InputError(f'Malformed native recording dataset: {exc}') from exc
    return result


def audit_named_actions(capture, *, expected_action_names=None, expected_driver_namespace=None,
                        require_canonical_matches_driver_return=False, require_recorded_matches_driver_return=False, atol=1e-6, rtol=1e-5):
    """Audit native named boundaries; legal limiting is not a fault without a criterion."""
    if type(require_recorded_matches_driver_return) is not bool:
        raise InputError('require_recorded_matches_driver_return must be a strict Boolean')
    if type(require_canonical_matches_driver_return) is not bool:
        raise InputError('require_canonical_matches_driver_return must be a strict Boolean')
    if expected_action_names is not None:
        expected_action_names = _names(expected_action_names)
    if expected_driver_namespace is not None and not _text(expected_driver_namespace):
        raise InputError('Expected driver namespace must be bounded explicit text')
    _tolerances(atol, rtol)
    root, data, hashes, supported = _load(capture)
    events, covered = _analyze(data, supported, atol, rtol)
    recording = _recording(root, data, hashes, supported, events, atol, rtol)
    numeric = ('error' if any(e['numeric_replay_status'] == 'error' for e in events) else
               'pass' if all(e['numeric_replay_status'] == 'pass' for e in events) else 'unknown')
    binding = 'pass' if all(e['binding_status'] == 'pass' for e in events) else 'unknown'
    qualified = supported and covered and numeric == 'pass' and binding == 'pass'
    def evidence(pointer):
        return {'path': str(root / 'capture.json'), 'sha256': hashes['capture.json'], 'pointer': pointer}
    first_driver_change = None
    for event, tick in zip(events, data['ticks'], strict=True):
        if event['driver_input_matches_return'] is False:
            changes = [{'name': key, 'driver_request': tick['driver_request'][key], 'driver_return': tick['driver_return'][key]}
                       for key in tick['driver_request']
                       if not math.isclose(tick['driver_request'][key], tick['driver_return'][key], abs_tol=atol, rel_tol=rtol)]
            first_driver_change = {'tick_id': event['tick_id'], 'changed_channel_count': len(changes), 'channels': changes[:8],
                'channels_truncated': len(changes) > 8, 'evidence': [evidence(f"/ticks/{event['tick_id']}/driver_request"),
                                                                    evidence(f"/ticks/{event['tick_id']}/driver_return")]}
            break
    violations = []
    if qualified:
        if expected_action_names is not None:
            for field in ('action_names', 'engine_action_names'):
                if data[field] != expected_action_names:
                    violations.append({'field': field, 'observed': data[field], 'expected': expected_action_names,
                                       'evidence': [evidence('/' + field)]})
        if expected_driver_namespace is not None and data['driver_namespace'] != expected_driver_namespace:
            violations.append({'field': 'driver_namespace', 'observed': data['driver_namespace'], 'expected': expected_driver_namespace,
                               'evidence': [evidence('/driver_namespace')]})
        if require_canonical_matches_driver_return:
            for event in events:
                if event['canonical_matches_driver_return'] is False:
                    violations.append({'field': 'canonical_matches_driver_return', 'tick_id': event['tick_id'],
                        'expected': True, 'observed': False,
                        'evidence': [evidence(f"/ticks/{event['tick_id']}/canonical_at_processor"), evidence(f"/ticks/{event['tick_id']}/driver_return")]})
    if qualified and require_recorded_matches_driver_return and recording['status'] == 'pass' and recording['driver_difference_events']:
        first = recording['first_difference']
        violations.append({'field': 'recorded_matches_driver_return', 'expected': True, 'observed': False,
                           'difference_count': recording['driver_difference_events'], 'tick_id': first['tick_id'], 'evidence': first['evidence']})
    explicit = expected_action_names is not None or expected_driver_namespace is not None or require_canonical_matches_driver_return or require_recorded_matches_driver_return
    comparable = not require_canonical_matches_driver_return or all(e['canonical_matches_driver_return'] is not None for e in events)
    comparable = comparable and (not require_recorded_matches_driver_return or
        (recording['status'] == 'pass' and recording['driver_comparison_complete']))
    criterion = 'unknown' if not qualified or not explicit or not comparable else 'error' if violations else 'pass'
    report = {'kind': 'named_action_audit', 'status': 'error' if numeric == 'error' else criterion,
              'binding_status': binding, 'numeric_replay_status': numeric,
              'source_supported': supported, 'coverage_supported': covered,
              'action_names': data['action_names'], 'engine_action_names': data['engine_action_names'],
              'driver_namespace': data['driver_namespace'], 'tick_count': len(events),
              'ticks': events[:12], 'tick_preview_truncated': len(events) > 12,
              'driver_change_events': sum(e['driver_input_matches_return'] is False for e in events),
              'canonical_mutation_events': sum(e['canonical_changed_during_call'] for e in events),
              'first_driver_change': first_driver_change, 'robot_processor_semantics_status': 'unknown',
              'acceptance_contract': {'status': criterion, 'expected_action_names': deepcopy(expected_action_names),
                  'expected_driver_namespace': expected_driver_namespace, 'require_canonical_matches_driver_return': require_canonical_matches_driver_return,
                  'require_recorded_matches_driver_return': require_recorded_matches_driver_return,
                  'violations': violations[:16], 'violation_count': len(violations), 'violations_truncated': len(violations) > 16,
                  'evidence': [evidence('/ticks')],
                  'scope': 'Explicit declared order/software-driver namespace and optional canonical-input or recorded-row/driver-return equality; permits new values. Equality is a caller requirement, not a universal native rule.'},
              'first_intrinsic_contradiction': next(({'tick_id': e['tick_id'], 'evidence': [evidence(f"/ticks/{e['tick_id']}")]}
                                                     for e in events if e['numeric_replay_status'] == 'error'), None),
              'recorded_action_status': recording['status'], 'recording': recording, 'physical_execution_status': 'unknown',
              'policy_selection_status': 'unknown', 'policy_processor_status': 'unknown',
              'tolerances': {'atol': atol, 'rtol': rtol}, 'scope': SCOPE,
              'artifacts': [{'path': str(root / name), 'sha256': digest} for name, digest in sorted(hashes.items())]}
    _recheck(root, hashes)
    if len(json.dumps(report).encode()) > 192 * 1024:
        raise InputError('Named-action report exceeds bounded response budget')
    return report
