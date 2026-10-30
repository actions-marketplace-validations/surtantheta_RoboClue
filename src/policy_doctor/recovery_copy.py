"""Apply a freshly computed, bounded recording-prefix plan to a new local directory.

No user-supplied plan is accepted. Statistics are aggregated from complete retained
native episode summaries, not copied from the damaged recording. This is not a
training, recorder-resume, image-content, or robot-behavior certification.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

from .artifacts import InputError, _pairs, finite_tree
from .salvage import _RECOVERY_COPY_RECHECK, check_salvage

MAX_COPY_BYTES = 1024 ** 3
MAX_TABLE_BYTES = 64 * 1024 ** 2
MAX_JSON_BYTES = 8 * 1024 ** 2
INCOMPLETE = '.roboclue-copy-incomplete.json'
RECEIPT = 'roboclue-recovery.json'
STATS_SOURCE = ('https://github.com/huggingface/lerobot/blob/'
                '146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/datasets/compute_stats.py')
SCOPE = ('Separate retained-prefix copy only. Rows and video headers are rechecked; complete supported numeric '
         'mean/std/min/max are compared with copied rows. Statistics are aggregated from retained episode '
         'summaries. Sampled image summary accuracy, native dataset '
         'reload, recorder resume, policy normalization and training behavior are not verified.')
CANDIDATE_SCOPE = ('Separate source-anchored native rollover copy. Numeric summaries are regenerated from actual '
                   'rows and RGB image summaries from decoded frames for every retained/included episode. '
                   'Native dataset reload, recorder resume, policy normalization and training behavior are not '
                   'verified by this copy operation.')


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _parts(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative:
        raise InputError('Recovery paths must be nonempty relative native paths')
    path = PurePosixPath(relative)
    if path.is_absolute() or any(p in ('', '.', '..') for p in relative.split('/')):
        raise InputError(f'Unsafe recovery path: {relative!r}')
    return path.parts


def _open_directory(path):
    path = Path(os.path.abspath(os.fspath(path)))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return path, fd
    except BaseException:
        os.close(fd)
        raise


@contextlib.contextmanager
def _file(root_fd, relative, *, writing=False):
    parts = _parts(relative)
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            if writing:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= os.O_WRONLY | os.O_CREAT | os.O_EXCL if writing else os.O_RDONLY
        fd = os.open(parts[-1], flags, 0o600, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise InputError(f'Recovery refuses non-regular file: {relative}')
            with os.fdopen(fd, 'wb' if writing else 'rb') as stream:
                fd = None
                yield stream
        finally:
            if fd is not None:
                os.close(fd)
    finally:
        os.close(parent)


def _inventory(root_fd, max_files, max_bytes):
    """Hash every source file, including omitted files, without following any link."""
    files, directories, total = {}, {}, 0

    def visit(fd, prefix, depth):
        nonlocal total
        if depth > 32:
            raise InputError('Recovery source directory depth exceeds32 levels')
        directories[prefix] = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        if len(directories) > max_files * 4 + 32:
            raise InputError('Recovery source directory budget exceeded')
        with os.scandir(fd) as entries:
            for entry in entries:
                relative = f'{prefix}/{entry.name}' if prefix else entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        visit(child, relative, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    if len(files) >= max_files:
                        raise InputError(f'Recovery source file budget exceeds {max_files}')
                    total += info.st_size
                    if total > max_bytes:
                        raise InputError(f'Recovery source byte budget exceeds {max_bytes}')
                    with _file(root_fd, relative) as stream:
                        before = os.fstat(stream.fileno())
                        if _identity(before) != _identity(info):
                            raise InputError(f'Source changed while opening: {relative}')
                        digest, read = hashlib.sha256(), 0
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            read += len(chunk)
                            if read > info.st_size:
                                raise InputError(f'Source grew while hashing: {relative}')
                            digest.update(chunk)
                        if read != info.st_size or _identity(os.fstat(stream.fileno())) != _identity(before):
                            raise InputError(f'Source changed while hashing: {relative}')
                    files[relative] = {'sha256': digest.hexdigest(), 'size_bytes': read,
                                       'identity': list(_identity(info))}
                else:
                    kind = 'symlink' if stat.S_ISLNK(info.st_mode) else 'special file (FIFO/socket/device)'
                    raise InputError(f'Recovery refuses {kind}: {relative}')
    visit(root_fd, '', 0)
    return {'files': dict(sorted(files.items())), 'directories': directories, 'bytes': total}


def _json(root_fd, relative):
    with _file(root_fd, relative) as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise InputError(f'Recovery JSON budget exceeded: {relative}')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
        finite_tree(value)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(f'Invalid recovery JSON {relative}: {exc}') from exc
    if not isinstance(value, dict):
        raise InputError(f'Expected JSON object: {relative}')
    return value


def _write_json(root_fd, relative, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with _file(root_fd, relative, writing=True) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _table(root_fd, relative, max_rows):
    import pyarrow.parquet as pq
    with _file(root_fd, relative) as stream:
        parquet = pq.ParquetFile(stream, thrift_string_size_limit=MAX_JSON_BYTES,
                                 thrift_container_size_limit=1_000_000)
        metadata = parquet.metadata
        size = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        if metadata.num_rows > max_rows or size > MAX_TABLE_BYTES:
            raise InputError(f'Recovery Parquet rewrite budget exceeded: {relative}')
        table = parquet.read()
        if table.nbytes > MAX_TABLE_BYTES:
            raise InputError(f'Recovery decoded table budget exceeded: {relative}')
        return table


def _aggregate_episode_stats(rows, info, original_stats):
    """Restricted native aggregation; incomplete or unmodeled summaries are refused.

    Quantiles are native conservative envelopes, NOT exact pooled quantiles.
    Image summaries retain their native sampling semantics; no pixels are decoded.
    """
    import numpy as np
    features = info.get('features')
    if not isinstance(features, dict) or not features or not rows:
        raise ValueError('No complete retained episode feature summaries')
    expected = set(features)
    if original_stats is not None and set(original_stats) != expected:
        raise ValueError('Global statistics include missing or extra feature families')
    episodes = []
    for index, row in enumerate(rows):
        if row.get('episode_index') != index or not isinstance(row.get('length'), int) or row['length'] <= 0:
            raise ValueError('Retained metadata is not an ordered positive-length prefix')
        summary = {}
        for key, value in row.items():
            if key.startswith('stats/'):
                feature, sep, name = key[6:].rpartition('/')
                if not sep or feature not in expected:
                    raise ValueError('Unmodeled episode statistics key')
                try:
                    array = np.asarray(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError('Unmodeled episode statistics array') from exc
                if array.dtype.kind not in 'iuf' or not 1 <= array.ndim <= 3 or array.size > 65536:
                    raise ValueError('Statistics must be bounded numeric arrays')
                if not np.isfinite(array).all():
                    raise ValueError('Episode statistics contain non-finite values')
                summary.setdefault(feature, {})[name] = array.astype('float64')
        if set(summary) != expected:
            raise ValueError('Per-episode statistics missing for declared features')
        for feature, stats in summary.items():
            if not {'min', 'max', 'mean', 'std', 'count'} <= set(stats):
                raise ValueError(f'Incomplete retained summary: {feature}')
            if any(k not in {'min', 'max', 'mean', 'std', 'count'} and
                   not (k.startswith('q') and k[1:].isdigit() and 0 <= int(k[1:]) <= 100) for k in stats):
                raise ValueError(f'Unmodeled retained statistic: {feature}')
            count = stats['count']
            if count.shape != (1,) or count[0] <= 0 or int(count[0]) != count[0] or count[0] > row['length']:
                raise ValueError(f'Invalid retained summary count: {feature}')
            spec = features[feature]
            if not isinstance(spec, dict):
                raise ValueError('Unmodeled feature description')
            image = spec.get('dtype') in ('image', 'video')
            shape = stats['mean'].shape
            if image:
                if shape not in ((3, 1, 1), (1, 1, 1)):
                    raise ValueError(f'Unmodeled image-summary shape: {feature}')
            elif shape != tuple(spec.get('shape', ())) or count[0] != row['length']:
                raise ValueError(f'Numeric retained-summary shape/count mismatch: {feature}')
            if any(v.shape != shape for k, v in stats.items() if k != 'count'):
                raise ValueError(f'Inconsistent retained-summary shape: {feature}')
            if (stats['std'] < 0).any() or (stats['min'] > stats['max']).any():
                raise ValueError(f'Invalid retained-summary range/variance: {feature}')
            if ((stats['std'] == 0) & (stats['min'] < stats['max'])).any():
                raise ValueError(f'Zero retained-summary variance with a nonzero range: {feature}')
            tolerance = np.maximum(np.maximum(np.abs(stats['min']), np.abs(stats['max'])), 1.) * 1e-6
            if (stats['mean'] < stats['min'] - tolerance).any() or (stats['mean'] > stats['max'] + tolerance).any():
                raise ValueError(f'Retained-summary mean outside its range: {feature}')
            quantiles = sorted((k for k in stats if k.startswith('q')), key=lambda k: int(k[1:]))
            if len({int(k[1:]) for k in quantiles}) != len(quantiles):
                raise ValueError(f'Duplicate retained-summary quantile percent: {feature}')
            previous = stats['min']
            for key in quantiles:
                if (stats[key] < previous - tolerance).any() or (stats[key] > stats['max'] + tolerance).any():
                    raise ValueError(f'Unordered or out-of-range retained quantile: {feature}/{key}')
                previous = stats[key]
            stats['count'] = count.astype('int64')
        episodes.append(summary)
    result = {}
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for feature in sorted(expected):
            values = [s[feature] for s in episodes]
            keys = set(values[0])
            if any(set(v) != keys for v in values):
                raise ValueError(f'Incomplete statistic family across retained episodes: {feature}')
            means = np.stack([v['mean'] for v in values])
            variances = np.stack([v['std'] ** 2 for v in values])
            counts = np.stack([v['count'] for v in values])
            total = counts.sum(axis=0)
            while counts.ndim < means.ndim:
                counts = np.expand_dims(counts, axis=-1)
            mean = (means * counts).sum(axis=0) / total
            variance = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total
            merged = {'min': np.min(np.stack([v['min'] for v in values]), axis=0),
                      'max': np.max(np.stack([v['max'] for v in values]), axis=0),
                      'mean': mean, 'std': np.sqrt(variance), 'count': total}
            for key in sorted(keys - set(merged)):
                arrays = np.stack([v[key] for v in values])
                merged[key] = arrays.min(axis=0) if int(key[1:]) <= 50 else arrays.max(axis=0)
            if any(not np.isfinite(v).all() for v in merged.values()):
                raise ValueError('Aggregation overflow/nonfinite result')
            result[feature] = {key: value.tolist() for key, value in merged.items()}
    return result


def _prepare(source_fd, report, inventory, max_rows, *, include_candidates=False):
    """Validate the recomputed plan and prepare only its supported rewrites in memory."""
    import pyarrow as pa
    plan = report.get('plan') or {}
    files = plan.get('files') or {}
    if (plan.get('strategy') not in ('metadata_copy', 'keep_as_is', 'none_needed', 'none_needed_rows_only') or
            files.get('lists_complete') is not True or report.get('counts', {}).get('findings', {}).get('unknown', 0)):
        raise InputError('Recovery needs a complete supported plan; source plan is unsupported or incomplete')
    if plan.get('rebuild'):
        raise InputError('Recovery cannot automatically rebuild missing metadata/tasks; restore a retained source first')
    copied = files.get('copy_unchanged', [])
    rewrites = files.get('rewrite', [])
    trims = plan.get('required_data_rewrite', [])
    selected = set(copied) | {r['file'] for r in rewrites} | {r['file'] for r in trims}
    if 'meta/info.json' not in selected or 'meta/tasks.parquet' not in selected or not plan.get('kept_episodes'):
        raise InputError('Recovery plan does not retain a complete nonempty metadata prefix')
    if not selected <= set(inventory['files']):
        raise InputError('A planned source file is missing from the stable source inventory')
    for name in selected:
        _parts(name)
    prepared, decoded = {}, {}

    def account(name, table):
        decoded.setdefault(name, table.nbytes)
        if sum(decoded.values()) > MAX_TABLE_BYTES:
            raise InputError('Recovery cumulative decoded-table budget exceeded')
    for rewrite in rewrites:
        name = rewrite['file']
        if name == 'meta/info.json':
            doc = _json(source_fd, name)
            if set(rewrite) != {'file', 'set'} or set(rewrite['set']) - {'total_episodes', 'total_frames', 'total_tasks', 'splits'}:
                raise InputError('Unsupported info.json rewrite')
            doc.update(rewrite['set'])
            prepared[name] = doc
        elif name.startswith('meta/episodes/'):
            table = _table(source_fd, name, max_rows)
            account(name, table)
            first, last = rewrite['keep_physical_rows']
            if first != 0 or not 0 <= last < table.num_rows:
                raise InputError('Recovery supports leading metadata-prefix trims only')
            prepared[name] = table.slice(first, last + 1)
        else:
            raise InputError(f'Unsupported recovery rewrite: {name}')
    for trim in trims:
        name = trim['file']
        if not name.startswith('data/') or trim.get('range_end') != 'exclusive':
            raise InputError('Unsupported data trim')
        table = _table(source_fd, name, max_rows)
        account(name, table)
        first, end = trim['keep_physical_rows']
        if first != 0 or not 0 < end < table.num_rows or trim['rows_to_exclude'] != table.num_rows - end:
            raise InputError('Recovery supports declared leading data-prefix trims only')
        prepared[name] = table.slice(0, end)
    corrections = {p['episode_index']: dict(p['to']) for p in plan.get('pointer_corrections', [])}
    for repair in plan.get('row_span_corrections', []):
        corrections.setdefault(repair['episode_index'], {}).update(repair['to'])
    metadata_names = sorted(name for name in selected if name.startswith('meta/episodes/') and name.endswith('.parquet'))
    retained_rows = []
    for name in metadata_names:
        table = prepared.get(name)
        if table is None:
            table = _table(source_fd, name, max_rows)
        account(name, table)
        rows = table.to_pylist()
        changed = False
        for row in rows:
            for key, value in corrections.get(row.get('episode_index'), {}).items():
                if (key not in ('data/chunk_index', 'data/file_index', 'dataset_from_index', 'dataset_to_index')
                        or type(value) is not int or value < 0):
                    raise InputError('Unsupported recovery pointer correction')
                row[key] = value
                changed = True
        if changed:
            prepared[name] = pa.Table.from_pylist(rows, schema=table.schema)
        retained_rows.extend(rows)
    if len(retained_rows) != plan['kept_episodes'] or sum(r.get('length', 0) for r in retained_rows) != plan['kept_frames']:
        raise InputError('Retained metadata does not match the recomputed plan')
    info = prepared.get('meta/info.json') or _json(source_fd, 'meta/info.json')
    candidate_details = None
    if include_candidates:
        from .recovery_candidates import include_candidates as extend_prefix
        retained_rows, candidate_details = extend_prefix(
            source_fd, report, inventory, selected, prepared, retained_rows,
            info, metadata_names, account, max_rows)
    statistics = {'status': 'unknown', 'method': 'retained_episode_summary_aggregation',
                  'source': STATS_SOURCE, 'scope': 'No image decoding; summaries are assumed to describe their retained rows. '
                  'Quantiles are conservative native envelopes, not exact pooled quantiles.'}
    if candidate_details is not None:
        statistics.update(method='source_row_and_decoded_frame_regeneration',
                          scope=candidate_details['video_statistics'], candidates=candidate_details)
    try:
        old = _json(source_fd, 'meta/stats.json') if 'meta/stats.json' in inventory['files'] else None
        stats = _aggregate_episode_stats(retained_rows, info, old)
        prepared['meta/stats.json'] = stats
        statistics.update(status='pass', retained_episodes=len(retained_rows), feature_count=len(stats))
    except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError) as exc:
        statistics['reason'] = str(exc)[:300]
    # Never propagate an aggregate that includes omitted episodes, even if aggregation is unavailable.
    selected.discard('meta/stats.json')
    return selected, prepared, statistics


def write_recovery_copy(source, destination, *, max_rows=200_000, max_files=1024,
                        max_copy_bytes=MAX_COPY_BYTES, include_candidates=False):
    """Create a new copy and verify a supported retained prefix; never change source.

    The explicit destination must not exist. Invalid/unsupported plans fail before
    creation. Failures after creation retain an incomplete marker and raise InputError.
    A PASS is scoped to the structural recheck and retained-summary aggregation;
    it is never training or native resume approval.
    """
    if type(include_candidates) is not bool:
        raise InputError('include_candidates must be a Boolean')
    scope = CANDIDATE_SCOPE if include_candidates else SCOPE
    for value, name, maximum in ((max_rows, 'max_rows', 1_000_000), (max_files, 'max_files', 4096),
                                 (max_copy_bytes, 'max_copy_bytes', 64 * MAX_COPY_BYTES)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise InputError(f'{name} must be a positive integer no larger than {maximum}')
    if str(source).startswith('hf://'):
        raise InputError('Recovery writes require a local source directory; download a separate local copy first')
    source_fd = parent_fd = output_fd = None
    created = False
    try:
        source, source_fd = _open_directory(source)
        destination = Path(os.path.abspath(os.fspath(destination)))
        try:
            parent, parent_fd = _open_directory(destination.parent)
        except FileNotFoundError as exc:
            raise InputError(f'Recovery destination parent does not exist: {destination.parent}. '
                             'Create that directory or choose an existing parent before retrying.') from exc
        destination = parent / destination.name
        receipt = destination.with_name(destination.name + '.' + RECEIPT)
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise InputError('Recovery destination must be separate from the source tree and its ancestors')
        if os.path.lexists(destination):
            raise InputError(f'Recovery destination already exists; choose a new directory: {destination}')
        if os.path.lexists(receipt):
            raise InputError(f'Recovery receipt already exists; choose a new destination: {receipt}')
        try:
            fcntl.flock(source_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InputError('Another cooperating RoboClue process holds the source directory lock') from exc
        original_root = os.fstat(source_fd)
        before = _inventory(source_fd, max_files, max_copy_bytes)
        if INCOMPLETE in before['files']:
            raise InputError('Source is an incomplete RoboClue copy; keep it unqualified and return to the original')
        report = check_salvage(source, max_rows=max_rows, max_files=max_files, video_headers=True)
        selected, prepared, statistics = _prepare(source_fd, report, before, max_rows,
                                                  include_candidates=include_candidates)
        if _inventory(source_fd, max_files, max_copy_bytes) != before:
            raise InputError('Source changed while computing the recovery plan; no copy was created')
        os.mkdir(destination.name, mode=0o700, dir_fd=parent_fd)
        created = True
        output_fd = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        _write_json(output_fd, INCOMPLETE, {'verified': False, 'scope': scope,
                                          'reason': 'Copy creation or verification has not finished'})
        for name in sorted(selected - set(prepared)):
            with _file(source_fd, name) as src, _file(output_fd, name, writing=True) as dst:
                digest, count = hashlib.sha256(), 0
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > before['files'][name]['size_bytes']:
                        raise InputError(f'Source grew while copying: {name}')
                    digest.update(chunk)
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
                if count != before['files'][name]['size_bytes'] or digest.hexdigest() != before['files'][name]['sha256']:
                    raise InputError(f'Source changed while copying: {name}')
        import pyarrow.parquet as pq
        for name, value in sorted(prepared.items()):
            if name.endswith('.json'):
                _write_json(output_fd, name, value)
            else:
                with _file(output_fd, name, writing=True) as stream:
                    pq.write_table(value, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
        written = _inventory(output_fd, max_files + 4, max_copy_bytes + MAX_TABLE_BYTES)
        for name in selected - set(prepared):
            expected, actual = before['files'][name], written['files'][name]
            if any(expected[key] != actual[key] for key in ('sha256', 'size_bytes')):
                raise InputError(f'Destination changed while writing unchanged file: {name}')
        candidate = check_salvage(destination, max_rows=max_rows, max_files=max_files, video_headers=True,
                                  _recovery_copy_token=_RECOVERY_COPY_RECHECK)
        after = _inventory(source_fd, max_files, max_copy_bytes)
        _, named_source_fd = _open_directory(source)
        try:
            root_same = _identity(os.fstat(named_source_fd))[:2] == _identity(original_root)[:2]
        finally:
            os.close(named_source_fd)
        if not root_same or after != before:
            raise InputError('Source changed during copy/verification; output remains explicitly incomplete')
        _, named_output_fd = _open_directory(destination)
        try:
            if _identity(os.fstat(named_output_fd))[:2] != _identity(os.fstat(output_fd))[:2]:
                raise InputError('Destination changed during copy/verification')
        finally:
            os.close(named_output_fd)
        status = candidate['status']
        statistics['aggregation_status'] = statistics['status']
        comparison = candidate.get('statistics_comparison', {})
        statistics['row_comparison_status'] = comparison.get('status', 'unknown')
        statistics['observed_numeric_rows_compared'] = bool(comparison.get('compared_features'))
        if comparison.get('status') == 'error':
            statistics['status'] = 'error'
        elif comparison.get('status') != 'pass' and statistics['status'] == 'pass':
            statistics['status'] = 'unknown'
        if statistics['status'] != 'pass' and status != 'error':
            status = 'unknown'
        output = _inventory(output_fd, max_files + 4, max_copy_bytes + MAX_TABLE_BYTES)
        if output != written:
            raise InputError('Destination changed during candidate verification; output remains incomplete')
        result = {'schema_version': 1, 'kind': 'recovery_copy', 'status': status,
                  'copy_created': True, 'destination': str(destination), 'original_unchanged': True,
                  'receipt': str(receipt),
                  'source_report': report, 'candidate_report': candidate, 'statistics': statistics,
                  'scope': scope,
                  'verification': {'recording_integrity': candidate['status'], 'statistics': statistics['status'],
                                   'native_reload': 'not_run', 'training_qualification': 'not_assessed'},
                  'source_manifest': before['files'],
                  'output_manifest': {k: v for k, v in output['files'].items() if k != INCOMPLETE},
                  'files_copied_unchanged': sorted(selected - set(prepared)),
                  'files_rewritten': sorted(prepared),
                  'files_omitted': sorted(set(before['files']) - selected - set(prepared)),
                  'bounds': {'max_rows': max_rows, 'max_files': max_files, 'max_copy_bytes': max_copy_bytes}}
        _write_json(parent_fd, receipt.name, result)
        os.fsync(parent_fd)
        os.fsync(output_fd)
        os.unlink(INCOMPLETE, dir_fd=output_fd)
        return result
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        suffix = f'; new directory {destination} is incomplete and unverified' if created else '; no copy created'
        raise InputError(f'Recovery copy refused: {exc}{suffix}') from exc
    finally:
        for fd in (output_fd, parent_fd, source_fd):
            if fd is not None:
                os.close(fd)
