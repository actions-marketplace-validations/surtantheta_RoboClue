"""Opt-in, bounded recovery for a source-anchored one-episode-per-file layout.

Missing summaries are computed from actual numeric rows and decoded RGB frames.
This deliberately supports a narrow native rollover layout; it never borrows an
image summary or infers a task/camera mapping from a neighboring episode.
"""
from __future__ import annotations

import math
import re

from .artifacts import InputError

MAX_DECODED_VIDEO_BYTES = 128 * 1024 ** 2
DATA_TEMPLATE = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
VIDEO_TEMPLATE = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'


def _refuse(reason):
    raise InputError('Including recovery candidates refused: ' + reason)


def _numeric_summary(values, spec, names):
    import numpy as np
    shape = spec.get('shape')
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or not 1 <= shape[0] <= 65536:
        _refuse('numeric recovery requires declared scalar/vector feature shapes')
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in 'biuf':
            _refuse('declared numeric source columns must contain actual numeric values')
        array = raw.astype(np.float64).reshape(len(values), -1)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputError('Candidate numeric values cannot be decoded') from exc
    if array.shape != (len(values), shape[0]) or not np.isfinite(array).all():
        _refuse('candidate numeric feature shape/finite-value mismatch')
    summaries = {'min': array.min(0), 'max': array.max(0), 'mean': array.mean(0),
                 'std': array.std(0), 'count': np.asarray([len(array)])}
    for key in names:
        if key.startswith('q') and key[1:].isdigit() and 0 <= int(key[1:]) <= 100:
            summaries[key] = np.quantile(array, int(key[1:]) / 100, axis=0)
    if set(summaries) != names:
        _refuse('unmodeled per-episode statistic names')
    return {key: value.tolist() for key, value in summaries.items()}


def _decode_summary(source_fd, name, spec, length, fps, names):
    import numpy as np

    from .recovery_copy import _file
    try:
        import av
    except ImportError as exc:
        raise InputError('Candidate video recovery needs PyAV; install roboclue[recovery] and retry') from exc
    height, width, channels = spec['shape']
    histogram = np.zeros((3, 256), dtype=np.int64)
    n = 0
    try:
        with _file(source_fd, name) as handle, av.open(handle, mode='r') as container:
            if len(container.streams.video) != 1:
                _refuse(f'exactly one video stream is required: {name}')
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (width, height):
                _refuse(f'video dimensions disagree with the declared camera: {name}')
            stream.thread_count = 1
            for frame in container.decode(stream):
                if n >= length or (frame.width, frame.height) != (width, height):
                    _refuse(f'video frame count/dimensions disagree with rows: {name}')
                if frame.pts is None or frame.time_base is None or abs(float(frame.pts * frame.time_base) - n / fps) > 1e-4:
                    _refuse(f'video timestamps do not match the per-episode row timeline: {name}')
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape != (height, width, channels) or rgb.dtype != np.uint8:
                    _refuse(f'unsupported decoded RGB layout: {name}')
                for channel in range(3):
                    histogram[channel] += np.bincount(rgb[:, :, channel].ravel(), minlength=256)
                n += 1
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f'Candidate video decoding failed for {name}: {type(exc).__name__}: {exc}') from exc
    if n != length:
        _refuse(f'video ended after {n} frames, expected {length}: {name}')
    pixels = length * height * width
    axis = np.arange(256, dtype=np.float64)
    mean = (histogram * axis).sum(1) / pixels
    variance = (histogram * (axis[None, :] - mean[:, None]) ** 2).sum(1) / pixels
    occupied = histogram > 0
    stats = {'min': occupied.argmax(1), 'max': 255 - occupied[:, ::-1].argmax(1),
             'mean': mean, 'std': np.sqrt(variance)}
    cumulative = histogram.cumsum(1)
    for key in names:
        if key.startswith('q') and key[1:].isdigit() and 0 <= int(key[1:]) <= 100:
            rank = (pixels - 1) * int(key[1:]) / 100
            low, high = math.floor(rank), math.ceil(rank)
            a = np.asarray([np.searchsorted(row, low + 1) for row in cumulative])
            b = np.asarray([np.searchsorted(row, high + 1) for row in cumulative])
            stats[key] = a + (b - a) * (rank - low)
    output = {key: (np.asarray(value).reshape(3, 1, 1) / 255).tolist() for key, value in stats.items()}
    output['count'] = [length]
    if set(output) != names:
        _refuse('unmodeled image-summary statistic names')
    return output


def include_candidates(source_fd, report, inventory, selected, prepared, retained_rows,
                       info, metadata_names, account, max_rows):
    """Extend only a contiguous, anchored native rollover prefix, before mkdir."""
    import pyarrow as pa

    from .recovery_copy import _json, _table
    ranges = report.get('episodes', {}).get('recovery_candidates', [])
    candidates = []
    for pair in ranges:
        if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
            _refuse('candidate episode ranges are incomplete')
        first, last = pair
        if first < 0 or last < first or last - first > max_rows:
            _refuse('candidate episode range exceeds the row budget')
        candidates.extend(range(first, last + 1))
    if not candidates:
        _refuse('the source has no complete row/video candidates to include')
    start = len(retained_rows)
    if candidates != list(range(start, start + len(candidates))):
        _refuse('only contiguous candidates immediately after the retained prefix are supported')
    if len(metadata_names) != 1:
        _refuse('candidate recovery currently requires one source episode-metadata file')
    if info.get('data_path') != DATA_TEMPLATE or info.get('video_path') not in (None, VIDEO_TEMPLATE):
        _refuse('unmodeled native data/video path templates')
    chunk_size, fps = info.get('chunks_size'), info.get('fps')
    if type(chunk_size) is not int or chunk_size < 1 or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        _refuse('missing source chunk-size or frame-rate declaration')
    features = info.get('features')
    if not isinstance(features, dict) or not features:
        _refuse('missing source feature schema')
    videos = {key: spec for key, spec in features.items() if spec.get('dtype') == 'video'}
    for key, spec in features.items():
        dtype = spec.get('dtype', '')
        if key in videos:
            shape = spec.get('shape')
            if (not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape)
                    or shape[2] != 3 or spec.get('names') != ['height', 'width', 'channels']
                    or spec.get('info', {}).get('video.is_depth_map', False)):
                _refuse(f'only declared height/width/3 RGB video features are supported: {key}')
        elif not (dtype.startswith(('float', 'int', 'uint')) or dtype == 'bool'):
            _refuse(f'unmodeled feature type: {key}')
    thresholds = {'data': info.get('data_files_size_in_mb')}
    if videos:
        thresholds['video'] = info.get('video_files_size_in_mb')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        _refuse('source rollover size thresholds are missing')
    source_info = _json(source_fd, 'meta/info.json')
    if source_info.get('splits') != {'train': f"0:{source_info.get('total_episodes')}"}:
        _refuse('custom train/evaluation splits cannot be reconstructed safely')
    tasks_table = _table(source_fd, 'meta/tasks.parquet', max_rows)
    account('meta/tasks.parquet', tasks_table)
    tasks = {}
    for row in tasks_table.to_pylist():
        index, text = row.get('task_index'), row.get('task')
        if type(index) is not int or index in tasks or not isinstance(text, str):
            _refuse('source task table does not uniquely map task_index to task text')
        tasks[index] = text
    if not tasks:
        _refuse('source task table is empty')
    meta_name = metadata_names[0]
    location = re.fullmatch(r'meta/episodes/chunk-(\d+)/file-(\d+)\.parquet', meta_name)
    if location is None:
        _refuse('unmodeled episode-metadata file location')
    meta_chunk, meta_file = map(int, location.groups())
    meta_table = prepared.get(meta_name)
    if meta_table is None:
        meta_table = _table(source_fd, meta_name, max_rows)
        account(meta_name, meta_table)
    template = retained_rows[-1]
    base_fields = {'episode_index', 'tasks', 'length', 'data/chunk_index', 'data/file_index',
                   'dataset_from_index', 'dataset_to_index', 'meta/episodes/chunk_index', 'meta/episodes/file_index'}
    for key in videos:
        base_fields.update(f'videos/{key}/{suffix}' for suffix in
                           ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp'))
    if set(template) - base_fields - {k for k in template if k.startswith('stats/')}:
        _refuse('unknown episode metadata fields cannot be copied into missing episode rows')
    for field in template:
        if field.startswith('stats/'):
            feature, sep, _metric = field[6:].rpartition('/')
            if not sep or feature not in features:
                _refuse('unknown episode-summary feature cannot be reconstructed')
    if not {'meta/episodes/chunk_index', 'meta/episodes/file_index'} <= set(template):
        _refuse('source lacks native episode-metadata file coordinates')
    if any((r.get('meta/episodes/chunk_index'), r.get('meta/episodes/file_index')) != (meta_chunk, meta_file)
           for r in retained_rows):
        _refuse('source metadata file coordinates do not match the actual file')
    statistic_names = {}
    for feature in features:
        names = {key[len(f'stats/{feature}/'):] for key in template if key.startswith(f'stats/{feature}/')}
        if not {'min', 'max', 'mean', 'std', 'count'} <= names:
            _refuse(f'source does not declare complete episode-summary fields: {feature}')
        statistic_names[feature] = names
    all_rows, next_index, total_video_bytes = [], 0, 0
    for ep in range(start + len(candidates)):
        chunk, file = divmod(ep, chunk_size)
        name = DATA_TEMPLATE.format(chunk_index=chunk, file_index=file)
        if name not in inventory['files']:
            _refuse(f'missing intact per-episode data file: {name}')
        table = _table(source_fd, name, max_rows)
        account(name, table)
        n = table.num_rows
        if not n or next_index + n > max_rows:
            _refuse('candidate recovery exceeds the complete numeric-row budget')
        data = table.to_pydict()
        if (data.get('episode_index') != [ep] * n or data.get('frame_index') != list(range(n))
                or data.get('index') != list(range(next_index, next_index + n))):
            _refuse(f'one intact, contiguous episode per source data file is required: {name}')
        timestamps = data.get('timestamp', [])
        if len(timestamps) != n or any(type(v) not in (int, float) or not math.isfinite(v)
                                      or abs(v - i / fps) > 1e-4 for i, v in enumerate(timestamps)):
            _refuse(f'numeric timestamps do not follow declared fps: {name}')
        indices = data.get('task_index', [])
        if len(indices) != n or any(type(v) is not int or v not in tasks for v in indices):
            _refuse(f'candidate task indices are missing from the retained source task table: {name}')
        task_texts = list(dict.fromkeys(tasks[v] for v in indices))
        row = dict(retained_rows[ep] if ep < start else template)
        if ep < start and (row.get('data/chunk_index'), row.get('data/file_index'), row.get('length'),
                           row.get('dataset_from_index'), row.get('dataset_to_index'), row.get('tasks')) != (
                           chunk, file, n, next_index, next_index + n, task_texts):
            _refuse('existing metadata does not anchor the one-episode-per-file layout')
        row.update(episode_index=ep, length=n, tasks=task_texts, dataset_from_index=next_index,
                   dataset_to_index=next_index + n)
        row.update({'data/chunk_index': chunk, 'data/file_index': file})
        if inventory['files'][name]['size_bytes'] <= thresholds['data'] * 1024 ** 2:
            _refuse('data file size does not establish the declared per-episode rollover layout')
        selected.add(name)
        for feature, spec in features.items():
            if feature not in videos:
                if feature not in data:
                    _refuse(f'declared numeric feature is missing from source rows: {feature}')
                summary = _numeric_summary(data[feature], spec, statistic_names[feature])
            else:
                prefix = f'videos/{feature}/'
                if ep < start and any(abs(row.get(prefix + suffix, -1) - value) > 1e-4 for suffix, value in (
                        ('chunk_index', chunk), ('file_index', file), ('from_timestamp', 0.), ('to_timestamp', n / fps))):
                    _refuse('existing camera metadata does not anchor per-episode rollover files')
                video = VIDEO_TEMPLATE.format(video_key=feature, chunk_index=chunk, file_index=file)
                if video not in inventory['files'] or inventory['files'][video]['size_bytes'] <= thresholds['video'] * 1024 ** 2:
                    _refuse(f'missing or non-rollover candidate video: {video}')
                total_video_bytes += n * math.prod(spec['shape'])
                if total_video_bytes > MAX_DECODED_VIDEO_BYTES:
                    _refuse('decoded RGB video budget exceeds128 MiB')
                summary = _decode_summary(source_fd, video, spec, n, fps, statistic_names[feature])
                selected.add(video)
                row.update({prefix + 'chunk_index': chunk, prefix + 'file_index': file,
                            prefix + 'from_timestamp': 0., prefix + 'to_timestamp': n / fps})
            row.update({f'stats/{feature}/{key}': value for key, value in summary.items()})
        all_rows.append(row)
        next_index += n
    # The already-computed default plan may trim a file; source-backed candidates
    # in this qualified layout are whole files and must never receive that trim.
    for name in list(prepared):
        if name.startswith('data/'):
            prepared.pop(name)
    prepared[meta_name] = pa.Table.from_pylist(all_rows, schema=meta_table.schema)
    if prepared[meta_name].nbytes > 64 * 1024 ** 2:
        _refuse('regenerated metadata exceeds the decoded-table budget')
    info.update(total_episodes=len(all_rows), total_frames=next_index, total_tasks=len(tasks),
                splits={'train': f'0:{len(all_rows)}'})
    prepared['meta/info.json'] = info
    return all_rows, {'included_episodes': candidates, 'included_frames': next_index - sum(r['length'] for r in retained_rows),
                     'source_layout': 'source-anchored native per-episode data/video rollover',
                     'video_statistics': 'All retained and included RGB frames decoded; channel statistics use every pixel. '
                     'Per-episode quantiles are exact linear histogram quantiles; global quantiles use native conservative '
                     'episode envelopes. These replace original sampled/precompression image summaries.',
                     'numeric_statistics': 'All retained and included numeric rows; exact per-episode linear quantiles.',
                     'decoded_video_bytes': total_video_bytes, 'max_decoded_video_bytes': MAX_DECODED_VIDEO_BYTES}
