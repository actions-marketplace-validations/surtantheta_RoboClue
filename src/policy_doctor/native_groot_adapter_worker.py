"""Fixed CPU GR00T adapter dry run. Standalone Python 3.10; no artifact code imports."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import string
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

MAX_JSON = 2 * 1024**2
MAX_FILE = 128 * 1024**2
MAX_ARRAYS = 2 * 1024**2
MODEL_FILES = ('processor_config.json', 'statistics.json')
STATE_KEYS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
LAYOUT = ['x', 'y', 'z', 'axis_angle1', 'axis_angle2', 'axis_angle3', 'gripper', 'gripper']
STREAMS = [{'input': 'agentview', 'model': 'image', 'source': 'observation.images.image'},
           {'input': 'wrist', 'model': 'wrist_image', 'source': 'observation.images.wrist_image'}]
CONTRACT = {'kind': 'IPEC_LIBERO_selected_source_preservation', 'association': 'explicit',
            'boundary': 'policy_get_action_input_pre_normalization', 'state_layout': LAYOUT,
            'ordered_source_streams': STREAMS}


def relative(value):
    if (not isinstance(value, str) or not value or len(value) > 512 or '\\' in value or ':' in value
            or Path(value).is_absolute() or any(x in {'', '.', '..'} for x in value.split('/'))):
        raise ValueError('Expected a safe contained relative dependency path')
    return value


def stamp(info):
    return tuple(getattr(info, x) for x in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def open_regular(path):
    path = Path(path).expanduser().absolute()
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ValueError('Selected dependency is not a regular file')
        return fd, info
    finally:
        os.close(parent)


def read_bytes(path, maximum=MAX_JSON):
    fd, before = open_regular(path)
    try:
        if before.st_size > maximum:
            raise ValueError('Selected dependency exceeds byte bound')
        chunks, count = [], 0
        while block := os.read(fd, min(1024**2, maximum + 1 - count)):
            chunks.append(block)
            count += len(block)
            if count > maximum:
                raise ValueError('Selected dependency grew beyond byte bound')
        if stamp(before) != stamp(os.fstat(fd)) or count != before.st_size:
            raise ValueError('Selected dependency changed during read')
        other, current = open_regular(path)
        os.close(other)
        if stamp(before) != stamp(current):
            raise ValueError('Selected dependency identity changed during read')
        return b''.join(chunks)
    finally:
        os.close(fd)


def no_code(value):
    pending, count = [(value, 0)], 0
    forbidden = {'auto_map', 'class_path', '_target_', 'module', 'python', 'trust_remote_code', 'code', 'callable'}
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 100_000 or depth > 24:
            raise ValueError('Metadata structure exceeds bound')
        if isinstance(item, dict):
            if any(not isinstance(k, str) or k.lower() in forbidden for k in item):
                raise ValueError('Artifact code declarations are unsupported')
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Nonfinite metadata is unsupported')
        elif isinstance(item, str) and len(item) > 16_384:
            raise ValueError('Metadata string exceeds bound')


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    no_code(value)
    return value


def read_json(path):
    return loads(read_bytes(path))


def validate_contract(contract):
    if contract != CONTRACT or type(contract) is not dict:
        raise ValueError('Only the explicit supported LIBERO source-preservation contract is supported')
    return json.loads(json.dumps(contract))


def validate_samples(samples):
    if (type(samples) is not list or len(samples) != 1 or type(samples[0]) is not dict
            or set(samples[0]) != {'episode_index', 'frame_index'}
            or any(type(v) is not int or not 0 <= v <= 10_000_000 for v in samples[0].values())):
        raise ValueError('Exactly one explicit nonnegative native episode/frame selector is required')
    return samples[0]


def selected_path(info, key, episode, video_key=None):
    template = info.get(key)
    if not isinstance(template, str) or len(template) > 512:
        raise ValueError('Missing bounded native v2.1 path template')
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is not None and (field not in {'episode_chunk', 'episode_index', 'video_key'}
                                  or spec not in {'', '03d', '06d'} or conversion is not None):
            raise ValueError('Unsupported native path template field')
    chunks = info.get('chunks_size')
    if type(chunks) is not int or not 1 <= chunks <= 100_000:
        raise ValueError('Invalid native episode chunk size')
    path = relative(template.format(episode_chunk=episode // chunks, episode_index=episode, video_key=video_key))
    if not path.startswith('data/' if key == 'data_path' else 'videos/') or not path.endswith('.parquet' if key == 'data_path' else '.mp4'):
        raise ValueError('Unsupported native selected payload path')
    return path


def validate_info(info):
    if type(info) is not dict or info.get('codebase_version') != 'v2.1':
        raise ValueError('Only original supported v2.1 episode-file metadata is supported')
    if type(info.get('fps')) is not int or not 1 <= info['fps'] <= 240:
        raise ValueError('Native dataset FPS must be a bounded positive integer')
    features = info.get('features')
    if not isinstance(features, dict):
        raise ValueError('Missing native feature metadata')
    state = features.get('observation.state', {})
    if state.get('dtype') != 'float32' or state.get('shape') != [8] or state.get('names') != {'motors': LAYOUT}:
        raise ValueError('Unsupported explicit native LIBERO state layout')
    for name, dtype, shape in [('action', 'float32', [7]), ('timestamp', 'float32', [1]),
                               ('frame_index', 'int64', [1]), ('episode_index', 'int64', [1]),
                               ('index', 'int64', [1]), ('task_index', 'int64', [1])]:
        feature = features.get(name, {})
        if feature.get('dtype') != dtype or feature.get('shape') != shape:
            raise ValueError('Unsupported original native feature schema: ' + name)
    for item in STREAMS:
        feature = features.get(item['source'], {})
        if feature.get('dtype') != 'video' or feature.get('shape') != [256, 256, 3]:
            raise ValueError('Unsupported native available camera feature')
    return info


def validate_model(model):
    model = Path(model)
    config, stats = (read_json(model / name) for name in MODEL_FILES)
    if (type(config) is not dict or set(config) != {'processor_class', 'processor_kwargs'}
            or config['processor_class'] != 'Gr00tN1d6Processor'):
        raise ValueError('Unsupported processor constructor declaration')
    kwargs = config['processor_kwargs']
    allowed = {'modality_configs', 'image_crop_size', 'image_target_size', 'use_albumentations',
               'random_rotation_angle', 'color_jitter_params', 'shortest_image_edge', 'crop_fraction',
               'model_name', 'model_type', 'formalize_language', 'max_state_dim', 'max_action_dim',
               'max_action_horizon', 'use_percentiles', 'clip_outliers', 'apply_sincos_state_encoding',
               'use_relative_action'}
    flags = ('use_percentiles', 'clip_outliers', 'apply_sincos_state_encoding', 'use_relative_action')
    if type(kwargs) is not dict or set(kwargs) - allowed or any(type(kwargs.get(k)) is not bool for k in flags):
        raise ValueError('Unsupported processor schema or native processing flags')
    modalities = kwargs.get('modality_configs', {}).get('libero_panda')
    if type(modalities) is not dict or set(modalities) != {'video', 'state', 'action', 'language'}:
        raise ValueError('Unsupported selected embodiment modalities')
    fields = {'delta_indices', 'modality_keys', 'sin_cos_embedding_keys', 'mean_std_embedding_keys', 'action_configs'}
    for _name, value in modalities.items():
        if type(value) is not dict or set(value) - fields:
            raise ValueError('Unsupported native modality constructor fields')
        keys, indices = value.get('modality_keys'), value.get('delta_indices')
        if (type(keys) is not list or not 1 <= len(keys) <= 64
                or any(not isinstance(k, str) or not k or len(k) > 256 for k in keys)
                or len(set(keys)) != len(keys) or type(indices) is not list or not 1 <= len(indices) <= 128
                or any(type(i) is not int or not -128 <= i <= 128 for i in indices)):
            raise ValueError('Invalid native modality keys or indices')
        for field in ('sin_cos_embedding_keys', 'mean_std_embedding_keys'):
            if value.get(field) is not None and (type(value[field]) is not list or any(k not in keys for k in value[field])):
                raise ValueError('Invalid native embedding keys')
        actions = value.get('action_configs')
        if actions is not None:
            if type(actions) is not list or len(actions) != len(keys):
                raise ValueError('Invalid native action configuration list')
            for action in actions:
                if (type(action) is not dict or set(action) != {'rep', 'type', 'format', 'state_key'}
                        or action['rep'] not in {'ABSOLUTE', 'RELATIVE', 'DELTA'}
                        or action['type'] not in {'NON_EEF', 'EEF'}
                        or action['format'] not in {'DEFAULT', 'XYZ_ROT6D', 'XYZ_ROTVEC'}
                        or action['state_key'] is not None and action['state_key'] not in STATE_KEYS):
                    raise ValueError('Unsupported native action constructor')
    if (modalities['video']['modality_keys'] != ['image', 'wrist_image']
            or modalities['state']['modality_keys'] != STATE_KEYS
            or modalities['action']['modality_keys'] != STATE_KEYS
            or any(modalities[k]['delta_indices'] != [0] for k in ('state', 'video', 'language'))):
        raise ValueError('Unsupported selected native adapter recipe')
    selected_stats = stats.get('libero_panda') if isinstance(stats, dict) else None
    if not isinstance(selected_stats, dict) or not {'state', 'action'} <= set(selected_stats):
        raise ValueError('Missing selected original processor statistics')
    for role in ('state', 'action'):
        groups = selected_stats[role]
        if type(groups) is not dict or set(groups) != set(STATE_KEYS):
            raise ValueError('Invalid selected statistics group schema')
        for key, values in groups.items():
            width = 2 if role == 'state' and key == 'gripper' else 1
            if (type(values) is not dict or not {'mean', 'std', 'min', 'max', 'q01', 'q99'} <= set(values)
                    or any(type(values[k]) is not list or len(values[k]) != width
                           or any(type(x) not in (int, float) or abs(x) > 1e100 for x in values[k])
                           for k in ('mean', 'std', 'min', 'max', 'q01', 'q99'))):
                raise ValueError('Invalid selected native statistic widths/values')
    return kwargs, selected_stats


def jsonl(path):
    raw = read_bytes(path)
    lines = raw.splitlines()
    if len(lines) > 20_000:
        raise ValueError('Native JSONL metadata exceeds record bound')
    return [loads(line) for line in lines if line.strip()]


def selected_episode(path, sample):
    records = jsonl(path)
    if any(type(x) is not dict or type(x.get('episode_index')) is not int or x['episode_index'] < 0 for x in records):
        raise ValueError('Native episode metadata requires positive typed original IDs')
    if len({x['episode_index'] for x in records}) != len(records):
        raise ValueError('Native episode metadata contains duplicate identities')
    selected = [x for x in records if x['episode_index'] == sample['episode_index']]
    if (len(selected) != 1 or type(selected[0].get('length')) is not int
            or not 0 <= sample['frame_index'] < selected[0]['length'] <= 100_000):
        raise ValueError('Selected native episode/frame metadata is absent or ambiguous')
    tasks = selected[0].get('tasks')
    if (type(tasks) is not list or not 1 <= len(tasks) <= 256
            or any(not isinstance(x, str) or not x or len(x) > 16_384 for x in tasks)):
        raise ValueError('Selected native episode tasks must be an original list of strings')
    return selected[0]


def validate_request(request):
    common = {'operation', 'source_root', 'groot_source_root', 'model', 'samples', 'contract', 'output'}
    if type(request) is not dict or request.get('operation') not in {'probe', 'replay'}:
        raise ValueError('Unsupported fixed native worker operation')
    required = common | ({'dataset'} if request['operation'] == 'probe' else {'raw_samples'})
    if set(request) != required:
        raise ValueError('Unsupported worker request fields')
    validate_samples(request['samples'])
    validate_contract(request['contract'])
    for name in ('source_root', 'groot_source_root', 'model', 'output'):
        if not isinstance(request[name], str) or not Path(request[name]).is_absolute():
            raise ValueError('Worker code/input/output paths must be operator-selected absolute paths')
    if request['operation'] == 'probe':
        if not isinstance(request['dataset'], str) or not Path(request['dataset']).is_absolute():
            raise ValueError('Dataset path must be absolute')
    else:
        raw = request['raw_samples']
        if (type(raw) is not list or len(raw) != 1 or type(raw[0]) is not dict
                or set(raw[0]) != {'arrays', 'metadata'}
                or any(not isinstance(x, str) or not Path(x).is_absolute() for x in raw[0].values())):
            raise ValueError('One bounded retained arrays/metadata pair is required')


def _npz(raw):
    shapes = {'state': ('<f4', (8,)), 'recorded_action': ('<f4', (7,)),
              'image': ('|u1', (256, 256, 3)), 'wrist_image': ('|u1', (256, 256, 3))}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if set(archive.namelist()) != {x + '.npy' for x in shapes} or len(archive.infolist()) != 4:
            raise ValueError('Retained sample NPZ has unsupported members')
        if sum(x.file_size for x in archive.infolist()) > MAX_ARRAYS:
            raise ValueError('Retained sample NPZ decoded bytes exceed bound')
        for name, (dtype, shape) in shapes.items():
            with archive.open(name + '.npy') as handle:
                prefix = handle.read(8)
                if prefix[:6] != b'\x93NUMPY' or prefix[6:8] not in {b'\x01\x00', b'\x02\x00'}:
                    raise ValueError('Unsupported retained NPY format')
                size = int.from_bytes(handle.read(2 if prefix[6] == 1 else 4), 'little')
                if not 1 <= size <= 16_384:
                    raise ValueError('Retained NPY header exceeds bound')
                header = ast.literal_eval(handle.read(size).decode('latin1'))
                if header != {'descr': dtype, 'fortran_order': False, 'shape': shape}:
                    raise ValueError('Retained NPY dtype/shape is unsupported')
    import numpy as np
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        result = {key: arrays[key].copy() for key in shapes}
    if not all(np.isfinite(result[k]).all() for k in ('state', 'recorded_action')):
        raise ValueError('Retained numerical input is not finite')
    return result


def _probe(dataset, sample):
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(dataset)
    info = validate_info(read_json(root / 'meta/info.json'))
    episode, frame = sample['episode_index'], sample['frame_index']
    episode_record = selected_episode(root / 'meta/episodes.jsonl', sample)
    path = selected_path(info, 'data_path', episode)
    # The sibling guard is fixed installed producer code, never an artifact-supplied import.
    spec = importlib.util.spec_from_file_location('_pd_groot_parquet_guard', Path(__file__).with_name('native_parquet_guard.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.guard_file(root / path)
    raw = read_bytes(root / path, MAX_FILE)
    parquet = pq.ParquetFile(io.BytesIO(raw), thrift_string_size_limit=2 * 1024**2, thrift_container_size_limit=100_000)
    if parquet.metadata.num_rows != episode_record['length']:
        raise ValueError('Native episode file row count differs from original metadata')
    columns = ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
    table = parquet.read(columns=columns)
    for key in ('observation.state', 'action'):
        dtype = table.schema.field(key).type
        if not (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)) or not pa.types.is_float32(dtype.value_type):
            raise ValueError('Native state/action dtype is unsupported')
    for key in ('frame_index', 'episode_index', 'index', 'task_index'):
        if not pa.types.is_int64(table.schema.field(key).type):
            raise ValueError('Native row identifiers must be original int64 values')
    if not pa.types.is_float32(table.schema.field('timestamp').type):
        raise ValueError('Native timestamp dtype is unsupported')
    rows = table.to_pylist()
    if ([x['frame_index'] for x in rows] != list(range(len(rows)))
            or any(x['episode_index'] != episode for x in rows)
            or len({x['index'] for x in rows}) != len(rows)):
        raise ValueError('Native original row identity is ambiguous')
    row = rows[frame]
    if type(row['timestamp']) not in (int, float) or not math.isfinite(row['timestamp']) or abs(row['timestamp'] - frame / info['fps']) > 1e-5:
        raise ValueError('Selected native timestamp does not match supported episode clock')
    task_records = jsonl(root / 'meta/tasks.jsonl')
    if (any(type(x) is not dict or type(x.get('task_index')) is not int or x['task_index'] < 0 for x in task_records)
            or len({x['task_index'] for x in task_records}) != len(task_records)):
        raise ValueError('Native task metadata requires unique typed original IDs')
    tasks = [x for x in task_records if x['task_index'] == row['task_index']]
    if (len(tasks) != 1 or not isinstance(tasks[0].get('task'), str) or len(tasks[0]['task']) > 16_384
            or tasks[0]['task'] not in episode_record['tasks']):
        raise ValueError('Selected original task identity is absent or ambiguous')
    arrays = {'state': np.asarray(row['observation.state'], dtype=np.float32),
              'recorded_action': np.asarray(row['action'], dtype=np.float32)}
    if arrays['state'].shape != (8,) or arrays['recorded_action'].shape != (7,) or not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Selected native state/action shape or values are unsupported')
    meta = {'episode_index': episode, 'frame_index': frame, 'global_index': row['index'],
            'timestamp': row['timestamp'], 'task_index': row['task_index'], 'task': tasks[0]['task'],
            'state_dtype': arrays['state'].dtype.str, 'state_shape': [8], 'state_values': arrays['state'].tolist(),
            'state_sha256': hashlib.sha256(arrays['state'].tobytes()).hexdigest(),
            'recorded_action_scope': 'original recorded label, not model output',
            'parquet_source_sha256': hashlib.sha256(raw).hexdigest(), 'parquet_relative_path': path, 'cameras': {}}
    for stream in STREAMS:
        relative_video = selected_path(info, 'video_path', episode, stream['source'])
        video = read_bytes(root / relative_video, MAX_FILE)
        matches = []
        with av.open(io.BytesIO(video)) as container:
            stream_obj = container.streams.video[0]
            stream_obj.codec_context.thread_count = 1
            if stream_obj.width != 256 or stream_obj.height != 256:
                raise ValueError('Original video dimensions exceed supported camera recipe')
            previous, count = -math.inf, 0
            for ordinal, decoded in enumerate(container.decode(stream_obj)):
                count += 1
                if count > episode_record['length'] or decoded.pts is None or decoded.time_base is None:
                    raise ValueError('Video exceeds bounded native episode or lacks PTS')
                timestamp = float(decoded.pts * decoded.time_base)
                if timestamp <= previous or abs(timestamp - ordinal / info['fps']) > 1e-5:
                    raise ValueError('Video PTS does not follow supported native episode clock')
                previous = timestamp
                if abs(timestamp - row['timestamp']) <= 1e-5:
                    if decoded.width != 256 or decoded.height != 256:
                        raise ValueError('Decoded original camera dimensions changed')
                    rgb = decoded.to_ndarray(format='rgb24')
                    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                        raise ValueError('Unsupported decoded original camera shape/dtype')
                    arrays[stream['model']] = rgb
                    matches.append({'ordinal': ordinal, 'pts': decoded.pts, 'time_base': str(decoded.time_base),
                                    'time_seconds': timestamp, 'pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                    'pixel_dtype': rgb.dtype.str, 'pixel_shape': list(rgb.shape),
                                    'source_key': stream['source'], 'source_file': str(root / relative_video),
                                    'source_relative_path': relative_video, 'source_sha256': hashlib.sha256(video).hexdigest(),
                                    'codec': stream_obj.codec_context.name, 'stream_time_base': str(stream_obj.time_base)})
        if count != episode_record['length'] or len(matches) != 1 or matches[0]['ordinal'] != frame:
            raise ValueError('Original row/video timestamp association is absent or ambiguous')
        meta['cameras'][stream['model']] = matches[0]
    return arrays, meta


def _sample(request, output):
    sample = request['samples'][0]
    if request['operation'] == 'probe':
        arrays, meta = _probe(request['dataset'], sample)
        import numpy as np
        np.savez(output / 'samples/arrays.npz', **arrays)
        meta['arrays_sha256'] = hashlib.sha256(read_bytes(output / 'samples/arrays.npz', MAX_ARRAYS)).hexdigest()
        (output / 'samples/metadata.json').write_bytes(_canonical(meta))
    else:
        raw = request['raw_samples'][0]
        data = read_bytes(raw['arrays'], MAX_ARRAYS)
        meta_bytes = read_bytes(raw['metadata'])
        meta = loads(meta_bytes)
        arrays = _npz(data)
        if meta.get('arrays_sha256') != hashlib.sha256(data).hexdigest():
            raise ValueError('Retained input NPZ digest differs from metadata')
        (output / 'samples/arrays.npz').write_bytes(data)
        (output / 'samples/metadata.json').write_bytes(meta_bytes)
    if (meta.get('episode_index') != sample['episode_index'] or meta.get('frame_index') != sample['frame_index']
            or type(meta.get('episode_index')) is not int or type(meta.get('frame_index')) is not int
            or type(meta.get('global_index')) is not int or type(meta.get('task_index')) is not int
            or type(meta.get('timestamp')) not in (float, int) or not math.isfinite(meta['timestamp'])
            or meta.get('state_dtype') != arrays['state'].dtype.str or meta.get('state_shape') != [8]
            or meta.get('state_values') != arrays['state'].tolist()
            or meta.get('state_sha256') != hashlib.sha256(arrays['state'].tobytes()).hexdigest()
            or not isinstance(meta.get('task'), str) or len(meta['task']) > 16_384
            or not isinstance(meta.get('parquet_source_sha256'), str) or len(meta['parquet_source_sha256']) != 64):
        raise ValueError('Missing or contradictory retained original row witness')
    relative(meta.get('parquet_relative_path'))
    for stream in STREAMS:
        camera = meta.get('cameras', {}).get(stream['model'], {})
        if (camera.get('source_key') != stream['source'] or camera.get('pixel_dtype') != '|u1'
                or camera.get('pixel_shape') != [256, 256, 3]
                or not isinstance(camera.get('source_sha256'), str) or len(camera['source_sha256']) != 64
                or camera.get('pixel_sha256') != hashlib.sha256(arrays[stream['model']].tobytes()).hexdigest()
                or type(camera.get('ordinal')) is not int or camera.get('ordinal') != sample['frame_index'] or type(camera.get('pts')) is not int
                or type(camera.get('time_seconds')) not in (int, float)
                or abs(camera['time_seconds'] - meta['timestamp']) > 1e-5):
            raise ValueError('Missing or contradictory retained decoded camera witness')
        relative_video = relative(camera.get('source_relative_path'))
        if not relative_video.startswith('videos/') or not relative_video.endswith('.mp4'):
            raise ValueError('Unsupported retained original video binding')
    return arrays, meta


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _capture(request, arrays, meta, kwargs, stats):
    import copy

    import numpy as np
    source, groot = Path(request['source_root']), Path(request['groot_source_root'])
    sys.path[:0] = [str(source / 'src') if (source / 'src').is_dir() else str(source), str(groot)]
    import vla_eval.model_servers.groot as native_module
    from vla_eval.model_servers.base import SessionContext
    from vla_eval.model_servers.groot import GR00TModelServer
    if not Path(native_module.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('Native adapter import escaped operator source root')
    with patch.object(GR00TModelServer, '_init_policy', lambda self: None):
        server = GR00TModelServer(embodiment_tag='LIBERO_PANDA', image_resolution=None, invert_gripper=False)
    # Only known native dataclass constructors are used; artifact class/path fields were refused above.
    import gr00t.data.state_action.state_action_processor as component_module
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.utils import parse_modality_configs
    if not Path(component_module.__file__).resolve().is_relative_to(groot.resolve()):
        raise ValueError('Native state-component import escaped operator source root')
    mods = parse_modality_configs({'libero_panda': kwargs['modality_configs']['libero_panda']})
    server._modality_config = mods['libero_panda']
    server._language_key = mods['libero_panda']['language'].modality_keys[0]
    server._state_dims = {k: len(stats['state'][k]['mean']) for k in STATE_KEYS}
    params = server.get_observation_params()
    if type(params) is not dict or type(params.get('send_wrist_image', False)) is not bool:
        raise ValueError('Native requested-image recipe is unsupported')
    images = {'agentview': arrays['image']}
    supplied = params.get('send_wrist_image', False)
    if supplied:
        images['wrist'] = arrays['wrist_image']
    captured = {}
    class Sink:
        def get_action(self, observation):
            captured.update(copy.deepcopy(observation))
            return {key: np.zeros((1, 1, len(stats['action'][key]['mean'])), dtype=np.float32)
                    for key in STATE_KEYS}, None
    server._policy = Sink()
    def leaf(value, values=False):
        value = np.ascontiguousarray(value)
        result = {'dtype': value.dtype.str, 'shape': list(value.shape),
                  'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        if values:
            result['values'] = value.reshape(-1).tolist()
        return result
    report = {'kind': 'native_adapter_boundary_capture', 'embodiment': 'LIBERO_PANDA',
              'contract': request['contract'], 'route': 'native_request', 'views': 'native_request',
              'frame': meta['frame_index'], 'control': 'original',
              'source_module_sha256': hashlib.sha256(read_bytes(native_module.__file__)).hexdigest(),
              'state_component_module_sha256': hashlib.sha256(read_bytes(component_module.__file__)).hexdigest(),
              'input_identity': {k: meta[k] for k in ('episode_index', 'frame_index', 'global_index', 'timestamp', 'task_index')},
              'input_state': arrays['state'].tolist(), 'state_supplied': True,
              'row_witness': {'parquet_source_sha256': meta['parquet_source_sha256'],
                              'state_sha256': meta['state_sha256'], 'dtype': '<f4', 'shape': [8]},
              'statistics_state_dims': server._state_dims,
              'input_images': {s['model']: {'sha256': hashlib.sha256(arrays[s['model']].tobytes()).hexdigest(),
                                          'association': 'explicit', 'supplied': s['model'] == 'image' or supplied,
                                          'source_key': s['source']} for s in STREAMS},
              'native_request_params': params,
              'policy_loader': 'explicitly substituted; no weights',
              'policy_sink': 'capture+generated zero action return; not inference',
              'normalization_scope': 'optional separate actual native StateActionProcessor on captured named state',
              'native_processing_status': 'unknown'}
    try:
        server.predict_batch([{'images': images, 'states': arrays['state'], 'task_description': meta['task']}],
                             [SessionContext('saved-original', 'original-episode')])
        if not captured:
            raise ValueError('Native call did not produce a positive policy-call boundary capture')
        report['captured'] = {'state': {k: leaf(v, True) for k, v in captured['state'].items()},
                              'video': {k: leaf(v) for k, v in captured['video'].items()}}
        report['native_processing_status'] = 'complete'
    except Exception as exc:
        report['native_processing_status'] = 'error'
        report['exception'] = {'type': type(exc).__name__, 'message': str(exc)[:512]}
    try:
        if report['native_processing_status'] != 'complete':
            raise ValueError('No complete adapter boundary')
        processor = StateActionProcessor(mods, statistics={'libero_panda': stats},
                                         **{k: kwargs[k] for k in ('use_percentiles', 'clip_outliers', 'apply_sincos_state_encoding', 'use_relative_action')})
        processor.eval()
        processed = processor.apply_state(captured['state'], 'libero_panda')
        report['native_state_component'] = {'status': 'complete', 'output': {k: leaf(v, True) for k, v in processed.items()}}
    except Exception as exc:
        report['native_state_component'] = {'status': 'error', 'exception': {'type': type(exc).__name__, 'message': str(exc)[:512]}}
    return report


def run(request):
    validate_request(request)
    kwargs, stats = validate_model(request['model'])
    output = Path(request['output'])
    for ancestor in (output, *output.parents):
        if ancestor.is_symlink():
            raise ValueError('Worker output cannot traverse symbolic links')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite an existing worker output')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    (output / 'samples').mkdir(mode=0o700)
    arrays, meta = _sample(request, output)
    report = _capture(request, arrays, meta, kwargs, stats)
    raw = _canonical(report)
    if len(raw) > MAX_JSON:
        raise ValueError('Native raw report exceeds bound')
    (output / 'result.json').write_bytes(raw)
    files = []
    for name in ('result.json', 'samples/arrays.npz', 'samples/metadata.json'):
        data = read_bytes(output / name, MAX_ARRAYS)
        files.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)})
    receipt = {'provider': 'groot_adapter', 'operation': request['operation'],
               'execution_status': report['native_processing_status'], 'result': files[0], 'payload_files': files}
    (output / 'worker_receipt.json').write_bytes(_canonical(receipt))
    return receipt


def main():
    request = None
    try:
        raw = sys.stdin.buffer.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            raise ValueError('Worker request exceeds bound')
        request = loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            receipt = run(request)
    except Exception as exc:
        receipt = {'provider': 'groot_adapter', 'execution_status': 'error', 'result': None,
                   'payload_files': [], 'native_execution_error': {'type': type(exc).__name__, 'message': str(exc)[:512]}}
    if receipt['execution_status'] == 'error' and isinstance(request, dict):
        # Preserve a bounded machine-readable failure for the fixed parent launcher.
        # Unsafe/nonabsolute outputs are never followed merely to report an error.
        candidate = request.get('output')
        if isinstance(candidate, str) and Path(candidate).is_absolute():
            output = Path(candidate)
            if not any(p.is_symlink() for p in (output, *output.parents)):
                output.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not (output / 'worker_receipt.json').exists():
                    (output / 'worker_receipt.json').write_bytes(_canonical(receipt))
    data = _canonical(receipt)
    if len(data) > 128 * 1024:
        raise ValueError('Worker completion exceeds bound')
    sys.stdout.buffer.write(data)


if __name__ == '__main__':
    main()
