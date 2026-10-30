"""Resolve bounded exact native dependencies from pre-staged metadata only."""

import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath

MAX_JSON = 2 * 1024**2
MAX_REQUEST = 128 * 1024
MAX_EPISODE_FILES = 128
MAX_ENTRIES = 1024


def _relative(value):
    if type(value) is not str or not value or len(value) > 1024 or '\\' in value:
        raise ValueError('Invalid native dependency path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')) or ':' in value:
        raise ValueError('Native dependency must be a contained relative path')
    return path.as_posix()


def _within(root, relative):
    relative = _relative(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Native metadata must not contain symlinks')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Native metadata escapes its root')
    return path


def _json(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise ValueError('Native metadata JSON exceeds regular-file bounds')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate native metadata JSON key')
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 100_000:
            raise ValueError('Native metadata JSON exceeds structure bounds')
        if type(item) is dict:
            if set(item) & {'auto_map', 'class_path', 'module', 'module_path', '_target_'}:
                raise ValueError('Artifact code references are unsupported')
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError('Non-finite native metadata number')
    return value


def _integer(value, label, maximum=100_000_000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Invalid native ' + label)
    return value


def _samples(value):
    if type(value) is not list or not 1 <= len(value) <= 2:
        raise ValueError('Plan requires one or two native frame selectors')
    seen = set()
    for sample in value:
        if type(sample) is not dict or set(sample) != {'episode_index', 'frame_index'}:
            raise ValueError('Plan requires explicit native episode/frame selectors')
        pair = (_integer(sample['episode_index'], 'episode'), _integer(sample['frame_index'], 'frame'))
        if pair in seen:
            raise ValueError('Duplicate native sample selector')
        seen.add(pair)
    return value


def _episode_files(dataset):
    root = _within(dataset, 'meta/episodes')
    if not root.is_dir():
        raise ValueError('Native episode metadata is missing')
    files, entries = [], 0
    with os.scandir(root) as directories:
        for directory in directories:
            entries += 1
            if entries > MAX_ENTRIES or directory.is_symlink():
                raise ValueError('Native metadata discovery exceeds safe entry bounds')
            if not directory.is_dir(follow_symlinks=False):
                continue
            with os.scandir(directory.path) as children:
                for child in children:
                    entries += 1
                    if entries > MAX_ENTRIES or child.is_symlink():
                        raise ValueError('Native metadata discovery exceeds safe entry bounds')
                    if child.name.endswith('.parquet'):
                        if not child.is_file(follow_symlinks=False):
                            raise ValueError('Native episode Parquet must be a regular file')
                        files.append(Path(child.path).relative_to(dataset).as_posix())
                        if len(files) > MAX_EPISODE_FILES:
                            raise ValueError('Native episode metadata exceeds file bounds')
    if not files:
        raise ValueError('Native episode metadata is missing')
    return sorted(files)


def _guard(paths):
    # Fixed installed sibling; no artifact-supplied module or import path.
    spec = importlib.util.spec_from_file_location('_policy_doctor_parquet_guard',
                                                Path(__file__).with_name('native_parquet_guard.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run({'operation': 'guard_parquet', 'paths': [str(path) for path in paths]})


def _load_native_metadata(source, dataset, output):
    runtime = output / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update({'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_CACHE': str(runtime / 'datasets-cache'),
                       'HF_HUB_CACHE': str(runtime / 'hub'), 'CUDA_VISIBLE_DEVICES': '',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    import_root = source if (source / 'lerobot/__init__.py').is_file() else source.parent
    if not (import_root / 'lerobot/__init__.py').is_file():
        raise ValueError('Trusted source must contain the native lerobot package')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(import_root))
    from lerobot.datasets import io_utils
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    source_path = Path(inspect.getsourcefile(LeRobotDatasetMetadata)).resolve()
    if not source_path.is_relative_to(source):
        raise ValueError('Native metadata import is outside the trusted source')
    metadata = LeRobotDatasetMetadata('native-local-plan', root=dataset, force_cache_sync=False, token=False)
    symbols = [LeRobotDatasetMetadata, type(metadata.info), io_utils.load_info, io_utils.load_tasks,
               io_utils.load_stats, io_utils.load_episodes, io_utils.load_nested_dataset]
    return metadata, symbols


def _request_paths(request):
    if (type(request) is not dict or set(request) != {'source_root', 'dataset', 'samples', 'output'}
            or any(type(request[key]) is not str or not request[key] or len(request[key]) > 4096
                   for key in ('source_root', 'dataset', 'output'))):
        raise ValueError('Unsupported fixed native metadata plan request')
    _samples(request['samples'])
    dataset, source, output = (Path(request[key]).absolute() for key in ('dataset', 'source_root', 'output'))
    for path in (dataset, output):
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError('Native plan paths must not contain symlinks')
    if dataset.is_symlink() or source.is_symlink() or not dataset.is_dir() or not source.is_dir():
        raise ValueError('Native plan roots must be regular existing directories')
    dataset, source = dataset.resolve(), source.resolve()
    if (any(output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()) for root in (dataset, source))
            or (output.exists() and (not output.is_dir() or any(output.iterdir())))):
        raise ValueError('Native plan output must be an empty separate directory')
    return dataset, source, output


def run(request):
    dataset, source, output = _request_paths(request)
    samples = request['samples']
    info = _json(_within(dataset, 'meta/info.json'))
    if type(info) is not dict or info.get('storage_format', 'parquet') not in (None, 'parquet'):
        raise ValueError('Only native parquet metadata planning is supported')
    if info.get('data_path') != 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet':
        raise ValueError('Unsupported native data path template')
    if info.get('video_path') not in (None, 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'):
        raise ValueError('Unsupported native video path template')
    total_episodes = _integer(info.get('total_episodes'), 'episode count', 100_000)
    total_tasks = _integer(info.get('total_tasks'), 'task count', 100_000)
    if not total_episodes or any(sample['episode_index'] >= total_episodes for sample in samples):
        raise ValueError('Requested native episode is outside metadata bounds')
    features = info.get('features')
    if (type(features) is not dict or not 1 <= len(features) <= 32
            or any(type(key) is not str or not re.fullmatch(r'[A-Za-z0-9_.]+', key)
                   or type(value) is not dict for key, value in features.items())):
        raise ValueError('Unsupported native feature inventory')
    for feature in features.values():
        if feature.get('dtype') not in {'video', 'image', 'float32', 'float64', 'int32', 'int64', 'bool', 'string'}:
            raise ValueError('Unsupported native feature dtype')
        shape = feature.get('shape')
        if (type(shape) is not list or not 1 <= len(shape) <= 4
                or any(type(value) is not int or not 1 <= value <= 4096 for value in shape)
                or math.prod(shape) > 4 * 1024**2):
            raise ValueError('Unsupported native feature dimensions')
    metadata_files = ['meta/info.json']
    parquet_files = []
    if total_tasks:
        tasks = _within(dataset, 'meta/tasks.parquet')
        if not tasks.is_file():
            raise ValueError('Required native task metadata is missing')
        metadata_files.append('meta/tasks.parquet')
        parquet_files.append(tasks)
    episodes = _episode_files(dataset)
    metadata_files.extend(episodes)
    parquet_files.extend(_within(dataset, path) for path in episodes)
    stats = _within(dataset, 'meta/stats.json')
    if stats.exists():
        _json(stats)
        metadata_files.append('meta/stats.json')
    guard = _guard(parquet_files)
    metadata, symbols = _load_native_metadata(source, dataset, output)
    if Path(metadata.root).resolve() != dataset or metadata.episodes is None:
        raise ValueError('Native metadata did not load the exact staged root')
    if len(metadata.episodes) != total_episodes:
        raise ValueError('Native episode table is incomplete or inconsistent')
    files, resolved = {}, []
    cameras = metadata.video_keys
    if (type(cameras) is not list or len(cameras) > 8
            or any(type(key) is not str or not re.fullmatch(r'[A-Za-z0-9_.]+', key) for key in cameras)):
        raise ValueError('Unsupported native camera inventory')
    for sample in samples:
        episode, frame = sample['episode_index'], sample['frame_index']
        row = metadata.episodes[episode]
        if type(row) is not dict or _integer(row.get('episode_index'), 'episode identity') != episode:
            raise ValueError('Native positional episode identity differs from the request')
        length = _integer(row.get('length'), 'episode length')
        if not length or frame >= length:
            raise ValueError('Requested native frame is outside its original episode')
        for key in ('data/chunk_index', 'data/file_index'):
            _integer(row.get(key), 'data file index')
        data_path = _relative(str(metadata.get_data_file_path(episode)))
        if not data_path.startswith('data/') or not data_path.endswith('.parquet'):
            raise ValueError('Native data dependency has an unsupported role/path')
        files[data_path] = 'native_data'
        videos = {}
        for key in cameras:
            for suffix in ('chunk_index', 'file_index'):
                _integer(row.get(f'videos/{key}/{suffix}'), 'video file index')
            video_path = _relative(str(metadata.get_video_file_path(episode, key)))
            if not video_path.startswith('videos/') or not video_path.endswith('.mp4'):
                raise ValueError('Native video dependency has an unsupported role/path')
            files[video_path] = 'native_video'
            videos[key] = video_path
        resolved.append({**sample, 'episode_length': length, 'data_path': data_path, 'video_paths': videos})
    sources = {}
    for symbol in symbols:
        path = Path(inspect.getsourcefile(symbol)).resolve()
        if not path.is_relative_to(source) or path.is_symlink() or path.stat().st_size > MAX_JSON:
            raise ValueError('Native planner source is outside bounded trusted code')
        relative = path.relative_to(source).as_posix()
        sources[relative] = {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'size_bytes': path.stat().st_size}
    return {'provider': 'lerobot', 'operation': 'plan', 'execution_status': 'complete',
            'files': [{'path': path, 'role': role} for path, role in sorted(files.items())],
            'metadata_files': sorted(metadata_files), 'samples': resolved,
            'source_files': sorted(sources.values(), key=lambda item: item['path']),
            'parquet_guard': {'decoded_bound_bytes': guard['decoded_bound_bytes']}}


def main():
    request = None
    output = None
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(raw) > MAX_REQUEST:
            raise ValueError('Native plan request exceeds byte bounds')
        request = json.loads(raw)
        _, _, output = _request_paths(request)
        output.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
    except Exception as error:
        result = {'provider': 'lerobot', 'operation': 'plan', 'execution_status': 'error',
                  'error': {'type': type(error).__name__, 'message': str(error)[:2000]}}
    if output is not None:
        (output / 'result.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print(json.dumps({'execution_status': result['execution_status']}))
    return 0 if result['execution_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
