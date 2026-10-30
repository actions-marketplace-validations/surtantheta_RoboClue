"""Bounded exact-byte dependency views of ordinary native artifact folders.

Only fixed metadata and native-planned files are selected. This is file-granular
capture, not a whole-root inventory or an alternative native dataset reader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

from .artifacts import InputError

MAX_FILE = 128 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_FILES = 256
MAX_STATE = 32 * 1024**2
MAX_NOTICE = 256 * 1024
MAX_NOTICES = 1024**2
NOTICE_NAMES = ('README.md', 'MODEL_CARD.md', 'DATASET_CARD.md', 'LICENSE', 'LICENSE.md',
                'LICENSE.txt', 'LICENCE', 'NOTICE', 'NOTICE.txt', 'COPYING')
MODEL_SEEDS = {'lerobot': ('config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'),
               'groot_state': ('processor_config.json', 'statistics.json')}


def _relative(value):
    if (not isinstance(value, str) or not value or len(value) > 512 or '\\' in value or ':' in value
            or Path(value).is_absolute() or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise InputError('Selected dependencies require safe contained relative paths')
    return value


def _stamp(info):
    return {name: getattr(info, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def _directory_fd(path):
    """Open every absolute ancestor without following links, including the root."""
    path = Path(path).expanduser().absolute()
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_file(root, relative):
    parts = _relative(relative).split('/')
    parent = _directory_fd(root)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise InputError('Selected dependencies must be regular files')
        return fd, info
    finally:
        os.close(parent)


def _entry(root, relative):
    """Stat a fixed contained candidate, never enumerate its unrelated siblings."""
    parts = _relative(relative).split('/')
    parent = _directory_fd(root)
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            except FileNotFoundError:
                return None
            os.close(parent)
            parent = child
        try:
            info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise InputError('Selected metadata/notice symlink refused')
        return info
    finally:
        os.close(parent)


def _list_directory(root, relative, maximum):
    fd = _directory_fd(Path(root) / _relative(relative))
    try:
        result = []
        with os.scandir(fd) as entries:
            for entry in entries:
                if len(result) >= maximum:
                    raise InputError('Metadata directory entry bound exceeded')
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise InputError('Metadata directory symlink refused')
                result.append((entry.name, info.st_mode))
        return sorted(result)
    finally:
        os.close(fd)


def _episode_metadata(root):
    if _entry(root, 'meta/episodes') is None:
        return []
    first = _list_directory(root, 'meta/episodes', 1024)
    count, selected = len(first), []
    for name, mode in first:
        if not stat.S_ISDIR(mode):
            if not stat.S_ISREG(mode):
                raise InputError('Unsupported native metadata entry')
            continue
        children = _list_directory(root, 'meta/episodes/' + name, 1024 - count)
        count += len(children)
        for child, child_mode in children:
            if not stat.S_ISREG(child_mode):
                raise InputError('Native episode metadata exceeds supported two-level layout')
            if child.endswith('.parquet'):
                selected.append('meta/episodes/' + name + '/' + child)
                if len(selected) > 128:
                    raise InputError('Native episode metadata exceeds 128 shards')
    return sorted(selected)


def _membership(provider, roots):
    metadata = {'model': list(MODEL_SEEDS[provider])}
    if 'dataset' not in roots:
        pass
    elif provider == 'groot_state':
        metadata['dataset'] = ['meta/info.json']
        metadata['dataset'].append('meta/modality.json')
    else:
        metadata['dataset'] = ['meta/info.json']
        for optional in ('meta/tasks.parquet', 'meta/stats.json'):
            if _entry(roots['dataset'], optional) is not None:
                metadata['dataset'].append(optional)
        metadata['dataset'].extend(_episode_metadata(roots['dataset']))
    notices = {role: [name for name in NOTICE_NAMES if _entry(root, name) is not None]
               for role, root in roots.items()}
    return {'metadata': {k: sorted(v) for k, v in metadata.items()}, 'notices': notices}


def _hash_selected(root, relative, limit=MAX_FILE):
    fd, before = _open_file(root, relative)
    try:
        if before.st_size > limit:
            raise InputError('Selected native file exceeds byte bound: ' + relative)
        digest, count = hashlib.sha256(), 0
        while block := os.read(fd, 1024**2):
            count += len(block)
            if count > limit:
                raise InputError('Selected native file grew beyond byte bound')
            digest.update(block)
        after = os.fstat(fd)
        if _stamp(before) != _stamp(after) or count != before.st_size:
            raise InputError('Selected native source changed while reading')
        reopened, current = _open_file(root, relative)
        os.close(reopened)
        if _stamp(current) != _stamp(before):
            raise InputError('Selected native path changed while reading')
        return digest.hexdigest(), _stamp(before)
    finally:
        os.close(fd)


def _copy_selected(selection, root_name, relative, role, limit=MAX_FILE):
    relative = _relative(relative)
    for existing in selection['files']:
        if existing['root'] == root_name and existing['path'] == relative:
            return existing
    if len(selection['files']) >= MAX_FILES:
        raise InputError('Selected native closure exceeds 256 files')
    root = Path(selection['roots'][root_name])
    fd, before = _open_file(root, relative)
    destination = Path(selection['staged_roots'][root_name]) / relative
    try:
        if before.st_size > limit or selection['total_bytes'] + before.st_size > MAX_TOTAL:
            raise InputError('Selected native closure exceeds its file/total byte bound: ' + relative)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        digest, count = hashlib.sha256(), 0
        try:
            while block := os.read(fd, 1024**2):
                count += len(block)
                if count > limit or selection['total_bytes'] + count > MAX_TOTAL:
                    raise InputError('Selected native source grew beyond capture byte bound')
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(target, view)
                    view = view[written:]
        finally:
            os.close(target)
        after = os.fstat(fd)
        if _stamp(before) != _stamp(after) or count != before.st_size:
            raise InputError('Selected native source changed during exact-byte copy')
        reopened, current = _open_file(root, relative)
        os.close(reopened)
        if _stamp(before) != _stamp(current):
            raise InputError('Selected native source path changed during copy')
        copied_hash, copied_stamp = _hash_selected(Path(selection['staged_roots'][root_name]), relative, limit)
        if copied_hash != digest.hexdigest() or copied_stamp['st_size'] != count:
            raise InputError('Selected native source/copy hash mismatch')
        item = {'root': root_name, 'path': relative, 'role': role, 'sha256': digest.hexdigest(),
                'size_bytes': count, 'source_stamp': _stamp(before), 'copy_stamp': copied_stamp}
        selection['files'].append(item)
        selection['total_bytes'] += count
        return item
    finally:
        os.close(fd)


def _validate_state(path):
    """Bound safetensors header, tensor spans and decoded size without loading tensors."""
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise InputError('Invalid processor safetensors header')
        length = int.from_bytes(prefix, 'little')
        if not 2 <= length <= 1024**2 or length + 8 > path.stat().st_size:
            raise InputError('Processor safetensors header exceeds bound')
        raw = stream.read(length)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate state header key')
            result[key] = value
        return result
    try:
        header = json.loads(raw, object_pairs_hook=pairs)
        if type(header) is not dict or len(header) > 257:
            raise ValueError('Unsupported state tensor inventory')
        widths = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2,
                  'I8': 1, 'U8': 1, 'BOOL': 1}
        spans = []
        payload = path.stat().st_size - length - 8
        for name, item in header.items():
            if name == '__metadata__':
                if type(item) is not dict or any(type(k) is not str or type(v) is not str for k, v in item.items()):
                    raise ValueError('Invalid state metadata')
                continue
            if type(item) is not dict or set(item) != {'dtype', 'shape', 'data_offsets'} or item['dtype'] not in widths:
                raise ValueError('Unsupported processor state tensor')
            shape, offsets = item['shape'], item['data_offsets']
            if (type(shape) is not list or len(shape) > 4 or any(type(v) is not int or not 0 <= v <= 10**7 for v in shape)
                    or type(offsets) is not list or len(offsets) != 2
                    or any(type(v) is not int for v in offsets) or not 0 <= offsets[0] <= offsets[1] <= payload):
                raise ValueError('Invalid processor state shape/offsets')
            size = math.prod(shape) * widths[item['dtype']]
            if size != offsets[1] - offsets[0] or size > MAX_STATE:
                raise ValueError('Processor state decoded allocation disagrees with bounded bytes')
            spans.append(offsets)
        end = 0
        for begin, stop in sorted(spans):
            if begin != end:
                raise ValueError('Processor state spans overlap or have holes')
            end = stop
        if end != payload:
            raise ValueError('Processor state payload is not fully described')
    except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
        raise InputError('Invalid bounded processor state: ' + str(exc)) from exc


def _model_dependencies(selection, model, model_family='xvla'):
    from .native_lerobot_worker import _features, _no_artifact_code

    if model_family == 'xvla':
        from . import native_lerobot_worker as validator
    elif model_family == 'smolvla':
        from . import native_processor_source_worker as validator
    else:
        raise InputError('Unsupported fixed native model family')
    step_fields = validator.STEP_FIELDS
    tokenizer_files = validator.TOKENIZER_FILES
    from .native_processing import _json

    config = _json(model / 'config.json')
    _no_artifact_code(config)
    if model_family == 'smolvla':
        if type(config) is not dict or config.get('type') != 'smolvla':
            raise InputError('Only the supported native SmolVLA config is accepted')
        for key in ('input_features', 'output_features'):
            _features(config.get(key, {}))
        origin = config.get('pretrained_path')
        if origin is not None and (type(origin) is not str or not origin or len(origin) > 512 or '\x00' in origin):
            raise InputError('Unsupported literal model source reference')
    else:
        if type(config) is not dict or config.get('type') != 'xvla' or config.get('tokenizer_name', 'facebook/bart-large') != 'facebook/bart-large':
            raise InputError('Only the supported native XVLA config/tokenizer is accepted')
        for key in ('input_features', 'output_features'):
            _features(config.get(key, {}))
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        pipeline = _json(model / filename)
        _no_artifact_code(pipeline)
        if type(pipeline) is not dict or set(pipeline) - {'name', 'steps'} or type(pipeline.get('steps')) is not list or not 1 <= len(pipeline['steps']) <= 16:
            raise InputError('Unsupported processor pipeline declaration')
        for entry in pipeline['steps']:
            if type(entry) is not dict or set(entry) - {'registry_name', 'config', 'state_file', 'artifacts'}:
                raise InputError('Only fixed native registry processor steps are supported')
            name, cfg = entry.get('registry_name'), entry.get('config', {})
            if name not in step_fields or type(cfg) is not dict or set(cfg) - step_fields[name]:
                raise InputError('Unsupported processor constructor/registry')
            if 'state_file' in entry:
                relative = _relative(entry['state_file'])
                if name not in {'normalizer_processor', 'unnormalizer_processor'} or not relative.endswith('.safetensors'):
                    raise InputError('Only declared normalization step state files are supported')
                _copy_selected(selection, 'model', relative, 'processor_state', MAX_STATE)
                _validate_state(model / relative)
            artifacts = entry.get('artifacts', {})
            if type(artifacts) is not dict or set(artifacts) - {'tokenizer_name'} or (artifacts and name != 'tokenizer_processor'):
                raise InputError('Unsupported processor artifact reference')
            if artifacts:
                token = artifacts.get('tokenizer_name')
                if token != cfg.get('tokenizer_name'):
                    raise InputError('Tokenizer artifact/config disagreement')
                token = _relative(token)
                files = _list_directory(Path(selection['roots']['model']), token, 16)
                if not files or any(name not in tokenizer_files or not stat.S_ISREG(mode) for name, mode in files):
                    raise InputError('Unsupported local tokenizer dependency tree')
                selection['tokenizer_membership'][token] = [name for name, _ in files]
                for name, _ in files:
                    relative = token + '/' + name
                    _copy_selected(selection, 'model', relative, 'tokenizer_asset', MAX_STATE)
                    if name.endswith('.json'):
                        asset = _json(model / relative, limit=MAX_STATE if model_family == 'smolvla' else 2 * 1024**2)
                        if model_family == 'xvla':
                            _no_artifact_code(asset)
                        # SmolVLA's fixed validator distinguishes vocabulary token
                        # dictionaries from actual tokenizer configuration hooks.
        validator._validate_pipeline(model, filename)
    if model_family == 'smolvla':
        validator.validate_model(model)


def _guard_parquets(paths, workspace, invoke):
    if not paths:
        return
    workspace.mkdir()
    response = invoke(Path(__file__).with_name('native_parquet_guard.py'), {'python': sys.executable},
                      {'operation': 'guard_parquet', 'paths': [str(path) for path in paths],
                       'output': str(workspace / 'output')}, workspace)
    expected = {str(path): path.stat().st_size for path in paths}
    if (type(response) is not dict or response.get('operation') != 'guard_parquet'
            or response.get('execution_status') != 'complete' or type(response.get('files')) is not list
            or len(response['files']) != len(expected)):
        raise InputError('Whole-file native Parquet allocation preflight did not complete')
    observed, total = set(), 0
    for item in response['files']:
        if (type(item) is not dict or type(item.get('path')) is not str or item['path'] not in expected
                or item['path'] in observed or type(item.get('size_bytes')) is not int
                or item['size_bytes'] != expected[item['path']]
                or type(item.get('decoded_bound_bytes')) is not int
                or not 0 <= item['decoded_bound_bytes'] <= 32 * 1024**2):
            raise InputError('Whole-file Parquet guard returned mismatched dependency membership/bounds')
        observed.add(item['path'])
        total += item['decoded_bound_bytes']
    if (observed != set(expected) or type(response.get('decoded_bound_bytes')) is not int
            or response['decoded_bound_bytes'] != total or total > 32 * 1024**2):
        raise InputError('Whole-file Parquet guard aggregate membership/allocation bound is invalid')


def _template_matches(template, path, video_key=None):
    pattern = re.escape(template)
    for field, width in (('chunk_index', 3), ('file_index', 3), ('episode_chunk', 3), ('episode_index', 6)):
        for spec in ('', f'0{width}d'):
            token = '{' + field + (':' + spec if spec else '') + '}'
            pattern = pattern.replace(re.escape(token), r'[0-9]{' + str(width if spec else 1) + ',}')
    pattern = pattern.replace(re.escape('{video_key}'), re.escape(video_key or ''))
    return re.fullmatch(pattern, path) is not None


def _planned_files(plan, samples, info, metadata):
    if (type(plan) is not dict or plan.get('provider') != 'lerobot' or plan.get('operation') != 'plan'
            or plan.get('execution_status') != 'complete' or type(plan.get('files')) is not list
            or not 1 <= len(plan['files']) <= 64 or type(plan.get('metadata_files')) is not list
            or any(type(path) is not str for path in plan['metadata_files'])
            or sorted(plan['metadata_files']) != sorted(metadata)
            or type(plan.get('samples')) is not list or len(plan['samples']) != len(samples)):
        raise InputError('Native metadata planner returned an invalid or incomplete dependency plan')
    files = {}
    for item in plan['files']:
        if type(item) is not dict or set(item) != {'path', 'role'} or item['role'] not in {'native_data', 'native_video'}:
            raise InputError('Native metadata planner returned an unsupported dependency role')
        relative = _relative(item['path'])
        if relative in files:
            raise InputError('Native plan must deduplicate selected dependency files')
        files[relative] = item['role']
    expected = {}
    if (type(info) is not dict or type(info.get('features')) is not dict
            or any(type(feature) is not dict for feature in info['features'].values())):
        raise InputError('Native planned feature metadata must be an object map')
    videos = {key for key, feature in info['features'].items() if feature.get('dtype') == 'video'}
    for requested, observed in zip(samples, plan['samples'], strict=True):
        if (type(observed) is not dict or any(type(observed.get(key)) is not int or observed[key] != value
                                             for key, value in requested.items())
                or type(observed.get('episode_length')) is not int or not 0 <= requested['frame_index'] < observed['episode_length']
                or type(observed.get('video_paths')) is not dict or set(observed['video_paths']) != videos):
            raise InputError('Native planner did not preserve selected episode/frame/camera identity')
        data = _relative(observed.get('data_path'))
        if not _template_matches(info['data_path'], data):
            raise InputError('Native planned data path disagrees with the supported original template')
        expected[data] = 'native_data'
        for key, path in observed['video_paths'].items():
            relative = _relative(path)
            if not _template_matches(info.get('video_path', ''), relative, key):
                raise InputError('Native planned video path disagrees with the supported original template')
            expected[relative] = 'native_video'
    if files != expected:
        raise InputError('Native planner files exceed or omit the selected native path closure')
    return files


def _capture_notices(selection, omissions):
    roots = {key: Path(value) for key, value in selection['roots'].items()}
    notice_bytes = 0
    for key, names in selection['seed_membership']['notices'].items():
        if not names:
            omissions.append(key + ': no fixed local notice/card candidate was present.')
        for name in names:
            info = _entry(roots[key], name)
            if info is None or not stat.S_ISREG(info.st_mode):
                raise InputError('Local notice candidate is not a regular file')
            if info.st_size > MAX_NOTICE or notice_bytes + info.st_size > MAX_NOTICES:
                omissions.append(key + '/' + name + ': notice exceeded bounded intake; content/hash not inspected.')
                continue
            _copy_selected(selection, key, name, 'local_notice', MAX_NOTICE)
            notice_bytes += info.st_size


def verify_native_inputs(selection):
    """Recheck only selected source/copy files and seed dependency membership."""
    try:
        roots = {key: Path(value) for key, value in selection['roots'].items()}
        if _membership(selection['provider'], roots) != selection['seed_membership']:
            raise InputError('Native seed metadata membership changed during capture')
        for token, expected in selection['tokenizer_membership'].items():
            if [name for name, _ in _list_directory(roots['model'], token, 16)] != expected:
                raise InputError('Selected tokenizer dependency membership changed')
        for item in selection['files']:
            original_hash, original_stamp = _hash_selected(roots[item['root']], item['path'])
            copied_hash, copied_stamp = _hash_selected(Path(selection['staged_roots'][item['root']]), item['path'])
            if (original_hash != item['sha256'] or copied_hash != item['sha256']
                    or original_stamp != item['source_stamp'] or copied_stamp != item['copy_stamp']):
                raise InputError('Selected native source/copy bytes or identity changed: ' + item['path'])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise InputError('Unable to verify selected native dependency capture: ' + str(exc)) from exc


def prepare_native_inputs(provider, model: Path, dataset: Path, samples: list,
                          workspace: Path, runtime: dict, invoke, *, model_family='xvla') -> dict:
    """Prepare fixed seeds and actual native-planned exact files, never sweep roots."""
    from .native_lerobot_worker import _no_artifact_code, _validate_dataset_info
    from .native_processing import _json, _samples, _validate_inputs

    try:
        if provider not in MODEL_SEEDS:
            raise InputError('Unsupported native dependency provider')
        if model_family not in {'xvla', 'smolvla'} or (provider != 'lerobot' and model_family != 'xvla'):
            raise InputError('Unsupported fixed native model family/provider')
        samples = _samples(samples)
        if provider == 'lerobot' and any('physical_row' in sample for sample in samples):
            raise InputError('LeRobot planning requires explicit native frame IDs')
        roots = {'model': Path(model).expanduser().absolute(), 'dataset': Path(dataset).expanduser().absolute()}
        for root in roots.values():
            os.close(_directory_fd(root))
        workspace = Path(workspace).absolute()
        os.close(_directory_fd(workspace))
        if any(workspace.is_relative_to(root) for root in roots.values()):
            raise InputError('Native staging workspace must be outside original input roots')
        view = workspace / 'input-view'
        view.mkdir(mode=0o700)
        staged = {key: view / key for key in roots}
        for root in staged.values():
            root.mkdir(mode=0o700)
        selection = {'provider': provider, 'roots': {key: str(value) for key, value in roots.items()},
                     'staged_roots': {key: str(value) for key, value in staged.items()}, 'files': [],
                     'total_bytes': 0, 'seed_membership': _membership(provider, roots), 'tokenizer_membership': {},
                     'model_family': model_family}
        for key, paths in selection['seed_membership']['metadata'].items():
            for relative in paths:
                _copy_selected(selection, key, relative, 'native_metadata')
                if relative.endswith('.json'):
                    _no_artifact_code(_json(staged[key] / relative))
        info = _json(staged['dataset'] / 'meta/info.json')
        if (type(info) is not dict or type(info.get('features')) is not dict
                or any(type(feature) is not dict for feature in info['features'].values())):
            raise InputError('Native dataset info/features must be JSON objects')
        if provider == 'lerobot':
            for key in ('total_tasks', 'total_episodes'):
                value = info.get(key)
                if type(value) is not int or not 0 <= value <= 10**9:
                    raise InputError('Native metadata requires bounded declared ' + key)
            metadata = selection['seed_membership']['metadata']['dataset']
            if info['total_tasks'] and 'meta/tasks.parquet' not in metadata:
                raise InputError('Declared native tasks metadata is missing')
            if info['total_episodes'] and not any(p.startswith('meta/episodes/') for p in metadata):
                raise InputError('Declared native episode metadata is missing')
            _model_dependencies(selection, staged['model'], model_family)
            _validate_dataset_info(staged['dataset'])
        else:
            processor_config = _json(staged['model'] / 'processor_config.json')
            if type(processor_config) is not dict:
                raise InputError('Original native GR1 processor configuration must be an object')
            kwargs = processor_config.get('processor_kwargs')
            if type(kwargs) is not dict or any(type(kwargs.get(k)) is not bool for k in
                    ('use_percentiles', 'clip_outliers', 'apply_sincos_state_encoding', 'use_relative_action')):
                raise InputError('Original native GR1 processor configuration is missing')
        # This old helper inventories only the newly selected private view.
        # It must never receive either ordinary original root here.
        _validate_inputs(provider, staged['model'], staged['dataset'])
        seed_parquets = [staged['dataset'] / item['path'] for item in selection['files']
                         if item['root'] == 'dataset' and item['path'].endswith('.parquet')]
        _guard_parquets(seed_parquets, workspace / 'metadata-guard', invoke)
        if provider == 'lerobot':
            plan_workspace = workspace / 'native-plan'
            plan_workspace.mkdir()
            plan = invoke(Path(__file__).with_name('native_lerobot_plan_worker.py'), runtime,
                          {'source_root': runtime['source_root'], 'dataset': str(staged['dataset']),
                           'samples': samples, 'output': str(plan_workspace / 'output')}, plan_workspace)
            files = _planned_files(plan, samples, info, selection['seed_membership']['metadata']['dataset'])
            selection['native_plan'] = plan
        else:
            chunk_size = info.get('chunks_size')
            if type(chunk_size) is not int or not 1 <= chunk_size <= 10**9:
                raise InputError('Original native GR1 episode chunks_size is missing')
            files = {info['data_path'].format(episode_chunk=s['episode_index'] // chunk_size,
                                             episode_index=s['episode_index']): 'native_data' for s in samples}
        for relative, role in files.items():
            _copy_selected(selection, 'dataset', relative, role)
        _guard_parquets([staged['dataset'] / path for path, role in files.items() if role == 'native_data'],
                        workspace / 'data-guard', invoke)
        omissions = [
            'Only the listed dependency files were inspected and copied; selection is file-granular.',
            'Selected Parquet/video files can contain other episodes, rows or frames; this is not observation-level redaction.',
            'Weight payloads, unselected data/media and unrelated files were not inspected. Omitted-file counts and hashes are unknown.',
            'A private exact-byte selected view is supplied to native code; original root membership beyond the selected closure is unknown.',
            'Local notices are evidence only; redistribution rights, native-source notices outside authorized roots and licensing resolution remain unknown.',
        ]
        _capture_notices(selection, omissions)
        verify_native_inputs(selection)
        receipts = [{key: item[key] for key in ('root', 'path', 'role', 'sha256', 'size_bytes')}
                    for item in selection['files']]
        return {'model': staged['model'], 'dataset': staged['dataset'], 'selection': selection,
                'receipts': receipts, 'omissions': omissions}
    except InputError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise InputError('Unable to prepare bounded native dependencies: ' + str(exc)) from exc


def prepare_native_model(model: Path, workspace: Path, *, model_family='smolvla') -> dict:
    """Stage one caller-authorized source bundle without dataset planning.

    Serialized source/repository identifiers remain inert metadata. The caller
    must authorize alternate source directories and revisions independently.
    """
    from .native_lerobot_worker import _no_artifact_code
    from .native_processing import _json

    try:
        if model_family not in {'xvla', 'smolvla'}:
            raise InputError('Unsupported fixed native model family')
        roots = {'model': Path(model).expanduser().absolute()}
        os.close(_directory_fd(roots['model']))
        workspace = Path(workspace).absolute()
        os.close(_directory_fd(workspace))
        if workspace.is_relative_to(roots['model']):
            raise InputError('Native staging workspace must be outside original source root')
        view = workspace / 'input-view'
        view.mkdir(mode=0o700)
        staged = view / 'model'
        staged.mkdir(mode=0o700)
        selection = {'provider': 'lerobot', 'model_family': model_family,
                     'roots': {'model': str(roots['model'])}, 'staged_roots': {'model': str(staged)},
                     'files': [], 'total_bytes': 0, 'seed_membership': _membership('lerobot', roots),
                     'tokenizer_membership': {}}
        for relative in selection['seed_membership']['metadata']['model']:
            _copy_selected(selection, 'model', relative, 'native_metadata')
            _no_artifact_code(_json(staged / relative))
        _model_dependencies(selection, staged, model_family)
        omissions = [
            'Only the listed model-source dependency files were inspected and copied; selection is file-granular.',
            'Weight payloads and unrelated files were not inspected. Omitted-file counts and hashes are unknown.',
            'Serialized repository/path identifiers were retained as data; no alternate source or revision was followed or authorized.',
            'No dataset metadata planning, native imports, processor execution or numerical qualification occurred during source intake.',
            'Local notices are evidence only; copying them establishes no redistribution permission.',
        ]
        _capture_notices(selection, omissions)
        verify_native_inputs(selection)
        receipts = [{key: item[key] for key in ('root', 'path', 'role', 'sha256', 'size_bytes')}
                    for item in selection['files']]
        return {'model': staged, 'selection': selection, 'receipts': receipts, 'omissions': omissions}
    except InputError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RecursionError, ImportError) as exc:
        raise InputError('Unable to prepare bounded native model source: ' + str(exc)) from exc
