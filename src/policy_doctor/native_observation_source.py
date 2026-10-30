"""Evaluate retained GR00T adapter input witnesses without running native code."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

import numpy as np

from .artifacts import InputError
from .rollouts import _npz

PARAMETER_KEYS = {'contract', 'model_identity'}
MAX_JSON = 2 * 1024**2
MAX_FILE = 128 * 1024**2
MAX_TOTAL = 256 * 1024**2
LAYOUT = ['x', 'y', 'z', 'axis_angle1', 'axis_angle2', 'axis_angle3', 'gripper', 'gripper']
STREAMS = [dict(input='agentview', model='image', source='observation.images.image'),
           dict(input='wrist', model='wrist_image', source='observation.images.wrist_image')]
CONTRACT = dict(kind='IPEC_LIBERO_selected_source_preservation', association='explicit',
                boundary='policy_get_action_input_pre_normalization', state_layout=LAYOUT,
                ordered_source_streams=STREAMS)
SCOPE = ('Conditional preservation of supplied source state and ordered camera API values before '
         'normalization. Retained hashes and typed decode witnesses establish consistency, not '
         'independent video re-decoding, author authenticity, physical camera identity, equal-value '
         'source flow, rotation semantics, training membership or policy competence. Policy loading '
         'is explicitly substituted; zero sink actions are generated, not predictions. No native '
         'code is executed by this audit.')


class _Missing(Exception):
    pass


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _relative(value):
    if (type(value) is not str or not value or len(value) > 1024 or '\\' in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise InputError('Retained paths must be bounded plain relative paths')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in ('', '.', '..') for p in value.split('/')):
        raise InputError('Unsafe retained relative path')
    return value


def _read(root, relative, limit=MAX_JSON, *, expected=None, size=None):
    """Walk no-follow directory descriptors and hash the same bounded open file."""
    relative = _relative(relative)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptors = []
    try:
        fd = os.open(root, flags | os.O_DIRECTORY)
        descriptors.append(fd)
        root_stat = os.fstat(fd)
        ancestors = [(root_stat.st_dev, root_stat.st_ino)]
        for part in relative.split('/')[:-1]:
            fd = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            descriptors.append(fd)
            current_directory = os.fstat(fd)
            ancestors.append((current_directory.st_dev, current_directory.st_ino))
        leaf = relative.split('/')[-1]
        fd = os.open(leaf, flags, dir_fd=fd)
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise InputError('Retained evidence must be a bounded regular file')
        if size is not None and before.st_size != size:
            raise InputError('Retained binding size mismatch')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        current = os.stat(leaf, dir_fd=descriptors[-2], follow_symlinks=False)
        def signature(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if len(raw) > limit or signature(before) != signature(after) or signature(before) != signature(current):
            raise InputError('Retained evidence changed during reading')
        # Held directory FDs alone can keep reading a detached tree after its
        # public name is replaced. Re-open the complete path without following
        # links and compare every ancestor before accepting the retained bytes.
        current_fd = os.open(root, flags | os.O_DIRECTORY)
        descriptors.append(current_fd)
        directory_stat = os.fstat(current_fd)
        if (directory_stat.st_dev, directory_stat.st_ino) != ancestors[0]:
            raise InputError('Capture root changed during reading')
        for index, part in enumerate(relative.split('/')[:-1], 1):
            current_fd = os.open(part, flags | os.O_DIRECTORY, dir_fd=current_fd)
            descriptors.append(current_fd)
            directory_stat = os.fstat(current_fd)
            if (directory_stat.st_dev, directory_stat.st_ino) != ancestors[index]:
                raise InputError('Retained evidence ancestor changed during reading')
        if signature(os.stat(leaf, dir_fd=current_fd, follow_symlinks=False)) != signature(before):
            raise InputError('Retained evidence path changed during reading')
        if expected is not None and (not _sha(expected) or _hash(raw) != expected):
            raise InputError('Retained evidence digest mismatch')
        return raw
    except OSError as exc:
        raise InputError(f'Unsafe or unavailable retained evidence: {relative}') from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _json(raw):
    if len(raw) > MAX_JSON:
        raise InputError('Evidence JSON exceeds audit budget')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError('Duplicate JSON object key')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(InputError('Non-finite JSON')))
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError('Invalid bounded evidence JSON') from exc


def validate_parameters(parameters):
    if type(parameters) is not dict or set(parameters) - PARAMETER_KEYS or 'contract' not in parameters:
        raise InputError('native_observation_source requires contract and optional model_identity')
    if parameters['contract'] != CONTRACT or type(parameters['contract']) is not dict:
        raise InputError('Unsupported explicit native observation source contract')
    identity = parameters.get('model_identity')
    if identity is not None and not _sha(identity):
        raise InputError('model_identity must be a SHA-256 content identity')
    result = {'contract': json.loads(json.dumps(CONTRACT))}
    if identity is not None:
        result['model_identity'] = identity
    return result


def _descriptor(root, value, limit):
    if type(value) is not dict or set(value) != {'path', 'sha256'} or not _sha(value.get('sha256')):
        raise InputError('Evidence descriptors require exact path and SHA-256')
    return _read(root, value['path'], limit, expected=value['sha256'])


def _capture(directory):
    root = Path(directory).absolute()
    if root.is_symlink() or not root.is_dir():
        raise InputError('Capture root must be a real directory')
    wrapper = _json(_read(root, 'capture.json'))
    if (type(wrapper) is not dict or wrapper.get('schema_version') != 1
            or type(wrapper.get('schema_version')) is not int
            or wrapper.get('kind') != 'native_observation_source_capture'
            or wrapper.get('evidence_role') != 'native_adapter_dry_run'
            or set(wrapper) != {'schema_version', 'kind', 'evidence_role', 'result', 'sample', 'contract', 'bindings'}):
        raise InputError('Invalid native observation source capture wrapper')
    raw = _json(_descriptor(root, wrapper['result'], MAX_JSON))
    sample = wrapper['sample']
    if type(sample) is not dict or set(sample) != {'arrays', 'metadata'}:
        raise InputError('Capture requires exact arrays/metadata sample descriptors')
    if (wrapper['result'].get('path') != 'raw/result.json'
            or type(sample.get('arrays')) is not dict or sample['arrays'].get('path') != 'samples/arrays.npz'
            or type(sample.get('metadata')) is not dict or sample['metadata'].get('path') != 'samples/metadata.json'):
        raise InputError('Capture requires fixed raw and sample evidence paths')
    arrays_raw = _descriptor(root, sample['arrays'], 16 * 1024**2)
    metadata = _json(_descriptor(root, sample['metadata'], MAX_JSON))
    _, arrays = _npz(arrays_raw, True)
    if set(arrays) != {'state', 'recorded_action', 'image', 'wrist_image'}:
        raise InputError('Native sample NPZ requires only state, recorded_action and both image streams')
    if type(metadata) is not dict or metadata.get('arrays_sha256') != _hash(arrays_raw):
        raise InputError('Sample metadata does not bind retained NPZ')
    if type(raw) is not dict:
        raise InputError('Raw capture must be an object')
    bindings = wrapper['bindings']
    if type(bindings) is not list or not bindings or len(bindings) > 64:
        raise InputError('Capture requires a bounded original binding closure')
    originals, retained, total = {}, set(), 0
    for binding in bindings:
        if (type(binding) is not dict or set(binding) != {'root', 'path', 'retained_path', 'sha256', 'size_bytes'}
                or binding.get('root') not in ('model', 'dataset') or not _sha(binding.get('sha256'))
                or type(binding.get('size_bytes')) is not int or not 0 < binding['size_bytes'] <= MAX_FILE):
            raise InputError('Invalid original binding receipt')
        key = (binding['root'], _relative(binding['path']))
        path = _relative(binding['retained_path'])
        if path != f'originals/{binding["root"]}/{binding["path"]}':
            raise InputError('Original bindings must retain their exact root-relative path')
        total += binding['size_bytes']
        if key in originals or path in retained or total > MAX_TOTAL:
            raise InputError('Duplicate or excessive retained original closure')
        retained.add(path)
        originals[key] = (_read(root, path, MAX_FILE, expected=binding['sha256'], size=binding['size_bytes']), binding)
    return wrapper, raw, metadata, arrays, originals


def _original(originals, kind, path):
    value = originals.get((kind, path))
    if value is None:
        raise _Missing(f'Missing original {kind} binding: {path}')
    return value[0]


def _number(value):
    return type(value) in (int, float) and abs(value) <= 2**53 and math.isfinite(value)


def _numeric_list(value, length):
    return type(value) is list and len(value) == length and all(_number(v) for v in value)


def _integer_list(value, expected):
    return type(value) is list and value == expected and all(type(item) is int for item in value)


def _identity(value):
    keys = ('episode_index', 'frame_index', 'global_index', 'task_index')
    if (type(value) is not dict or any(type(value.get(k)) is not int or not 0 <= value[k] <= 2**53 for k in keys)
            or not _number(value.get('timestamp')) or value['timestamp'] < 0):
        raise _Missing('Missing typed original row identity')
    return {k: value[k] for k in (*keys, 'timestamp')}


def _tensor(leaf, shape, dtype, *, values=True):
    if (type(leaf) is not dict or leaf.get('shape') != shape or leaf.get('dtype') != dtype
            or type(leaf.get('shape')) is not list or any(type(n) is not int for n in leaf['shape'])
            or not _sha(leaf.get('sha256'))):
        raise _Missing('Missing typed captured tensor witness')
    if not values:
        return leaf['sha256']
    value = leaf.get('values')
    if type(value) is not list or len(value) != math.prod(shape) or not all(_number(v) for v in value):
        raise _Missing('Missing complete finite captured tensor values')
    with np.errstate(over='ignore', invalid='ignore'):
        array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all() or _hash(array.tobytes()) != leaf['sha256']:
        raise _Missing('Captured typed tensor values do not match their digest')
    return array


def _jsonlines(raw):
    if len(raw) > MAX_JSON:
        raise InputError('Original metadata JSONL exceeds audit budget')
    lines = raw.splitlines()
    if len(lines) > 100000:
        raise InputError('Original metadata has excessive records')
    return [_json(line) for line in lines if line.strip()]


def _model(originals):
    config_raw = _original(originals, 'model', 'processor_config.json')
    stats_raw = _original(originals, 'model', 'statistics.json')
    if len(config_raw) > MAX_JSON or len(stats_raw) > MAX_JSON:
        raise InputError('Original model metadata exceeds audit budget')
    config, stats = _json(config_raw), _json(stats_raw)
    try:
        modality = config['processor_kwargs']['modality_configs']['libero_panda']
        groups = modality['state']['modality_keys']
        video = modality['video']['modality_keys']
        state_stats = stats['libero_panda']['state']
        if (groups != ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
                or video != ['image', 'wrist_image'] or not _integer_list(modality['state']['delta_indices'], [0])
                or not _integer_list(modality['video']['delta_indices'], [0])):
            raise _Missing('Unsupported original model modality order')
        widths = {}
        for name in groups:
            group = state_stats[name]
            if type(group) is not dict or not group:
                raise _Missing('Missing original statistics-derived group widths')
            width = None
            for values in group.values():
                if type(values) is not list or not values or not all(_number(v) for v in values):
                    raise _Missing('Invalid original state statistics')
                if width is not None and len(values) != width:
                    raise _Missing('Contradictory original state statistics widths')
                width = len(values)
            widths[name] = width
        if list(widths.values()) != [1, 1, 1, 1, 1, 1, 2]:
            raise _Missing('Unsupported original state group widths')
    except (KeyError, TypeError, AttributeError) as exc:
        raise _Missing('Missing original model modality/statistics evidence') from exc
    identity = _hash(_canonical({'processor_config.json': _hash(config_raw), 'statistics.json': _hash(stats_raw)}))
    return identity, widths


def audit_native_observation_source(directory, *, contract=None, model_identity=None):
    wrapper, raw, metadata, arrays, originals = _capture(directory)
    parameters = None if contract is None and model_identity is None else validate_parameters(
        {'contract': wrapper['contract'] if contract is None else contract, 'model_identity': model_identity})
    report = dict(kind='native_observation_source_audit', status='unknown', input_digest=None,
                  native_identity=None, model_identity=None, native_processing_status='unknown',
                  native_state_component_status='unknown', findings=[], finding_count=0, findings_omitted=0,
                  scope=SCOPE, original_authenticity_status='unknown', native_replay_status='unknown')

    def finish(status, reason, findings=None):
        report['status'] = status
        report['source_criterion'] = dict(status=status, code={'error': 'source_contradiction', 'pass': 'source_preserved',
                                                            'unknown': 'insufficient_evidence'}[status], reason=reason)
        report['findings'] = findings or []
        report['finding_count'] = len(report['findings'])
        return report
    try:
        if wrapper['contract'] != CONTRACT or raw.get('contract') != CONTRACT:
            raise _Missing('Missing supported explicit source-preservation association')
        if parameters is not None and parameters['contract'] != wrapper['contract']:
            raise _Missing('Current capture does not match frozen explicit contract')
        identity, widths = _model(originals)
        report['model_identity'] = identity
        if parameters is not None and parameters.get('model_identity', identity) != identity:
            raise _Missing('Original model metadata differs from the frozen criterion')
        native_identity = _identity(raw.get('input_identity'))
        if native_identity != _identity(metadata):
            raise _Missing('Raw current identity does not match retained sample identity')
        report['native_identity'] = native_identity
        state = arrays['state']
        if (state.dtype.str != '<f4' or state.shape != (8,) or arrays['recorded_action'].dtype.str != '<f4'
                or arrays['recorded_action'].shape != (7,)):
            raise _Missing('Missing supported typed original state/action sample')
        state_sha = _hash(state.tobytes())
        if (metadata.get('state_dtype') != '<f4' or metadata.get('state_shape') != [8]
                or not _numeric_list(metadata.get('state_values'), 8) or not _numeric_list(raw.get('input_state'), 8)
                or metadata.get('state_sha256') != state_sha or metadata.get('state_values') != state.tolist()
                or raw.get('input_state') != state.tolist()):
            raise _Missing('Original state summaries do not bind current retained values')
        info = _json(_original(originals, 'dataset', 'meta/info.json'))
        state_feature = info.get('features', {}).get('observation.state', {}) if type(info) is dict else {}
        if (type(info) is not dict or info.get('codebase_version') != 'v2.1'
                or state_feature.get('names', {}).get('motors') != LAYOUT
                or state_feature.get('dtype') != 'float32' or not _integer_list(state_feature.get('shape'), [8])
                or type(info.get('chunks_size')) is not int or info['chunks_size'] <= 0
                or info.get('data_path') != 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
                or info.get('video_path') != 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'):
            raise _Missing('Missing supported original dataset layout')
        ep = native_identity['episode_index']
        frame = native_identity['frame_index']
        episodes = _jsonlines(_original(originals, 'dataset', 'meta/episodes.jsonl'))
        tasks = _jsonlines(_original(originals, 'dataset', 'meta/tasks.jsonl'))
        matched = [r for r in episodes if type(r) is dict and type(r.get('episode_index')) is int and r['episode_index'] == ep]
        task = [r for r in tasks if type(r) is dict and type(r.get('task_index')) is int
                and r['task_index'] == native_identity['task_index']]
        if (len(matched) != 1 or type(matched[0].get('length')) is not int or not frame < matched[0]['length']
                or len(task) != 1 or task[0].get('task') != metadata.get('task')
                or type(metadata.get('task')) is not str or not metadata['task']
                or type(matched[0].get('tasks')) is not list or not matched[0]['tasks']
                or any(type(value) is not str or not value for value in matched[0]['tasks'])
                or metadata.get('task') not in matched[0].get('tasks', [])):
            raise _Missing('Missing positive episode/task row association')
        try:
            substitutions = dict(episode_index=ep, episode_chunk=ep // info['chunks_size'])
            parquet_path = _relative(info['data_path'].format(**substitutions))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise _Missing('Unsupported original dataset path recipe') from exc
        parquet = _original(originals, 'dataset', parquet_path)
        witness = raw.get('row_witness')
        if (type(witness) is not dict or witness.get('parquet_source_sha256') != _hash(parquet)
                or witness.get('state_sha256') != state_sha or witness.get('dtype') != '<f4' or witness.get('shape') != [8]
                or type(witness.get('shape')) is not list or any(type(n) is not int for n in witness['shape'])):
            raise _Missing('Missing typed row witness bound to retained original Parquet')
        if ('parquet_source_sha256' in metadata and metadata['parquet_source_sha256'] != _hash(parquet)
                or 'parquet_relative_path' in metadata and metadata['parquet_relative_path'] != parquet_path):
            raise _Missing('Retained sample row receipt contradicts original Parquet association')
        cameras = metadata.get('cameras')
        if type(cameras) is not dict:
            raise _Missing('Missing original decoded camera association')
        image_digests = {}
        for stream in STREAMS:
            name, source = stream['model'], stream['source']
            image, camera = arrays[name], cameras.get(name)
            if image.dtype.str != '|u1' or image.shape != (256, 256, 3) or type(camera) is not dict:
                raise _Missing('Missing supported decoded RGB image witness')
            digest = _hash(image.tobytes())
            if (camera.get('pixel_dtype') != '|u1' or camera.get('pixel_shape') != [256, 256, 3]
                    or camera.get('pixel_sha256') != digest or camera.get('source_key') != source
                    or type(camera.get('ordinal')) is not int or camera['ordinal'] < 0
                    or type(camera.get('pts')) is not int or not 0 <= camera['pts'] <= 2**53
                    or not _number(camera.get('time_seconds'))
                    or abs(camera['time_seconds'] - native_identity['timestamp']) > 1e-5
                    or type(camera.get('time_base')) is not str
                    or re.fullmatch(r'[1-9][0-9]{0,9}/[1-9][0-9]{0,9}', camera['time_base']) is None):
                raise _Missing('Missing typed video timestamp/source association')
            numerator, denominator = map(int, camera['time_base'].split('/'))
            if abs(camera['pts'] * numerator / denominator - camera['time_seconds']) > 1e-5:
                raise _Missing('Video PTS does not match declared decode time')
            try:
                video_path = _relative(info['video_path'].format(**substitutions, video_key=source))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise _Missing('Unsupported original video path recipe') from exc
            video_bytes = _original(originals, 'dataset', video_path)
            if (camera.get('source_relative_path') != video_path
                    or camera.get('source_sha256') != _hash(video_bytes)):
                raise _Missing('Decoded camera witness does not bind the retained original video bytes')
            # Host paths are capture assertions, never opened or followed.
            source_file = camera.get('source_file')
            if type(source_file) is not str or not source_file.endswith('/' + video_path):
                raise _Missing('Decoded video witness does not identify the retained stream path')
            feature = info.get('features', {}).get(source, {})
            if feature.get('dtype') != 'video' or feature.get('shape') != [256, 256, 3]:
                raise _Missing('Original dataset does not declare the decoded video stream')
            image_digests[name] = digest
        report['input_digest'] = _hash(_canonical(dict(native_identity=native_identity, state=state_sha, images=image_digests)))
        if raw.get('embodiment') != 'LIBERO_PANDA' or raw.get('state_supplied') is not True:
            raise _Missing('No positive original state supplied for the supported embodiment')
        if (not _sha(raw.get('source_module_sha256'))
                or raw.get('policy_loader') != 'explicitly substituted; no weights'
                or raw.get('policy_sink') != 'capture+generated zero action return; not inference'):
            raise _Missing('Missing actual adapter source or explicit dry-run substitution witness')
        captured = raw.get('captured')
        images = raw.get('input_images')
        if (type(captured) is not dict or type(captured.get('state')) is not dict
                or type(captured.get('video')) is not dict or type(images) is not dict
                or raw.get('statistics_state_dims') != widths
                or type(raw.get('statistics_state_dims')) is not dict
                or any(type(v) is not int for v in raw['statistics_state_dims'].values())):
            raise _Missing('Missing actual named state/video payload and original group-width association')
        findings, offset = [], 0
        for name, width in widths.items():
            actual = _tensor(captured['state'].get(name), [1, 1, width], '<f4')
            expected = state[offset:offset + width]
            offset += width
            if not np.array_equal(actual, expected):
                findings.append(dict(kind='state_value_not_preserved_under_declared_layout', group=name,
                                     expected_original_values=expected.tolist(), actual_values=actual.tolist()))
        for stream in STREAMS:
            name = stream['model']
            supplied = images.get(name)
            if (type(supplied) is not dict or supplied.get('association') != 'explicit'
                    or type(supplied.get('supplied')) is not bool or supplied.get('source_key') != stream['source']
                    or supplied.get('sha256') != image_digests[name]):
                raise _Missing('Missing positive supplied/omitted original camera association: ' + name)
            leaf = captured['video'].get(name)
            if type(leaf) is not dict or leaf.get('dtype') != '|u1' or not _sha(leaf.get('sha256')):
                raise _Missing('Missing positive actual native image tensor: ' + name)
            shape = leaf.get('shape')
            if (type(shape) is not list or len(shape) != 5 or any(type(n) is not int or not 0 < n <= 4096 for n in shape)
                    or shape[:2] != [1, 1] or shape[-1] != 3):
                raise _Missing('Missing complete supported native image tensor shape')
            if not supplied['supplied'] or leaf['sha256'] != image_digests[name] or shape != [1, 1, 256, 256, 3]:
                findings.append(dict(kind='camera_value_or_supplied_view_requirement_not_met', camera=name,
                                     supplied=supplied['supplied'], actual_shape=shape))
        report['native_processing_status'] = 'complete'
        component = raw.get('native_state_component')
        if type(component) is dict and type(component.get('output')) is dict:
            try:
                for name, width in widths.items():
                    _tensor(component['output'].get(name), [1, 1, width], '<f4')
                report['native_state_component_status'] = 'complete'
            except _Missing:
                pass
        return finish('error' if findings else 'pass',
                      'Supported pre-normalization source-preservation contradiction' if findings else
                      'Captured native input preserves the explicitly required original API values', findings)
    except _Missing as exc:
        return finish('unknown', str(exc))
    except InputError:
        raise
    except (KeyError, TypeError, AttributeError, OverflowError, ValueError):
        return finish('unknown', 'Malformed or unsupported positive observation witness')


def bind_native_observation_source_parameters(directory, parameters=None):
    wrapper, _, _, _, _ = _capture(directory)
    result = validate_parameters({'contract': wrapper['contract']} if parameters is None else parameters)
    report = audit_native_observation_source(directory, **result)
    if not _sha(report.get('model_identity')):
        raise InputError('Cannot bind criterion without complete original model metadata')
    if 'model_identity' in result and result['model_identity'] != report['model_identity']:
        raise InputError('Supplied model identity does not match retained metadata')
    result['model_identity'] = report['model_identity']
    return result
