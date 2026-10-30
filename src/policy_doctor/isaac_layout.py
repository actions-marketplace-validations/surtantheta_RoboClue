"""Capture cached IsaacLab observation evidence without importing its runtime.

Call this hook at the observation read point, passing the data object already
returned by the original sensor.data read. It never fetches sensor.data, reruns
the term, chooses body ordering or manufactures observed component labels.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import math
import sys
import threading
from pathlib import Path

import numpy as np

from .artifacts import InputError

REVISION = '32a1f330511130ae75ffeda96304c1bf569b5acb'
MAX_VALUES = 1_000_000
MAX_SOURCE_BYTES = 512 * 1024
MAX_SOURCE_TOTAL = 2 * 1024 * 1024
_ACTIVE_CALL = threading.local()


def _write_capture(output, metadata, arrays, *, sources):
    from .layouts import write_layout_capture

    return write_layout_capture(output, metadata, arrays, sources=sources)


def _text(value, label, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 256 or not value.isprintable():
        raise InputError(f'{label} must be a nonempty printable string of at most 256 characters')
    return value


def _native_shape(value, label, rank, allow_tensor_snapshot):
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif allow_tensor_snapshot and type(value).__module__.startswith('torch'):
        torch = importlib.import_module('torch')
        if not isinstance(value, torch.Tensor):
            raise InputError(f'{label} is not a supported native Torch tensor')
        shape = tuple(value.shape)
    else:
        raise InputError(f'{label} requires a NumPy array; explicit Torch snapshot opt-in is available')
    if len(shape) != rank or any(type(size) is not int or size <= 0 for size in shape):
        raise InputError(f'{label} must be nonempty rank-{rank} native data')
    return shape


def _environment_rows(value, count):
    if value is None:
        if count > 32:
            raise InputError('Source batch exceeds 32 environments; supply explicit env_indices (at most 32 row positions)')
        return list(range(count))
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 32 or
            any(type(index) is not int or not 0 <= index < count for index in value) or
            len(set(value)) != len(value)):
        raise InputError('env_indices must contain 1–32 distinct in-range native integer row positions')
    return list(value)


def _snapshot(value, label, rank, allow_tensor_snapshot, env_indices=None):
    """No np.asarray/duck conversion; explicit tensor copying is opt-in."""
    if value is None:
        return None
    _native_shape(value, label, rank, allow_tensor_snapshot)
    if env_indices is not None:
        value = value[env_indices]  # native selection before Torch CPU copy or NumPy snapshot
    if not isinstance(value, np.ndarray):
        if not allow_tensor_snapshot or not type(value).__module__.startswith('torch'):
            raise InputError(f'{label} requires a NumPy array; explicit Torch snapshot opt-in is available')
        try:
            torch = importlib.import_module('torch')
            if not isinstance(value, torch.Tensor):
                raise InputError(f'{label} is not a supported native Torch tensor')
            if (value.numel() > MAX_VALUES or len(value.shape) != rank or value.element_size() > 8 or
                    value.numel() * value.element_size() > 8 * MAX_VALUES):
                raise InputError(f'{label} exceeds bounded shape/value limits')
            value = value.detach().cpu().numpy()
        except (ImportError, TypeError, RuntimeError) as exc:
            raise InputError(f'{label} cannot be explicitly snapshotted as NumPy: {exc}') from exc
    if (value.ndim != rank or not value.size or value.size > MAX_VALUES or
            value.dtype.kind not in 'fiu' or value.dtype.itemsize > 8 or
            value.nbytes > 8 * MAX_VALUES or not np.isfinite(value).all()):
        raise InputError(f'{label} must be bounded, finite real numeric rank-{rank} data')
    return value.copy(order='C')


def _body_ids(value, count, allow_tensor_snapshot=False):
    if type(value).__module__.startswith('torch'):
        if not allow_tensor_snapshot:
            raise InputError('Resolved Torch body IDs require explicit allow_tensor_snapshot=True for bounded CPU copy')
        torch = importlib.import_module('torch')
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1 or
                not 1 <= value.numel() <= 256):
            raise InputError('Resolved Torch body IDs must be a bounded nonempty rank-1 torch.long vector')
        try:
            value = value.detach().cpu().tolist()
        except (TypeError, RuntimeError) as exc:
            raise InputError(f'Resolved Torch body IDs cannot be explicitly copied: {exc}') from exc
    if isinstance(value, slice):
        if any(v is not None and type(v) is not int for v in (value.start, value.stop, value.step)):
            raise InputError('Resolved body slice must use native integers')
        try:
            values = list(range(count))[value]
        except ValueError as exc:
            raise InputError('Resolved body slice is invalid') from exc
    elif type(value) is int:
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise InputError('Resolved body IDs must be native integers or a resolved slice')
    if (not values or len(values) > 256 or any(type(v) is not int or not 0 <= v < count for v in values)
            or len(set(values)) != len(values)):
        raise InputError('Resolved body IDs must be unique in-range native integers')
    return values


def _scale(value, width, allow_tensor_snapshot):
    if value is None:
        return None
    if type(value) in (int, float):
        try:
            if not math.isfinite(value):
                raise ValueError
        except (ValueError, OverflowError):
            raise InputError('Manager scale must be finite') from None
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != width or any(type(v) not in (int, float) for v in value):
            raise InputError('Manager scale vector must match captured raw width')
        try:
            if any(not math.isfinite(v) for v in value):
                raise ValueError
        except (ValueError, OverflowError):
            raise InputError('Manager scale vector must be finite') from None
        return list(value)
    if ((isinstance(value, np.ndarray) or type(value).__module__.startswith('torch'))
            and getattr(value, 'ndim', None) == 0):
        return _snapshot(value, 'Manager scalar scale', 0, allow_tensor_snapshot).item()
    array = _snapshot(value, 'Manager scale', 1, allow_tensor_snapshot)
    if len(array) != width:
        raise InputError('Manager scale vector must match captured raw width')
    return array.tolist()


def _source_identity(obj, sources, source_paths):
    module = _text(getattr(obj, '__module__', 'unknown'), 'Source module')
    qualname = _text(getattr(obj, '__qualname__', type(obj).__qualname__), 'Source qualified name')
    identity = {'module': module, 'qualname': qualname, 'source_file': None, 'sha256': None}
    try:
        filename = inspect.getsourcefile(obj)
        if filename is None:
            return identity
        path = Path(filename).resolve(strict=True)
        if path.stat().st_size > MAX_SOURCE_BYTES or not path.is_file():
            return identity
        raw = path.read_bytes()
        if len(raw) > MAX_SOURCE_BYTES:
            return identity
        if path not in source_paths:
            if len(sources) >= 16 or sum(map(len, sources.values())) + len(raw) > MAX_SOURCE_TOTAL:
                return identity
            key = f'source_{len(sources):02d}.py'
            sources[key] = raw
            source_paths[path] = key
        key = source_paths[path]
        identity.update(source_file=key, sha256=hashlib.sha256(sources[key]).hexdigest())
    except (OSError, TypeError, ValueError):
        pass
    return identity


def _license_source():
    filename = 'LICENSE-ISAACLAB-BSD-3-CLAUSE'
    paths = [Path(__file__).resolve().parents[2] / filename]
    try:
        dist = importlib.metadata.distribution('roboclue')
        paths.extend(Path(dist.locate_file(item)) for item in (dist.files or []) if item.name == filename)
    except importlib.metadata.PackageNotFoundError:
        pass
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(raw).hexdigest() == '6acf26ea8f8c42c454e30e9ad4206bd90d8fce85a54c011568faaf10c7ccab61':
            return raw
    raise InputError('Exact bundled IsaacLab BSD license is unavailable for portable source capture')


def _collect_capture(*, sensor, sensor_cfg, cached_data, raw_term_output, term_function,
                     term_cfg=None, processed_term_output=None, observation_slice=None,
                     allow_tensor_snapshot=False, cached_fields=None, env_indices=None):
    if type(allow_tensor_snapshot) is not bool:
        raise InputError('allow_tensor_snapshot must be a Boolean')
    if not inspect.isfunction(term_function):
        raise InputError('term_function must be the actual Python observation function')
    if observation_slice is not None:
        if (type(observation_slice) is not dict or len(observation_slice) > 8 or
                any(type(k) is not str or not k.isprintable() or len(k) > 64 for k in observation_slice) or
                any(v is not None and (type(v) not in (str, int) or
                    (type(v) is str and (len(v) > 256 or not v.isprintable())) or
                    (type(v) is int and abs(v) >= 2**63)) for v in observation_slice.values())):
            raise InputError('Observation slice must be a small explicit string/integer mapping')
        observation_slice = dict(observation_slice)
    names = sensor.body_names  # deliberately never sensor.data
    if not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 256:
        raise InputError('Live sensor.body_names must contain 1–256 names')
    names = [_text(name, 'Body name') for name in names]
    if len(set(names)) != len(names):
        raise InputError('Live sensor.body_names must be unique')
    ids = _body_ids(getattr(sensor_cfg, 'body_ids', None), len(names), allow_tensor_snapshot)
    preserve = getattr(sensor_cfg, 'preserve_order', False)
    if type(preserve) is not bool:
        raise InputError('Native preserve_order must be a Boolean')
    requested = getattr(sensor_cfg, 'body_names', None)
    if isinstance(requested, str):
        requested = [_text(requested, 'Requested body selector')]
    elif requested is not None:
        if not isinstance(requested, (list, tuple)) or len(requested) > 256:
            raise InputError('Requested native body selectors must be bounded strings')
        requested = [_text(name, 'Requested body selector') for name in requested]
    values = {'raw_term_output': raw_term_output, 'processed_term_output': processed_term_output}
    for label, field in [('source_force', 'force'), ('source_torque', 'torque')]:
        wrapper = (cached_fields.get(field) if cached_fields is not None else
                   getattr(cached_data, field, None))
        values[label] = None if wrapper is None else getattr(wrapper, 'torch', None)
    shapes = {key: _native_shape(value, key, 3 if key.startswith('source_') else 2,
                                  allow_tensor_snapshot)
              for key, value in values.items() if value is not None}
    counts = {shape[0] for shape in shapes.values()}
    if len(counts) != 1:
        raise InputError('Cached buffers and outputs must evidence one matching source environment count')
    source_count = counts.pop()
    if source_count > MAX_VALUES:
        raise InputError('Source environment count exceeds the supported 1,000,000-row bound')
    rows = _environment_rows(env_indices, source_count)
    for key, shape in shapes.items():
        if key.startswith('source_') and shape[1:] != (len(names), 3):
            raise InputError(f'{key} shape must match actual body_names and three native coordinates')
        if not key.startswith('source_') and shape[1] != len(ids) * 6:
            raise InputError('Output width does not match captured resolved body count')
    arrays = {}
    missing = []
    for label in ('source_force', 'source_torque'):
        array = _snapshot(values[label], label, 3, allow_tensor_snapshot, rows)
        if array is None:
            missing.append(label)
        else:
            if array.shape[1:] != (len(names), 3):
                raise InputError(f'{label} shape must match actual body_names and three native coordinates')
            arrays[label] = array
    if 'source_force' in arrays and 'source_torque' in arrays:
        if arrays['source_force'].shape != arrays['source_torque'].shape:
            raise InputError('Cached force and torque shapes differ')
    raw = _snapshot(raw_term_output, 'raw_term_output', 2, allow_tensor_snapshot, rows)
    if raw is None:
        missing.append('raw_term_output')
    else:
        if raw.shape[1] != len(ids) * 6:
            raise InputError('Raw output width does not match captured resolved body count')
        if any(array.shape[0] != raw.shape[0] for array in arrays.values()):
            raise InputError('Cached buffers and raw output have different environment counts')
        arrays['raw_term_output'] = raw
    processed = _snapshot(processed_term_output, 'processed_term_output', 2, allow_tensor_snapshot, rows)
    if processed is not None:
        if raw is not None and processed.shape != raw.shape:
            raise InputError('Processed term shape differs from raw term output')
        arrays['processed_term_output'] = processed
    namespace = getattr(getattr(sensor, 'cfg', None), 'prim_path', None)
    identity = None if namespace is None else {'prim_path': _text(namespace, 'Sensor prim_path')}
    asset = getattr(getattr(sensor, 'cfg', None), 'asset_name', None)
    if identity is not None and asset is not None:
        identity['asset_name'] = _text(asset, 'Sensor asset_name')
    sources, paths = {}, {}
    term_identity = _source_identity(term_function, sources, paths)
    cls = type(sensor)
    class_identity = _source_identity(cls, sources, paths)
    mro = [_source_identity(base, sources, paths) for base in cls.__mro__ if base is not object]
    config_identity = _source_identity(type(sensor_cfg), sources, paths)
    sources['LICENSE.txt'] = _license_source()
    width = len(ids) * 6
    scale = _scale(getattr(term_cfg, 'scale', None), width, allow_tensor_snapshot)
    absent = object()
    modifiers = getattr(term_cfg, 'modifiers', absent)
    clip = getattr(term_cfg, 'clip', absent)
    history = getattr(term_cfg, 'history_length', absent)
    noise = getattr(term_cfg, 'noise', absent)
    delay_values = [getattr(term_cfg, key, absent) for key in
                    ('delay_min_lag', 'delay_max_lag', 'delay_hold_prob')]
    metadata = {
        'schema_version': 1, 'kind': 'isaaclab_joint_wrench_layout',
        'source': {'repository': 'https://github.com/isaac-sim/IsaacLab', 'revision': REVISION,
                   'rule': 'body_incoming_wrench.force_torque_per_body.v1',
                   'term': term_identity, 'sensor_class': class_identity, 'sensor_mro': mro,
                   'sensor_config_class': config_identity,
                   'files': {key: {'sha256': hashlib.sha256(value).hexdigest()} for key, value in sources.items()}},
        'runtime': {'sensor_body_names': names, 'requested_body_names': requested,
                    'resolved_body_ids': ids, 'preserve_order': preserve, 'sensor_identity': identity,
                    'source_num_envs': source_count, 'env_indices': rows,
                    'omitted_env_count': source_count - len(rows),
                    'environment_scope': 'Selected source row positions; not temporal or physical environment identities.'},
        'processing': {'scale': scale, 'observation_slice': observation_slice,
                       'modifiers': 'unknown' if modifiers is absent else
                                    ([] if isinstance(modifiers, (list, tuple)) and not modifiers else
                                     ('present' if modifiers is not None else None)),
                       'clip': 'unknown' if clip is absent else (None if clip is None else 'present'),
                       'history': 'unknown' if history is absent else
                                  (history if type(history) is int and history == 0 else
                                   ('present' if history is not None else None)),
                       'noise': 'unknown' if noise is absent else (None if noise is None else 'present'),
                       'delay': 'unknown' if any(value is absent for value in delay_values) else
                                ('present' if any(delay_values) else None)},
        'provenance': {'origin': 'in_process_cached_read', 'physical_sample_identity': 'unknown',
                       'authenticity': 'Caller runtime evidence, not authenticated source or physical attestation.',
                       'source_fields': {'force': 'cached_data.force.torch', 'torque': 'cached_data.torque.torch'},
                       'missing_buffers': missing, 'explicit_tensor_snapshot': allow_tensor_snapshot,
                       'capture_method': 'caller_supplied_cached_data',
                       'same_read_context': 'caller_asserted_unverified',
                       'reference_revision_not_runtime_attestation': True},
    }
    return metadata, arrays, sources


def capture_isaac_joint_wrench_layout(output, *, sensor, sensor_cfg, cached_data,
                                     raw_term_output, term_function, term_cfg=None,
                                     processed_term_output=None, observation_slice=None,
                                     allow_tensor_snapshot=False, env_indices=None):
    """Publish copied cached buffers without another sensor.data read or term call.

    Explicit ``allow_tensor_snapshot=True`` permits detach/CPU-copy of real Torch
    tensors. Caller-supplied same-read association is unverified. No native module
    import, component-name assertion or physical simultaneity claim is made.
    ``env_indices`` selects at most 32 original batch rows before CPU copies;
    omitting it preserves every row only when the source batch is already bounded.
    """
    metadata, arrays, sources = _collect_capture(
        sensor=sensor, sensor_cfg=sensor_cfg, cached_data=cached_data,
        raw_term_output=raw_term_output, term_function=term_function, term_cfg=term_cfg,
        processed_term_output=processed_term_output, observation_slice=observation_slice,
        allow_tensor_snapshot=allow_tensor_snapshot, env_indices=env_indices)
    return _write_capture(output, metadata, arrays, sources=sources)


def capture_isaac_joint_wrench_call(output, *, term_function, env, sensor_cfg,
                                   term_cfg=None, allow_tensor_snapshot=False, env_indices=None):
    """Call a trusted Python term once and capture its original return-frame locals.

    Returns ``(original_result, capture_receipt)``. Existing profilers are refused
    before invocation. A successful native result is retained if capture fails;
    its receipt then states ``published=False`` and ``status='unknown'``. Native
    exceptions propagate unchanged. Profiling/snapshotting adds overhead and is
    not an atomic physical sensor read or an authenticated execution attestation.
    Environment selection bounds only saved evidence; the original full result
    is returned unchanged. Large batches require explicit ``env_indices``.
    """
    if not inspect.isfunction(term_function):
        raise InputError('term_function must be the actual trusted Python function')
    if getattr(_ACTIVE_CALL, 'active', False):
        raise InputError('Recursive producer capture is refused')
    previous = sys.getprofile()
    if previous is not None:
        raise InputError('An existing per-thread profiler prevents single-call capture')
    target_frame = None
    count = 0
    payload = None
    capture_error = None
    ownership_lost = False

    def profile(frame, event, arg):
        nonlocal target_frame, count, payload, capture_error
        if frame.f_code is not term_function.__code__:
            return
        if event == 'call':
            count += 1
            if count == 1:
                target_frame = frame
            else:
                capture_error = 'Multiple or recursive target-function invocations prevent capture binding'
        elif event == 'return' and frame is target_frame:
            try:
                # Exception unwinding also emits return(None); outer success is checked below.
                if arg is None:
                    capture_error = capture_error or 'No supported tensor output was returned'
                    return
                local = frame.f_locals
                if not all(key in local for key in ('sensor', 'sensor_data', 'force_data', 'torque_data')):
                    capture_error = capture_error or 'Original return frame lacks the required cached native locals'
                    return
                if capture_error is None:
                    payload = _collect_capture(
                        sensor=local['sensor'], sensor_cfg=sensor_cfg, cached_data=local['sensor_data'],
                        raw_term_output=arg, term_function=term_function, term_cfg=term_cfg,
                        allow_tensor_snapshot=allow_tensor_snapshot,
                        env_indices=env_indices,
                        cached_fields={'force': local['force_data'], 'torque': local['torque_data']})
                    payload[0]['provenance'].update(capture_method='profiled_exact_function_return',
                                                     call_boundary='original_term_return_frame',
                                                     same_read_context='captured_original_frame_locals',
                                                     target_function_calls=count)
            except Exception as exc:
                capture_error = f'Snapshot failed: {type(exc).__name__}: {str(exc)[:512]}'
            finally:
                target_frame = None

    _ACTIVE_CALL.active = True
    try:
        sys.setprofile(profile)
        result = term_function(env, sensor_cfg=sensor_cfg)
    finally:
        if sys.getprofile() is profile:
            sys.setprofile(previous)
        else:
            ownership_lost = True  # preserve a hook deliberately installed by the callee
        target_frame = None
        _ACTIVE_CALL.active = False
    if ownership_lost:
        capture_error = 'Target execution replaced the capture profiler; the replacement was preserved'
    if capture_error is not None or payload is None or count != 1:
        return result, {'status': 'unknown', 'kind': 'layout_capture_not_published', 'published': False,
                        'reason': capture_error or 'No unique original return-frame capture was observed'}
    metadata, arrays, sources = payload
    try:
        receipt = _write_capture(output, metadata, arrays, sources=sources)
    except Exception as exc:
        return result, {'status': 'unknown', 'kind': 'layout_capture_not_published', 'published': False,
                        'reason': f'Publication failed: {type(exc).__name__}: {str(exc)[:512]}'}
    return result, receipt
