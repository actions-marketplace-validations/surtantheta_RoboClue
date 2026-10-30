"""Opt-in observation of ordinary ACT selection calls; no native calls or replacements.

The first prototype captures CPU arrays only. Saved state supports offline numeric
replay; it does not authenticate query lineage, command issuance or physical action.
"""
from __future__ import annotations

import dis
import hashlib
import inspect
import json
import math
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .act_bindings import BindingTracker
from .artifacts import InputError
from .publication import publish_capture

MODELING_SHA256 = '83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af'
CONFIG_SHA256 = '35a60c0dd3c811b137b5eb035bdbd40d40e2c24ba0cda61756052e6e1c829282'
REVISION = '146d8474932c8b566717eb2738ba459337ab71c4'
MAX_EVENTS = 256
MAX_ARRAYS = 4096
MAX_VALUES = 1_000_000
MAX_BYTES = 32 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
MAX_SOURCE = 512 * 1024
_ACTIVE = threading.local()
SCOPE = ('Saved ACT policy-output selection state and actual observed returns only. '
         'Query lineage preceding capture, decoded/issued commands, physical execution '
         'and authenticated runtime origin are not certified.')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(path, maximum=MAX_SOURCE):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise InputError('Native source exceeds capture size bounds')
    return raw



def _ensemble_definition(policy, module, modeling_path):
    helper = getattr(policy, 'temporal_ensembler', None)
    cls = getattr(module, 'ACTTemporalEnsembler', None)
    if cls is None or type(helper) is not cls:
        raise InputError('Ensemble capture requires the exact original native ACTTemporalEnsembler')
    verified = {}
    for name in ('update', 'reset'):
        method = getattr(helper, name, None)
        if (not inspect.ismethod(method) or method.__self__ is not helper
                or method.__func__ is not cls.__dict__.get(name)):
            raise InputError('Replaced native ACT ensemble methods are unsupported')
        function = method.__func__
        if (not inspect.isfunction(function) or function.__module__ != module.__name__
                or function.__qualname__ != f'ACTTemporalEnsembler.{name}'
                or Path(inspect.getsourcefile(function)) != modeling_path):
            raise InputError('Unsupported native ACT ensemble method source')
        verified[name] = function
    return {'update': verified['update'], 'ensemble_class': cls}


def _native_definition(policy):
    cls = type(policy)
    module = sys.modules.get('lerobot.policies.act.modeling_act')
    config_module = sys.modules.get('lerobot.policies.act.configuration_act')
    if (module is None or config_module is None or cls is not getattr(module, 'ACTPolicy', None)
            or type(getattr(policy, 'config', None)) is not getattr(config_module, 'ACTConfig', None)):
        raise InputError('Capture requires the exact supported native ACTPolicy and ACTConfig')
    functions, bridges = {}, set()
    for role, name in [('select', 'select_action'), ('predict', 'predict_action_chunk'), ('reset', 'reset')]:
        method = getattr(policy, name, None)
        if not inspect.ismethod(method) or method.__self__ is not policy:
            raise InputError('Native ACT methods must remain original bound methods')
        outer = method.__func__
        if outer is not cls.__dict__.get(name):
            raise InputError('Replaced native ACT method is unsupported')
        function = inspect.unwrap(outer)
        if (not inspect.isfunction(function) or function.__module__ != module.__name__
                or function.__qualname__ != f'ACTPolicy.{name}'
                or inspect.isgeneratorfunction(function) or inspect.iscoroutinefunction(function)):
            raise InputError('Unsupported native ACT method definition')
        if outer is not function:
            if (getattr(outer, '__wrapped__', None) is not function
                    or outer.__globals__.get('__name__') != 'torch.utils._contextlib'
                    or outer.__code__.co_name != 'decorate_context'):
                raise InputError('Only the original Torch context decorator is supported')
            closure = inspect.getclosurevars(outer).nonlocals
            if closure.get('func') is not function:
                raise InputError('Native Torch decorator targets a different function')
            bridges.add(outer.__code__)
        functions[role] = function
    modeling_path = Path(inspect.getsourcefile(functions['select']))
    if any(Path(inspect.getsourcefile(fn)) != modeling_path for fn in functions.values()):
        raise InputError('Native ACT methods do not share the pinned source module')
    config_path = Path(inspect.getsourcefile(type(policy.config)))
    modeling, config = _bytes(modeling_path), _bytes(config_path)
    if _sha(modeling) != MODELING_SHA256 or _sha(config) != CONFIG_SHA256:
        raise InputError('Unsupported native ACT source revision; no capture instrumentation installed')
    if policy.config.temporal_ensemble_coeff is not None:
        functions.update(_ensemble_definition(policy, module, modeling_path))
    license_path = next((p / 'LICENSE' for p in modeling_path.parents if (p / 'LICENSE').is_file()), None)
    if license_path is None:
        raise InputError('Original native license is unavailable')
    sources = {'modeling_act.py': modeling, 'configuration_act.py': config,
               'LICENSE.txt': _bytes(license_path)}
    return dict(functions, bridges=bridges, sources=sources,
                paths={'modeling_act.py': modeling_path, 'configuration_act.py': config_path,
                       'LICENSE.txt': license_path},
                source={'repository': 'https://github.com/huggingface/lerobot', 'revision': REVISION,
                        'modeling_sha256': _sha(modeling), 'config_sha256': _sha(config)})


def _destination(value):
    path = Path(value).expanduser().absolute()
    if '..' in path.parts:
        raise InputError('Capture destination must not contain parent traversal')
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise InputError('Capture destination must not traverse symlinks')
    if path.exists():
        raise InputError('Capture destination already exists; no files will be replaced')
    return path


def _settings(policy):
    cfg = policy.config
    horizon, chunk, coefficient = cfg.n_action_steps, cfg.chunk_size, cfg.temporal_ensemble_coeff
    if (type(horizon) is not int or type(chunk) is not int
            or not 1 <= horizon <= chunk <= 256):
        raise InputError('Native action horizon/chunk must be bounded positive integers')
    if coefficient is not None and (type(coefficient) not in (int, float) or not math.isfinite(coefficient)):
        raise InputError('Native temporal ensemble coefficient must be finite, not Boolean')
    return {'n_action_steps': horizon, 'chunk_size': chunk, 'temporal_ensemble_coeff': coefficient}


def _normal_return(frame):
    return (frame.f_lasti >= 0
            and dis.opname[frame.f_code.co_code[frame.f_lasti]] in ('RETURN_VALUE', 'RETURN_CONST'))


def _host_copy(value, rows):
    selected = value if rows is None else value[rows]
    if type(selected) is np.ndarray:
        return selected.copy()
    return selected.detach().numpy().copy()  # CPU was checked before selecting/copying.


class _SelectionCapture:
    def __init__(self, policy, output, *, env_indices, allow_tensor_snapshot, domain):
        if type(allow_tensor_snapshot) is not bool:
            raise InputError('allow_tensor_snapshot must be Boolean')
        if domain != 'policy_output' or type(domain) is not str:
            raise InputError("This capture supports only domain='policy_output'; no command-role inference")
        if env_indices is not None and (type(env_indices) not in (list, tuple)
                or not 1 <= len(env_indices) <= 32
                or any(type(v) is not int or not 0 <= v < 1_000_000 for v in env_indices)
                or len(set(env_indices)) != len(env_indices)):
            raise InputError('env_indices must be 1–32 distinct bounded native integers')
        self._policy = policy
        self._output = output
        self._allow = allow_tensor_snapshot
        self._requested_rows = None if env_indices is None else list(env_indices)
        self._used = False
        self._uninstalling = False
        self._owner = None
        self._definition = None
        self._frame = self._predict_frame = self._update_frame = None
        self._event = None
        self._events, self._arrays = [], {}
        self._values = self._array_bytes = self._query_count = 0
        self._source_envs = self._rows = None
        self._error = None
        self._bindings = BindingTracker()
        self.receipt = self._unpublished('No selected native ACT operation observed')

    @staticmethod
    def _unpublished(reason):
        return {'kind': 'act_selection_capture_receipt', 'status': 'unknown',
                'published': False, 'reason': reason}

    def __enter__(self):
        if self._used:
            raise InputError('ACT capture contexts are single use')
        if getattr(_ACTIVE, 'active', False) or sys.getprofile() is not None:
            raise InputError('Nested capture or existing per-thread profiler prevents ACT observation')
        if not self._allow:
            raise InputError('ACT Tensor capture requires explicit allow_tensor_snapshot=True')
        self._output = _destination(self._output)
        self._definition = _native_definition(self._policy)
        _settings(self._policy)
        if sum(len(raw) for raw in self._definition['sources'].values()) > MAX_BYTES:
            raise InputError('Native source payload exceeds capture bounds')
        self._used = True
        self._owner = threading.get_ident()
        self._hook = self._profile
        _ACTIVE.active = True
        try:
            sys.setprofile(self._hook)
        except BaseException:
            _ACTIVE.active = False
            self._clear()
            raise
        return self

    def _shape(self, value, rank, integer):
        if type(value) is np.ndarray:
            shape, dtype = value.shape, value.dtype
        else:
            if not self._allow:
                raise InputError('Tensor snapshots require explicit opt-in')
            import torch
            if type(value) is not torch.Tensor or value.device.type != 'cpu' or value.layout != torch.strided:
                raise InputError('Prototype supports ordinary dense CPU Tensors only; no device transfer performed')
            shape = tuple(value.shape)
            if value.dtype not in ((torch.int64, torch.int32) if integer else
                                   (torch.float16, torch.float32, torch.float64)):
                raise InputError('Unsupported native ACT tensor dtype')
            dtype = np.dtype(str(value.dtype).split('.')[-1])
        if len(shape) != rank or any(type(n) is not int or n < 0 for n in shape):
            raise InputError('Native ACT array rank/shape is unsupported')
        if (dtype.hasobject or dtype.itemsize > 8
                or (integer and dtype.kind not in 'iu')
                or (not integer and dtype not in (np.dtype('float16'), np.dtype('float32'), np.dtype('float64')))):
            raise InputError('Native ACT dtype must be supported <=8-byte numeric values, not objects/Boolean')
        return shape, dtype

    def _snapshot(self, value, key, *, rank, integer=False, batch=False):
        shape, dtype = self._shape(value, rank, integer)
        rows = None
        saved_shape = shape
        if batch:
            if not 1 <= shape[0] <= 1_000_000 or not 1 <= shape[-1] <= 256:
                raise InputError('Native ACT batch/action width exceeds capture bounds')
            if self._source_envs is None:
                if self._requested_rows is None:
                    if shape[0] > 32:
                        raise InputError('Select <=32 explicit env_indices before capturing a larger native batch')
                    self._rows = list(range(shape[0]))
                else:
                    self._rows = self._requested_rows.copy()
                self._source_envs = shape[0]
            if shape[0] != self._source_envs or any(i >= shape[0] for i in self._rows):
                raise InputError('Native ACT batch changed or selected environment index is out of range')
            rows = self._rows
            saved_shape = (len(rows), *shape[1:])
            if rank == 3 and shape[1] > 256:
                raise InputError('Native ACT chunk/state length exceeds capture bounds')
        elif any(n > 256 for n in shape):
            raise InputError('Native ACT weight/count dimensions exceed capture bounds')
        cells = math.prod(saved_shape)
        count_bytes = cells * dtype.itemsize
        source_bytes = sum(len(raw) for raw in self._definition['sources'].values())
        if (len(self._arrays) >= MAX_ARRAYS or self._values + cells > MAX_VALUES
                or self._array_bytes + count_bytes + source_bytes > MAX_BYTES):
            raise InputError('ACT capture numeric/payload budget exceeded before host copy; reduce captured calls/rows')
        copied = _host_copy(value, rows)
        if copied.shape != saved_shape or copied.dtype != dtype or not np.isfinite(copied).all():
            raise InputError('Native ACT array changed or contains unsupported nonfinite values')
        self._arrays[key] = copied
        self._values += cells
        self._array_bytes += count_bytes
        return key

    def _state(self, prefix, settings):
        if settings['temporal_ensemble_coeff'] is None:
            queue = self._policy._action_queue
            if (type(queue) is not deque or queue.maxlen is None
                    or not 1 <= queue.maxlen <= 4096 or len(queue) > 256):
                raise InputError('Native ACT action queue is unsupported')
            return {'mode': 'queue', 'queue_maxlen': queue.maxlen,
                    'queue': [self._snapshot(v, f'{prefix}_q{i:03d}', rank=2, batch=True)
                              for i, v in enumerate(queue)]}
        helper = self._policy.temporal_ensembler
        aggregate, counts = helper.ensembled_actions, helper.ensembled_actions_count
        return {'mode': 'ensemble',
                'aggregate': None if aggregate is None else self._snapshot(
                    aggregate, prefix + '_aggregate', rank=3, batch=True),
                'counts': None if counts is None else self._snapshot(
                    counts, prefix + '_counts', rank=2, integer=True),
                'weights': self._snapshot(helper.ensemble_weights, prefix + '_weights', rank=1),
                'weights_cumsum': self._snapshot(helper.ensemble_weights_cumsum, prefix + '_weights_cumsum', rank=1)}

    def _inside_select(self, frame, target='predict'):
        caller = frame.f_back
        for _ in range(8):
            if caller is self._frame:
                return self._event is not None and self._event['type'] == 'select'
            if caller is None or caller.f_code not in self._definition['bridges']:
                return False
            if caller.f_locals.get('func') is not self._definition[target]:
                return False
            caller = caller.f_back
        return False

    def _profile(self, frame, event, arg):
        if event == 'c_call' and arg is sys.setprofile and not self._uninstalling:
            self._error = 'Per-thread profiler changed during capture; complete call coverage is unavailable'
            return
        if self._error is not None or event not in ('call', 'return'):
            return
        try:
            definition = self._definition
            role = next((r for r in ('select', 'reset', 'predict', 'update')
                         if r in definition and frame.f_code is definition[r].__code__), None)
            if role == 'update':
                helper = getattr(self._policy, 'temporal_ensembler', None)
                if frame.f_locals.get('self') is not helper:
                    return
                if event == 'call':
                    inside = (type(helper) is definition.get('ensemble_class')
                              and self._inside_select(frame, 'update'))
                    self._bindings.update_start(helper, frame.f_locals.get('actions'), inside_select=inside)
                    if self._bindings.scope()['off_path_update_calls'] > MAX_EVENTS:
                        raise InputError('Off-path ACT update call budget exceeded')
                    if inside:
                        self._update_frame = frame
                elif frame is self._update_frame:
                    if not _normal_return(frame) or arg is None:
                        raise InputError('Native ACT ensemble update did not return normally')
                    self._bindings.update_return(arg)
                    self._update_frame = None
                return
            if role is None or frame.f_locals.get('self') is not self._policy:
                return
            if role == 'predict':
                if event == 'call' and self._inside_select(frame):
                    if self._predict_frame is not None:
                        raise InputError('Recursive native prediction prevents capture binding')
                    self._predict_frame = frame
                elif event == 'return' and frame is self._predict_frame:
                    if not _normal_return(frame) or arg is None:
                        raise InputError('Native prediction did not return normally')
                    query = {'query_id': self._query_count,
                             'array': self._snapshot(arg, f'e{self._event["event_id"]:03d}_prediction{len(self._event["predictions"]):03d}', rank=3, batch=True)}
                    self._query_count += 1
                    self._event['predictions'].append(query)
                    self._bindings.prediction(arg, query)
                    self._predict_frame = None
                return
            if event == 'call':
                if self._frame is not None:
                    raise InputError('Nested native ACT selection/reset prevents ordered event binding')
                if len(self._events) >= MAX_EVENTS:
                    raise InputError('ACT event budget exceeded; reduce observed calls')
                index = len(self._events)
                settings = _settings(self._policy)
                self._event = {'event_id': index, 'type': 'select' if role == 'select' else 'reset',
                               'settings': settings, 'pre_state': self._state(f'e{index:03d}_pre', settings),
                               'predictions': [], 'selected': None}
                self._frame = frame
                if role == 'select':
                    mode = self._event['pre_state']['mode']
                    queue = self._policy._action_queue if mode == 'queue' else None
                    self._bindings.start(
                        self._event, queue_head=queue[0] if queue else None,
                        helper=getattr(self._policy, 'temporal_ensembler', None) if mode == 'ensemble' else None)
            elif frame is self._frame:
                if not _normal_return(frame):
                    raise InputError('Native ACT operation did not return normally')
                record = self._event
                if role == 'select':
                    if arg is None:
                        raise InputError('Native ACT selection has no supported normal return')
                    record['selected'] = self._snapshot(arg, f'e{record["event_id"]:03d}_selected', rank=2, batch=True)
                if _settings(self._policy) != record['settings']:
                    raise InputError('Native selection settings changed within a captured operation')
                record['post_state'] = self._state(f'e{record["event_id"]:03d}_post', record['settings'])
                record['completion'] = 'normal'
                if role == 'select':
                    record['local_binding'] = self._bindings.finish(record, arg)
                self._events.append(record)
                self._event = self._frame = None
        except Exception as exc:
            self._error = f'ACT snapshot failed: {type(exc).__name__}: {str(exc)[:512]}'
            self._frame = self._predict_frame = self._update_frame = None

    def _write(self):
        self._bindings.finalize(self._events)
        definition = self._definition
        for name, path in definition.get('paths', {}).items():
            if _bytes(path) != definition['sources'][name]:
                raise InputError('Native ACT source changed during capture')
        output = _destination(self._output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.act-selection-', dir=output.parent) as temporary:
            stage = Path(temporary)
            (stage / 'sources').mkdir()
            np.savez(stage / 'arrays.npz', **self._arrays)
            for name, raw in definition['sources'].items():
                if name not in ('modeling_act.py', 'configuration_act.py', 'LICENSE.txt'):
                    raise InputError('Unexpected native capture source filename')
                (stage / 'sources' / name).write_bytes(raw)
            files = [{'path': str(p.relative_to(stage)), 'sha256': _sha(p.read_bytes())}
                     for p in sorted(stage.rglob('*')) if p.is_file()]
            selection = {'schema_version': 1, 'kind': 'act_selection_capture', 'domain': 'policy_output',
                         'source': dict(definition['source'], files={name: {'sha256': _sha(raw)}
                                                                   for name, raw in definition['sources'].items()}),
                         'environment_selection': {'source_num_envs': self._source_envs, 'env_indices': self._rows,
                                                   'saved_rows': None if self._rows is None else len(self._rows),
                                                   'omitted_envs': None if self._rows is None else self._source_envs - len(self._rows)},
                         'binding_scope': self._bindings.scope(),
                         'coverage': {'target': 'one_policy_on_owner_thread',
                                      'complete_target_call_history': True,
                                      'external_state_mutations': 'untracked'},
                         'events': self._events, 'arrays': {key: {'shape': list(array.shape), 'dtype': str(array.dtype)}
                                                          for key, array in self._arrays.items()},
                         'files': files, 'requested_env_indices': self._requested_rows,
                         'provenance': {'capture_method': 'profiled_original_ACT_policy_returns',
                                        'origin': 'caller-runtime-evidence-not-attestation',
                                        'prior_query_lineage': 'unknown', 'downstream_association': 'unknown'},
                         'scope': SCOPE}
            manifest = json.dumps(selection, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
            if len(manifest) > MAX_MANIFEST or len(manifest) + sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()) > MAX_BYTES:
                raise InputError('ACT capture manifest/total payload exceeds bounds')
            (stage / 'selection.json').write_bytes(manifest)
            (stage / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': _sha(manifest)}))
            publish_capture(stage, output)
        return {'kind': 'act_selection_capture_receipt', 'status': 'captured', 'published': True,
                'directory': str(output), 'event_count': len(self._events),
                'selection_count': sum(e['type'] == 'select' for e in self._events),
                'reset_count': sum(e['type'] == 'reset' for e in self._events), 'domain': 'policy_output'}

    def _clear(self):
        self._frame = self._predict_frame = self._update_frame = self._event = None
        self._bindings.clear()
        self._policy = self._definition = None
        self._events.clear()
        self._arrays.clear()
        self._hook = None

    def __exit__(self, exc_type, exc, tb):
        if threading.get_ident() != self._owner:
            raise InputError('ACT capture must exit on its owning thread')
        owned = sys.getprofile() is self._hook
        if owned:
            self._uninstalling = True
            sys.setprofile(None)
        _ACTIVE.active = False
        try:
            if exc_type is not None:
                self.receipt = self._unpublished('Caller exited with an exception; no capture published')
            elif not owned:
                self.receipt = self._unpublished('Profiler replaced; replacement preserved and no capture published')
            elif self._error is not None or self._frame is not None or not self._events:
                self.receipt = self._unpublished(self._error or 'No complete normal native ACT operation captured')
            else:
                self.receipt = self._write()
        except Exception as failure:
            self.receipt = self._unpublished(f'Publication failed: {type(failure).__name__}: {str(failure)[:512]}')
        finally:
            self._clear()
        return False


def capture_act_selection(policy, output, *, env_indices=None, allow_tensor_snapshot=False,
                          domain='policy_output'):
    """Observe bounded normal select_action/reset calls after policy initialization.

    The context never invokes or replaces native methods. Inspect ``receipt``
    after exit: capture/publication success is separate from native execution.
    CPU snapshots require explicit opt-in; physical/issued roles remain unknown.
    """
    return _SelectionCapture(policy, output, env_indices=env_indices,
                             allow_tensor_snapshot=allow_tensor_snapshot, domain=domain)
