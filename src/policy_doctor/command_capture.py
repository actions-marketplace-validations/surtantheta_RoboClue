"""Opt-in observation of a pinned synchronous ACT software-command path.

No methods, hooks, policy calls or consumer calls are installed or invoked. Saved
relations are caller-runtime evidence, never authenticated origin or robot motion.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
import tempfile
import threading
import types
import weakref
from pathlib import Path

import numpy as np

from .act_bindings import OpaqueTokens
from .act_capture import _bytes, _destination, _normal_return, _sha
from .act_capture import _native_definition as _act_definition
from .artifacts import InputError
from .command_rules import (
    MAX_BYTES,
    MAX_DIM,
    MAX_MANIFEST,
    MAX_MULTIPLIER,
    MAX_TICKS,
    MAX_VALUES,
    REPOSITORY,
    REVISION,
    SOURCE_HASHES,
    SOURCE_MODULES,
)
from .publication import publish_capture

_ACTIVE = threading.local()
_TOKEN_ROLES = ('selected', 'post_input', 'post_output', 'engine_output', 'add_input',
                'get_output', 'requested', 'returned', 'dispatch_return')
_RELATIONS = ('selected_to_post_input', 'post_return_is_engine_action', 'engine_to_add_input',
              'get_prebuffer_to_return', 'get_return_to_consumer', 'consumer_to_dispatch_target')
SCOPE = ('One observed synchronous native dispatch chain to an explicitly supplied software consumer. '
         'External writes, prior query lineage, physical sampling, recorded actions and robot motion remain unknown.')


def _same_callable(left, right):
    if inspect.ismethod(left) and inspect.ismethod(right):
        return left.__self__ is right.__self__ and left.__func__ is right.__func__
    return left is right


def _method(obj, name, cls):
    method = getattr(obj, name, None)
    original = getattr(cls, name, None)
    if not inspect.ismethod(method) or method.__self__ is not obj or method.__func__ is not original:
        raise InputError('Replaced native command-path method is unsupported')
    return original


def _native_definition(engine, interpolator, consumer):
    modules, sources, paths = {}, {}, {}
    for filename, name in SOURCE_MODULES.items():
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__file__', None):
            raise InputError(f'Load the supported native module before capture: {name}')
        path = Path(module.__file__)
        raw = _bytes(path)
        if _sha(raw) != SOURCE_HASHES[filename]:
            raise InputError(f'Unsupported native command-path source: {filename}')
        modules[name] = module
        sources[filename], paths[filename] = raw, path
    sync = modules['lerobot.inference.sync']
    interp_module = modules['lerobot.utils.action_interpolator']
    pipeline_module = modules['lerobot.processor.pipeline']
    if type(engine) is not sync.SyncInferenceEngine or type(interpolator) is not interp_module.ActionInterpolator:
        raise InputError('Command capture requires exact supported native synchronous engine/interpolator')
    act = _act_definition(engine._policy)
    license_raw = act['sources']['LICENSE.txt']
    if _sha(license_raw) != SOURCE_HASHES['LICENSE.txt']:
        raise InputError('Unsupported original native license')
    sources['LICENSE.txt'] = license_raw
    paths['LICENSE.txt'] = act['paths']['LICENSE.txt']
    pipeline = engine._postprocessor
    if type(pipeline) is not pipeline_module.DataProcessorPipeline:
        raise InputError('Command capture requires the native DataProcessorPipeline instance')
    if not isinstance(pipeline.steps, (list, tuple)) or len(pipeline.steps) > 16:
        raise InputError('Native postprocessor exceeds bounded step scope')
    norm = modules['lerobot.processor.normalize_processor']
    device = modules['lerobot.processor.device_processor']
    converters = modules['lerobot.processor.converters']
    dispatch = modules['lerobot.inference.dispatch']
    base = modules['lerobot.inference.base']
    functions = {
        'tick': dispatch.dispatch_tick, 'engine': _method(engine, 'get_action', sync.SyncInferenceEngine),
        'select': act['select'], 'pipeline': _method(pipeline, '__call__', pipeline_module.DataProcessorPipeline),
        'add': _method(interpolator, 'add', interp_module.ActionInterpolator),
        'get': _method(interpolator, 'get', interp_module.ActionInterpolator),
        'ack': _method(engine, 'record_dispatch', base.InferenceEngine),
    }
    supported = len(pipeline.steps) == 2 and type(pipeline.steps[0]) is norm.UnnormalizerProcessorStep and type(pipeline.steps[1]) is device.DeviceProcessorStep
    if supported:
        functions['step0'] = _method(pipeline.steps[0], '__call__', norm.UnnormalizerProcessorStep)
        functions['step1'] = _method(pipeline.steps[1], '__call__', device.DeviceProcessorStep)
        functions['transform'] = _method(pipeline.steps[0], '_apply_transform', norm._NormalizationMixin)
    function = consumer.__func__ if inspect.ismethod(consumer) else consumer
    if (not inspect.isfunction(function) or inspect.isgeneratorfunction(function)
            or inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function)):
        raise InputError('Consumer must be an explicit synchronous Python function or bound method')
    parameters = list(inspect.signature(consumer).parameters.values())
    if (len(parameters) != 1 or parameters[0].kind not in
            (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)):
        raise InputError('Consumer must accept one explicit positional action argument')
    functions['consumer'] = function
    # Compare loaded native code with compiled pinned bytes without executing artifact code.
    for role, native in functions.items():
        if role == 'consumer':
            continue
        filename = next((name for name, module in modules.items() if module.__name__ == native.__module__), None)
        source_name = next((name for name, module_name in SOURCE_MODULES.items() if module_name == filename), None)
        if source_name is None:
            raise InputError('Unsupported loaded native function identity')
        compiled = compile(sources[source_name], native.__code__.co_filename, 'exec', dont_inherit=True)
        stack, matches = [compiled], []
        while stack:
            code = stack.pop()
            if code.co_qualname == native.__qualname__:
                matches.append(code)
            stack.extend(item for item in code.co_consts if isinstance(item, types.CodeType))
        if len(matches) != 1 or matches[0] != native.__code__:
            raise InputError('Loaded native code differs from its pinned original source')
    return {'functions': functions, 'sources': sources, 'paths': paths,
            'source': {'repository': REPOSITORY, 'revision': REVISION},
            'consumer_argument': parameters[0].name, 'consumer_owner': getattr(consumer, '__self__', None),
            'consumer_owner_argument': function.__code__.co_varnames[0] if inspect.ismethod(consumer) else None,
            'pipeline_supported': supported,
            'converters_supported': (pipeline.to_transition is converters.policy_action_to_transition
                                     and pipeline.to_output is converters.transition_to_policy_action)}


class _CommandCapture:
    def __init__(self, engine, interpolator, consumer, output, *, action_names, consumer_namespace, allow_tensor_snapshot):
        def text(value):
            return (type(value) is str and 1 <= len(value.encode('utf8')) <= 128
                    and value.isprintable() and value == value.strip())
        if (type(action_names) is not list or not 1 <= len(action_names) <= MAX_DIM
                or any(not text(n) for n in action_names)
                or len(set(action_names)) != len(action_names)):
            raise InputError('action_names must be distinct explicit bounded channel names')
        if not text(consumer_namespace):
            raise InputError('consumer_namespace must be an explicit bounded nonempty string')
        if type(allow_tensor_snapshot) is not bool:
            raise InputError('allow_tensor_snapshot must be Boolean')
        self._engine, self._interpolator, self._consumer = engine, interpolator, consumer
        self._output, self._names, self._namespace = output, list(action_names), consumer_namespace
        self._allow = allow_tensor_snapshot
        self._used = self._uninstalling = False
        self._definition = self._hook = self._owner = self._runtime_guard = None
        self._frames, self._live, self._arrays, self._ticks = {}, {}, {}, []
        self._tokens = OpaqueTokens()
        self._tick = self._error = None
        self._values = 0
        self.receipt = self._unpublished('No complete dispatch tick observed')

    @staticmethod
    def _unpublished(reason):
        return {'kind': 'command_path_capture_receipt', 'status': 'unknown', 'published': False, 'reason': reason}

    def __enter__(self):
        if self._used or getattr(_ACTIVE, 'active', False) or sys.getprofile() is not None:
            raise InputError('Existing profiler, nested or reused command capture is unsupported')
        self._output = _destination(self._output)
        self._definition = _native_definition(self._engine, self._interpolator, self._consumer)
        if (self._definition['source'].get('repository') == REPOSITORY
                and getattr(self._engine, '_ordered_action_keys', None) != self._names):
            self._clear()
            raise InputError('action_names must match the actual native engine ordered_action_keys')
        pipeline = self._engine._postprocessor
        self._runtime_guard = {
            'policy': self._engine._policy, 'pipeline': pipeline, 'steps': tuple(pipeline.steps),
            'converters': (getattr(pipeline, 'to_transition', None), getattr(pipeline, 'to_output', None)),
            'hooks': (tuple(getattr(pipeline, 'before_step_hooks', ())), tuple(getattr(pipeline, 'after_step_hooks', ()))),
            'names': list(self._engine._ordered_action_keys) if hasattr(self._engine, '_ordered_action_keys') else None,
        }
        self._used = True
        self._owner = threading.get_ident()
        self._hook = self._profile
        _ACTIVE.active = True
        sys.setprofile(self._hook)
        return self

    def _check_runtime(self):
        guard = self._runtime_guard
        pipeline = self._engine._postprocessor
        if (self._engine._policy is not guard['policy'] or pipeline is not guard['pipeline']
                or len(pipeline.steps) != len(guard['steps'])
                or any(a is not b for a, b in zip(pipeline.steps, guard['steps'], strict=True))
                or getattr(pipeline, 'to_transition', None) is not guard['converters'][0]
                or getattr(pipeline, 'to_output', None) is not guard['converters'][1]
                or tuple(getattr(pipeline, 'before_step_hooks', ())) != guard['hooks'][0]
                or tuple(getattr(pipeline, 'after_step_hooks', ())) != guard['hooks'][1]
                or (guard['names'] is not None and self._engine._ordered_action_keys != guard['names'])):
            raise InputError('Native policy/pipeline/steps/converters/hooks or declared action order changed during capture')

    def _snapshot(self, value, rank=None):
        if value is None:
            return None
        if type(value) is np.ndarray:
            shape, dtype = value.shape, value.dtype
        else:
            torch = sys.modules.get('torch')
            if not self._allow:
                raise InputError('Tensor snapshots require explicit allow_tensor_snapshot=True')
            if (torch is None or type(value) is not torch.Tensor or value.device.type != 'cpu'
                    or value.layout != torch.strided or value.dtype != torch.float32):
                raise InputError('Command snapshots support only ordinary finite CPU float32 tensors')
            shape, dtype = tuple(value.shape), np.dtype('float32')
        if (dtype != np.dtype('float32') or len(shape) not in (1, 2) or not 1 <= shape[-1] <= MAX_DIM
                or (len(shape) == 2 and shape[0] != 1) or shape[-1] != len(self._names)
                or (rank is not None and len(shape) != rank)):
            raise InputError('Command arrays require float32 batch1 vectors matching explicit action_names')
        count = math.prod(shape)
        if self._values + count > MAX_VALUES or len(self._arrays) >= 4096:
            raise InputError('Command numeric payload exceeds capture bounds')
        if (self._values + count) * 4 + sum(map(len, self._definition['sources'].values())) > MAX_BYTES:
            raise InputError('Command byte payload exceeds capture bounds before host copy')
        copied = value.copy() if type(value) is np.ndarray else value.detach().numpy().copy()
        if not np.isfinite(copied).all():
            raise InputError('Command values must be finite')
        key = f'a{len(self._arrays):04d}'
        self._arrays[key] = copied
        self._values += count
        return key

    def _remember(self, role, value):
        token = self._tokens.token(value) if value is not None else None
        self._tick['bindings']['tokens'][role] = token
        try:
            self._live[role] = weakref.ref(value)
        except TypeError:
            self._live[role] = lambda: None
        return token

    def _is(self, role, value):
        ref = self._live.get(role)
        return ref is not None and ref() is not None and ref() is value

    def _state(self):
        interp = self._interpolator
        multiplier, index, emitted = interp.multiplier, interp._idx, interp._emitted_policy_action
        if (type(multiplier) is not int or not 1 <= multiplier <= MAX_MULTIPLIER
                or type(index) is not int or type(emitted) is not bool
                or type(interp._buffer) is not list or len(interp._buffer) > MAX_MULTIPLIER
                or not 0 <= index <= len(interp._buffer)):
            raise InputError('Unsupported or unbounded native interpolation state')
        return {'previous': self._snapshot(interp._prev), 'previous_token': self._tokens.token(interp._prev),
                'buffer': [self._snapshot(value) for value in interp._buffer],
                'buffer_tokens': [self._tokens.token(value) for value in interp._buffer],
                'index': index, 'emitted': emitted}

    def _configuration(self, step, index):
        if index == 1:
            return {'device': str(getattr(step, 'device', 'unknown')),
                    'float_dtype': getattr(step, 'float_dtype', None)}
        norm_map = getattr(step, 'norm_map', {})
        modes = [v for k, v in norm_map.items() if getattr(k, 'name', k) == 'ACTION']
        mode = getattr(modes[0], 'value', modes[0]) if len(modes) == 1 else 'IDENTITY'
        eps = getattr(step, 'eps', None)
        if type(eps) not in (int, float) or not math.isfinite(eps):
            raise InputError('Native processor epsilon must be finite')
        stats = self._stats(step)
        declared = getattr(step, 'device', None)
        return {'mode': str(mode), 'eps': eps, 'stats': stats, 'device': 'cpu',
                'declared_device': None if declared is None else str(declared)}

    def _stats(self, step):
        stats = getattr(step, '_tensor_stats', {}).get('action', {})
        if type(stats) is not dict:
            raise InputError('Native effective action statistics are malformed')
        return {name: self._snapshot(stats[name]) for name in ('mean', 'std') if name in stats}

    def _inside(self, frame, role):
        target = self._frames.get(role)
        current = frame.f_back
        for _ in range(32):
            if current is None:
                break
            if current is target:
                return True
            current = current.f_back
        return False

    def _view_pair(self, post, engine):
        torch = sys.modules.get('torch')
        if torch is None or type(post) is not torch.Tensor or type(engine) is not torch.Tensor:
            return None
        try:
            if (post.device.type != 'cpu' or engine.device.type != 'cpu' or post.dtype != torch.float32
                    or engine.dtype != torch.float32 or post.layout != torch.strided or engine.layout != torch.strided):
                return None
            same = post.untyped_storage()._cdata == engine.untyped_storage()._cdata
            token = self._tokens.token(post)
            def view(value):
                shape, stride = list(value.shape), list(value.stride())
                offset = value.storage_offset()
                if (any(type(v) is not int or not 0 <= v <= 2**53 - 1 for v in [*stride, offset])
                        or value.element_size() != 4):
                    raise InputError('Unsupported native tensor view bounds')
                return {'base_token': token, 'shape': shape, 'stride': stride,
                        'storage_offset': offset, 'element_size': 4}
            return {'post': view(post), 'engine': view(engine), 'same_live_storage': same,
                    'basis': 'live_post_output_storage'}
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return None

    def _start_tick(self, frame):
        self._check_runtime()
        if self._tick is not None or len(self._ticks) >= MAX_TICKS:
            raise InputError('Nested dispatch or command tick budget exceeded')
        if not _same_callable(frame.f_locals.get('send'), self._consumer):
            raise InputError('Dispatch callback is not the explicitly supplied consumer')
        self._frames['tick'] = frame
        self._tick = {'tick_id': len(self._ticks), 'completion': None, 'outcome': None,
                      'selected': None, 'postprocessor': None, 'engine_output': None,
                      'interpolation': {'multiplier': self._interpolator.multiplier, 'pre': self._state(), 'add': None, 'get': None},
                      'requested': None, 'returned': None, 'dispatch_return': None,
                      'consumer_called': False, 'consumer_completed': False, 'ack_count': 0,
                      'bindings': {'tokens': dict.fromkeys(_TOKEN_ROLES), 'relations': dict.fromkeys(_RELATIONS),
                                   'post_to_engine_view': None}, 'reason': None}
        self._live.clear()

    def _entry(self, role, frame):
        if role in self._frames:
            raise InputError('Recursive command boundary is unsupported')
        self._frames[role] = frame
        record = self._tick
        if role == 'pipeline':
            self._check_runtime()
            value = frame.f_locals['data']
            self._remember('post_input', value)
            record['bindings']['relations']['selected_to_post_input'] = self._is('selected', value)
            pipeline = self._engine._postprocessor
            hooks = getattr(pipeline, 'before_step_hooks', []) or getattr(pipeline, 'after_step_hooks', [])
            record['postprocessor'] = {'input': self._snapshot(value, rank=2), 'output': None, 'steps': [],
                                      'hooks_present': bool(hooks), 'converters_supported': self._definition['converters_supported'],
                                      'steps_supported': self._definition['pipeline_supported']}
        elif role in ('step0', 'step1'):
            index = int(role[-1])
            step = self._engine._postprocessor.steps[index]
            record['postprocessor']['steps'].append({'index': index, 'kind': 'unnormalizer' if index == 0 else 'device',
                'input': self._snapshot(frame.f_locals['transition'].get('action'), rank=2), 'output': None,
                'configuration': self._configuration(step, index), 'transform': None})
        elif role == 'transform':
            record['postprocessor']['steps'][-1]['transform'] = {
                'input': self._snapshot(frame.f_locals['tensor'], rank=2), 'output': None,
                'key': frame.f_locals['key'], 'inverse': frame.f_locals.get('inverse'), 'stats': {}}
        elif role == 'add':
            value = frame.f_locals['action']
            self._remember('add_input', value)
            record['bindings']['relations']['engine_to_add_input'] = self._is('engine_output', value)
            record['interpolation']['add'] = {'input': self._snapshot(value, rank=1), 'post': None}
        elif role == 'get':
            record['interpolation']['get'] = {'pre': self._state(), 'output': None, 'post': None}
        elif role == 'consumer':
            if record['consumer_called']:
                raise InputError('Multiple consumers per tick are unsupported')
            value = frame.f_locals[self._definition['consumer_argument']]
            record['consumer_called'] = True
            record['requested'] = self._snapshot(value, rank=1)
            self._remember('requested', value)
            record['bindings']['relations']['get_return_to_consumer'] = self._is('get_output', value)

    def _return(self, role, frame, value):
        normal = _normal_return(frame)
        record = self._tick
        if role in ('tick', 'pipeline'):
            self._check_runtime()
        if not normal and role in ('pipeline', 'step0', 'step1', 'transform'):
            raise InputError('Partial processor exception is outside supported capture publication scope')
        if role == 'tick':
            record['completion'] = 'normal' if normal else 'error'
            record['outcome'] = ('error' if not normal else 'sent' if record['consumer_completed'] else 'not_sent')
            record['dispatch_return'] = self._snapshot(value, rank=1) if normal else None
            if value is not None:
                self._remember('dispatch_return', value)
                record['bindings']['relations']['consumer_to_dispatch_target'] = self._is('requested', value)
            self._ticks.append(record)
            self._tick = None
            self._frames.clear()
            self._live.clear()
            return
        if normal:
            if role == 'select':
                record['selected'] = self._snapshot(value, rank=2)
                self._remember('selected', value)
            elif role == 'pipeline':
                record['postprocessor']['output'] = self._snapshot(value, rank=2)
                self._remember('post_output', value)
            elif role in ('step0', 'step1'):
                record['postprocessor']['steps'][-1]['output'] = self._snapshot(value.get('action'), rank=2)
            elif role == 'transform':
                transform = record['postprocessor']['steps'][-1]['transform']
                transform['output'] = self._snapshot(value, rank=2)
                # Pin actual operands retained by the original transform frame, not
                # a mapping that another actor could replace after those reads.
                transform['stats'] = {name: self._snapshot(frame.f_locals[name], rank=1)
                                      for name in ('mean', 'std') if name in frame.f_locals}
            elif role == 'engine':
                record['engine_output'] = self._snapshot(value, rank=1)
                self._remember('engine_output', value)
                post = frame.f_locals.get('action')
                record['bindings']['relations']['post_return_is_engine_action'] = self._is('post_output', post)
                record['bindings']['post_to_engine_view'] = self._view_pair(post, value)
            elif role == 'add':
                record['interpolation']['add']['post'] = self._state()
            elif role == 'get':
                operation = record['interpolation']['get']
                operation['output'], operation['post'] = self._snapshot(value, rank=1), self._state()
                token = self._remember('get_output', value)
                pre = operation['pre']
                index = pre['index']
                record['bindings']['relations']['get_prebuffer_to_return'] = (
                    index < len(pre['buffer_tokens']) and pre['buffer_tokens'][index] is not None
                    and pre['buffer_tokens'][index] == token)
            elif role == 'consumer':
                record['returned'] = self._snapshot(value, rank=1)
                self._remember('returned', value)
                record['consumer_completed'] = True
            elif role == 'ack':
                record['ack_count'] += 1
        self._frames.pop(role, None)

    def _profile(self, frame, event, arg):
        if event == 'c_call' and arg is sys.setprofile and not self._uninstalling:
            self._error = 'Profiler coverage gap observed'
        if self._error is not None or event not in ('call', 'return'):
            return
        try:
            role = next((key for key, function in self._definition['functions'].items()
                         if frame.f_code is function.__code__), None)
            if role is None:
                return
            if role == 'tick':
                if frame.f_locals.get('engine') is not self._engine or frame.f_locals.get('interpolator') is not self._interpolator:
                    return
                if event == 'call':
                    self._start_tick(frame)
                elif frame is self._frames.get('tick'):
                    self._return(role, frame, arg)
                return
            if self._tick is None or not self._inside(frame, 'tick'):
                return
            owners = {'engine': self._engine, 'ack': self._engine, 'select': self._engine._policy,
                      'pipeline': self._engine._postprocessor, 'add': self._interpolator, 'get': self._interpolator}
            if role in ('step0', 'step1', 'transform'):
                owners[role] = self._engine._postprocessor.steps[0 if role == 'transform' else int(role[-1])]
            if role in owners and frame.f_locals.get('self') is not owners[role]:
                return
            if role == 'consumer' and self._definition['consumer_owner'] is not None and frame.f_locals.get(self._definition.get('consumer_owner_argument') or 'self') is not self._definition['consumer_owner']:
                return
            if role in ('step0', 'step1') and not self._inside(frame, 'pipeline'):
                return
            if role == 'transform' and (not self._inside(frame, 'step0') or frame.f_locals.get('key') != 'action'):
                return
            if role in ('select', 'pipeline') and not self._inside(frame, 'engine'):
                return
            if event == 'call':
                self._entry(role, frame)
            elif frame is self._frames.get(role):
                self._return(role, frame, arg)
        except Exception as exc:
            self._error = f'Command capture failed: {type(exc).__name__}: {str(exc)[:512]}'
            self._frames.clear()
            self._live.clear()

    def _write(self):
        self._check_runtime()
        definition = self._definition
        for name, path in definition['paths'].items():
            if _bytes(path) != definition['sources'][name]:
                raise InputError('Native command source changed during capture')
        output = _destination(self._output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.command-path-', dir=output.parent) as temporary:
            stage = Path(temporary)
            (stage / 'sources').mkdir()
            np.savez(stage / 'arrays.npz', **self._arrays)
            for name, raw in definition['sources'].items():
                if not name or Path(name).name != name:
                    raise InputError('Unsafe native source filename')
                (stage / 'sources' / name).write_bytes(raw)
            files = [{'path': str(path.relative_to(stage)), 'sha256': _sha(path.read_bytes())}
                     for path in sorted(stage.rglob('*')) if path.is_file()]
            packet = {'schema_version': 1, 'kind': 'command_path_capture', 'domain': 'software_consumer',
                      'source': {**definition['source'], 'files': {name: {'sha256': _sha(raw)} for name, raw in definition['sources'].items()}},
                      'action_names': self._names, 'consumer_namespace': self._namespace,
                      'coverage': {'target': 'one_sync_engine_dispatch_on_owner_thread', 'complete_target_call_history': True,
                                   'external_state_mutations': 'untracked'},
                      'ticks': self._ticks,
                      'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in self._arrays.items()},
                      'files': files, 'provenance': {'origin': 'caller-runtime-evidence-not-attestation',
                      'prior_query_lineage': 'unknown', 'recorded_action': 'unknown', 'physical_execution': 'unknown'}, 'scope': SCOPE}
            raw = json.dumps(packet, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
            if len(raw) > MAX_MANIFEST or len(raw) + sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()) > MAX_BYTES:
                raise InputError('Command manifest or total payload exceeds capture bounds')
            (stage / 'capture.json').write_bytes(raw)
            (stage / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': _sha(raw)}))
            publish_capture(stage, output)
        return {'kind': 'command_path_capture_receipt', 'status': 'captured', 'published': True,
                'directory': str(output), 'tick_count': len(self._ticks), 'domain': 'software_consumer'}

    def _clear(self):
        self._frames.clear()
        self._live.clear()
        self._arrays.clear()
        self._ticks.clear()
        self._tokens.clear()
        self._engine = self._interpolator = self._consumer = self._definition = self._tick = self._hook = self._runtime_guard = None

    def __exit__(self, exc_type, exc, tb):
        if threading.get_ident() != self._owner:
            raise InputError('Command capture must exit on the owner thread')
        owned = sys.getprofile() is self._hook
        if owned:
            self._uninstalling = True
            sys.setprofile(None)
        _ACTIVE.active = False
        try:
            if exc_type is not None:
                self.receipt = self._unpublished('Caller exception preserved; no capture published')
            elif not owned:
                self.receipt = self._unpublished('Profiler replaced; replacement preserved')
            elif self._error or self._tick is not None or not self._ticks:
                self.receipt = self._unpublished(self._error or 'No complete dispatch tick captured')
            else:
                self.receipt = self._write()
        except Exception as failure:
            self.receipt = self._unpublished(f'Publication failed: {type(failure).__name__}: {str(failure)[:512]}')
        finally:
            self._clear()
        return False


def capture_command_path(engine, interpolator, consumer, output, *, action_names, consumer_namespace,
                         allow_tensor_snapshot=False):
    """Observe normal native dispatch_tick calls without invoking native/consumer code.

    Tensor CPU snapshots require explicit consent. A consumer returning None can
    complete successfully; missing returned values are not fabricated feedback.
    Inspect the context receipt after exit, separately from native call results.
    """
    return _CommandCapture(engine, interpolator, consumer, output, action_names=action_names,
                           consumer_namespace=consumer_namespace, allow_tensor_snapshot=allow_tensor_snapshot)
