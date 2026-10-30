"""Opt-in read-only capture around a normal, already initialized IsaacLab manager.

No native runtime import, config replacement or manager invocation is performed
here. The caller executes its normal compute call inside a one-shot context.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
import threading
from pathlib import Path

from . import isaac_layout as producer
from .artifacts import InputError

MANAGER_SHA256 = '969adad80aaf837e4c95b05c40809d70ecd514ded9d6a2ef85ad3ec6c649a393'


def _manager_definition(manager):
    method = getattr(manager, 'compute_group', None)
    if (not inspect.ismethod(method) or method.__self__ is not manager
            or not inspect.isfunction(method.__func__)):
        raise InputError('Capture requires the supported native ObservationManager.compute_group')
    function = method.__func__
    if (function.__module__ != 'isaaclab.managers.observation_manager'
            or function.__qualname__ != 'ObservationManager.compute_group'):
        raise InputError('Capture requires the supported native observation-manager source')
    try:
        path = Path(inspect.getsourcefile(function))
        if path.stat().st_size > producer.MAX_SOURCE_BYTES:
            raise InputError('Native observation-manager source exceeds capture bounds')
        if hashlib.sha256(path.read_bytes()).hexdigest() != MANAGER_SHA256:
            raise InputError('Native observation-manager source revision is unsupported')
    except (TypeError, OSError) as exc:
        raise InputError('Cannot verify native observation-manager source') from exc
    return function


def _member(container, name):
    return container[name] if isinstance(container, dict) else getattr(container, name)


class _ObservationCapture:
    def __init__(self, manager, output, *, group, term, env_indices, allow_tensor_snapshot):
        self._manager = manager
        self._output = output
        self._group = producer._text(group, 'Observation group')
        self._term = producer._text(term, 'Observation term')
        if type(allow_tensor_snapshot) is not bool:
            raise InputError('allow_tensor_snapshot must be a Boolean')
        if env_indices is not None and (
                not isinstance(env_indices, (list, tuple)) or not 1 <= len(env_indices) <= 32
                or any(type(index) is not int or index < 0 for index in env_indices)
                or len(set(env_indices)) != len(env_indices)):
            raise InputError('env_indices must contain 1–32 distinct nonnegative native integers')
        self._allow = allow_tensor_snapshot
        self._rows = list(env_indices) if isinstance(env_indices, (list, tuple)) else env_indices
        self._used = False
        self._owner = None
        self._group_frame = None
        self._term_frame = None
        self._payload = None
        self._error = None
        self._group_calls = 0
        self._term_calls = 0
        self._completed_groups = 0
        self.receipt = self._unpublished('No normal selected manager call captured')

    @staticmethod
    def _unpublished(reason):
        return {'kind': 'layout_capture_not_published', 'status': 'unknown',
                'published': False, 'reason': reason}

    def __enter__(self):
        if self._used:
            raise InputError('Observation capture contexts are single use')
        if getattr(producer._ACTIVE_CALL, 'active', False):
            raise InputError('An active producer capture prevents nested instrumentation')
        if sys.getprofile() is not None:
            raise InputError('An existing per-thread profiler prevents manager capture')
        self._group_function = _manager_definition(self._manager)
        try:
            terms = self._manager.active_terms[self._group]
            if (not isinstance(terms, (list, tuple)) or len(terms) > 256
                    or terms.count(self._term) != 1):
                raise InputError('Selected observation term must be uniquely active in the group')
            self._cfg = _member(_member(self._manager.cfg, self._group), self._term)
            self._function = self._cfg.func
            if (not inspect.isfunction(self._function) or inspect.isgeneratorfunction(self._function)
                    or inspect.iscoroutinefunction(self._function)):
                raise InputError('Selected observation term must be a normal trusted Python function')
        except (AttributeError, KeyError, TypeError) as exc:
            raise InputError('Selected native observation group/term is unavailable') from exc
        self._used = True
        self._owner = threading.get_ident()
        self._profile_hook = self._profile
        producer._ACTIVE_CALL.active = True
        try:
            sys.setprofile(self._profile_hook)
        except BaseException:
            producer._ACTIVE_CALL.active = False
            self._clear()
            raise
        return self

    def _profile(self, frame, event, arg):
        try:
            if frame.f_code is self._group_function.__code__:
                local = frame.f_locals
                if local.get('self') is not self._manager or local.get('group_name') != self._group:
                    return
                if event == 'call':
                    self._group_calls += 1
                    if self._group_calls != 1:
                        self._error = 'More than one selected group invocation prevents one-shot binding'
                    else:
                        self._group_frame = frame
                elif event == 'return' and frame is self._group_frame:
                    if arg is None:
                        self._error = self._error or 'Selected group did not return normally'
                    elif self._error is None:
                        self._finish_group(local)
                    self._group_frame = None
                return
            if frame.f_code is not self._function.__code__ or self._group_frame is None:
                return
            if event == 'call':
                if self._term_frame is not None:
                    self._error = 'Recursive selected term invocation prevents binding'
                    return
                caller = frame.f_back
                if (caller is not self._group_frame or caller.f_locals.get('term_name') != self._term):
                    return
                if caller.f_locals.get('term_cfg') is not self._cfg:
                    self._error = 'Selected runtime term config differs from public manager config'
                    return
                self._term_calls += 1
                if self._term_calls != 1:
                    self._error = 'More than one selected term invocation prevents binding'
                else:
                    self._term_frame = frame
            elif event == 'return' and frame is self._term_frame:
                if arg is None:
                    self._error = self._error or 'Selected term did not return a supported normal value'
                elif self._error is None:
                    local = frame.f_locals
                    if not all(key in local for key in
                               ('sensor', 'sensor_cfg', 'sensor_data', 'force_data', 'torque_data')):
                        raise InputError('Selected term return lacks cached native wrench locals')
                    self._payload = producer._collect_capture(
                        sensor=local['sensor'], sensor_cfg=local['sensor_cfg'],
                        cached_data=local['sensor_data'], raw_term_output=arg,
                        term_function=self._function, term_cfg=self._cfg,
                        allow_tensor_snapshot=self._allow, env_indices=self._rows,
                        cached_fields={'force': local['force_data'], 'torque': local['torque_data']})
                self._term_frame = None
        except Exception as exc:
            self._error = f'Manager snapshot failed: {type(exc).__name__}: {str(exc)[:512]}'
            self._term_frame = None

    def _finish_group(self, local):
        if self._payload is None or self._term_calls != 1:
            raise InputError('No unique selected native term return was captured')
        metadata, arrays, sources = self._payload
        group_obs = local.get('group_obs')
        if not isinstance(group_obs, dict) or self._term not in group_obs:
            raise InputError('Normal native group return lacks the selected processed term')
        value = group_obs[self._term]
        shape = producer._native_shape(value, 'Processed manager term', 2, self._allow)
        if shape != (metadata['runtime']['source_num_envs'], arrays['raw_term_output'].shape[1]):
            raise InputError('Processed manager term shape differs from its raw term')
        arrays['processed_term_output'] = producer._snapshot(
            value, 'Processed manager term', 2, self._allow, metadata['runtime']['env_indices'])
        identity = producer._source_identity(self._group_function, sources, {})
        if identity['source_file'] is None or identity['sha256'] != MANAGER_SHA256:
            raise InputError('Original observation-manager source changed or could not be retained')
        metadata['source']['files'] = {
            name: {'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in sources.items()}
        metadata['provenance'].update(
            capture_method='profiled_manager_and_selected_term_returns',
            call_boundary='selected_term_return_then_normal_group_return',
            same_read_context='captured_original_selected_term_frame_locals',
            target_function_calls=self._term_calls,
            manager_observation={'group': self._group, 'term': self._term,
                                 'source': identity, 'selected_group_calls': self._group_calls,
                                 'processed_source': 'original_compute_group.group_obs[selected_term]',
                                 'group_or_policy_binding': 'unknown'})
        self._completed_groups += 1

    def _clear(self):
        self._group_frame = self._term_frame = self._payload = None
        self._manager = self._cfg = self._function = self._group_function = None
        self._profile_hook = None

    def __exit__(self, exc_type, exc, tb):
        if threading.get_ident() != self._owner:
            raise InputError('Manager capture must exit on its owning thread')
        owned = sys.getprofile() is self._profile_hook
        if owned:
            sys.setprofile(None)
        producer._ACTIVE_CALL.active = False
        self._group_frame = self._term_frame = None
        try:
            if exc_type is not None:
                self.receipt = self._unpublished('Caller exited with an exception; staged capture discarded')
            elif not owned:
                self.receipt = self._unpublished('Capture profiler replaced; replacement preserved')
            elif (self._error is not None or self._group_calls != 1 or self._term_calls != 1
                  or self._completed_groups != 1 or self._payload is None):
                self.receipt = self._unpublished(self._error or 'No unique normal selected group/term completed')
            else:
                metadata, arrays, sources = self._payload
                self.receipt = producer._write_capture(self._output, metadata, arrays, sources=sources)
                self.receipt['published'] = True
        except Exception as failure:
            self.receipt = self._unpublished(f'Publication failed: {type(failure).__name__}: {str(failure)[:512]}')
        finally:
            self._clear()
        return False


def capture_isaac_observation(manager, output, *, group, term,
                              env_indices=None, allow_tensor_snapshot=False):
    """Observe one normal compute call after native manager initialization.

    The context does not call the manager or replace its config/function. The
    original full result and exceptions remain the caller's. Read ``receipt``
    after exit; capture/publication failure is separate from native completion.
    Saved raw/processed term evidence does not certify a whole group, policy
    input, physical sampling identity or authenticated execution.
    """
    return _ObservationCapture(manager, output, group=group, term=term,
                               env_indices=env_indices, allow_tensor_snapshot=allow_tensor_snapshot)
