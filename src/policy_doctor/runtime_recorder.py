"""Opt-in recording proxy for an existing synchronous Robot-style software loop.

Only explicit caller invocations dispatch send_action/get_observation. No robot
is connected, configured, retried or stopped here; no MCP control is exposed.
Software API return association is intrinsic. Physical feedback association and
units are never inferred from order, coordinate suffixes or numerical values.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .artifacts import InputError
from .flight_recorder import FlightRecorder, _clocks, _metadata, _text, _vector

LEROBOT_INTERFACE_REVISION = '5831498454e93c71d209ff2e5e823e434f1efd15'
LEROBOT_INTERFACE_GIT_BLOB = 'fa61ce9cee79f3593eb2980a0742e56b9eb3b180'


@dataclass(frozen=True)
class SignalSpec:
    """Explicit ordered flat-dictionary fields, coordinate units and semantics.

    Units may be None when unknown. Declaring the same field name at two
    observation points is a caller declaration, not robot identity verification.
    """

    fields: tuple[str, ...]
    units: tuple[str | None, ...]
    semantics: str

    def __post_init__(self):
        if type(self.fields) not in (list, tuple) or type(self.units) not in (list, tuple):
            raise InputError('SignalSpec fields and units require explicit sequences')
        names, units = tuple(self.fields), tuple(self.units)
        _vector(list(names), list(units), [0.0] * len(names), self.semantics)
        object.__setattr__(self, 'fields', names)
        object.__setattr__(self, 'units', units)

    def select(self, data):
        if not isinstance(data, Mapping):
            raise InputError('Runtime signals require the original flat action/observation mapping')
        values = []
        for name in self.fields:
            if name not in data:
                raise InputError(f'Declared runtime signal field is missing: {name!r}')
            value = data[name]
            if isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.integer):
                value = int(value)
            values.append(value)
        return _vector(list(self.fields), list(self.units), values, self.semantics)


@dataclass(frozen=True)
class SendReceipt:
    request_id: str
    returned_feedback_id: str
    returned_command_id: str
    policy_command_id: str | None


class RuntimeRecorder:
    """Observe explicit calls to an already-created synchronous robot interface.

    Compatible signatures are send_action(action)->flat mapping and
    get_observation()->flat mapping, as declared by LeRobot Robot. This is a
    proxy, not a Robot subclass or validation of any hardware implementation.
    The caller owns connection lifetime, producer exceptions and robot cleanup.
    """

    def __init__(self, robot, path, *, action_spec, returned_spec, observation_spec,
                 provenance, camera_keys=(), policy_spec=None, timestamp=None,
                 clock=None, max_events=10000):
        if not all(callable(getattr(robot, name, None)) for name in ('send_action', 'get_observation')):
            raise InputError('Use an existing robot interface with send_action and get_observation methods')
        if any(not isinstance(spec, SignalSpec) for spec in (action_spec, returned_spec, observation_spec)):
            raise InputError('Declare action_spec, returned_spec and observation_spec as SignalSpec objects')
        if policy_spec is not None and not isinstance(policy_spec, SignalSpec):
            raise InputError('policy_spec must be an explicit SignalSpec')
        if type(camera_keys) not in (list, tuple) or len(camera_keys) > 16:
            raise InputError('Declare at most 16 camera keys')
        camera_keys = tuple(_text(key, 'Camera observation key') for key in camera_keys)
        if len(set(camera_keys)) != len(camera_keys):
            raise InputError('Camera observation keys must be distinct')
        if timestamp is None:
            if clock is not None:
                raise InputError('Custom clock declarations require their actual timestamp provider')
            timestamp = time.monotonic_ns
            clock = {'unit': 'ns', 'domain': 'runtime_host_monotonic'}
            self.timestamp_origin = 'local_host_monotonic_software_observation'
        else:
            if not callable(timestamp) or clock is None:
                raise InputError('Custom timestamps require a callable provider and explicit unit/domain')
            self.timestamp_origin = 'caller_supplied_timestamp_provider'
        self.clock = _clocks({'runtime': clock})['runtime']
        provenance = _metadata(provenance)
        if not provenance:
            raise InputError('Explicit runtime recording provenance is required')
        self.robot = robot
        self.action_spec, self.returned_spec, self.observation_spec = action_spec, returned_spec, observation_spec
        self.policy_spec, self.camera_keys, self.timestamp = policy_spec, camera_keys, timestamp
        self.flight = FlightRecorder(path, clocks={'runtime': self.clock}, provenance=provenance,
                                     max_events=max_events)
        self.active, self.used = False, False
        self.sequence = 0
        self.policy_ids, self.sent_ids = set(), set()
        self.last_send = None

    def _active(self):
        if not self.active:
            raise InputError('Use RuntimeRecorder inside its context manager')

    def _id(self, prefix):
        self.sequence += 1
        return f'{prefix}:{self.sequence}'

    def __enter__(self):
        if self.used:
            raise InputError('RuntimeRecorder cannot be reused')
        self.used = True
        self.flight.__enter__()
        try:
            self.flight.interface('runtime:interface', timestamp=self.timestamp(), clock='runtime',
                stream='runtime_recorder', metadata={
                    'api': 'send_action(action)->mapping;get_observation()->mapping',
                    'timestamp_origin': self.timestamp_origin,
                    'observation_point': 'software_call_or_return_not_sensor_acquisition',
                    'policy_output_requires_explicit_call': True,
                    'physical_feedback_association': 'caller_explicit_or_unknown',
                    'camera_content_inspected': False,
                })
        except BaseException as error:
            self.flight.__exit__(type(error), error, error.__traceback__)
            raise
        self.active = True
        return self

    def record_policy_command(self, action):
        """Record caller-selected policy output at this explicit observation point."""
        self._active()
        if self.policy_spec is None:
            raise InputError('Declare policy_spec before recording policy output')
        event_id = self._id('policy')
        self.flight.command(event_id, timestamp=self.timestamp(), clock='runtime', stream='policy.command',
                            metadata={'observation_point': 'caller_policy_output'}, **self.policy_spec.select(action))
        self.policy_ids.add(event_id)
        return event_id

    def send_action(self, action, *args, policy_command_id=None, **kwargs):
        """Call the original once, recording its request and actual returned mapping.

        Logging may raise; no command is retried. In particular an error after
        the original call does not mean that no hardware command was sent.
        """
        self._active()
        if policy_command_id is not None:
            _text(policy_command_id, 'policy_command_id')
            if policy_command_id not in self.policy_ids:
                raise InputError('policy_command_id must name a policy output recorded in this context')
        requested = self.action_spec.select(action)
        request_id = self._id('send-request')
        before = self.timestamp()
        self.flight.command(request_id, timestamp=before, clock='runtime', stream='send_action.request',
                            metadata={'observation_point': 'before_original_send_action'}, **requested)
        if policy_command_id is not None:
            self.flight.feedback(self._id('policy-request'), command_id=policy_command_id,
                timestamp=before, clock='runtime', stream='policy_to_send_request',
                metadata={'association': 'explicit_caller_policy_to_request_not_physical_feedback'}, **requested)
        returned = self.robot.send_action(action, *args, **kwargs)
        after = self.timestamp()
        values = self.returned_spec.select(returned)
        returned_feedback_id = self._id('send-return')
        self.flight.feedback(returned_feedback_id, command_id=request_id, timestamp=after,
            clock='runtime', stream='send_action.return',
            metadata={'association': 'same_original_send_action_call_return', 'physical_delivery_verified': False}, **values)
        sent_id = self._id('returned-command')
        self.flight.command(sent_id, timestamp=after, clock='runtime', stream='returned_software_command',
            metadata={'observation_point': 'original_send_action_return_not_motor_measurement'}, **values)
        self.sent_ids.add(sent_id)
        self.last_send = SendReceipt(request_id, returned_feedback_id, sent_id, policy_command_id)
        return returned

    def get_observation(self, *args, associated_command_id=None, frame_ids=None, **kwargs):
        """Call the original once; physical feedback association defaults UNKNOWN.

        A supplied association is a caller declaration. Clock values identify
        software return time, not camera exposure or device acquisition time.
        """
        self._active()
        if associated_command_id is not None:
            _text(associated_command_id, 'associated_command_id')
            if associated_command_id not in self.sent_ids:
                raise InputError('associated_command_id must name a returned command recorded in this context')
        if frame_ids is not None:
            if type(frame_ids) is not dict or not set(frame_ids) <= set(self.camera_keys):
                raise InputError('frame_ids must map declared camera keys to caller-supplied frame identities')
            for value in frame_ids.values():
                _text(value, 'Caller-supplied camera frame identity')
        observation = self.robot.get_observation(*args, **kwargs)
        after = self.timestamp()
        values = self.observation_spec.select(observation)
        observation_id = self._id('observation')
        self.flight.feedback(observation_id, command_id=associated_command_id, timestamp=after,
            clock='runtime', stream='get_observation.return',
            metadata={'association': 'explicit_caller_declaration' if associated_command_id else 'unknown',
                      'observation_point': 'software_return_not_sensor_acquisition'}, **values)
        for key in self.camera_keys:
            if key not in observation:
                raise InputError(f'Declared runtime camera field is missing: {key!r}')
            frame = observation[key]
            if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or not all(frame.shape):
                raise InputError('Declared camera fields require returned NumPy image arrays with explicit dimensions')
            frame_id = frame_ids.get(key) if frame_ids else None
            self.flight.camera(self._id('camera'), timestamp=after, clock='runtime', stream=key,
                frame_id=frame_id or f'{observation_id}/{key}', metadata={
                    'height': int(frame.shape[0]), 'width': int(frame.shape[1]),
                    'channels': int(frame.shape[2]) if frame.ndim == 3 else 1,
                    'frame_identity': 'caller_declared' if frame_id else 'software_observation_id_only',
                    'image_content_inspected': False, 'acquisition_timestamp_known': False,
                })
        return observation

    def __exit__(self, exc_type, exc, traceback):
        try:
            return self.flight.__exit__(exc_type, exc, traceback)
        finally:
            self.active = False
