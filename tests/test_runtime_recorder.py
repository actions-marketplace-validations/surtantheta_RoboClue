"""Controlled interface tests; actual public replay is a separate qualification."""
import json

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.flight_recorder import triage_flight_recording
from policy_doctor.runtime_recorder import RuntimeRecorder, SignalSpec


class Interface:
    def __init__(self):
        self.actions, self.observations = [], []
        self.returned = {'elbow.pos': 0.5}
        self.observation = {'elbow.pos': 0.45, 'front': np.zeros((3, 4, 3), dtype=np.uint8)}
        self.error = None

    def send_action(self, action, *args, **kwargs):
        self.actions.append((action, args, kwargs))
        if self.error:
            raise self.error
        return self.returned

    def get_observation(self, *args, **kwargs):
        self.observations.append((args, kwargs))
        if self.error:
            raise self.error
        return self.observation


def spec(unit='rad'):
    return SignalSpec(('elbow.pos',), (unit,), 'joint_position')


def recorder(tmp_path, robot=None, **kwargs):
    robot = robot or Interface()
    defaults = dict(action_spec=spec(), returned_spec=spec(), observation_spec=spec(), policy_spec=spec(),
                    camera_keys=('front',), provenance={'kind': 'controlled_interface_test'})
    defaults.update(kwargs)
    return RuntimeRecorder(robot, tmp_path / 'runtime.jsonl', **defaults)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()][1:-1]


def test_no_native_method_called_on_construction_enter_or_exit(tmp_path):
    robot = Interface()
    with recorder(tmp_path, robot):
        assert not robot.actions and not robot.observations
    assert not robot.actions and not robot.observations


def test_original_arguments_objects_and_results_preserved(tmp_path):
    robot, action = Interface(), {'elbow.pos': np.float32(0.5), 'extra': object()}
    with recorder(tmp_path, robot) as runtime:
        policy_id = runtime.record_policy_command(action)
        result = runtime.send_action(action, 'positional', policy_command_id=policy_id, mode='native')
        assert result is robot.returned
        assert robot.actions == [(action, ('positional',), {'mode': 'native'})]
        assert robot.actions[0][0] is action
        observed = runtime.get_observation('arg', associated_command_id=runtime.last_send.returned_command_id,
                                           frame_ids={'front': 'declared:frame'}, option=3)
        assert observed is robot.observation and robot.observations == [(('arg',), {'option': 3})]
    saved = events(tmp_path / 'runtime.jsonl')
    camera = next(item for item in saved if item['kind'] == 'camera')
    assert camera['metadata']['width'] == 4 and camera['metadata']['height'] == 3
    assert camera['metadata']['image_content_inspected'] is False
    assert camera['frame_id'] == 'declared:frame'


def test_default_feedback_association_remains_unknown_and_camera_is_software_identity(tmp_path):
    with recorder(tmp_path) as runtime:
        runtime.send_action({'elbow.pos': 0.5})
        runtime.get_observation()
    saved = events(tmp_path / 'runtime.jsonl')
    observed = next(item for item in saved if item['stream'] == 'get_observation.return')
    assert observed['command_id'] is None and observed['metadata']['association'] == 'unknown'
    camera = next(item for item in saved if item['kind'] == 'camera')
    assert camera['metadata']['frame_identity'] == 'software_observation_id_only'
    assert camera['metadata']['acquisition_timestamp_known'] is False
    report = triage_flight_recording(tmp_path / 'runtime.jsonl', max_latency_s=1,
        tracking_tolerance={'elbow.pos': {'unit': 'rad', 'absolute': .01}})
    assert any(f['code'] == 'feedback_association' and f['status'] == 'unknown' for f in report['findings'])


def test_real_api_return_difference_identifies_request_and_return_then_correction(tmp_path):
    for folder, returned in [('fault', .8), ('corrected', .5)]:
        root = tmp_path / folder
        root.mkdir()
        robot = Interface()
        robot.returned['elbow.pos'] = returned
        with recorder(root, robot) as runtime:
            runtime.send_action({'elbow.pos': .5})
            receipt = runtime.last_send
        report = triage_flight_recording(root / 'runtime.jsonl', max_latency_s=1,
            tracking_tolerance={'elbow.pos': {'unit': 'rad', 'absolute': .01}})
        warnings = [f for f in report['findings'] if f['code'] == 'tracking' and f['status'] == 'warning']
        if folder == 'fault':
            assert len(warnings) == 1
            assert warnings[0]['event_ids'] == [receipt.request_id, receipt.returned_feedback_id]
            assert warnings[0]['details']['absolute_residual'] == pytest.approx(.3)
            assert warnings[0]['details']['comparison_boundary'] == 'request_to_return'
            assert 'send request → software return' in warnings[0]['message']
            assert 'controller logs' not in warnings[0]['remedy']
        else:
            assert not warnings


@pytest.mark.parametrize('associated', [False, True])
def test_software_boundaries_do_not_verify_physical_feedback(tmp_path, associated):
    from policy_doctor.first_use_views import flight_view

    with recorder(tmp_path) as runtime:
        policy_id = runtime.record_policy_command({'elbow.pos': .5})
        runtime.send_action({'elbow.pos': .5}, policy_command_id=policy_id)
        runtime.get_observation(associated_command_id=runtime.last_send.returned_command_id if associated else None)
    report = triage_flight_recording(tmp_path / 'runtime.jsonl', max_latency_s=1,
        tracking_tolerance={'elbow.pos': {'unit': 'rad', 'absolute': .1}})
    view = flight_view(report)
    assert {r['boundary'] for r in view['software_comparisons']} == {'policy_to_request', 'request_to_return'}
    assert view['software_counts'] == {'pass': 2, 'warning': 0, 'unknown': 0}
    assert view['physical_feedback_association']['status'] == 'unknown'
    assert view['physical_feedback_association']['unassociated_observation_returns'] == (0 if associated else 1)
    assert report['status'] == 'unknown'


@pytest.mark.parametrize('method', ['send_action', 'get_observation'])
def test_original_native_error_identity_and_exception_safe_prefix(tmp_path, method):
    robot = Interface()
    original = RuntimeError('private robot failure detail')
    with pytest.raises(RuntimeError) as caught:
        with recorder(tmp_path, robot) as runtime:
            robot.error = original
            if method == 'send_action':
                runtime.send_action({'elbow.pos': .5})
            else:
                runtime.get_observation()
    assert caught.value is original
    assert len(robot.actions if method == 'send_action' else robot.observations) == 1
    report = triage_flight_recording(tmp_path / 'runtime.jsonl')
    assert report['termination']['status'] == 'producer_exception'
    assert report['coverage']['requested_run_coverage'] == 'unknown'
    assert b'private robot failure detail' not in (tmp_path / 'runtime.jsonl').read_bytes()


@pytest.mark.parametrize('action', [{}, {'elbow.pos': float('nan')}, {'elbow.pos': np.array([.5])}])
def test_invalid_request_never_dispatches_original(tmp_path, action):
    robot = Interface()
    with recorder(tmp_path, robot) as runtime:
        with pytest.raises(InputError):
            runtime.send_action(action)
    assert not robot.actions


def test_return_validation_failure_does_not_retry_native_command(tmp_path):
    robot = Interface()
    robot.returned = {}
    with pytest.raises(InputError, match='missing'):
        with recorder(tmp_path, robot) as runtime:
            runtime.send_action({'elbow.pos': .5})
    assert len(robot.actions) == 1
    report = triage_flight_recording(tmp_path / 'runtime.jsonl')
    assert report['termination']['status'] == 'producer_exception'


def test_unknown_units_are_not_inferred_from_pos_suffix(tmp_path):
    with recorder(tmp_path, action_spec=spec(None), returned_spec=spec(None)) as runtime:
        runtime.send_action({'elbow.pos': .5})
    report = triage_flight_recording(tmp_path / 'runtime.jsonl', max_latency_s=1,
        tracking_tolerance={'elbow.pos': {'unit': 'rad', 'absolute': .01}})
    assert all(f['status'] == 'unknown' for f in report['findings'] if f['code'] == 'tracking')


def test_clock_override_requires_declaration_and_preserves_unknown_domain(tmp_path):
    with pytest.raises(InputError, match='explicit unit/domain'):
        recorder(tmp_path, timestamp=lambda: 0)
    with pytest.raises(InputError, match='actual timestamp provider'):
        recorder(tmp_path, clock={'unit': 's', 'domain': None})
    with recorder(tmp_path, timestamp=lambda: 3, clock={'unit': 's', 'domain': None}) as runtime:
        runtime.send_action({'elbow.pos': .5})
    report = triage_flight_recording(tmp_path / 'runtime.jsonl', max_latency_s=1)
    assert all(f['status'] == 'unknown' for f in report['findings'] if f['code'] == 'latency')


def test_policy_instrumentation_is_explicit_and_bad_associations_refused(tmp_path):
    robot = Interface()
    with recorder(tmp_path, robot, policy_spec=None) as runtime:
        with pytest.raises(InputError, match='policy_spec'):
            runtime.record_policy_command({'elbow.pos': .5})
        with pytest.raises(InputError, match='policy_command_id'):
            runtime.send_action({'elbow.pos': .5}, policy_command_id='guessed')
        with pytest.raises(InputError, match='associated_command_id'):
            runtime.get_observation(associated_command_id='row:0')
    assert not robot.actions and not robot.observations


def test_inactive_or_reused_context_refused_without_native_call(tmp_path):
    robot = Interface()
    runtime = recorder(tmp_path, robot)
    with pytest.raises(InputError, match='context manager'):
        runtime.send_action({'elbow.pos': .5})
    with runtime:
        pass
    with pytest.raises(InputError, match='cannot be reused'):
        with runtime:
            pass
    assert not robot.actions


@pytest.mark.parametrize('value', [[], {}, True])
def test_malformed_association_ids_refused_before_native_calls(tmp_path, value):
    robot = Interface()
    with recorder(tmp_path, robot) as runtime:
        with pytest.raises(InputError, match='policy_command_id'):
            runtime.send_action({'elbow.pos': .5}, policy_command_id=value)
        with pytest.raises(InputError, match='associated_command_id'):
            runtime.get_observation(associated_command_id=value)
    assert not robot.actions and not robot.observations


def test_missing_declared_camera_cannot_silently_look_healthy(tmp_path):
    robot = Interface()
    del robot.observation['front']
    with pytest.raises(InputError, match='camera field is missing'):
        with recorder(tmp_path, robot) as runtime:
            runtime.get_observation()
    assert len(robot.observations) == 1
    report = triage_flight_recording(tmp_path / 'runtime.jsonl')
    assert report['termination']['status'] == 'producer_exception'
    assert report['coverage']['requested_run_coverage'] == 'unknown'


@pytest.mark.parametrize('kwargs', [{'camera_keys': ['front', 'front']}, {'camera_keys': 'front'},
                                    {'action_spec': None}, {'provenance': {}}])
def test_invalid_configuration_refused_before_dispatch(tmp_path, kwargs):
    with pytest.raises(InputError):
        recorder(tmp_path, **kwargs)
