import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import InputError
from policy_doctor.dispatch import audit_dispatch, make_dispatch_event, snapshot_action


@pytest.fixture
def boundary(tmp_path):
    recording = tmp_path / 'data.parquet'
    # Native frame IDs need not equal selected-file row offsets or row order.
    pq.write_table(pa.table({'episode_index': [2, 2], 'frame_index': [101, 100], 'action': [[2., 0.], [2., 0.]]}), recording)
    events = [make_dispatch_event(str(frame), frame_index=frame,
        canonical_dispatch={'base.vx': 2., 'base.wz': 0.}, canonical_endpoint={'base.vx': 2., 'base.wz': 0.},
        driver_input={'base.vx': 2., 'base.wz': 0.}, driver_return={'base.vx': .5, 'base.wz': 0.},
        recorded_role='driver_return') for frame in [100, 101]]
    trace = tmp_path / 'trace.json'
    data = {'schema_version': 1, 'kind': 'action_dispatch_evidence', 'events': events,
            'recording_sha256': hashlib.sha256(recording.read_bytes()).hexdigest(), 'episode_index': 2}

    def put():
        trace.write_text(json.dumps(data))
        return trace

    def audit(**kwargs):
        put()
        return audit_dispatch(str(trace), str(recording), episode=2, action_names=['base.vx', 'base.wz'], **kwargs)
    return {'data': data, 'recording': recording, 'trace': trace, 'put': put, 'audit': audit}


def test_native_wrong_recording_role_localizes_first_contradiction(boundary):
    r = boundary['audit']()
    assert r['status'] == 'error' and r['counts'] == {'pass': 0, 'error': 2, 'unknown': 0}
    first = r['first_recording_contradiction']
    assert first['frame_index'] == 100 and first['selected_episode_row'] == 1
    assert first['recording_difference_preview'] == [{'channel': 'base.vx', 'expected': .5, 'observed': 2.}]
    assert first['matches_roles'] == ['canonical_dispatch', 'canonical_endpoint', 'driver_input']
    assert r['driver_change_events'] == 2 and r['physical_cause'] == 'unknown'


def test_intentional_canonical_recording_is_healthy_despite_driver_clipping(boundary):
    for event in boundary['data']['events']:
        event['recorded_role'] = 'canonical_endpoint'
    r = boundary['audit']()
    assert r['status'] == 'pass' and r['driver_change_events'] == 2


def test_corrected_native_recording_and_fresh_binding_pass(boundary):
    p = boundary['recording']
    pq.write_table(pa.table({'episode_index': [2, 2], 'frame_index': [101, 100], 'action': [[.5, 0.], [.5, 0.]]}), p)
    boundary['data']['recording_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert boundary['audit']()['status'] == 'pass'


def test_intermediate_dispatch_is_distinct_from_recorded_endpoint(boundary):
    for event in boundary['data']['events']:
        event['canonical_dispatch']['base.vx'] = 1.
        event['recorded_role'] = 'canonical_endpoint'
    assert boundary['audit']()['status'] == 'pass'


@pytest.mark.parametrize('missing', ['role', 'driver_return', 'frame', 'binding', 'episode', 'names'])
def test_missing_evidence_stays_unknown(boundary, missing):
    data = boundary['data']
    if missing == 'binding':
        data['recording_sha256'] = None
    elif missing == 'episode':
        data['episode_index'] = None
    for event in data['events']:
        if missing == 'role':
            event['recorded_role'] = 'unknown'
        elif missing == 'driver_return':
            event['driver_return'] = None
        elif missing == 'frame':
            event['frame_index'] = None
        elif missing == 'names':
            event['driver_return'] = {'different.namespace': .5}
    assert boundary['audit']()['status'] == 'unknown'


def test_without_native_recording_cannot_certify_saved_action(boundary):
    boundary['put']()
    r = audit_dispatch(str(boundary['trace']))
    assert r['status'] == 'unknown' and r['native_recording_rows'] == 0


@pytest.mark.parametrize('fault', ['hash', 'episode', 'duplicate_event', 'duplicate_frame', 'new_schema', 'command'])
def test_ambiguous_or_unbound_inputs_refused(boundary, fault):
    data = boundary['data']
    if fault == 'hash':
        data['recording_sha256'] = '0' * 64
    elif fault == 'episode':
        data['episode_index'] = 3
    elif fault == 'duplicate_event':
        data['events'][1]['event_id'] = data['events'][0]['event_id']
    elif fault == 'duplicate_frame':
        data['events'][1]['frame_index'] = data['events'][0]['frame_index']
    elif fault == 'new_schema':
        data['schema_version'] = 2
    else:
        data['command'] = 'never execute'
    with pytest.raises(InputError):
        boundary['audit']()


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), 2**53 + 1, [1.], '2'])
def test_snapshot_refuses_unsafe_or_ambiguous_values(value):
    with pytest.raises(InputError):
        snapshot_action({'joint': value})


def test_capture_snapshots_native_dictionary_before_driver_mutation():
    action = {'base.vx': 2.}
    before = snapshot_action(action)
    action['base.vx'] = .5
    event = make_dispatch_event('tick', driver_input=before, driver_return=action)
    action['base.vx'] = 0.
    assert event['driver_input'] == {'base.vx': 2.} and event['driver_return'] == {'base.vx': .5}


@pytest.mark.parametrize('fault', ['duplicate_frames', 'wrong_width', 'float_frames'])
def test_native_recording_identity_and_layout_are_not_guessed(boundary, fault):
    p = boundary['recording']
    frames = [100, 100] if fault == 'duplicate_frames' else [100., 101.] if fault == 'float_frames' else [100, 101]
    actions = [[2.], [2.]] if fault == 'wrong_width' else [[2., 0.], [2., 0.]]
    pq.write_table(pa.table({'episode_index': [2, 2], 'frame_index': frames, 'action': actions}), p)
    boundary['data']['recording_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(InputError):
        boundary['audit']()


def test_actual_cli_and_read_only_mcp(boundary, tmp_path):
    boundary['put']()
    project = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(project / 'src'))
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'audit-dispatch', str(boundary['trace']),
        '--recording', str(boundary['recording']), '--episode', '2', '--action-names', 'base.vx', 'base.wz'],
        capture_output=True, text=True, env=env)
    assert result.returncode == 1 and 'expected 0.5; recorded 2.0' in result.stdout
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path)], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('audit_action_boundary', {'evidence': str(boundary['trace']),
                    'recording': str(boundary['recording']), 'episode': 2, 'action_names': ['base.vx', 'base.wz']})
                assert not result.isError and result.structuredContent['status'] == 'error'
                denied = await client.call_tool('audit_action_boundary', {'evidence': '/etc/passwd'})
                assert denied.isError
    asyncio.run(asyncio.wait_for(exercise(), 30))
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('field', ['atol', 'rtol', 'episode'])
def test_mcp_cannot_coerce_boolean_controls_into_numbers(boundary, tmp_path, field):
    p = boundary['recording']
    pq.write_table(pa.table({'episode_index': [1], 'frame_index': [100], 'action': [[2., 0.]]}), p)
    boundary['data']['recording_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    boundary['data']['episode_index'] = 1
    boundary['data']['events'] = boundary['data']['events'][:1]
    boundary['put']()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path)], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                args = {'evidence': str(boundary['trace']), 'recording': str(p), 'episode': 1,
                        'action_names': ['base.vx', 'base.wz'], field: True}
                result = await client.call_tool('audit_action_boundary', args)
                assert result.isError
    asyncio.run(asyncio.wait_for(exercise(), 30))
