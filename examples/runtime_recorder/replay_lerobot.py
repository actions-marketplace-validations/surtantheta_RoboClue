"""Replay actual LeRobot-format public simulation rows; never connect hardware.

    python examples/runtime_recorder/replay_lerobot.py --dataset LOCAL_EXPORT --output NEW_FOLDER

The optional nominal timeline is explicitly a recorded frame-index timeline,
not live execution duration or camera acquisition. See RUNTIME_RECORDER.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from policy_doctor.flight_recorder import triage_flight_recording
from policy_doctor.runtime_recorder import RuntimeRecorder, SignalSpec


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RecordedInterface:
    """Recorded software interface, explicitly not an executed native Robot."""

    def __init__(self, rows, names, state_names, frames, scenario):
        self.rows, self.names, self.state_names, self.frames = rows, names, state_names, frames
        self.scenario, self.index = scenario, 0
        self.calls, self.returned, self.observations = 0, [], []

    def send_action(self, action):
        self.calls += 1
        # A controlled software adapter fault; never label it a real robot fault.
        result = dict(action)
        if self.scenario == 'fault' and self.index == 7:
            result[self.names[0]] += 1.0
        self.returned.append(result)
        return result

    def get_observation(self):
        row = self.rows[self.index]
        result = dict(zip(self.state_names, row['observation.state'], strict=True))
        if self.frames:
            result['ego_view'] = self.frames[self.index]
        self.observations.append(result)
        return result


def run(dataset, output, *, rows=32, video=None):
    import pyarrow.parquet as pq

    dataset, output = Path(dataset), Path(output)
    if not 8 <= rows <= 128:
        raise ValueError('Choose between 8 and 128 public rows')
    output.mkdir(exist_ok=False)
    info_path = dataset / 'meta/info.json'
    info = json.loads(info_path.read_text())
    parquet = dataset / 'data/chunk-000/episode_000000.parquet'
    inputs = {str(path): sha(path) for path in (info_path, parquet)}
    table = pq.read_table(parquet, columns=['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index'])
    selected = table.slice(0, rows).to_pylist()
    assert len(selected) == rows
    names = info['features']['action']['names']
    state_names = info['features']['observation.state']['names']
    frames = []
    if video is not None:
        import av

        video = Path(video)
        inputs[str(video)] = sha(video)
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format='rgb24'))
                if len(frames) == rows:
                    break
        assert len(frames) == rows
    source_spec = SignalSpec(tuple(names), ('stored_action_value',) * len(names), 'recorded_action_value')
    unknown_spec = SignalSpec(tuple(names), (None,) * len(names), 'recorded_action_value')
    state_spec = SignalSpec(tuple(state_names), (None,) * len(state_names), 'recorded_state_value')
    # This criterion is independent of every candidate and is never regenerated
    # from the injected faulty return. Software-value identity is not a unit claim.
    criteria = {'max_command_gap_s': .03, 'max_feedback_gap_s': .03,
                'max_camera_gap_s': .03, 'max_latency_s': .03,
                'tracking_tolerance': {name: {'unit': 'stored_action_value', 'absolute': 0.0} for name in names}}
    criterion_path = output / 'criteria.json'
    criterion_path.write_text(json.dumps(criteria, sort_keys=True, separators=(',', ':')) + '\n')
    criterion_pin = sha(criterion_path)
    (output / 'criterion-retained-before-candidates.sha256').write_text(criterion_pin + '\n')
    results = {}
    for scenario in ('healthy', 'fault', 'corrected', 'unknown', 'exception'):
        native = RecordedInterface(selected, names, state_names, frames, scenario)
        known = scenario != 'unknown'
        spec = source_spec if known else unknown_spec
        original_error = RuntimeError('controlled replay producer exception; no hardware')
        caught = False
        path = output / (scenario + '.jsonl')
        try:
            with RuntimeRecorder(native, path, action_spec=spec, returned_spec=spec, policy_spec=spec,
                observation_spec=state_spec, camera_keys=('ego_view',) if frames else (),
                timestamp=lambda native=native: selected[native.index]['timestamp'],
                clock={'unit': 's', 'domain': 'declared_recorded_episode_timeline' if known else None},
                provenance={'kind': 'public_recorded_simulation_software_interface_replay',
                    'source_sha256': inputs[str(parquet)], 'metadata_sha256': inputs[str(info_path)],
                    'timestamp_origin': 'original_stored_dataset_values_not_live_execution',
                    'physical_robot_execution': False, 'scenario': scenario}) as recording:
                for index, row in enumerate(selected):
                    native.index = index
                    action = dict(zip(names, row['action'], strict=True))
                    policy_id = recording.record_policy_command(action)
                    returned = recording.send_action(action, policy_command_id=policy_id)
                    assert returned is native.returned[-1]
                    observation = recording.get_observation(frame_ids={'ego_view': str(row['frame_index'])})
                    assert observation is native.observations[-1]
                    if scenario == 'exception' and index == 7:
                        raise original_error
        except RuntimeError as error:
            assert scenario == 'exception' and error is original_error
            caught = True
        assert caught is (scenario == 'exception')
        report = triage_flight_recording(path, **criteria)
        tracking_warnings = [f for f in report['findings'] if f['code'] == 'tracking' and f['status'] == 'warning']
        if scenario == 'fault':
            assert len(tracking_warnings) == 1
            assert tracking_warnings[0]['details']['absolute_residual'] == 1.0
            assert tracking_warnings[0]['details']['joint'] == names[0]
        else:
            assert not tracking_warnings and report['counts']['warning'] == 0
        if scenario == 'unknown':
            assert all(f['status'] == 'unknown' for f in report['findings'] if f['code'] in ('latency', 'tracking'))
        if scenario == 'exception':
            assert report['coverage']['requested_run_coverage'] == 'unknown'
            assert report['termination']['status'] == 'producer_exception'
        # Physical observations are never implicitly associated with returned commands.
        assert any(f['code'] == 'feedback_association' and f['status'] == 'unknown' for f in report['findings'])
        (output / (scenario + '-triage.json')).write_text(json.dumps(report, indent=2) + '\n')
        results[scenario] = {'events': report['recording']['events'], 'counts': report['counts'],
                             'status': report['status'], 'calls': native.calls,
                             'tracking_warning_events': [f['event_ids'] for f in tracking_warnings]}
        assert sha(criterion_path) == criterion_pin
    assert all(sha(path) == pin for path, pin in inputs.items())
    summary = {'status': 'PASS_PUBLIC_RECORDED_INTERFACE_RUNTIME_LOOP', 'rows': rows,
               'actual_image_arrays_decoded': len(frames), 'source_inputs_postchecked': inputs,
               'criterion_sha256': criterion_pin, 'scenarios': results,
               'native_robot_or_policy_execution': False, 'physical_feedback_association_inferred': False,
               'fault_scope': 'Injected one-coordinate software send_action return corruption on actual public action rows',
               'timeline_scope': 'Explicit nominal recorded timeline; not measured API duration or sensor acquisition'}
    (output / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--rows', type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, rows=args.rows, video=args.video), indent=2))
