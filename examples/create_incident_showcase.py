"""Generate a public, synthetic saved-decoding investigation; no policy/robot is run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file

from policy_doctor.capture import capture_runtime
from policy_doctor.investigation import investigate_run
from policy_doctor.reports import markdown_report
from policy_doctor.rollouts import compare_rollouts
from policy_doctor.runtime_diff import compare_runtime_captures
from policy_doctor.trace import trace_actions


class FixturePipeline:
    """Trusted in-memory fixture implementing the capture API, not a real robot pipeline."""
    def __init__(self, seed: Path, side: str, shift: float):
        self.config = json.loads((seed / 'checkpoint' / f'policy_{side}processor.json').read_text())
        self.states = {step['state_file'][:-12]: load_file(str(seed / 'checkpoint' / step['state_file']))
                       for step in self.config['steps'] if step.get('state_file')}
        for state in self.states.values():
            state['action.mean'][0] += shift
        self.steps = [SimpleNamespace() for _ in self.config['steps']]

    def get_config(self):
        return copy.deepcopy(self.config)

    def state_dict(self):
        return copy.deepcopy(self.states)


def create_showcase(destination: Path, seed: Path | None = None) -> dict:
    seed = seed or Path(__file__).parent / 'demo'
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((seed / 'checkpoint/config.json').read_text())
    deployment = json.loads((seed / 'healthy.json').read_text())
    sample = json.loads((seed / 'sample.json').read_text())

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

    recordings, traces = {}, {}
    for episode, (name, shift) in enumerate([('reference', 0.), ('candidate', 2.), ('corrected', 0.)]):
        run = destination / name
        pre, post = FixturePipeline(seed, 'pre', shift), FixturePipeline(seed, 'post', shift)
        post.steps[-1].relative_step = pre.steps[0]
        captured = capture_runtime(run / 'capture', policy_config=config, preprocessor=pre,
                                   postprocessor=post, deployment=deployment)
        assert captured['status'] == 'pass'
        write(run / 'saved_output.json', sample)
        trace = trace_actions(str(run / 'capture/checkpoint'), str(run / 'saved_output.json'),
                              str(run / 'capture/deployment.json'))
        traces[name] = trace
        write(run / 'trace.json', trace)
        rows = len(trace['commands'])
        path = run / 'dataset/data/chunk-000/file-000.parquet'
        path.parent.mkdir(parents=True)
        # Preserve native keys; model_output is an explicit extra numeric fixture stream.
        pq.write_table(pa.table({'episode_index': [episode] * rows, 'frame_index': list(range(rows)),
            'index': list(range(episode * rows, (episode + 1) * rows)), 'task_index': [0] * rows,
            'timestamp': [i / 30 for i in range(rows)], 'action': trace['commands'],
            'observation.state': [sample['observation.state']] * rows,
            'model_output': sample['model_output']}), path)
        info = json.loads((seed / 'dataset/meta/info.json').read_text())
        info.update(total_episodes=1, total_frames=rows,
                    data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet')
        for value in info['features'].values():
            value['dtype'] = 'float64'
        info['features'].update({key: {'dtype': 'int64', 'shape': [1], 'names': None}
                                 for key in ('episode_index', 'frame_index', 'index', 'task_index')})
        info['features']['timestamp'] = {'dtype': 'float64', 'shape': [1], 'names': None}
        info['features']['model_output'] = {'dtype': 'float64', 'shape': [2], 'names': ['shoulder', 'gripper']}
        write(run / 'dataset/meta/info.json', info)
        recordings[name] = path

    changed = compare_rollouts(str(recordings['reference']), str(recordings['candidate']),
                               alignment_key='frame_index', reference_episode=0, candidate_episode=1)
    corrected = compare_rollouts(str(recordings['reference']), str(recordings['corrected']),
                                 alignment_key='frame_index', reference_episode=0, candidate_episode=2)
    parameters = compare_runtime_captures(str(destination / 'reference/capture'), str(destination / 'candidate/capture'))
    investigation = investigate_run(str(destination / 'candidate'))

    # Negative control: observed state drift with matching commands and no controller-memory evidence.
    control = destination / 'unknown_cause'
    control.mkdir()
    for name, offset in [('reference', 0.), ('candidate', .25)]:
        state = np.zeros((8, 2))
        state[4:, 0] = offset
        np.savez(control / f'{name}.npz', step=np.arange(8), commands=np.zeros((8, 2)), state=state)
    unknown = compare_rollouts(str(control / 'reference.npz'), str(control / 'candidate.npz'), alignment_key='step')
    for name, report in [('recordings', changed), ('corrected', corrected), ('parameters', parameters),
                         ('investigation', investigation), ('unknown_cause', unknown)]:
        write(destination / 'reports' / f'{name}.json', report)
        (destination / 'reports' / f'{name}.md').write_text(markdown_report(report))
    summary = {'scope': 'Generated diagnostic fixture; no trained model, policy inference, simulator or robot execution.',
        'matching_saved_outputs': next(s['equal'] for s in changed['streams'] if s['key'] == 'model_output'),
        'first_command_difference': changed['first_observed_difference'],
        'native_bookkeeping_disclosed_separately': changed['bookkeeping_difference'],
        'effective_parameter_changes': [c['pointer'] for c in parameters['changes']],
        'candidate_saved_replay': traces['candidate']['status'], 'corrected_saved_replay': traces['corrected']['status'],
        'corrected_recordings': corrected['status'],
        'unknown_control_first_difference': {'stream': unknown['first_observed_difference']['stream'],
                                             'row': unknown['first_observed_difference']['row']},
        'unknown_control_causal_conclusion': unknown['causal_conclusion'],
        'investigation_status': investigation['status']}
    assert summary['matching_saved_outputs']
    assert summary['first_command_difference']['stream'] == 'action'
    assert len(summary['effective_parameter_changes']) == 2
    assert summary['candidate_saved_replay'] == 'error'
    assert summary['corrected_saved_replay'] == summary['corrected_recordings'] == 'pass'
    assert summary['unknown_control_causal_conclusion'] == 'unknown'
    write(destination / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New directory; never overwritten')
    args = parser.parse_args()
    print(json.dumps(create_showcase(args.destination), indent=2))
