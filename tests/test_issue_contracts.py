"""Bounded mechanisms from public issues; these are not original incident replays."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from safetensors.numpy import load_file, save_file

from policy_doctor.artifacts import InputError
from policy_doctor.diagnostics import audit
from policy_doctor.rollouts import compare_rollouts
from policy_doctor.trace import trace_actions


@pytest.mark.parametrize('namespaces', [('so100.buffer',), ('so100.buffer', 'so100-blue.buffer')])
def test_issue4415_namespaced_stats_never_become_an_implicit_first_match(case, namespaces):
    # Same tensor dimensions, but no exact native 'action' statistic key.
    for name in ['pre.safetensors', 'post.safetensors']:
        path = case['checkpoint'] / name
        stats = load_file(str(path))
        remapped = {key: value for key, value in stats.items() if not key.startswith('action.')}
        for index, prefix in enumerate(namespaces):
            for key, value in stats.items():
                if key.startswith('action.'):
                    remapped[prefix + '.' + key] = value + index
        save_file(remapped, str(path))
    report = audit(str(case['checkpoint']), str(case['dataset']), str(case['deployment']))
    assert report['status'] == 'error'
    assert any(f['status'] == 'error' and f['code'] == 'normalization.statistics' for f in report['findings'])
    with pytest.raises(InputError):
        trace_actions(str(case['checkpoint']), str(case['sample']), str(case['deployment']))


def test_issue4679_requested_and_returned_named_streams_stay_separate(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    for path, requested in [(a, 2.), (b, 3.)]:
        pq.write_table(pa.table({'episode_index': [0, 0], 'frame_index': [0, 1],
            'action': [[.5], [.5]], 'action.requested': [[requested], [requested]]}), path)
    report = compare_rollouts(str(a), str(b), alignment_key='frame_index')
    assert report['status'] == 'different'
    assert {'action', 'action.requested'} <= {stream['key'] for stream in report['streams']}
    assert report['first_observed_difference']['stream'] == 'action.requested'
    # Numeric field names alone do not certify what hardware actually executed.
    assert report['causal_conclusion'] == 'unknown'
