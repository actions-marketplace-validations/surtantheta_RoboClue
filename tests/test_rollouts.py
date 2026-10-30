"""Recorded behavior comparison: observed divergence is not causal diagnosis."""
import csv
import io
import zipfile

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.rollouts import compare_rollouts, inspect_rollout


def pair(tmp_path, change=True):
    reference = {'step': np.arange(20), 'model_output': np.zeros((20, 2)),
                 'commands': np.zeros((20, 2)), 'state': np.zeros((20, 2))}
    candidate = {k: v.copy() for k, v in reference.items()}
    if change:
        candidate['commands'][5:, 1] = 0.5
        candidate['state'][8:, 1] = 0.1
    a, b = tmp_path / 'reference.npz', tmp_path / 'candidate.npz'
    np.savez_compressed(a, **reference)
    np.savez_compressed(b, **candidate)
    return a, b


def test_same_prediction_different_command_then_state(tmp_path):
    a, b = pair(tmp_path)
    report = compare_rollouts(str(a), str(b), alignment_key='step')
    assert report['status'] == 'different' and report['comparison_complete']
    assert report['first_observed_difference']['row'] == 5
    streams = {s['key']: s for s in report['streams']}
    assert streams['model_output']['equal']
    assert streams['commands']['first_difference']['row'] == 5
    assert streams['state']['first_difference']['row'] == 8
    assert streams['commands']['first_difference']['reference_value'] == 0.
    assert streams['commands']['first_difference']['candidate_value'] == .5
    assert report['causal_conclusion'] == 'unknown'
    assert report['alignment']['sample_phase'] == 'unknown'
    assert all(a['sha256'] for a in report['artifacts'])


def test_healthy_control_and_header_inventory(tmp_path):
    a, b = pair(tmp_path, False)
    report = compare_rollouts(str(a), str(b), alignment_key='step')
    assert report['status'] == 'pass' and report['first_observed_difference'] is None
    inventory = inspect_rollout(str(a))
    assert inventory['streams']['commands']['shape'] == [20, 2]
    assert not inventory['values_validated']


def test_row_alignment_is_explicitly_unknown_without_matching_index(tmp_path):
    a, b = pair(tmp_path, False)
    result = compare_rollouts(str(a), str(b))
    assert result['status'] == 'unknown'
    assert result['alignment']['status'] == 'unknown'


def test_truncation_never_establishes_complete_equivalence(tmp_path):
    a, b = pair(tmp_path)
    result = compare_rollouts(str(a), str(b), alignment_key='step', max_rows=3)
    assert result['status'] == 'unknown' and not result['comparison_complete']
    assert result['first_observed_difference'] is None


def test_gapped_or_different_steps_not_silently_aligned(tmp_path):
    a, b = pair(tmp_path, False)
    np.savez_compressed(b, step=np.arange(20) + 1, commands=np.zeros((20, 2)))
    result = compare_rollouts(str(a), str(b), alignment_key='step')
    assert result['status'] == 'unknown'
    assert not result['comparison_complete']


def test_empty_overlap_and_trailing_dimension_mismatch(tmp_path):
    a, b = pair(tmp_path, False)
    np.savez_compressed(b, other=np.zeros((20, 3)))
    result = compare_rollouts(str(a), str(b))
    assert result['status'] == 'unknown' and not result['streams']
    np.savez_compressed(b, commands=np.zeros((20, 3)), step=np.arange(20))
    result = compare_rollouts(str(a), str(b), alignment_key='step')
    assert result['status'] == 'unknown'
    assert any(s['status'] == 'incompatible' for s in result['streams'])


def test_exact_integer_comparison_above_float_precision(tmp_path):
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    np.savez(a, step=np.arange(2), counter=np.array([2**60, 2**60], dtype=np.uint64))
    np.savez(b, step=np.arange(2), counter=np.array([2**60, 2**60+1], dtype=np.uint64))
    result = compare_rollouts(str(a), str(b), alignment_key='step')
    assert result['status'] == 'different'
    assert result['streams'][0]['first_difference']['candidate_value'] == 2**60+1


@pytest.mark.parametrize('bad', ['pickle', 'nan', 'duplicate', 'oversized_header'])
def test_unsafe_or_invalid_arrays_are_refused(tmp_path, bad):
    path = tmp_path / 'bad.npz'
    if bad in ('pickle', 'nan'):
        values = np.array([{'execute': 'never'}], dtype=object) if bad == 'pickle' else np.array([[float('nan')]])
        np.savez(path, commands=values)
    elif bad == 'duplicate':
        payload = io.BytesIO()
        np.save(payload, np.ones((2, 2)))
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('commands.npy', payload.getvalue())
            with pytest.warns(UserWarning, match='Duplicate name'):
                z.writestr('commands.npy', payload.getvalue())
    else:
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {'descr': '<f8', 'fortran_order': False, 'shape': (10**12, 2)})
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('commands.npy', payload.getvalue())
    with pytest.raises(InputError):
        compare_rollouts(str(path), str(path))


def test_csv_numeric_streams_with_explicit_step(tmp_path):
    paths = [tmp_path / 'a.csv', tmp_path / 'b.csv']
    for path in paths:
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step', 'joint'])
            writer.writerows([[0, 1.], [1, 2.]])
    assert compare_rollouts(*map(str, paths), alignment_key='step')['status'] == 'pass'
