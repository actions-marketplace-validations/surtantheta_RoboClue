"""A complete row cohort can contradict otherwise self-consistent saved values."""
import json

import numpy as np
import pytest
from test_salvage import meta_for, rows_for, write

from policy_doctor.dataset import scan_dataset
from policy_doctor.numeric_statistics import NumericStatistics


def dataset(tmp_path, *, shift=False):
    episodes = [(0, 3), (1, 4)]
    rows = rows_for(episodes)
    root = write(tmp_path / 'dataset', [rows], [meta_for(episodes)], 2, 7)
    values = np.array([r['action'] for r in rows])
    stats = {'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(),
             'min': values.min(0).tolist(), 'max': values.max(0).tolist(), 'count': [len(values)]}
    if shift:
        stats['mean'] = [x + .2 for x in stats['mean']]
    (root / 'meta/stats.json').write_text(json.dumps({'action': stats}))
    return root


def test_complete_native_rows_confirm_mean_std_min_max(tmp_path):
    report = scan_dataset(dataset(tmp_path))
    assert report['status'] == 'pass'
    comparison = report['statistics_comparison']
    assert comparison['status'] == 'pass' and comparison['rows'] == 7
    assert comparison['compared_features'] == ['action']
    assert comparison['compared_metrics'] == ['max', 'mean', 'min', 'std']
    assert report['coverage']['observed_row_statistics_compared'] is True


def test_stale_inbounds_mean_cannot_pass_after_row_comparison(tmp_path):
    report = scan_dataset(dataset(tmp_path, shift=True))
    assert report['status'] == 'error'
    comparison = report['statistics_comparison']
    assert comparison['status'] == 'error'
    mismatch, = comparison['mismatches']
    assert mismatch['feature'] == 'action' and mismatch['metric'] == 'mean'
    assert 'complete 7-row cohort' in mismatch['message']
    assert any(f['code'] == 'dataset.statistics_rows_mismatch' for f in report['findings'])


def test_correct_count_but_stale_mean_blocks_recovery_and_resume(tmp_path):
    from policy_doctor.resume_guard import assess_resume
    from policy_doctor.salvage import check_salvage

    root = dataset(tmp_path, shift=True)
    report = check_salvage(root)
    assert report['status'] == 'error'
    assert report['statistics_comparison']['status'] == 'error'
    assert any(f['code'] == 'dataset.statistics_rows_mismatch' for f in report['findings'])
    gate = assess_resume(root, ['lerobot-record', '--dataset.root=' + str(root), '--resume=true'])
    assert gate['status'] == 'blocked' and gate['launched'] is False


@pytest.mark.parametrize('max_rows', [1, 3, 6])
def test_partial_scan_never_compares_a_subset_to_global_stats(tmp_path, max_rows):
    report = scan_dataset(dataset(tmp_path, shift=True), max_rows=max_rows)
    assert report['status'] == 'unknown'
    assert report['statistics_comparison']['status'] == 'unknown'
    assert report['statistics_comparison']['compared_features'] == []
    assert not any(f['code'] == 'dataset.statistics_rows_mismatch' for f in report['findings'])


def test_explicit_optout_retains_unassessed_comparison(tmp_path):
    report = scan_dataset(dataset(tmp_path, shift=True), compare_stats=False)
    assert report['status'] == 'pass'
    assert report['statistics_comparison']['status'] == 'unknown'
    assert report['statistics_comparison']['reason'] == 'Explicitly disabled.'
    assert report['coverage']['observed_row_statistics_compared'] is False


def test_welford_matches_direct_unequal_length_oracle():
    rng = np.random.default_rng(3)
    rows = rng.normal(73.5, .3, size=(103, 4))
    entry = {name: values.tolist() for name, values in {
        'mean': rows.mean(0), 'std': rows.std(0), 'min': rows.min(0), 'max': rows.max(0)}.items()}
    state = NumericStatistics({'codebase_version': 'v3.0', 'stats': {'action': entry}},
                              {'action': {'shape': [4], 'dtype': 'float32'}})
    for row in rows:
        state.update({'action': row})
    assert state.finish(complete=True, declared_rows=len(rows))['status'] == 'pass'


def test_images_and_quantile_sampling_are_not_recomputed_as_numeric_rows():
    state = NumericStatistics({'codebase_version': 'v3.0', 'stats': {
        'observation.images.front': {'mean': [[[.5]]], 'count': [2]},
        'action': {'q01': [.1], 'q99': [.9]}}}, {
        'observation.images.front': {'shape': [1, 1, 1], 'dtype': 'video', 'type': 'VISUAL'},
        'action': {'shape': [1], 'dtype': 'float32'}})
    state.update({'action': [.5]})
    report = state.finish(complete=True, declared_rows=1)
    assert report['status'] == 'unknown' and report['compared_features'] == []
    assert 'observation.images.front' not in state.columns


def test_native_float32_cancellation_is_unassessed_not_a_stale_mean_accusation():
    # Native computes E[x²] - E[x]² in float32, which can lose the spread.
    rows = np.array([[73.25], [73.5], [73.75]] * 35, dtype=np.float32)
    mean = rows.mean(0)
    native_std = np.sqrt(np.maximum(0, (rows ** 2).mean(0) - mean ** 2))
    entry = {'mean': mean.tolist(), 'std': native_std.tolist(),
             'min': rows.min(0).tolist(), 'max': rows.max(0).tolist()}
    state = NumericStatistics({'codebase_version': 'v3.0', 'stats': {'action': entry}},
                              {'action': {'shape': [1], 'dtype': 'float32'}})
    for row in rows:
        state.update({'action': row})
    result = state.finish(complete=True, declared_rows=len(rows))
    assert result['status'] == 'unknown' and not result['mismatches']
    assert result['unassessed_metrics'] == {'action': ['std']}
    assert 'higher precision' in result['precision_unassessed'][0]['reason']


def test_precision_uncertainty_cannot_hide_independent_stale_mean():
    rows = np.array([[73.25], [73.5], [73.75]] * 35, dtype=np.float32)
    mean = rows.mean(0)
    native_std = np.sqrt(np.maximum(0, (rows ** 2).mean(0) - mean ** 2))
    entry = {'mean': (mean + 2).tolist(), 'std': native_std.tolist(),
             'min': rows.min(0).tolist(), 'max': rows.max(0).tolist()}
    state = NumericStatistics({'codebase_version': 'v3.0', 'stats': {'action': entry}},
                              {'action': {'shape': [1], 'dtype': 'float32'}})
    for row in rows:
        state.update({'action': row})
    result = state.finish(complete=True, declared_rows=len(rows))
    assert result['status'] == 'error'
    assert result['mismatches'][0]['metric'] == 'mean'
    assert result['unassessed_metrics'] == {'action': ['std']}
