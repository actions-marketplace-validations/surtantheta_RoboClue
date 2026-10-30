"""Numeric cohort metadata must not approve stale copied statistics."""
import json

import pytest
from test_salvage import meta_for, rows_for, write

from policy_doctor.artifacts import load_dataset
from policy_doctor.diagnostics import dataset_statistics_findings
from policy_doctor.resume_guard import assess_resume
from policy_doctor.salvage import check_salvage
from policy_doctor.statistics_counts import statistics_count_findings


def info():
    return {'total_frames': 165, 'features': {
        'action': {'dtype': 'float32', 'shape': [2]},
        'index': {'dtype': 'int64', 'shape': [1]},
        'observation.images.front': {'dtype': 'video', 'shape': [3, 32, 32]},
    }}


def test_numeric_count_mismatch_is_a_supported_error_with_both_sources():
    findings = statistics_count_findings(info(), {'action': {'count': [259]}},
        info_artifact={'path': 'meta/info.json', 'sha256': 'info'},
        stats_artifact={'path': 'meta/stats.json', 'sha256': 'stats'})
    assert len(findings) == 1
    result = findings[0]
    assert result['code'] == 'dataset.statistics_count_mismatch' and result['status'] == 'error'
    assert result['saved_count'] == [259] and result['declared_total_frames'] == 165
    assert {e['sha256'] for e in result['evidence']} == {'info', 'stats'}
    assert 'Do not change the count alone' in result['remedy']


@pytest.mark.parametrize('count', [[165], [165.0]])
def test_matching_numeric_count_is_not_a_statistic_values_certificate(count):
    assert statistics_count_findings(info(), {'action': {'count': count}}) == []


@pytest.mark.parametrize('count', [[12], [32], [165], [259]])
def test_native_sampled_visual_counts_are_never_compared_as_numeric_rows(count):
    assert statistics_count_findings(info(), {'observation.images.front': {'count': count}}) == []


@pytest.mark.parametrize('count', [None, 259, [], [259, 259], [True], ['259']])
def test_unsupported_count_encodings_are_not_guessed(count):
    assert statistics_count_findings(info(), {'action': {'count': count}}) == []


@pytest.mark.parametrize('count', [[-1], [float('inf')], [1.5], [10**100]])
def test_invalid_or_extreme_numeric_cohort_cannot_pass(count):
    assert statistics_count_findings(info(), {'action': {'count': count}})[0]['status'] == 'error'


def stale_copy(tmp_path):
    episodes = [(0, 3), (1, 4)]
    root = write(tmp_path / 'dataset', [rows_for(episodes)], [meta_for(episodes)], 2, 7)
    (root / 'meta/stats.json').write_text(json.dumps({
        'action': {'mean': [.5, 1.], 'std': [1., 1.], 'min': [0., 0.], 'max': [2., 2.], 'count': [12]},
        'index': {'mean': [3.], 'std': [2.], 'min': [0.], 'max': [6.], 'count': [7]},
    }))
    return root


def test_generic_statistics_recovery_and_resume_all_block_stale_numeric_count(tmp_path):
    root = stale_copy(tmp_path)
    findings = dataset_statistics_findings(load_dataset(root))
    assert any(f['code'] == 'dataset.statistics_count_mismatch' for f in findings)
    report = check_salvage(root)
    assert report['status'] == 'error'
    assert any(f['code'] == 'dataset.statistics_count_mismatch' for f in report['findings'])
    command = ['lerobot-record', '--dataset.root=' + str(root), '--resume=true']
    gate = assess_resume(root, command)
    assert gate['status'] == 'blocked' and gate['exit_code'] != 0 and gate['launched'] is False


def test_clean_row_ledger_with_total_typo_is_metadata_mismatch(tmp_path):
    root = write(tmp_path/'dataset', [rows_for([(0, 3), (1, 4)])], [meta_for([(0, 3), (1, 4)])], 2, 7)
    path = root/'meta/info.json'
    value = json.loads(path.read_text())
    value['total_frames'] = 8
    path.write_text(json.dumps(value))
    report = check_salvage(root)
    assert report['assessment_kind'] == 'metadata_totals_mismatch'
    assert report['counts']['files_without_footer'] == 0
    assert report['episodes']['recovery_candidates'] == []
