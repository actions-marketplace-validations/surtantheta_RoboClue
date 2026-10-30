"""A complete native cohort must not turn an unresolved saved metric into green.

The small JSON fixture retains actual get_feature_stats float32 output and the
103 source vectors. No native framework is imported by these integration tests.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from policy_doctor.dataset import scan_dataset
from policy_doctor.health_cards import health_card, render_badge, verify_health_card
from policy_doctor.preflight import preflight

FIXTURE = Path(__file__).parent / 'data/native_float32_precision.json'


def dataset(root, *, precise=False, missing=False, wrong_mean=False):
    value = json.loads(FIXTURE.read_text())
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps(value['info']))
    stats = copy.deepcopy(value['stats'])
    values = np.asarray([row['action'] for row in value['rows']], dtype=np.float64)
    if precise:
        stats['action']['std'] = values.std(0).tolist()
    if wrong_mean:
        stats['action']['mean'] = (values.mean(0) + 2).tolist()
    if not missing:
        (root / 'meta/stats.json').write_text(json.dumps(stats))
    table = pa.Table.from_pylist(value['rows'])
    table = table.set_column(table.schema.get_field_index('action'), 'action',
                             pa.array([row['action'] for row in value['rows']], type=pa.list_(pa.float32(), 2)))
    pq.write_table(table, root / 'data/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(value['episodes']), root / 'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(value['tasks']), root / 'meta/tasks.parquet')
    return root


def reports(root):
    return (scan_dataset(root), preflight(str(root), groot_contract=False, units=False),
            health_card(str(root), config={'groot_contract': False, 'units': False}))


def test_actual_native_precision_unknown_reaches_complete_preflight_and_health_card(tmp_path):
    root = dataset(tmp_path / 'native')
    before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    scan, pre, card = reports(root)
    assert scan['coverage']['complete_parquet_scan'] and scan['coverage']['rows_scanned'] == 103
    assert scan['statistics_comparison']['unassessed_metrics'] == {'action': ['std']}
    assert scan['statistics_comparison']['precision_unassessed_count'] == 1
    assert scan['status'] == pre['status'] == card['status'] == 'unknown'
    assert pre['components']['dataset_statistics']['status'] == 'pass'
    assert pre['components']['episode_windows']['status'] == pre['components']['video_index']['status'] == 'pass'
    assert pre['components']['numeric_rows']['status'] == 'unknown'
    finding, = scan['findings']
    assert finding['code'] == 'dataset.statistics_values_unassessed' and finding['status'] == 'unknown'
    assert 'float32' in finding['message'] and 'higher precision' in finding['message']
    assert finding['compared_metrics'] == ['max', 'mean', 'min', 'std']
    assert scan['total_findings'] == 1 and not scan['findings_truncated']
    assert verify_health_card(card) and 'fill="#276b37"' not in render_badge(card)
    assert before == {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*') if p.is_file()}


def test_precisely_recomputed_metric_restores_pass_for_same_complete_rows(tmp_path):
    scan, pre, card = reports(dataset(tmp_path / 'corrected', precise=True))
    assert scan['status'] == pre['status'] == card['status'] == 'pass'
    assert scan['statistics_comparison']['unassessed_metrics'] == {}
    assert not scan['findings'] and 'fill="#276b37"' in render_badge(card)


def test_definite_error_wins_over_native_precision_uncertainty(tmp_path):
    scan, pre, card = reports(dataset(tmp_path / 'wrong', wrong_mean=True))
    assert scan['statistics_comparison']['unassessed_metrics'] == {'action': ['std']}
    assert scan['status'] == pre['status'] == card['status'] == 'error'
    assert any(f['code'] == 'dataset.statistics_rows_mismatch' for f in scan['findings'])
    assert scan['total_findings'] > 0


def test_no_saved_stats_retains_row_only_pass_without_green_health_card(tmp_path):
    scan, pre, card = reports(dataset(tmp_path / 'rows', missing=True))
    assert scan['status'] == 'pass' and not scan['findings']
    assert scan['statistics_comparison']['status'] == 'unknown'
    assert not scan['statistics_comparison']['compared_features']
    assert pre['status'] == card['status'] == 'unknown'


def test_explicit_disabled_comparison_keeps_row_only_legacy_scope(tmp_path):
    scan = scan_dataset(dataset(tmp_path / 'disabled'), compare_stats=False)
    assert scan['status'] == 'pass' and not scan['findings']
    assert scan['statistics_comparison']['status'] == 'unknown'
    assert scan['statistics_comparison']['reason'] == 'Explicitly disabled.'
    assert scan['coverage']['observed_row_statistics_compared'] is False
