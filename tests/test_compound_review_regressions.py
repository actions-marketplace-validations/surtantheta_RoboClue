"""Independent critic regressions: uncertainty must not erase proven damage."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_compound_metadata_recovery import META, make_variant
from test_salvage import hashes

from policy_doctor import product_cli
from policy_doctor.artifacts import InputError
from policy_doctor.recovery_copy import write_recovery_copy
from policy_doctor.salvage import check_salvage
from policy_doctor.sim_salvage_frontdoor import concise_salvage


def damage(root, kind):
    if kind == 'tasks':
        path = root / 'meta/tasks.parquet'
        table = pq.read_table(path)
        pq.write_table(table.slice(0, 0), path)
    elif kind == 'stats':
        (root / 'meta/stats.json').unlink()
    elif kind in ('empty', 'delete', 'truncate', 'index_local', 'index_offset'):
        path = root / f'data/chunk-000/file-{1 if kind == "empty" else 2:03d}.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        if kind == 'empty':
            rows = []
        elif kind == 'delete':
            rows = [r for r in rows if r['episode_index'] != 40]
        elif kind == 'truncate':
            rows = rows[:-10]
        else:
            for i, row in enumerate(rows):
                row['index'] = i if kind == 'index_local' else row['index'] + 1
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


@pytest.mark.parametrize('kind', ['frame', 'empty', 'delete', 'truncate', 'tasks'])
def test_independent_damage_remains_error_through_all_ci_gates(tmp_path, capsys, kind):
    root = make_variant(tmp_path, 'frame' if kind in ('frame', 'tasks') else 'clean', videos=False)
    if kind != 'frame':
        damage(root, kind)
    before = hashes(root)
    report = check_salvage(root)
    codes = {f['code']: f['status'] for f in report['findings']}
    assert report['assessment_kind'] == 'metadata_rows_unresolved'
    assert report['status'] == 'error'
    assert codes['salvage.resume_hazard'] == 'error'
    assert report['plan']['validity']['current_directory_resume_status'] == 'unsafe'
    if kind == 'tasks':
        assert codes['salvage.tasks_unusable'] == 'error'
    if kind in ('empty', 'delete'):
        assert codes['salvage.episode_integrity'] == 'error'
    if kind == 'frame':
        assert codes['salvage.row_identity'] == 'error'
    for fail_on in ('error', 'warning'):
        for args in (['check', str(root)], ['check', str(root), '--question', 'recovery'],
                     ['advanced', 'check-salvage', str(root)],
                     ['advanced', 'check-salvage', str(root), '--no-video-headers']):
            assert product_cli.main([*args, '--fail-on', fail_on, '--json']) == 1
            raw = json.loads(capsys.readouterr().out)
            assert raw['status'] == 'error' and raw['plan']['strategy'] == 'unknown'
            assert raw['episodes']['drop'] == []
    assert hashes(root) == before


def test_missing_statistics_warning_is_not_lost_to_uncertainty(tmp_path):
    root = make_variant(tmp_path, 'frame', videos=False)
    damage(root, 'stats')
    codes = {f['code']: f['status'] for f in check_salvage(root)['findings']}
    assert codes['salvage.stats_missing'] == 'warning'
    assert codes['salvage.resume_hazard'] == 'error'


@pytest.mark.parametrize('kind', ['index_offset', 'index_local', 'delete'])
def test_unresolved_data_keeps_only_physically_verified_prefix_evidence(tmp_path, kind):
    root = make_variant(tmp_path, corrected=True, videos=False)
    damage(root, kind)
    report = check_salvage(root)
    assert report['assessment_kind'] == 'metadata_rows_unresolved'
    expected = 40 if kind == 'delete' else 36
    assert report['episodes']['keep'] == [[0, expected - 1]]
    assert report['episodes']['unresolved'] == [[expected, 49]]
    assert report['episodes']['drop'] == []
    assert report['status'] == 'error'
    # Independently check the native row-position criterion for every kept row.
    data = pa.concat_tables([pq.read_table(p) for p in sorted((root / 'data').rglob('*.parquet'))]).to_pydict()
    assert data['index'][:expected * 400] == list(range(expected * 400))
    for ep in range(expected):
        a, b = ep * 400, (ep + 1) * 400
        assert data['episode_index'][a:b] == [ep] * 400
        assert data['frame_index'][a:b] == list(range(400))


def test_file_local_spans_and_indices_never_authorize_exclusion_or_copy(tmp_path):
    root = make_variant(tmp_path, corrected=True, videos=False)
    path = root / META
    table = pq.read_table(path)
    meta = table.to_pylist()
    for shard in sorted((root / 'data').rglob('*.parquet')):
        data = pq.read_table(shard)
        rows = data.to_pylist()
        for position, row in enumerate(rows):
            row['index'] = position
        pq.write_table(pa.Table.from_pylist(rows, schema=data.schema), shard)
        for item in meta:
            if item['data/file_index'] == int(shard.stem.split('-')[1]):
                start = (item['episode_index'] % 18) * 400
                item.update(dataset_from_index=start, dataset_to_index=start + 400)
    pq.write_table(pa.Table.from_pylist(meta, schema=table.schema), path)
    before = hashes(root)
    report = check_salvage(root)
    assert report['status'] == 'error'
    assert report['episodes']['keep'] == [[0, 17]]
    assert report['episodes']['drop'] == []
    assert report['episodes']['unresolved'] == [[18, 49]]
    assert report['plan']['strategy'] == 'unknown'
    conflict = report['metadata_row_conflicts']['observations'][0]
    assert conflict['observed_index_span'] == conflict['declared_span']
    assert conflict['observed_position_span'] != conflict['declared_span']
    with pytest.raises(InputError, match='complete supported plan'):
        write_recovery_copy(root, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists() and hashes(root) == before


@pytest.mark.parametrize('path_length', [6, 20, 30, 35, 75, 250])
def test_mcp_budget_covers_pretty_text_and_structured_content_at_any_path_length(tmp_path, path_length):
    report = check_salvage(make_variant(tmp_path, 'totals'))
    report['findings'] *= 5
    for finding in report['findings']:
        for evidence in finding.get('evidence', []):
            evidence['path'] = '/' + 'x' * path_length
    view = concise_salvage(report)
    assert view['findings_shown'][0]['code'] == 'salvage.metadata_row_conflict'
    assert len(json.dumps(view, indent=2).encode()) <= 8192 - 256
    assert len(json.dumps(view).encode()) <= 8192 - 256
    assert 'salvage.resume_hazard' in view['independent_error_codes']


def test_frame_error_is_named_without_blame_on_metadata(tmp_path, capsys):
    report = check_salvage(make_variant(tmp_path, 'frame', corrected=True, videos=False))
    assert report['episodes']['drop'][0]['episode_index'] == 30
    # The pre-existing coherent-span recovery remains a bounded operation.
    assert report['episodes']['keep'] == [[0, 29]]
    assert not any(f['code'] == 'salvage.metadata_row_conflict' for f in report['findings'])


def test_huge_declared_episode_count_is_bounded_before_ledger_enumeration(tmp_path):
    root = make_variant(tmp_path, videos=False)
    path = root / 'meta/info.json'
    info = json.loads(path.read_text())
    info['total_episodes'] = 10 ** 12
    path.write_text(json.dumps(info))
    report = check_salvage(root)
    assert report['status'] == 'unknown' and report['plan']['strategy'] == 'unknown'
    assert not report['episodes'].get('drop')
    assert any(f['code'] == 'salvage.coverage_unknown' and 'bounded ledger limit' in f['message']
               for f in report['findings'])
