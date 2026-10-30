"""Episode count/span contradictions found on real public LeRobot v3 datasets (census protocol B).

The strict window engine ends unknown when declared totals cannot anchor completeness; these
contradictions must still be reported, with evidence, rather than hidden behind that unknown.
Fixtures are generated; they reproduce the declaration patterns, not the public bytes.
"""
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from policy_doctor import windows
from policy_doctor.windows import audit_episode_windows


def make(tmp_path, *, lengths=(30, 30), rows_per_episode=None, totals=None, extra_meta=0, meta_columns=0):
    root = tmp_path / 'dataset'
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    actual = list(rows_per_episode or lengths)
    rows, index = [], 0
    for ep, count in enumerate(actual):
        for frame in range(count):
            rows.append({'index': index, 'episode_index': ep, 'frame_index': frame, 'action': [float(index), 0.0]})
            index += 1
    meta, start = [], 0
    for ep, length in enumerate(list(lengths) + [10] * extra_meta):
        item = {'episode_index': ep, 'length': length, 'dataset_from_index': start, 'dataset_to_index': start + length,
                'data/chunk_index': 0, 'data/file_index': 0}
        item.update({f'stats/feature_{c}/mean': [0.0] for c in range(meta_columns)})
        meta.append(item)
        start += length
    info = {'codebase_version': 'v3.0', 'total_frames': len(rows), 'total_episodes': len(meta),
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {'action': {'dtype': 'float32', 'shape': [2]}}}
    info.update(totals or {})
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / 'file-000.parquet')
    return root


def codes(report):
    return {(f['code'], f['status']) for f in report['findings']}


def test_zero_declared_totals_are_an_error_not_an_unknown(tmp_path):
    # SO-101-robot-learning/so101_pick_place: totals 0 over 50 real episodes.
    report = audit_episode_windows(str(make(tmp_path, totals={'total_frames': 0, 'total_episodes': 0})))
    assert report['status'] == 'error'
    totals = [f for f in report['findings'] if f['code'] == 'episode.declared_totals']
    assert {(f['field'], f['declared'], f['observed']) for f in totals} == {('total_frames', 0, 60), ('total_episodes', 0, 2)}
    assert {('dataset_metadata.py:222' in f['message'], f['field']) for f in totals} == {
        (True, 'total_episodes'), (False, 'total_frames')}
    assert totals[0]['evidence'][0]['sha256'] == hashlib.sha256((tmp_path / 'dataset/meta/info.json').read_bytes()).hexdigest()


def test_wrong_spans_and_orphan_metadata_are_reported(tmp_path):
    # EPITECH-LILLE/eval_training: 80 metadata episodes over 60 data episodes, lengths 446 vs 1041 rows.
    root = make(tmp_path, lengths=(10, 10), rows_per_episode=(23, 24), extra_meta=1,
                totals={'total_frames': 0, 'total_episodes': 0})
    report = audit_episode_windows(str(root))
    found = codes(report)
    assert ('episode.length_mismatch', 'error') in found and ('episode.metadata_without_rows', 'error') in found
    mismatch = next(f for f in report['findings'] if f['code'] == 'episode.length_mismatch')
    assert mismatch['episodes'][0] == {'episode_index': 0, 'declared_length': 10, 'actual_rows': 23}
    assert {e['column'] for e in mismatch['evidence']} == {'length', 'episode_index'}


def test_single_frame_episode_is_a_warning_even_when_windows_pass(tmp_path):
    # lerobot/svla_so100_stacking: episode 20 holds one frame; spans are otherwise consistent.
    report = audit_episode_windows(str(make(tmp_path, lengths=(30, 1, 29))))
    assert report['status'] == 'warning'  # validator rounds 1 and 2: a warning must reach the status
    short = [f for f in report['findings'] if f['code'] == 'episode.too_short']
    assert short and short[0]['status'] == 'warning' and short[0]['episodes'] == [1]


def test_many_statistics_columns_do_not_block_the_strict_audit(tmp_path):
    # recast-robotics rollout: richer features push episode metadata past 128 columns.
    report = audit_episode_windows(str(make(tmp_path, meta_columns=200)))
    assert report['status'] == 'pass' and report['coverage']['identity_verified']


def test_healthy_dataset_report_is_unchanged(tmp_path):
    report = audit_episode_windows(str(make(tmp_path)))
    assert report['status'] == 'pass'
    assert not [f for f in report['findings'] if f['code'].startswith('episode.') and f['status'] != 'pass']


def test_repeated_episode_metadata_rows_are_an_error(tmp_path):
    # EPITECH-LILLE/eval_training: 80 metadata rows for 60 episodes; indices 0-19 appear twice.
    root = make(tmp_path)
    meta = root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
    table = pq.read_table(meta)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), meta)
    report = audit_episode_windows(str(root))
    duplicate = [f for f in report['findings'] if f['code'] == 'episode.metadata_duplicate']
    assert report['status'] == 'error' and duplicate and duplicate[0]['episodes'] == [0]
    assert 'dataset_reader.py:304' in duplicate[0]['message']


def test_nonzero_wrong_totals_cite_the_matching_consequence(tmp_path):
    # kmg0620/so101_bomb_pick_place_v1: total_episodes=103 against 150 distinct episodes.
    report = audit_episode_windows(str(make(tmp_path, totals={'total_episodes': 1})))
    episodes = next(f for f in report['findings'] if f['code'] == 'episode.declared_totals' and f['field'] == 'total_episodes')
    assert 'dataset_reader.py:264' in episodes['message'] and 'dataset_metadata.py:222' not in episodes['message']


def test_empty_repository_is_an_explicit_warning(tmp_path):
    # Embodied-AI-6/eval_act: only meta/info.json, zero totals.
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'total_frames': 0, 'total_episodes': 0,
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet', 'features': {}}))
    report = audit_episode_windows(str(root))
    assert ('episode.empty_dataset', 'warning') in codes(report) and report['status'] == 'unknown'


def test_stray_v2_files_are_loaded_counted_and_reported(tmp_path):
    # lerobot/berkeley_cable_routing: v2-style episode_*.parquet files left beside the v3 shard.
    root = make(tmp_path)
    for ep in range(2):
        rows = [{'index': 100 + i, 'episode_index': ep, 'frame_index': i, 'action': [0.0, 0.0]} for i in range(30)]
        pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / f'episode_{ep:06d}.parquet')
    report = audit_episode_windows(str(root))
    found = codes(report)
    assert ('episode.nonstandard_data_files', 'warning') in found and ('episode.length_mismatch', 'error') in found
    stray = next(f for f in report['findings'] if f['code'] == 'episode.nonstandard_data_files')
    assert 'io_utils.py:76' in stray['message'] and stray['files'][0].endswith('episode_000000.parquet')


def test_many_shards_stay_within_the_response_bound(tmp_path):
    root = make(tmp_path)
    for i in range(300):
        rows = [{'index': 1000 + i, 'episode_index': 0, 'frame_index': 30 + i, 'action': [0.0, 0.0]}]
        pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / f'episode_{i:06d}.parquet')
    report = audit_episode_windows(str(root))
    assert report['status'] == 'error' and len(json.dumps(report).encode()) < 192 * 1024


def test_repeated_global_indices_are_an_error(tmp_path):
    # EPITECH-LILLE/eval_training: 29,775 distinct global indices among 41,686 rows (found by the validator).
    root = make(tmp_path, totals={'total_frames': 0, 'total_episodes': 0})
    data = root / 'data' / 'chunk-000' / 'file-000.parquet'
    table = pq.read_table(data).to_pylist()
    table[40]['index'] = 5
    pq.write_table(pa.Table.from_pylist(table), data)
    report = audit_episode_windows(str(root))
    dup = [f for f in report['findings'] if f['code'] == 'episode.duplicate_global_index']
    assert dup and dup[0]['status'] == 'error' and 'dataset_reader.py:200-201' in dup[0]['message']


def test_metadata_pointing_at_a_missing_data_file_is_a_warning(tmp_path):
    # allenai/eval_molmoact_cup_stacking_in-distribution: 7 episodes point at data/chunk-000/file-001.parquet,
    # which does not exist; all rows live in file-000.
    root = make(tmp_path, totals={'total_frames': 0, 'total_episodes': 0})
    meta = root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
    rows = pq.read_table(meta).to_pylist()
    rows[1]['data/file_index'] = 1
    pq.write_table(pa.Table.from_pylist(rows), meta)
    report = audit_episode_windows(str(root))
    missing = [f for f in report['findings'] if f['code'] == 'episode.missing_data_file']
    assert missing and missing[0]['status'] == 'warning' and missing[0]['episodes'] == [1]
    assert 'dataset_writer.py:454-460' in missing[0]['message']


# ---- Validator round 2 (d5fdd9c) ----

def _rewrite_data(root, change):
    data = root / 'data' / 'chunk-000' / 'file-000.parquet'
    pq.write_table(pa.Table.from_pylist([change(dict(row)) for row in pq.read_table(data).to_pylist()]), data)


def test_missing_or_null_global_index_keeps_the_totals_checks(tmp_path):
    # Validator W5/W6: without an integer index column the consistency pass dropped every finding.
    for name, change in (('missing', lambda row: {k: v for k, v in row.items() if k != 'index'}),
                         ('null', lambda row: dict(row, index=None)),
                         # Validator W6: one null inside an integer column.
                         ('one_null', lambda row: dict(row, index=None) if row['index'] == 3 else row)):
        root = make(tmp_path / name, totals={'total_frames': 0, 'total_episodes': 0})
        _rewrite_data(root, change)
        report = audit_episode_windows(str(root))
        found = codes(report)
        assert ('episode.declared_totals', 'error') in found, name
        assert ('episode.global_index_unchecked', 'unknown') in found, name
        assert report['status'] == 'error'


def test_wide_shard_numbers_are_not_stray_files(tmp_path):
    # The strict engine accepts chunk-NNN.../file-NNN... with three or more digits; so must the stray check.
    root = make(tmp_path)
    (root / 'data' / 'chunk-1000').mkdir()
    rows = [{'index': 500 + i, 'episode_index': 0, 'frame_index': 60 + i, 'action': [0.0, 0.0]} for i in range(2)]
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-1000' / 'file-1000.parquet')
    report = audit_episode_windows(str(root))
    assert 'episode.nonstandard_data_files' not in {code for code, _ in codes(report)}


def test_skipped_consistency_pass_is_recorded_in_coverage(tmp_path, monkeypatch):
    # Validator: above the file budget the pass was skipped without saying so.
    root = make(tmp_path, lengths=(30, 1, 29))
    assert audit_episode_windows(str(root))['coverage']['episode_consistency'] == 'checked'
    monkeypatch.setattr(windows, 'CONSISTENCY_MAX_FILES', 0)
    report = audit_episode_windows(str(root))
    assert report['coverage']['episode_consistency'].startswith('not checked: ')
    assert report['status'] == 'pass' and not any(f['code'] == 'episode.too_short' for f in report['findings'])


def test_human_output_shows_consistency_explanations(tmp_path, capsys):
    # The README's Hub example prints human output; consistency findings must carry their message there.
    from policy_doctor.cli import main
    root = make(tmp_path, lengths=(30, 1, 29))
    assert main(['audit-windows', str(root)]) == 0
    out = capsys.readouterr().out
    assert 'episode windows: warning' in out and 'hold at most one frame' in out and 'episode None' not in out


def test_many_healthy_shards_are_audited_with_a_bounded_artifact_list(tmp_path):
    # Validator round 3 (defect 5): 1000 standard 2-frame shards exited 2 because every shard was an artifact.
    root = tmp_path / 'dataset'
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    count = 1000
    meta = [{'episode_index': e, 'length': 2, 'dataset_from_index': 2 * e, 'dataset_to_index': 2 * e + 2,
             'data/chunk_index': 0, 'data/file_index': e} for e in range(count)]
    for e in range(count):
        rows = [{'index': 2 * e + i, 'episode_index': e, 'frame_index': i, 'action': [0.0, 0.0]} for i in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / f'file-{e:03d}.parquet')
    pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'total_frames': 2 * count,
        'total_episodes': count, 'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'features': {'action': {'dtype': 'float32', 'shape': [2]}}}))
    report = audit_episode_windows(str(root))
    assert report['status'] == 'pass' and report['artifacts_truncated'] and report['artifact_count'] > 1000
    assert len(report['artifacts_manifest_sha256']) == 64 and len(json.dumps(report).encode()) <= 192 * 1024


def test_non_ascii_digits_are_stray_files(tmp_path):
    # Validator round 3 (defect 9): Unicode \\d matched Arabic-Indic digits; the strict engine accepts [0-9] only.
    root = make(tmp_path)
    rows = [{'index': 900, 'episode_index': 0, 'frame_index': 30, 'action': [0.0, 0.0]}]
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / 'file-٠٠١.parquet')
    assert ('episode.nonstandard_data_files', 'warning') in codes(audit_episode_windows(str(root)))


def test_metadata_rows_do_not_spend_the_data_row_budget(tmp_path):
    # Validator round 3 (W13): with max_rows equal to the data rows, the pass was skipped for counting metadata too.
    report = audit_episode_windows(str(make(tmp_path, lengths=(30, 1, 29))), max_rows=60)
    assert report['coverage']['episode_consistency'] == 'checked'
    assert ('episode.too_short', 'warning') in codes(report)


def test_truncated_artifact_digest_is_location_independent_and_keeps_query_files(tmp_path):
    # Validator round 4: the digest hashed absolute paths, and query-cited shards could be dropped.
    import shutil

    def many(root, count=1000):
        (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
        (root / 'data' / 'chunk-000').mkdir(parents=True)
        meta = [{'episode_index': e, 'length': 2, 'dataset_from_index': 2 * e, 'dataset_to_index': 2 * e + 2,
                 'data/chunk_index': 0, 'data/file_index': e} for e in range(count)]
        for e in range(count):
            rows = [{'index': 2 * e + i, 'episode_index': e, 'frame_index': i, 'action': [0.0, 0.0]} for i in range(2)]
            pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / f'file-{e:03d}.parquet')
        pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
        (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'total_frames': 2 * count,
            'total_episodes': count, 'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {'action': {'dtype': 'float32', 'shape': [2]}}}))
        return root

    first = many(tmp_path / 'a')
    second = tmp_path / 'b' / 'elsewhere'
    shutil.copytree(first, second)
    one, two = audit_episode_windows(str(first)), audit_episode_windows(str(second))
    assert one['artifacts_truncated'] and one['artifacts_manifest_sha256'] == two['artifacts_manifest_sha256']
    query = audit_episode_windows(str(first), episode_index=990, anchor_index=1980, delta_indices=[0, 1])
    if query.get('artifacts_truncated'):
        assert any(a['path'].endswith('file-990.parquet') for a in query['artifacts'])


def test_coverage_counters_agree_when_a_duplicated_shard_stops_the_strict_engine(tmp_path):
    # FT-LLM-2026-RAMEN/aloha_sim_transfer_cube_scripted: file-001 repeats rows of file-000. The strict engine stops
    # at file-001's first row; it reported data_files_read=1 beside data_rows_read=27200, the rows of both files.
    root = make(tmp_path)
    first = root / 'data' / 'chunk-000' / 'file-000.parquet'
    pq.write_table(pa.Table.from_pylist(pq.read_table(first).to_pylist()[:10]), first.with_name('file-001.parquet'))
    report = audit_episode_windows(str(root))
    coverage = report['coverage']
    assert report['status'] == 'error' and not coverage['complete']
    assert (coverage['data_files_read'], coverage['data_rows_read']) == (1, 60)
    assert coverage['episode_consistency_read'] == {'data_files_read': 2, 'data_rows_read': 70,
                                                    'metadata_files_read': 1, 'metadata_rows_read': 2}
    assert ('episode.duplicate_global_index', 'error') in codes(report)

