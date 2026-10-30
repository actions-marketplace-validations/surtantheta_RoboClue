"""Unknown shard positions never become evidence for deleting intact episodes."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_compound_metadata_recovery import META, make_variant
from test_salvage import hashes, strip_footer
from test_salvage_release_blockers import mutate, patch_num_rows

from policy_doctor import product_cli
from policy_doctor.artifacts import InputError
from policy_doctor.recovery_copy import write_recovery_copy
from policy_doctor.salvage import check_salvage
from policy_doctor.windows import audit_episode_windows


def forbid_copy(root, destination):
    with pytest.raises(InputError, match='complete supported plan'):
        write_recovery_copy(root, destination)
    assert not destination.exists()


@pytest.mark.parametrize('headers', [True, False])
def test_unreadable_middle_does_not_accuse_the_correct_readable_tail(tmp_path, headers):
    root = make_variant(tmp_path, corrected=True)
    strip_footer(root / 'data/chunk-000/file-001.parquet')
    before = hashes(root)
    report = check_salvage(root, video_headers=headers)
    conflict = report['metadata_row_conflicts']
    assert conflict['physical_span_conflicts'] == conflict['row_index_conflicts'] == 0
    assert conflict['position_unresolved'] == 14
    assert all(o['observed_position_span'] is None for o in conflict['observations'])
    assert next(f for f in report['findings'] if f['code'] == 'salvage.metadata_row_conflict')['status'] == 'unknown'
    assert not any(f['code'] == 'salvage.row_identity' for f in report['findings'])
    assert any(f['code'] == 'salvage.file_unreadable' and f['status'] == 'error' for f in report['findings'])
    assert report['status'] == 'error' and report['plan']['strategy'] == 'unknown'
    assert report['episodes']['keep'] == [[0, 17]] and report['episodes']['unresolved'] == [[18, 49]]
    assert report['episodes']['drop'] == []
    forbid_copy(root, tmp_path / 'unsafe')
    assert hashes(root) == before


@pytest.mark.parametrize('defect', ['before_hole', 'tail_frame', 'tail_index'])
def test_unreadable_middle_preserves_independent_known_or_local_errors(tmp_path, defect):
    root = make_variant(tmp_path, corrected=True, videos=False)
    strip_footer(root / 'data/chunk-000/file-001.parquet')
    if defect == 'before_hole':
        mutate(root, 'swap_rows')
    else:
        path = root / 'data/chunk-000/file-002.parquet'
        table = pq.read_table(path)
        data = table.to_pylist()
        position = next(i for i, r in enumerate(data) if r['episode_index'] == 40 and r['frame_index'] == 0)
        if defect == 'tail_frame':
            data[position]['frame_index'], data[position + 1]['frame_index'] = 1, 0
        else:
            data[position + 1]['index'] += 1000
        pq.write_table(pa.Table.from_pylist(data, schema=table.schema), path)
    report = check_salvage(root)
    assert report['status'] == 'error' and report['plan']['strategy'] == 'unknown'
    assert not report['episodes']['drop']
    if defect == 'before_hole':
        assert report['metadata_row_conflicts']['physical_span_conflicts'] == 2
        assert all(o['episode_index'] in (3, 4) for o in report['metadata_row_conflicts']['observations']
                   if o['physical_span_conflict'])
    else:
        assert report['metadata_row_conflicts']['physical_span_conflicts'] == 0
        identity = next(f for f in report['findings'] if f['code'] == 'salvage.row_identity')
        assert identity['status'] == 'error' and identity['episodes'] == [[40, 40]]
    forbid_copy(root, tmp_path / 'unsafe')


@pytest.mark.parametrize('where', [0, 26, 50, 'reorder'])
@pytest.mark.parametrize('headers', [True, False])
def test_metadata_identity_defects_cannot_exclude_an_intact_tail(tmp_path, where, headers):
    root = make_variant(tmp_path, corrected=True)
    path = root / META
    table = pq.read_table(path)
    rows = table.to_pylist()
    if where == 'reorder':
        rows[25], rows[26] = rows[26], rows[25]
    else:
        rows.insert(where, dict(rows[max(0, where - 1)]))
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    before = hashes(root)
    report = check_salvage(root, video_headers=headers)
    assert report['status'] == 'error' and report['plan']['strategy'] == 'unknown'
    assert report['episodes']['drop'] == [] and report['counts']['drop'] == 0
    assert report['metadata_row_conflicts']['metadata_identity_conflicts'] > 0
    assert report['metadata_row_conflicts']['physical_span_conflicts'] == 0
    assert not any(f['code'] in ('salvage.metadata_plan', 'salvage.episodes_dropped') for f in report['findings'])
    assert report['plan']['validity']['current_directory_resume_status'] == 'unsafe'
    # All fifty original runs still exist; metadata order does not prove data loss.
    data = pa.concat_tables([pq.read_table(p) for p in sorted((root / 'data').rglob('*.parquet'))]).to_pydict()
    assert set(data['episode_index']) == set(range(50)) and len(data['index']) == 20000
    forbid_copy(root, tmp_path / 'unsafe')
    assert hashes(root) == before


@pytest.mark.parametrize('count', [10 ** 9, 10 ** 12])
def test_consistent_extreme_footer_is_unknown_not_an_invented_integrity_error(tmp_path, count):
    root = make_variant(tmp_path, corrected=True, videos=False)
    path = root / 'data/chunk-000/file-002.parquet'
    patch_num_rows(path, file_rows=count, rg_rows=count)
    for report in (check_salvage(root), audit_episode_windows(root)):
        assert report['status'] == 'unknown'
        assert not any(f['status'] == 'error' for f in report['findings'])
    assert check_salvage(root)['plan']['strategy'] == 'unknown'


@pytest.mark.parametrize('headers', [True, False])
def test_footer_contradiction_is_error_through_recovery_and_generic_gates(tmp_path, capsys, headers):
    root = make_variant(tmp_path, corrected=True, videos=False)
    patch_num_rows(root / 'data/chunk-000/file-002.parquet', rg_rows=10 ** 12)
    flags = [] if headers else ['--no-video-headers']
    for prefix in (['check', str(root)], ['check', str(root), '--question', 'recovery'],
                   ['advanced', 'check-salvage', str(root)], ['advanced', 'preflight', str(root)]):
        assert product_cli.main([*prefix, *flags, '--json', '--fail-on', 'error']) == 1
        output = capsys.readouterr()
        assert 'Traceback' not in output.out + output.err
        assert json.loads(output.out)['status'] == 'error'


def test_exact_negative_span_repair_keeps_every_data_and_video_byte(tmp_path):
    root = make_variant(tmp_path, corrected=True)
    mutate(root, 'negative')
    before = hashes(root)
    report = check_salvage(root)
    assert report['status'] == 'error' and report['plan']['strategy'] == 'metadata_copy'
    assert report['episodes']['keep'] == [[0, 49]] and not report['episodes']['drop']
    destination = tmp_path / 'exact-copy'
    write_recovery_copy(root, destination)
    copied = hashes(destination)
    assert all(copied[n] == digest for n, digest in before.items() if n.startswith(('data/', 'videos/')))
    assert hashes(root) == before
    assert not any(f['status'] == 'error' for f in check_salvage(destination)['findings'])
