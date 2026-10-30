"""Explicit copy controls: source retention, native summary math and refusal paths."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_salvage import hashes, meta_for, rows_for, write

from policy_doctor import recovery_copy as rc
from policy_doctor.artifacts import InputError


def recording(tmp_path, *, orphan=0, drop=False, summaries=True):
    episodes = [(0, 3), (1, 4)]
    rows = rows_for(episodes + ([(2, orphan)] if orphan else []))
    meta = meta_for(episodes)
    if summaries:
        for row in meta:
            values = rows[row['dataset_from_index']:row['dataset_to_index']]
            for name in ('action', 'index'):
                data = np.array([v[name] if name == 'action' else [v[name]] for v in values], dtype=float)
                stats = {'min': data.min(axis=0), 'max': data.max(axis=0), 'mean': data.mean(axis=0),
                         'std': data.std(axis=0), 'count': np.array([len(data)]),
                         'q01': np.quantile(data, .01, axis=0), 'q99': np.quantile(data, .99, axis=0)}
                row.update({f'stats/{name}/{k}': v.tolist() for k, v in stats.items()})
    if drop:
        data_files = [rows_for(episodes), rows_for([(2, 5)], start=7)]
    else:
        data_files = [rows]
    root = write(tmp_path / 'source', data_files, [meta], 3 if drop else 2, 12 if drop else 7)
    (root / 'meta/stats.json').write_text(json.dumps({
        key: {'mean': [9999.] * size, 'std': [1.] * size} for key, size in [('action', 2), ('index', 1)]}))
    return root


def test_broken_drop_corrected_copy_rebuilds_stats(tmp_path):
    source = recording(tmp_path, drop=True)
    original = hashes(source)
    report = rc.write_recovery_copy(source, tmp_path / 'new')
    assert report['status'] == 'pass'
    assert report['source_report']['status'] == 'error'
    assert report['candidate_report']['status'] == 'pass'
    assert report['statistics']['status'] == 'pass'
    assert report['verification']['training_qualification'] == 'not_assessed'
    assert report['verification']['native_reload'] == 'not_run'
    assert hashes(source) == original
    assert report['original_unchanged'] is True
    assert report['files_omitted'] == ['data/chunk-000/file-001.parquet']
    stats = json.loads((tmp_path / 'new/meta/stats.json').read_text())
    np.testing.assert_allclose(stats['action']['mean'], [4 / 7, 9 / 7])
    assert stats['action']['count'] == [7]
    assert stats['index']['mean'] == [3.]
    assert (tmp_path / 'new.roboclue-recovery.json').is_file()
    assert report['receipt'] == str(tmp_path / 'new.roboclue-recovery.json')
    assert not (tmp_path / 'new/roboclue-recovery.json').exists()
    assert not (tmp_path / 'new' / rc.INCOMPLETE).exists()
    for name in report['files_copied_unchanged']:
        src, dst = source / name, tmp_path / 'new' / name
        assert src.read_bytes() == dst.read_bytes()
        assert src.stat().st_ino != dst.stat().st_ino
    for name, manifest in report['output_manifest'].items():
        assert hashlib.sha256((tmp_path / 'new' / name).read_bytes()).hexdigest() == manifest['sha256']


def test_orphan_rows_trim_preserves_arrow_metadata(tmp_path):
    source = recording(tmp_path, orphan=5)
    path = source / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).replace_schema_metadata({b'owner': b'original Arrow metadata'})
    pq.write_table(table, path)
    before = hashes(source)
    result = rc.write_recovery_copy(source, tmp_path / 'new')
    assert result['status'] == 'pass'
    copied = pq.read_table(tmp_path / 'new' / path.relative_to(source))
    assert copied.num_rows == 7 and copied.schema.metadata == table.schema.metadata
    assert hashes(source) == before


def test_healthy_copy_does_not_edit_source(tmp_path):
    source = recording(tmp_path)
    before = hashes(source)
    result = rc.write_recovery_copy(source, tmp_path / 'new')
    assert result['status'] == 'pass'
    assert result['source_report']['plan']['strategy'] == 'none_needed'
    assert hashes(source) == before


def test_missing_summaries_omit_stale_stats_and_never_qualify(tmp_path):
    source = recording(tmp_path, drop=True, summaries=False)
    before = hashes(source)
    result = rc.write_recovery_copy(source, tmp_path / 'new')
    assert result['status'] == 'unknown' and result['statistics']['status'] == 'unknown'
    assert result['verification']['statistics'] == 'unknown'
    assert not (tmp_path / 'new/meta/stats.json').exists()
    assert any(f['code'] == 'salvage.stats_missing' for f in result['candidate_report']['findings'])
    assert hashes(source) == before


def test_unsupported_version_no_creation(tmp_path):
    source = recording(tmp_path)
    path = source / 'meta/info.json'
    info = json.loads(path.read_text())
    info['codebase_version'] = 'v2.1'
    path.write_text(json.dumps(info))
    with pytest.raises(InputError, match='unsupported|incomplete'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('existing', ['directory', 'file', 'dangling_link'])
def test_no_overwrite(tmp_path, existing):
    source = recording(tmp_path)
    target = tmp_path / 'new'
    if existing == 'directory':
        target.mkdir()
        (target / 'sentinel').write_bytes(b'leave me')
    elif existing == 'file':
        target.write_bytes(b'leave me')
    else:
        target.symlink_to(tmp_path / 'absent')
    with pytest.raises(InputError, match='already exists'):
        rc.write_recovery_copy(source, target)
    if existing == 'directory':
        assert (target / 'sentinel').read_bytes() == b'leave me'


@pytest.mark.parametrize('kind', ['fifo', 'symlink', 'directory_symlink'])
def test_source_special_entries_are_never_opened(tmp_path, kind):
    source = recording(tmp_path)
    path = source / 'unused'
    if kind == 'fifo':
        os.mkfifo(path)
    elif kind == 'symlink':
        path.symlink_to(source / 'meta/info.json')
    else:
        path.symlink_to(source / 'meta', target_is_directory=True)
    with pytest.raises(InputError, match='special|symlink'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_output_parent_symlink_refused(tmp_path):
    source = recording(tmp_path)
    (tmp_path / 'alias').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(InputError):
        rc.write_recovery_copy(source, tmp_path / 'alias/new')
    assert not (tmp_path / 'new').exists()


def test_source_symlink_refused(tmp_path):
    source = recording(tmp_path)
    (tmp_path / 'alias').symlink_to(source, target_is_directory=True)
    with pytest.raises(InputError):
        rc.write_recovery_copy(tmp_path / 'alias', tmp_path / 'new')


def test_nested_destination_refused(tmp_path):
    source = recording(tmp_path)
    with pytest.raises(InputError, match='separate'):
        rc.write_recovery_copy(source, source / 'new')
    assert not (source / 'new').exists()


@pytest.mark.parametrize('budget', [{'max_copy_bytes': 1}, {'max_files': 1}, {'max_rows': 1}])
def test_budgets_do_not_create_copy(tmp_path, budget):
    source = recording(tmp_path)
    with pytest.raises(InputError):
        rc.write_recovery_copy(source, tmp_path / 'new', **budget)
    assert not (tmp_path / 'new').exists()


def test_incomplete_plan_refused_not_trusted(monkeypatch, tmp_path):
    source = recording(tmp_path)
    original = rc.check_salvage
    def incomplete(*args, **kwargs):
        report = original(*args, **kwargs)
        report['plan']['files']['lists_complete'] = False
        return report
    monkeypatch.setattr(rc, 'check_salvage', incomplete)
    with pytest.raises(InputError, match='incomplete'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_race_after_plan_rejected_before_write(monkeypatch, tmp_path):
    source = recording(tmp_path)
    original = rc._prepare
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        (source / 'changed').write_bytes(b'new source bytes')
        return result
    monkeypatch.setattr(rc, '_prepare', changing)
    with pytest.raises(InputError, match='Source changed'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_race_during_verification_keeps_incomplete_marker(monkeypatch, tmp_path):
    source = recording(tmp_path)
    original = rc.check_salvage
    def changing(path, **kwargs):
        result = original(path, **kwargs)
        if Path(path) != source:
            (source / 'changed').write_bytes(b'new source bytes')
        return result
    monkeypatch.setattr(rc, 'check_salvage', changing)
    with pytest.raises(InputError, match='incomplete and unverified'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    marker = json.loads((tmp_path / 'new' / rc.INCOMPLETE).read_text())
    assert marker['verified'] is False
    assert not (tmp_path / 'new' / rc.RECEIPT).exists()
    with pytest.raises(InputError, match='incomplete RoboClue copy'):
        rc.write_recovery_copy(tmp_path / 'new', tmp_path / 'again')


def test_source_directory_lock_blocks_copy(tmp_path):
    import fcntl
    source = recording(tmp_path)
    fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(InputError, match='lock'):
            rc.write_recovery_copy(source, tmp_path / 'new')
        assert not (tmp_path / 'new').exists()
    finally:
        os.close(fd)


@pytest.mark.parametrize('change', ['missing_feature', 'missing_stat', 'count', 'nan', 'shape', 'extra_feature',
                                  'zero_variance', 'quantile_order', 'quantile_bounds'])
def test_unsupported_summaries_are_unknown_not_stale_pass(tmp_path, change):
    source = recording(tmp_path, drop=True)
    path = source / 'meta/episodes/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    rows = table.to_pylist()
    if change == 'missing_feature':
        rows = [{k: v for k, v in row.items() if not k.startswith('stats/index/')} for row in rows]
    elif change == 'missing_stat':
        rows = [{k: v for k, v in row.items() if k != 'stats/action/std'} for row in rows]
    elif change == 'count':
        rows[0]['stats/action/count'] = [999]
    elif change == 'nan':
        rows[0]['stats/action/mean'] = [float('nan'), 0.]
    elif change == 'shape':
        rows[0]['stats/action/mean'] = [1.]
    elif change == 'zero_variance':
        rows[0]['stats/action/std'] = [0., 0.]
    elif change == 'quantile_order':
        rows[0]['stats/action/q01'] = [0., 1.8]
        rows[0]['stats/action/q99'] = [0., .2]
    elif change == 'quantile_bounds':
        rows[0]['stats/action/q99'] = [0., 999.]
    else:
        for row in rows:
            row['stats/extra/count'] = [1]
    pq.write_table(pa.Table.from_pylist(rows), path)
    result = rc.write_recovery_copy(source, tmp_path / 'new')
    assert result['status'] == 'unknown'
    assert result['statistics']['status'] == 'unknown'
    assert not (tmp_path / 'new/meta/stats.json').exists()


def test_no_user_plan_file_is_accepted(tmp_path):
    source = recording(tmp_path)
    with pytest.raises(TypeError):
        rc.write_recovery_copy(source, tmp_path / 'new', plan={'files': []})
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('token', [None, True, 'verified', object()])
def test_incomplete_marker_cannot_be_bypassed_with_serialized_or_foreign_token(tmp_path, token):
    source = recording(tmp_path)
    (source / rc.INCOMPLETE).write_text('{"verified":true}')
    (source / rc.RECEIPT).write_text('{"status":"pass","original_unchanged":true}')
    with pytest.raises(InputError, match='incomplete RoboClue'):
        rc.check_salvage(source, _recovery_copy_token=token)


@pytest.mark.parametrize('kind', ['regular', 'link', 'fifo', 'directory'])
def test_any_incomplete_marker_presence_blocks_resume_before_child(tmp_path, kind, capsys):
    from policy_doctor.product_cli import main
    source = recording(tmp_path)
    marker = source / rc.INCOMPLETE
    if kind == 'regular':
        marker.write_text('{"verified":true}')
    elif kind == 'link':
        marker.symlink_to(tmp_path / 'absent')
    elif kind == 'fifo':
        os.mkfifo(marker)
    else:
        marker.mkdir()
    child = tmp_path / 'lerobot-record'
    sentinel = tmp_path / 'child-started'
    child.write_text('#!/bin/sh\ntouch ' + str(sentinel) + '\n')
    child.chmod(0o700)
    args = ['guard', 'resume', str(source), '--dry-run', '--json', '--', str(child),
            '--dataset.root=' + str(source), '--resume=true']
    assert main(args) != 0
    assert not sentinel.exists()
    assert 'incomplete' in capsys.readouterr().err.lower()


def test_pointer_correction_and_metadata_tail_rewrite(tmp_path):
    source = recording(tmp_path)
    path = source / 'meta/episodes/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    rows[1]['data/file_index'] = 99
    pq.write_table(pa.Table.from_pylist(rows), path)
    before = hashes(source)
    result = rc.write_recovery_copy(source, tmp_path / 'new')
    assert result['status'] == 'pass'
    corrected = pq.read_table(tmp_path / 'new/meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert corrected[1]['data/file_index'] == 0
    assert hashes(source) == before


def test_partial_metadata_file_is_trimmed_to_kept_rows(tmp_path):
    source = recording(tmp_path)
    path = source / 'meta/episodes/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    rows = table.to_pylist()
    extra = dict(rows[-1])
    extra.update(episode_index=2, length=3, dataset_from_index=7, dataset_to_index=10)
    rows.append(extra)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    before = hashes(source)
    result = rc.write_recovery_copy(source, tmp_path / 'new')
    assert result['status'] == 'pass'
    assert pq.read_table(tmp_path / 'new/meta/episodes/chunk-000/file-000.parquet').num_rows == 2
    assert hashes(source) == before


def test_sampled_image_summaries_use_native_envelopes_without_decoding():
    features = {'observation.images.cam': {'dtype': 'video', 'shape': [480, 640, 3]}}
    rows = []
    for episode, mean in enumerate((.2, .8)):
        values = {'min': 0., 'max': 1., 'mean': mean, 'std': .1, 'q01': mean - .1, 'q99': mean + .1}
        row = {'episode_index': episode, 'length': 1000}
        row.update({f'stats/observation.images.cam/{k}': np.full((3, 1, 1), v).tolist() for k, v in values.items()})
        row['stats/observation.images.cam/count'] = [100]  # sampled frames, not 1,000 images
        rows.append(row)
    result = rc._aggregate_episode_stats(rows, {'features': features}, None)['observation.images.cam']
    assert result['count'] == [200]
    np.testing.assert_allclose(result['mean'], np.full((3, 1, 1), .5))
    np.testing.assert_allclose(result['std'], np.full((3, 1, 1), np.sqrt(.1)))
    np.testing.assert_allclose(result['q01'], np.full((3, 1, 1), .1))
    np.testing.assert_allclose(result['q99'], np.full((3, 1, 1), .9))


def test_decoded_parquet_bound_refuses_before_copy(monkeypatch, tmp_path):
    source = recording(tmp_path, orphan=2)
    monkeypatch.setattr(rc, 'MAX_TABLE_BYTES', 1)
    with pytest.raises(InputError, match='budget'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('kind', ['unchanged_data', 'trimmed_data', 'statistics'])
def test_output_mutation_after_write_never_returns_verified_copy(monkeypatch, tmp_path, kind):
    source = recording(tmp_path, orphan=3 if kind == 'trimmed_data' else 0)
    before = hashes(source)
    original = rc.check_salvage
    def changed(path, **kwargs):
        result = original(path, **kwargs)
        if Path(path) != source:
            if kind == 'statistics':
                target = Path(path) / 'meta/stats.json'
                document = json.loads(target.read_text())
                document['action']['mean'] = [9999., 9999.]
                target.write_text(json.dumps(document))
            else:
                target = Path(path) / 'data/chunk-000/file-000.parquet'
                table = pq.read_table(target)
                rows = table.to_pylist()
                rows[0]['action'] = [9999., 9999.]
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target)
        return result
    monkeypatch.setattr(rc, 'check_salvage', changed)
    with pytest.raises(InputError, match='Destination changed'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert hashes(source) == before
    assert (tmp_path / 'new' / rc.INCOMPLETE).exists()
    assert not (tmp_path / 'new' / rc.RECEIPT).exists()


def test_cumulative_metadata_decoding_budget_refuses_before_creation(monkeypatch, tmp_path):
    episodes = [(n, 1) for n in range(64)]
    metadata = meta_for(episodes)
    for index, row in enumerate(metadata):
        row['meta/episodes/file_index'] = index
    source = write(tmp_path / 'source', [rows_for(episodes)], [[row] for row in metadata], 64, 64)
    paths = sorted((source / 'meta/episodes').rglob('*.parquet'))
    per_file = max(pq.ParquetFile(p).metadata.row_group(0).total_byte_size for p in paths) + 1
    assert sum(pq.read_table(p).nbytes for p in paths) > per_file
    monkeypatch.setattr(rc, 'MAX_TABLE_BYTES', per_file)
    with pytest.raises(InputError, match='cumulative'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_directory_sync_failure_leaves_marker_even_with_completed_receipt(monkeypatch, tmp_path):
    import stat
    source = recording(tmp_path)
    original = os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError('injected directory sync failure')
        return original(fd)
    monkeypatch.setattr(rc.os, 'fsync', fail_directory)
    with pytest.raises(InputError, match='incomplete and unverified'):
        rc.write_recovery_copy(source, tmp_path / 'new')
    assert (tmp_path / 'new' / rc.INCOMPLETE).exists()
    with pytest.raises(InputError, match='incomplete RoboClue'):
        rc.check_salvage(tmp_path / 'new')


def test_tiny_positive_variance_remains_supported():
    rows = []
    for episode in range(2):
        row = {'episode_index': episode, 'length': 2}
        for key, value in {'min': 0., 'max': 2e-40, 'mean': 1e-40, 'std': 1e-40}.items():
            row[f'stats/action/{key}'] = [value]
        row['stats/action/count'] = [2]
        rows.append(row)
    result = rc._aggregate_episode_stats(rows, {'features': {'action': {'dtype': 'float64', 'shape': [1]}}}, None)
    assert result['action']['std'][0] > 0
    np.testing.assert_allclose(result['action']['std'], [1e-40], rtol=1e-12, atol=0)
