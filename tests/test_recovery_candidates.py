"""Candidate inclusion uses surviving source rows; ambiguous reconstruction fails closed."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_salvage import hashes, meta_for, rows_for, write

from policy_doctor.artifacts import InputError
from policy_doctor.recovery_copy import write_recovery_copy


def source_recording(tmp_path):
    first = rows_for([(0, 3)])
    second = rows_for([(1, 4)], start=3)
    meta = meta_for([(0, 3)])
    meta[0]['tasks'] = ['t']
    for name, size in [('action', 2), ('index', 1)]:
        values = np.asarray([r[name] for r in first], dtype=float).reshape(3, size)
        stats = {'mean': values.mean(0), 'std': values.std(0), 'min': values.min(0),
                 'max': values.max(0), 'count': np.array([3]), 'q50': np.quantile(values, .5, axis=0)}
        meta[0].update({f'stats/{name}/{k}': v.tolist() for k, v in stats.items()})
    root = write(tmp_path/'source', [first, second], [meta], 2, 7)
    info_path = root/'meta/info.json'
    info = json.loads(info_path.read_text())
    info['data_files_size_in_mb'] = .0001
    info_path.write_text(json.dumps(info))
    saved = {}
    for name, size in [('action', 2), ('index', 1)]:
        values = np.asarray([r[name] for r in first + second], dtype=float).reshape(7, size)
        saved[name] = {'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(),
                       'min': values.min(0).tolist(), 'max': values.max(0).tolist(), 'count': [7]}
    (root/'meta/stats.json').write_text(json.dumps(saved))
    return root


def test_included_numeric_episode_reconstructs_own_source_stats_and_tasks(tmp_path):
    source = source_recording(tmp_path)
    before = hashes(source)
    result = write_recovery_copy(source, tmp_path/'new', include_candidates=True)
    assert result['status'] == 'pass'
    assert result['statistics']['candidates']['included_episodes'] == [1]
    assert result['statistics']['candidates']['included_frames'] == 4
    rows = pq.read_table(tmp_path/'new/meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert rows[1]['tasks'] == ['t'] and rows[1]['length'] == 4
    assert rows[1]['stats/action/mean'] == [1., 1.5]
    stats = json.loads((tmp_path/'new/meta/stats.json').read_text())
    np.testing.assert_allclose(stats['action']['mean'], [4/7, 9/7])
    assert stats['index']['count'] == [7]
    assert hashes(source) == before


@pytest.mark.parametrize('value', [1, 'true', None, []])
def test_include_flag_strict(tmp_path, value):
    source = source_recording(tmp_path)
    with pytest.raises(InputError, match='Boolean'):
        write_recovery_copy(source, tmp_path/'new', include_candidates=value)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('fault', ['threshold', 'template', 'tasks', 'layout', 'unknown_metadata', 'splits',
                                  'missing_summary', 'unknown_stats_feature', 'string_numeric'])
def test_ambiguous_layout_refused_before_creation(tmp_path, fault):
    source = source_recording(tmp_path)
    info_path, meta_path = source/'meta/info.json', source/'meta/episodes/chunk-000/file-000.parquet'
    info, rows = json.loads(info_path.read_text()), pq.read_table(meta_path).to_pylist()
    if fault == 'threshold':
        info['data_files_size_in_mb'] = 100
    elif fault == 'template':
        info['video_path'] = 'videos/custom.mp4'
    elif fault == 'tasks':
        table = pq.read_table(source/'data/chunk-000/file-001.parquet').to_pylist()
        for r in table:
            r['task_index'] = 27
        pq.write_table(pa.Table.from_pylist(table), source/'data/chunk-000/file-001.parquet')
    elif fault == 'layout':
        rows[0]['tasks'] = ['wrong task']
    elif fault == 'unknown_metadata':
        rows[0]['unproven_operator_label'] = 'do not borrow me'
    elif fault == 'splits':
        info['splits'] = {'train': '0:1', 'test': '1:2'}
    elif fault == 'missing_summary':
        del rows[0]['stats/action/std']
    elif fault == 'unknown_stats_feature':
        rows[0]['stats/borrowed_camera/mean'] = [1.]
    elif fault == 'string_numeric':
        table = pq.read_table(source/'data/chunk-000/file-001.parquet').to_pylist()
        for r in table:
            r['action'] = [str(v) for v in r['action']]
        pq.write_table(pa.Table.from_pylist(table), source/'data/chunk-000/file-001.parquet')
    info_path.write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist(rows), meta_path)
    before = hashes(source)
    with pytest.raises(InputError):
        write_recovery_copy(source, tmp_path/'new', include_candidates=True)
    assert not (tmp_path/'new').exists() and hashes(source) == before


def test_missing_destination_parent_named_and_unchanged(tmp_path):
    source = source_recording(tmp_path)
    parent = tmp_path/'missing-parent'
    before = hashes(source)
    with pytest.raises(InputError, match='destination parent does not exist') as caught:
        write_recovery_copy(source, parent/'new')
    assert str(parent) in str(caught.value) and hashes(source) == before
    assert not parent.exists()


@pytest.mark.parametrize('kind', ['file', 'link', 'directory'])
def test_sibling_receipt_collision_refuses_before_creation(tmp_path, kind):
    source = source_recording(tmp_path)
    sidecar = tmp_path/'new.roboclue-recovery.json'
    if kind == 'file':
        sidecar.write_text('private existing receipt')
    elif kind == 'link':
        sidecar.symlink_to(tmp_path/'absent')
    else:
        sidecar.mkdir()
    with pytest.raises(InputError, match='receipt already exists'):
        write_recovery_copy(source, tmp_path/'new')
    assert not (tmp_path/'new').exists()
