"""Declared native feature names, never guessed state aliases, select scan columns."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.dataset import scan_dataset
from policy_doctor.evidence import inspect_evidence


def dataset(tmp_path, *, declared=None, values=None):
    root = tmp_path / 'native-dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    declared = {'state': {'dtype': 'float32', 'shape': [2]}} if declared is None else declared
    features = {'action': {'dtype': 'float32', 'shape': [2]}, **declared,
                'observation.images.front': {'dtype': 'video', 'shape': [32, 32, 3]}}
    rows = [{'action': [0., 1.], 'timestamp': i / 20, 'episode_index': 0, 'frame_index': i,
             **{key: [float(i), 0.5] for key in declared}} for i in range(3)]
    if values is not None:
        values(rows)
    (root / 'meta' / 'info.json').write_text(json.dumps({'fps': 20, 'total_frames': 3, 'features': features}))
    pq.write_table(pa.Table.from_pylist(rows), root / 'data' / 'chunk-000' / 'file-000.parquet')
    return root


@pytest.mark.parametrize('key', ['state', 'robot.state', 'sensors/joints~raw'])
def test_healthy_declared_native_vector_name_is_not_replaced(tmp_path, key):
    root = dataset(tmp_path, declared={key: {'dtype': 'float32', 'shape': [2]}})
    report = scan_dataset(str(root))
    assert report['status'] == 'pass' and not report['findings']
    assert key in report['coverage']['numeric_vector_features']
    assert 'observation.state' not in report['coverage']['numeric_vector_features']


def test_state_and_observation_state_remain_distinct_and_both_are_checked(tmp_path):
    def faulty(rows):
        rows[1]['observation.state'] = [float('nan'), 1.]

    root = dataset(tmp_path, declared={key: {'dtype': 'float32', 'shape': [2]}
                                     for key in ['state', 'observation.state']}, values=faulty)
    report = scan_dataset(str(root))
    invalid = [f for f in report['findings'] if f['code'] == 'dataset.vector_invalid']
    assert report['status'] == 'error'
    assert [f['detail']['feature'] for f in invalid] == ['observation.state']
    assert {'state', 'observation.state'} <= set(report['coverage']['numeric_vector_features'])


def test_missing_native_declaration_is_error_even_if_canonical_name_exists(tmp_path):
    def renamed(rows):
        for row in rows:
            row['observation.state'] = row.pop('state')

    root = dataset(tmp_path, values=renamed)
    report = scan_dataset(str(root))
    missing = [f['detail'] for f in report['findings'] if f['code'] == 'dataset.column_missing']
    assert report['status'] == 'error' and missing == ['state']


def test_undeclared_canonical_column_does_not_override_a_valid_native_state(tmp_path):
    def extra(rows):
        for row in rows:
            row['observation.state'] = [float('nan')]

    root = dataset(tmp_path, values=extra)
    report = scan_dataset(str(root))
    assert report['status'] == 'pass'
    assert 'observation.state' not in report['coverage']['numeric_vector_features']


@pytest.mark.parametrize('bad', [[1.], [float('nan'), 0.]])
def test_custom_vector_shape_and_finiteness_checks_use_native_metadata(tmp_path, bad):
    def faulty(rows):
        rows[1]['sensors/joints~raw'] = bad

    root = dataset(tmp_path, declared={'sensors/joints~raw': {'dtype': 'float32', 'shape': [2]}}, values=faulty)
    report = scan_dataset(str(root))
    finding = next(f for f in report['findings'] if f['code'] == 'dataset.vector_invalid')
    assert report['status'] == 'error' and finding['detail']['feature'] == 'sensors/joints~raw'
    evidence = finding['evidence']
    assert evidence['pointer'] == '/features/sensors~1joints~0raw/shape'
    checked = inspect_evidence(evidence['path'], evidence['pointer'], evidence['sha256'])
    assert checked['status'] == 'pass' and checked['value'] == [2]


def test_other_declared_numeric_vectors_are_checked_without_state_name_guess(tmp_path):
    def faulty(rows):
        rows[2]['measured.effort'] = [float('inf'), 0.]

    root = dataset(tmp_path, declared={'measured.effort': {'dtype': 'float64', 'shape': [2]}}, values=faulty)
    report = scan_dataset(str(root))
    assert report['status'] == 'error'
    assert any(f['code'] == 'dataset.vector_invalid' and f['detail']['feature'] == 'measured.effort'
               for f in report['findings'])
    assert not any(f['code'] == 'dataset.column_missing' for f in report['findings'])


@pytest.mark.parametrize('key,dtype', [('index', 'int64'), ('task_index', 'int64'),
                                      ('next.reward', 'float32'), ('next.done', 'bool'),
                                      ('contact.force', 'float32')])
def test_native_shape_one_scalar_storage_is_checked_without_false_vector_error(tmp_path, key, dtype):
    def scalar(rows):
        for i, row in enumerate(rows):
            row[key] = bool(i) if dtype == 'bool' else i if dtype == 'int64' else float(i)

    root = dataset(tmp_path, declared={'state': {'dtype': 'float32', 'shape': [2]},
                                      key: {'dtype': dtype, 'shape': [1]}}, values=scalar)
    report = scan_dataset(str(root))
    assert report['status'] == 'pass' and not report['findings']
    assert key in report['coverage']['numeric_vector_features']


def test_native_scalar_scientific_value_is_still_checked_for_finiteness(tmp_path):
    def scalar(rows):
        for row in rows:
            row['contact.force'] = float('nan')

    root = dataset(tmp_path, declared={'contact.force': {'dtype': 'float32', 'shape': [1]}}, values=scalar)
    report = scan_dataset(str(root))
    invalid = [f for f in report['findings'] if f['code'] == 'dataset.vector_invalid']
    assert report['status'] == 'error' and len(invalid) == 3
    assert all(f['detail']['feature'] == 'contact.force' for f in invalid)


def test_scalar_storage_cannot_satisfy_a_declared_multi_coordinate_vector(tmp_path):
    def scalar(rows):
        for row in rows:
            row['state'] = 1.0

    report = scan_dataset(str(dataset(tmp_path, values=scalar)))
    assert report['status'] == 'error'
    assert any(f['code'] == 'dataset.vector_invalid' and f['detail']['feature'] == 'state'
               for f in report['findings'])


def test_list_stored_shape_one_vector_still_requires_one_coordinate(tmp_path):
    def vector(rows):
        for row in rows:
            row['contact.force'] = [1., 2.]

    root = dataset(tmp_path, declared={'contact.force': {'dtype': 'float32', 'shape': [1]}}, values=vector)
    report = scan_dataset(str(root))
    assert report['status'] == 'error'
    assert any(f['code'] == 'dataset.vector_invalid' and f['detail']['feature'] == 'contact.force'
               for f in report['findings'])
