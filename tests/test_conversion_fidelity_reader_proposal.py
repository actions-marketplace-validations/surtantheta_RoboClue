"""Generated HDF5/Parquet controls only; no original arrays/native conversion."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from policy_doctor.artifacts import InputError
from policy_doctor.conversion_fidelity import check_conversion_fidelity, prepare_conversion_inputs

FIXTURES = Path(__file__).parent / 'fixtures' / 'arena43'
ROOT = 'isaaclab_arena_gr00t/'


def permutation():
    source = yaml.safe_load((FIXTURES / (ROOT + 'embodiments/g1/43dof_joint_space.yaml')).read_bytes())['joints']
    target = yaml.safe_load((FIXTURES / (ROOT + 'embodiments/g1/gr00t_43dof_joint_space.yaml')).read_bytes())['joints']
    groups = json.loads((FIXTURES / (ROOT + 'embodiments/g1/modality.json')).read_bytes())['action']
    omitted = {'left_wrist_pose', 'right_wrist_pose', 'base_height_command', 'navigate_command', 'torso_orientation_rpy_command'}
    return [source[joint] for group in groups if group not in omitted for joint in target[group]]


class Generated:
    def __init__(self, temporary):
        self.root = temporary
        self.source = temporary / 'converter'
        self.source.mkdir()
        for name, row in json.loads((FIXTURES / 'UPSTREAM.json').read_bytes())['files'].items():
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((FIXTURES / row['fixture_path']).read_bytes())
        self.original = temporary / 'generated-only.hdf5'
        self.export = temporary / 'export'
        (self.export / 'meta').mkdir(parents=True)
        self.config_path = self.source / (ROOT + 'lerobot/config/g1_static_apple_config.yaml')
        self.config = yaml.safe_load(self.config_path.read_bytes())
        self.config.update(data_root=str(temporary), hdf5_name=self.original.name)
        self.write_config()
        self.arrays = {}
        self.auxiliary = {}
        with h5py.File(self.original, 'w') as file:
            data = file.create_group('data', track_order=True)
            # Nonlexical insertion order is deliberate; native list(keys()) must win.
            for ordinal, (key, length) in enumerate([('demo_z', 8), ('demo_initial', None), ('demo_a', 6)]):
                group = data.create_group(key)
                if length is None:
                    group.create_group('initial_state')
                    continue
                action = (np.arange(length * 43).reshape(length, 43) / 64 + ordinal * 8).astype('float32')
                state = action + np.float32(100)
                group.create_group('obs').create_dataset('robot_joint_pos', data=state)
                group.create_dataset('processed_actions', data=action)
                self.arrays[ordinal] = (key, state, action)
                commands = group.create_group('action')
                self.auxiliary[ordinal] = {}
                for name, width, offset in [('base_height_cmd', 1, 1), ('navigate_cmd', 3, 1000), ('torso_orientation_rpy_cmd', 3, 2000)]:
                    values = (action[:, :width] + np.float32(offset)).astype('float32')
                    commands.create_dataset(name, data=values)
                    self.auxiliary[ordinal][name] = values
        self.episodes = [{'episode_index': eid, 'length': len(a) - 1,
                          'tasks': [self.config['language_instruction']]} for eid, (_, _, a) in self.arrays.items()]
        self.write_metadata()
        self.restore()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config, sort_keys=False))

    def write_metadata(self):
        info = {'fps': self.config['fps'], 'data_path': self.config['data_path'], 'total_episodes': 3}
        (self.export / 'meta/info.json').write_text(json.dumps(info))
        (self.export / 'meta/episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.episodes))
        (self.export / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': self.config['task_index'], 'task': self.config['language_instruction']}) + '\n')

    def path(self, eid):
        return self.export / self.config['data_path'].format(episode_chunk=eid // 1000, episode_index=eid)

    def table(self, eid):
        _, state, action = self.arrays[eid]
        state, action = state[:-1, permutation()], action[:-1, permutation()]
        length = len(action)
        start = sum(len(a) - 1 for i, (_, _, a) in self.arrays.items() if i < eid)
        columns = {'observation.state': pa.array(state.tolist(), type=pa.list_(pa.float32())),
                         'action': pa.array(action.tolist(), type=pa.list_(pa.float32())),
                         'timestamp': pa.array(np.arange(length, dtype='float64') * (1 / self.config['fps'])),
                         'frame_index': pa.array(np.arange(length, dtype='int64')),
                         'episode_index': pa.array(np.full(length, eid, dtype='int64')),
                         'index': pa.array(np.arange(length, dtype='int64') + start),
                         'task_index': pa.array(np.full(length, self.config['task_index'], dtype='int64')),
                         self.config['task_description_lerobot']: pa.array(np.full(length, self.config['task_index'], dtype='int64'))}
        for source, target in [('base_height_cmd', 'teleop.base_height_command'), ('navigate_cmd', 'teleop.navigate_command'), ('torso_orientation_rpy_cmd', 'teleop.torso_orientation_rpy_command')]:
            columns[target] = pa.array(self.auxiliary[eid][source][:-1].tolist(), type=pa.list_(pa.float32()))
        return pa.table(columns)

    def write(self, eid, table, **kwargs):
        path = self.path(eid)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, **kwargs)

    def restore(self):
        for eid in self.arrays:
            self.write(eid, self.table(eid))

    def kwargs(self, **extra):
        return dict(original_recording=str(self.original), converter_source=str(self.source),
                    converter_config=str(self.config_path), converter_directory=str(self.source),
                    read_roots=[str(self.root)], **extra)

    def check(self, **extra):
        return check_conversion_fidelity(str(self.export), **self.kwargs(**extra))


@pytest.fixture
def generated(tmp_path):
    return Generated(tmp_path)


def test_actual_generated_membership_sparse_native_order_and_scoped_pass(generated):
    result = generated.check()
    assert result['status'] == result['numeric_association_status'] == 'pass', result
    assert result['source_membership']['status'] == 'pass'
    assert result['coverage']['native_episode_ids'] == [0, 2]
    assert result['coverage']['original_source_groups'] == 3
    assert result['coverage']['compared_numeric_rows'] == 12
    assert result['coverage']['initial_state_only_omissions'][0]['native_episode_index'] == 1
    assert result['coverage']['whole_numeric_source_to_export']
    assert not result['coverage']['whole_dataset_verified']
    assert result['native_consumption']['status'] == result['pixel_fidelity']['status'] == 'unknown'
    assert result['worker']['isolated'] and not result['worker']['torch_imported']
    original = next(a for a in result['artifacts'] if a.get('kind') == 'original_hdf5')
    assert original['sha256'] == hashlib.sha256(generated.original.read_bytes()).hexdigest()
    actual = next(a for a in result['artifacts'] if a['path'] == str(generated.config_path))
    assert actual['sha256'] != '3dc9168a54d30f5d86c259b8620af075fd648d2236d8bb6c3646c72f51782c33'


@pytest.mark.parametrize('change', ['state_for_action', 'joint_permutation', 'forward_shift'])
def test_shape_valid_changes_and_corrected_export_keep_identical_criterion(generated, change):
    healthy = generated.check()
    changed_rows = 0
    for eid in generated.arrays:
        table = generated.table(eid)
        action = np.asarray(table['action'].to_pylist(), dtype='float32')
        if change == 'state_for_action':
            action = np.asarray(table['observation.state'].to_pylist(), dtype='float32')
        elif change == 'joint_permutation':
            action[:, [0, 1]] = action[:, [1, 0]]
        else:
            action = np.concatenate([action[1:], action[-1:]])
        changed_rows += int(np.any(action != np.asarray(table['action'].to_pylist()), axis=1).sum())
        generated.write(eid, table.set_column(table.schema.get_field_index('action'), 'action', pa.array(action.tolist(), type=pa.list_(pa.float32()))))
    bad = generated.check()
    assert bad['status'] == bad['numeric_association_status'] == 'error', bad
    assert bad['source_membership']['status'] == 'pass'
    assert bad['criterion_sha256'] == healthy['criterion_sha256']
    finding = next(f for f in bad['findings'] if f['code'] == 'conversion.numeric_joint_association')
    assert len(finding['evidence']) == 2 and all(len(e['sha256']) == 64 for e in finding['evidence'])
    assert 'source frame' in finding['message'] and 'joint' in finding['message'] and 'row' in finding['message']
    assert changed_rows == (10 if change == 'forward_shift' else 12)
    generated.restore()
    assert generated.check()['criterion_sha256'] == bad['criterion_sha256']
    assert generated.check()['status'] == 'pass'


def test_legitimate_parquet_encoding_change_preserves_value_criterion(generated):
    before = generated.check()
    for eid in generated.arrays:
        generated.write(eid, generated.table(eid), compression='gzip', use_dictionary=False)
    after = generated.check()
    assert after['status'] == 'pass'
    assert after['criterion_sha256'] == before['criterion_sha256']
    assert [a['sha256'] for a in after['artifacts'] if a.get('kind') == 'converted_parquet'] != [a['sha256'] for a in before['artifacts'] if a.get('kind') == 'converted_parquet']


@pytest.mark.parametrize('change', ['float64', 'nonfinite', 'truncated', 'time', 'global_index', 'native_id'])
def test_exact_dtype_extent_and_row_time_identity(generated, change):
    table = generated.table(0)
    if change in {'float64', 'nonfinite'}:
        values = np.asarray(table['action'].to_pylist(), dtype='float64' if change == 'float64' else 'float32')
        if change == 'nonfinite':
            values[0, 0] = np.inf
        value = pa.array(values.tolist(), type=pa.list_(pa.float64() if change == 'float64' else pa.float32()))
        table = table.set_column(table.schema.get_field_index('action'), 'action', value)
    elif change == 'truncated':
        table = table.slice(0, len(table) - 1)
    else:
        field = {'time': 'timestamp', 'global_index': 'index', 'native_id': 'episode_index'}[change]
        values = table[field].to_numpy() + 1
        table = table.set_column(table.schema.get_field_index(field), field, pa.array(values))
    generated.write(0, table)
    result = generated.check()
    assert result['status'] == 'error', result
    assert not result['coverage']['whole_numeric_source_to_export']


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'nominal_truncation', 'dense_ids', 'extra_file'])
def test_complete_sparse_membership_not_selected_convenient_subset(generated, change):
    if change == 'missing':
        generated.path(2).unlink()
    elif change == 'duplicate':
        generated.episodes.append(dict(generated.episodes[0]))
    elif change == 'nominal_truncation':
        generated.episodes[0]['length'] -= 1
    elif change == 'dense_ids':
        generated.episodes[1]['episode_index'] = 1
    else:
        generated.write(9, generated.table(0))
    generated.write_metadata()
    result = generated.check()
    assert result['status'] == 'error', result
    assert result['source_membership']['status'] == 'error', result
    assert not result['coverage']['whole_numeric_source_to_export']


def test_known_value_error_survives_other_export_refusal(generated):
    table = generated.table(0)
    values = np.asarray(table['action'].to_pylist(), dtype='float32') + 1
    generated.write(0, table.set_column(1, 'action', pa.array(values.tolist(), type=pa.list_(pa.float32()))))
    generated.path(2).write_bytes(b'not a parquet')
    result = generated.check()
    assert result['status'] == 'error' and result['counts']['unknown'] > 0, result
    assert any(f['code'] == 'conversion.numeric_joint_association' for f in result['findings'])


@pytest.mark.parametrize('change', ['soft_link', 'hard_link', 'external', 'wrong_width'])
def test_source_unsupported_members_never_false_pass(generated, change):
    with h5py.File(generated.original, 'r+') as file:
        group = file['data/demo_z']
        if change == 'wrong_width':
            del group['processed_actions']
            group.create_dataset('processed_actions', data=np.zeros((8, 42), dtype='float32'))
        elif change == 'hard_link':
            group['alias'] = group['processed_actions']
        else:
            del group['processed_actions']
            group['processed_actions'] = h5py.SoftLink('/outside') if change == 'soft_link' else h5py.ExternalLink('never-open.hdf5', '/outside')
    result = generated.check()
    assert result['status'] != 'pass' and result['counts']['unknown'] > 0, result
    assert not result['coverage'].get('whole_numeric_source_to_export')


def test_limits_missing_cwd_and_strict_args(generated):
    for bounds in [{'max_rows': 1}, {'max_files': 13}]:
        result = generated.check(**bounds)
        assert result['status'] == 'unknown' and not result['coverage']['whole_numeric_source_to_export'], result
    kwargs = generated.kwargs()
    kwargs['converter_directory'] = None
    with pytest.raises(InputError, match='explicit converter_directory'):
        prepare_conversion_inputs(str(generated.export), **kwargs)
    for extra in [{'max_rows': True}, {'max_files': 1.5}]:
        with pytest.raises(InputError):
            generated.check(**extra)


def test_extracted_path_authority_before_worker_or_original_open(generated, monkeypatch):
    import policy_doctor.conversion_fidelity as module
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('worker launched before authorization'))
    generated.config['modality_template_path'] = '/outside/not-authorized.json'
    generated.write_config()
    with pytest.raises(InputError, match='outside authorized'):
        generated.check()
    generated.config['modality_template_path'] = ROOT + 'embodiments/g1/modality.json'
    generated.config['hdf5_name'] = 'different.hdf5'
    generated.write_config()
    with pytest.raises(InputError, match='must match'):
        generated.check()


@pytest.mark.parametrize('change', ['semantic_task', 'code', 'alias_yaml', 'duplicate_yaml'])
def test_unsupported_recipe_never_imported_or_constructed(generated, change):
    if change == 'semantic_task':
        generated.config['state_name_sim'] = 'unqualified_state'
        generated.write_config()
    elif change == 'code':
        path = generated.source / (ROOT + 'utils/io_utils.py')
        path.write_text('raise RuntimeError("must never execute")')
    elif change == 'alias_yaml':
        generated.config_path.write_text('x: &alias []\ny: *alias\n')
    else:
        generated.config_path.write_text('data_root: a\ndata_root: b\n')
    if change.endswith('yaml'):
        with pytest.raises(InputError):
            generated.check()
    else:
        result = generated.check()
        assert result['status'] == 'unknown' and not any(a.get('kind') == 'original_hdf5' for a in result['artifacts'])


def test_prepared_readset_drift_and_argument_substitution(generated):
    plan = prepare_conversion_inputs(str(generated.export), **generated.kwargs())
    generated.config_path.write_text(generated.config_path.read_text() + '\n# changed after authorization\n')
    result = check_conversion_fidelity(str(generated.export), **generated.kwargs(), _prepared=plan)
    assert result['status'] == 'unknown'
    assert not any(a.get('kind') == 'original_hdf5' for a in result['artifacts'])
    with pytest.raises(InputError, match='does not match'):
        check_conversion_fidelity(str(generated.export), **generated.kwargs(max_rows=1), _prepared=plan)


def test_relocated_candidate_keeps_same_criterion(generated):
    import shutil
    initial = generated.check()
    moved = generated.root / 'later-relocated-candidate'
    shutil.copytree(generated.export, moved)
    generated.export = moved
    result = generated.check()
    assert result['status'] == 'pass'
    assert result['criterion_sha256'] == initial['criterion_sha256']


def test_bounded_yaml_depth_before_worker(generated):
    generated.config_path.write_text('nested: ' + '[' * 33 + '0' + ']' * 33)
    with pytest.raises(InputError, match='depth32'):
        generated.check()


def test_known_error_survives_empty_other_episode(generated):
    table = generated.table(0)
    values = np.asarray(table['action'].to_pylist(), dtype='float32') + 1
    generated.write(0, table.set_column(1, 'action', pa.array(values.tolist(), type=pa.list_(pa.float32()))))
    generated.write(2, generated.table(2).slice(0, 0))
    result = generated.check()
    assert result['status'] == 'error' and result['counts']['unknown'] > 0, result
    assert result['numeric_association_status'] == 'error', result


def test_source_stamp_drift_refused_before_value_adjudication(generated, monkeypatch, tmp_path):
    import os

    import policy_doctor.conversion_fidelity_worker as worker
    plan = prepare_conversion_inputs(str(generated.export), **generated.kwargs())
    original_get = h5py.Dataset.__getitem__
    def changed(dataset, key):
        result = original_get(dataset, key)
        if dataset.name.endswith('processed_actions'):
            before = generated.original.stat()
            os.utime(generated.original, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        return result
    monkeypatch.setattr(h5py.Dataset, '__getitem__', changed)
    monkeypatch.setattr(worker, 'evaluate_numeric', lambda *a: pytest.fail('stale original hash reached adjudication'))
    with pytest.raises(InputError, match='changed before numeric adjudication'):
        worker.read_conversion(plan, tmp_path)


def test_membership_reenumeration_blocks_false_complete(generated, monkeypatch, tmp_path):
    import policy_doctor.conversion_fidelity_worker as worker
    plan = prepare_conversion_inputs(str(generated.export), **generated.kwargs())
    original = worker._data_files
    calls = []
    def membership(*args):
        result = original(*args)
        calls.append(result)
        return result | {'data/chunk-000/episode_999999.parquet'} if len(calls) == 2 else result
    monkeypatch.setattr(worker, '_data_files', membership)
    result = worker.read_conversion(plan, tmp_path)
    assert result['status'] == 'unknown' and result['numeric_association_status'] == 'pass'
    assert not result['coverage']['whole_numeric_source_to_export']
    assert any(f['code'] == 'conversion.export_membership_changed' for f in result['findings'])


@pytest.mark.parametrize('change', ['vector42', 'inner_null', 'missing_action', 'integer_action'])
def test_readable_export_contract_contradictions_are_errors(generated, change):
    table = generated.table(0)
    if change == 'missing_action':
        table = table.drop(['action'])
    else:
        values = table['action'].to_pylist()
        if change == 'vector42':
            values = [row[:-1] for row in values]
        elif change == 'inner_null':
            values[0][0] = None
        else:
            values = [[int(value) for value in row] for row in values]
        dtype = pa.int32() if change == 'integer_action' else pa.float32()
        table = table.set_column(1, 'action', pa.array(values, type=pa.list_(dtype)))
    generated.write(0, table)
    result = generated.check()
    assert result['status'] == result['numeric_association_status'] == 'error', result
    assert not result['coverage']['whole_numeric_source_to_export']
    assert any(f['code'] in {'conversion.export_vector_width', 'conversion.null_export',
        'conversion.missing_export_columns', 'conversion.dtype_changed'} for f in result['findings'])
