import copy

import pytest

from policy_doctor import native_groot_adapter_worker as worker


def contract():
    return {'kind': 'IPEC_LIBERO_selected_source_preservation', 'association': 'explicit',
            'boundary': 'policy_get_action_input_pre_normalization',
            'state_layout': ['x', 'y', 'z', 'axis_angle1', 'axis_angle2', 'axis_angle3', 'gripper', 'gripper'],
            'ordered_source_streams': [{'input': 'agentview', 'model': 'image', 'source': 'observation.images.image'},
                                       {'input': 'wrist', 'model': 'wrist_image', 'source': 'observation.images.wrist_image'}]}


def test_contract_is_explicit_and_ordered():
    assert worker.validate_contract(contract()) == contract()
    bad = contract()
    bad['ordered_source_streams'].reverse()
    with pytest.raises(ValueError):
        worker.validate_contract(bad)


@pytest.mark.parametrize('sample', [[], [{'episode_index': True, 'frame_index': 0}],
                                   [{'episode_index': 0, 'frame_index': -1}],
                                   [{'episode_index': 0, 'frame_index': 0, 'row': 0}]])
def test_sample_schema(sample):
    with pytest.raises(ValueError):
        worker.validate_samples(sample)


def test_code_keys_refused_before_native_import():
    with pytest.raises(ValueError):
        worker.no_code({'safe': {'auto_map': {'module': 'bad'}}})
    worker.no_code({'task': 'pick up module and python book'})


def test_duplicate_and_nonfinite_json_refused(tmp_path):
    p = tmp_path / 'bad.json'
    for value in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}']:
        p.write_text(value)
        with pytest.raises(ValueError):
            worker.read_json(p)


def test_path_templates_are_data_not_code():
    info = {'chunks_size': 10, 'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'}
    assert worker.selected_path(info, 'data_path', 17) == 'data/chunk-001/episode_000017.parquet'
    for template in ['../../{episode_index}.parquet', '{episode_index.__class__}.parquet', '/{episode_index}.parquet']:
        bad = copy.deepcopy(info)
        bad['data_path'] = template
        with pytest.raises(ValueError):
            worker.selected_path(bad, 'data_path', 17)


def test_worker_error_receipt_before_native_import(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, '-B', '-I', str(Path(worker.__file__))],
                            input=json.dumps({'operation': 'bad', 'output': str(output)}),
                            capture_output=True, text=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt['execution_status'] == 'error'
    assert receipt['payload_files'] == []
    assert json.loads((output / 'worker_receipt.json').read_text()) == receipt


@pytest.mark.parametrize('key', ['state', 'recorded_action', 'image', 'wrist_image'])
def test_npz_rejects_wrong_dtype_before_values(key):
    import io
    np = pytest.importorskip('numpy')
    arrays = {'state': np.zeros(8, dtype='<f4'), 'recorded_action': np.zeros(7, dtype='<f4'),
              'image': np.zeros((256, 256, 3), dtype='u1'), 'wrist_image': np.zeros((256, 256, 3), dtype='u1')}
    arrays[key] = arrays[key].astype('float64')
    data = io.BytesIO()
    np.savez(data, **arrays)
    with pytest.raises(ValueError):
        worker._npz(data.getvalue())


@pytest.mark.parametrize('record', [{'episode_index': False, 'length': 3, 'tasks': ['task']},
                                    {'episode_index': 0, 'length': 3, 'tasks': 'task'},
                                    {'episode_index': 0, 'length': True, 'tasks': ['task']}])
def test_original_episode_schema_is_not_bool_or_string_coercion(tmp_path, record):
    import json
    path = tmp_path / 'episodes.jsonl'
    path.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError):
        worker.selected_episode(path, {'episode_index': 0, 'frame_index': 0})
