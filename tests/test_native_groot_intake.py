import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_groot_intake import prepare_native_groot_inputs, verify_native_groot_inputs


@pytest.fixture
def folders(tmp_path):
    from policy_doctor.native_groot_adapter_worker import LAYOUT, STATE_KEYS, STREAMS
    model, dataset, workspace = (tmp_path / n for n in ['model', 'dataset', 'workspace'])
    for p in (model, dataset / 'meta', workspace):
        p.mkdir(parents=True)
    modalities = {k: {'delta_indices': [0], 'modality_keys': v}
                  for k, v in [('video', ['image', 'wrist_image']), ('state', STATE_KEYS),
                               ('action', STATE_KEYS), ('language', ['task'])]}
    config = {'processor_class': 'Gr00tN1d6Processor', 'processor_kwargs': {
        'modality_configs': {'libero_panda': modalities}, 'use_percentiles': False,
        'clip_outliers': True, 'apply_sincos_state_encoding': False, 'use_relative_action': False}}
    stats = {'libero_panda': {role: {k: {n: [1.0] * (2 if role == 'state' and k == 'gripper' else 1)
                                       for n in ['mean', 'std', 'min', 'max', 'q01', 'q99']}
                                      for k in STATE_KEYS} for role in ['state', 'action']}}
    (model / 'processor_config.json').write_text(json.dumps(config))
    (model / 'statistics.json').write_text(json.dumps(stats))
    info = {'codebase_version': 'v2.1', 'fps': 20, 'chunks_size': 10,
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': {'observation.state': {'dtype': 'float32', 'shape': [8], 'names': {'motors': LAYOUT}},
                         'action': {'dtype': 'float32', 'shape': [7]},
                         **{k: {'dtype': 'float32' if k == 'timestamp' else 'int64', 'shape': [1]}
                            for k in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']},
                         **{s['source']: {'dtype': 'video', 'shape': [256, 256, 3]} for s in STREAMS}}}
    (dataset / 'meta/info.json').write_text(json.dumps(info))
    (dataset / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 13, 'length': 3, 'tasks': ['generated fixture']}) + '\n')
    (dataset / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'generated fixture'}) + '\n')
    payload = dataset / 'data/chunk-001/episode_000013.parquet'
    payload.parent.mkdir(parents=True)
    # Intake is byte/header-only; actual Arrow allocation/value checks are a separate worker gate.
    payload.write_bytes(b'PAR1' + b'x' * 8 + (8).to_bytes(4, 'little') + b'PAR1')
    for stream in STREAMS:
        video = dataset / f"videos/chunk-001/{stream['source']}/episode_000013.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b'generated opaque video fixture')
    return model, dataset, workspace


def test_ordinary_inputs_exact_selection(folders):
    model, dataset, workspace = folders
    (model / 'unrelated.bin').write_bytes(b'x')
    result = prepare_native_groot_inputs(model, dataset, [{'episode_index': 13, 'frame_index': 1}], workspace)
    assert result['model'].is_dir() and result['dataset'].is_dir()
    assert not (result['model'] / 'unrelated.bin').exists()
    assert len([x for x in result['receipts'] if x['root'] == 'dataset']) == 6
    for item in result['receipts']:
        assert (result[item['root']] / item['path']).read_bytes() == ({'model': model, 'dataset': dataset}[item['root']] / item['path']).read_bytes()
    verify_native_groot_inputs(result['selection'])


@pytest.mark.parametrize('mutation', ['source', 'copy', 'extra', 'symlink'])
def test_selected_identity_and_membership_guard(folders, mutation):
    model, dataset, workspace = folders
    result = prepare_native_groot_inputs(model, dataset, [{'episode_index': 13, 'frame_index': 0}], workspace)
    if mutation == 'source':
        (model / 'statistics.json').write_text('{}')
    elif mutation == 'copy':
        (result['model'] / 'statistics.json').write_text('{}')
    elif mutation == 'extra':
        (result['dataset'] / 'extra.json').write_text('{}')
    else:
        p = result['model'] / 'statistics.json'
        p.unlink()
        p.symlink_to(model / 'statistics.json')
    with pytest.raises(InputError):
        verify_native_groot_inputs(result['selection'])


def test_reject_custom_processor_before_copying_payloads(folders):
    model, dataset, workspace = folders
    p = model / 'processor_config.json'
    value = json.loads(p.read_text())
    value['processor_class'] = 'AttackerProcessor'
    p.write_text(json.dumps(value))
    with pytest.raises(InputError):
        prepare_native_groot_inputs(model, dataset, [{'episode_index': 13, 'frame_index': 0}], workspace)


def test_reject_symlink_root(folders):
    model, dataset, workspace = folders
    linked = workspace.parent / 'linked'
    linked.symlink_to(dataset, target_is_directory=True)
    with pytest.raises(InputError):
        prepare_native_groot_inputs(model, linked, [{'episode_index': 13, 'frame_index': 0}], workspace)
