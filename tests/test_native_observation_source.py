"""Conditional adapter-contract checks use typed evidence, never imported verdicts."""
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_observation_source import (
    audit_native_observation_source,
    bind_native_observation_source_parameters,
    validate_parameters,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(root):
    """Generated unit evidence; actual native original captures are qualified separately."""
    root.mkdir()
    state = np.array([.1, .2, .3, 3.1, .005, -.08, .03, -.03], dtype='<f4')
    images = dict(image=np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
                  wrist_image=np.zeros((256, 256, 3), dtype=np.uint8))
    identity = dict(episode_index=0, frame_index=0, global_index=0, timestamp=0.0, task_index=0)
    layout = ['x', 'y', 'z', 'axis_angle1', 'axis_angle2', 'axis_angle3', 'gripper', 'gripper']
    streams = [dict(input='agentview', model='image', source='observation.images.image'),
               dict(input='wrist', model='wrist_image', source='observation.images.wrist_image')]
    contract = dict(kind='IPEC_LIBERO_selected_source_preservation', association='explicit',
                    boundary='policy_get_action_input_pre_normalization', state_layout=layout,
                    ordered_source_streams=streams)
    groups = dict(x=1, y=1, z=1, roll=1, pitch=1, yaw=1, gripper=2)
    def tensor(value, shape):
        value = np.asarray(value, dtype='<f4')
        return dict(dtype='<f4', shape=shape, values=value.tolist(), sha256=sha(value.tobytes()))
    captured_state, offset = {}, 0
    for name, width in groups.items():
        captured_state[name] = tensor(state[offset:offset + width], [1, 1, width])
        offset += width
    raw = dict(kind='native_adapter_boundary_capture', embodiment='LIBERO_PANDA', contract=contract,
               input_identity=identity, input_state=state.tolist(), state_supplied=True,
               row_witness=dict(state_sha256=sha(state.tobytes()), dtype='<f4', shape=[8]),
               statistics_state_dims=groups, input_images={}, source_module_sha256='a' * 64,
               policy_loader='explicitly substituted; no weights',
               policy_sink='capture+generated zero action return; not inference',
               captured=dict(state=captured_state, video={}),
               native_state_component=dict(output=captured_state), criterion=dict(status='pass'))
    metadata = dict(**identity, state_dtype='<f4', state_shape=[8], state_values=state.tolist(),
                    state_sha256=sha(state.tobytes()), cameras={}, task='pick object')
    for name, image in images.items():
        digest = sha(image.tobytes())
        raw['input_images'][name] = dict(association='explicit', supplied=True,
                                        source_key='observation.images.' + name, sha256=digest)
        raw['captured']['video'][name] = dict(dtype='|u1', shape=[1, 1, 256, 256, 3], sha256=digest)
        metadata['cameras'][name] = dict(ordinal=0, pts=0, time_base='1/20', time_seconds=0.0,
                                         pixel_sha256=digest, pixel_dtype='|u1', pixel_shape=[256, 256, 3],
                                         source_key='observation.images.' + name,
                                         source_sha256=sha(b'video receipt ' + name.encode()),
                                         source_relative_path='videos/chunk-000/observation.images.' + name + '/episode_000000.mp4',
                                         source_file='/unavailable/videos/chunk-000/observation.images.' + name + '/episode_000000.mp4')
    (root / 'samples').mkdir()
    np.savez(root / 'samples/arrays.npz', state=state, recorded_action=np.zeros(7, dtype='<f4'), **images)
    metadata['arrays_sha256'] = sha((root / 'samples/arrays.npz').read_bytes())
    (root / 'samples/metadata.json').write_text(json.dumps(metadata))
    bindings = []
    def bind(kind, name, data):
        path = root / 'originals' / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        bindings.append(dict(root=kind, path=name, retained_path=path.relative_to(root).as_posix(),
                             sha256=sha(data), size_bytes=len(data)))
    modality = dict(state=dict(modality_keys=list(groups), delta_indices=[0]),
                    video=dict(modality_keys=['image', 'wrist_image'], delta_indices=[0]))
    bind('model', 'processor_config.json', json.dumps(dict(processor_kwargs=dict(modality_configs=dict(libero_panda=modality)))).encode())
    bind('model', 'statistics.json', json.dumps(dict(libero_panda=dict(state={k: dict(min=[0.] * w, max=[1.] * w) for k, w in groups.items()}))).encode())
    features = {'observation.state': dict(dtype='float32', shape=[8], names=dict(motors=layout))}
    features.update({stream['source']: dict(dtype='video', shape=[256, 256, 3]) for stream in streams})
    info = dict(codebase_version='v2.1', chunks_size=1000, features=features,
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    bind('dataset', 'meta/info.json', json.dumps(info).encode())
    bind('dataset', 'meta/episodes.jsonl', json.dumps(dict(episode_index=0, length=100, tasks=['pick object'])).encode())
    bind('dataset', 'meta/tasks.jsonl', json.dumps(dict(task_index=0, task='pick object')).encode())
    parquet = b'generated unit receipt; no Parquet decoder or origin claim'
    bind('dataset', 'data/chunk-000/episode_000000.parquet', parquet)
    raw['row_witness']['parquet_source_sha256'] = sha(parquet)
    for name in ['image', 'wrist_image']:
        bind('dataset', f'videos/chunk-000/observation.images.{name}/episode_000000.mp4', b'video receipt ' + name.encode())
    (root / 'raw').mkdir()
    capture = dict(schema_version=1, kind='native_observation_source_capture', evidence_role='native_adapter_dry_run',
                   result={}, sample={}, contract=raw['contract'], bindings=bindings)
    for key in ['arrays', 'metadata']:
        path = f'samples/{"arrays.npz" if key == "arrays" else "metadata.json"}'
        capture['sample'][key] = dict(path=path, sha256=sha((root / path).read_bytes()))
    save(root, capture, raw)
    return capture, raw


def save(root, capture, raw):
    (root / 'raw/result.json').write_text(json.dumps(raw))
    capture['result'] = dict(path='raw/result.json', sha256=sha((root / 'raw/result.json').read_bytes()))
    (root / 'capture.json').write_text(json.dumps(capture))


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / 'view'
    capture, raw = fixture(root)
    return root, capture, raw


def test_corrected_passes_independently_of_imported_verdict(evidence):
    root, capture, raw = evidence
    raw['criterion'] = {'status': 'error'}
    raw['native_processing_status'] = 'failed'
    save(root, capture, raw)
    result = audit_native_observation_source(root)
    assert result['status'] == 'pass'
    assert result['native_processing_status'] == 'complete'
    assert result['native_replay_status'] == result['original_authenticity_status'] == 'unknown'


def test_equal_width_wrong_values_error(evidence):
    root, capture, raw = evidence
    leaf = raw['captured']['state']['roll']
    leaf['values'] = raw['captured']['state']['pitch']['values'][:]
    leaf['sha256'] = sha(np.array(leaf['values'], dtype='<f4').tobytes())
    raw['criterion'] = {'status': 'pass'}
    save(root, capture, raw)
    assert audit_native_observation_source(root)['source_criterion']['code'] == 'source_contradiction'


@pytest.mark.parametrize('field', ['state_supplied', 'input_state', 'row_witness', 'input_identity', 'captured', 'source_module_sha256', 'policy_loader', 'policy_sink'])
def test_missing_positive_witness_unknown(evidence, field):
    root, capture, raw = evidence
    raw.pop(field)
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


@pytest.mark.parametrize('field', ['supplied', 'association', 'source_key', 'sha256'])
def test_missing_camera_witness_unknown(evidence, field):
    root, capture, raw = evidence
    raw['input_images']['wrist_image'].pop(field)
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


def test_positive_omission_is_error(evidence):
    root, capture, raw = evidence
    raw['input_images']['wrist_image']['supplied'] = False
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'error'


def test_bound_metadata_and_input_identity(evidence):
    root, capture, raw = evidence
    params = bind_native_observation_source_parameters(root)
    assert set(params) == {'contract', 'model_identity'}
    assert audit_native_observation_source(root, **params)['status'] == 'pass'
    assert audit_native_observation_source(root, model_identity=params['model_identity'])['status'] == 'pass'
    assert audit_native_observation_source(root, contract=params['contract'], model_identity='0' * 64)['status'] == 'unknown'


@pytest.mark.parametrize('mutation', ['escape', 'symlink', 'tamper', 'duplicate', 'size_bool', 'raw_hash'])
def test_unsafe_or_false_retained_receipt_refused(evidence, mutation, tmp_path):
    root, capture, raw = evidence
    binding = capture['bindings'][0]
    if mutation == 'escape':
        binding['retained_path'] = '../escape'
    elif mutation == 'symlink':
        path = root / binding['retained_path']
        other = tmp_path / 'other'
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    elif mutation == 'tamper':
        (root / binding['retained_path']).write_bytes(b'changed')
    elif mutation == 'duplicate':
        capture['bindings'].append(dict(binding))
    elif mutation == 'size_bool':
        binding['size_bytes'] = True
    elif mutation == 'raw_hash':
        capture['result']['sha256'] = '0' * 64
    if mutation != 'raw_hash':
        save(root, capture, raw)
    else:
        (root / 'capture.json').write_text(json.dumps(capture))
    with pytest.raises(InputError):
        audit_native_observation_source(root)


@pytest.mark.parametrize('value', [None, [], {'extra': 1}, {'contract': {}}, {'model_identity': True}])
def test_parameters_strict(value):
    with pytest.raises(InputError):
        validate_parameters(value)


@pytest.mark.parametrize('mutation', ['boolean_shape', 'boolean_width', 'bad_tensor_digest', 'giant_pts', 'missing_component', 'missing_video_shape'])
def test_typed_payload_controls(evidence, mutation):
    root, capture, raw = evidence
    if mutation == 'boolean_shape':
        raw['captured']['state']['x']['shape'] = [True, True, 1]
    elif mutation == 'boolean_width':
        raw['statistics_state_dims']['x'] = True
    elif mutation == 'bad_tensor_digest':
        raw['captured']['state']['x']['sha256'] = 'f' * 64
    elif mutation == 'missing_component':
        raw.pop('native_state_component')
    elif mutation == 'missing_video_shape':
        raw['captured']['video']['image'].pop('shape')
    else:
        meta = json.loads((root / 'samples/metadata.json').read_text())
        meta['cameras']['image']['pts'] = 10**400
        data = json.dumps(meta).encode()
        (root / 'samples/metadata.json').write_bytes(data)
        capture['sample']['metadata']['sha256'] = sha(data)
    save(root, capture, raw)
    result = audit_native_observation_source(root)
    if mutation == 'missing_component':
        assert result['status'] == 'pass'
        assert result['native_state_component_status'] == 'unknown'
    else:
        assert result['status'] == 'unknown'


def test_black_supplied_image_is_preserved(evidence):
    root, _, _ = evidence
    assert audit_native_observation_source(root)['status'] == 'pass'


def test_positive_distinct_camera_swap_is_error(evidence):
    root, capture, raw = evidence
    videos = raw['captured']['video']
    videos['image'], videos['wrist_image'] = videos['wrist_image'], videos['image']
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'error'


def test_raw_dictionary_order_does_not_define_group_order(evidence):
    root, capture, raw = evidence
    raw['statistics_state_dims'] = dict(reversed(list(raw['statistics_state_dims'].items())))
    raw['captured']['state'] = dict(reversed(list(raw['captured']['state'].items())))
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'pass'


@pytest.mark.parametrize('path', ['processor_config.json', 'statistics.json', 'meta/info.json', 'meta/tasks.jsonl',
                                  'meta/episodes.jsonl', 'data/chunk-000/episode_000000.parquet',
                                  'videos/chunk-000/observation.images.image/episode_000000.mp4'])
def test_missing_original_binding_unknown(evidence, path):
    root, capture, raw = evidence
    capture['bindings'] = [b for b in capture['bindings'] if b['path'] != path]
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


@pytest.mark.parametrize('mutation', ['object', 'extra', 'missing', 'duplicate', 'huge_header'])
def test_pickle_and_ambiguous_npz_refused(evidence, mutation):
    root, capture, raw = evidence
    path = root / 'samples/arrays.npz'
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    if mutation == 'object':
        arrays['state'] = np.array([object()], dtype=object)
    elif mutation == 'extra':
        arrays['unexpected'] = np.zeros(1)
    elif mutation == 'missing':
        arrays.pop('state')
    if mutation in ['object', 'extra', 'missing']:
        np.savez(path, **arrays)
    elif mutation == 'duplicate':
        with zipfile.ZipFile(path, 'a') as archive:
            payload = archive.read('state.npy')
            with pytest.warns(UserWarning, match='Duplicate'):
                archive.writestr('state.npy', payload)
    else:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(descr='<f8', fortran_order=False, shape=(10**12,)))
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('state.npy', header.getvalue())
    meta_path = root / 'samples/metadata.json'
    metadata = json.loads(meta_path.read_text())
    metadata['arrays_sha256'] = sha(path.read_bytes())
    meta_path.write_text(json.dumps(metadata))
    capture['sample']['arrays']['sha256'] = sha(path.read_bytes())
    capture['sample']['metadata']['sha256'] = sha(meta_path.read_bytes())
    save(root, capture, raw)
    with pytest.raises(InputError):
        audit_native_observation_source(root)


def test_fifo_original_refused_without_waiting(evidence):
    root, _, _ = evidence
    path = root / 'originals/model/processor_config.json'
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(InputError):
        audit_native_observation_source(root)


def test_duplicate_json_key_refused(evidence):
    root, _, _ = evidence
    path = root / 'capture.json'
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(InputError):
        audit_native_observation_source(root)


def test_model_statistics_dictionary_reordering_is_lawful_unbound(evidence):
    root, capture, raw = evidence
    binding = next(b for b in capture['bindings'] if b['path'] == 'statistics.json')
    path = root / binding['retained_path']
    data = json.loads(path.read_text())
    data['libero_panda']['state'] = dict(reversed(list(data['libero_panda']['state'].items())))
    path.write_text(json.dumps(data))
    binding['sha256'] = sha(path.read_bytes())
    binding['size_bytes'] = path.stat().st_size
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'pass'


def test_input_digest_separates_available_sources_from_omission_recipe(evidence):
    root, capture, raw = evidence
    before = audit_native_observation_source(root)
    raw['input_images']['wrist_image']['supplied'] = False
    save(root, capture, raw)
    after = audit_native_observation_source(root)
    assert before['status'] == 'pass' and after['status'] == 'error'
    assert before['input_digest'] == after['input_digest']


@pytest.mark.parametrize('field', ['input_identity', 'row_witness', 'statistics_state_dims', 'captured', 'input_images'])
def test_malformed_positive_shapes_stay_unknown(evidence, field):
    root, capture, raw = evidence
    raw[field] = [1, 2, 3]
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


@pytest.mark.parametrize('field', ['state_delta_float', 'video_delta_float', 'state_shape_float', 'episode_tasks_empty'])
def test_original_metadata_exact_types(evidence, field):
    root, capture, raw = evidence
    filename = 'processor_config.json' if 'delta' in field else 'meta/info.json' if 'shape' in field else 'meta/episodes.jsonl'
    binding = next(b for b in capture['bindings'] if b['path'] == filename)
    path = root / binding['retained_path']
    data = json.loads(path.read_text())
    if 'delta' in field:
        modality = 'state' if field.startswith('state') else 'video'
        data['processor_kwargs']['modality_configs']['libero_panda'][modality]['delta_indices'] = [0.0]
    elif 'shape' in field:
        data['features']['observation.state']['shape'] = [8.0]
    else:
        data['tasks'] = []
    path.write_text(json.dumps(data))
    binding['sha256'] = sha(path.read_bytes())
    binding['size_bytes'] = path.stat().st_size
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


@pytest.mark.parametrize('field', ['source_sha256', 'source_relative_path'])
def test_camera_source_receipt_required(evidence, field):
    root, capture, raw = evidence
    path = root / 'samples/metadata.json'
    metadata = json.loads(path.read_text())
    metadata['cameras']['image'].pop(field)
    path.write_text(json.dumps(metadata))
    capture['sample']['metadata']['sha256'] = sha(path.read_bytes())
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


def test_replaced_original_video_cannot_reuse_old_decode_witness(evidence):
    root, capture, raw = evidence
    binding = next(b for b in capture['bindings'] if b['path'].startswith('videos/') and 'wrist' not in b['path'])
    path = root / binding['retained_path']
    path.write_bytes(b'changed original video; old decode remains')
    binding['sha256'] = sha(path.read_bytes())
    binding['size_bytes'] = path.stat().st_size
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'


def test_core_rechecks_original_directory_after_same_fd_read(evidence, tmp_path, monkeypatch):
    root, capture, _ = evidence
    binding = capture['bindings'][-1]
    path = root / binding['retained_path']
    directory = path.parent
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / path.name).write_bytes(path.read_bytes())
    original_fstat = os.fstat
    calls = 0

    def replace_parent_on_final_fstat(fd):
        nonlocal calls
        current = Path(f'/proc/self/fd/{fd}').resolve()
        if current == path:
            calls += 1
            if calls == 2:
                directory.rename(directory.with_name('retired-stream'))
                directory.symlink_to(outside, target_is_directory=True)
        return original_fstat(fd)

    monkeypatch.setattr(os, 'fstat', replace_parent_on_final_fstat)
    with pytest.raises(InputError):
        audit_native_observation_source(root)
