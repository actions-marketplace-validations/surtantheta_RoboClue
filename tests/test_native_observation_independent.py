"""Independent generated controls; no native execution or original-payload mutation."""
import json

import pytest
from test_native_observation_source import fixture, save, sha

from policy_doctor.native_observation_source import audit_native_observation_source


def _rewrite_binding(root, capture, path, mutate):
    binding = next(b for b in capture['bindings'] if b['path'] == path)
    retained = root / binding['retained_path']
    value = json.loads(retained.read_text())
    mutate(value)
    retained.write_text(json.dumps(value))
    binding['sha256'] = sha(retained.read_bytes())
    binding['size_bytes'] = retained.stat().st_size


@pytest.mark.parametrize('field', ['episode_identity', 'task_identity', 'episode_tasks', 'state_delta', 'video_delta', 'state_dtype', 'state_shape'])
def test_inconsistent_or_untyped_original_metadata_cannot_certify_pass(tmp_path, field):
    root = tmp_path / 'view'
    capture, raw = fixture(root)
    if field == 'episode_identity':
        _rewrite_binding(root, capture, 'meta/episodes.jsonl', lambda v: v.update(episode_index=False))
    elif field == 'task_identity':
        _rewrite_binding(root, capture, 'meta/tasks.jsonl', lambda v: v.update(task_index=False))
    elif field == 'episode_tasks':
        _rewrite_binding(root, capture, 'meta/episodes.jsonl', lambda v: v.update(tasks='pick object'))
    elif field in {'state_delta', 'video_delta'}:
        group = 'state' if field == 'state_delta' else 'video'
        _rewrite_binding(root, capture, 'processor_config.json',
                         lambda v: v['processor_kwargs']['modality_configs']['libero_panda'][group].update(delta_indices=[False]))
    elif field == 'state_dtype':
        _rewrite_binding(root, capture, 'meta/info.json',
                         lambda v: v['features']['observation.state'].update(dtype='string'))
    else:
        _rewrite_binding(root, capture, 'meta/info.json',
                         lambda v: v['features']['observation.state'].update(shape=[7]))
    save(root, capture, raw)
    result = audit_native_observation_source(root)
    assert result['status'] == 'unknown', (field, result)


def test_healthy_generated_control_passes(tmp_path):
    root = tmp_path / 'healthy'
    fixture(root)
    assert audit_native_observation_source(root)['status'] == 'pass'


def test_copy_rejects_original_parent_replaced_by_external_symlink(tmp_path, monkeypatch):
    """A path recheck must not follow a replaced ancestor outside its source root."""
    import os

    from policy_doctor import native_observation_processing as parent
    from policy_doctor.artifacts import InputError

    source = tmp_path / 'source'
    source.mkdir()
    nested = source / 'nested'
    nested.mkdir()
    data = b'generated original bytes'
    (nested / 'payload.bin').write_bytes(data)
    outside = tmp_path / 'outside-source-root'
    outside.mkdir()
    (outside / 'payload.bin').write_bytes(data)
    destination = tmp_path / 'destination'
    destination.mkdir()
    record = parent._record(source, 'nested/payload.bin')
    original_read = os.read
    changed = False

    def replace_parent_after_open(fd, count):
        nonlocal changed
        block = original_read(fd, count)
        if not changed:
            changed = True
            nested.rename(source / 'retired-nested')
            nested.symlink_to(outside, target_is_directory=True)
        return block

    monkeypatch.setattr(os, 'read', replace_parent_after_open)
    with pytest.raises(InputError):
        parent._copy(source, 'nested/payload.bin', destination, 'payload.bin', record)


@pytest.mark.parametrize('mutation', ['missing_source_sha', 'wrong_source_sha', 'wrong_source_path', 'changed_retained_video'])
def test_camera_decode_receipt_must_bind_retained_original_media(tmp_path, mutation):
    """Stored pixels cannot certify the source when their current video receipt differs."""
    root = tmp_path / 'media'
    capture, raw = fixture(root)
    metadata_path = root / 'samples/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    camera = metadata['cameras']['wrist_image']
    if mutation == 'missing_source_sha':
        camera.pop('source_sha256')
    elif mutation == 'wrong_source_sha':
        camera['source_sha256'] = 'b' * 64
    elif mutation == 'wrong_source_path':
        camera['source_relative_path'] = 'videos/chunk-000/observation.images.image/episode_000000.mp4'
    else:
        binding = next(b for b in capture['bindings'] if 'wrist_image' in b['path'])
        retained = root / binding['retained_path']
        retained.write_bytes(b'different generated video bytes; old decoded pixels remain')
        binding['sha256'] = sha(retained.read_bytes())
        binding['size_bytes'] = retained.stat().st_size
    metadata_path.write_text(json.dumps(metadata))
    capture['sample']['metadata']['sha256'] = sha(metadata_path.read_bytes())
    save(root, capture, raw)
    assert audit_native_observation_source(root)['status'] == 'unknown'
