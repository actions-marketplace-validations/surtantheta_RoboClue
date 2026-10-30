"""Boundary tests for the standalone worker. Generated JSON is not native evidence."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).parents[1] / 'src/policy_doctor/native_lerobot_worker.py'
SPEC = importlib.util.spec_from_file_location('isolated_native_lerobot_worker', WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


@pytest.fixture(autouse=True)
def restore_standalone_process_settings_after_unit_calls():
    # Production calls are isolated processes; direct boundary tests must restore globals.
    environment, path, tempdir = dict(os.environ), list(sys.path), tempfile.tempdir
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(environment)
        sys.path[:] = path
        tempfile.tempdir = tempdir


def pipeline(tmp_path, entry=None):
    root = tmp_path / 'model'
    root.mkdir()
    value = {'name': 'policy_preprocessor', 'steps': [entry or {
        'registry_name': 'to_batch_processor', 'config': {}}]}
    (root / 'policy_preprocessor.json').write_text(json.dumps(value))
    return root


def request(tmp_path, **overrides):
    paths = {}
    for key in ['source_root', 'model', 'dataset']:
        path = tmp_path / key
        path.mkdir(exist_ok=True)
        paths[key] = str(path)
    return {'operation': 'probe', **paths, 'output': str(tmp_path / 'output'),
            'samples': [{'episode_index': 2, 'frame_index': 0}], **overrides}


def test_standalone_import_does_not_load_native_or_product_modules():
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not any(x == 'torch' or x == 'numpy' or x.startswith(('lerobot', 'policy_doctor')) for x in sys.modules)"
    completed = subprocess.run([sys.executable, '-I', '-c', code, str(WORKER_PATH)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    tree = ast.parse(WORKER_PATH.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert not any(name.startswith(('torch', 'numpy', 'lerobot', 'policy_doctor')) for name in names)


@pytest.mark.parametrize('relative', ['../escape', '/absolute', 'x/../escape', 'x/./a', r'x\a', 'C:/escape', 'x//a', ''])
def test_relative_paths_reject_traversal_and_platform_ambiguity(relative):
    with pytest.raises(ValueError):
        worker._relative(relative)


def test_symlinked_artifact_ancestor_is_refused(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'state.safetensors').write_bytes(b'not executed')
    (root / 'alias').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        worker._within(root, 'alias/state.safetensors')


@pytest.mark.parametrize('entry', [
    {'class': 'artifact.evil.Run', 'config': {}},
    {'registry_name': 'to_batch_processor', 'class': 'artifact.evil.Run', 'config': {}},
    {'registry_name': 'not_registered', 'config': {}},
    {'registry_name': 'to_batch_processor', 'config': {'callback': 'artifact.run'}},
    {'registry_name': 'normalizer_processor', 'config': {'stats': {'callable': 'artifact.run'}}},
    {'registry_name': 'to_batch_processor', 'config': {}, 'state_file': '../outside.safetensors'},
    {'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': 'unknown/repo'}},
    {'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': 'facebook/bart-large', 'max_length': 100_000}},
    {'registry_name': 'device_processor', 'config': {'device': 'cuda:99'}},
    {'registry_name': 'to_batch_processor', 'config': {}, 'artifacts': {'callback': 'evil'}},
])
def test_artifact_imports_callbacks_unknown_steps_and_unsafe_constructor_fields_are_refused(tmp_path, entry):
    root = pipeline(tmp_path, entry)
    with pytest.raises(ValueError):
        worker._validate_pipeline(root, 'policy_preprocessor.json')


def test_valid_registry_scaffold_and_original_cuda_declaration_allowed_only_with_probe_cpu_override(tmp_path):
    root = pipeline(tmp_path, {'registry_name': 'device_processor', 'config': {'device': 'cuda'}})
    worker._validate_pipeline(root, 'policy_preprocessor.json')
    with pytest.raises(ValueError, match='CPU only'):
        worker._validate_pipeline(root, 'policy_preprocessor.json', replay=True)


def test_replay_requires_self_contained_tokenizer_not_hidden_cache(tmp_path):
    root = pipeline(tmp_path, {'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': 'facebook/bart-large'}})
    worker._validate_pipeline(root, 'policy_preprocessor.json')
    with pytest.raises(ValueError, match='exported local tokenizer'):
        worker._validate_pipeline(root, 'policy_preprocessor.json', replay=True)


def test_tokenizer_artifact_rejects_remote_code_and_unselected_python(tmp_path):
    root = pipeline(tmp_path, {'registry_name': 'tokenizer_processor',
        'config': {'tokenizer_name': 'tokenizer'}, 'artifacts': {'tokenizer_name': 'tokenizer'}})
    tok = root / 'tokenizer'
    tok.mkdir()
    (tok / 'tokenizer_config.json').write_text(json.dumps({'auto_map': {'AutoTokenizer': 'evil.Code'}}))
    with pytest.raises(ValueError, match='Executable'):
        worker._validate_pipeline(root, 'policy_preprocessor.json', replay=True)
    (tok / 'tokenizer_config.json').write_text(json.dumps({'tokenizer_class': 'RobertaTokenizer'}))
    worker._validate_pipeline(root, 'policy_preprocessor.json', replay=True)
    (tok / 'evil.py').write_text('raise AssertionError("must never run")')
    with pytest.raises(ValueError, match='artifact file'):
        worker._validate_pipeline(root, 'policy_preprocessor.json', replay=True)


@pytest.mark.parametrize('samples', [[], [{'episode_index': True, 'frame_index': 0}],
    [{'episode_index': 0, 'frame_index': -1}], [{'episode_index': 0, 'frame_index': 0}] * 2,
    [{'episode_index': 0, 'frame_index': 0}] * 3,
    [{'episode_index': 0, 'frame_index': 0, 'callback': 'evil'}]])
def test_sample_request_identity_and_size_boundaries(tmp_path, samples):
    with pytest.raises(ValueError):
        worker._validate_request(request(tmp_path, samples=samples))


def test_rejected_output_never_mutates_existing_or_input_directory(tmp_path):
    req = request(tmp_path)
    destination = Path(req['model'])
    (destination / 'original.json').write_text('{}')
    req['output'] = str(destination)
    before = sorted(p.name for p in destination.iterdir())
    result = worker.run(req)
    assert result['execution_status'] == 'error'
    assert sorted(p.name for p in destination.iterdir()) == before
    assert not (destination / 'native_error.txt').exists()
    req['output'] = str(destination / 'nested')
    assert worker.run(req)['execution_status'] == 'error'
    assert not (destination / 'nested').exists()


def test_duplicate_nonfinite_and_oversized_json_are_refused(tmp_path):
    path = tmp_path / 'config.json'
    for raw in ['{"a":1,"a":2}', '{"a":NaN}']:
        path.write_text(raw)
        with pytest.raises(ValueError):
            worker._read_json(path)
    path.write_bytes(b' ' * 11)
    with pytest.raises(ValueError, match='oversized'):
        worker._read_json(path, maximum=10)


@pytest.mark.parametrize('template', ['../data/{file_index}.parquet', 'data/{chunk_index.__class__}.parquet',
    'data/{file_index!r}.parquet', '/data/{file_index}.parquet'])
def test_native_dataset_templates_cannot_escape_or_access_arbitrary_attributes(tmp_path, template):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({'data_path': template, 'video_path': None}))
    with pytest.raises(ValueError):
        worker._validate_dataset_info(tmp_path)


def test_hash_is_streamed_and_bounded(tmp_path):
    path = tmp_path / 'large.bin'
    path.write_bytes(b'x' * 25)
    with pytest.raises(ValueError, match='oversized'):
        worker._hash(path, 24)
    assert worker._hash(path, 25) == __import__('hashlib').sha256(b'x' * 25).hexdigest()


def test_projection_compares_json_representation_without_weakening_numeric_or_shape_values():
    native = {'shape': (1, 3, 256, 256), 'sha256': 'actual bytes', 'preview': [1.0]}
    retained = json.loads(worker._canonical(native))
    assert worker._canonical(native) == worker._canonical(retained)
    retained['shape'][-1] = 224
    assert worker._canonical(native) != worker._canonical(retained)
    retained['shape'][-1] = 256
    retained['sha256'] = 'different bytes'
    assert worker._canonical(native) != worker._canonical(retained)


def test_probe_blocks_dynamic_artifact_constructor_before_any_native_import(tmp_path):
    req = request(tmp_path)
    source = Path(req['source_root']) / 'lerobot'
    source.mkdir()
    marker = tmp_path / 'code_ran'
    (source / '__init__.py').write_text(f'open({str(marker)!r}, "w").write("must never execute")')
    model = Path(req['model'])
    (model / 'config.json').write_text(json.dumps({'type': 'xvla', 'input_features': {}, 'output_features': {}}))
    (model / 'policy_preprocessor.json').write_text(json.dumps({'steps': [{'class': 'artifact.Run', 'config': {}}]}))
    result = worker.run(req)
    assert result['execution_status'] == 'error'
    assert 'registry' in result['native_execution_error']['message']
    assert not marker.exists()
    assert (Path(req['output']) / 'native_error.txt').exists()


def test_replay_refuses_corrupted_export_before_native_import(tmp_path):
    req = request(tmp_path)
    source = Path(req['source_root']) / 'lerobot'
    source.mkdir()
    marker = tmp_path / 'code_ran'
    (source / '__init__.py').write_text(f'open({str(marker)!r}, "w").write("must never execute")')
    capsule = tmp_path / 'capsule'
    export = capsule / 'exports/saved'
    export.mkdir(parents=True)
    (export / 'pipeline.json').write_text('{}')
    manifest = {'schema': worker.SCHEMA, 'provider': 'lerobot', 'samples': [{}], 'exports': [
        {'route': 'saved', 'path': 'exports/saved', 'config_filename': 'pipeline.json',
         'files': [{'path': 'pipeline.json', 'sha256': 'bad digest'}]}]}
    (capsule / 'worker_manifest.json').write_text(json.dumps(manifest))
    req = {'operation': 'replay', 'source_root': req['source_root'],
           'capsule_root': str(capsule), 'output': req['output']}
    result = worker.run(req)
    assert result['execution_status'] == 'error'
    assert 'digest mismatch' in result['native_execution_error']['message']
    assert not marker.exists()


def test_main_invalid_model_writes_bounded_result_and_tiny_stdout_only(tmp_path):
    req = request(tmp_path)
    source = Path(req['source_root']) / 'lerobot'
    source.mkdir()
    (source / '__init__.py').write_text('raise AssertionError("not a native claim")')
    (Path(req['model']) / 'config.json').write_text(json.dumps({'type': 'unknown'}))
    completed = subprocess.run([sys.executable, str(WORKER_PATH)], input=json.dumps(req),
                               capture_output=True, text=True)
    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert len(completed.stdout) < 1024
    assert receipt['execution_status'] == 'error'
    path = Path(req['output']) / 'result.json'
    result = json.loads(path.read_text())
    assert path.stat().st_size < worker.MAX_RESPONSE
    assert result['native_execution_error']['type'] == 'ValueError'


@pytest.mark.parametrize('features', [
    {'observation.images.escape/../../other': {'dtype': 'video'}},
    {'observation.images.camera': {'dtype': 'image'}},
])
def test_dataset_camera_keys_cannot_escape_native_video_templates_or_load_external_image_paths(tmp_path, features):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4', 'features': features}))
    with pytest.raises(ValueError):
        worker._validate_dataset_info(tmp_path)


def test_normalization_constructor_key_list_type_is_checked_before_native_loader(tmp_path):
    root = pipeline(tmp_path, {'registry_name': 'normalizer_processor',
        'config': {'normalize_observation_keys': {'callback': 'unsupported'}}})
    with pytest.raises(ValueError, match='observation key list'):
        worker._validate_pipeline(root, 'policy_preprocessor.json')
