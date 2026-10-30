"""Prospective source-observer intake controls, generated data only.

No native runtime, processor call, tokenizer loading or numerical oracle is used.
The fixed source-worker validation API must remain standard-library only.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from policy_doctor import native_intake as intake
from policy_doctor.artifacts import InputError

TOKENIZER = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def state(path, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({key: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, 'little') + header + b'\x00\x00\x00\x00')


@pytest.fixture
def source_model(tmp_path):
    root, workspace = tmp_path / 'model', tmp_path / 'workspace'
    root.mkdir()
    workspace.mkdir()
    features = {'observation.state': {'type': 'STATE', 'shape': [8]}}
    put(root / 'config.json', {'type': 'smolvla', 'input_features': features,
        'output_features': {'action': {'type': 'ACTION', 'shape': [7]}}, 'device': 'cuda',
        'vlm_model_name': TOKENIZER, 'pretrained_path': 'lerobot/smolvla_base',
        'normalization_mapping': {'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD', 'VISUAL': 'IDENTITY'},
        'tokenizer_max_length': 48})
    put(root / 'policy_preprocessor.json', {'name': 'policy_preprocessor', 'steps': [
        {'registry_name': 'to_batch_processor', 'config': {}},
        {'registry_name': 'smolvla_new_line_processor', 'config': {}},
        {'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': TOKENIZER,
            'max_length': 48, 'task_key': 'task', 'padding_side': 'right', 'padding': 'max_length', 'truncation': True}},
        {'registry_name': 'normalizer_processor', 'config': {'features': features, 'norm_map': {'STATE': 'MEAN_STD'}},
         'state_file': 'pre.safetensors'}]})
    put(root / 'policy_postprocessor.json', {'name': 'policy_postprocessor', 'steps': [
        {'registry_name': 'unnormalizer_processor', 'config': {}, 'state_file': 'post.safetensors'},
        {'registry_name': 'device_processor', 'config': {'device': 'cpu'}}]})
    state(root / 'pre.safetensors', 'observation.state.mean')
    state(root / 'post.safetensors', 'action.mean')
    (root / 'LICENSE').write_bytes(b'Generated local notice; no licensing inference\r\n')
    return root, workspace


def prepare(source_model, family='smolvla'):
    return intake.prepare_native_model(source_model[0], source_model[1], model_family=family)


def test_prospective_fixed_family_api_and_old_default_are_explicit():
    assert inspect.signature(intake.prepare_native_inputs).parameters['model_family'].default == 'xvla'
    assert inspect.signature(intake.prepare_native_model).parameters['model_family'].default == 'smolvla'


def test_model_only_exact_closure_and_no_dataset_planning(source_model):
    result = prepare(source_model)
    assert set(result) == {'model', 'selection', 'receipts', 'omissions'}
    assert set(result['selection']['roots']) == {'model'}
    assert set(result['selection']['staged_roots']) == {'model'}
    expected = {'config.json', 'policy_preprocessor.json', 'policy_postprocessor.json',
                'pre.safetensors', 'post.safetensors', 'LICENSE'}
    assert {r['path'] for r in result['receipts']} == expected
    for receipt in result['receipts']:
        assert receipt['root'] == 'model'
        original, staged = source_model[0] / receipt['path'], result['model'] / receipt['path']
        assert original.read_bytes() == staged.read_bytes()
        assert original.stat().st_ino != staged.stat().st_ino
        assert receipt['sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert not (source_model[1] / 'native-plan').exists()
    assert not (source_model[1] / 'input-view/dataset').exists()
    intake.verify_native_inputs(result['selection'])


def test_unknown_model_family_is_refused_before_any_copy(source_model):
    with pytest.raises(InputError, match='family'):
        prepare(source_model, family='artifact.module:callable')
    assert not list(source_model[1].iterdir())


def test_xvla_explicit_family_preserves_its_original_registry_boundaries(source_model):
    with pytest.raises(InputError, match='XVLA|xvla'):
        prepare(source_model, family='xvla')


@pytest.mark.parametrize('malformed', [[], None, {'type': 'xvla'}, {'type': 'smolvla', 'auto_map': {'processor': 'module.Callback'}}])
def test_wrong_or_executable_model_configuration_is_refused(source_model, malformed):
    put(source_model[0] / 'config.json', malformed)
    with pytest.raises(InputError):
        prepare(source_model)


@pytest.mark.parametrize('reference', ['../outside.safetensors', '/outside.safetensors', 'https://host/state.safetensors'])
def test_normalization_state_refs_never_escape_source(source_model, reference):
    path = source_model[0] / 'policy_preprocessor.json'
    value = json.loads(path.read_text())
    value['steps'][-1]['state_file'] = reference
    put(path, value)
    with pytest.raises(InputError):
        prepare(source_model)


def test_unknown_registry_never_imports_an_artifact_callable(source_model):
    put(source_model[0] / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'client_chosen_callback', 'config': {}}]})
    with pytest.raises(InputError):
        prepare(source_model)


@pytest.mark.parametrize('where', ['model_root', 'state_leaf', 'state_parent'])
def test_source_symlink_paths_are_refused(source_model, tmp_path, where):
    root, workspace = source_model
    if where == 'model_root':
        link = tmp_path / 'root-link'
        link.symlink_to(root, target_is_directory=True)
        source_model = link, workspace
    elif where == 'state_leaf':
        source = root / 'pre.safetensors'
        destination = tmp_path / 'external-state.safetensors'
        destination.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(destination)
    else:
        target = tmp_path / 'external-state'
        target.mkdir()
        (root / 'pre.safetensors').rename(target / 'pre.safetensors')
        (root / 'state').symlink_to(target, target_is_directory=True)
        path = root / 'policy_preprocessor.json'
        value = json.loads(path.read_text())
        value['steps'][-1]['state_file'] = 'state/pre.safetensors'
        put(path, value)
    with pytest.raises(InputError):
        prepare(source_model)


def test_unrelated_weight_and_serialized_source_are_not_opened(source_model, monkeypatch):
    root, _ = source_model
    with (root / 'model.safetensors').open('wb') as stream:
        stream.truncate(3 * 1024**3)
    (root / 'not-selected.json').write_text('invalid unrelated JSON')
    actual = os.open
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in {'model.safetensors', 'not-selected.json', 'smolvla_base'}
        return actual(path, *args, **kwargs)
    monkeypatch.setattr(intake.os, 'open', guarded)
    result = prepare(source_model)
    assert len(result['receipts']) == 6
    assert json.loads((result['model'] / 'config.json').read_text())['pretrained_path'] == 'lerobot/smolvla_base'
    assert any('counts and hashes are unknown' in value for value in result['omissions'])


@pytest.mark.parametrize('change', ['config', 'state', 'notice', 'new_notice', 'staged'])
def test_model_only_reverification_retains_source_and_notice_binding(source_model, change):
    result = prepare(source_model)
    root = source_model[0]
    if change == 'config':
        path = root / 'config.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif change == 'state':
        path = root / 'pre.safetensors'
        path.touch()
    elif change == 'notice':
        (root / 'LICENSE').write_bytes(b'different notice')
    elif change == 'new_notice':
        (root / 'README.md').write_bytes(b'new local notice')
    else:
        (result['model'] / 'policy_preprocessor.json').write_text('{}')
    with pytest.raises(InputError, match='changed|membership'):
        intake.verify_native_inputs(result['selection'])


def test_small_normalizer_state_is_required_exact_bytes_not_rebuilt_statistics(source_model):
    result = prepare(source_model)
    original = source_model[0] / 'pre.safetensors'
    staged = result['model'] / 'pre.safetensors'
    assert original.read_bytes() == staged.read_bytes()
    assert any(r['path'] == 'pre.safetensors' and r['role'] == 'processor_state' for r in result['receipts'])


def local_tokenizer(source_model):
    root, _ = source_model
    folder = root / 'assets/tokenizer'
    put(folder / 'vocab.json', {'auto_map': 0, '_target_': 1, '__import__': 2, 'word': 3})
    put(folder / 'tokenizer.json', {'version': '1.0', 'model': {
        'type': 'BPE', 'vocab': {'auto_map': 0, '_target_': 1, '__import__': 2, 'word': 3}, 'merges': []}})
    put(folder / 'tokenizer_config.json', {'tokenizer_class': 'GPT2Tokenizer'})
    (folder / 'chat_template.jinja').write_text('{{ messages }}')
    pipeline = root / 'policy_preprocessor.json'
    value = json.loads(pipeline.read_text())
    step = next(step for step in value['steps'] if step['registry_name'] == 'tokenizer_processor')
    step['config']['tokenizer_name'] = 'assets/tokenizer'
    step['artifacts'] = {'tokenizer_name': 'assets/tokenizer'}
    put(pipeline, value)
    return folder


def test_local_vocabulary_token_names_are_data_not_import_hooks(source_model):
    folder = local_tokenizer(source_model)
    result = prepare(source_model)
    for path in folder.iterdir():
        assert (result['model'] / 'assets/tokenizer' / path.name).read_bytes() == path.read_bytes()
    assert any(r['path'] == 'assets/tokenizer/vocab.json' and r['role'] == 'tokenizer_asset' for r in result['receipts'])
    (folder / 'added_tokens.json').write_text('{}')
    with pytest.raises(InputError, match='tokenizer.*membership'):
        intake.verify_native_inputs(result['selection'])


def test_real_local_tokenizer_configuration_hook_is_still_refused(source_model):
    folder = local_tokenizer(source_model)
    put(folder / 'tokenizer_config.json', {'tokenizer_class': 'GPT2Tokenizer', 'auto_map': {'AutoTokenizer': 'client.module'}})
    with pytest.raises(InputError):
        prepare(source_model)


def test_unapproved_tokenizer_repo_never_becomes_native_authority(source_model):
    pipeline = source_model[0] / 'policy_preprocessor.json'
    value = json.loads(pipeline.read_text())
    step = next(step for step in value['steps'] if step['registry_name'] == 'tokenizer_processor')
    step['config']['tokenizer_name'] = 'client/arbitrary_repository'
    put(pipeline, value)
    with pytest.raises(InputError):
        prepare(source_model)


def test_intake_uses_no_native_or_model_imports(source_model, monkeypatch):
    import builtins

    actual = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'transformers', 'lerobot', 'tokenizers', 'pyarrow'}, 'Native dependency imported during source-only intake'
        return actual(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    prepare(source_model)


def test_old_xvla_registry_remains_unchanged():
    from policy_doctor.native_lerobot_worker import STEP_FIELDS

    assert 'smolvla_new_line_processor' not in STEP_FIELDS


def test_model_only_xvla_uses_existing_validator_without_smol_steps(source_model):
    root, _ = source_model
    config = json.loads((root / 'config.json').read_text())
    config['type'] = 'xvla'
    config['tokenizer_name'] = 'facebook/bart-large'
    put(root / 'config.json', config)
    pipeline = json.loads((root / 'policy_preprocessor.json').read_text())
    pipeline['steps'] = [step for step in pipeline['steps'] if step['registry_name'] != 'smolvla_new_line_processor']
    next(step for step in pipeline['steps'] if step['registry_name'] == 'tokenizer_processor')['config']['tokenizer_name'] = 'facebook/bart-large'
    put(root / 'policy_preprocessor.json', pipeline)
    result = prepare(source_model, family='xvla')
    assert result['selection']['model_family'] == 'xvla'
    intake.verify_native_inputs(result['selection'])


@pytest.mark.parametrize('malformed', [[], {}, True, 34])
def test_serialized_source_reference_requires_literal_string_or_null(source_model, malformed):
    path = source_model[0] / 'config.json'
    config = json.loads(path.read_text())
    config['pretrained_path'] = malformed
    put(path, config)
    with pytest.raises(InputError):
        prepare(source_model)


def test_smollvla_family_reuses_exact_native_dataset_plan_intake(source_model):
    root, workspace = source_model
    dataset = root.parent / 'dataset'
    info = {'total_tasks': 0, 'total_episodes': 1,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {'observation.state': {'dtype': 'float32', 'shape': [8]}}}
    put(dataset / 'meta/info.json', info)
    episode = dataset / 'meta/episodes/chunk-000/file-000.parquet'
    episode.parent.mkdir(parents=True)
    episode.write_bytes(b'generated unit metadata only')
    payload = dataset / 'data/chunk-002/file-007.parquet'
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b'generated unit payload only')
    calls = []
    def invoke(script, runtime, request, scratch):
        calls.append(script.name)
        if script.name == 'native_parquet_guard.py':
            return {'operation': 'guard_parquet', 'execution_status': 'complete',
                    'decoded_bound_bytes': len(request['paths']) * 8,
                    'files': [{'path': path, 'size_bytes': Path(path).stat().st_size, 'decoded_bound_bytes': 8}
                              for path in request['paths']]}
        assert script.name == 'native_lerobot_plan_worker.py'
        return {'provider': 'lerobot', 'operation': 'plan', 'execution_status': 'complete',
                'metadata_files': ['meta/info.json', 'meta/episodes/chunk-000/file-000.parquet'],
                'files': [{'path': 'data/chunk-002/file-007.parquet', 'role': 'native_data'}],
                'samples': [{'episode_index': 0, 'frame_index': 0, 'episode_length': 3,
                             'data_path': 'data/chunk-002/file-007.parquet', 'video_paths': {}}]}
    result = intake.prepare_native_inputs('lerobot', root, dataset, [{'episode_index': 0, 'frame_index': 0}],
        workspace, {'source_root': '/operator/native-source', 'python': '/operator/python'}, invoke,
        model_family='smolvla')
    assert result['selection']['model_family'] == 'smolvla'
    assert calls == ['native_parquet_guard.py', 'native_lerobot_plan_worker.py', 'native_parquet_guard.py']
    assert (result['dataset'] / 'data/chunk-002/file-007.parquet').read_bytes() == payload.read_bytes()
    assert (result['model'] / 'pre.safetensors').read_bytes() == (root / 'pre.safetensors').read_bytes()
    intake.verify_native_inputs(result['selection'])
