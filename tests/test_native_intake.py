"""Ordinary-folder dependency selection unit controls; no native execution.

The fixed planner/guard invoke fixture returns generated plans. It does not
qualify native metadata planning, Parquet decoding or original numerical parity.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from policy_doctor import native_intake as intake
from policy_doctor.artifacts import InputError


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def folders(tmp_path):
    model, dataset, workspace = (tmp_path / name for name in ('model', 'dataset', 'workspace'))
    model.mkdir()
    dataset.mkdir()
    workspace.mkdir()
    put(model / 'config.json', {'type': 'xvla', 'input_features': {}, 'output_features': {}})
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        put(model / name, {'steps': [{'registry_name': 'to_batch_processor', 'config': {}}]})
    camera = 'observation.images.alternate_camera'
    put(dataset / 'meta/info.json', {
        'codebase_version': 'v3.0', 'total_tasks': 1, 'total_episodes': 10,
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'features': {'observation.state': {'dtype': 'float32', 'shape': [2]},
                     camera: {'dtype': 'video', 'shape': [8, 8, 3]}}})
    metadata = ['meta/info.json', 'meta/tasks.parquet', 'meta/episodes/chunk-000/file-000.parquet',
                'meta/episodes/chunk-003/file-004.parquet']
    for relative in metadata[1:]:
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('generated unit metadata: ' + relative).encode())
    routes = {
        7: {'data_path': 'data/chunk-002/file-017.parquet',
            'video_paths': {camera: f'videos/{camera}/chunk-006/file-029.mp4'}},
        9: {'data_path': 'data/chunk-004/file-023.parquet',
            'video_paths': {camera: f'videos/{camera}/chunk-009/file-031.mp4'}},
    }
    for route in routes.values():
        for relative in [route['data_path'], *route['video_paths'].values()]:
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('generated unit selected payload: ' + relative).encode())
    (model / 'README.md').write_bytes(b'Exact unit model notice bytes\r\n')
    (dataset / 'LICENSE').write_bytes(b'Exact unit dataset notice; no licence inference\n')
    calls = []
    def invoke(script, runtime, request, execution_workspace):
        calls.append({'script': script.name, 'runtime': copy.deepcopy(runtime), 'request': copy.deepcopy(request)})
        if script.name == 'native_parquet_guard.py':
            assert request['operation'] == 'guard_parquet'
            assert all(Path(path).is_file() and Path(path).is_relative_to(workspace / 'input-view') for path in request['paths'])
            return {'operation': 'guard_parquet', 'execution_status': 'complete',
                    'decoded_bound_bytes': 8 * len(request['paths']),
                    'files': [{'path': path, 'size_bytes': Path(path).stat().st_size, 'decoded_bound_bytes': 8}
                              for path in request['paths']]}
        assert script.name == 'native_lerobot_plan_worker.py'
        assert Path(request['dataset']) == workspace / 'input-view/dataset'
        assert not (Path(request['dataset']) / 'data').exists()
        samples, files = [], {}
        for sample in request['samples']:
            route = copy.deepcopy(routes[sample['episode_index']])
            samples.append(dict(sample, episode_length=20, **route))
            files[route['data_path']] = 'native_data'
            files.update({path: 'native_video' for path in route['video_paths'].values()})
        return {'provider': 'lerobot', 'operation': 'plan', 'execution_status': 'complete',
                'metadata_files': sorted(metadata), 'samples': samples,
                'files': [{'path': path, 'role': role} for path, role in files.items()]}
    return {'model': model, 'dataset': dataset, 'workspace': workspace, 'metadata': metadata,
            'routes': routes, 'calls': calls, 'invoke': invoke,
            'samples': [{'episode_index': 7, 'frame_index': 2}],
            'runtime': {'python': '/operator/python', 'source_root': '/operator/native-source'}}


def prepare(folders, *, invoke=None, samples=None, provider='lerobot'):
    return intake.prepare_native_inputs(provider, folders['model'], folders['dataset'],
        samples or folders['samples'], folders['workspace'], folders['runtime'], invoke or folders['invoke'])


def test_exact_seed_dependency_view_and_native_planned_alternate_paths(folders):
    result = prepare(folders)
    assert result['model'] == folders['workspace'] / 'input-view/model'
    assert result['dataset'] == folders['workspace'] / 'input-view/dataset'
    receipts = {(r['root'], r['path']): r for r in result['receipts']}
    selected = folders['routes'][7]
    assert ('dataset', selected['data_path']) in receipts
    assert all(('dataset', path) in receipts for path in selected['video_paths'].values())
    assert ('dataset', folders['routes'][9]['data_path']) not in receipts
    for (root, relative), receipt in receipts.items():
        original = folders[root] / relative
        copied = result[root] / relative
        assert copied.read_bytes() == original.read_bytes()
        assert copied.stat().st_ino != original.stat().st_ino
        assert receipt['sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
        assert receipt['size_bytes'] == original.stat().st_size
    assert [c['script'] for c in folders['calls']] == ['native_parquet_guard.py', 'native_lerobot_plan_worker.py', 'native_parquet_guard.py']
    assert sorted(folders['calls'][0]['request']['paths']) == sorted(str(result['dataset'] / p) for p in folders['metadata'] if p.endswith('.parquet'))
    assert any('other episodes, rows or frames' in value for value in result['omissions'])
    assert any('counts and hashes are unknown' in value for value in result['omissions'])
    intake.verify_native_inputs(result['selection'])


def test_ordinary_weight_and_unrelated_contents_are_never_opened_or_walked(folders, monkeypatch):
    weight = folders['model'] / 'model.safetensors'
    with weight.open('wb') as stream:
        stream.truncate(3 * 1024**3)  # sparse metadata-only generated scale control
    (folders['model'] / 'unrelated.json').write_text('not JSON; must remain unopened')
    for index in range(64):
        (folders['model'] / f'irrelevant-{index}').mkdir()
    ignored = folders['dataset'] / 'data/unselected.parquet'
    ignored.write_bytes(b'unselected encoded payload')
    actual_open, actual_scan = os.open, os.scandir
    forbidden = {'model.safetensors', 'unrelated.json', 'unselected.parquet'}
    def guarded_open(path, *args, **kwargs):
        assert Path(path).name not in forbidden, 'Unrelated payload was opened'
        return actual_open(path, *args, **kwargs)
    def guarded_scan(path):
        if not isinstance(path, int):
            assert Path(path) not in {folders['model'], folders['dataset'], folders['dataset'] / 'data', folders['dataset'] / 'videos'}
        return actual_scan(path)
    monkeypatch.setattr(intake.os, 'open', guarded_open)
    monkeypatch.setattr(intake.os, 'scandir', guarded_scan)
    result = prepare(folders)
    assert all(r['path'] not in forbidden for r in result['receipts'])
    weight.touch()  # unselected source changes are outside the integrity claim
    intake.verify_native_inputs(result['selection'])


def test_reversed_real_selector_order_and_distinct_files_are_preserved(folders):
    samples = [{'episode_index': 9, 'frame_index': 3}, {'episode_index': 7, 'frame_index': 2}]
    result = prepare(folders, samples=samples)
    assert [{k: row[k] for k in ('episode_index', 'frame_index')} for row in result['selection']['native_plan']['samples']] == samples
    selected = {r['path'] for r in result['receipts'] if r['role'] == 'native_data'}
    assert selected == {folders['routes'][7]['data_path'], folders['routes'][9]['data_path']}


@pytest.mark.parametrize('which', ['source_metadata', 'source_data', 'source_inode', 'staged_bytes'])
def test_selected_mutation_or_identity_change_refuses_capture(folders, which):
    result = prepare(folders)
    if which == 'source_metadata':
        path = folders['model'] / 'config.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif which == 'source_data':
        path = folders['dataset'] / folders['routes'][7]['data_path']
        path.write_bytes(path.read_bytes() + b'changed')
    elif which == 'source_inode':
        path = folders['dataset'] / folders['routes'][7]['data_path']
        contents = path.read_bytes()
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(contents)
        replacement.replace(path)
    else:
        (result['model'] / 'config.json').write_text('{}')
    with pytest.raises(InputError, match='changed'):
        intake.verify_native_inputs(result['selection'])


@pytest.mark.parametrize('membership', ['add_shard', 'remove_shard', 'optional_stats', 'notice'])
def test_seed_metadata_membership_change_is_not_hidden(folders, membership):
    result = prepare(folders)
    if membership == 'add_shard':
        (folders['dataset'] / 'meta/episodes/chunk-000/new.parquet').write_bytes(b'new metadata shard')
    elif membership == 'remove_shard':
        (folders['dataset'] / folders['metadata'][-1]).unlink()
    elif membership == 'optional_stats':
        put(folders['dataset'] / 'meta/stats.json', {})
    else:
        (folders['model'] / 'LICENSE').write_text('new local notice')
    with pytest.raises(InputError, match='membership'):
        intake.verify_native_inputs(result['selection'])


def test_source_swap_during_copy_is_rejected(folders, monkeypatch):
    actual = os.read
    changed = False
    source = folders['model'] / 'config.json'
    def changing_read(fd, count):
        nonlocal changed
        block = actual(fd, count)
        if not changed and os.readlink(f'/proc/self/fd/{fd}') == str(source):
            changed = True
            source.write_bytes(source.read_bytes() + b' ')
        return block
    monkeypatch.setattr(intake.os, 'read', changing_read)
    with pytest.raises(InputError, match='changed during exact-byte copy'):
        prepare(folders)
    assert not folders['calls']


@pytest.mark.parametrize('unsafe', ['../model.safetensors', '/outside/state.safetensors', 'https://host/state.safetensors'])
def test_unsafe_processor_state_reference_never_reaches_native(folders, unsafe):
    put(folders['model'] / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'normalizer_processor', 'config': {}, 'state_file': unsafe}]})
    with pytest.raises(InputError):
        prepare(folders)
    assert not folders['calls']


def test_unknown_registry_and_executable_config_are_refused_before_plan(folders):
    put(folders['model'] / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'client_chosen_callable', 'config': {}}]})
    with pytest.raises(InputError, match='registry'):
        prepare(folders)
    assert not folders['calls']


@pytest.mark.parametrize('link_location', ['source_root', 'state_ancestor', 'metadata_leaf', 'metadata_directory'])
def test_source_symlinks_are_refused_without_following(folders, tmp_path, link_location):
    if link_location == 'source_root':
        link = tmp_path / 'model-link'
        link.symlink_to(folders['model'], target_is_directory=True)
        folders['model'] = link
    elif link_location == 'state_ancestor':
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        (folders['model'] / 'state').symlink_to(elsewhere, target_is_directory=True)
        put(folders['model'] / 'policy_preprocessor.json', {'steps': [
            {'registry_name': 'normalizer_processor', 'config': {}, 'state_file': 'state/normalizer.safetensors'}]})
    elif link_location == 'metadata_leaf':
        path = folders['dataset'] / 'meta/info.json'
        contents = path.read_bytes()
        outside = tmp_path / 'outside-info.json'
        outside.write_bytes(contents)
        path.unlink()
        path.symlink_to(outside)
    else:
        (folders['dataset'] / 'meta/episodes/chunk-link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(InputError):
        prepare(folders)
    assert not folders['calls']


def test_complete_bounded_metadata_required_without_synthetic_stats(folders):
    (folders['dataset'] / 'meta/tasks.parquet').unlink()
    with pytest.raises(InputError, match='tasks metadata'):
        prepare(folders)
    assert not folders['calls']


def test_metadata_shard_count_refused_before_native_loader(folders):
    chunk = folders['dataset'] / 'meta/episodes/chunk-000'
    for index in range(128):
        (chunk / f'additional-{index:03d}.parquet').write_bytes(b'generated metadata')
    with pytest.raises(InputError, match='128 shards'):
        prepare(folders)
    assert not folders['calls']


@pytest.mark.parametrize('failure', ['incomplete', 'traversal', 'wrong_metadata', 'extra_payload', 'wrong_camera', 'wrong_identity', 'boolean_identity'])
def test_native_plan_is_inert_bounded_and_independently_checked(folders, failure):
    def bad(script, runtime, request, workspace):
        result = folders['invoke'](script, runtime, request, workspace)
        if script.name != 'native_lerobot_plan_worker.py':
            return result
        if failure == 'incomplete':
            result['execution_status'] = 'error'
        elif failure == 'traversal':
            result['files'][0]['path'] = '../unrelated.parquet'
        elif failure == 'wrong_metadata':
            result['metadata_files'].pop()
        elif failure == 'extra_payload':
            result['files'].append({'path': 'data/chunk-999/file-999.parquet', 'role': 'native_data'})
        elif failure == 'wrong_camera':
            result['samples'][0]['video_paths'] = {'invented_camera': 'videos/invented_camera/chunk-000/file-000.mp4'}
        elif failure == 'boolean_identity':
            result['samples'][0]['frame_index'] = True
        else:
            result['samples'][0]['episode_index'] = 9
        return result
    with pytest.raises(InputError):
        prepare(folders, invoke=bad, samples=[{'episode_index': 7, 'frame_index': 1}] if failure == 'boolean_identity' else None)
    assert not (folders['workspace'] / 'input-view/dataset/data').exists()


def test_metadata_allocation_guard_precedes_native_planner(folders):
    def refused(script, runtime, request, workspace):
        assert script.name == 'native_parquet_guard.py'
        return {'execution_status': 'error'}
    with pytest.raises(InputError, match='allocation preflight'):
        prepare(folders, invoke=refused)
    assert not folders['calls']


def test_missing_selected_payload_fails_without_substitution(folders):
    (folders['dataset'] / folders['routes'][7]['data_path']).unlink()
    with pytest.raises(InputError):
        prepare(folders)
    assert not any(r['request'].get('operation') == 'probe' for r in folders['calls'])


def test_notice_bytes_roles_and_oversized_notice_omission(folders, monkeypatch):
    oversized = folders['model'] / 'LICENSE'
    with oversized.open('wb') as stream:
        stream.truncate(intake.MAX_NOTICE + 1)
    original_open = os.open
    def guarded(path, *args, **kwargs):
        if Path(path).name == 'LICENSE' and kwargs.get('dir_fd') is not None:
            target = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}")) / str(path)
            assert target != oversized, 'Oversized notice content should remain unopened'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(intake.os, 'open', guarded)
    result = prepare(folders)
    assert not (result['model'] / 'LICENSE').exists()
    assert (result['model'] / 'README.md').read_bytes() == (folders['model'] / 'README.md').read_bytes()
    assert ('model', 'README.md', 'local_notice') in {(r['root'], r['path'], r['role']) for r in result['receipts']}
    assert any('model/LICENSE' in message and 'hash not inspected' in message for message in result['omissions'])


def _local_tokenizer(folders):
    directory = folders['model'] / 'assets/tokenizer'
    put(directory / 'tokenizer.json', {})
    put(directory / 'tokenizer_config.json', {'tokenizer_class': 'BartTokenizerFast'})
    put(folders['model'] / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'tokenizer_processor', 'config': {'tokenizer_name': 'assets/tokenizer', 'max_length': 64},
         'artifacts': {'tokenizer_name': 'assets/tokenizer'}}]})
    return directory


def test_safe_declared_tokenizer_assets_are_exact_dependencies(folders):
    directory = _local_tokenizer(folders)
    result = prepare(folders)
    assert result['selection']['tokenizer_membership'] == {'assets/tokenizer': ['tokenizer.json', 'tokenizer_config.json']}
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        assert (result['model'] / 'assets/tokenizer' / name).read_bytes() == (directory / name).read_bytes()
    (directory / 'added_tokens.json').write_text('{}')
    with pytest.raises(InputError, match='tokenizer.*membership'):
        intake.verify_native_inputs(result['selection'])


@pytest.mark.parametrize('unsafe', ['extra_file', 'nested_directory', 'code_hook'])
def test_unsafe_tokenizer_tree_refused_before_native(folders, unsafe):
    directory = _local_tokenizer(folders)
    if unsafe == 'extra_file':
        (directory / 'client_code.py').write_text('raise RuntimeError("must never execute")')
    elif unsafe == 'nested_directory':
        (directory / 'nested').mkdir()
    else:
        put(directory / 'tokenizer_config.json', {'tokenizer_class': 'BartTokenizerFast', 'auto_map': {'tokenizer': 'client.module'}})
    with pytest.raises(InputError):
        prepare(folders)
    assert not folders['calls']


def test_declared_processor_state_header_cannot_expand_allocation(folders):
    header = json.dumps({'x': {'dtype': 'F32', 'shape': [10_000_000], 'data_offsets': [0, 4]}}).encode()
    (folders['model'] / 'normalizer.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + b'1234')
    put(folders['model'] / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'normalizer_processor', 'config': {}, 'state_file': 'normalizer.safetensors'}]})
    with pytest.raises(InputError, match='allocation'):
        prepare(folders)
    assert not folders['calls']


def test_gr1_uses_original_per_episode_template_without_lerobot_assumptions(folders):
    model = folders['model']
    put(model / 'processor_config.json', {'processor_kwargs': {key: value for key, value in (
        ('use_percentiles', False), ('clip_outliers', True), ('apply_sincos_state_encoding', True), ('use_relative_action', True))}})
    put(model / 'statistics.json', {})
    put(folders['dataset'] / 'meta/info.json', {'chunks_size': 1000,
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'features': {'observation.state': {'dtype': 'float64', 'shape': [44]}}})
    put(folders['dataset'] / 'meta/modality.json', {'state': {'right_arm': {'start': 22, 'end': 29}}})
    relative = 'data/chunk-001/episode_001007.parquet'
    path = folders['dataset'] / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'generated unit GR1 original bytes')
    result = prepare(folders, provider='groot_state', samples=[{'episode_index': 1007, 'physical_row': 2}])
    assert any(r['path'] == relative for r in result['receipts'])
    assert [c['script'] for c in folders['calls']] == ['native_parquet_guard.py']
    assert not (result['model'] / 'config.json').exists()
    assert not (result['dataset'] / 'meta/tasks.parquet').exists()


def test_valid_declared_small_processor_state_is_copied_without_weight_discovery(folders):
    header = json.dumps({'observation.state.mean': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    state = folders['model'] / 'states/normalizer.safetensors'
    state.parent.mkdir()
    state.write_bytes(len(header).to_bytes(8, 'little') + header + b'\x00\x00\x00\x00')
    put(folders['model'] / 'policy_preprocessor.json', {'steps': [
        {'registry_name': 'normalizer_processor', 'config': {}, 'state_file': 'states/normalizer.safetensors'}]})
    result = prepare(folders)
    assert (result['model'] / 'states/normalizer.safetensors').read_bytes() == state.read_bytes()
    assert any(r['path'] == 'states/normalizer.safetensors' and r['role'] == 'processor_state' for r in result['receipts'])
    state.touch()
    with pytest.raises(InputError, match='changed'):
        intake.verify_native_inputs(result['selection'])


def test_oversized_selected_video_is_refused_without_content_read(folders, monkeypatch):
    selected = folders['dataset'] / next(iter(folders['routes'][7]['video_paths'].values()))
    with selected.open('wb') as stream:
        stream.truncate(intake.MAX_FILE + 1)  # sparse selected-file bound control
    actual = os.read
    def guarded_read(fd, count):
        assert os.readlink(f'/proc/self/fd/{fd}') != str(selected), 'Oversized selected payload was read'
        return actual(fd, count)
    monkeypatch.setattr(intake.os, 'read', guarded_read)
    with pytest.raises(InputError, match='byte bound'):
        prepare(folders)


@pytest.mark.parametrize('malformed', [[], 1, None, {'features': []}, {'features': {'observation.state': []}}])
def test_public_malformed_info_is_input_error_before_native(folders, malformed):
    put(folders['dataset'] / 'meta/info.json', malformed)
    with pytest.raises(InputError):
        prepare(folders)
    assert not folders['calls']


@pytest.mark.parametrize('malformed', [[], 1, None])
def test_public_gr1_nonobject_processor_config_is_input_error(folders, malformed):
    put(folders['model'] / 'processor_config.json', malformed)
    put(folders['model'] / 'statistics.json', {})
    put(folders['dataset'] / 'meta/modality.json', {})
    with pytest.raises(InputError, match='must be an object'):
        prepare(folders, provider='groot_state')
    assert not folders['calls']


@pytest.mark.parametrize('malformed', ['wrong_path', 'missing_file', 'duplicate', 'boolean_budget', 'wrong_size', 'wrong_aggregate'])
def test_guard_success_requires_exact_selected_membership_and_bounds(folders, malformed):
    def bad(script, runtime, request, workspace):
        response = folders['invoke'](script, runtime, request, workspace)
        if script.name != 'native_parquet_guard.py':
            return response
        if malformed == 'wrong_path':
            response['files'][0]['path'] = '/unrequested/file.parquet'
        elif malformed == 'missing_file':
            response['files'].pop()
        elif malformed == 'duplicate':
            response['files'][1] = response['files'][0]
        elif malformed == 'boolean_budget':
            response['files'][0]['decoded_bound_bytes'] = True
        elif malformed == 'wrong_size':
            response['files'][0]['size_bytes'] += 1
        else:
            response['decoded_bound_bytes'] += 1
        return response
    with pytest.raises(InputError, match='Parquet'):
        prepare(folders, invoke=bad)
    assert not any(call['script'] == 'native_lerobot_plan_worker.py' for call in folders['calls'])
