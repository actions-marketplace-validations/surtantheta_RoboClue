"""Safety controls for the file/source image contract, not a native-run claim."""
import asyncio
import hashlib
import json
import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from policy_doctor import fastwam_contract as fw
from policy_doctor.artifacts import InputError
from policy_doctor.product_cli import main
from policy_doctor.product_mcp import BASIC_TOOLS, create_product_server
from policy_doctor.workflow_readiness import choose_workflow


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    cp, source = tmp_path / 'checkpoint', tmp_path / 'source'
    cp.mkdir()
    file = source / 'src/lerobot/processor/normalize_processor.py'
    file.parent.mkdir(parents=True)
    (source / 'src/lerobot/policies/fastwam').mkdir(parents=True)
    file.write_text('Never executed: bounded fake source for unit controls.\n')
    monkeypatch.setattr(fw, 'PROFILES', {'unit': {'files': {'processor/normalize_processor.py': hashlib.sha256(file.read_bytes()).hexdigest()}, 'identity': 'unit-only', 'map_lines': [1298, 1319]}})
    visual = {'observation.images.image': {'type': 'VISUAL', 'shape': [3, 8, 8]}}
    write_json(cp / 'config.json', {'type': 'fastwam', 'input_features': visual, 'output_features': {}})
    pipeline = {'steps': [
        {'registry_name': 'rename_observations_processor', 'config': {'rename_map': {}}},
        {'registry_name': 'to_batch_processor', 'config': {}},
        {'registry_name': 'device_processor', 'config': {'device': 'cuda', 'float_dtype': None}},
        {'registry_name': 'normalizer_processor', 'config': {
            'features': visual, 'norm_map': {'VISUAL': 'MEAN_STD'}, 'eps': 1e-8}, 'state_file': 'stats.safetensors'}]}
    write_json(cp / 'policy_preprocessor.json', pipeline)
    save_file({f'observation.images.image.{key}': np.full((3, 1, 1), 0.5, dtype=np.float32)
               for key in ('mean', 'std')}, cp / 'stats.safetensors')
    return cp, source, pipeline


def test_reported_map_is_not_a_mode_label_test(inputs):
    cp, source, _ = inputs
    report = fw.check_image_contract(cp, source)
    assert report['status'] == 'error'
    image = report['images']['observation.images.image']
    assert image['slope'] == [4.0] * 3 and image['offset'] == [-3.0] * 3
    assert report['native_execution'] == 'not_run'
    assert report['active_runtime'] == 'not_attested'


@pytest.mark.parametrize('healthy', ['identity', 'valid_mean_std', 'skipped_image', 'per_channel_tolerance'])
def test_healthy_image_maps(inputs, healthy):
    cp, source, pipeline = inputs
    normalizer = pipeline['steps'][-1]['config']
    if healthy == 'identity':
        normalizer['norm_map']['VISUAL'] = 'IDENTITY'
    elif healthy == 'skipped_image':
        normalizer['normalize_observation_keys'] = ['observation.state']
    else:
        mean = np.array([0, 1e-8, -1e-8] if healthy == 'per_channel_tolerance' else [0, 0, 0], dtype=np.float32)
        save_file({'observation.images.image.mean': mean,
                   'observation.images.image.std': np.ones(3, dtype=np.float32)}, cp / 'stats.safetensors')
    write_json(cp / 'policy_preprocessor.json', pipeline)
    report = fw.check_image_contract(cp, source)
    assert report['status'] == 'pass'
    assert report['images']['observation.images.image']['consistent']
    assert 'Other modalities' in report['findings'][0]['message']


@pytest.mark.parametrize('missing', ['source', 'older_code', 'processor', 'state', 'images', 'image_stats'])
def test_missing_or_unrecognized_evidence_never_passes_or_errors(inputs, missing):
    cp, source, pipeline = inputs
    if missing == 'source':
        source = None
    elif missing == 'older_code':
        (source / 'src/lerobot/processor/normalize_processor.py').write_text('Older valid implementation without internal model scaling.\n')
    elif missing == 'processor':
        (cp / 'policy_preprocessor.json').unlink()
    elif missing == 'state':
        (cp / 'stats.safetensors').unlink()
    elif missing == 'images':
        write_json(cp / 'config.json', {'type': 'fastwam', 'input_features': {}})
    elif missing == 'image_stats':
        save_file({'action.mean': np.ones(1, dtype=np.float32)}, cp / 'stats.safetensors')
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


@pytest.mark.parametrize('change', ['custom_step', 'renaming', 'double_normalizer', 'dtype', 'minmax', 'custom_stats', 'features', 'eps'])
def test_unsupported_processing_never_produces_false_error(inputs, change):
    cp, source, pipeline = inputs
    normalizer = pipeline['steps'][-1]['config']
    if change == 'custom_step':
        pipeline['steps'].insert(0, {'registry_name': 'custom_image_transform', 'config': {}})
    elif change == 'renaming':
        pipeline['steps'][0]['config']['rename_map'] = {'a': 'b'}
    elif change == 'double_normalizer':
        pipeline['steps'].append(pipeline['steps'][-1])
    elif change == 'dtype':
        pipeline['steps'][2]['config']['float_dtype'] = 'float16'
    elif change == 'minmax':
        normalizer['norm_map']['VISUAL'] = 'MIN_MAX'
    elif change == 'custom_stats':
        normalizer['stats'] = {'action': {'mean': [0]}}
    elif change == 'features':
        normalizer['features'] = {}
    else:
        normalizer['eps'] = -1
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


@pytest.mark.parametrize('policy', ['act', 'lingbot_va', 'cosmos', 'wan'])
def test_other_policies_have_no_image_finding(inputs, policy):
    cp, source, _ = inputs
    write_json(cp / 'config.json', {'type': policy, 'input_features': {}})
    assert fw.check_image_contract(cp, source) is None


def test_source_symlink_cannot_escape(inputs, tmp_path):
    cp, source, _ = inputs
    file = source / 'src/lerobot/processor/normalize_processor.py'
    outside = tmp_path / 'outside.py'
    file.rename(outside)
    file.symlink_to(outside)
    with pytest.raises(InputError, match='escapes'):
        fw.check_image_contract(cp, source)


def test_special_source_file_never_blocks(inputs):
    cp, source, _ = inputs
    file = source / 'src/lerobot/processor/normalize_processor.py'
    file.unlink()
    os.mkfifo(file)
    with pytest.raises(InputError, match='FIFO'):
        fw.check_image_contract(cp, source)


def test_cli_fault_correction_same_gate_and_missing_source(inputs, capsys):
    cp, source, pipeline = inputs
    argv = ['check', str(cp), '--runtime-source-root', str(source), '--fail-on', 'error']
    assert main(argv) == 1
    text = capsys.readouterr().out
    assert 'ERROR' in text and 'VISUAL=IDENTITY' in text and 'slope=' in text and ':1298' in text
    pipeline['steps'][-1]['config']['norm_map']['VISUAL'] = 'IDENTITY'
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert main(argv) == 0
    assert 'image map PASS' in capsys.readouterr().out
    assert main(['check', str(cp), '--fail-on', 'error', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['image_contract']['status'] == 'unknown'


def test_chooser_routes_to_existing_default_tool(inputs):
    cp, source, _ = inputs
    plan = choose_workflow('training', checkpoint=str(cp), runtime_source_root=str(source))
    assert plan['assessment_status'] == 'not_run'
    call = plan['can_do_now'][0]['mcp_request']
    assert call == {'name': 'preflight_check', 'arguments': {'source': str(cp), 'runtime_source_root': str(source), 'runtime_mode': 'training'}}


def test_null_image_declarations_are_missing_evidence(inputs):
    cp, source, _ = inputs
    document = json.loads((cp / 'config.json').read_text())
    document['input_features'] = None
    write_json(cp / 'config.json', document)
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


@pytest.mark.parametrize('epsilon', [10 ** 30, -(10 ** 30), 10 ** 1000])
def test_extreme_integer_epsilon_is_unknown_without_numpy_type_error(inputs, epsilon):
    cp, source, pipeline = inputs
    pipeline['steps'][-1]['config']['eps'] = epsilon
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


def test_native_float32_dtype_is_supported(inputs):
    cp, source, pipeline = inputs
    pipeline['steps'][-1]['config']['dtype'] = 'float32'
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'error'


def test_checkpoint_mcp_never_ignores_converter_directory(inputs, tmp_path):
    cp, source, _ = inputs
    server = create_product_server([str(cp), str(source)])

    async def run():
        with pytest.raises(Exception, match='separate'):
            await server.call_tool('preflight_check', {'source': str(cp), 'runtime_source_root': str(source),
                                                       'converter_directory': str(tmp_path)})
    asyncio.run(run())


def test_default_mcp_has_contract_path_and_refuses_outside_source(inputs, tmp_path):
    cp, source, _ = inputs
    server = create_product_server([str(cp), str(source)])
    assert {t.name for t in server._tool_manager.list_tools()} == BASIC_TOOLS
    tool = server._tool_manager.get_tool('preflight_check')
    assert 'runtime_source_root' in tool.parameters['properties']

    async def run():
        value = await server.call_tool('preflight_check', {'source': str(cp), 'runtime_source_root': str(source)})
        # FastMCP versions return either a CallToolResult or the content/structured tuple.
        report = value.structuredContent if hasattr(value, 'structuredContent') else value[1]
        assert report['status'] == 'error'
        assert report['image_contract']['images']['observation.images.image']['slope'] == [4.0] * 3
        assert len(json.dumps(report).encode()) < 8192
        with pytest.raises(Exception, match='outside'):
            await server.call_tool('preflight_check', {'source': str(cp), 'runtime_source_root': str(tmp_path)})
    asyncio.run(run())


@pytest.mark.parametrize('value', [None, [], False, 0, '', 'wrong', [1]])
def test_entire_wrong_type_feature_class_is_bounded(inputs, value):
    cp, source, _ = inputs
    document = json.loads((cp / 'config.json').read_text())
    document['input_features'] = value
    write_json(cp / 'config.json', document)
    try:
        report = fw.check_image_contract(cp, source)
    except InputError:
        return
    assert report['status'] == 'unknown'


@pytest.mark.parametrize('change', ['missing_visual', 'reorder', 'eps', 'second_camera_bad', 'both_cameras_bad'])
def test_mutation_sensitive_processing_controls(inputs, change):
    cp, source, pipeline = inputs
    normalizer = pipeline['steps'][-1]['config']
    if change == 'missing_visual':
        normalizer['norm_map'] = {}
        expected = 'pass'
    elif change == 'reorder':
        pipeline['steps'][0]['config'] = {}
        pipeline['steps'][0], pipeline['steps'][1] = pipeline['steps'][1], pipeline['steps'][0]
        expected = 'unknown'
    elif change == 'eps':
        normalizer['eps'] = 0.5
        expected = 'error'
    else:
        config = json.loads((cp / 'config.json').read_text())
        config['input_features']['observation.images.second'] = {'type': 'VISUAL', 'shape': [3, 8, 8]}
        normalizer['features'] = config['input_features']
        write_json(cp / 'config.json', config)
        save_file({'observation.images.image.mean': np.full(3, 0.5 if change == 'both_cameras_bad' else 0),
                   'observation.images.image.std': np.full(3, 0.5 if change == 'both_cameras_bad' else 1),
                   'observation.images.second.mean': np.full(3, 0.5),
                   'observation.images.second.std': np.full(3, 0.5)}, cp / 'stats.safetensors')
        expected = 'error'
    write_json(cp / 'policy_preprocessor.json', pipeline)
    report = fw.check_image_contract(cp, source)
    assert report['status'] == expected
    if change == 'eps':
        assert report['images']['observation.images.image']['slope'] == [2.] * 3
        assert report['images']['observation.images.image']['offset'] == [-2.] * 3
    if 'cameras_bad' in change or change == 'second_camera_bad':
        images = report['images']
        assert len(images) == 2
        assert images['observation.images.second']['consistent'] is False
        assert images['observation.images.image']['consistent'] is (change != 'both_cameras_bad')
        assert 'second' in report['findings'][0]['message']


def test_saved_remedy_does_not_certify_fresh_training(inputs):
    cp, source, pipeline = inputs
    config = json.loads((cp / 'config.json').read_text())
    config['normalization_mapping'] = {'VISUAL': 'MEAN_STD'}
    write_json(cp / 'config.json', config)
    pipeline['steps'][-1]['config']['norm_map']['VISUAL'] = 'IDENTITY'
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'pass'
    report = fw.check_image_contract(cp, source, 'training')
    assert report['status'] == 'unknown'
    assert 'config.json' in report['findings'][0]['remedy']
    config['normalization_mapping']['VISUAL'] = 'IDENTITY'
    write_json(cp / 'config.json', config)
    assert fw.check_image_contract(cp, source, 'training')['status'] == 'pass'


def test_generic_errors_survive_image_pass_and_unknown(inputs, capsys):
    from policy_doctor.hub import check_checkpoint

    cp, source, pipeline = inputs
    config = json.loads((cp / 'config.json').read_text())
    config['normalization_mapping'] = {'VISUAL': 'MEAN_STD'}
    write_json(cp / 'config.json', config)
    pipeline['steps'][-1]['config']['norm_map']['VISUAL'] = 'IDENTITY'
    write_json(cp / 'policy_preprocessor.json', pipeline)
    original = check_checkpoint(str(cp))
    errors = [f for f in original['findings'] if f['status'] == 'error']
    assert errors
    assert main(['check', str(cp), '--runtime-source-root', str(source), '--json', '--fail-on', 'error']) == 1
    report = json.loads(capsys.readouterr().out)
    assert all(f in report['findings'] for f in errors)
    assert report['image_contract']['status'] == 'pass'
    assert main(['check', str(cp), '--json', '--fail-on', 'error']) == 1
    report = json.loads(capsys.readouterr().out)
    assert all(f in report['findings'] for f in errors)


@pytest.mark.parametrize('change', ['missing_state_identity', 'lower_visual', 'device_nonblocking'])
def test_native_unloadable_processors_never_claim_consistency(inputs, change):
    cp, source, pipeline = inputs
    if change == 'missing_state_identity':
        pipeline['steps'][-1]['config']['norm_map']['VISUAL'] = 'IDENTITY'
        (cp / 'stats.safetensors').unlink()
    elif change == 'lower_visual':
        pipeline['steps'][-1]['config']['norm_map'] = {'visual': 'MEAN_STD'}
    else:
        pipeline['steps'][2]['config']['non_blocking'] = True
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


@pytest.mark.parametrize('selector', [{'revision': 'zzz'}, {'checkpoint_revision': 'zzz'}, {'max_download_mb': 0},
                                    {'video_headers': False}, {'max_rows': 1}, {'max_files': 1}, {'groot_contract': False}])
def test_mcp_checkpoint_rejects_inapplicable_selectors(inputs, selector):
    cp, source, _ = inputs
    server = create_product_server([str(cp), str(source)])

    async def run():
        with pytest.raises(Exception, match='separate'):
            await server.call_tool('preflight_check', dict(source=str(cp), runtime_source_root=str(source), **selector))
    asyncio.run(run())


def test_concise_checkpoint_retains_every_generic_error_and_counts(inputs):
    from policy_doctor.hub import check_checkpoint

    cp, source, _ = inputs
    full = fw.augment_checkpoint_report(check_checkpoint(str(cp)), fw.check_image_contract(cp, source))
    concise = fw.concise_checkpoint_report(full)
    assert concise['counts'] == full['counts'] and concise['status'] == full['status']
    assert [(f['code'], f['status']) for f in concise['findings']] == [(f['code'], f['status']) for f in full['findings']]
    assert concise['image_contract']['images'] == full['image_contract']['images']
    assert 'omitted' in concise['detail_omissions']
    assert len(json.dumps(concise).encode()) < 8192


@pytest.mark.parametrize('character', ['k', '\u754c'])
def test_eight_long_camera_keys_keep_error_and_numeric_witness_with_explicit_omissions(inputs, character):
    from policy_doctor.hub import check_checkpoint

    cp, source, pipeline = inputs
    keys = ['observation.images.' + str(i) + character * 236 for i in range(8)]
    features = {key: {'type': 'VISUAL', 'shape': [3, 8, 8]} for key in keys}
    write_json(cp / 'config.json', {'type': 'fastwam', 'input_features': features, 'output_features': {}})
    pipeline['steps'][-1]['config']['features'] = features
    write_json(cp / 'policy_preprocessor.json', pipeline)
    # Only the final camera is wrong: dropping the last row by input order
    # would conceal the most useful numeric witness.
    save_file({f'{key}.{stat}': np.full((3, 1, 1), 0.5 if key == keys[-1] else
               (0 if stat == 'mean' else 1), dtype=np.float32)
               for key in keys for stat in ('mean', 'std')}, cp / 'stats.safetensors')
    full = fw.augment_checkpoint_report(check_checkpoint(str(cp)), fw.check_image_contract(cp, source))
    original = json.dumps(full, sort_keys=True, allow_nan=False)
    compact = fw.concise_checkpoint_report(full)
    assert compact['status'] == full['status'] == 'error'
    assert compact['counts'] == full['counts']
    assert [(row['code'], row['status']) for row in compact['findings']] == [
        (row['code'], row['status']) for row in full['findings']]
    image = compact['image_contract']
    assert image['images_total'] == 8 and image['images_omitted'] > 0
    assert image['image_status_counts'] == {'consistent': 7, 'inconsistent': 1}
    assert next(iter(image['images'])) == keys[-1]
    assert image['images'][keys[-1]]['slope'] == [4.0] * 3
    assert image['images'][keys[-1]]['offset'] == [-3.0] * 3
    assert compact['full_report_sha256'] == hashlib.sha256(original.encode()).hexdigest()
    assert image['full_images_sha256'] == hashlib.sha256(
        json.dumps(full['image_contract']['images'], sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert json.dumps(full, sort_keys=True, allow_nan=False) == original
    assert len(json.dumps(compact, allow_nan=False, indent=2).encode()) <= 8192

    async def run():
        server = create_product_server([str(cp), str(source)])
        value = await server.call_tool('preflight_check', {'source': str(cp), 'runtime_source_root': str(source)})
        report = value.structuredContent if hasattr(value, 'structuredContent') else value[1]
        assert report['status'] == 'error'
        assert len(json.dumps(report, allow_nan=False, indent=2).encode()) <= 8192
        assert report['image_contract']['images'][keys[-1]]['slope'] == [4.0] * 3
    asyncio.run(run())


def test_non_fastwam_mcp_does_not_authorize_unrelated_installed_source(inputs, tmp_path, monkeypatch):
    cp, _, _ = inputs
    config = json.loads((cp / 'config.json').read_text())
    config['type'] = 'act'
    write_json(cp / 'config.json', config)
    outside = tmp_path / 'outside-installed-source'
    outside.mkdir()
    monkeypatch.setattr(fw, 'discover_source_root', lambda: outside)
    server = create_product_server([str(cp)])

    async def run():
        value = await server.call_tool('preflight_check', {'source': str(cp)})
        report = value.structuredContent if hasattr(value, 'structuredContent') else value[1]
        assert report['kind'] == 'checkpoint_check'
        assert 'image_contract' not in report
    asyncio.run(run())


def test_runtime_mode_is_not_silently_ignored_on_config(inputs, capsys):
    cp, _, _ = inputs
    assert main(['check', str(cp / 'config.json'), '--kind', 'config', '--runtime-mode', 'training', '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'input_error'


@pytest.mark.parametrize('mapping', [{}, {'VISUAL': 'IDENTITY'}])
def test_training_uses_resolved_config_identity_over_saved_meanstd(inputs, mapping):
    cp, source, _ = inputs
    config = json.loads((cp / 'config.json').read_text())
    config['normalization_mapping'] = mapping
    write_json(cp / 'config.json', config)
    assert fw.check_image_contract(cp, source, 'saved')['status'] == 'error'
    training = fw.check_image_contract(cp, source, 'training')
    assert training['status'] == 'pass'
    assert training['images']['observation.images.image']['saved']['mode'] == 'IDENTITY'
    assert training['images']['observation.images.image']['slope'] == [2.] * 3


@pytest.mark.parametrize('change', ['bool_epsilon', 'negative_std_positive_denominator', 'wrong_broadcast_shape',
                                   'nine_images', 'normalizer_bfloat16'])
def test_observable_native_input_boundaries_remain_unknown(inputs, change):
    cp, source, pipeline = inputs
    normalizer = pipeline['steps'][-1]['config']
    if change == 'bool_epsilon':
        normalizer['eps'] = True
    elif change == 'negative_std_positive_denominator':
        normalizer['eps'] = 0.75
        save_file({'observation.images.image.mean': np.zeros(3, dtype=np.float32),
                   'observation.images.image.std': np.full(3, -0.25, dtype=np.float32)}, cp / 'stats.safetensors')
    elif change == 'wrong_broadcast_shape':
        save_file({'observation.images.image.mean': np.zeros((1, 3), dtype=np.float32),
                   'observation.images.image.std': np.ones((1, 3), dtype=np.float32)}, cp / 'stats.safetensors')
    elif change == 'nine_images':
        config = json.loads((cp / 'config.json').read_text())
        config['input_features'] = {f'observation.images.camera{i}': {'type': 'VISUAL', 'shape': [3, 8, 8]}
                                    for i in range(9)}
        normalizer['features'] = config['input_features']
        write_json(cp / 'config.json', config)
        save_file({f'{key}.{stat}': np.full(3, value, dtype=np.float32)
                   for key in config['input_features'] for stat, value in [('mean', 0), ('std', 1)]},
                  cp / 'stats.safetensors')
    else:
        normalizer['dtype'] = 'bfloat16'
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


def test_resuming_is_refused_by_direct_contract_api(inputs):
    cp, source, _ = inputs
    with pytest.raises(InputError, match='saved or training'):
        fw.check_image_contract(cp, source, 'resume')


def test_source_loop_is_a_bounded_input_error_on_cli_and_mcp(inputs, capsys):
    cp, source, _ = inputs
    path = source / 'src/lerobot/processor/normalize_processor.py'
    path.unlink()
    sibling = path.with_name('loop.py')
    path.symlink_to(sibling.name)
    sibling.symlink_to(path.name)
    assert main(['check', str(cp), '--runtime-source-root', str(source), '--json', '--fail-on', 'error']) == 2
    response = json.loads(capsys.readouterr().out)
    assert response['status'] == 'input_error'
    assert 'resolve' in response['error']['message'].lower()
    server = create_product_server([str(cp), str(source)])

    async def run():
        with pytest.raises(Exception, match='resolve'):
            await server.call_tool('preflight_check', {'source': str(cp), 'runtime_source_root': str(source)})
    asyncio.run(run())


def test_discovered_source_must_be_authorized_before_any_profile_read(inputs, monkeypatch):
    cp, source, _ = inputs
    calls = []
    monkeypatch.setattr(fw, 'discover_source_root', lambda: source)

    def deny(path):
        calls.append(path)
        raise InputError('outside read roots')

    def forbidden_read(path):
        pytest.fail('An unauthorized source reached the profile reader')

    monkeypatch.setattr(fw, '_source_profile', forbidden_read)
    report = fw.check_image_contract(cp, authorize_source=deny)
    assert calls == [str(source)]
    assert report['status'] == 'unknown'
    assert 'authorized' in report['findings'][0]['message']
    with pytest.raises(InputError, match='outside'):
        fw.check_image_contract(cp, source, authorize_source=deny)


def test_mcp_discovered_source_outside_roots_preserves_generic_audit(inputs, monkeypatch):
    from policy_doctor.hub import check_checkpoint

    cp, source, _ = inputs
    generic = check_checkpoint(str(cp))
    monkeypatch.setattr(fw, 'discover_source_root', lambda: source)
    server = create_product_server([str(cp)])

    async def run():
        response = await server.call_tool('preflight_check', {'source': str(cp)})
        report = response.structuredContent if hasattr(response, 'structuredContent') else response[1]
        assert report['image_contract']['status'] == 'unknown'
        assert all(any((item['code'], item['status']) == (original['code'], original['status'])
                       for item in report['findings']) for original in generic['findings'])
        with pytest.raises(Exception, match='outside'):
            await server.call_tool('preflight_check', {'source': str(cp), 'runtime_source_root': str(source)})
    asyncio.run(run())


def test_concise_messages_remedies_references_and_full_digest_are_preserved(inputs):
    from policy_doctor.hub import check_checkpoint

    cp, source, _ = inputs
    full = fw.augment_checkpoint_report(check_checkpoint(str(cp)), fw.check_image_contract(cp, source))
    full['findings'].append({'code': 'independent.long_condition', 'status': 'error',
                             'message': 'condition ' * 40, 'remedy': 'repair ' * 50,
                             'evidence': [{'path': 'config.json', 'pointer': '/type', 'sha256': 'a' * 64}]})
    full['counts']['error'] += 1
    concise = fw.concise_checkpoint_report(full)
    for original, compact in zip(full['findings'], concise['findings'], strict=True):
        if original['code'] != 'fastwam.image_map':
            assert compact.get('message') == original.get('message', '')
            if original['status'] == 'error' and original.get('remedy'):
                assert compact.get('remedy') == original['remedy']
            if original['status'] == 'error' and original.get('evidence'):
                reference = {k: v for k, v in original['evidence'][0].items() if k in ('path', 'pointer', 'sha256')}
                assert compact.get('evidence') == [reference]
    expected = hashlib.sha256(json.dumps(full, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert concise['full_report_sha256'] == expected
    assert concise['image_contract']['full_report_sha256'] == hashlib.sha256(
        json.dumps(full['image_contract'], sort_keys=True, allow_nan=False).encode()).hexdigest()


def test_large_compact_text_marks_exact_original_length_and_digest(inputs):
    from policy_doctor.hub import check_checkpoint

    cp, source, _ = inputs
    full = fw.augment_checkpoint_report(check_checkpoint(str(cp)), fw.check_image_contract(cp, source))
    text = 'critical condition ' * 200
    full['findings'].append({'code': 'independent.oversize_text', 'status': 'error', 'message': text, 'remedy': text})
    full['counts']['error'] += 1
    item = fw.concise_checkpoint_report(full)['findings'][-1]
    for field in ('message', 'remedy'):
        assert '[truncated;' in item[field]
        assert item[field + '_truncation'] == {'characters': len(text), 'sha256': hashlib.sha256(text.encode()).hexdigest()}


def test_saved_training_visual_echo_is_bounded_and_digest_bound(inputs):
    cp, source, _ = inputs
    config = json.loads((cp / 'config.json').read_text())
    original = 'X' * (2 * 1024 * 1024)
    config['normalization_mapping'] = {'VISUAL': original}
    write_json(cp / 'config.json', config)
    report = fw.check_image_contract(cp, source)
    assert len(json.dumps(report).encode()) < 8192
    assert report['training_visual_mode_truncation'] == {
        'characters': len(original), 'sha256': hashlib.sha256(original.encode()).hexdigest()}


@pytest.mark.parametrize('policy', ['lingbot_va', 'lingbot', 'cosmos'])
@pytest.mark.parametrize('selector', ['mode', 'source'])
def test_non_fastwam_selectors_are_refused_in_cli_and_mcp(inputs, capsys, policy, selector):
    cp, source, _ = inputs
    config = json.loads((cp / 'config.json').read_text())
    config['type'] = policy
    write_json(cp / 'config.json', config)
    arguments = ['--runtime-mode', 'training'] if selector == 'mode' else ['--runtime-source-root', str(source)]
    assert main(['check', str(cp), '--json', *arguments]) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'input_error'
    server = create_product_server([str(cp), str(source)])

    async def run():
        extra = {'runtime_mode': 'training'} if selector == 'mode' else {'runtime_source_root': str(source)}
        with pytest.raises(Exception, match='FastWAM'):
            await server.call_tool('preflight_check', {'source': str(cp), **extra})
    asyncio.run(run())


@pytest.mark.parametrize('where,value', [('normalizer', 2), ('normalizer', []), ('normalizer', 'x' * 1000),
                                         ('normalizer', 'banana'), ('device', None), ('device', 1), ('device', [])])
def test_unsupported_device_declarations_are_unassessed(inputs, where, value):
    cp, source, pipeline = inputs
    pipeline['steps'][-1 if where == 'normalizer' else 2]['config']['device'] = value
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'unknown'


def test_null_normalizer_device_is_a_supported_native_default(inputs):
    cp, source, pipeline = inputs
    pipeline['steps'][-1]['config']['device'] = None
    write_json(cp / 'policy_preprocessor.json', pipeline)
    assert fw.check_image_contract(cp, source)['status'] == 'error'


def test_home_absolute_command_survives_actual_shell_argument_parsing():
    import subprocess
    import sys
    from pathlib import Path

    from policy_doctor.console_text import display_command

    path = str(Path.home() / "robot's data; $(printf should-not-expand)" / 'checkpoint')
    command = display_command([sys.executable, '-c', 'import sys; print(sys.argv[1])', path])
    completed = subprocess.run(['bash', '-c', command], text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0
    assert completed.stdout.rstrip('\n') == path
    assert completed.stderr == ''
