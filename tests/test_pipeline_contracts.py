"""Contract behavior and gates; native public-source proofs run separately."""
import asyncio
import hashlib
import json
import os
from types import MethodType, SimpleNamespace

import pytest

from policy_doctor import pipeline_contracts as pc
from policy_doctor.artifacts import InputError
from policy_doctor.product_cli import main
from policy_doctor.product_mcp import BASIC_TOOLS, create_product_server
from policy_doctor.workflow_readiness import choose_workflow

CAMERAS = ['observation.images.front', 'observation.images.wrist', 'observation.images.empty_camera_0']


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / 'runtime'
    source = b'raise RuntimeError("This source is matched as text, never imported")\n'
    digest = hashlib.sha256(source).hexdigest()
    path = root / pc.OPENPI_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    monkeypatch.setattr(pc, 'OPENPI_HASH', digest)
    profiles = {}
    for policy, (relative, _, method) in pc.CAMERA_PROFILES.items():
        path = root / 'src' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        profiles[policy] = relative, {digest: ('0' * 40, 1)}, method
    monkeypatch.setattr(pc, 'CAMERA_PROFILES', profiles)
    return root


def initialization(path=None, *, intent='pretrained', resume=False, backend='pytorch'):
    return {'schema_version': 1, 'kind': 'openpi_initialization', 'backend': backend,
            'expected_initialization': intent, 'train_config': {
                'name': 'actual-config', 'resume': resume, 'pytorch_weight_path': path,
                'weight_loader': {'class': 'openpi.training.weight_loaders.CheckpointWeightLoader',
                                  'params_path': 'gs://openpi-assets/checkpoints/pi0_base/params'}}}


def cameras(policy='pi05', *, supplied=None, expected=None, empty=2):
    return {'schema_version': 1, 'kind': 'lerobot_camera_inputs', 'policy_type': policy,
            'expected_image_keys': list(CAMERAS if expected is None else expected),
            'provided_image_keys': list(CAMERAS if supplied is None else supplied),
            'empty_cameras': empty if policy == 'smolvla' else None}


@pytest.mark.parametrize(('path', 'intent', 'resume', 'backend', 'status'), [
    (None, 'pretrained', False, 'pytorch', 'error'),
    ('/owned/converted-model', 'pretrained', False, 'pytorch', 'pass'),
    (None, 'scratch', False, 'pytorch', 'pass'),
    (None, 'unspecified', False, 'pytorch', 'warning'),
    ('/owned/converted-model', 'scratch', False, 'pytorch', 'error'),
    (None, 'pretrained', True, 'pytorch', 'unknown'),
    (None, 'pretrained', False, 'jax', 'unknown'),
])
def test_initialization_selection_and_lawful_controls(runtime, path, intent, resume, backend, status):
    document = initialization(path, intent=intent, resume=resume, backend=backend)
    report = pc.check_pipeline_document(document, runtime_source_root=runtime)
    assert report['status'] == status
    if status == 'pass':
        assert report['weight_file_checked'] is False
        assert report['pretrained_load_selected'] is (path is not None)


def test_unknown_source_never_approves_a_config(runtime):
    (runtime / pc.OPENPI_FILE).write_text('changed implementation')
    report = pc.check_pipeline_document(initialization('/model'), runtime_source_root=runtime)
    assert report['status'] == 'unknown'
    assert 'SHA256' in report['findings'][0]['message']
    assert report['findings'][0]['evidence'][0]['sha256']


@pytest.mark.parametrize('policy', ['pi0', 'pi05', 'smolvla'])
@pytest.mark.parametrize(('supplied', 'status', 'slots'), [
    (CAMERAS, 'pass', CAMERAS),
    (CAMERAS[1:2], 'error', [CAMERAS[1], None, None]),
    ([CAMERAS[0], CAMERAS[2]], 'error', [CAMERAS[0], CAMERAS[2], None]),
    (CAMERAS[:2], 'pass', [CAMERAS[0], CAMERAS[1], None]),
    ([], 'error', None),
])
def test_camera_leading_interior_trailing_full_and_empty(runtime, policy, supplied, status, slots):
    report = pc.check_pipeline_document(cameras(policy, supplied=supplied), runtime_source_root=runtime)
    assert report['status'] == status
    if slots is not None:
        assert report['native_slot_cameras'] == slots
        assert len(report['shifted_cameras']) == sum(key is not None and key != CAMERAS[i] for i, key in enumerate(slots))


def test_every_shifted_camera_is_retained_and_provided_order_cannot_change_config_order(runtime):
    keys = ['observation.images.a', 'observation.images.b', 'observation.images.c', 'observation.images.d']
    report = pc.check_pipeline_document(cameras(expected=keys, supplied=[keys[3], keys[2]]), runtime_source_root=runtime)
    assert report['shifted_cameras'] == [
        {'camera': keys[2], 'expected_slot': 2, 'actual_slot': 0},
        {'camera': keys[3], 'expected_slot': 3, 'actual_slot': 1}]


def test_smolvla_preserves_native_empty_camera_cap(runtime):
    report = pc.check_pipeline_document(cameras('smolvla', supplied=CAMERAS[:1], empty=0), runtime_source_root=runtime)
    assert report['status'] == 'pass'
    assert report['native_slot_cameras'] == CAMERAS[:1]
    assert report['missing_image_keys'] == CAMERAS[1:]


@pytest.mark.parametrize('policy', ['act', 'fastwam', 'lingbot_va', 'cosmos_policy'])
def test_other_policies_are_not_given_this_slot_contract(runtime, policy):
    assert pc.check_pipeline_document(cameras(policy), runtime_source_root=runtime)['status'] == 'unknown'


@pytest.mark.parametrize('kind', ['weights', 'cameras'])
def test_source_omission_stays_unknown(kind):
    document = initialization('/model') if kind == 'weights' else cameras()
    assert pc.check_pipeline_document(document)['status'] == 'unknown'


@pytest.mark.parametrize('mutation', ['version_bool', 'resume_integer', 'missing_path', 'unexpected_field', 'empty_path',
                                      'camera_scalar', 'camera_duplicate', 'camera_unknown', 'camera_control', 'too_many'])
def test_bad_contracts_refused_without_guessing(runtime, mutation):
    document = initialization()
    if mutation == 'version_bool':
        document['schema_version'] = True
    elif mutation == 'resume_integer':
        document['train_config']['resume'] = 1
    elif mutation == 'missing_path':
        del document['train_config']['pytorch_weight_path']
    elif mutation == 'unexpected_field':
        document['resume'] = True
    elif mutation == 'empty_path':
        document['train_config']['pytorch_weight_path'] = ''
    else:
        document = cameras()
        if mutation == 'camera_scalar':
            document['expected_image_keys'] = None
        elif mutation == 'camera_duplicate':
            document['provided_image_keys'] = CAMERAS[:1] * 2
        elif mutation == 'camera_unknown':
            document['provided_image_keys'] = ['observation.images.unknown']
        elif mutation == 'camera_control':
            document['expected_image_keys'][0] += '\n'
        else:
            document['expected_image_keys'] = ['observation.images.' + str(i) for i in range(9)]
    with pytest.raises(InputError):
        pc.check_pipeline_document(document, runtime_source_root=runtime)


def test_fifo_and_source_symlink_escape_are_refused(runtime, tmp_path):
    source = runtime / pc.OPENPI_FILE
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(InputError, match='FIFO'):
        pc.check_pipeline_document(initialization(), runtime_source_root=runtime)
    source.unlink()
    outside = tmp_path / 'outside.py'
    outside.write_text('outside')
    source.symlink_to(outside)
    with pytest.raises(InputError, match='escapes'):
        pc.check_pipeline_document(initialization(), runtime_source_root=runtime)


def test_actual_config_capture_and_guard_stop_before_consumer(runtime):
    config = SimpleNamespace(name='real-object', pytorch_weight_path=None, resume=False,
                             weight_loader=SimpleNamespace(params_path='gs://base'))
    assert pc.capture_openpi_initialization(config)['train_config']['pytorch_weight_path'] is None
    consumed = []
    with pytest.raises(pc.PipelineContractError) as failure:
        pc.guard_openpi_initialization(config, runtime_source_root=runtime)
        consumed.append(True)
    assert failure.value.report['status'] == 'error' and not consumed
    config.pytorch_weight_path = '/owned/pytorch-model'
    assert pc.guard_openpi_initialization(config, runtime_source_root=runtime)['status'] == 'pass'


def test_actual_batch_capture_guard_and_genuine_camera_restoration(runtime):
    config = SimpleNamespace(type='pi05', image_features=dict.fromkeys(CAMERAS))
    batch = {CAMERAS[1]: object()}
    before = pc.capture_lerobot_camera_inputs(config, batch)
    assert before['provided_image_keys'] == CAMERAS[1:2]
    with pytest.raises(pc.PipelineContractError):
        pc.guard_lerobot_camera_inputs(config, batch, runtime_source_root=runtime)
    batch[CAMERAS[0]] = object()  # The caller restores a genuine source; no zero image is fabricated.
    report = pc.guard_lerobot_camera_inputs(config, batch, runtime_source_root=runtime)
    assert report['status'] == 'pass' and report['shifted_cameras'] == []


@pytest.mark.parametrize('kind', ['weights', 'cameras'])
def test_cli_same_gate_fault_correction_and_readable_evidence(runtime, tmp_path, capsys, kind):
    path = tmp_path / 'actual.json'
    document = initialization() if kind == 'weights' else cameras(supplied=CAMERAS[1:2])
    path.write_text(json.dumps(document))
    argv = ['check', str(path), '--runtime-source-root', str(runtime), '--fail-on', 'error']
    assert main(argv) == 1
    text = capsys.readouterr().out
    assert 'ERROR' in text and 'Evidence:' in text and 'Scope:' in text
    if kind == 'weights':
        document['train_config']['pytorch_weight_path'] = '/owned/converted'
        assert 'pytorch_weight_path=null' in text
    else:
        document['provided_image_keys'] = CAMERAS[:2]
        assert CAMERAS[1] in text and 'slot 1 to slot 0' in text
    path.write_text(json.dumps(document))
    assert main(argv) == 0
    assert 'CONSISTENT' in capsys.readouterr().out


def test_default_mcp_and_chooser_use_same_existing_tool(runtime, tmp_path):
    path = tmp_path / 'actual.json'
    path.write_text(json.dumps(initialization()))
    server = create_product_server([str(tmp_path)])
    assert {t.name for t in server._tool_manager.list_tools()} == BASIC_TOOLS
    assert 'runtime_source_root' in server._tool_manager.get_tool('preflight_check').parameters['properties']

    async def run():
        response = await server.call_tool('preflight_check', {'source': str(path), 'runtime_source_root': str(runtime)})
        report = response.structuredContent if hasattr(response, 'structuredContent') else response[1]
        assert report['status'] == 'error'
        assert len(json.dumps(report).encode()) < 7 * 1024
        with pytest.raises(Exception, match='separate'):
            await server.call_tool('preflight_check', {'source': str(path), 'runtime_source_root': str(runtime), 'converter_directory': str(tmp_path)})
    asyncio.run(run())
    plan = choose_workflow('training', pipeline_contract=str(path), runtime_source_root=str(runtime))
    assert plan['can_do_now'][0]['mcp_request'] == {
        'name': 'preflight_check', 'arguments': {'source': str(path), 'runtime_source_root': str(runtime)}}


def test_mcp_refuses_outside_source_before_reading_contract(runtime, tmp_path):
    path = tmp_path / 'actual.json'
    path.write_text('{broken')
    server = create_product_server([str(tmp_path)])
    async def run():
        with pytest.raises(Exception, match='outside'):
            await server.call_tool('preflight_check', {'source': str(path), 'runtime_source_root': '/etc'})
    asyncio.run(run())


def test_cli_does_not_reinterpret_an_unrelated_checkpoint_as_pipeline_contract(runtime, tmp_path, capsys):
    cp = tmp_path / 'cp'
    cp.mkdir()
    (cp / 'config.json').write_text(json.dumps({'type': 'act', 'input_features': {}, 'output_features': {}}))
    assert main(['check', str(cp), '--runtime-source-root', str(runtime)]) == 2
    assert 'exported' in capsys.readouterr().err


@pytest.fixture
def native_hook_actor(runtime, monkeypatch):
    relative, _, method = pc.CAMERA_PROFILES['pi05']
    path = runtime / 'src' / relative
    source = 'def _preprocess_images(self, batch):\n    self.consumed += 1\n    return batch\n'
    path.write_text(source)
    profiles = dict(pc.CAMERA_PROFILES)
    profiles['pi05'] = relative, {hashlib.sha256(source.encode()).hexdigest(): ('0' * 40, 1)}, method
    monkeypatch.setattr(pc, 'CAMERA_PROFILES', profiles)
    namespace = {}
    exec(compile(source, str(path), 'exec'), namespace)
    actor = SimpleNamespace(config=SimpleNamespace(type='pi05', image_features=dict.fromkeys(CAMERAS)), consumed=0)
    actor._preprocess_images = MethodType(namespace['_preprocess_images'], actor)
    return actor, path


def test_installed_hook_blocks_at_native_boundary_and_can_be_removed(native_hook_actor):
    actor, _ = native_hook_actor
    original = actor._preprocess_images
    guard = pc.install_lerobot_camera_guard(actor)
    with pytest.raises(pc.PipelineContractError):
        actor._preprocess_images({CAMERAS[1]: object()})
    assert actor.consumed == 0 and guard.last_report['status'] == 'error'
    with pytest.raises(InputError, match='already installed'):
        pc.install_lerobot_camera_guard(actor)
    batch = dict.fromkeys(CAMERAS[:2])
    assert actor._preprocess_images(batch) is batch
    assert actor.consumed == 1 and guard.last_report['status'] == 'pass'
    assert guard.last_contract['runtime_source_root']
    guard.remove()
    assert actor._preprocess_images == original


def test_camera_hook_rejects_a_method_bound_to_another_policy(native_hook_actor):
    actor, _ = native_hook_actor
    other = SimpleNamespace(config=SimpleNamespace(type='pi05', image_features=dict.fromkeys(reversed(CAMERAS))), consumed=0)
    actor._preprocess_images = MethodType(actor._preprocess_images.__func__, other)
    with pytest.raises(InputError, match='different policy instance'):
        pc.install_lerobot_camera_guard(actor)
    assert actor.consumed == other.consumed == 0


def test_camera_hook_rejects_same_filename_and_name_with_different_code(native_hook_actor):
    actor, path = native_hook_actor
    original = actor._preprocess_images
    namespace = {}
    exec(compile('def _preprocess_images(self, batch):\n    self.consumed += 2\n    return batch\n', str(path), 'exec'), namespace)
    replacement = namespace['_preprocess_images']
    replacement.__qualname__ = original.__func__.__qualname__
    actor._preprocess_images = MethodType(replacement, actor)
    with pytest.raises(InputError, match='body differs'):
        pc.install_lerobot_camera_guard(actor)
    assert actor.consumed == 0


def test_camera_hook_refuses_code_change_after_installation(native_hook_actor):
    actor, path = native_hook_actor
    original_function = actor._preprocess_images.__func__
    guard = pc.install_lerobot_camera_guard(actor)
    namespace = {}
    exec(compile('def _preprocess_images(self, batch):\n    return None\n', str(path), 'exec'), namespace)
    original_function.__code__ = namespace['_preprocess_images'].__code__
    with pytest.raises(InputError, match='changed after installation'):
        actor._preprocess_images(dict.fromkeys(CAMERAS))
    assert guard.last_report is None and actor.consumed == 0


def test_camera_hook_refuses_rebinding_the_guard_itself(native_hook_actor):
    actor, _ = native_hook_actor
    guard = pc.install_lerobot_camera_guard(actor)
    other = SimpleNamespace(config=actor.config, consumed=0)
    rebound = MethodType(guard.bound.__func__, other)
    with pytest.raises(InputError, match='changed after installation'):
        rebound(dict.fromkeys(CAMERAS))
    assert guard.last_report is None and actor.consumed == other.consumed == 0


@pytest.mark.parametrize('replacement', ['other_receiver', 'not_a_method'])
def test_camera_hook_refuses_replacement_of_the_retained_method(native_hook_actor, replacement):
    actor, _ = native_hook_actor
    guard = pc.install_lerobot_camera_guard(actor)
    other = SimpleNamespace(config=actor.config, consumed=0)
    guard.original = MethodType(guard.original.__func__, other) if replacement == 'other_receiver' else object()
    with pytest.raises(InputError, match='changed after installation'):
        actor._preprocess_images(dict.fromkeys(CAMERAS))
    assert guard.last_report is None and actor.consumed == other.consumed == 0


def test_embedded_source_context_routes_without_new_flags(runtime, tmp_path, capsys):
    path = tmp_path / 'actual.json'
    document = initialization()
    document['runtime_source_root'] = str(runtime)
    path.write_text(json.dumps(document))
    assert main(['check', str(path), '--fail-on', 'error']) == 1
    assert 'pytorch_weight_path=null' in capsys.readouterr().out


def test_mcp_authorizes_embedded_source_and_explicit_override(runtime, tmp_path):
    path = tmp_path / 'actual.json'
    document = initialization()
    document['runtime_source_root'] = '/etc'
    path.write_text(json.dumps(document))
    server = create_product_server([str(tmp_path)])
    async def run():
        with pytest.raises(Exception, match='outside'):
            await server.call_tool('preflight_check', {'source': str(path)})
        response = await server.call_tool('preflight_check', {'source': str(path), 'runtime_source_root': str(runtime)})
        report = response.structuredContent if hasattr(response, 'structuredContent') else response[1]
        assert report['status'] == 'error'
    asyncio.run(run())
