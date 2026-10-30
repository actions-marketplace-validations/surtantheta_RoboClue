"""Native default/argument facts must stay visible at the consumer boundary."""
import copy
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.consumer_version import (
    SOURCE_PROFILES,
    apply_consumer_version,
    consumer_version_context,
    supported_consumer_profiles,
    verify_consumer_source,
)
from policy_doctor.robot_contract import check_declared_camera_keys, validate_robot_declaration
from policy_doctor.robot_interface import from_command

BASE = 'lerobot-rollout --robot.type=so101_follower --robot.port=/not-opened'


def dataset(tmp_path, cameras=('up', 'side')):
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'features': {
        'observation.state': {'dtype': 'float32', 'shape': [6]},
        **{'observation.images.' + name: {'dtype': 'video', 'shape': [480, 640, 3]} for name in cameras}}}))
    return root


@pytest.mark.parametrize('version', [None, '0.6.2'])
@pytest.mark.parametrize('kind', ['so100_follower', 'so101_follower'])
def test_omitted_camera_mapping_resolves_empty_native_default(tmp_path, version, kind):
    root = dataset(tmp_path)
    result = check_declared_camera_keys(root, robot_command=BASE.replace('so101_follower', kind), lerobot_version=version)
    assert result['requested'] is True and result['compared'] is True
    assert result['status'] == 'warning' and result['findings'][0]['code'] == 'camera.default_empty'
    assert result['declared_keys'] == [] and result['missing'] == ['observation.images.side', 'observation.images.up']
    assert result['coverage']['explicit_camera_declaration'] is False
    assert result['coverage']['default_camera_declaration'] is True
    assert result['default_source']['sha256'] and result['findings'][0]['evidence'][-1]['value'] == []
    assert 'supplies none' in result['findings'][0]['message']
    assert 'not certify' in result['scope']


def test_unknown_version_does_not_invent_an_empty_camera_default(tmp_path):
    result = check_declared_camera_keys(dataset(tmp_path), robot_command=BASE, lerobot_version='0.6.3')
    assert result['status'] == 'unknown' and result['requested'] is True and result['compared'] is False
    assert result['coverage']['default_camera_declaration'] is False
    assert result['findings'][0]['code'] == 'camera.default_unqualified'


def test_no_image_keys_and_no_cameras_agree_without_claiming_policy_compatibility(tmp_path):
    result = check_declared_camera_keys(dataset(tmp_path, ()), robot_command=BASE, lerobot_version='0.6.2')
    assert result['status'] == 'pass' and result['dataset_keys'] == result['declared_keys'] == []
    assert result['compared'] is True and 'policy compatibility' in result['not_checked']


def test_explicit_matching_cameras_clear_the_default_empty_warning(tmp_path):
    root = dataset(tmp_path)
    command = BASE + " --robot.cameras='{up: {type: opencv, index_or_path: 0}, side: {type: opencv, index_or_path: 1}}'"
    result = check_declared_camera_keys(root, robot_command=command, lerobot_version='0.6.2')
    assert result['status'] == 'pass' and result['coverage']['explicit_camera_declaration'] is True
    assert result['coverage']['default_camera_declaration'] is False


@pytest.mark.parametrize('key', ['use_degrees', 'disable_torque_on_disconnect', 'port', 'cameras', 'left_arm_config.use_degrees'])
@pytest.mark.parametrize('tail', ['', ' --policy.path=local/checkpoint'])
def test_valueless_robot_flags_cannot_turn_into_true(key, tail):
    text = BASE.replace('so101_follower', 'bi_so_follower') if key.startswith('left_') else BASE
    if key == 'port':
        text = text.replace(' --robot.port=/not-opened', '')
    with pytest.raises(InputError, match='explicit value'):
        validate_robot_declaration(robot_command=text + ' --robot.' + key + tail)


@pytest.mark.parametrize('value', ['true', 'false'])
@pytest.mark.parametrize('separator', ['=', ' '])
def test_explicit_boolean_native_forms_remain_valid(value, separator):
    result = validate_robot_declaration(robot_command=BASE + ' --robot.use_degrees' + separator + value)
    assert result['flags']['use_degrees'] is (value == 'true')


def test_legacy_parser_remains_distinct_from_strict_native_input_validation():
    assert from_command(BASE + ' --robot.use_degrees')['flags']['use_degrees'] is True


def report():
    return {'kind': 'units_manifest', 'status': 'pass', 'findings': [{'code': 'units.joint_mode_match', 'status': 'pass'}],
            'manifest': {'features': {'observation.state': {'coordinates': [{'unit': 'DEGREES'}]}},
                'deployment': {'robot_type': 'so101_follower', 'program': 'rollout', 'unknown': None,
                    'arms': {'': {'use_degrees': True, 'mode': 'DEGREES', 'origin': 'the default of pinned source', 'basis': 'conditional'}}}}}


def test_required_consumer_version_cannot_approve_an_omitted_label():
    original = report()
    saved = copy.deepcopy(original)
    result = apply_consumer_version(original, require_version=True)
    assert result['status'] == 'unknown' and result['consumer_version']['comparison_allowed'] is False
    assert result['consumer_version']['declared_version'] is None
    assert 'Specify the LeRobot version' in result['consumer_version']['reason']
    assert not any(f['code'] == 'units.joint_mode_match' for f in result['findings'])
    assert result['conditional_comparison']['findings'][0]['code'] == 'units.joint_mode_match'
    assert original == saved


def test_required_consumer_version_accepts_the_qualified_label_only():
    original = report()
    assert apply_consumer_version(original, '0.6.2', require_version=True)['status'] == 'pass'
    assert apply_consumer_version(original, '0.6.3', require_version=True)['status'] == 'unknown'
    assert consumer_version_context(deployment=original['manifest']['deployment'])['command_role'] == 'policy_rollout'


@pytest.mark.parametrize('value', ['true', 1, None])
def test_require_version_is_a_strict_boolean(value):
    with pytest.raises(InputError, match='boolean'):
        apply_consumer_version(report(), require_version=value)


@pytest.mark.parametrize('version', ['0.6.1', '0.6.2'])
@pytest.mark.parametrize('kind,expected', [
    ('so100_follower', 'warning'), ('so101_follower', 'warning'),
    ('koch_follower', 'warning'), ('omx_follower', 'warning'),
    ('bi_so_follower', 'warning'), ('lekiwi', 'pass'),
])
def test_recorded_native_camera_defaults_cover_each_qualified_family(tmp_path, version, kind, expected):
    root = dataset(tmp_path, ('front', 'wrist'))
    command = (BASE.replace('so101_follower', kind) if kind != 'bi_so_follower' else
               'lerobot-rollout --robot.type=bi_so_follower --robot.left_arm_config.port=/not-opened --robot.right_arm_config.port=/not-opened')
    result = check_declared_camera_keys(root, robot_command=command, lerobot_version=version)
    assert result['status'] == expected and result['requested'] is True and result['compared'] is True
    assert result['coverage']['default_camera_declaration'] is True
    expected_keys = ['observation.images.front', 'observation.images.wrist'] if kind == 'lekiwi' else []
    assert result['declared_keys'] == expected_keys
    assert result['default_source']['artifact']['sha256'] if version == '0.6.1' else result['default_source']['source_revision']


@pytest.mark.parametrize('version', ['0.6.1', '0.6.2'])
@pytest.mark.parametrize('kind', ['so100_follower', 'so101_follower', 'koch_follower', 'omx_follower', 'bi_so_follower', 'lekiwi'])
def test_qualified_family_label_preserves_comparison_without_claiming_robot_execution(version, kind):
    original = report()
    original['manifest']['deployment']['robot_type'] = kind
    result = apply_consumer_version(original, version, require_version=True)
    context = result['consumer_version']
    assert context['comparison_allowed'] is True and context['profile_version'] == version
    assert result['status'] == 'pass' and context['fleet_verified'] is False
    assert context['source_basis']['critical_files']
    assert context['camera_default']['keys'] == (['front', 'wrist'] if kind == 'lekiwi' else [])
    assert 'native command execution' in context['scope']


@pytest.mark.parametrize('suffix', ['json', 'yaml'])
def test_flat_robot_config_gets_an_actionable_input_error(tmp_path, suffix):
    path = tmp_path / ('robot.' + suffix)
    value = {'type': 'so101_follower', 'port': '/not-opened', 'use_degrees': False}
    if suffix == 'json':
        path.write_text(json.dumps(value))
    else:
        import yaml
        path.write_text(yaml.safe_dump(value))
    with pytest.raises(InputError, match='flat robot configuration.*top-level "robot"'):
        validate_robot_declaration(robot_config=path)


def test_public_source_matrix_is_a_copy_and_does_not_become_a_version_range():
    matrix = supported_consumer_profiles()
    assert {row['lerobot_version'] for row in matrix} == {'0.6.1', '0.6.2'}
    assert all(len(row['robot_types']) == 6 for row in matrix)
    matrix[0]['robot_types'].append('invented_robot')
    matrix[0]['artifact']['sha256'] = 'tampered'
    second = supported_consumer_profiles()
    assert 'invented_robot' not in second[0]['robot_types']
    assert second[0]['artifact']['sha256'] == SOURCE_PROFILES['0.6.1']['artifact']['sha256']
    assert consumer_version_context('0.6.1.post1', deployment=report()['manifest']['deployment'])['comparison_allowed'] is False


@pytest.mark.parametrize('version', ['0.6.1', '0.6.2'])
@pytest.mark.parametrize('kind', ['so101_follower', 'koch_follower', 'omx_follower', 'bi_so_follower', 'lekiwi'])
def test_source_matching_checks_the_selected_family_and_refuses_changed_bytes(tmp_path, monkeypatch, version, kind):
    import hashlib

    from policy_doctor import consumer_version as module

    # Separate fixed per-profile bytes exercise matching and source-over-label
    # precedence. Actual upstream byte identities are checked by the native gate.
    pins = copy.deepcopy(module.SOURCE_PROFILES)
    for label, source in pins.items():
        for name in source['files']:
            source['files'][name] = hashlib.sha256((label + name).encode()).hexdigest()
    monkeypatch.setattr(module, 'SOURCE_PROFILES', pins)
    # Keep the legacy SO constant tied to its own source identity.
    monkeypatch.setattr(module, 'CRITICAL_FILES', {
        name: pins['0.6.2']['files'][name] for name in module.CRITICAL_FILES})
    selected = module._critical_files(version, kind)
    for name in selected:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((version + name).encode())
    matched = verify_consumer_source(tmp_path, robot_type=kind)
    assert matched['matches_tested_profile'] is True and matched['profile_version'] == version
    assert len(matched['critical_files']) == (7 if kind == 'bi_so_follower' else 5)
    opposite = '0.6.1' if version == '0.6.2' else '0.6.2'
    context = consumer_version_context(opposite, deployment={'robot_type': kind, 'program': 'rollout'}, source_root=tmp_path)
    assert context['comparison_allowed'] is True and context['profile_version'] == version
    (tmp_path / 'motors/motors_bus.py').write_bytes(b'changed source')
    assert verify_consumer_source(tmp_path, robot_type=kind)['matches_tested_profile'] is False
    assert consumer_version_context(version, deployment={'robot_type': kind}, source_root=tmp_path)['comparison_allowed'] is False


def test_required_unversioned_record_role_is_a_conditional_note_only():
    context = consumer_version_context(deployment={'robot_type': 'so101_follower', 'program': 'record'}, require_version=True)
    assert context['command_role'] == 'not_established'
    assert context['conditional_command_role'] == 'collection_only'
    assert context['comparison_allowed'] is False
    assert 'declare the actual version' in context['command_role_note']


@pytest.mark.parametrize('kind', ['so101_follower', 'lekiwi'])
def test_required_unversioned_camera_default_cannot_approve_even_matching_names(tmp_path, kind):
    root = dataset(tmp_path, ('front', 'wrist'))
    result = check_declared_camera_keys(root, robot_command=BASE.replace('so101_follower', kind), require_version=True)
    assert result['status'] == 'unknown' and result['compared'] is False
    assert result['coverage']['default_camera_declaration'] is False
    assert result['findings'][0]['code'] == 'camera.default_unqualified'


def test_explicit_camera_identities_remain_independent_of_a_missing_units_version(tmp_path):
    command = BASE + " --robot.cameras='{up: {type: opencv, index_or_path: 0}, side: {type: opencv, index_or_path: 1}}'"
    result = check_declared_camera_keys(dataset(tmp_path), robot_command=command, require_version=True)
    assert result['status'] == 'pass' and result['compared'] is True
    assert result['coverage']['explicit_camera_declaration'] is True
    assert result['coverage']['default_camera_declaration'] is False


@pytest.mark.parametrize('value', ['true', 1, None])
def test_camera_require_version_is_a_strict_boolean(tmp_path, value):
    with pytest.raises(InputError, match='boolean'):
        check_declared_camera_keys(dataset(tmp_path), robot_command=BASE, require_version=value)


@pytest.mark.parametrize('version', ['0.6.1', '0.6.2'])
@pytest.mark.parametrize('kind', ['so100_follower', 'so101_follower', 'koch_follower', 'omx_follower', 'bi_so_follower', 'lekiwi'])
def test_qualified_profile_detects_and_clears_the_same_actual_units_comparison(tmp_path, version, kind):
    # Generated encoder-grid control, not a public incident or hardware test.
    pytest.importorskip('pyarrow')
    import numpy as np
    from test_units_manifest import arm, write

    from policy_doctor.units_manifest import units_manifest

    degrees = kind in ('koch_follower', 'omx_follower')
    mode = 'deg' if degrees else 'm100'
    if kind == 'bi_so_follower':
        left, right = arm(400, mode, 'left_', 1), arm(400, mode, 'right_', 2)
        state = (left[0] + right[0], np.hstack([left[1], right[1]]))
        command = 'lerobot-rollout --robot.type=bi_so_follower --robot.left_arm_config.port=/not-opened --robot.right_arm_config.port=/not-opened'
        correction = ' --robot.left_arm_config.use_degrees=false --robot.right_arm_config.use_degrees=false'
    else:
        state = arm(400, mode, 'arm_' if kind == 'lekiwi' else '', 1)
        command = BASE.replace('so101_follower', kind)
        correction = ' --robot.use_degrees=' + str(degrees).lower()
    root = write(tmp_path, {'observation.state': state}, robot_type=kind)
    bad = apply_consumer_version(units_manifest(root, robot_command=command), version, require_version=True)
    assert bad['status'] == 'error'
    assert any(f['code'] == 'units.joint_mode_mismatch' for f in bad['findings'])
    validate_robot_declaration(robot_command=command + correction)
    good = apply_consumer_version(units_manifest(root, robot_command=command + correction), version, require_version=True)
    assert good['status'] == 'pass'
    assert any(f['code'] == 'units.joint_mode_match' for f in good['findings'])
    assert good['consumer_version']['comparison_allowed'] is True
