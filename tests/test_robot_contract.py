"""Strict supplied declarations and key-only camera comparisons; no native robot execution."""
from __future__ import annotations

import hashlib
import json
import shlex

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.robot_contract import check_declared_camera_keys, validate_robot_declaration
from policy_doctor.robot_interface import from_command

BASE = 'lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0'


def dataset(tmp_path, names=('up', 'side')):
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'fps': 30,
        'features': {**{'observation.images.' + name: {'dtype': 'video', 'shape': [480, 640, 3]} for name in names},
                     'observation.state': {'dtype': 'float32', 'shape': [6]}}}))
    return root


def cameras(names, camera_type='opencv', **extras):
    return {name: {'type': camera_type, 'index_or_path': index, 'width': 640, 'height': 480, 'fps': 30, **extras}
            for index, name in enumerate(names)}


def command(names=None, **extras):
    result = BASE + ' --robot.use_degrees=false'
    if names is not None:
        result += " --robot.cameras='" + json.dumps(cameras(names, **extras)) + "'"
    return result


@pytest.mark.parametrize('key', ['use_degres', 'usedegrees', 'use_degree', 'use_degress', 'disble_torque_on_disconnect', 'poert'])
@pytest.mark.parametrize('value', ['false', '$UNRESOLVED'])
def test_misspelled_robot_options_cannot_disappear_before_validation(key, value):
    with pytest.raises(InputError, match=key):
        validate_robot_declaration(robot_command=BASE + ' --robot.' + key + '=' + value)


@pytest.mark.parametrize('key', ['use_degres', 'usedegrees'])
@pytest.mark.parametrize('arm', ['left', 'right'])
def test_nested_bimanual_typo_is_not_a_default(arm, key):
    text = 'lerobot-record --robot.type=bi_so_follower --robot.' + arm + '_arm_config.' + key + '=false'
    with pytest.raises(InputError, match=key):
        validate_robot_declaration(robot_command=text)


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('field', ['use_degrees', 'port'])
def test_unresolved_nested_scalar_option_cannot_disappear_into_arm_defaults(arm, field):
    text = 'lerobot-record --robot.type=bi_so_follower --robot.' + arm + '_arm_config.' + field + '.typo=$UNRESOLVED'
    with pytest.raises(InputError, match='nested.*typo'):
        validate_robot_declaration(robot_command=text)


@pytest.mark.parametrize('flag', ['robot.cameras.front.width', 'robot.max_relative_target.shoulder_pan',
                                 'robot.left_arm_config.cameras.front.width'])
def test_dictionary_fields_use_the_registered_whole_mapping_cli_option(flag):
    kind = 'bi_so_follower' if 'left_arm_config' in flag else 'so101_follower'
    with pytest.raises(InputError, match='nested'):
        validate_robot_declaration(robot_command='lerobot-record --robot.type=' + kind + ' --' + flag + '=640')


def test_unresolved_valid_robot_value_is_named_unknown_not_validated_as_a_default():
    result = validate_robot_declaration(robot_command=BASE + ' --robot.use_degrees=$MODE')
    assert result['declaration']['status'] == 'unknown'
    assert any('robot.use_degrees' in reason for reason in result['declaration']['unknown'])


@pytest.mark.parametrize('value', ['flase', 'tru', '1', 'null', '{typo: false}'])
def test_unit_flag_type_errors_are_invalid_input(value):
    with pytest.raises(InputError, match='boolean|mapping'):
        validate_robot_declaration(robot_command=BASE + ' --robot.use_degrees=' + shlex.quote(value))


@pytest.mark.parametrize('suffix', ['json', 'yaml'])
def test_config_typos_receive_the_same_validation(tmp_path, suffix):
    path = tmp_path / ('robot.' + suffix)
    config = {'robot': {'type': 'so101_follower', 'use_degres': False}}
    if suffix == 'json':
        path.write_text(json.dumps(config))
    else:
        import yaml
        path.write_text(yaml.safe_dump(config))
    with pytest.raises(InputError, match='use_degres'):
        validate_robot_declaration(robot_config=path)


def test_valid_native_settings_and_legacy_parser_compatibility():
    result = validate_robot_declaration(robot_command=BASE + ' --robot.use_degrees=false --robot.id=arm '
        '--robot.calibration_dir=/not-opened --robot.disable_torque_on_disconnect=true '
        '--robot.max_relative_target=15 --robot.position_p_coefficient=16 --robot.position_i_coefficient=0 '
        '--robot.position_d_coefficient=32 --robot.num_read_retries=2')
    assert result['declaration']['status'] == 'pass' and result['flags']['use_degrees'] is False
    # The legacy source-profile parser remains explicitly distinct from the public strict helper.
    assert from_command(BASE + ' --robot.use_degres=false')['flags']['use_degrees'] is True


def test_scalar_robot_field_cannot_hide_nested_unknown_flags():
    with pytest.raises(InputError, match='nested|mapping'):
        validate_robot_declaration(robot_command='lerobot-record --robot.type=so101_follower --robot.port.typo=x')


def test_duplicate_command_settings_and_flow_keys_are_refused():
    with pytest.raises(InputError, match='Repeated'):
        validate_robot_declaration(robot_command=BASE + ' --robot.use_degrees=true --robot.use_degrees=false')
    with pytest.raises(InputError, match='parse'):
        validate_robot_declaration(robot_command=BASE + " --robot.cameras='{front: {type: opencv, type: intelrealsense}}'")


def test_unmodeled_robot_and_camera_profiles_never_get_camera_pass(tmp_path):
    root = dataset(tmp_path)
    unknown = validate_robot_declaration(robot_command='lerobot-record --robot.type=custom_robot --robot.any_new_field=1')
    assert unknown['declaration']['status'] == 'unknown'
    custom = 'lerobot-record --robot.type=custom_robot --robot.cameras=\'{"up":{"type":"plugin"}}\''
    report = check_declared_camera_keys(root, robot_command=custom)
    assert report['requested'] is True and report['status'] == 'unknown' and report['compared'] is False
    plugin = BASE + ' --robot.cameras=\'{"up":{"type":"plugin"},"side":{"type":"plugin"}}\''
    assert check_declared_camera_keys(root, robot_command=plugin)['status'] == 'unknown'


def test_camera_key_fault_correction_same_metadata_and_scope_without_policy(tmp_path):
    root = dataset(tmp_path)
    info = root / 'meta/info.json'
    before = hashlib.sha256(info.read_bytes()).hexdigest()
    wrong = check_declared_camera_keys(root, robot_command=command(['front']))
    right = check_declared_camera_keys(root, robot_command=command(['up', 'side']))
    assert wrong['status'] == 'error' and wrong['compared'] is True and wrong['requested'] is True
    assert wrong['missing'] == ['observation.images.side', 'observation.images.up']
    assert wrong['extra'] == ['observation.images.front']
    assert right['status'] == 'pass' and right['compared'] is True and right['missing'] == []
    assert right['dataset_keys'] == wrong['dataset_keys']
    assert right['not_checked'] == ['image size', 'frame rate', 'joint order', 'physical calibration', 'policy compatibility']
    assert right['findings'][0]['evidence'][0]['sha256'] == before
    assert hashlib.sha256(info.read_bytes()).hexdigest() == before
    assert 'not certify' in right['scope']


def test_known_camera_defaults_are_compared_but_unqualified_client_defaults_stay_unknown(tmp_path):
    root = dataset(tmp_path, ('front', 'wrist'))
    result = check_declared_camera_keys(root, robot_command=BASE)
    assert result['status'] == 'warning' and result['requested'] is True and result['compared'] is True
    assert result['findings'][0]['code'] == 'camera.default_empty'
    result = check_declared_camera_keys(root, robot_command='lerobot-record --robot.type=lekiwi')
    assert result['status'] == 'pass' and result['requested'] is True and result['compared'] is True
    result = check_declared_camera_keys(root, robot_command='lerobot-record --robot.type=lekiwi_client')
    assert result['status'] == 'unknown' and result['requested'] is False and result['compared'] is False
    assert result['findings'][0]['code'] == 'camera.keys_not_requested'


def test_explicit_empty_and_additional_camera_keys_are_distinct(tmp_path):
    root = dataset(tmp_path)
    empty = check_declared_camera_keys(root, robot_command=command([]))
    extra = check_declared_camera_keys(root, robot_command=command(['up', 'side', 'extra']))
    assert empty['requested'] is True and empty['status'] == 'warning'
    assert extra['status'] == 'warning' and not extra['missing'] and extra['extra'] == ['observation.images.extra']


def test_intentional_partial_view_selection_is_reviewable_not_a_pipeline_error(tmp_path):
    root = dataset(tmp_path)
    info = root / 'meta/info.json'
    before = hashlib.sha256(info.read_bytes()).hexdigest()
    result = check_declared_camera_keys(root, robot_command=command(['up']))
    assert result['status'] == 'warning' and result['requested'] is True and result['compared'] is True
    assert result['missing'] == ['observation.images.side']
    assert result['findings'][0]['code'] == 'camera.partial_keys'
    assert 'may be intentional' in result['findings'][0]['message']
    assert result['counts']['error'] == 0 and hashlib.sha256(info.read_bytes()).hexdigest() == before


def test_camera_config_and_bimanual_names_are_compared(tmp_path):
    root = dataset(tmp_path, ('left_up', 'right_side'))
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps({'robot': {'type': 'bi_so_follower',
        'left_arm_config': {'port': '/unused', 'cameras': cameras(['up'])},
        'right_arm_config': {'port': '/unused', 'cameras': cameras(['side'])}}}))
    result = check_declared_camera_keys(root, robot_config=path)
    assert result['status'] == 'pass' and result['effective_declared_keys'] == result['dataset_keys']


def test_native_bimanual_camera_name_collision_does_not_create_a_predicted_pass(tmp_path):
    root = dataset(tmp_path, ('front', 'left_front'))
    path = tmp_path / 'collision.json'
    path.write_text(json.dumps({'robot': {'type': 'bi_so_follower', 'cameras': cameras(['front']),
        'left_arm_config': {'port': '/unused', 'cameras': cameras(['front'])}}}))
    with pytest.raises(InputError, match='collide'):
        check_declared_camera_keys(root, robot_config=path)


def test_explicit_rename_map_direction_is_raw_robot_to_dataset_keys(tmp_path):
    root = dataset(tmp_path)
    mapping = {'observation.images.front': 'observation.images.up', 'observation.images.wrist': 'observation.images.side'}
    result = check_declared_camera_keys(root, robot_command=command(['front', 'wrist']) + " --rename_map='" + json.dumps(mapping) + "'")
    assert result['status'] == 'pass' and result['declared_keys'] != result['effective_declared_keys']
    assert result['effective_declared_keys'] == result['dataset_keys']
    unsupported = check_declared_camera_keys(root, robot_command=command(['up', 'side']) + " --rename_map='" + json.dumps(mapping) + "'")
    assert unsupported['status'] == 'unknown'
    with pytest.raises(InputError):
        check_declared_camera_keys(root, robot_command=command(['up', 'side']) + " --rename_map='{bad: {nested: value}}'")


def test_camera_key_check_does_not_claim_size_or_rate_comparison(tmp_path):
    root = dataset(tmp_path)
    mapping = cameras(['up', 'side'])
    for camera in mapping.values():
        camera['width'] = 1280
        camera['height'] = 720
        camera['fps'] = 10
    result = check_declared_camera_keys(root, robot_command=BASE + " --robot.cameras='" + json.dumps(mapping) + "'")
    assert result['status'] == 'pass' and 'image size' in result['not_checked'] and 'frame rate' in result['not_checked']


@pytest.mark.parametrize('bad', ['{front: {type: opencv', 'null', '[]', '{front: 1}'])
def test_explicit_malformed_camera_values_never_get_pass(tmp_path, bad):
    root = dataset(tmp_path)
    with pytest.raises(InputError):
        check_declared_camera_keys(root, robot_command=BASE + " --robot.cameras='" + bad + "'")


def test_nested_camera_field_typo_is_invalid_not_dropped(tmp_path):
    root = dataset(tmp_path)
    with pytest.raises(InputError, match='widht'):
        check_declared_camera_keys(root, robot_command=BASE + " --robot.cameras='{front: {type: opencv, widht: 640}}'")


def test_read_roots_and_symlinks_apply_to_config_and_metadata(tmp_path):
    root = dataset(tmp_path)
    other = tmp_path / 'other'
    other.mkdir()
    config = other / 'robot.json'
    config.write_text(json.dumps({'robot': {'type': 'so101_follower', 'cameras': cameras(['up', 'side'])}}))
    with pytest.raises(InputError, match='read roots'):
        check_declared_camera_keys(root, robot_config=config, read_roots=[root])
    with pytest.raises(InputError, match='read roots'):
        check_declared_camera_keys(root, robot_command=command(['up', 'side']), read_roots=[other])
    alias = tmp_path / 'link'
    alias.symlink_to(root)
    with pytest.raises(InputError, match='Symlink'):
        check_declared_camera_keys(alias, robot_command=command(['up', 'side']))


def test_hub_input_does_not_fetch_or_infer_camera_names():
    result = check_declared_camera_keys('hf://some/dataset', robot_command=command(['up', 'side']))
    assert result['status'] == 'unknown' and result['requested'] is True and result['compared'] is False
    assert 'materialized local' in result['findings'][0]['message']


def test_yaml_aliases_duplicates_and_recursive_declarations_are_refused(tmp_path):
    path = tmp_path / 'robot.yaml'
    for raw in ['robot: {type: so101_follower, use_degrees: true, use_degrees: false}',
                'robot: &robot {type: so101_follower, cameras: *robot}',
                'robot: {type: so101_follower}\ncopy: &a [1]\nother: *a',
                'robot: {? [type]: so101_follower}']:
        path.write_text(raw)
        with pytest.raises(InputError):
            validate_robot_declaration(robot_config=path)


def test_canonical_check_rejects_typo_before_units_default_can_clear(tmp_path, capsys):
    from policy_doctor.product_cli import main

    root = dataset(tmp_path)
    for key in ['use_degres', 'usedegrees']:
        result = main(['check', str(root), '--robot-command', BASE + ' --robot.' + key + '=false',
                       '--require', 'deployment-units', '--json'])
        assert result == 2
        output = capsys.readouterr()
        assert key in output.out + output.err


def test_canonical_command_camera_fault_correction_uses_same_dataset_without_policy(tmp_path, capsys):
    from test_units_manifest import so_dataset

    from policy_doctor.product_cli import main

    root = so_dataset(tmp_path, 'm100')
    info = root / 'meta/info.json'
    document = json.loads(info.read_text())
    document['features'].update({'observation.images.' + key: {'dtype': 'video', 'shape': [480, 640, 3]}
                                 for key in ['up', 'side']})
    info.write_text(json.dumps(document))
    before = hashlib.sha256(info.read_bytes()).hexdigest()
    wrong = main(['check', str(root), '--robot-command', command(['front']), '--lerobot-version', '0.6.2'])
    wrong_text = capsys.readouterr().out
    right = main(['check', str(root), '--robot-command', command(['up', 'side']), '--lerobot-version', '0.6.2'])
    right_text = capsys.readouterr().out
    assert wrong == 1, wrong_text
    assert 'camera' in wrong_text.lower() and 'up' in wrong_text and 'side' in wrong_text
    assert right == 0, right_text
    assert 'camera' in right_text.lower()
    assert hashlib.sha256(info.read_bytes()).hexdigest() == before
