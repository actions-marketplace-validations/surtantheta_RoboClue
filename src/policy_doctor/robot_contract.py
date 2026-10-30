"""Strict public robot declarations and a dataset/declared-camera key comparison.

These checks read supported declarations only. They do not execute configuration
code, load a policy, contact a robot, or establish native pipeline compatibility.
Legacy robot_interface parsing stays opt-in compatible; public paths use strict.
"""
from __future__ import annotations

import difflib
import hashlib
import json

from .artifacts import LEROBOT_REVISION, InputError, finite_tree
from .framework_versions import _regular_bytes

MAX_CONFIG_BYTES = 256 * 1024
MAX_INFO_BYTES = 4 * 1024 * 1024
MAX_CAMERA_KEYS = 256
MAX_TREE_NODES = 16384
PROFILE = 'LeRobot ' + LEROBOT_REVISION[:7]
BASE_FIELDS = {'type', 'id', 'calibration_dir'}
ARM_FIELDS = {'port', 'disable_torque_on_disconnect', 'max_relative_target', 'cameras', 'use_degrees'}
SO_FIELDS = ARM_FIELDS | {'position_p_coefficient', 'position_i_coefficient', 'position_d_coefficient', 'num_read_retries'}
ROBOT_FIELDS = {
    'so100_follower': BASE_FIELDS | SO_FIELDS,
    'so101_follower': BASE_FIELDS | SO_FIELDS,
    'koch_follower': BASE_FIELDS | ARM_FIELDS,
    'omx_follower': BASE_FIELDS | ARM_FIELDS,
    'lekiwi': BASE_FIELDS | ARM_FIELDS | {'num_read_retries'},
    'lekiwi_client': BASE_FIELDS | {'remote_ip', 'port_zmq_cmd', 'port_zmq_observations', 'teleop_keys', 'cameras', 'polling_timeout_ms', 'connect_timeout_s'},
    'bi_so_follower': BASE_FIELDS | {'left_arm_config', 'right_arm_config', 'cameras'},
}
CAMERA_BASE = {'type', 'fps', 'width', 'height'}
CAMERA_FIELDS = {
    'opencv': CAMERA_BASE | {'index_or_path', 'color_mode', 'rotation', 'warmup_s', 'fourcc', 'backend'},
    'intelrealsense': CAMERA_BASE | {'serial_number_or_name', 'color_mode', 'use_rgb', 'use_depth', 'rotation', 'warmup_s', 'exposure', 'gain', 'white_balance'},
}
SOURCE_PATHS = ['src/lerobot/robots/config.py', 'src/lerobot/robots/so_follower/config_so_follower.py',
                'src/lerobot/robots/koch_follower/config_koch_follower.py', 'src/lerobot/robots/omx_follower/config_omx_follower.py',
                'src/lerobot/robots/lekiwi/config_lekiwi.py', 'src/lerobot/robots/bi_so_follower/config_bi_so_follower.py',
                'src/lerobot/cameras/configs.py', 'src/lerobot/cameras/opencv/configuration_opencv.py',
                'src/lerobot/cameras/realsense/configuration_realsense.py']
NOT_CHECKED = ['image size', 'frame rate', 'joint order', 'physical calibration', 'policy compatibility']
SCOPE = ('Camera feature-key declarations only, after an explicitly supplied rename_map when supported. '
         'No policy/checkpoint, image size, frame rate, joint order, decoded images or physical calibration was compared. '
         'A key match does not certify native training, deployment or task success.')


def _safe_local(path, read_roots=None):
    from .product_cli import _safe_path

    resolved = _safe_path(path)
    if read_roots is not None:
        roots = [_safe_path(root) for root in read_roots]
        if not roots or not any(resolved.is_relative_to(root) for root in roots):
            raise InputError('Robot/camera declaration input is outside the configured read roots.')
    return resolved


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or not key or not all(c.isprintable() for c in key):
            raise InputError('Declaration keys must be nonempty printable strings.')
        if key in result:
            raise InputError('Duplicate declaration key ' + str(key) + ' is ambiguous.')
        result[key] = value
    return result


def _bounded_tree(value):
    seen, count = set(), 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if count > MAX_TREE_NODES or depth > 16:
            raise InputError('Robot declaration is too large or deeply nested.')
        if isinstance(node, (dict, list)):
            if id(node) in seen:
                raise InputError('YAML aliases and recursive robot declarations are not supported.')
            seen.add(id(node))
            if isinstance(node, dict):
                for key, child in node.items():
                    if not isinstance(key, str) or not key or not all(c.isprintable() for c in key):
                        raise InputError('Robot/configuration keys must be nonempty printable strings.')
                    visit(child, depth + 1)
            else:
                for child in node:
                    visit(child, depth + 1)
    visit(value, 0)


def read_robot_config(path):
    """One bounded, no-follow read and source hash for strict JSON/YAML configs."""
    path = _safe_local(path)
    try:
        raw = _regular_bytes(path, MAX_CONFIG_BYTES)
        if raw is None:
            raise InputError('Robot configuration file was not found.')
        if path.suffix.lower() == '.json':
            config = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
        else:
            try:
                import yaml
            except ImportError as error:
                raise InputError('YAML robot configuration needs PyYAML; install roboclue[dataset].') from error
            class UniqueLoader(yaml.SafeLoader):
                pass
            def unique_mapping(loader, node, deep=False):
                loader.flatten_mapping(node)
                return _pairs([(loader.construct_object(key, deep=deep), loader.construct_object(value, deep=deep)) for key, value in node.value])
            UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
            try:
                config = yaml.load(raw.decode('utf-8'), Loader=UniqueLoader)
            except yaml.YAMLError as error:
                raise InputError('Invalid YAML robot configuration.') from error
        if not isinstance(config, dict):
            raise InputError('A robot rollout/record configuration must be a mapping.')
        if 'robot' not in config and config.get('type') in ROBOT_FIELDS:
            raise InputError('This file is a flat robot configuration. Put these settings under a top-level "robot" mapping, or provide the full rollout/record configuration.')
        _bounded_tree(config)
        finite_tree(config)
    except InputError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise InputError('Robot configuration is unreadable, malformed, linked or outside supported bounds.') from error
    return config, {'kind': 'config', 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def _unknown_keys(value, allowed, prefix):
    for key in value:
        if key not in allowed:
            suggestion = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=.55)
            hint = ' Did you mean --' + prefix + '.' + suggestion[0] + '?' if suggestion else ''
            raise InputError('Unknown or unsupported robot setting --' + prefix + '.' + key + ' for the ' + PROFILE + ' declaration profile.' + hint)


def _camera_maps(value, prefix, unknown):
    if not isinstance(value, dict):
        raise InputError('--' + prefix + ' must be a camera-name mapping; it cannot be null or a scalar.')
    if len(value) > MAX_CAMERA_KEYS:
        unknown.append(prefix + ' exceeds the supported camera-key budget.')
        return
    for name, camera in value.items():
        if not isinstance(name, str) or not name or len(name) > 256 or not all(c.isprintable() for c in name):
            raise InputError('Camera names must be bounded nonempty printable strings.')
        path = prefix + '.' + name
        if not isinstance(camera, dict):
            raise InputError('--' + path + ' must be a camera configuration mapping.')
        camera_type = camera.get('type')
        if not isinstance(camera_type, str) or camera_type not in CAMERA_FIELDS:
            unknown.append(path + ': camera type is missing or outside the supported declaration profile; output keys are not established.')
            continue
        _unknown_keys(camera, CAMERA_FIELDS[camera_type], path)
        for flag in ('use_rgb', 'use_depth'):
            if flag in camera and type(camera[flag]) is not bool:
                raise InputError('--' + path + '.' + flag + ' must be a boolean, not a misspelled or numeric value.')


def validate_robot_flag_names(config, keys):
    """Validate supplied names even when shell-dependent values were not parsed."""
    robot = config.get('robot')
    kind = robot.get('type') if isinstance(robot, dict) else None
    if not isinstance(kind, str) or kind not in ROBOT_FIELDS:
        return
    for key in keys:
        if key == 'robot':
            continue
        parts = key.split('.')
        if len(parts) < 2:
            continue
        _unknown_keys({parts[1]: None}, ROBOT_FIELDS[kind], 'robot')
        if parts[1] in {'left_arm_config', 'right_arm_config'} and len(parts) > 2:
            _unknown_keys({parts[2]: None}, SO_FIELDS, 'robot.' + parts[1])
            if len(parts) > 3:
                raise InputError('Unknown nested robot setting --' + key + '; this arm field is not a configuration mapping.')
        elif len(parts) > 2:
            raise InputError('Unknown nested robot setting --' + key + '; supply this native field as one correctly quoted value or mapping.')


def validate_robot_config(config):
    """Validate modeled robot field names and unit booleans, not every native value."""
    _bounded_tree(config)
    if 'rename_map' in config:
        from .mapping import validate_rename_map

        validate_rename_map(config['rename_map'])
    robot = config.get('robot')
    unknown = []
    if not isinstance(robot, dict):
        if 'robot' in config:
            raise InputError('robot must be a configuration mapping.')
        return {'status': 'unknown', 'unknown': ['No robot declaration was supplied.'], 'explicit_camera_declaration': False}
    robot_type = robot.get('type')
    requested = 'cameras' in robot
    if not isinstance(robot_type, str) or robot_type not in ROBOT_FIELDS:
        return {'status': 'unknown', 'unknown': ['Robot type is missing or outside the supported declaration profile; robot keys were not validated.'],
                'explicit_camera_declaration': requested}
    _unknown_keys(robot, ROBOT_FIELDS[robot_type], 'robot')
    arms = [('robot', robot)]
    if robot_type == 'bi_so_follower':
        for side in ('left', 'right'):
            key = side + '_arm_config'
            if key not in robot:
                continue
            arm = robot[key]
            if not isinstance(arm, dict):
                raise InputError('--robot.' + key + ' must be an arm configuration mapping.')
            _unknown_keys(arm, SO_FIELDS, 'robot.' + key)
            requested = requested or 'cameras' in arm
            arms.append(('robot.' + key, arm))
    for prefix, arm in arms:
        for field, value in arm.items():
            if isinstance(value, dict) and field not in {'cameras', 'left_arm_config', 'right_arm_config', 'max_relative_target', 'teleop_keys'}:
                raise InputError('--' + prefix + '.' + field + ' is not a configuration mapping; nested settings are unsupported.')
        for flag in ('use_degrees', 'disable_torque_on_disconnect'):
            if flag in arm and type(arm[flag]) is not bool:
                raise InputError('--' + prefix + '.' + flag + ' must be a boolean, not a misspelled or numeric value.')
        if 'cameras' in arm:
            _camera_maps(arm['cameras'], prefix + '.cameras', unknown)
    if robot_type == 'bi_so_follower':
        top = set(robot.get('cameras', {}))
        collisions = top.intersection(name for _, arm in arms[1:] for name in arm.get('cameras', {}))
        if collisions:
            raise InputError('Top-level camera names collide with per-arm camera names in the supported bimanual profile: ' + ', '.join(sorted(collisions)))
    return {'status': 'unknown' if unknown else 'pass', 'unknown': unknown,
            'explicit_camera_declaration': requested,
            'profile': PROFILE, 'source_revision': LEROBOT_REVISION,
            'scope': 'Supported robot/camera field names and unit/stream booleans only; other native value constraints and runtime behavior are not certified.'}


def declaration_details(config, interface):
    declaration = validate_robot_config(config)
    unresolved = interface.get('unresolved', [])
    camera_unresolved = any(key == 'robot' or key.startswith('robot.') and (key.endswith('.cameras') or '.cameras.' in key) for key in unresolved)
    declaration['explicit_camera_declaration'] = declaration['explicit_camera_declaration'] or camera_unresolved
    declaration['camera_declaration_complete'] = declaration['status'] == 'pass' and not camera_unresolved and not interface.get('source', {}).get('program_index')
    if camera_unresolved:
        declaration['unknown'].append('An explicitly supplied camera declaration could not be resolved; camera keys were not compared.')
    if interface.get('source', {}).get('program_index'):
        declaration['unknown'].append('Multiple robot commands were supplied; only the first declaration was read.')
    unresolved_robot = [key for key in unresolved if key == 'robot' or key.startswith('robot.')]
    if unresolved_robot:
        declaration['status'] = 'unknown'
        declaration['unknown'].append('Robot declaration values remain unresolved: ' + ', '.join(unresolved_robot[:8]))
    return declaration


def validate_robot_declaration(*, robot_command=None, robot_config=None, read_roots=None):
    """Strict public selector validation; legacy parser behavior is not changed."""
    from . import robot_interface

    if (robot_command is None) == (robot_config is None):
        raise InputError('Supply exactly one robot command or robot config for the declared consumer.')
    if robot_command is not None:
        return robot_interface.from_command(robot_command, strict=True)
    path = _safe_local(robot_config, read_roots)
    return robot_interface.from_config(path, strict=True)


def check_declared_camera_keys(source, *, robot_command=None, robot_config=None, read_roots=None,
                               lerobot_version=None, require_version=False):
    """Compare local camera keys with declarations or qualified native defaults."""
    if type(require_version) is not bool:
        raise InputError('require_version must be a boolean.')
    interface = validate_robot_declaration(robot_command=robot_command, robot_config=robot_config, read_roots=read_roots)
    declaration = interface['declaration']
    explicit = declaration['explicit_camera_declaration']
    default_cameras = False
    default_unqualified = False
    from .consumer_version import ROBOT_TYPES, consumer_version_context

    context = None
    if not explicit and interface.get('robot_type') in ROBOT_TYPES:
        context = consumer_version_context(lerobot_version, require_version=require_version, deployment={
            'robot_type': interface['robot_type'], 'program': interface.get('program')})
        default_cameras = context['default_inference_allowed']
        default_unqualified = not default_cameras
    requested = explicit or default_cameras or default_unqualified
    result = {'schema_version': 1, 'kind': 'camera_key_comparison', 'status': 'unknown', 'requested': requested,
              'compared': False, 'dataset_keys': [], 'declared_keys': [], 'effective_declared_keys': [], 'missing': [], 'extra': [],
              'findings': [], 'counts': {}, 'coverage': {'dataset_declarations_read': False, 'robot_camera_declaration_read': False,
              'explicit_camera_declaration': explicit, 'default_camera_declaration': default_cameras},
              'scope': SCOPE, 'not_checked': list(NOT_CHECKED), 'robot_source': interface['source']}
    def finish(code, status, message, evidence=(), remedy=''):
        result['findings'].append({'code': code, 'status': status, 'message': message, 'evidence': list(evidence), 'remedy': remedy})
        result['status'] = status
        result['counts'] = {state: sum(f['status'] == state for f in result['findings']) for state in ('error', 'warning', 'unknown', 'pass')}
        return result
    if not requested:
        return finish('camera.keys_not_requested', 'unknown', 'No camera configuration was explicitly supplied; dataset and robot cameras were not compared.',
                      remedy='Supply the actual --robot.cameras mapping or its config equivalent to compare camera names.')
    if default_unqualified:
        return finish('camera.default_unqualified', 'unknown',
                      'No cameras were supplied, and no tested camera default was established for the declared LeRobot version.',
                      remedy='Declare the actual camera configuration and a supported consumer version; an unknown default is not an empty or matching camera set.')
    if not declaration.get('camera_declaration_complete'):
        return finish('camera.declaration_unassessed', 'unknown', 'Explicit camera output keys are unresolved: ' + '; '.join(declaration['unknown']),
                      remedy='Supply a supported explicit camera-name/config mapping; do not infer output keys from an unsupported camera plugin.')
    if str(source).startswith('hf://'):
        return finish('camera.dataset_unassessed', 'unknown', 'Camera-key comparison needs a materialized local dataset metadata file; no Hub or policy was fetched.')
    path = _safe_local(source, read_roots)
    if path.is_dir():
        info = _safe_local(path / 'meta/info.json', read_roots)
    elif path.name == 'info.json' and path.parent.name == 'meta':
        info = path
    else:
        raise InputError('Camera comparison needs a dataset directory or its meta/info.json.')
    try:
        raw = _regular_bytes(info, MAX_INFO_BYTES)
        if raw is None:
            raise InputError('Dataset meta/info.json was not found.')
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
        finite_tree(document)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        raise InputError('Dataset camera metadata is unreadable, malformed or outside supported bounds.') from error
    features = document.get('features') if isinstance(document, dict) else None
    if not isinstance(features, dict):
        return finish('camera.dataset_unassessed', 'unknown', 'Dataset metadata has no supported feature-key mapping.')
    keys = sorted(key for key in features if isinstance(key, str) and key.startswith('observation.images.'))
    result['dataset_keys'] = keys[:MAX_CAMERA_KEYS]
    result['coverage']['dataset_camera_keys_total'] = len(keys)
    result['coverage']['dataset_keys_truncated'] = len(keys) > MAX_CAMERA_KEYS
    result['coverage']['dataset_declarations_read'] = True
    if len(keys) > MAX_CAMERA_KEYS:
        return finish('camera.coverage_unknown', 'unknown', 'Dataset camera declarations exceed the bounded key budget; the complete set was not compared.')
    declared = sorted(key for key in interface['features'] if key.startswith('observation.images.'))
    if default_cameras and declared != sorted('observation.images.' + key for key in context['camera_default']['keys']):
        return finish('camera.default_unqualified', 'unknown',
                      'The parsed camera defaults do not match the tested source profile; no key comparison was approved.',
                      remedy='Supply the actual camera mapping and review the consumer source before comparing names.')
    result['declared_keys'] = declared[:MAX_CAMERA_KEYS]
    result['coverage']['declared_camera_keys_total'] = len(declared)
    result['coverage']['declared_keys_truncated'] = len(declared) > MAX_CAMERA_KEYS
    result['coverage']['robot_camera_declaration_read'] = True
    if len(declared) > MAX_CAMERA_KEYS:
        return finish('camera.coverage_unknown', 'unknown', 'Robot camera declarations exceed the bounded key budget; the complete set was not compared.')
    mapping = interface.get('rename_map') or {}
    if 'rename_map' in interface.get('unresolved', []):
        return finish('camera.rename_unassessed', 'unknown', 'The explicitly supplied camera rename_map was not resolved.')
    unused = [key for key in mapping if key.startswith('observation.images.') and key not in declared]
    if unused:
        return finish('camera.rename_unassessed', 'unknown', 'Camera rename_map names source keys absent from the declared robot: ' + ', '.join(unused[:8]))
    effective = [mapping.get(key, key) for key in declared]
    if len(set(effective)) != len(effective):
        raise InputError('Camera rename_map collides with another declared camera feature key.')
    result['effective_declared_keys'] = sorted(effective)
    result['missing'] = sorted(set(keys) - set(effective))
    result['extra'] = sorted(set(effective) - set(keys))
    result['compared'] = True
    evidence = [{'path': str(info), 'sha256': hashlib.sha256(raw).hexdigest(), 'pointer': '/features', 'value': keys},
                {'path': interface['source'].get('path'), 'sha256': interface['source']['sha256'], 'pointer': '/robot/cameras', 'value': declared}]
    if default_cameras:
        result['default_source'] = context['camera_default']
        evidence.append({'path': context['camera_default']['path'],
                         'sha256': context['camera_default']['sha256'],
                         'pointer': 'native_config.cameras',
                         'value': context['camera_default']['keys']})
    if result['missing']:
        if default_cameras and not effective:
            return finish('camera.default_empty', 'warning',
                          'The dataset contains cameras ' + ', '.join(key.removeprefix('observation.images.') for key in result['missing'])
                          + ', but this robot configuration supplies none. Its tested LeRobot default is an empty camera mapping.',
                          evidence, 'Provide the real --robot.cameras mapping with the required dataset names; choose device indexes or paths from the actual robot setup, not from dataset metadata.')
        if effective and not set(keys).intersection(effective):
            return finish('camera.keys_mismatch', 'error', 'Declared camera-key identities have no overlap without a matching explicit rename_map; dataset keys missing from the declared inputs: ' + ', '.join(result['missing']) + '. This is a key-identity mismatch, not proof of native failure; policy view selection was not assessed.',
                          evidence, 'Review the intended camera names and explicit rename_map; recheck the matching owned command/config. Do not rename public dataset features merely to silence this check.')
        return finish('camera.partial_keys', 'warning', 'Dataset camera keys absent from the declared input set: ' + ', '.join(result['missing']) + '. Empty or partial view selection may be intentional; the policy view-selection contract was not inspected.',
                      evidence, 'Confirm the intended policy view selection; compare the actual required keys or review the explicit mapping before changing data.')
    if result['extra']:
        return finish('camera.extra_keys', 'warning', 'The robot declares additional camera keys absent from dataset metadata: ' + ', '.join(result['extra']) + '. Extra views may be intentional; no policy selection was inspected.', evidence)
    return finish('camera.keys_consistent', 'pass', 'Dataset and compared robot camera feature-key sets agree. Image size, rate, joint order, calibration and policy compatibility remain unassessed.', evidence)
