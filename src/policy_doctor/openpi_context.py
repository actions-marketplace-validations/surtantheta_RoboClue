"""Static raw-input contracts from fingerprint-verified OpenPI TrainConfigs.

The result assesses declared dataset keys, not a LeRobot loader or training run.
Source is parsed as data and never imported or evaluated.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from .artifacts import InputError, dataset_root
from .conversion_fidelity import _read_small, _reference_json
from .openpi_interface import MANIFEST, Unsupported, defaults, definition, dotted, source_data

CONFIG = 'src/openpi/training/config.py'
SCOPE = ('Static raw-key contract only. Dataset feature declarations are not a loaded batch. '
         'Loader-version compatibility, task lookup, decoded images, transforms after repacking, '
         'normalization, model targets and physical behavior remain unassessed.')


def _kwargs(call):
    if not isinstance(call, ast.Call) or call.args or any(item.arg is None for item in call.keywords):
        raise Unsupported('The selected constructor has positional or expanded arguments.')
    return {item.arg: item.value for item in call.keywords}


def _factory_defaults(tree, cls, seen=None):
    seen = set() if seen is None else seen
    if cls.name in seen:
        raise Unsupported('Recursive data-config inheritance is unsupported.')
    seen.add(cls.name)
    result = {}
    for base in cls.bases:
        name = dotted(base)
        if name != 'abc.ABC':
            result.update(_factory_defaults(tree, definition(tree, name), seen))
    result.update(defaults(cls))
    return result


def _selected(tree, name):
    configs = [item.value for item in tree.body if isinstance(item, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == '_CONFIGS' for target in item.targets)]
    if len(configs) != 1 or not isinstance(configs[0], (ast.List, ast.Tuple)):
        raise Unsupported('The builtin TrainConfig list is not statically resolved.')
    matches = [item for item in configs[0].elts if isinstance(item, ast.Call)
               and dotted(item.func) == 'TrainConfig' and any(
                   field.arg == 'name' and isinstance(field.value, ast.Constant) and field.value.value == name
                   for field in item.keywords)]
    if len(matches) != 1:
        raise Unsupported('The selected TrainConfig is absent, duplicated or dynamically supplied outside the verified manifest.')
    call = matches[0]
    data = _kwargs(call).get('data')
    if not isinstance(data, ast.Call):
        raise Unsupported('The selected TrainConfig has no literal data-factory constructor.')
    factory = dotted(data.func)
    cls = definition(tree, factory)
    values = _factory_defaults(tree, cls)
    values.update(_kwargs(data))
    return call, cls, values


def _field_default(node):
    if isinstance(node, ast.Call) and dotted(node.func) == 'dataclasses.field':
        fields = _kwargs(node)
        if 'default' not in fields:
            raise Unsupported('A repack default factory is not a literal input-key contract.')
        return fields['default']
    return node


def _group_structure(group):
    if not isinstance(group, ast.Call) or dotted(group.func) != '_transforms.Group':
        raise Unsupported('The repack group is not a literal verified Group.')
    inputs = _kwargs(group).get('inputs')
    if not isinstance(inputs, (ast.List, ast.Tuple)) or len(inputs.elts) != 1:
        raise Unsupported('Only one literal input RepackTransform is modeled; other input-transform sequences remain unknown.')
    repack = inputs.elts[0]
    if (not isinstance(repack, ast.Call) or dotted(repack.func) != '_transforms.RepackTransform'
            or len(repack.args) != 1 or repack.keywords):
        raise Unsupported('The input transform is not a literal RepackTransform structure.')
    try:
        structure = ast.literal_eval(repack.args[0])
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise Unsupported('The RepackTransform mapping is dynamic.') from exc
    leaves = []

    def walk(value, depth=0):
        if depth > 8 or len(leaves) > 256:
            raise Unsupported('The literal RepackTransform exceeds the supported structure bound.')
        if isinstance(value, dict) and value:
            if any(type(key) is not str or not key for key in value):
                raise Unsupported('Repack output names must be nonempty strings.')
            for child in value.values():
                walk(child, depth + 1)
        elif type(value) is str and value:
            leaves.append(value)
        else:
            raise Unsupported('Repack structures require dictionaries with string input-key leaves.')

    walk(structure)
    return sorted(set(leaves)), repack, structure


def _repack(cls, values):
    creates = [item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'create']
    if len(creates) != 1:
        raise Unsupported('The data factory has no unique create method.')
    create = creates[0]
    returns = [item.value for item in ast.walk(create) if isinstance(item, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0], ast.Call) or dotted(returns[0].func) != 'dataclasses.replace':
        raise Unsupported('The data factory return is not a modeled dataclasses.replace.')
    fields = {item.arg: item.value for item in returns[0].keywords if item.arg is not None}
    if any(item.arg is None for item in returns[0].keywords):
        raise Unsupported('Expanded data-factory return fields are unsupported.')
    group = fields.get('repack_transforms')
    if isinstance(group, ast.Attribute) and isinstance(group.value, ast.Name) and group.value.id == 'self':
        if group.attr not in values:
            raise Unsupported('The repack field has no resolved declared/default value.')
        group = _field_default(values[group.attr])
    elif isinstance(group, ast.Name):
        assignments = [item.value for item in create.body if isinstance(item, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == group.id for target in item.targets)]
        if len(assignments) != 1:
            raise Unsupported('The local repack group is not a unique straight-line assignment.')
        group = assignments[0]
    if group is None:
        raise Unsupported('This data factory declares no literal repack contract; its later input transforms are not inferred.')
    return _group_structure(group)


def _prompt_setting(tree, values):
    base = values.get('base_config')
    if isinstance(base, ast.Constant) and base.value is None:
        base = None
    base_fields = {}
    if base is not None:
        if not isinstance(base, ast.Call) or dotted(base.func) != 'DataConfig':
            raise Unsupported('The base DataConfig and prompt injection declaration are dynamic.')
        base_fields = _kwargs(base)
    value = base_fields.get('prompt_from_task', defaults(definition(tree, 'DataConfig')).get('prompt_from_task'))
    if not isinstance(value, ast.Constant) or type(value.value) is not bool:
        raise Unsupported('prompt_from_task is not a literal Boolean.')
    return value.value


def _lerobot_pin(lock):
    packages = [package for package in lock.get('package', []) if package.get('name') == 'lerobot']
    if len(packages) != 1:
        raise Unsupported('The verified lockfile has no unique LeRobot package pin.')
    source = packages[0].get('source') or {}
    git = source.get('git')
    if type(git) is not str:
        raise Unsupported('The verified lockfile does not identify a LeRobot git pin.')
    revision = urlsplit(git).fragment
    if len(revision) != 40 or any(char not in '0123456789abcdef' for char in revision):
        raise Unsupported('The verified LeRobot git source has no full commit identity.')
    return {'git': git, 'revision': revision, 'version': packages[0].get('version'),
            'scope': 'Verified OpenPI lockfile dependency; not an observed installed loader.'}


def check_openpi_context(source, *, config_name, source_root):
    """Compare literal raw repack inputs with declared dataset feature keys."""
    if type(config_name) is not str or not config_name.strip() or len(config_name) > 128:
        raise InputError('openpi_config must be a nonempty name of at most 128 characters')
    report = {'kind': 'openpi_input_context', 'status': 'unknown', 'selected_config': config_name,
              'data_factory': None, 'backend': None, 'raw_source_keys': [], 'missing_keys': [],
              'prompt_from_task': None, 'dataset_codebase_version': None, 'consumer_lerobot_pin': None,
              'scope': SCOPE, 'findings': [], 'artifacts': []}

    def finding(code, status, message, evidence, remedy=''):
        report['status'] = status
        report['findings'].append({'code': code, 'status': status, 'message': message,
                                    'evidence': evidence, 'remedy': remedy})
        return report

    try:
        trees, identities, _ = source_data(source_root)
        for relative, identity in identities.items():
            report['artifacts'].append({'path': str(Path(source_root) / relative), **identity, 'role': 'verified_openpi_source'})
        report['consumer_lerobot_pin'] = _lerobot_pin(trees['uv.lock'])
        report['source_revision'] = MANIFEST['revision']
        tree = trees[CONFIG]
        call, cls, values = _selected(tree, config_name)
        report['data_factory'] = cls.name
        config_ref = {'path': str(Path(source_root) / CONFIG), 'sha256': identities[CONFIG]['sha256'],
                      'pointer': f'L{call.lineno}', 'value': {'selected_config': config_name, 'data_factory': cls.name}}
        root = dataset_root(source)
        info_path = root / 'meta/info.json'
        raw, _ = _read_small(info_path, limit=8 * 1024 * 1024)
        info = _reference_json(raw)
        if type(info) is not dict or type(info.get('features')) is not dict:
            raise InputError('The dataset info.json must declare a features object.')
        digest = hashlib.sha256(raw).hexdigest()
        report['dataset_codebase_version'] = info.get('codebase_version')
        report['artifacts'].append({'path': str(info_path), 'sha256': digest, 'size_bytes': len(raw), 'role': 'dataset_metadata'})
        info_ref = {'path': str(info_path), 'sha256': digest, 'pointer': '/features'}
        if cls.name == 'RLDSDroidDataConfig':
            report['backend'] = 'rlds'
            return finding('openpi.rlds_backend', 'unknown',
                'The selected config uses RLDS input. This LeRobot dataset metadata does not establish its RLDS raw-input contract.',
                [config_ref, info_ref], 'Choose the config and dataset backend used by the actual training run.')
        report['backend'] = 'lerobot'
        prompt = _prompt_setting(tree, values)
        keys, repack, mapping = _repack(cls, values)
        report.update(raw_source_keys=keys, prompt_from_task=prompt,
                      prompt_injection_scope='Declared prompt_from_task only; task metadata and loader injection were not executed.',
                      repack_mapping=mapping)
        injected = ['prompt'] if prompt and 'prompt' in keys else []
        required = sorted(set(keys) - set(injected))
        missing = sorted(set(required) - set(info['features']))
        report.update(required_dataset_keys=required, declared_injected_keys=injected, missing_keys=missing)
        mapping_ref = {'path': str(Path(source_root) / CONFIG), 'sha256': identities[CONFIG]['sha256'],
                       'pointer': f'L{repack.lineno}', 'value': mapping}
        if missing:
            return finding('openpi.raw_key_missing', 'error',
                'The selected OpenPI repack requires dataset keys that are not declared: ' + ', '.join(missing) + '.',
                [config_ref, mapping_ref, info_ref],
                'Use the matching dataset/config or explicitly adapt the selected repack mapping; do not rename public data blindly.')
        return finding('openpi.raw_keys_consistent', 'pass',
            'The dataset declares the required raw repack keys. This does not establish loader-version compatibility or training readiness.',
            [config_ref, mapping_ref, info_ref])
    except (Unsupported, OSError, ValueError, SyntaxError, UnicodeError) as exc:
        return finding('openpi.input_context_unresolved', 'unknown',
                       'The raw input contract is unassessed: ' + str(exc), [],
                       'Supply the selected builtin config and its unchanged supported OpenPI source checkout; inspect detailed evidence.')
