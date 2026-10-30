"""Isolated source-data prototype: actual serving command to a pinned builtin selection.

No source import, expression evaluation, checkpoint read or network request occurs.
"""
import ast
import hashlib
import json
import os
import shlex
import stat
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

MANIFEST = json.loads(Path(__file__).with_name('source_manifest.json').read_text())
SUPPORTED = {'pi0_droid', 'pi05_droid', 'pi0_fast_droid'}
MAX_COMMAND = 16 * 1024
MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 8 * 1024 * 1024


class Unsupported(ValueError):
    pass


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    raise Unsupported('unsupported source expression')


def literal(node):
    if not isinstance(node, ast.Constant) or type(node.value) not in (str, int, bool, type(None)):
        raise Unsupported('nonliteral builtin field')
    return node.value


def keywords(node, expected):
    if not isinstance(node, ast.Call) or dotted(node.func) != expected or node.args:
        raise Unsupported('unsupported builtin constructor')
    if any(item.arg is None for item in node.keywords):
        raise Unsupported('expanded builtin arguments')
    return {item.arg: item.value for item in node.keywords}


def definition(tree, name, kind=ast.ClassDef):
    matches = [node for node in tree.body if isinstance(node, kind) and node.name == name]
    if len(matches) != 1:
        raise Unsupported('unsupported source declaration')
    return matches[0]


def defaults(node):
    return {item.target.id: item.value for item in node.body if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name) and item.value is not None}


def source_root_fd(root):
    root = Path(root)
    if not root.is_absolute() or '..' in root.parts:
        raise Unsupported('source root must be absolute without parent traversal')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def source_data(root):
    root_fd = source_root_fd(root)
    trees, evidence = {}, {}
    total = 0
    try:
        for relative, expected in MANIFEST['files'].items():
            fd = os.dup(root_fd)
            try:
                parts = Path(relative).parts
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = child
                leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                try:
                    meta = os.fstat(leaf)
                    if not stat.S_ISREG(meta.st_mode) or meta.st_size > MAX_FILE:
                        raise Unsupported('source file size or type refused')
                    if meta.st_size > MAX_TOTAL - total:
                        raise Unsupported('source byte limit exceeded')
                    limit = min(MAX_FILE, MAX_TOTAL - total)
                    raw = b''
                    while len(raw) < limit:
                        chunk = os.read(leaf, min(65536, limit - len(raw)))
                        if not chunk:
                            break
                        raw += chunk
                    if os.fstat(leaf).st_size != len(raw):
                        raise Unsupported('source size changed or exceeded read limit')
                finally:
                    os.close(leaf)
            finally:
                os.close(fd)
            total += len(raw)
            if total > MAX_TOTAL or len(raw) > MAX_FILE:
                raise Unsupported('source byte limit exceeded')
            actual = {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
            if actual != expected:
                raise Unsupported('source fingerprint drift')
            trees[relative] = tomllib.loads(raw.decode('utf-8')) if relative == 'uv.lock' else ast.parse(raw, filename=relative)
            evidence[relative] = actual
    finally:
        os.close(root_fd)
    return trees, evidence, total


def command_fields(command):
    if not isinstance(command, str) or len(command.encode()) > MAX_COMMAND:
        raise Unsupported('command type or size refused')
    if any(character in command for character in ';|&<>'):
        raise Unsupported('shell operators are unsupported')
    words = shlex.split(command)
    if words[:3] != ['uv', 'run', 'scripts/serve_policy.py']:
        raise Unsupported('unsupported serving entry point')
    fields, route, unknown = {}, None, []
    remaining = words[3:]
    index = 0
    while index < len(remaining):
        word = remaining[index]
        if word in {'policy:checkpoint', 'policy:default'}:
            if route is not None:
                raise Unsupported('multiple policy selections')
            route = word
        elif word.startswith('--'):
            key, separator, value = word[2:].partition('=')
            if key not in {'policy.config', 'policy.dir', 'env'}:
                unknown.append(key)
                index += 1
                if not separator and index < len(remaining) and not remaining[index].startswith('-'):
                    index += 1
                continue
            if key == 'env' and route is not None:
                raise Unsupported('environment argument must precede policy subcommand')
            if key.startswith('policy.') and route != 'policy:checkpoint':
                raise Unsupported('checkpoint fields must follow checkpoint subcommand')
            if not separator:
                if index + 1 == len(remaining) or remaining[index + 1].startswith('--'):
                    raise Unsupported('missing argument value')
                index += 1
                value = remaining[index]
            if key in fields:
                raise Unsupported('duplicate serving argument')
            if key in {'policy.config', 'policy.dir', 'env'} and any(token in value for token in ['$','`','\n']):
                raise Unsupported('unresolved serving argument')
            fields[key] = value
        else:
            raise Unsupported('unsupported shell or positional token')
        index += 1
    if unknown:
        raise Unsupported('unmodeled argument names: ' + ', '.join(sorted(set(unknown))))
    route = route or 'policy:default'
    if route == 'policy:checkpoint' and not {'policy.config', 'policy.dir'} <= fields.keys():
        raise Unsupported('checkpoint config and dir are required')
    if route == 'policy:default' and ({'policy.config', 'policy.dir'} & fields.keys()):
        raise Unsupported('checkpoint fields require checkpoint route')
    return route, fields


def checkpoint_asset(reference, asset_id):
    parsed = urlsplit(reference)
    if parsed.query or parsed.fragment or parsed.username or parsed.password or reference.startswith('~'):
        raise Unsupported('credentialed or unresolved checkpoint reference')
    if parsed.scheme:
        if parsed.scheme != 'gs' or not parsed.netloc or '@' in parsed.netloc:
            raise Unsupported('unsupported checkpoint reference scheme')
    elif not Path(reference).is_absolute():
        raise Unsupported('checkpoint reference must be absolute or public gs URI')
    if any(part == '..' for part in parsed.path.split('/')):
        raise Unsupported('checkpoint parent traversal refused')
    return reference.rstrip('/') + '/assets/' + asset_id + '/norm_stats.json'


def resolve_serving_command(command, source_root):
    result = {'status': 'unknown', 'schema_version': 1, 'kind': 'openpi_builtin_source_resolution',
              'source_revision': MANIFEST['revision'], 'command_sha256': None,
              'checkpoint_io_performed': False, 'native_factory_executed': False,
              'historical_training_mode': 'unknown', 'checkpoint_reference_immutability': 'unassessed'}
    try:
        if not isinstance(command, str) or len(command) > MAX_COMMAND:
            raise Unsupported('command type or size refused')
        if any(ord(character) < 32 or ord(character) == 127 for character in command):
            raise Unsupported('control characters and multiline shell syntax are unsupported')
        if not isinstance(source_root, (str, Path)):
            raise Unsupported('source root path type refused')
        raw_command = command.encode('utf-8', errors='strict')
        if len(raw_command) > MAX_COMMAND:
            raise Unsupported('command type or size refused')
        result['command_sha256'] = hashlib.sha256(raw_command).hexdigest()
        route, fields = command_fields(command)
        trees, fingerprints, read_bytes = source_data(source_root)
        tyro_version = next(item['version'] for item in trees['uv.lock']['package'] if item['name'] == 'tyro')
        if tyro_version != '0.9.22':
            raise Unsupported('unsupported locked parser version')
        result['parser_dependency_coverage'] = {'supported_native_lock_tyro': tyro_version,
                                                'actual_user_runtime': 'uninspected',
                                                'condition': 'Resolution assumes this pinned source and supported lock parser semantics.'}
        serve = trees['scripts/serve_policy.py']
        env_enum = definition(serve, 'EnvMode')
        env_names = {target.id for node in env_enum.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name)}
        if 'env' in fields and fields['env'] not in env_names:
            raise Unsupported('unsupported native environment enum spelling')
        if route == 'policy:default':
            env = fields.get('env')
            if env is None:
                env = dotted(defaults(definition(serve, 'Args'))['env']).split('.')[-1]
            mapping = next(node.value for node in serve.body if isinstance(node, ast.AnnAssign)
                           and isinstance(node.target, ast.Name) and node.target.id == 'DEFAULT_CHECKPOINT')
            matches = [value for key, value in zip(mapping.keys, mapping.values, strict=True)
                       if dotted(key).split('.')[-1] == env]
            if len(matches) != 1:
                raise Unsupported('unsupported default environment')
            choice = keywords(matches[0], 'Checkpoint')
            config_id, checkpoint = literal(choice['config']), literal(choice['dir'])
        else:
            config_id, checkpoint = fields['policy.config'], fields['policy.dir']
        if config_id not in SUPPORTED:
            raise Unsupported('custom or unsupported builtin config')
        config = trees['src/openpi/training/config.py']
        configurations = next(node.value for node in config.body if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == '_CONFIGS' for target in node.targets))
        candidates = [call for call in configurations.elts if isinstance(call, ast.Call)
                      and any(item.arg == 'name' and literal(item.value) == config_id for item in call.keywords)]
        if len(candidates) != 1:
            raise Unsupported('builtin config declaration not unique')
        selected = keywords(candidates[0], 'TrainConfig')
        model_call = selected['model']
        constructor = dotted(model_call.func)
        if constructor == 'pi0_config.Pi0Config':
            model_file, model_class = 'src/openpi/models/pi0_config.py', 'Pi0Config'
        elif constructor == 'pi0_fast.Pi0FASTConfig':
            model_file, model_class = 'src/openpi/models/pi0_fast.py', 'Pi0FASTConfig'
        else:
            raise Unsupported('custom model config')
        declaration = definition(trees[model_file], model_class)
        model_defaults = defaults(declaration)
        overrides = keywords(model_call, constructor)
        model_fields = {key: literal(overrides.get(key, model_defaults[key])) for key in ['action_dim', 'action_horizon']}
        method = definition(declaration, 'model_type', ast.FunctionDef)
        if model_class == 'Pi0Config':
            pi05 = literal(overrides.get('pi05', model_defaults['pi05']))
            if type(pi05) is not bool or not isinstance(method.body[0], ast.If) or dotted(method.body[0].test) != 'self.pi05':
                raise Unsupported('unmodeled model type derivation')
            return_node = method.body[0].body[0] if pi05 else method.body[1]
        else:
            return_node = method.body[0]
        model_enum = dotted(return_node.value).split('.')[-1]
        enum_class = definition(trees['src/openpi/models/model.py'], 'ModelType')
        enum_value = next(literal(node.value) for node in enum_class.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == model_enum for target in node.targets))
        data = keywords(selected['data'], 'SimpleDataConfig')
        asset_id = literal(keywords(data['assets'], 'AssetsConfig')['asset_id'])
        if not isinstance(asset_id, str) or not asset_id or '/' in asset_id or asset_id in {'.', '..'}:
            raise Unsupported('unsupported asset ID')
        factory = definition(config, 'DataConfigFactory')
        base_method = definition(factory, 'create_base_config', ast.FunctionDef)
        replace = next(node.value for node in base_method.body if isinstance(node, ast.Return))
        replacement = {item.arg: item.value for item in replace.keywords}
        mode_rule = replacement['use_quantile_norm']
        if not (isinstance(mode_rule, ast.Compare) and dotted(mode_rule.left) == 'model_config.model_type'
                and len(mode_rule.ops) == 1 and isinstance(mode_rule.ops[0], ast.NotEq)
                and dotted(mode_rule.comparators[0]) == 'ModelType.PI0'):
            raise Unsupported('unmodeled effective normalization rule')
        asset_assignment = next(node.value for node in base_method.body if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == 'asset_id' for target in node.targets))
        if not (isinstance(asset_assignment, ast.BoolOp) and isinstance(asset_assignment.op, ast.Or)
                and [dotted(value) for value in asset_assignment.values] == ['self.assets.asset_id', 'repo_id']):
            raise Unsupported('unmodeled effective asset rule')
        base = keywords(data['base_config'], 'DataConfig')
        base_flag = literal(base.get('use_quantile_norm', defaults(definition(config, 'DataConfig'))['use_quantile_norm']))
        group = keywords(data['data_transforms'].body, '_transforms.Group')
        data_outputs = [dotted(item.func) for item in group['outputs'].elts]
        if data_outputs != ['droid_policy.DroidOutputs']:
            raise Unsupported('unmodeled data output transform')
        droid = definition(trees['src/openpi/policies/droid_policy.py'], 'DroidOutputs')
        slices = [node for node in ast.walk(droid) if isinstance(node, ast.Slice) and node.upper is not None]
        final_width = literal(slices[0].upper) if len(slices) == 1 else None
        policy = trees['src/openpi/policies/policy_config.py']
        policy_constructor = next(node for node in ast.walk(policy) if isinstance(node, ast.Call)
                                  and isinstance(node.func, ast.Attribute) and dotted(node.func) == '_policy.Policy')
        arguments = {item.arg: item.value for item in policy_constructor.keywords}
        ordering = {}
        for name in ['transforms', 'output_transforms']:
            ordering[name] = [dotted(item.value) if isinstance(item, ast.Starred) else dotted(item.func).split('.')[-1]
                              for item in arguments[name].elts]
        if ordering['output_transforms'].index('Unnormalize') >= ordering['output_transforms'].index('data_config.data_transforms.outputs'):
            raise Unsupported('unmodeled inverse/output order')
        stats_path = checkpoint_asset(checkpoint, asset_id)
        result.update(status='resolved_supported_builtin', config_id=config_id, route=route,
                      checkpoint_reference=checkpoint, statistics_reference=stats_path, asset_id=asset_id,
                      model={'class': model_class, 'model_type_enum': model_enum, 'model_type': enum_value, **model_fields},
                      normalization={'use_quantile_norm': model_enum != 'PI0', 'base_config_flag': base_flag,
                                     'basis': 'DataConfigFactory.create_base_config model_type!=ModelType.PI0 overrides base flag'},
                      action_boundaries={'declared_model_action_width': model_fields['action_dim'],
                                         'pre_inverse_width_basis': 'declared model dimension; PI0_FAST first decodes model tokens to action_dim',
                                         'final_DroidOutputs_slice_width': final_width,
                                         'full_native_chain_executed': False},
                      transform_order=ordering, source_fingerprints=fingerprints, source_bytes_read=read_bytes,
                      evidence={'config_line': candidates[0].lineno, 'model_type_file': model_file,
                                'model_type_line': method.lineno, 'factory_line': base_method.lineno,
                                'output_slice_line': droid.lineno},
                      limits=['Source-derived builtin selection only, not checkpoint content authentication or model execution.',
                              'Physical action schema, historical mode and task suitability require separate evidence.'])
    except (Unsupported, OSError, SyntaxError, ValueError, KeyError, StopIteration, AttributeError, IndexError) as exc:
        result['reason'] = str(exc) if isinstance(exc, Unsupported) else 'source or command shape is unsupported'
    return result
