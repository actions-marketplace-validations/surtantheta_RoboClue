"""Evidence-only native processor source checks; never imports a native runtime."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

from .artifacts import InputError

PARAMETER_KEYS = {'expected_source', 'require_same_input_native_reference'}
KIND = 'native_processor_source_capture'
MAX_JSON = 2 * 1024**2
MAX_FILE = 128 * 1024**2
MAX_TOTAL = 256 * 1024**2
SCOPE = ('Evidence-only check of explicit caller restoration intent, witnessed source bytes and captured '
         'same-input native reference parity. No native code or numerical transformation is executed. '
         'Hashes establish retained integrity, not original author authenticity. Native replay, training '
         'correctness, policy competence and physical execution remain unverified.')


class _Missing(Exception):
    pass


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _text(value, maximum=1024):
    return type(value) is str and 0 < len(value) <= maximum and not any(ord(x) < 32 for x in value)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _relative(value, *, empty=False):
    if empty and value == '':
        return value
    if (not _text(value) or '\\' in value or ':' in value or PurePosixPath(value).is_absolute()
            or PurePosixPath(value).as_posix() != value or any(p in ('', '.', '..') for p in value.split('/'))):
        raise InputError('Source evidence requires normalized contained relative paths')
    return value


def _inside(root, relative):
    path = root
    for part in PurePosixPath(_relative(relative)).parts:
        path /= part
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise InputError('Source evidence symlink or root escape refused')
    return path


def _hash(path, maximum=MAX_FILE):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise InputError('Source evidence exceeds regular-file byte bounds')
            digest, size = hashlib.sha256(), 0
            while chunk := stream.read(512 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise InputError('Source evidence exceeds byte bounds')
                digest.update(chunk)
            def stamp(s):
                return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
            if stamp(before) != stamp(os.fstat(stream.fileno())) or stamp(before) != stamp(path.stat(follow_symlinks=False)):
                raise InputError('Source evidence changed during read')
            return digest.hexdigest(), size
    except OSError as exc:
        raise InputError('Cannot read retained source evidence: ' + str(exc)) from exc


def _json(path, expected_sha=None):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError('Duplicate source evidence JSON key')
            result[key] = value
        return result

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON:
                raise InputError('Source evidence JSON exceeds regular-file byte bounds')
            # Bound this actual read, rather than checking a preliminary hash
            # and subsequently allocating from a changed or replaced path.
            raw = stream.read(MAX_JSON + 1)
            if len(raw) > MAX_JSON:
                raise InputError('Source evidence JSON exceeds byte bounds')

            def stamp(value):
                return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

            if (stamp(before) != stamp(os.fstat(stream.fileno()))
                    or stamp(before) != stamp(path.stat(follow_symlinks=False))):
                raise InputError('Source evidence JSON changed during its bounded read')
        if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
            raise InputError('Parsed source JSON differs from its bound bytes')
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(InputError('Non-finite source JSON')))
    except (ValueError, RecursionError, UnicodeDecodeError, OSError) as exc:
        raise InputError('Invalid bounded source evidence JSON') from exc
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > 32 or count > 200_000 or (type(item) is float and not math.isfinite(item)):
            raise InputError('Source evidence JSON exceeds finite structure bounds')
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
    return value


def _file_record(value):
    if (type(value) is not dict or set(value) != {'path', 'sha256', 'size_bytes'}
            or not _sha(value.get('sha256')) or type(value.get('size_bytes')) is not int
            or not 0 <= value['size_bytes'] <= MAX_FILE):
        raise InputError('Invalid source bundle file record')
    return {'path': _relative(value['path']), 'sha256': value['sha256'], 'size_bytes': value['size_bytes']}


def _bundle(value):
    if (type(value) is not dict or set(value) != {'identity', 'config', 'files'}
            or not _text(value.get('identity'), 512) or type(value.get('files')) is not list
            or not 2 <= len(value['files']) <= 64):
        raise InputError('Expected source requires a nonempty immutable source bundle')
    config = _file_record(value['config'])
    files = [_file_record(item) for item in value['files']]
    if len({item['path'] for item in files}) != len(files):
        raise InputError('Duplicate expected source bundle path')
    if not {'policy_preprocessor.json', 'policy_postprocessor.json'} <= {item['path'] for item in files}:
        raise InputError('Expected source lacks native processor declarations')
    return {'identity': value['identity'], 'config': config, 'files': sorted(files, key=lambda x: x['path'])}


def validate_parameters(parameters):
    if type(parameters) is not dict or set(parameters) - PARAMETER_KEYS:
        raise InputError('Only allowlisted native processor-source criteria are supported')
    required = parameters.get('require_same_input_native_reference', True)
    if type(required) is not bool or not required:
        raise InputError('A same-input native reference is required as a strict Boolean true')
    expected = parameters.get('expected_source', 'selected_checkpoint')
    if expected != 'selected_checkpoint':
        expected = _bundle(expected)
    return {'expected_source': expected, 'require_same_input_native_reference': True}


def _capture(directory):
    original = Path(directory).expanduser()
    if original.is_symlink() or not original.is_dir():
        raise InputError('Expected a regular native source capture directory')
    root = original.resolve()
    wrapper = _json(_inside(root, 'capture.json'))
    allowed = {'schema_version', 'kind', 'evidence_role', 'result', 'consumer_route', 'reference_route',
               'restore_intent', 'selected_source', 'bindings', 'derivation'}
    if (type(wrapper) is not dict or set(wrapper) - allowed or type(wrapper.get('schema_version')) is not int
            or wrapper['schema_version'] != 1 or wrapper.get('kind') != KIND
            or wrapper.get('evidence_role') != 'derived_route_view'):
        raise InputError('Unsupported native processor-source capture schema')
    result = wrapper.get('result')
    if type(result) is not dict or set(result) != {'path', 'sha256'} or not _sha(result.get('sha256')):
        raise InputError('Source capture requires a typed raw-result digest')
    result_path = _inside(root, result['path'])
    if _hash(result_path, MAX_JSON)[0] != result['sha256']:
        raise InputError('Source raw-result digest mismatch')
    raw = _json(result_path, result['sha256'])
    if type(raw) is not dict:
        raise InputError('Native source result must be an object')
    derivation = wrapper.get('derivation')
    if (type(derivation) is not dict or set(derivation) != {'attempt_sha256', 'consumer_call_id', 'reference_call_id', 'description'}
            or derivation.get('attempt_sha256') != result['sha256']
            or any(not _text(derivation.get(key), 2048) for key in ('consumer_call_id', 'reference_call_id', 'description'))
            or any(not _text(wrapper.get(key), 256) for key in ('consumer_route', 'reference_route'))):
        raise InputError('Source capture requires an explicit same-attempt derivation')
    if ((wrapper['consumer_route'] == wrapper['reference_route']) !=
            (derivation['consumer_call_id'] == derivation['reference_call_id'])):
        raise InputError('Source route and observed call identities conflict')
    selected = wrapper.get('selected_source')
    if (type(selected) is not dict or set(selected) != {'identity', 'root', 'prefix'}
            or not _text(selected.get('identity'), 512) or not _text(selected.get('root'), 64)):
        raise InputError('Source capture requires explicit selected source identity')
    _relative(selected['prefix'], empty=True)
    bindings = wrapper.get('bindings')
    if type(bindings) is not list or not 1 <= len(bindings) <= 128:
        raise InputError('Source capture requires bounded original byte bindings')
    index, paths, total = {}, set(), 0
    for item in bindings:
        if (type(item) is not dict or set(item) != {'root', 'path', 'retained_path', 'sha256', 'size_bytes'}
                or not _text(item.get('root'), 64)):
            raise InputError('Invalid original source binding')
        record = _file_record({key: item.get(key) for key in ('path', 'sha256', 'size_bytes')})
        key = (item['root'], record['path'])
        retained = _relative(item['retained_path'])
        if key in index or retained in paths:
            raise InputError('Duplicate source binding or retained path')
        if _hash(_inside(root, retained)) != (record['sha256'], record['size_bytes']):
            raise InputError('Original source binding hash/size mismatch')
        total += record['size_bytes']
        if total > MAX_TOTAL:
            raise InputError('Original source bindings exceed total byte bounds')
        index[key] = dict(item)
        paths.add(retained)
    return root, wrapper, raw, index


def _bound_json(root, bindings, source_root, path):
    binding = bindings.get((source_root, _relative(path)))
    if binding is None:
        raise _Missing('Original source bytes are not retained')
    return _json(_inside(root, binding['retained_path']), binding['sha256'])


def _prefix(prefix, name):
    return _relative(prefix + '/' + name if prefix else name)


def _processor_files(root, bindings, source_root, prefix):
    files = {}
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        path = _prefix(prefix, filename)
        config = _bound_json(root, bindings, source_root, path)
        if type(config) is not dict or type(config.get('steps')) is not list or not 1 <= len(config['steps']) <= 32:
            raise _Missing('Native processor declaration has no supported step inventory')
        required = [filename]
        for step in config['steps']:
            if type(step) is not dict:
                raise InputError('Invalid original processor step declaration')
            if 'state_file' in step:
                state = _relative(step['state_file'])
                if not state.endswith('.safetensors'):
                    raise InputError('Only relative native safetensors state references are supported')
                required.append(state)
        for relative in required:
            binding = bindings.get((source_root, _prefix(prefix, relative)))
            if binding is None:
                raise _Missing('Declared original processor state bytes are missing')
            files[relative] = {key: binding[key] for key in ('sha256', 'size_bytes')} | {'path': relative}
    return sorted(files.values(), key=lambda item: item['path'])


def _witnessed(receipts, bindings):
    if type(receipts) is not list or len(receipts) > 128:
        raise _Missing('Actual native loaded-file witnesses are missing')
    witnessed = {}
    for item in receipts:
        if (type(item) is not dict or set(item) - {'root', 'path', 'sha256', 'size_bytes', 'witnesses'}
                or not _text(item.get('root'), 64)):
            raise InputError('Invalid native loaded-file witness')
        record = _file_record({key: item.get(key) for key in ('path', 'sha256', 'size_bytes')})
        key = (item['root'], record['path'])
        if key in witnessed:
            raise InputError('Duplicate native loaded-file receipt')
        binding = bindings.get(key)
        if binding is None:
            raise _Missing('Native loaded file has no retained byte binding')
        if any(binding[k] != record[k] for k in ('sha256', 'size_bytes')):
            raise InputError('Loaded-file receipt contradicts retained source bytes')
        methods = item.get('witnesses')
        if type(methods) is not list or not 1 <= len(methods) <= 8 or any(not _text(x, 256) for x in methods):
            raise _Missing('Native loaded-file witness mechanism is absent')
        witnessed[key] = record
    return witnessed


def _loaded_bundle(root, bindings, route):
    witnessed = _witnessed(route.get('loaded_files'), bindings)
    declarations = []
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        matches = [key for key in witnessed if PurePosixPath(key[1]).name == filename]
        if len(matches) != 1:
            raise _Missing('Need exactly one witnessed native processor declaration per role')
        declarations.append(matches[0])
    source_root, pre_path = declarations[0]
    prefix = str(PurePosixPath(pre_path).parent)
    prefix = '' if prefix == '.' else prefix
    if declarations[1] != (source_root, _prefix(prefix, 'policy_postprocessor.json')):
        raise _Missing('Native pre/post source association is ambiguous')
    files = _processor_files(root, bindings, source_root, prefix)
    if any((source_root, _prefix(prefix, item['path'])) not in witnessed for item in files):
        raise _Missing('A required native processor state load witness is missing')
    return files


def _signature(value, *, nonempty=False):
    tensors = 0
    nonempty_tensors = 0

    def visit(item, depth=0):
        nonlocal tensors, nonempty_tensors
        if depth > 24:
            raise _Missing('Native digest tree exceeds depth bounds')
        if type(item) is dict:
            if any(key in item for key in ('sha256', 'dtype', 'shape')):
                shape = item.get('shape')
                if (not _sha(item.get('sha256')) or not _text(item.get('dtype'), 32)
                        or item['dtype'] not in {'bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                                 'float16', 'float32', 'float64', 'bfloat16'}
                        or type(shape) is not list or len(shape) > 8
                        or any(type(x) is not int or not 0 <= x <= 100_000_000 for x in shape)):
                    raise _Missing('Native tensor evidence needs typed nonempty SHA256/dtype/shape')
                tensors += 1
                nonempty_tensors += math.prod(shape) > 0
                return {key: item[key] for key in ('sha256', 'dtype', 'shape')}
            if len(item) > 256 or any(not _text(key, 512) for key in item):
                raise _Missing('Invalid native digest mapping')
            return {key: visit(child, depth + 1) for key, child in sorted(item.items())}
        if type(item) is list:
            if len(item) > 256:
                raise _Missing('Native digest sequence exceeds bounds')
            return [visit(child, depth + 1) for child in item]
        if item is None or type(item) in (bool, int, str) or (type(item) is float and math.isfinite(item)):
            return item
        raise _Missing('Unsupported native digest value')

    result = visit(value)
    if nonempty and (type(result) is not dict or not result or not tensors or not nonempty_tensors):
        raise _Missing('Native input/output evidence must contain nonempty typed tensor digests')
    return hashlib.sha256(_encode(result)).hexdigest()


def _native_identity(value):
    if (type(value) is not dict or any(type(value.get(key)) is not int or value[key] < 0
                                      for key in ('episode_index', 'frame_index', 'index'))):
        raise _Missing('Original native episode/frame/index identity is missing')
    result = {key: value[key] for key in ('episode_index', 'frame_index', 'index')}
    if 'task_index' in value:
        if type(value['task_index']) is not int or value['task_index'] < 0:
            raise _Missing('Invalid native task identity')
        result['task_index'] = value['task_index']
    if 'timestamp' in value:
        timestamp = value['timestamp']
        if (type(timestamp) not in (int, float)
                or (type(timestamp) is int and abs(timestamp) > 2**53)
                or not math.isfinite(timestamp)):
            raise _Missing('Invalid native timestamp')
        result['timestamp'] = timestamp
    return result


def _stats(route):
    result = []
    for field in ('steps', 'steps_after'):
        steps = route.get(field)
        if type(steps) is not list or not steps:
            raise _Missing('Native before/after step inventory is missing')
        normalizers = []
        for step in steps:
            if type(step) is not dict:
                raise _Missing('Invalid native step evidence')
            if step.get('registry_name') in ('normalizer_processor', 'unnormalizer_processor'):
                if step.get('stats_explicitly_provided') is not False:
                    raise _Missing('Explicit/missing statistics override cannot establish saved-source statistics')
                if type(step.get('effective_tensor_stats')) is not dict or type(step.get('native_state_dict')) is not dict:
                    raise _Missing('Native effective statistics/state witness is missing')
                if type(step.get('parameters')) is not dict or not step['parameters']:
                    raise _Missing('Native normalizer parameter witness is missing')
                normalizers.append({'registry_name': step['registry_name'],
                    'parameters': step.get('parameters'),
                    'effective_tensor_stats': _signature(step['effective_tensor_stats'], nonempty=True),
                    'native_state_dict': _signature(step['native_state_dict'], nonempty=True)})
        if not normalizers:
            raise _Missing('Native normalizer evidence is missing')
        result.append(normalizers)
    return hashlib.sha256(_encode(result)).hexdigest()


def _trace_endpoint(transition, batch, *, input_endpoint=False):
    """Join captured native transition fields to a reported batch, without math.

    The pinned native batch converters nest observations/complementary data and
    rename next.reward/done/truncated. Sparse inputs omit the default control
    fields that the converter supplies; outputs retain those explicit fields.
    """
    fields = {'observation', 'complementary_data', 'action', 'reward', 'done', 'truncated', 'info'}
    if type(transition) is not dict or set(transition) - fields or type(batch) is not dict:
        raise _Missing('Unsupported native transition/batch endpoint evidence')
    flattened = {}
    for field in ('observation', 'complementary_data'):
        value = transition.get(field)
        if value is None:
            continue
        if type(value) is not dict or set(flattened) & set(value):
            raise _Missing('Ambiguous native transition endpoint fields')
        flattened.update(value)
    for field, name, default in (
        ('action', 'action', None), ('reward', 'next.reward', 0.0),
        ('done', 'next.done', False), ('truncated', 'next.truncated', False), ('info', 'info', {}),
    ):
        if field not in transition:
            continue
        if name in flattened:
            raise _Missing('Overlapping native transition endpoint fields')
        value = transition[field]
        if input_endpoint and name not in batch and _encode(value) == _encode(default):
            continue
        flattened[name] = value
    if _signature(flattened, nonempty=True) != _signature(batch, nonempty=True):
        raise _Missing('Native step trace endpoint contradicts its reported batch')


def _trace(root, bindings, route):
    trace = route.get('trace')
    if type(trace) is not dict or not _sha(trace.get('sha256')):
        raise _Missing('Native step trace witness is missing')
    binding = bindings.get(('attempt', _relative(trace.get('path'))))
    if binding is None:
        raise _Missing('Native trace bytes are not retained')
    if binding['sha256'] != trace['sha256']:
        raise InputError('Native trace digest contradicts retained bytes')
    value = _json(_inside(root, binding['retained_path']), binding['sha256'])
    if type(value) is not list or not value:
        raise _Missing('Native trace has no before/after observations')
    steps = route.get('steps')
    if type(steps) is not list or not steps:
        raise _Missing('Native trace step association is missing')
    pairs = []
    for event in value:
        if (type(event) is not dict or event.get('position') not in ('before', 'after')
                or type(event.get('step_index')) is not int or not 0 <= event['step_index'] < len(steps)
                or type(event.get('transition')) is not dict or not event['transition']):
            raise _Missing('Native trace has invalid step/phase association')
        pairs.append((event['position'], event['step_index']))
    expected = [(phase, index) for index in range(len(steps)) for phase in ('before', 'after')]
    if (len(set(pairs)) != len(pairs) or pairs != expected[:len(pairs)]
            or (route.get('status') == 'complete' and pairs != expected)):
        raise _Missing('Native trace does not retain the observed before/after step sequence')
    _trace_endpoint(value[0]['transition'], route.get('input'), input_endpoint=True)
    if route.get('status') == 'complete':
        _trace_endpoint(value[-1]['transition'], route.get('output'))
    return value


def _stat_keys(route, trace):
    """Report evidenced input/stat-key correspondence, not inferred math usage."""
    result = []
    for index, step in enumerate(route['steps']):
        if step.get('registry_name') != 'normalizer_processor':
            continue
        before = next((event['transition'] for event in trace
                       if event['position'] == 'before' and event['step_index'] == index), {})
        observation = before.get('observation', {})
        features = step.get('parameters', {}).get('features', {})
        norm_map = step.get('parameters', {}).get('norm_map', {})
        stats = step.get('effective_tensor_stats', {})
        matching = []
        if all(type(value) is dict for value in (observation, features, norm_map, stats)):
            for key in sorted(set(observation) & set(features) & set(stats)):
                feature = features[key]
                if (type(feature) is dict and type(feature.get('type')) is str
                        and norm_map.get(feature['type']) not in (None, 'IDENTITY')):
                    matching.append(key)
        result.append({'step_index': index, 'observed_input_stat_keys': matching,
                       'scope': 'Recorded input keys with declared nonidentity normalization and matching captured statistics.'})
    return result


def _source_selection(wrapper, raw):
    """Expose only bounded source argument diagnostics, never arbitrary kwargs."""
    result = {'selected_checkpoint': wrapper['selected_source']['identity']}
    routes = raw.get('routes')
    if type(routes) is not dict:
        return result
    for role in ('consumer', 'reference'):
        route = routes.get(wrapper[role + '_route'])
        arguments = route.get('factory_arguments') if type(route) is dict else None
        if type(arguments) is not dict or 'pretrained_path' not in arguments:
            continue
        value = arguments['pretrained_path']
        if value is None or (type(value) is str and len(value) <= 512
                             and not any(ord(char) < 32 for char in value)):
            result[role + '_argument'] = value
    return result


def audit_native_processor_source(directory, *, expected_source='selected_checkpoint', require_same_input_native_reference=True):
    parameters = validate_parameters({'expected_source': expected_source,
                                      'require_same_input_native_reference': require_same_input_native_reference})
    root, wrapper, raw, bindings = _capture(directory)
    report = {'schema_version': 1, 'kind': 'native_processor_source_audit', 'status': 'unknown',
        'source_criterion': {'status': 'unknown', 'code': 'missing_evidence', 'reason': ''},
        'source_binding_status': 'unknown', 'input_association_status': 'unknown', 'input_digest': None,
        'native_identity': None,
        'native_processing_status': 'unknown', 'captured_output_parity_status': 'unknown',
        'effective_statistics_status': 'unknown', 'native_replay_status': 'unknown', 'applicability': 'unknown',
        'selected_source': None, 'expected_source': parameters['expected_source'], 'original_authenticity_status': 'unknown',
        'source_selection': _source_selection(wrapper, raw),
        'evidence_role': wrapper['evidence_role'], 'derivation': wrapper['derivation'],
        'artifacts': [{'path': 'capture.json', 'sha256': _hash(root / 'capture.json')[0]},
                      {'path': wrapper['result']['path'], 'sha256': wrapper['result']['sha256']}],
        'findings': [], 'finding_count': 0, 'findings_omitted': 0, 'scope': SCOPE}

    def finish(status, code, reason):
        if _json(root / 'capture.json') != wrapper or _hash(root / wrapper['result']['path'])[0] != wrapper['result']['sha256']:
            raise InputError('Source capture changed during its criterion check')
        for binding in bindings.values():
            if _hash(_inside(root, binding['retained_path'])) != (binding['sha256'], binding['size_bytes']):
                raise InputError('Retained source bytes changed during the criterion check')
        report['status'] = status
        report['source_criterion'] = {'status': status, 'code': code, 'reason': reason}
        report['findings'] = [{'status': status, 'code': code, 'message': reason}]
        report['finding_count'] = 1
        return report

    intent = wrapper.get('restore_intent')
    if intent != 'restore_selected_checkpoint':
        report['applicability'] = 'not_applicable' if intent in ('fresh_training', 'restore_explicit_source') else 'unknown'
        return finish('unknown', 'intent_not_applicable' if report['applicability'] == 'not_applicable' else 'missing_intent',
                      'Restoration of the selected checkpoint was not explicitly requested.')
    report['applicability'] = 'applicable'
    try:
        routes = raw.get('routes')
        if type(routes) is not dict:
            raise _Missing('Native route observations are missing')
        consumer, reference = (routes.get(wrapper[key]) for key in ('consumer_route', 'reference_route'))
        if type(consumer) is not dict or type(reference) is not dict:
            raise _Missing('Consumer/reference native route is missing')
        for key, route in (('consumer_route', consumer), ('reference_route', reference)):
            if route.get('name') != wrapper[key]:
                raise InputError('Native route identity conflicts with capture view')
            if route.get('status') not in ('complete', 'partial', 'error'):
                raise _Missing('Native route completion status is missing or invalid')
        report['native_processing_status'] = consumer.get('status', 'unknown')
        report['native_identity'] = _native_identity(raw.get('identity'))
        selected = wrapper['selected_source']
        config_path = _prefix(selected['prefix'], 'config.json')
        config = bindings.get((selected['root'], config_path))
        if config is None:
            raise _Missing('Selected original config bytes are missing')
        original_config = _witnessed(raw.get('original_config_loaded_files'), bindings)
        if (selected['root'], config_path) not in original_config:
            raise _Missing('Original selected config load witness is missing')
        selected_files = _processor_files(root, bindings, selected['root'], selected['prefix'])
        bundle = {'identity': selected['identity'], 'config': {'path': 'config.json', **{
            key: config[key] for key in ('sha256', 'size_bytes')}}, 'files': selected_files}
        report['selected_source'] = bundle
        frozen = bundle if parameters['expected_source'] == 'selected_checkpoint' else parameters['expected_source']
        report['expected_source'] = frozen
        raw_input = _signature(raw.get('input'), nonempty=True)
        consumer_input = _signature(consumer.get('input'), nonempty=True)
        reference_input = _signature(reference.get('input'), nonempty=True)
        if not raw_input == consumer_input == reference_input:
            raise _Missing('Consumer/reference inputs are not the same captured observation')
        report.update(input_association_status='pass', input_digest=raw_input)
        if reference.get('status') != 'complete':
            raise _Missing('The same-input selected-source native reference is incomplete')
        reference_output = _signature(reference.get('output'), nonempty=True)
        reference_files = _loaded_bundle(root, bindings, reference)
        if reference_files != selected_files:
            raise _Missing('The native reference is not witnessed from the selected processor source')
        reference_trace = _trace(root, bindings, reference)
        reference_stats = _stats(reference)
        consumer_files = _loaded_bundle(root, bindings, consumer)
        consumer_trace = _trace(root, bindings, consumer)
        consumer_stats = _stats(consumer)
        report.update(reference_output_digest=reference_output,
                      consumer_effective_statistics_digest=consumer_stats,
                      reference_effective_statistics_digest=reference_stats,
                      effective_statistics_status='pass' if consumer_stats == reference_stats else 'error',
                      normalizer_input_statistics={'consumer': _stat_keys(consumer, consumer_trace),
                                                   'reference': _stat_keys(reference, reference_trace)})
        output = None
        if consumer.get('status') == 'complete':
            try:
                output = _signature(consumer.get('output'), nonempty=True)
            except _Missing:
                pass
        if output is not None:
            report.update(consumer_output_digest=output,
                          captured_output_parity_status='pass' if output == reference_output else 'error')
        if bundle != frozen:
            report['source_binding_status'] = 'error'
            return finish('error', 'selected_source_changed', 'Candidate selected-source identity/bytes differ from the frozen case bundle.')
        if consumer_files != frozen['files']:
            report['source_binding_status'] = 'error'
            return finish('error', 'source_contradiction', 'Witnessed native processor/state source differs from the caller-selected restoration source.')
        report['source_binding_status'] = 'pass'
        if consumer_stats != reference_stats:
            report['effective_statistics_status'] = 'error'
            return finish('error', 'effective_statistics_mismatch', 'Captured effective statistics differ from the same-input selected-source reference.')
        if consumer.get('status') != 'complete' or output is None:
            raise _Missing('Consumer native processing is incomplete; no source contradiction was observed')
        return finish('pass' if output == reference_output else 'error',
                      'source_and_reference_match' if output == reference_output else 'captured_output_mismatch',
                      'Selected source bytes, effective statistics and captured same-input reference parity match.'
                      if output == reference_output else 'Captured output differs from the same-input selected-source reference.')
    except _Missing as exc:
        return finish('unknown', 'missing_evidence', str(exc))


def bind_native_processor_source_parameters(directory, parameters=None):
    result = validate_parameters({} if parameters is None else parameters)
    report = audit_native_processor_source(directory, **result)
    if report['status'] != 'pass' or type(report['selected_source']) is not dict:
        raise InputError('Corrected source evidence must pass the same source/reference criterion before freezing its bundle')
    return {'expected_source': report['expected_source'], 'require_same_input_native_reference': True}
