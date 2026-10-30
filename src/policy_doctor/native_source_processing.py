"""Opt-in native source observations, derived views and retained-input replay.

A source view is evidence from one fixed native attempt, never a complete
native-processing capsule or a policy/inference qualification.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re
import tempfile
from pathlib import Path

from .artifacts import InputError
from .native_intake import (
    _hash_selected,
    _open_file,
    prepare_native_inputs,
    prepare_native_model,
    verify_native_inputs,
)
from .native_processing import _encode, _fingerprint, _hash, _inside, _inventory, _invoke, _json, _samples
from .native_processor_source import audit_native_processor_source
from .publication import publish_capture

PROVIDER = 'lerobot_processor_source'
VIEW = 'native_processor_source_view'
ATTEMPT = 'native_processor_source_attempt'
MAX_RAW = 2 * 1024**2
SCOPE = ('Native source selection and same-input preprocessing observations only. Derived candidate/corrected views '
         'share one native attempt; they are not separate launches or complete numerical capsules. '
         'Replay repeats factory processing on retained arrays, without video re-decode, policy inference or competence claims.')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(_encode(value))


def _record(root, relative):
    sha, size = _hash(_inside(root, relative))
    return {'path': relative, 'sha256': sha, 'size_bytes': size}


def _file_record(value):
    if (type(value) is not dict or set(value) != {'path', 'sha256', 'size_bytes'}
            or type(value.get('sha256')) is not str or re.fullmatch(r'[a-f0-9]{64}', value['sha256']) is None
            or type(value.get('size_bytes')) is not int or not 0 <= value['size_bytes'] <= 128 * 1024**2):
        raise InputError('Invalid bounded native source file descriptor')
    relative = value.get('path')
    if (type(relative) is not str or not relative or len(relative) > 512 or '\\' in relative or ':' in relative
            or any(ord(char) < 32 for char in relative) or Path(relative).is_absolute()
            or any(part in {'', '.', '..'} for part in relative.split('/'))):
        raise InputError('Native source descriptors require safe relative paths')
    return dict(value)


def _copy(source_root, relative, destination_root, target_relative, expected):
    source = _inside(source_root, relative)
    if _hash(source) != (expected['sha256'], expected['size_bytes']):
        raise InputError('Native source payload hash/size disagrees with its descriptor')
    destination = _inside(destination_root, target_relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, _ = _open_file(source_root, relative)
    try:
        with destination.open('xb') as output:
            copied = 0
            while block := os.read(fd, min(1024**2, expected['size_bytes'] - copied + 1)):
                copied += len(block)
                if copied > expected['size_bytes']:
                    raise InputError('Native source grew beyond its declared bound during retention')
                output.write(block)
    finally:
        os.close(fd)
    if (_hash(source) != (expected['sha256'], expected['size_bytes'])
            or _hash(destination) != (expected['sha256'], expected['size_bytes'])):
        raise InputError('Native source bytes changed during retention')


def _verifier(runtime):
    result = {'native': _fingerprint('lerobot', runtime)}
    for label, filename in (('source_parent_sha256', 'native_source_processing.py'),
                            ('source_worker_sha256', 'native_processor_source_worker.py'),
                            ('source_auditor_sha256', 'native_processor_source.py'),
                            ('source_intake_sha256', 'native_intake.py')):
        result[label] = _hash(Path(__file__).with_name(filename))[0]
    return result


def _approved(value):
    if value is None:
        value = []
    if type(value) is not list or len(value) > 4:
        raise InputError('Approve at most four exact source references')
    normalized, seen = [], set()
    for item in value:
        if (type(item) is not dict or set(item) != {'repo_id', 'revision', 'directory'}
                or type(item.get('repo_id')) is not str
                or re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', item['repo_id']) is None
                or type(item.get('revision')) is not str or re.fullmatch(r'[a-f0-9]{40}', item['revision']) is None
                or not isinstance(item.get('directory'), (str, Path)) or not str(item['directory'])
                or item['repo_id'] in seen):
            raise InputError('Approved source requires a unique literal Hub ID, 40-hex revision and caller-approved directory')
        seen.add(item['repo_id'])
        normalized.append(dict(item, directory=Path(item['directory']).expanduser().absolute()))
    return normalized


def _selector(samples):
    samples = _samples(samples)
    if len(samples) != 1 or set(samples[0]) != {'episode_index', 'frame_index'}:
        raise InputError('Source observations require exactly one explicit native episode/frame sample')
    return samples


def _add_approved(allowed, prepared, *, model_root='selected', prefix=''):
    for receipt in prepared['receipts']:
        role = receipt['root']
        if role == 'model':
            label, relative, source_root = model_root, prefix + receipt['path'], prepared['model']
        elif role == 'dataset' and 'dataset' in prepared and model_root == 'selected':
            label, relative, source_root = 'dataset', receipt['path'], prepared['dataset']
        else:
            raise InputError('Native intake returned an unsupported original receipt root')
        source = _inside(Path(source_root), receipt['path'])
        if _hash(source) != (receipt['sha256'], receipt['size_bytes']):
            raise InputError('Prepared original source receipt disagrees with staged bytes')
        key = label, relative
        if key in allowed:
            raise InputError('Duplicate approved original binding')
        allowed[key] = {'root': label, 'path': relative, 'sha256': receipt['sha256'],
                        'size_bytes': receipt['size_bytes'], 'source_root': Path(source_root), 'source_path': receipt['path']}


def _needs_cache(models, processor_source):
    if processor_source == 'fresh_training':
        return True
    for model in models:
        for step in _json(Path(model) / 'policy_preprocessor.json').get('steps', []):
            if step.get('registry_name') == 'tokenizer_processor' and not step.get('artifacts'):
                return True
    return False


def _cache_snapshot(runtime, workspace, needed):
    """Clone fixed operator-authorized tokenizer files, including contained Hub blobs."""
    from .native_processor_source_worker import (
        TOKENIZER_FILES,
        TOKENIZER_ID,
        TOKENIZER_REVISION,
        _tokenizer_tree,
    )

    if not needed:
        return None, []
    if type(runtime.get('cache_root')) is not str:
        raise InputError('The fixed Smol tokenizer requires an explicitly configured operator cache')
    original = Path(runtime['cache_root']).resolve(strict=True)
    prefix = 'models--' + TOKENIZER_ID.replace('/', '--') + '/snapshots/' + TOKENIZER_REVISION + '/'
    copied = workspace / 'approved-tokenizer-cache'
    copied.mkdir()
    records = []
    for name in sorted(TOKENIZER_FILES):
        logical = original / (prefix + name)
        if not logical.exists():
            continue
        physical = logical.resolve(strict=True)
        if not physical.is_relative_to(original):
            raise InputError('Operator tokenizer cache file escapes its authorized cache')
        relative = physical.relative_to(original).as_posix()
        sha, stamp = _hash_selected(original, relative, 4 * 1024**2)
        if sum(r['size_bytes'] for r in records) + stamp['st_size'] > 6 * 1024**2:
            raise InputError('Fixed tokenizer snapshot exceeds the supported byte bound')
        item = {'root': 'hf_cache', 'path': prefix + name, 'sha256': sha, 'size_bytes': stamp['st_size'],
                'source_root': original, 'source_path': relative, 'logical': logical, 'physical': physical, 'stamp': stamp}
        _copy(original, relative, copied, prefix + name, item)
        records.append(item)
    if not records:
        raise InputError('Approved fixed tokenizer snapshot is unavailable')
    _tokenizer_tree(copied / prefix)
    return copied, records


def _verify_cache(records):
    for record in records:
        if record['logical'].resolve(strict=True) != record['physical']:
            raise InputError('Operator tokenizer cache association changed during processing')
        sha, stamp = _hash_selected(record['source_root'], record['source_path'], 4 * 1024**2)
        if sha != record['sha256'] or stamp != record['stamp']:
            raise InputError('Operator tokenizer cache bytes changed during processing')


def _descriptor(stage, descriptor, maximum):
    item = _file_record(descriptor)
    if item['size_bytes'] > maximum or _record(stage, item['path']) != item:
        raise InputError('Native source descriptor exceeds bounds or contradicts retained bytes')
    return item


def _payload(stage, completion, allowed, samples, operation):
    """Validate a positive durable closure, never copy the worker's runtime tree."""
    from .native_processor_source_worker import TOKENIZER_FILES, _validate_pipeline

    if (type(completion) is not dict or completion.get('provider') != PROVIDER
            or completion.get('operation') != operation or completion.get('execution_status') not in {'complete', 'partial', 'error'}):
        raise InputError('Invalid native source worker completion')
    result = _descriptor(stage, completion.get('result'), MAX_RAW)
    if result['path'] != 'result.json':
        raise InputError('Native source worker result path must be fixed')
    raw = _json(stage / 'result.json', MAX_RAW)
    if (type(raw) is not dict or raw.get('provider') != PROVIDER
            or raw.get('execution_status') != completion['execution_status']):
        raise InputError('Native source full result contradicts its completion')
    descriptors = {result['path']: result}
    manifest = None
    if completion.get('manifest') is not None:
        manifest_record = _descriptor(stage, completion['manifest'], 128 * 1024)
        if manifest_record['path'] != 'worker_manifest.json':
            raise InputError('Native source worker manifest path must be fixed')
        manifest = _json(stage / 'worker_manifest.json', 128 * 1024)
        if (type(manifest) is not dict or manifest.get('schema') != 'native_processor_source.v1'
                or manifest.get('provider') != PROVIDER or manifest.get('operation') != operation
                or manifest.get('result_path') != 'result.json'
                or manifest.get('bindings') != raw.get('bindings') or manifest.get('exports') != raw.get('exports')
                or type(manifest.get('samples')) is not list or len(manifest['samples']) != 1
                or type(raw.get('samples')) is not list or len(raw['samples']) != 1):
            raise InputError('Native source worker manifest is incomplete or contradictory')
        sample, observed = manifest['samples'][0], raw['samples'][0]
        identity = sample.get('identity')
        if (type(identity) is not dict or any(type(identity.get(key)) is not int or identity[key] != value
                                              for key, value in samples[0].items())
                or type(identity.get('index')) is not int or identity['index'] < 0
                or identity != raw.get('identity') or any(sample.get(key) != observed.get(key) for key in ('identity', 'arrays', 'metadata'))
                or raw.get('routes') != observed.get('routes') or raw.get('input') != observed.get('input')):
            raise InputError('Native source worker did not preserve the requested original sample identity')
        for field, suffix in (('arrays', '.npz'), ('metadata', '.json')):
            item = _descriptor(stage, sample.get(field), 32 * 1024**2 if field == 'arrays' else MAX_RAW)
            if not item['path'].startswith('samples/') or not item['path'].endswith(suffix):
                raise InputError('Retained sample must use a fixed supported numerical/metadata payload role')
            descriptors[item['path']] = item
        metadata = _json(_inside(stage, sample['metadata']['path']))
        if type(metadata) is not dict or metadata.get('identity') != identity:
            raise InputError('Raw sample metadata disagrees with its reported identity')
        if operation == 'probe':
            paths = metadata.get('source_bindings')
            if type(paths) is not list or not paths or any(('dataset', path) not in allowed for path in paths):
                raise InputError('Raw sample dataset membership exceeds the independently selected original closure')
        routes = raw.get('routes')
        if type(routes) is not dict or set(routes) != {'consumer', 'selected_reference'}:
            raise InputError('Native source worker must retain consumer and same-input selected-reference routes')
        for role, route in routes.items():
            if (type(route) is not dict or route.get('name') != role or type(route.get('call_id')) is not str
                    or re.fullmatch(r'native-call-[0-9]{4}', route['call_id']) is None
                    or route.get('status') not in {'complete', 'error', 'construction_error'}):
                raise InputError('Invalid observed native source route/call identity')
            if route.get('trace') is not None:
                trace = route['trace']
                item = {'path': trace.get('path'), 'sha256': trace.get('sha256'),
                        'size_bytes': trace.get('size_bytes', _inside(stage, trace.get('path')).stat().st_size)}
                item = _descriptor(stage, item, MAX_RAW)
                if not item['path'].startswith('traces/') or not item['path'].endswith('.json'):
                    raise InputError('Native trace has an unsupported durable role')
                descriptors[item['path']] = item
        aliases = list(routes.values())
        if aliases[0]['call_id'] == aliases[1]['call_id'] and (
                {k: v for k, v in aliases[0].items() if k != 'name'}
                != {k: v for k, v in aliases[1].items() if k != 'name'}):
            raise InputError('Shared actual native call identity has contradictory observed route evidence')
        if (type(manifest['bindings']) is not list or len(manifest['bindings']) > 128
                or type(manifest['exports']) is not list or len(manifest['exports']) > 2):
            raise InputError('Native original/export inventory exceeds bounds')
        for binding in manifest['bindings']:
            if type(binding) is not dict or not {'root', 'path', 'retained_path', 'sha256', 'size_bytes'} <= binding.keys():
                raise InputError('Invalid native original binding')
            key = binding['root'], binding['path']
            expected = allowed.get(key)
            if (expected is None or any(binding.get(k) != expected[k] for k in ('sha256', 'size_bytes'))
                    or binding['retained_path'] != 'originals/' + binding['root'] + '/' + binding['path']):
                raise InputError('Native binding contradicts selected/caller-approved original source bytes')
            item = _descriptor(stage, {k: binding[k] for k in ('sha256', 'size_bytes')} | {'path': binding['retained_path']}, 128 * 1024**2)
            if item['path'] in descriptors:
                raise InputError('Duplicate retained native original binding')
            descriptors[item['path']] = item
        seen_calls = set()
        for export in manifest['exports']:
            if (type(export) is not dict or type(export.get('call_id')) is not str or export['call_id'] in seen_calls
                    or export.get('route') not in routes or export['call_id'] != routes[export['route']]['call_id']
                    or export.get('path') != 'exports/' + export['call_id']
                    or type(export.get('config_filename')) is not str or Path(export['config_filename']).name != export['config_filename']
                    or type(export.get('files')) is not list or not 1 <= len(export['files']) <= 64):
                raise InputError('Invalid declared native export closure')
            seen_calls.add(export['call_id'])
            export_root = _inside(stage, export['path'])
            cfg = _validate_pipeline(export_root, export['config_filename'], replay=True)
            expected = {export['config_filename']}
            for step in cfg['steps']:
                if 'state_file' in step:
                    expected.add(step['state_file'])
                for directory in step.get('artifacts', {}).values():
                    for name in TOKENIZER_FILES:
                        if _inside(export_root, directory + '/' + name).exists():
                            expected.add(directory + '/' + name)
            actual = set()
            for record in export['files']:
                item = _file_record(record)
                if item['path'] in actual or item['path'] not in expected:
                    raise InputError('Native export contains duplicate/unapproved durable files')
                actual.add(item['path'])
                path = export['path'] + '/' + item['path']
                descriptors[path] = _descriptor(stage, item | {'path': path}, 32 * 1024**2)
            if actual != expected:
                raise InputError('Native export omits declared processor dependencies')
        if len(manifest['bindings']) > 128:
            raise InputError('Native source bindings exceed their bound')
        # Actual load witnesses must not grant new filesystem/source authority.
        collections = [raw.get('original_config_loaded_files', []), raw.get('original_dataset_receipts', [])]
        collections += [route.get('loaded_files', []) for route in routes.values()]
        for collection in collections:
            if type(collection) is not list or len(collection) > 128:
                raise InputError('Native loaded-file receipt collection exceeds its bound')
            seen = set()
            for item in collection:
                if type(item) is not dict:
                    raise InputError('Native loaded-file receipt must be an object')
                key = item.get('root'), item.get('path')
                expected = allowed.get(key)
                if (expected is None or key in seen
                        or any(item.get(field) != expected[field] for field in ('sha256', 'size_bytes'))):
                    raise InputError('Native loaded-file witness exceeds or contradicts approved source membership')
                seen.add(key)
        advertised = manifest.get('payload_files')
        if advertised != completion.get('payload_files'):
            raise InputError('Native source completion and manifest disagree about durable payload')
    elif raw['execution_status'] != 'error':
        raise InputError('Native successful/partial source observation lacks a durable manifest')
    advertised = completion.get('payload_files')
    if type(advertised) is not list or len(advertised) > 256:
        raise InputError('Native source payload inventory exceeds its bound')
    inventory = {}
    for item in advertised:
        item = _file_record(item)
        if item['path'] in inventory:
            raise InputError('Duplicate native source payload descriptor')
        inventory[item['path']] = item
    if inventory != descriptors or sum(item['size_bytes'] for item in inventory.values()) > 256 * 1024**2:
        raise InputError('Native source payload includes undeclared runtime/cache/code files or omits required evidence')
    return raw, manifest, descriptors


def _projection(raw):
    def receipts(values):
        return sorted([{k: item[k] for k in ('root', 'path', 'sha256', 'size_bytes')} for item in values],
                      key=lambda item: (item['root'], item['path']))
    routes = {}
    for name, route in raw.get('routes', {}).items():
        routes[name] = {key: route.get(key) for key in ('call_id', 'status', 'input', 'output', 'steps', 'steps_after', 'factory_arguments')}
        routes[name]['loaded_files'] = receipts(route.get('loaded_files', []))
        if route.get('native_execution_error'):
            routes[name]['native_error_type'] = route['native_execution_error'].get('type')
        if route.get('trace'):
            routes[name]['trace_sha256'] = route['trace'].get('sha256')
    return {'execution_status': raw.get('execution_status'), 'identity': raw.get('identity'),
            'input': raw.get('input'), 'routes': routes,
            'original_config_loaded_files': receipts(raw.get('original_config_loaded_files', []))}


def _commit(root, kind):
    inventory = _inventory(root)
    _write(root / 'manifest.json', {'kind': kind, 'schema_version': 1, 'files': inventory})
    _write(root / 'READY.json', {'manifest_sha256': _hash(root / 'manifest.json')[0]})
    _inventory(root)


def _load_committed(root, kind):
    if root.is_symlink():
        raise InputError('Source capture symlink root refused')
    before = _inventory(root)
    manifest, ready = _json(root / 'manifest.json'), _json(root / 'READY.json')
    if (type(manifest) is not dict or manifest.get('kind') != kind or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or ready != {'manifest_sha256': _hash(root / 'manifest.json')[0]}
            or manifest.get('files') != {k: v for k, v in before.items() if k not in {'manifest.json', 'READY.json'}}):
        raise InputError('Source view file membership, hashes or ready marker changed')
    return before


def _view(root, stage, descriptors, allowed, raw, manifest, verifier, replay, intent, identity, consumer):
    root.mkdir()
    for path, record in descriptors.items():
        if path.startswith('originals/'):
            continue  # Parent-approved originals below supersede worker copies.
        relative = 'raw/result.json' if path == 'result.json' else path
        _copy(stage, path, root, relative, record)
    if manifest is not None:
        _copy(stage, 'worker_manifest.json', root, 'raw/worker_manifest.json', _record(stage, 'worker_manifest.json'))
    bindings = []
    for (label, path), item in sorted(allowed.items()):
        relative = 'originals/' + label + '/' + path
        _copy(item['source_root'], item['source_path'], root, relative, item)
        bindings.append({key: item[key] for key in ('root', 'path', 'sha256', 'size_bytes')} | {'retained_path': relative})
    for route in raw.get('routes', {}).values():
        if route.get('trace'):
            path = route['trace']['path']
            record = _record(root, path)
            if not any(item['root'] == 'attempt' and item['path'] == path for item in bindings):
                bindings.append({'root': 'attempt', 'path': path, 'retained_path': path,
                                 'sha256': record['sha256'], 'size_bytes': record['size_bytes']})
    result_sha = _hash(root / 'raw/result.json')[0]
    reference = 'selected_reference'
    routes = raw.get('routes', {})
    def call(name):
        return routes.get(name, {}).get('call_id', name)
    if consumer == 'consumer' and call('consumer') == call(reference):
        consumer = reference
    wrapper = {'schema_version': 1, 'kind': 'native_processor_source_capture', 'evidence_role': 'derived_route_view',
               'result': {'path': 'raw/result.json', 'sha256': result_sha}, 'consumer_route': consumer,
               'reference_route': reference, 'restore_intent': intent,
               'selected_source': {'identity': identity, 'root': 'selected', 'prefix': ''}, 'bindings': bindings,
               'derivation': {'attempt_sha256': result_sha, 'consumer_call_id': result_sha + ':' + call(consumer),
                              'reference_call_id': result_sha + ':' + call(reference),
                              'description': 'Derived route view from one worker attempt. Shared call identities are aliases of one observed call; this view is not a separate native launch.'}}
    _write(root / 'capture.json', wrapper)
    _write(root / 'native_verifier.json', {'schema_version': 1, 'verifier': verifier, 'replay': replay})
    _write(root / 'expected_native.json', _projection(raw))
    with (root / 'README.md').open('x') as stream:
        stream.write(SCOPE + '\n\nLocal notices are preserved as source evidence; no redistribution permission is inferred.\n')
    report = audit_native_processor_source(root)
    _commit(root, VIEW)
    return report


def _summary(report):
    return {key: report[key] for key in ('status', 'source_criterion', 'source_binding_status', 'input_association_status',
        'native_processing_status', 'captured_output_parity_status', 'effective_statistics_status', 'applicability',
        'source_selection', 'native_identity', 'normalizer_input_statistics', 'consumer_output_digest',
        'reference_output_digest', 'native_replay_status') if key in report}


def _response(raw, candidate, corrected, **extra):
    result = {'kind': 'native_processor_source_observation', 'status': candidate['status'],
              'execution_status': raw.get('execution_status'), 'scope': SCOPE,
              'candidate': _summary(candidate), 'corrected': _summary(corrected), **extra}
    if len(_encode(result)) > 192 * 1024:
        raise InputError('Native source observation summary exceeds response bound')
    return result


def _input_errors(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except InputError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
            raise InputError('Invalid bounded native source observation: ' + str(exc)) from exc
    return guarded


@_input_errors
def _selected_identity(selected, cache_records):
    """Bind processing assets across relocation, excluding unrelated local notices."""
    records = [{'role': 'model', **{key: item[key] for key in ('path', 'sha256', 'size_bytes')}}
               for item in selected['receipts'] if item['root'] == 'model' and item['role'] != 'local_notice']
    records.extend({'role': 'tokenizer', **{key: item[key] for key in ('path', 'sha256', 'size_bytes')}}
                   for item in cache_records)
    records.sort(key=lambda item: (item['role'], item['path']))
    return 'sha256:' + hashlib.sha256(_encode(records)).hexdigest()


def probe_native_processor_source(model, dataset, *, runtimes, samples, destination=None,
                                  processor_source=None, intent=None, approved_sources=None):
    """Run one fixed native attempt and optionally retain two honest route views."""
    if type(runtimes) is not dict or 'lerobot' not in runtimes:
        raise InputError('Source observation requires an explicitly configured LeRobot runtime')
    samples = _selector(samples)
    processor_source = 'selected_checkpoint' if processor_source is None else processor_source
    if processor_source not in {'selected_checkpoint', 'serialized_pretrained_path', 'fresh_training'}:
        raise InputError('Unsupported fixed processor source mode')
    if intent is not None and (type(intent) is not str or len(intent) > 256):
        raise InputError('Intent must be a bounded literal string or null')
    approved = _approved(approved_sources)
    source_roots = [Path(model).expanduser().absolute(), Path(dataset).expanduser().absolute()]
    source_roots += [entry['directory'] for entry in approved]
    target = None
    if destination is not None:
        requested = Path(destination).expanduser().absolute()
        if requested.is_symlink():
            raise InputError('Source capture destination symlink refused')
        target = requested.parent.resolve() / requested.name
        if target.exists() or any(target.is_relative_to(root.resolve()) for root in source_roots):
            raise InputError('Choose a new source-capture destination outside original roots')
        target.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtimes['lerobot']
    verifier = _verifier(runtime)
    with tempfile.TemporaryDirectory(prefix='.policy-doctor-source-', dir=target.parent if target else None) as folder:
        work = Path(folder)
        selected_work = work / 'selected'
        selected_work.mkdir()
        selected = prepare_native_inputs('lerobot', Path(model), Path(dataset), samples, selected_work, runtime, _invoke, model_family='smolvla')
        prepared, allowed = [selected], {}
        _add_approved(allowed, selected)
        config = _json(Path(selected['model']) / 'config.json')
        origin = config.get('pretrained_path') if type(config) is dict else None
        active = []
        if processor_source == 'serialized_pretrained_path':
            active = [entry for entry in approved if entry['repo_id'] == origin]
            if type(origin) is not str or len(active) != 1:
                raise InputError('Serialized source is not an explicitly approved exact Hub reference')
        worker_approved = []
        for number, entry in enumerate(active):
            alternate_work = work / ('alternate-' + str(number))
            alternate_work.mkdir()
            alternate = prepare_native_model(entry['directory'], alternate_work, model_family='smolvla')
            prepared.append(alternate)
            prefix = 'models--' + entry['repo_id'].replace('/', '--') + '/snapshots/' + entry['revision'] + '/'
            _add_approved(allowed, alternate, model_root='hf_cache', prefix=prefix)
            worker_approved.append({'repo_id': entry['repo_id'], 'revision': entry['revision'], 'model': str(alternate['model'])})
        cache, cache_records = _cache_snapshot(runtime, work, _needs_cache([p['model'] for p in prepared], processor_source))
        for item in cache_records:
            key = item['root'], item['path']
            if key in allowed:
                raise InputError('Approved tokenizer/source snapshot roles overlap')
            allowed[key] = item
        identity = _selected_identity(selected, cache_records)
        request = {'operation': 'probe', 'source_root': runtime['source_root'], 'model': str(selected['model']),
                   'dataset': str(selected['dataset']), 'samples': samples, 'processor_source': processor_source,
                   'intent': intent, 'approved_sources': worker_approved, 'selected_identity': identity, 'output': str(work / 'worker-output')}
        if cache is not None:
            request['cache_root'] = str(cache)
        descriptor = _invoke(Path(__file__).with_name('native_processor_source_worker.py'), runtime, request, work)
        for item in prepared:
            verify_native_inputs(item['selection'])
        _verify_cache(cache_records)
        if _verifier(runtime) != verifier:
            raise InputError('Native source/runtime verifier changed during processing')
        stage = Path(request['output'])
        raw, manifest, descriptors = _payload(stage, descriptor, allowed, samples, 'probe')
        lineage = raw.get('config_lineage', {})
        if raw['execution_status'] != 'error' and (lineage.get('serialized_training_source') != origin or lineage.get('parsed_pretrained_path') != origin):
            raise InputError('Native parsed source lineage contradicts the selected original config')
        replay = {'samples': samples, 'processor_source': processor_source, 'intent': intent, 'selected_identity': identity,
                  'approved_sources': [{k: entry[k] for k in ('repo_id', 'revision')} for entry in active]}
        publication = work / 'publication'
        publication.mkdir()
        candidate = _view(publication / 'candidate', stage, descriptors, allowed, raw, manifest, verifier, replay, intent, identity, 'consumer')
        corrected = _view(publication / 'corrected', stage, descriptors, allowed, raw, manifest, verifier, replay, intent, identity, 'selected_reference')
        for item in prepared:
            verify_native_inputs(item['selection'])
        _verify_cache(cache_records)
        if _verifier(runtime) != verifier:
            raise InputError('Native source/runtime verifier changed before publication')
        if target is None:
            return _response(raw, candidate, corrected, retained=False,
                             retention_note='Temporary views/arrays/exports were removed; choose a destination to retain evidence.')
        _commit(publication, ATTEMPT)
        publish_capture(publication, target)
        return _response(raw, candidate, corrected, retained=True, directory=str(target),
                         candidate_directory=str(target / 'candidate'), corrected_directory=str(target / 'corrected'))


@_input_errors
def reproduce_native_processor_source(directory, *, runtimes):
    """Verify immutable evidence, then repeat fixed factory calls on retained arrays."""
    original = Path(directory).expanduser()
    if original.is_symlink():
        raise InputError('Native source view symlink root refused')
    root = original.resolve()
    attempt_root, attempt_before = None, None
    if _json(root / 'manifest.json').get('kind') == ATTEMPT:
        attempt_root, attempt_before = root, _load_committed(root, ATTEMPT)
        root = root / 'candidate'
    before = _load_committed(root, VIEW)
    candidate = audit_native_processor_source(root)
    wrapper = _json(root / 'capture.json')
    metadata = _json(root / 'native_verifier.json')
    if (type(metadata) is not dict or metadata.get('schema_version') != 1 or type(metadata.get('replay')) is not dict
            or type(runtimes) is not dict or 'lerobot' not in runtimes):
        raise InputError('Source replay requires retained verifier metadata and an explicitly configured runtime')
    runtime = runtimes['lerobot']
    current = _verifier(runtime)
    if current != metadata.get('verifier'):
        return {'kind': 'native_processor_source_reproduction', 'status': 'unknown', 'verifier_matches': False,
                'candidate_evaluated': False, 'scope': SCOPE}
    replay = metadata['replay']
    samples = _selector(replay.get('samples'))
    manifest_path = root / 'raw/worker_manifest.json'
    if not manifest_path.is_file():
        return {'kind': 'native_processor_source_reproduction', 'status': 'unknown', 'verifier_matches': True,
                'candidate_evaluated': False, 'scope': SCOPE, 'reason': 'Failure capture lacks retained numerical inputs.'}
    manifest = _json(manifest_path, 128 * 1024)
    if type(manifest.get('samples')) is not list or len(manifest['samples']) != 1:
        raise InputError('Source replay lacks one retained raw sample')
    with tempfile.TemporaryDirectory(prefix='policy-doctor-source-replay-') as folder:
        work = Path(folder)
        staged = work / 'originals'
        staged.mkdir()
        allowed = {}
        for binding in wrapper['bindings']:
            if binding['root'] == 'attempt':
                continue
            if binding['root'] not in {'selected', 'hf_cache', 'dataset'}:
                raise InputError('Unapproved retained source root')
            target = binding['root'] + '/' + binding['path']
            _copy(root, binding['retained_path'], staged, target, binding)
            allowed[(binding['root'], binding['path'])] = dict(binding, source_root=staged / binding['root'], source_path=binding['path'])
        worker_approved = []
        for entry in replay.get('approved_sources', []):
            if type(entry) is not dict or set(entry) != {'repo_id', 'revision'}:
                raise InputError('Invalid retained approved source identity')
            prefix = 'models--' + entry['repo_id'].replace('/', '--') + '/snapshots/' + entry['revision']
            worker_approved.append({'repo_id': entry['repo_id'], 'revision': entry['revision'], 'model': str(_inside(staged / 'hf_cache', prefix))})
        sample = manifest['samples'][0]
        request = {'operation': 'replay', 'source_root': runtime['source_root'], 'model': str(staged / 'selected'),
                   'samples': samples, 'processor_source': replay['processor_source'], 'intent': replay.get('intent'),
                   'approved_sources': worker_approved, 'selected_identity': replay['selected_identity'],
                   'raw_samples': [{key: str(_inside(root, sample[key]['path'])) for key in ('arrays', 'metadata')}],
                   'output': str(work / 'worker-output')}
        if (staged / 'hf_cache').is_dir():
            request['cache_root'] = str(staged / 'hf_cache')
        descriptor = _invoke(Path(__file__).with_name('native_processor_source_worker.py'), runtime, request, work)
        stage = Path(request['output'])
        observed, produced_manifest, descriptors = _payload(stage, descriptor, allowed, samples, 'replay')
        expected = _json(root / 'expected_native.json', MAX_RAW)
        matches = _encode(_projection(observed)) == _encode(expected)
        views = work / 'replayed-views'
        views.mkdir()
        after_candidate = _view(views / 'candidate', stage, descriptors, allowed, observed, produced_manifest,
                                current, replay, replay.get('intent'), replay['selected_identity'], wrapper['consumer_route'])
        after_corrected = _view(views / 'corrected', stage, descriptors, allowed, observed, produced_manifest,
                                current, replay, replay.get('intent'), replay['selected_identity'], 'selected_reference')
        if (before != _inventory(root) or current != _verifier(runtime)
                or (attempt_root is not None and attempt_before != _inventory(attempt_root))):
            raise InputError('Source evidence/runtime changed during native replay')
        matches = matches and after_candidate['source_criterion'] == candidate['source_criterion']
        return _response(observed, after_candidate, after_corrected, kind='native_processor_source_reproduction',
                         status='pass' if matches else 'error', reproduction_matches=matches,
                         verifier_matches=True, candidate_evaluated=True)
