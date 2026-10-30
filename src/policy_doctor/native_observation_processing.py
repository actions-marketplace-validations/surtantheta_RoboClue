"""Bounded GR00T adapter dry runs and replay of retained original observations.

Native code always comes from operator runtime configuration. A retained view
contains evidence and a fixed candidate/reference role, never executable code.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .artifacts import InputError
from .native_intake import _open_file
from .native_processing import _encode, _fingerprint, _hash, _inside, _inventory, _invoke, _json, _samples
from .native_source_processing import (
    _commit,
    _file_record,
    _input_errors,
    _load_committed,
    _record,
    _write,
)
from .publication import publish_capture

PROVIDER = 'groot_adapter'
VIEW = 'native_observation_source_view'
ATTEMPT = 'native_observation_source_attempt'
MAX_RAW = 2 * 1024**2
PAYLOAD = {'result.json': MAX_RAW, 'samples/arrays.npz': 16 * 1024**2,
           'samples/metadata.json': MAX_RAW}
SCOPE = ('CPU native adapter input dry run with policy loading explicitly substituted and a generated action sink. '
         'Native state processing is reported separately. Retained-input replay repeats one actual configured route '
         'without video re-decode, weights, inference, robot commands or task-competence claims.')


def validate_parameters(parameters):
    from .native_observation_source import validate_parameters as validate
    return validate(parameters)


def audit_native_observation_source(directory, **parameters):
    from .native_observation_source import audit_native_observation_source as audit
    return audit(directory, **parameters)


def prepare_native_groot_inputs(model, dataset, samples, workspace):
    from .native_groot_intake import prepare_native_groot_inputs as prepare
    return prepare(model, dataset, samples, workspace)


def verify_native_groot_inputs(selection):
    from .native_groot_intake import verify_native_groot_inputs as verify
    return verify(selection)


def _copy(source_root, relative, destination_root, target_relative, expected):
    """Copy only the bounded same-descriptor bytes, including under source races."""
    record = _file_record({'path': relative, **{key: expected[key] for key in ('sha256', 'size_bytes')}})
    destination = _inside(destination_root, target_relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, _ = _open_file(source_root, relative)
    digest, total = hashlib.sha256(), 0
    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    try:
        before = os.fstat(fd)
        if before.st_size != record['size_bytes']:
            raise InputError('Adapter source size changed before retention')
        with destination.open('xb') as output:
            while block := os.read(fd, min(1024**2, record['size_bytes'] - total + 1)):
                total += len(block)
                if total > record['size_bytes']:
                    raise InputError('Adapter source grew beyond its bounded receipt during retention')
                digest.update(block)
                output.write(block)
        if (total != record['size_bytes'] or digest.hexdigest() != record['sha256']
                or identity(os.fstat(fd)) != identity(before)):
            raise InputError('Adapter source bytes/identity changed during retention')
    finally:
        os.close(fd)
    try:
        reopened, current = _open_file(source_root, relative)
    except OSError as exc:
        raise InputError('Adapter source path changed during retention') from exc
    else:
        os.close(reopened)
    if identity(current) != identity(before):
        raise InputError('Adapter source path changed during retention')


def _verifier(runtime):
    result = {'native': _fingerprint(PROVIDER, runtime)}
    for role, name in [('parent', 'native_observation_processing.py'), ('worker', 'native_groot_adapter_worker.py'),
                       ('auditor', 'native_observation_source.py'), ('intake', 'native_groot_intake.py'),
                       ('lifecycle_helper', 'native_source_processing.py'), ('fd_helper', 'native_intake.py'),
                       ('native_helper', 'native_processing.py'), ('publication_helper', 'publication.py')]:
        result[role + '_sha256'] = _hash(Path(__file__).with_name(name))[0]
    return result


def _selector(samples):
    samples = _samples(samples)
    if len(samples) != 1 or set(samples[0]) != {'episode_index', 'frame_index'}:
        raise InputError('Adapter observations require exactly one explicit native episode/frame sample')
    return samples


def _runtime(runtimes):
    if type(runtimes) is not dict or type(runtimes.get(PROVIDER)) is not dict:
        raise InputError('Adapter observations require an operator-configured groot_adapter runtime')
    runtime = runtimes[PROVIDER]
    for key in ('python', 'source_root', 'groot_source_root'):
        if type(runtime.get(key)) is not str or not runtime[key]:
            raise InputError('Adapter runtime lacks an explicitly configured ' + key)
    if 'reference_source_root' in runtime and (type(runtime['reference_source_root']) is not str
                                             or not runtime['reference_source_root']):
        raise InputError('Adapter reference source must be an explicitly configured directory')
    return runtime


def _role_source(runtime, role):
    if role not in {'candidate', 'reference'}:
        raise InputError('Retained replay role must be candidate or reference')
    key = 'source_root' if role == 'candidate' else 'reference_source_root'
    if key not in runtime:
        raise InputError('Retained reference replay requires an operator-configured reference source')
    return runtime[key]


def _allowed(prepared):
    allowed = {}
    receipts = prepared.get('receipts')
    if type(receipts) is not list or not 1 <= len(receipts) <= 64:
        raise InputError('Adapter intake requires a bounded positive original inventory')
    for value in receipts:
        if type(value) is not dict or value.get('root') not in {'model', 'dataset'}:
            raise InputError('Unsupported adapter original root')
        record = _file_record({key: value.get(key) for key in ('path', 'sha256', 'size_bytes')})
        key = value['root'], record['path']
        if key in allowed:
            raise InputError('Duplicate adapter original receipt')
        root = Path(prepared[value['root']])
        if _hash(_inside(root, record['path'])) != (record['sha256'], record['size_bytes']):
            raise InputError('Adapter original receipt contradicts staged bytes')
        allowed[key] = dict(record, root=value['root'], source_root=root, source_path=record['path'])
    if sum(record['size_bytes'] for record in allowed.values()) > 256 * 1024**2:
        raise InputError('Adapter original closure exceeds its supported byte bound')
    return allowed


def _payload(stage, completion, samples, operation):
    if (type(completion) is not dict or completion.get('provider') != PROVIDER
            or completion.get('operation') != operation
            or completion.get('execution_status') not in {'complete', 'partial', 'error'}):
        raise InputError('Invalid native adapter worker completion')
    values = completion.get('payload_files')
    if type(values) is not list or len(values) != len(PAYLOAD):
        raise InputError('Adapter worker must declare its exact numerical evidence closure')
    records = {}
    for value in values:
        record = _file_record(value)
        path = record['path']
        if path not in PAYLOAD or path in records or record['size_bytes'] > PAYLOAD[path]:
            raise InputError('Adapter payload contains duplicate, unsupported or excessive files')
        if _record(stage, path) != record:
            raise InputError('Adapter payload bytes contradict their descriptor')
        records[path] = record
    if set(records) != set(PAYLOAD) or completion.get('result') != records['result.json']:
        raise InputError('Adapter result descriptor contradicts its payload inventory')
    raw = _json(stage / 'result.json', MAX_RAW)
    metadata = _json(stage / 'samples/metadata.json', MAX_RAW)
    if type(raw) is not dict or raw.get('kind') != 'native_adapter_boundary_capture' or type(metadata) is not dict:
        raise InputError('Adapter result/sample metadata has an unsupported format')
    identity = raw.get('input_identity')
    if (type(identity) is not dict or any(type(identity.get(key)) is not int or identity[key] != value
                                        for key, value in samples[0].items())
            or any(type(metadata.get(key)) is not int or metadata[key] != value for key, value in samples[0].items())):
        raise InputError('Adapter worker did not preserve the explicitly requested original row')
    for key in ('global_index', 'timestamp', 'task_index'):
        if key in identity and metadata.get(key) != identity[key]:
            raise InputError('Adapter raw/sample row identity witnesses disagree')
    return raw, metadata, records


def _projection(raw, metadata):
    """Preserve all captured native payloads; only host locator assertions vary."""
    retained_metadata = dict(metadata)
    if type(metadata.get('cameras')) is dict:
        retained_metadata['cameras'] = {
            name: {key: value for key, value in camera.items() if key != 'source_file'}
            if type(camera) is dict else camera for name, camera in metadata['cameras'].items()}
    # An imported helper answer has no authority and is not execution evidence.
    return {'raw': {key: value for key, value in raw.items() if key != 'criterion'},
            'sample_metadata': retained_metadata}


def _summary(report):
    return {key: report[key] for key in ('status', 'source_criterion', 'input_digest', 'native_identity',
            'model_identity', 'native_processing_status', 'native_state_component_status', 'native_replay_status')
            if key in report}


def _view(root, stage, records, allowed, raw, metadata, contract, verifier, replay):
    root.mkdir()
    for path, record in records.items():
        _copy(stage, path, root, 'raw/result.json' if path == 'result.json' else path, record)
    bindings = []
    for (label, path), value in sorted(allowed.items()):
        relative = 'originals/' + label + '/' + path
        _copy(value['source_root'], value['source_path'], root, relative, value)
        bindings.append({'root': label, 'path': path, 'retained_path': relative,
                         'sha256': value['sha256'], 'size_bytes': value['size_bytes']})
    wrapper = {'schema_version': 1, 'kind': 'native_observation_source_capture',
               'evidence_role': 'native_adapter_dry_run', 'contract': contract,
               'result': {'path': 'raw/result.json', 'sha256': records['result.json']['sha256']},
               'sample': {key: {field: records[path][field] for field in ('path', 'sha256')}
                          for key, path in [('arrays', 'samples/arrays.npz'), ('metadata', 'samples/metadata.json')]},
               'bindings': bindings}
    _write(root / 'capture.json', wrapper)
    _write(root / 'native_verifier.json', {'schema_version': 1, 'verifier': verifier, 'replay': replay})
    _write(root / 'expected_native.json', _projection(raw, metadata))
    with (root / 'README.md').open('x') as stream:
        stream.write(SCOPE + '\n\nWhole original files and notices are retained. No redistribution grant is inferred.\n')
    report = audit_native_observation_source(root, contract=contract)
    _commit(root, VIEW)
    return report


def _response(execution_status, candidate, corrected=None, **extra):
    result = {'kind': 'native_observation_source_observation', 'status': candidate['status'],
              'execution_status': execution_status, 'candidate': _summary(candidate),
              'corrected': _summary(corrected) if corrected is not None else None, 'scope': SCOPE, **extra}
    if len(_encode(result)) > 192 * 1024:
        raise InputError('Adapter observation summary exceeds its response bound')
    return result


@_input_errors
def probe_native_observation_source(model, dataset, *, runtimes, samples, destination=None, contract=None):
    """Observe candidate and optional operator-configured reference independently."""
    runtime = _runtime(runtimes)
    samples = _selector(samples)
    contract = validate_parameters({'contract': contract})['contract']
    source_roots = [Path(value).expanduser().resolve() for value in (model, dataset)]
    source_roots += [Path(runtime[key]).expanduser().resolve() for key in
                     ('source_root', 'groot_source_root', 'reference_source_root') if key in runtime]
    target = None
    if destination is not None:
        requested = Path(destination).expanduser().absolute()
        if requested.is_symlink():
            raise InputError('Adapter destination symlink refused')
        target = requested.parent.resolve() / requested.name
        if target.exists() or any(target.is_relative_to(root) for root in source_roots):
            raise InputError('Choose a new adapter capture destination outside original and trusted code roots')
        target.parent.mkdir(parents=True, exist_ok=True)
    verifier = _verifier(runtime)
    with tempfile.TemporaryDirectory(prefix='.policy-doctor-adapter-', dir=target.parent if target else None) as folder:
        work = Path(folder)
        intake = work / 'intake'
        intake.mkdir()
        prepared = prepare_native_groot_inputs(Path(model), Path(dataset), samples, intake)
        allowed = _allowed(prepared)
        publication = work / 'publication'
        publication.mkdir()
        reports, observations = {}, {}
        roles = ['candidate'] + (['reference'] if 'reference_source_root' in runtime else [])
        for role in roles:
            role_work = work / ('worker-' + role)
            role_work.mkdir()
            request = {'operation': 'probe', 'source_root': _role_source(runtime, role),
                       'groot_source_root': runtime['groot_source_root'], 'model': str(prepared['model']),
                       'dataset': str(prepared['dataset']), 'samples': samples, 'contract': contract,
                       'output': str(role_work / 'output')}
            completion = _invoke(Path(__file__).with_name('native_groot_adapter_worker.py'), runtime, request, role_work)
            verify_native_groot_inputs(prepared['selection'])
            if _verifier(runtime) != verifier:
                raise InputError('Adapter runtime/source verifier changed during processing')
            stage = Path(request['output'])
            raw, metadata, records = _payload(stage, completion, samples, 'probe')
            view_name = 'candidate' if role == 'candidate' else 'corrected'
            reports[role] = _view(publication / view_name, stage, records, allowed, raw, metadata, contract,
                                  verifier, {'role': role, 'samples': samples, 'contract': contract})
            observations[role] = (completion['execution_status'], records['samples/arrays.npz']['sha256'], metadata)
        if 'reference' in reports:
            first, second = reports['candidate'], reports['reference']
            if (observations['candidate'][1] != observations['reference'][1]
                    or _projection({}, observations['candidate'][2]) != _projection({}, observations['reference'][2])
                    or any(first.get(key) != second.get(key) for key in ('input_digest', 'native_identity', 'model_identity'))):
                raise InputError('Candidate/reference adapter views do not share the same original available input and model')
        verify_native_groot_inputs(prepared['selection'])
        if _verifier(runtime) != verifier:
            raise InputError('Adapter runtime/source verifier changed before publication')
        if target is None:
            return _response(observations['candidate'][0], reports['candidate'], reports.get('reference'), retained=False,
                             retention_note='Temporary evidence was removed; choose a destination to retain it.')
        _commit(publication, ATTEMPT)
        publish_capture(publication, target)
        return _response(observations['candidate'][0], reports['candidate'], reports.get('reference'), retained=True,
                         directory=str(target), candidate_directory=str(target / 'candidate'),
                         corrected_directory=str(target / 'corrected') if 'reference' in reports else None)


@_input_errors
def reproduce_native_observation_source(directory, *, runtimes):
    """Replay one retained fixed role using operator code and retained arrays."""
    original = Path(directory).expanduser()
    if original.is_symlink():
        raise InputError('Native adapter view symlink root refused')
    root = original.resolve()
    attempt_root, attempt_before = None, None
    if _json(root / 'manifest.json').get('kind') == ATTEMPT:
        attempt_root, attempt_before = root, _load_committed(root, ATTEMPT)
        root = root / 'candidate'
    before = _load_committed(root, VIEW)
    metadata = _json(root / 'native_verifier.json')
    if (type(metadata) is not dict or set(metadata) != {'schema_version', 'verifier', 'replay'}
            or type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1):
        raise InputError('Adapter replay lacks fixed verifier metadata')
    replay = metadata['replay']
    if type(replay) is not dict or set(replay) != {'role', 'samples', 'contract'} or replay['role'] not in {'candidate', 'reference'}:
        raise InputError('Adapter replay must specify only its fixed role, sample and contract')
    contract = validate_parameters({'contract': replay['contract']})['contract']
    samples = _selector(replay['samples'])
    runtime = _runtime(runtimes)
    current = _verifier(runtime)
    if current != metadata['verifier']:
        return {'kind': 'native_observation_source_reproduction', 'status': 'unknown', 'verifier_matches': False,
                'candidate_evaluated': False, 'scope': SCOPE}
    source_root = _role_source(runtime, replay['role'])
    candidate = audit_native_observation_source(root, contract=contract)
    wrapper = _json(root / 'capture.json')
    if wrapper.get('contract') != contract:
        raise InputError('Adapter replay contract contradicts retained capture')
    with tempfile.TemporaryDirectory(prefix='policy-doctor-adapter-replay-') as folder:
        work = Path(folder)
        staged = work / 'originals'
        staged.mkdir()
        allowed = {}
        for binding in wrapper['bindings']:
            if type(binding) is not dict or binding.get('root') not in {'model', 'dataset'}:
                raise InputError('Adapter replay has an unsupported original root')
            key = binding['root'], binding['path']
            if key in allowed:
                raise InputError('Adapter replay has duplicate original bindings')
            _copy(root, binding['retained_path'], staged, binding['root'] + '/' + binding['path'], binding)
            allowed[key] = dict(binding, source_root=staged / binding['root'], source_path=binding['path'])
        raw_samples = {}
        for key, fixed in [('arrays', 'samples/arrays.npz'), ('metadata', 'samples/metadata.json')]:
            record = wrapper['sample'][key]
            if record.get('path') != fixed or record.get('sha256') != _hash(_inside(root, fixed))[0]:
                raise InputError('Adapter replay sample descriptor contradicts its fixed retained role')
            private = work / fixed
            _copy(root, fixed, work, fixed, _record(root, fixed))
            raw_samples[key] = str(private)
        request = {'operation': 'replay', 'source_root': source_root, 'groot_source_root': runtime['groot_source_root'],
                   'model': str(staged / 'model'), 'samples': samples, 'contract': contract,
                   'raw_samples': [raw_samples], 'output': str(work / 'output')}
        completion = _invoke(Path(__file__).with_name('native_groot_adapter_worker.py'), runtime, request, work)
        stage = Path(request['output'])
        raw, sample_metadata, records = _payload(stage, completion, samples, 'replay')
        expected = _json(root / 'expected_native.json', 2 * MAX_RAW)
        matches = _encode(_projection(raw, sample_metadata)) == _encode(expected)
        observed = _view(work / 'replayed', stage, records, allowed, raw, sample_metadata, contract, current, replay)
        matches = matches and observed['source_criterion'] == candidate['source_criterion']
        if (before != _inventory(root) or current != _verifier(runtime)
                or (attempt_root is not None and attempt_before != _inventory(attempt_root))):
            raise InputError('Adapter evidence/runtime changed during native replay')
        return _response(completion['execution_status'], observed, kind='native_observation_source_reproduction',
                         status='pass' if matches else 'error', reproduction_matches=matches,
                         verifier_matches=True, candidate_evaluated=True, replay_role=replay['role'])
