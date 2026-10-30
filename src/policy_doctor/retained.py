"""Code-free portable checks over retained native metadata and numeric recordings.

A replay pass establishes repeatable diagnostics, including an original UNKNOWN.
It never establishes the solved-incident lifecycle or physical correctness.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from . import __version__
from .artifacts import InputError, read_json
from .incidents import MAX_FILE_BYTES, _hash, _inside, _inventory
from .publication import publish_capture
from .rollouts import compare_rollouts

KIND = 'robotics_retained_check'
CHECKS = {'policy_metadata', 'recording_equivalence'}
SCOPE = ('Repeatable offline diagnostics over retained native evidence. Reproduction PASS means the same '
         'complete report under the recorded verifier, including any original UNKNOWN or DIFFERENT. '
         'It does not establish a solved incident, whole-dataset compatibility, physical repair or causation.')
_MANIFEST = {'schema_version', 'kind', 'check', 'parameters', 'candidate_role', 'roles', 'files', 'verifier'}
_NUMERIC_PARAMETERS = {'alignment_key', 'atol', 'rtol', 'max_rows', 'reference_episode', 'candidate_episode', 'streams'}


def _encode(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, RecursionError) as exc:
        raise InputError('Retained checks require finite JSON data') from exc


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise InputError('Require a nonempty local evidence path')
    path = Path(value).expanduser().absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise InputError('Retained evidence symlinks are refused')
    return path.resolve()


def _parameters(check, parameters):
    if parameters is None:
        parameters = {}
    if type(parameters) is not dict:
        raise InputError('Retained parameters must be an object')
    if check == 'policy_metadata':
        from .retained_metadata import validate_metadata_parameters
        return validate_metadata_parameters(parameters)
    if set(parameters) - _NUMERIC_PARAMETERS:
        raise InputError('Unsupported recording comparison parameters')
    result = {'alignment_key': None, 'atol': 1e-6, 'rtol': 1e-5, 'max_rows': 100000,
              'reference_episode': None, 'candidate_episode': None, 'streams': None} | deepcopy(parameters)
    if (type(result['max_rows']) is not int or not 1 <= result['max_rows'] <= 100000 or
            any(type(result[k]) not in (int, float) or not math.isfinite(result[k]) or result[k] < 0 for k in ('atol', 'rtol'))):
        raise InputError('Require bounded max_rows and finite nonnegative tolerances')
    key, streams = result['alignment_key'], result['streams']
    if key is not None and (type(key) is not str or not key or len(key) > 512):
        raise InputError('alignment_key must be a bounded nonempty stream name')
    if streams is not None and (type(streams) is not list or not streams or len(streams) > 128 or
            any(type(s) is not str or not s or len(s) > 512 for s in streams) or len(set(streams)) != len(streams)):
        raise InputError('streams must contain bounded distinct nonempty names')
    if streams is not None and key is not None and key not in streams:
        raise InputError('The alignment_key must be explicitly included in streams')
    for name in ('reference_episode', 'candidate_episode'):
        value = result[name]
        if value is not None and (type(value) not in (int, str) or isinstance(value, str) and (not value or len(value) > 512)):
            raise InputError('Episode selections must be integers or bounded nonempty native keys')
    _encode(result)
    return result


def _fingerprint(manifest):
    root = Path(__file__).parent
    modules = {p.name: _digest(p.read_bytes()) for p in sorted(root.glob('*.py'))}
    versions = {}
    dependencies = {'numpy'}
    if manifest['check'] == 'policy_metadata':
        dependencies.add('safetensors')
    else:
        suffixes = {Path(name).suffix.lower() for name in manifest['roles'].values()}
        if suffixes & {'.h5', '.hdf5'}:
            dependencies.add('h5py')
        if '.parquet' in suffixes:
            dependencies.add('pyarrow')
    for name in sorted(dependencies):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'engine_version': __version__, 'python_version': list(sys.version_info[:2]),
            'module_sha256': modules, 'dependency_versions': versions}


def _inventory_complete(root):
    inventory = _inventory(root, case=True)
    # Empty directories carry no bound payload and are not accepted as hidden extras.
    for base, directories, _ in os.walk(root, followlinks=False):
        for name in directories:
            directory = Path(base) / name
            if not any(p.startswith(directory.relative_to(root).as_posix() + '/') for p in inventory):
                raise InputError('Unbound empty evidence directories are refused')
    return inventory


def _snapshot_recording(source, destination):
    if not source.is_file() or source.suffix.lower() not in ('.npz', '.csv', '.h5', '.hdf5', '.parquet'):
        raise InputError('Require a native NPZ, CSV, HDF5 or Parquet recording file')
    stamp = source.stat()
    identity = (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns)
    before = _hash(source)
    destination.mkdir()
    output = destination / ('recording' + source.suffix.lower())
    shutil.copyfile(source, output)
    _path(source)
    stamp = source.stat()
    after_identity = (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns)
    if identity != after_identity or _hash(source) != before or _hash(output) != before:
        raise InputError('Recording changed during retention')
    return {'entry': output.name, 'original_sha256': before[0], 'size_bytes': before[1],
            'retained_format': source.suffix.lower(), 'transformed': False}


def _snapshot(check, source, destination, role):
    if check == 'policy_metadata':
        from .retained_metadata import snapshot_metadata
        receipt = snapshot_metadata(source, destination, role=role)
    else:
        receipt = _snapshot_recording(source, destination)
    if type(receipt) is not dict or type(receipt.get('entry')) is not str:
        raise InputError('Snapshot helper did not declare its native entry')
    entry = destination if receipt['entry'] == '.' else _inside(destination, receipt['entry'])
    if not entry.exists():
        raise InputError('Snapshot entry does not exist')
    return entry, receipt


def _portable(value, roots):
    if isinstance(value, str):
        for prefix, label in sorted(roots, key=lambda p: len(p[0]), reverse=True):
            if value == prefix or value.startswith(prefix + os.sep):
                return label + value[len(prefix):]
        return value
    if isinstance(value, list):
        return [_portable(v, roots) for v in value]
    if isinstance(value, dict):
        return {k: _portable(v, roots) for k, v in value.items()}
    return value


def _portable_recording(report, roles):
    """Bind native ordered witnesses to roles, even when both roles share a file.

    A single robomimic file can supply different reference/candidate episodes.
    Prefix substitution alone loses that distinction for shared source paths.
    """
    result = deepcopy(report)
    ordered = ('reference', 'candidate')
    bound = set()

    def bind(entries):
        # The native report shares the first-difference evidence list between
        # its global witness and per-stream witness. Normalize that object once.
        if id(entries) in bound:
            return
        if type(entries) is not list or len(entries) != 2:
            raise InputError('Recording auditor did not supply its two ordered artifact witnesses')
        for entry, role in zip(entries, ordered, strict=True):
            if type(entry) is not dict or entry.get('path') != str(roles[role]):
                raise InputError('Recording witness does not identify its declared native role')
            entry['path'] = role
        bound.add(id(entries))

    bind(result['artifacts'])

    def visit(value):
        if isinstance(value, dict):
            if 'evidence' in value:
                bind(value['evidence'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def _evaluate(root, manifest, replacement=None):
    roles = {role: _inside(root, name) for role, name in manifest['roles'].items()}
    if replacement is not None:
        roles[manifest['candidate_role']] = replacement
    if manifest['check'] == 'policy_metadata':
        from .retained_metadata import evaluate_metadata
        report = evaluate_metadata(roles['checkpoint'], roles.get('dataset'), manifest['parameters'])
    else:
        report = compare_rollouts(str(roles['reference']), str(roles['candidate']), **manifest['parameters'])
    normalized = (_portable_recording(report, roles) if manifest['check'] == 'recording_equivalence' else
                  _portable(report, [(str(path), role) for role, path in roles.items()]))
    if len(_encode(normalized)) > MAX_FILE_BYTES:
        raise InputError('Diagnostic report exceeds the retained report byte budget')
    return normalized


def _summary(report, *, original=True):
    result = {'status': report.get('status'), 'source_report_sha256': _digest(_encode(report)),
              'detail_complete': False,
              'full_report': 'expected.json' if original else None,
              'report_scope': 'Original retained report' if original else 'Later candidate report; preview only, not persisted'}
    if 'findings' in report:
        findings = report['findings']
        result['counts'] = report.get('counts', {})
        result['assessment_scope'] = report.get('assessment_scope')
        result['deployment_status'] = report.get('deployment_status')
        result['unassessed_findings_count'] = len(report.get('unassessed_findings', []))
        # Keep complete evidence objects or omit the whole finding; never shorten its identifiers.
        shown = []
        for finding in sorted(findings, key=lambda f: {'error': 0, 'warning': 1, 'unknown': 2}.get(f.get('status'), 3)):
            if len(shown) < 3 and len(_encode(finding)) <= 2500:
                shown.append(finding)
        result.update(findings=shown, finding_count=len(findings), omitted_findings=len(findings) - len(shown))
    else:
        streams = report.get('streams', [])
        shown = [s for s in sorted(streams, key=lambda s: s.get('status') != 'different') if len(_encode(s)) <= 2500][:3]
        counts = {status: sum(s.get('status') == status for s in streams) for status in ('different', 'equal', 'incompatible')}
        result.update(comparison_complete=report.get('comparison_complete'), alignment=report.get('alignment'),
                      observed_difference=report.get('observed_difference'), stream_count=len(streams),
                      streams=shown, omitted_streams=len(streams) - len(shown),
                      counts=counts, unmatched_streams=report.get('unmatched_streams'), causal_conclusion='unknown')
        witness = report.get('first_observed_difference')
        result['first_observed_difference'] = witness if len(_encode(witness)) <= 2500 else None
        result['first_observed_difference_omitted'] = len(_encode(witness)) > 2500
    if len(_encode(result)) > 12000:
        # A huge unmatched-stream list must not silently turn into a complete report.
        result = {'status': report.get('status'), 'source_report_sha256': _digest(_encode(report)),
                  'detail_complete': False, 'preview_omitted': True, 'reason': 'Report preview exceeds byte budget',
                  'counts': result.get('counts', {}), 'full_report': 'expected.json' if original else None,
                  'comparison_complete': report.get('comparison_complete'),
                  'observed_difference': report.get('observed_difference'),
                  'report_scope': 'Original retained report' if original else 'Later candidate report; preview only, not persisted'}
    return result


def _bound(report):
    if len(_encode(report)) > 16384:
        raise InputError('Retained-check response exceeds 16 KiB preview budget')
    return report


def retain_check(check, source, destination, *, reference=None, dataset=None, candidate_role=None, parameters=None):
    """Retain original bytes and a complete diagnostic report, including UNKNOWNs."""
    if type(check) is not str or check not in CHECKS:
        raise InputError('Require policy_metadata or recording_equivalence')
    parameters = _parameters(check, parameters)
    original = _path(source)
    if check == 'policy_metadata':
        if reference is not None:
            raise InputError('Policy metadata accepts dataset, not recording reference')
        inputs = {'checkpoint': original}
        if dataset is not None:
            inputs['dataset'] = _path(dataset)
        candidate_role = ('dataset' if dataset is not None else 'checkpoint') if candidate_role is None else candidate_role
    else:
        if reference is None or dataset is not None:
            raise InputError('Recording checks require reference and do not accept dataset')
        inputs = {'candidate': original, 'reference': _path(reference)}
        candidate_role = 'candidate' if candidate_role is None else candidate_role
    if type(candidate_role) is not str or candidate_role not in inputs or (check == 'recording_equivalence' and candidate_role != 'candidate'):
        raise InputError('Candidate role must name a retained replaceable input')
    target = _path(destination)
    if target.exists():
        raise InputError('Retained-check destination already exists')
    if any(target == p or p.is_dir() and target.is_relative_to(p) for p in inputs.values()):
        raise InputError('Retained destination cannot be nested inside evidence')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.policy-doctor-retained-', dir=target.parent))
    try:
        roles, receipts = {}, {}
        for role, path in inputs.items():
            entry, receipt = _snapshot(check, path, staging / role, role)
            roles[role] = entry.relative_to(staging).as_posix()
            receipts[role] = receipt
        manifest = {'schema_version': 1, 'kind': KIND, 'check': check, 'parameters': parameters,
                    'candidate_role': candidate_role, 'roles': roles}
        manifest['verifier'] = _fingerprint(manifest)
        observed = _evaluate(staging, manifest)
        # Snapshot selection must preserve the original audit, including negatives.
        # Metadata receipts normalize accepted config.json/meta/info.json aliases.
        if check == 'policy_metadata':
            from .retained_metadata import evaluate_metadata
            original_report = evaluate_metadata(inputs['checkpoint'], inputs.get('dataset'), parameters)
            original_roots = [(str(receipts[role].get('source_root', path)), role) for role, path in inputs.items()]
            # A run pack may contain both config.json and meta/info.json. Its
            # precise native readsets distinguish the two roles at one root.
            original_files = {}
            for role, receipt in receipts.items():
                for entry in receipt['retained_files']:
                    path = str(Path(receipt['source_root']) / entry['path'])
                    label = role + '/' + entry['path']
                    if path in original_files and original_files[path] != label:
                        raise InputError('Metadata roles share an ambiguous native file locator')
                    original_files[path] = label
            original_roots.extend(original_files.items())
        else:
            original_report = compare_rollouts(str(inputs['reference']), str(inputs['candidate']), **parameters)
            original_roots = [(str(path), role) for role, path in inputs.items()]
        original_normalized = (_portable_recording(original_report, inputs) if check == 'recording_equivalence' else
                               _portable(original_report, original_roots))
        if _encode(original_normalized) != _encode(observed):
            raise InputError('Retained snapshot does not reproduce the original complete diagnostic report')
        if _fingerprint(manifest) != manifest['verifier']:
            raise InputError('Verifier changed during retention')
        (staging / 'expected.json').write_bytes(_encode(observed))
        (staging / 'provenance.json').write_bytes(_encode(receipts))
        inventory = _inventory_complete(staging)
        manifest['files'] = {p: v[0] for p, v in sorted(inventory.items())}
        raw = _encode(manifest)
        (staging / 'manifest.json').write_bytes(raw)
        (staging / 'READY.json').write_bytes(_encode({'schema_version': 1, 'manifest_sha256': _digest(raw)}))
        # Validate the fully committed shape before making the directory visible.
        _load(staging)
        response = _bound({'schema_version': 1, 'kind': 'retained_check_export', 'status': 'pass',
                           'check': check, 'original_status': observed['status'], 'candidate_role': candidate_role,
                           'directory': str(target), 'file_count': len(inventory), 'check_result': _summary(observed),
                           'scope': SCOPE})
        publish_capture(staging, target)
        return response
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load(root):
    inventory = _inventory_complete(root)
    manifest, ready = read_json(root / 'manifest.json'), read_json(root / 'READY.json')
    if (type(manifest) is not dict or set(manifest) != _MANIFEST or
            type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest['kind'] != KIND or
            type(manifest['check']) is not str or manifest['check'] not in CHECKS):
        raise InputError('Malformed retained-check manifest')
    if (type(ready) is not dict or set(ready) != {'schema_version', 'manifest_sha256'} or
            type(ready['schema_version']) is not int or ready['schema_version'] != 1 or
            ready['manifest_sha256'] != inventory['manifest.json'][0]):
        raise InputError('Retained-check READY digest mismatch')
    if _parameters(manifest['check'], manifest['parameters']) != manifest['parameters']:
        raise InputError('Retained parameters must record all defaults')
    roles, files = manifest['roles'], manifest['files']
    expected_roles = ({'checkpoint'}, {'checkpoint', 'dataset'}) if manifest['check'] == 'policy_metadata' else ({'reference', 'candidate'},)
    if type(roles) is not dict or set(roles) not in expected_roles:
        raise InputError('Unexpected retained input roles')
    if type(manifest['candidate_role']) is not str or manifest['candidate_role'] not in roles or (manifest['check'] == 'recording_equivalence' and manifest['candidate_role'] != 'candidate'):
        raise InputError('Invalid retained candidate role')
    for role, name in roles.items():
        path = _inside(root, name)
        if not (name == role or name.startswith(role + '/')) or not path.exists():
            raise InputError('Retained native entry escaped its declared role')
    if type(files) is not dict or not {'expected.json', 'provenance.json'}.issubset(files):
        raise InputError('Missing retained reports or provenance')
    if set(inventory) != set(files) | {'manifest.json', 'READY.json'}:
        raise InputError('Unlisted or missing retained evidence')
    for name, digest in files.items():
        _inside(root, name)
        if type(digest) is not str or digest != inventory[name][0]:
            raise InputError('Retained evidence digest mismatch')
        if name not in ('expected.json', 'provenance.json') and not any(name.startswith(role + '/') for role in roles):
            raise InputError('Unexpected retained payload location')
    if type(manifest['verifier']) is not dict:
        raise InputError('Malformed retained verifier')
    return manifest, inventory


def _replay(root):
    manifest, before = _load(root)
    verifier_matches = _fingerprint(manifest) == manifest['verifier']
    expected = read_json(root / 'expected.json')
    if type(expected) is not dict or type(expected.get('status')) is not str:
        raise InputError('Malformed expected diagnostic report')
    observed = _evaluate(root, manifest) if verifier_matches else None
    matches = observed is not None and _encode(observed) == _encode(expected)
    if verifier_matches and _fingerprint(manifest) != manifest['verifier']:
        raise InputError('Verifier changed during reproduction')
    if _inventory_complete(root) != before:
        raise InputError('Retained evidence changed during reproduction')
    result = _bound({'schema_version': 1, 'kind': 'retained_check_reproduction',
                     'status': 'unknown' if not verifier_matches else 'pass' if matches else 'error',
                     'check': manifest['check'], 'original_status': expected['status'],
                     'reproduction_matches': matches, 'verifier_matches': verifier_matches,
                     'recorded_report_sha256': _digest(_encode(expected)),
                     'current_report_sha256': _digest(_encode(observed)) if observed is not None else None,
                     'check_result': _summary(observed if observed is not None else expected), 'scope': SCOPE})
    return result, manifest, before


def reproduce_retained_check(directory):
    """Compare complete retained findings under the same verifier without writes."""
    root = _path(directory)
    if not root.is_dir():
        raise InputError('Require a retained-check directory')
    return _replay(root)[0]


def check_retained_candidate(directory, candidate):
    """Apply only a reproduced retained check to one later native input."""
    root = _path(directory)
    replay, manifest, before = _replay(root)
    if replay['status'] != 'pass':
        return _bound(dict(replay, kind='retained_candidate_check', status='unknown',
                           candidate_evaluated=False, candidate_status=None))
    source = _path(candidate)
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        raise InputError('Later candidate must be outside the retained check')
    if manifest['check'] == 'recording_equivalence':
        dependency = {'.h5': 'h5py', '.hdf5': 'h5py', '.parquet': 'pyarrow'}.get(source.suffix.lower())
        if dependency and dependency not in manifest['verifier']['dependency_versions']:
            raise InputError('Later format requires an unrecorded verifier dependency; retain a new check for that format')
    with tempfile.TemporaryDirectory(prefix='policy-doctor-later-') as folder:
        copied, _ = _snapshot(manifest['check'], source, Path(folder) / manifest['candidate_role'], manifest['candidate_role'])
        observed = _evaluate(root, manifest, copied)
    if _inventory_complete(root) != before:
        raise InputError('Retained evidence changed during candidate checking')
    if _fingerprint(manifest) != manifest['verifier']:
        raise InputError('Verifier changed during candidate checking')
    return _bound({'schema_version': 1, 'kind': 'retained_candidate_check', 'status': observed['status'],
                   'check': manifest['check'], 'candidate_role': manifest['candidate_role'],
                   'original_status': replay['original_status'], 'candidate_status': observed['status'],
                   'candidate_evaluated': True, 'reproduction_matches': True, 'verifier_matches': True,
                   'check_result': _summary(observed, original=False), 'scope': SCOPE})
