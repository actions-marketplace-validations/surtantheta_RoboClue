"""Bounded comparison of literal saved RoboLab configuration, without running it.

An equality result covers only explicitly selected env_cfg.json settings. HDF5
outcome labels are evidence about their own file, not cross-run episode matches.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

from .artifacts import InputError

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_ENTRIES = 100_000
MAX_FILES = 64
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_DISCOVERY_ENTRIES = 4096
MAX_OBJECTS = 4096
MAX_HDF5_DEPTH = 16
MAX_ATTRS = 64
MAX_ATTR_BYTES = 4096
MAX_TOTAL_ATTRS = 4096
MAX_TOTAL_ATTR_BYTES = 2 * 1024 * 1024
_ROOTS = {'instruction', 'actions', 'terminations', 'sim', 'decimation', 'scene', 'num_envs'}
_CALLABLE_REPR = re.compile(r'\bfunctools\.partial\b|<(?:function|bound method|built-in function|built-in method)\b|\bat 0x[0-9a-fA-F]+\b')
SCOPE = (
    'Exact canonical JSON equality of the selected literal recorded env_cfg.json settings only. '
    'Recorded outcome attributes retain within-file episode labels; nominal demo counters do not establish '
    'cross-run episode or cohort association. No trajectory alignment, dataset values, media, callable '
    'execution, simulator replay, physical success validation or causal diagnosis. '
    'Directory events and repeated file hashes detect observed changes; this is not an atomic '
    'filesystem snapshot and does not exclude changes made from other machines.'
)


class _Unknown(ValueError):
    pass


class _DirectoryEvents:
    """Retain local Linux change events independently of cached directory stats.

    A network filesystem may not report writes performed by a different client.
    The audit therefore does not claim an atomic snapshot or exclude remote writers.
    """

    def __init__(self):
        if not sys.platform.startswith('linux'):
            raise _Unknown('Native directory change monitoring requires Linux; no stable intake was established')
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        try:
            initialize = library.inotify_init1
            self._add = library.inotify_add_watch
        except AttributeError as exc:
            raise _Unknown('Native directory change monitoring is unavailable') from exc
        initialize.argtypes, initialize.restype = [ctypes.c_int], ctypes.c_int
        self._add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._add.restype = ctypes.c_int
        self._fd = initialize(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._fd < 0:
            raise _Unknown('Native directory change monitoring could not be initialized')
        self._directories = set()

    def __enter__(self):
        return self

    def watch(self, directory):
        key = os.fspath(directory)
        if key in self._directories:
            return
        if len(self._directories) >= MAX_DISCOVERY_ENTRIES:
            raise _Unknown('Native directory monitor budget exceeded')
        # MODIFY, ATTRIB, CLOSE_WRITE, moves, create/delete and self changes.
        # ONLYDIR | DONT_FOLLOW retains the discovery symlink refusal.
        mask = 0x00000FCE | 0x01000000 | 0x02000000
        if self._add(self._fd, os.fsencode(directory), mask) < 0:
            raise _Unknown('Native directory change monitoring could not cover an input directory')
        self._directories.add(key)

    def check(self):
        try:
            events = os.read(self._fd, 64 * 1024)
        except BlockingIOError:
            return
        if events:
            # Any change, queue overflow, watch removal or unmount invalidates
            # intake. We need no unbounded event parsing or retry loop.
            raise _Unknown('Native directory changed during checking; use an inactive snapshot')
        raise _Unknown('Native directory change monitoring ended before intake completed')

    def __exit__(self, *_):
        os.close(self._fd)


def validate_parameters(parameters) -> dict:
    """Validate the reusable case parameter object; references are separate inputs."""
    if not isinstance(parameters, dict) or set(parameters) != {'equal_fields'}:
        raise InputError('Native context parameters require only equal_fields')
    fields = parameters['equal_fields']
    if not isinstance(fields, list) or not 1 <= len(fields) <= 64:
        raise InputError('equal_fields must be a nonempty list of at most 64 strict JSON pointers')
    for pointer in fields:
        if not isinstance(pointer, str) or not 1 <= len(pointer) <= 1024 or not pointer.startswith('/'):
            raise InputError('equal_fields entries must be strict JSON pointers beginning with /')
        if re.search(r'~(?![01])', pointer) or any(ord(char) < 32 for char in pointer):
            raise InputError('Invalid JSON pointer escape or control character')
        if _tokens(pointer)[0] not in _ROOTS:
            raise InputError('Native context pointers must select an allowed recorded configuration root')
    if len(set(fields)) != len(fields):
        raise InputError('equal_fields must not contain duplicate pointers')
    return {'equal_fields': list(fields)}


def _tokens(pointer):
    return [part.replace('~1', '/').replace('~0', '~') for part in pointer[1:].split('/')]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _literal(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and _CALLABLE_REPR.search(item):
            return False
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return True


def _lookup(document, pointer):
    value = document
    for token in _tokens(pointer):
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r'0|[1-9][0-9]*', token) and int(token) < len(value):
            value = value[int(token)]
        else:
            return False, None
    return True, value


def _safe_path(path):
    """Refuse symlinks in the input or any existing ancestor before opening."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise _Unknown('Filesystem symlink refused: ' + str(current))
    return path


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def _open(path):
    _safe_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise _Unknown('Expected an ordinary native file: ' + str(path))
    if before.st_size > MAX_FILE_BYTES:
        raise _Unknown('Native source exceeds the 2 GiB per-file budget')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(descriptor, 'rb')
    if _signature(os.fstat(stream.fileno())) != _signature(before):
        stream.close()
        raise _Unknown('Native source changed before reading')
    return stream


def _hash(path):
    digest = hashlib.sha256()
    with _open(path) as stream:
        before = os.fstat(stream.fileno())
        count = 0
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            if count > MAX_FILE_BYTES:
                raise _Unknown('Native source grew past the byte budget')
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        _safe_path(path)
        if _signature(before) != _signature(after) or _signature(after) != _signature(path.lstat()):
            raise _Unknown('Native source changed while hashing')
    return digest.hexdigest(), count


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise _Unknown('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise _Unknown('Nonfinite JSON constant: ' + value)


def _read_config(path):
    with _open(path) as stream:
        before = _signature(os.fstat(stream.fileno()))
        content = stream.read(MAX_JSON_BYTES + 1)
        if before != _signature(os.fstat(stream.fileno())) or before != _signature(path.lstat()):
            raise _Unknown('Recorded configuration changed while reading')
    if len(content) > MAX_JSON_BYTES:
        raise _Unknown('env_cfg.json exceeds the 2 MiB JSON budget')
    document = json.loads(content.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if not isinstance(document, dict):
        raise _Unknown('env_cfg.json must contain a JSON object')
    pending, entries = [(document, 0)], 0
    while pending:
        value, depth = pending.pop()
        entries += 1
        if depth > MAX_JSON_DEPTH or entries > MAX_JSON_ENTRIES:
            raise _Unknown('Recorded JSON depth or entry budget exceeded')
        if isinstance(value, float) and not math.isfinite(value):
            raise _Unknown('Nonfinite recorded JSON number')
        if isinstance(value, dict):
            entries += len(value)
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return document


def _native_name(path):
    return (path.name == 'env_cfg.json' or path.suffix.lower() in {'.hdf5', '.h5'}
            or bool(re.fullmatch(r'log_[0-9]+\.json', path.name))
            or path.name in {'episode_results.json', 'episode_results.jsonl'})


def _discover(root, monitor):
    _safe_path(root)
    if not root.is_dir():
        raise _Unknown('Expected an existing native run directory')
    pending, paths, entries = [(root, 0)], [], 0
    directories = {}
    while pending:
        directory, depth = pending.pop()
        if depth > 8:
            raise _Unknown('Native directory depth budget exceeded')
        _safe_path(directory)
        monitor.watch(directory)
        before = _signature(directory.lstat())
        with os.scandir(directory) as children:
            for item in children:
                entries += 1
                if entries > MAX_DISCOVERY_ENTRIES:
                    raise _Unknown('Native directory entry budget exceeded')
                if item.is_symlink():
                    raise _Unknown('Filesystem symlink refused during native discovery')
                path = Path(item.path)
                if item.is_dir(follow_symlinks=False):
                    pending.append((path, depth + 1))
                elif _native_name(path):
                    paths.append(path)
                    if len(paths) > MAX_FILES:
                        raise _Unknown('Native file count budget exceeded')
        after = _signature(directory.lstat())
        if before != after:
            raise _Unknown('Native directory changed during discovery')
        directories[str(directory)] = after
    return sorted(paths), directories


def _attrs(obj, budget):
    """Read fixed, scalar attributes only; never allocate an unbounded vlen value."""
    if len(obj.attrs) > MAX_ATTRS:
        raise _Unknown('HDF5 attribute count budget exceeded')
    values, skipped = {}, []
    for name in obj.attrs:
        name_size = len(name.encode('utf-8'))
        budget['count'] += 1
        budget['bytes'] += name_size
        if name_size > MAX_ATTR_BYTES:
            raise _Unknown('HDF5 attribute name budget exceeded')
        if budget['count'] > MAX_TOTAL_ATTRS or budget['bytes'] > MAX_TOTAL_ATTR_BYTES:
            raise _Unknown('Total HDF5 attribute budget exceeded')
        attribute = obj.attrs.get_id(name)
        try:
            dtype = attribute.dtype
            if (attribute.shape != () or dtype.fields or dtype.hasobject or dtype.kind not in 'biufSU'
                    or dtype.kind == 'f' and dtype.itemsize > 8):
                skipped.append({'name': name, 'reason': 'Only bounded fixed scalar attributes decoded'})
                continue
            if dtype.itemsize > MAX_ATTR_BYTES or attribute.get_storage_size() > MAX_ATTR_BYTES:
                raise _Unknown('HDF5 scalar attribute byte budget exceeded')
            budget['bytes'] += dtype.itemsize
            if budget['bytes'] > MAX_TOTAL_ATTR_BYTES:
                raise _Unknown('Total HDF5 attribute byte budget exceeded')
            value = obj.attrs[name]
            if hasattr(value, 'item'):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            if isinstance(value, float) and not math.isfinite(value):
                raise _Unknown('Nonfinite HDF5 attribute')
            values[name] = value
        finally:
            attribute.close()
    return values, skipped


def _hdf5_metadata(path):
    import h5py

    outcomes, provenance, seen, objects = [], [], set(), 0
    attr_budget = {'count': 0, 'bytes': 0}
    with _open(path) as stream, h5py.File(stream, 'r') as handle:
        pending = [(handle, 0)]
        while pending:
            obj, depth = pending.pop()
            objects += 1
            if depth > MAX_HDF5_DEPTH or objects > MAX_OBJECTS:
                raise _Unknown('HDF5 object count or depth budget exceeded')
            address = h5py.h5o.get_info(obj.id).addr
            if address in seen:
                raise _Unknown('HDF5 hard-link alias or cycle refused')
            seen.add(address)
            attributes, skipped = _attrs(obj, attr_budget)
            if obj.name in {'/', '/data'}:
                provenance.append({'locator': obj.name, 'attrs': attributes, 'undecoded_attrs': skipped})
            if re.fullmatch(r'/data/demo_[0-9]+', obj.name):
                if not isinstance(obj, h5py.Group):
                    raise _Unknown('Native demo entry is not a group')
                outcomes.append({'episode': obj.name.rsplit('/', 1)[-1], 'locator': obj.name,
                                 'attrs': attributes, 'undecoded_attrs': skipped,
                                 'episode_association': 'within_file_only',
                                 'cross_run_episode_association': 'unknown'})
            if isinstance(obj, h5py.Dataset):
                if obj.is_virtual or obj.external:
                    raise _Unknown('HDF5 virtual or external dataset storage refused')
                creation = obj.id.get_create_plist()
                try:
                    if any(creation.get_filter(i)[0] not in {1, 2, 3, 4, 5, 6}
                           for i in range(creation.get_nfilters())):
                        raise _Unknown('HDF5 nonbuiltin filter refused')
                finally:
                    creation.close()
            elif isinstance(obj, h5py.Group):
                if len(obj) + objects + len(pending) > MAX_OBJECTS:
                    raise _Unknown('HDF5 object count budget exceeded')
                for name in obj:
                    if len(name.encode('utf-8')) > MAX_ATTR_BYTES:
                        raise _Unknown('HDF5 object name budget exceeded')
                    link = obj.get(name, getlink=True)
                    if not isinstance(link, h5py.HardLink):
                        raise _Unknown('HDF5 soft or external link refused')
                    pending.append((obj[name], depth + 1))
            else:
                raise _Unknown('Unsupported HDF5 object kind')
    return {'episodes': sorted(outcomes, key=lambda item: item['locator']), 'provenance': provenance,
            'object_count': objects, 'dataset_value_reads': 0}


def _finding(code, message, evidence=None):
    return {'code': code, 'status': 'unknown', 'message': message, 'evidence': evidence or []}


def _intake(directory):
    try:
        with _DirectoryEvents() as monitor:
            return _intake_monitored(directory, monitor)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return {'sources': [], 'configs': {}, 'outcomes': [], 'metadata': [],
                'findings': [_finding('native_intake_unknown', str(exc))]}


def _intake_monitored(directory, monitor):
    result = {'sources': [], 'configs': {}, 'outcomes': [], 'metadata': [], 'findings': []}
    try:
        supplied = Path(os.path.expanduser(os.fspath(directory)))
        if '..' in supplied.parts:
            raise _Unknown('Parent traversal in native input directory is unsupported')
        root = Path(os.path.abspath(supplied))
        paths, directories = _discover(root, monitor)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        result['findings'].append(_finding('native_intake_unknown', str(exc)))
        return result
    total, signatures = 0, {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        artifact = None
        try:
            size = path.lstat().st_size
            if total + size > MAX_TOTAL_BYTES:
                raise _Unknown('Total native source byte budget exceeded')
            signature = _signature(path.lstat())
            digest, size = _hash(path)
            if signature != _signature(path.lstat()):
                raise _Unknown('Native source changed before metadata intake')
            signatures[str(path)] = signature
            total += size
            artifact = {'path': str(path), 'relative_path': relative, 'sha256': digest, 'size_bytes': size}
            result['sources'].append(artifact)
            if path.name == 'env_cfg.json':
                result['configs'][relative] = {'document': _read_config(path), 'artifact': artifact}
            elif path.suffix.lower() in {'.hdf5', '.h5'}:
                from .native_evidence import read_context_metadata_hdf5

                metadata = read_context_metadata_hdf5(path)
                for episode in metadata['episodes']:
                    result['outcomes'].append({**artifact, **episode})
                result['metadata'].append({**artifact, **metadata})
        except (OSError, ValueError, TypeError, KeyError, ImportError, RecursionError, RuntimeError) as exc:
            result['findings'].append(_finding('native_source_unknown', str(exc), [artifact] if artifact else []))
    # Rehash after metadata access; directory events independently retain writes
    # even when enumeration and metadata return the same cached values.
    for artifact in result['sources']:
        try:
            digest, size = _hash(Path(artifact['path']))
            if (digest != artifact['sha256'] or size != artifact['size_bytes']
                    or signatures[artifact['path']] != _signature(Path(artifact['path']).lstat())):
                raise _Unknown('Native source changed during checking; retry a stable snapshot')
        except (OSError, ValueError) as exc:
            result['configs'].pop(artifact['relative_path'], None)
            result['outcomes'] = [item for item in result['outcomes'] if item['path'] != artifact['path']]
            result['metadata'] = [item for item in result['metadata'] if item['path'] != artifact['path']]
            result['findings'].append(_finding('native_source_changed', str(exc), [artifact]))
    try:
        current_paths, current_directories = _discover(root, monitor)
        if current_paths != paths or current_directories != directories:
            raise _Unknown('Native directory inventory changed during checking')
        monitor.check()
    except (OSError, ValueError, RuntimeError) as exc:
        result['findings'].append(_finding('native_inventory_changed', str(exc)))
        result['configs'].clear()
        result['outcomes'].clear()
        result['metadata'].clear()
    if not result['configs']:
        result['findings'].append(_finding('recorded_config_missing', 'Supply a stable native env_cfg.json sidecar'))
    return result


def _evidence(config, relative, pointer):
    if config is None:
        return {'relative_path': relative, 'pointer': pointer, 'present': False}
    present, value = _lookup(config['document'], pointer)
    result = {**config['artifact'], 'pointer': pointer, 'present': present}
    if present:
        encoded = _canonical(value)
        result['canonical_sha256'] = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
        if len(encoded) <= 8192:
            result['value'] = value
        else:
            result['value_preview'] = encoded[:512]
            result['value_truncated'] = True
    return result


def audit_native_run_context(directory, *, reference=None, equal_fields) -> dict:
    """Compare selected recorded literals; return a normal pass/error/unknown report."""
    fields = validate_parameters({'equal_fields': equal_fields})['equal_fields']
    candidate = _intake(directory)
    baseline = _intake(reference) if reference is not None else {
        'sources': [], 'configs': {}, 'outcomes': [], 'metadata': [],
        'findings': [_finding('reference_missing', 'Supply a reference native run folder for recorded-context equality')],
    }
    findings = [dict(item, side=side) for side, run in [('candidate', candidate), ('reference', baseline)]
                for item in run['findings']]
    comparisons = []
    for relative in sorted(set(candidate['configs']) | set(baseline['configs'])):
        before, after = baseline['configs'].get(relative), candidate['configs'].get(relative)
        for pointer in fields:
            evidence = {'reference': _evidence(before, relative, pointer),
                        'candidate': _evidence(after, relative, pointer)}
            if not all(item['present'] for item in evidence.values()):
                status, reason = 'unknown', 'Selected recorded field or matching config sidecar is missing'
            else:
                old = _lookup(before['document'], pointer)[1]
                new = _lookup(after['document'], pointer)[1]
                if not _literal(old) or not _literal(new):
                    status, reason = 'unknown', 'Callable or address-bearing representation is inert text, not a comparable literal contract'
                elif _canonical(old) != _canonical(new):
                    status, reason = 'error', 'Selected literal recorded setting differs from the reference'
                else:
                    status, reason = 'pass', 'Selected literal recorded setting equals the reference'
            comparison = {'pointer': pointer, 'config': relative, 'status': status, 'reason': reason, **evidence}
            comparisons.append(comparison)
            if status != 'pass':
                findings.append({'code': 'recorded_context_mismatch' if status == 'error' else 'recorded_context_unknown',
                                 'status': status, 'message': reason,
                                 'evidence': [item for item in evidence.values() if item.get('sha256')]})
    status = ('error' if any(item['status'] == 'error' for item in findings)
              else 'unknown' if findings or not comparisons else 'pass')
    next_action = (
        'Review each selected setting mismatch against the intended experiment; record intentional changes explicitly.'
        if status == 'error' else
        'Supply missing reference/sidecars or bounded stable supported evidence for each unknown finding.'
        if status == 'unknown' else
        'Selected saved settings match; separately verify runtime, source/assets and episode association before replay claims.'
    )
    return {
        'schema_version': 1, 'kind': 'native_run_context_audit', 'status': status, 'contract_status': status,
        'scope': SCOPE, 'equal_fields': fields, 'fields': comparisons, 'findings': findings,
        'sources': {'reference': baseline['sources'], 'candidate': candidate['sources']},
        'recorded_outcomes': {'reference': baseline['outcomes'], 'candidate': candidate['outcomes']},
        'native_metadata': {'reference': baseline['metadata'], 'candidate': candidate['metadata']},
        'unverified_attributes': {
            side: [{'path': metadata['path'], 'sha256': metadata['sha256'], 'locator': item['locator'], **attribute}
                   for metadata in run['metadata'] for item in metadata['provenance'] + metadata['episodes']
                   for attribute in item['undecoded_attrs']]
            for side, run in [('reference', baseline), ('candidate', candidate)]
        },
        'unverified': ['effective runtime configuration', 'callable implementations and source code',
                       'asset contents', 'cross-run episode/cohort association', 'physical outcome',
                       'trajectory alignment and policy/controller hidden state',
                       'atomic filesystem snapshot and changes made from other machines',
                       'undecoded variable-length/non-scalar HDF5 attributes'],
        'next_action': next_action,
        'next_actions': [{'tool': 'audit_native_run_context', 'purpose': next_action, 'arguments': None}],
        'coverage': {
            'selected_literal_fields': len(fields), 'field_results': len(comparisons),
            'source_hashes_recorded': {side: len(run['sources'])
                                       for side, run in [('reference', baseline), ('candidate', candidate)]},
            'hdf5_metadata_files': {side: len(run['metadata'])
                                   for side, run in [('reference', baseline), ('candidate', candidate)]},
            'dataset_value_reads': 0, 'media_decodes': 0, 'runtime_verified': False,
            'cross_run_episode_association_verified': False,
        },
    }
