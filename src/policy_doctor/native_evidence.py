"""Bounded, explicitly derived native HDF5 evidence; never execute artifact code."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from .artifacts import InputError

MAX_FILES = 64
MAX_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_SIDECAR_BYTES = 2 * 1024**2
MAX_SELECTED_BYTES = 32 * 1024**2
MAX_OBJECTS = 4096
MAX_DEPTH = 16
MAX_ATTRIBUTE_BYTES = 2 * 1024**2
_BUILTIN_FILTERS = {1, 2, 3, 4, 5, 6}


def _path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for item in [path, *path.parents]:
        if item.is_symlink():
            raise InputError('Native evidence refuses symlink paths')
    return path


def _stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _hash(path: Path) -> str:
    path = _path(path)
    initial = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        if _stamp(os.fstat(stream.fileno())) != _stamp(initial):
            raise InputError('Native source changed before hashing')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        if (_stamp(os.fstat(stream.fileno())) != _stamp(initial)
                or _stamp(path.stat()) != _stamp(initial)):
            raise InputError('Native source changed while hashing')
    return digest.hexdigest()


def fingerprint_native_context(source: Path) -> dict:
    """Hash every bounded source file, including explicitly unselected files."""
    source = _path(Path(source))
    if not source.is_dir():
        raise InputError('Native context source must be a directory')
    files, total, count = [], 0, 0
    stamps = {}

    def visit(directory: Path, depth: int) -> None:
        nonlocal total, count
        stamps[directory] = _stamp(directory.stat())
        if depth > MAX_DEPTH:
            raise InputError('Native source tree exceeds depth limit')
        # Bound enumeration before allocating an arbitrary list of names.
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > MAX_FILES:
                    raise InputError('Native source tree exceeds 64 entries')
                mode = entry.stat(follow_symlinks=False).st_mode
                path = Path(entry.path)
                if stat.S_ISDIR(mode):
                    visit(path, depth + 1)
                elif stat.S_ISREG(mode):
                    info = entry.stat(follow_symlinks=False)
                    size = info.st_size
                    stamps[path] = _stamp(info)
                    total += size
                    if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                        raise InputError('Native source exceeds file/total byte budget')
                    files.append({'path': path.relative_to(source).as_posix(),
                                  'bytes': size, 'sha256': _hash(path)})
                else:
                    raise InputError('Native source refuses links and special files')
    visit(source, 0)
    if any(_stamp(path.stat()) != stamp for path, stamp in stamps.items()):
        raise InputError('Native source changed during fingerprint inventory')
    files.sort(key=lambda item: item['path'])
    return {'schema_version': 1, 'files': files, 'total_bytes': total}


def _source_files(source: Path, fingerprint: dict) -> tuple[str, list[str], list[dict]]:
    names = [item['path'] for item in fingerprint['files']]
    hdf5 = [name for name in names if '/' not in name and name.endswith(('.hdf5', '.h5'))]
    if len(hdf5) != 1 or not re.fullmatch(r'(?:data|run_[0-9]+)\.hdf5', hdf5[0]):
        raise InputError('Require exactly one native data.hdf5 or run_N.hdf5')
    if 'env_cfg.json' not in names:
        raise InputError('Native source requires original env_cfg.json')
    copied = ['env_cfg.json']
    if 'source_provenance.json' in names:
        copied.append('source_provenance.json')
    for item in fingerprint['files']:
        name = item['path']
        if '/' not in name and (re.fullmatch(r'log_[A-Za-z0-9_]+\.json', name)
                                or name.startswith('LICENSE')):
            copied.append(name)
        if name in copied and item['bytes'] > MAX_SIDECAR_BYTES:
            raise InputError('Native JSON/license sidecar exceeds 2 MiB')
    omitted = [dict(item, reason='outside_selected_native_evidence')
               for item in fingerprint['files'] if item['path'] not in copied + hdf5]
    return hdf5[0], sorted(set(copied)), omitted


def _process_hdf5(source: Path, target: Path | None) -> dict:
    """Worker body. Called in a bounded process; exposed for mechanical tests only."""
    import h5py
    import numpy as np

    result = {'file_attrs': {}, 'data_attrs': {}, 'episodes': [], 'object_count': 0,
              'selected': [], 'omitted': [], 'datasets': []}
    seen, selected_bytes, attribute_bytes = set(), 0, 0
    output = h5py.File(target, 'w') if target is not None else None

    def attributes(obj, destination):
        nonlocal attribute_bytes
        if len(obj.attrs) > 64:
            raise InputError('Too many HDF5 attributes')
        values = {}
        for name in obj.attrs:
            aid = obj.attrs.get_id(name)
            dtype, shape = aid.dtype, aid.shape
            string = h5py.check_string_dtype(dtype)
            if dtype.fields or dtype.subdtype or h5py.check_dtype(ref=dtype):
                raise InputError('Unsupported compound/reference HDF5 attribute')
            if dtype.hasobject and not (string and string.encoding == 'utf-8' and shape == ()):
                raise InputError('Only scalar UTF8 variable-length attributes are supported')
            fixed_bytes = math.prod(shape or (1,)) * dtype.itemsize
            if fixed_bytes > MAX_ATTRIBUTE_BYTES:
                raise InputError('HDF5 attribute exceeds byte budget')
            # Vlen allocation is bounded by the worker address-space/CPU limits,
            # never by H5Aget_storage_size (which measures only its descriptor).
            value = obj.attrs[name]
            if dtype.hasobject:
                raw = value.encode('utf-8') if isinstance(value, str) else bytes(value)
                raw.decode('utf-8', errors='strict')
                size = len(raw)
            else:
                size = fixed_bytes
            attribute_bytes += size
            if size > MAX_ATTRIBUTE_BYTES or attribute_bytes > MAX_ATTRIBUTE_BYTES:
                raise InputError('HDF5 attributes exceed total byte budget')
            if destination is not None:
                destination.attrs.create(name, value, shape=shape, dtype=dtype)
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='strict')
            elif isinstance(value, np.ndarray):
                if value.dtype.kind not in 'biuf':
                    raise InputError('Unsupported HDF5 array attribute')
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (str, int, float, bool, list, type(None))):
                raise InputError('Unsupported HDF5 attribute value')
            # Refuse nonfinite attrs instead of silently normalizing them.
            json.dumps(value, allow_nan=False)
            values[name] = value
        return values

    try:
        with h5py.File(source, 'r') as native:
            def visit(obj, path, depth):
                nonlocal selected_bytes
                result['object_count'] += 1
                if result['object_count'] > MAX_OBJECTS or depth > MAX_DEPTH:
                    raise InputError('HDF5 object/depth budget exceeded')
                address = h5py.h5o.get_info(obj.id).addr
                if address in seen:
                    raise InputError('HDF5 hardlink aliases/cycles are unsupported')
                seen.add(address)
                if isinstance(obj, h5py.Group):
                    dest = output.require_group(path) if output is not None else None
                    attrs = attributes(obj, dest)
                    if path == '/':
                        result['file_attrs'] = attrs
                    elif path == '/data':
                        result['data_attrs'] = attrs
                    elif re.fullmatch(r'/data/demo_[0-9]+', path):
                        result['episodes'].append({'episode': path.rsplit('/', 1)[1],
                                                   'locator': path, 'attrs': attrs})
                    if len(obj) > MAX_OBJECTS:
                        raise InputError('HDF5 child count exceeds budget')
                    for name in obj:
                        if not isinstance(obj.get(name, getlink=True), h5py.HardLink):
                            raise InputError('HDF5 soft/external links are unsupported')
                        visit(obj[name], path.rstrip('/') + '/' + name, depth + 1)
                    return
                if obj.is_virtual or obj.external:
                    raise InputError('HDF5 virtual/external storage is unsupported')
                creation = obj.id.get_create_plist()
                for index in range(creation.get_nfilters()):
                    if creation.get_filter(index)[0] not in _BUILTIN_FILTERS:
                        raise InputError('HDF5 custom compression/filter plugins are unsupported')
                dtype, shape = obj.dtype, obj.shape
                if shape is None or len(shape) > MAX_DEPTH or any(n > 1_000_000_000 for n in shape):
                    raise InputError('Unsupported HDF5 dataspace')
                logical = math.prod(shape or (1,)) * dtype.itemsize
                entry = {'path': path, 'shape': list(shape), 'dtype': dtype.str,
                         'logical_bytes': logical}
                chunk_bytes = math.prod(obj.chunks) * dtype.itemsize if obj.chunks else 0
                result['datasets'].append(entry.copy())
                reason = None
                if (len(shape) > 2 or '/image_obs/' in path or '/images/' in path
                        or any(part in {'image', 'rgb', 'depth'} for part in path.split('/'))):
                    reason = 'media_or_rank_above_two'
                elif dtype.kind not in 'biuf' or dtype.hasobject or dtype.fields or dtype.itemsize > 8:
                    reason = 'unsupported_nonnumeric_dtype'
                elif logical > MAX_SELECTED_BYTES or selected_bytes + logical > MAX_SELECTED_BYTES:
                    reason = 'selected_decoded_byte_budget'
                if reason:
                    attributes(obj, None)
                    result['omitted'].append(dict(entry, reason=reason))
                    return
                if target is None:
                    attributes(obj, None)
                    return
                if chunk_bytes > MAX_SELECTED_BYTES:
                    raise InputError('HDF5 chunk decoded bytes exceed resource budget')
                array = obj[()]
                if np.asarray(array).nbytes != logical:
                    raise InputError('HDF5 decoded shape/dtype changed')
                dest = output.create_dataset(path, data=array, dtype=dtype)
                attributes(obj, dest)
                selected_bytes += logical
                entry.update(value_sha256=hashlib.sha256(np.asarray(array).tobytes(order='C')).hexdigest(),
                             row_range=[0, shape[0]] if shape else None)
                result['selected'].append(entry)
            visit(native, '/', 0)
    finally:
        if output is not None:
            output.close()
    result['selected_decoded_bytes'] = selected_bytes
    result['scope'] = 'selected saved numeric evidence; omitted media/fields are not validated'
    return result


def _worker(source: str, target: str, report: str, mode: str = 'extract') -> None:
    if mode not in {'extract', 'context'} or mode == 'context' and target:
        raise InputError('Unsupported fixed native HDF5 worker mode')
    if mode == 'context':
        from .native_context import _hdf5_metadata
        result = _hdf5_metadata(Path(source))
    else:
        result = _process_hdf5(Path(source), Path(target) if target else None)
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    if len(data) > 4 * 1024**2:
        raise InputError('HDF5 metadata report exceeds 4 MiB')
    Path(report).write_bytes(data)


def _isolated_hdf5(source: Path, target: Path | None, *, context: bool = False) -> dict:
    if type(context) is not bool:
        raise InputError('Native worker context mode must be Boolean')
    if importlib.util.find_spec('h5py') is None:
        raise InputError('HDF5 evidence requires policy-deployment-doctor[dataset] (h5py)')
    if os.name != 'posix':
        raise InputError('Bounded native HDF5 extraction requires POSIX resource limits')
    # Trusted product code only. Artifact modules, hooks and HDF5 plugins are never loaded.
    code = ('import resource,sys; '
            'resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824)); '
            'resource.setrlimit(resource.RLIMIT_CPU,(45,45)); '
            'resource.setrlimit(resource.RLIMIT_FSIZE,(67108864,67108864)); '
            'sys.path.insert(0,sys.argv[1]); '
            'from policy_doctor.native_evidence import _worker; _worker(*sys.argv[2:])')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HDF5_PLUGIN_PATH='', HDF5_PLUGIN_PRELOAD='::')
    with tempfile.TemporaryDirectory(prefix='native-hdf5-') as temporary:
        report = Path(temporary) / 'metadata.json'
        try:
            process = subprocess.run([sys.executable, '-I', '-c', code, str(Path(__file__).resolve().parent.parent),
                                      str(source), str(target) if target else '', str(report),
                                      'context' if context else 'extract'],
                                     env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InputError(f'Bounded native HDF5 worker unavailable: {exc}') from exc
        if process.returncode or not report.is_file():
            detail = process.stderr[-1500:].decode('utf-8', errors='replace')
            if 'ModuleNotFoundError' in detail and 'h5py' in detail:
                raise InputError('Install policy-deployment-doctor[dataset] in the active isolated interpreter')
            raise InputError(f'Bounded native HDF5 worker refused input: {detail}')
        if report.stat().st_size > 4 * 1024**2:
            raise InputError('Native HDF5 worker report exceeds budget')
        return json.loads(report.read_bytes())


def read_context_metadata_hdf5(path: Path) -> dict:
    """Run the trusted conservative context checker in the same bounded worker."""
    path = _path(Path(path))
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise InputError('Native HDF5 metadata input exceeds file budget')
    initial = _stamp(path.stat())
    result = _isolated_hdf5(path, None, context=True)
    if _stamp(path.stat()) != initial:
        raise InputError('Native HDF5 changed during context metadata intake')
    return result


def read_native_metadata_hdf5(path: Path) -> dict:
    """Read bounded native metadata in isolation, with no dataset value reads."""
    path = _path(Path(path))
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise InputError('Native HDF5 metadata input exceeds file budget')
    initial = _stamp(path.stat())
    before = _hash(path)
    result = _isolated_hdf5(path, None)
    if _hash(path) != before or _stamp(path.stat()) != initial:
        raise InputError('Native HDF5 changed during metadata intake')
    return result


def snapshot_native_context(source: Path, target: Path) -> Path:
    """Create a small derived native subset, preserving original source bytes."""
    source, target = _path(Path(source)), _path(Path(target))
    if target.exists() or target == source or source in target.parents or target in source.parents:
        raise InputError('Native evidence destination must be new and outside source')
    before = fingerprint_native_context(source)
    hdf5, copied, omitted = _source_files(source, before)
    observed_paths = {source}
    for entry in before['files']:
        file = source / entry['path']
        observed_paths.add(file)
        observed_paths.update(parent for parent in file.parents if parent == source or source in parent.parents)
    initial_stats = {path: _stamp(path.stat()) for path in observed_paths}
    created = False
    try:
        target.mkdir(mode=0o700)
        created = True
        subset = _isolated_hdf5(source / hdf5, target / hdf5)
        for name in copied:
            with (source / name).open('rb') as stream:
                data = stream.read(MAX_SIDECAR_BYTES + 1)
            if len(data) > MAX_SIDECAR_BYTES:
                raise InputError('Source sidecar grew during copy')
            (target / name).write_bytes(data)
        if (fingerprint_native_context(source) != before
                or any(_stamp(path.stat()) != stamp for path, stamp in initial_stats.items())):
            raise InputError('Native source changed during evidence selection')
        # Verify actual copied sidecars, rather than relying on a source reread alone.
        for name in copied:
            expected = next(item['sha256'] for item in before['files'] if item['path'] == name)
            if _hash(target / name) != expected:
                raise InputError('Native sidecar copy drifted')
        licenses = [name for name in copied if name.startswith('LICENSE')]
        receipt = {'schema_version': 1, 'kind': 'native_evidence_selection',
                   'original_bytes_preserved': False,
                   'original_origin': 'author_assertion_unless_original_available',
                   'source_fingerprint': before, 'hdf5': subset,
                   'derived_hdf5': {'path': hdf5, 'sha256': _hash(target / hdf5)},
                   'omitted_files': omitted,
                   'licenses': {'present': licenses, 'missing': not bool(licenses),
                                'scope': 'original source notices; no license compatibility determination'},
                   'scope': 'derived numeric excerpt, not a whole recording or physical replay certificate'}
        (target / 'evidence_selection.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n')
        return target
    except Exception:
        if created:
            shutil.rmtree(target)
        raise
