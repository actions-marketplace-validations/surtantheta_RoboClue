"""Exact-byte bounded selection for the fixed original GR00T/LIBERO v2.1 recipe."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from .artifacts import InputError
from .native_groot_adapter_worker import (
    MAX_JSON,
    MODEL_FILES,
    STREAMS,
    read_json,
    selected_episode,
    selected_path,
    validate_info,
    validate_model,
    validate_samples,
)
from .native_intake import (
    MAX_NOTICE,
    NOTICE_NAMES,
    _copy_selected,
    _directory_fd,
    _entry,
    _hash_selected,
    _list_directory,
    _open_file,
)

DATA_METADATA = ('meta/info.json', 'meta/episodes.jsonl', 'meta/tasks.jsonl')


def _notices(roots):
    return {name: [n for n in NOTICE_NAMES if _entry(Path(path), n) is not None]
            for name, path in roots.items()}


def _parquet_header(root, relative):
    fd, before = _open_file(root, relative)
    try:
        if before.st_size < 12 or os.read(fd, 4) != b'PAR1':
            raise InputError('Only bounded original unencrypted Parquet is supported')
        os.lseek(fd, -8, os.SEEK_END)
        trailer = os.read(fd, 8)
        if trailer[4:] != b'PAR1' or not 1 <= int.from_bytes(trailer[:4], 'little') <= min(MAX_JSON, before.st_size - 12):
            raise InputError('Selected Parquet footer exceeds metadata bound')
    finally:
        os.close(fd)


def _private_members(root, expected):
    pending, found = [('', Path(root))], set()
    ancestors = {parent.as_posix() for name in expected for parent in Path(name).parents if parent.as_posix() != '.'}
    while pending:
        relative, path = pending.pop()
        for name, info in _list_directory(path.parent, path.name, 256):
            child = name if not relative else relative + '/' + name
            if stat.S_ISDIR(info):
                if child not in ancestors:
                    raise InputError('Unexpected directory in private native view')
                pending.append((child, path / name))
            elif child not in expected:
                raise InputError('Unexpected retained payload in private native view')
            else:
                found.add(child)
    if found != set(expected):
        raise InputError('Selected private native dependency is absent')


def prepare_native_groot_inputs(model, dataset, samples, workspace):
    """Select original files without model imports, decoding, or source conversion.

    Parquet allocation/schema guards run in the fixed native worker immediately
    before reading values; intake itself uses only byte/FD/header bounds.
    """
    try:
        sample = validate_samples(samples)
        roots = {k: Path(v).expanduser().absolute() for k, v in [('model', model), ('dataset', dataset)]}
        workspace = Path(workspace).absolute()
        for path in [*roots.values(), workspace]:
            os.close(_directory_fd(path))
        if any(workspace.is_relative_to(path) for path in roots.values()):
            raise InputError('Staging workspace must be outside original source roots')
        view = workspace / 'groot-input-view'
        view.mkdir(mode=0o700)
        staged = {k: view / k for k in roots}
        for path in staged.values():
            path.mkdir(mode=0o700)
        selection = {'provider': 'groot_adapter', 'roots': {k: str(v) for k, v in roots.items()},
                     'staged_roots': {k: str(v) for k, v in staged.items()}, 'files': [], 'total_bytes': 0,
                     'notice_membership': _notices(roots)}
        for name in MODEL_FILES:
            _copy_selected(selection, 'model', name, 'native_metadata', MAX_JSON)
        validate_model(staged['model'])
        for name in DATA_METADATA:
            _copy_selected(selection, 'dataset', name, 'native_metadata', MAX_JSON)
        info = validate_info(read_json(staged['dataset'] / 'meta/info.json'))
        selected_episode(staged['dataset'] / 'meta/episodes.jsonl', sample)
        payload = selected_path(info, 'data_path', sample['episode_index'])
        _copy_selected(selection, 'dataset', payload, 'native_episode')
        _parquet_header(staged['dataset'], payload)
        for stream in STREAMS:
            payload = selected_path(info, 'video_path', sample['episode_index'], stream['source'])
            _copy_selected(selection, 'dataset', payload, 'native_video')
        for role, names in selection['notice_membership'].items():
            for name in names:
                _copy_selected(selection, role, name, 'local_notice', MAX_NOTICE)
        verify_native_groot_inputs(selection)
        receipts = [{k: item[k] for k in ('root', 'path', 'sha256', 'size_bytes', 'role')}
                    for item in selection['files']]
        return {'model': staged['model'], 'dataset': staged['dataset'], 'receipts': receipts,
                'selection': selection,
                'omissions': ['Only the listed original metadata/selected episode/video/notice files were copied.',
                              'Selection is whole-file, not privacy redaction; unrelated payloads were not inventoried.',
                              'Original v2.1 bytes were preserved; no native-v3 reader or conversion is claimed.',
                              'Local notice presence establishes no redistribution grant.']}
    except InputError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise InputError('Unable to prepare bounded native GR00T inputs: ' + str(exc)) from exc


def verify_native_groot_inputs(selection):
    """Recheck selected original/copy bytes and FD identities, and exact private membership."""
    try:
        roots = {k: Path(v) for k, v in selection['roots'].items()}
        if _notices(roots) != selection['notice_membership']:
            raise InputError('Selected native notice membership changed')
        for role in ('model', 'dataset'):
            expected = {x['path'] for x in selection['files'] if x['root'] == role}
            _private_members(selection['staged_roots'][role], expected)
        for item in selection['files']:
            source_hash, source_stamp = _hash_selected(roots[item['root']], item['path'])
            copy_hash, copy_stamp = _hash_selected(Path(selection['staged_roots'][item['root']]), item['path'])
            if (source_hash != item['sha256'] or copy_hash != item['sha256']
                    or source_stamp != item['source_stamp'] or copy_stamp != item['copy_stamp']):
                raise InputError('Selected original/private native bytes or identity changed')
    except InputError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputError('Unable to verify selected native GR00T inputs: ' + str(exc)) from exc
