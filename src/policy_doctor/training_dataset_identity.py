"""Portable content identity for the materialized inputs of a bounded dataset."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from .artifacts import InputError


def _signature(details):
    return (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)


def fingerprint_training_dataset(root, *, max_files=4096, max_bytes=1024**3):
    """Hash regular files below an explicit local root, without importing a loader.

    Move/copy invariant; every file is included. Hidden files, decoder caches and
    consumer-added files are inputs too. Choose a stable materialized snapshot.
    This is observed content identity, not historical producer authentication.
    """
    if (type(max_files) is not int or not 1 <= max_files <= 100_000
            or type(max_bytes) is not int or not 1 <= max_bytes <= 16 * 1024**3):
        raise InputError('Use max_files in 1..100000 and max_bytes in 1..16 GiB')
    path = Path(root).expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise InputError('Select an existing regular dataset directory, without a root symlink')
    files, pending, entries, total, original = [], [(path, 0)], 0, 0, {}
    while pending:
        directory, depth = pending.pop()
        if depth > 32 or directory.is_symlink():
            raise InputError('Dataset identity exceeds depth 32; select a smaller snapshot')
        for child in sorted(directory.iterdir()):
            entries += 1
            if entries > 2 * max_files + 128:
                raise InputError('Dataset inventory exceeds its entry bound; narrow the snapshot')
            details = child.lstat()
            if stat.S_ISDIR(details.st_mode):
                pending.append((child, depth + 1))
            elif stat.S_ISREG(details.st_mode):
                files.append(child)
                original[child] = _signature(details)
                total += details.st_size
                if len(files) > max_files or total > max_bytes:
                    raise InputError('Dataset identity exceeds max_files/max_bytes; select a smaller snapshot or increase the explicit bound')
            else:
                raise InputError('Dataset identity refuses symlinks and special files: ' + str(child))
    if not files:
        raise InputError('Dataset snapshot has no materialized files')
    records, observed = [], {}
    for child in sorted(files):
        descriptor = os.open(child, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes
                    or _signature(before) != original[child]):
                raise InputError('Dataset file changed type or exceeds its byte bound')
            digest, count = hashlib.sha256(), 0
            while block := os.read(descriptor, min(1024**2, max_bytes + 1 - count)):
                count += len(block)
                if count > before.st_size or count > max_bytes:
                    raise InputError('Dataset file grew during identity capture')
                digest.update(block)
            after = os.fstat(descriptor)
            if _signature(before) != _signature(after) or count != before.st_size:
                raise InputError('Dataset file changed during identity capture')
            observed[child] = _signature(after)
            records.append({'path': child.relative_to(path).as_posix(), 'sha256': digest.hexdigest(), 'size_bytes': count})
        finally:
            os.close(descriptor)
    actual = []
    pending = [(path, 0)]
    count = 0
    while pending:
        directory, depth = pending.pop()
        if depth > 32 or directory.is_symlink():
            raise InputError('Dataset membership changed during identity capture')
        for child in directory.iterdir():
            count += 1
            if count > 2 * max_files + 128:
                raise InputError('Dataset membership changed during identity capture')
            s = child.lstat()
            if stat.S_ISDIR(s.st_mode):
                pending.append((child, depth + 1))
            elif stat.S_ISREG(s.st_mode):
                actual.append(child)
                if observed.get(child) != (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns):
                    raise InputError('Dataset file changed after hashing')
            else:
                raise InputError('Dataset membership changed type during identity capture')
    if set(actual) != set(files):
        raise InputError('Dataset membership changed during identity capture')
    canonical = json.dumps(records, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return {'dataset_id': 'sha256:' + hashlib.sha256(canonical).hexdigest(), 'files': records,
            'file_count': len(records), 'bytes_hashed': sum(item['size_bytes'] for item in records),
            'scope': 'All observed regular files in this materialized snapshot; no historical provenance or dataset-health claim.'}
