"""No-replace runtime snapshot publication with a verifiable commit marker."""
from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
from pathlib import Path

from .artifacts import InputError


def publish_capture(staging: Path, destination: Path) -> None:
    """Publish without replacing existing data, including concurrent empty dirs.

    Linux renameat2 is preferred. Unsupported filesystems use exclusive directory
    and file creation, copying READY.json last. Readers must verify that marker.
    """
    if sys.platform == 'linux':
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, 'renameat2', None)
        if rename is not None:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if not rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
                return
            error = ctypes.get_errno()
            if error in (errno.EEXIST, errno.ENOTEMPTY):
                raise InputError('Capture destination already exists; no files were replaced')
            if error not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                raise InputError(f'No-replace publication failed: {os.strerror(error)}')
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise InputError('Capture destination already exists; no files were replaced') from exc
    created_files, created_dirs = [], {destination}
    try:
        sources = sorted((p for p in staging.rglob('*') if p.is_file()),
                         key=lambda p: (p.name == 'READY.json', str(p)))
        for source in sources:
            target = destination / source.relative_to(staging)
            if not target.parent.exists():
                target.parent.mkdir(parents=True)
                created_dirs.add(target.parent)
            with target.open('xb') as output:
                created_files.append(target)
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, output)
        shutil.rmtree(staging)
    except BaseException:
        # Remove only files this transaction created; preserve unexpected data.
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in sorted(created_dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        raise
