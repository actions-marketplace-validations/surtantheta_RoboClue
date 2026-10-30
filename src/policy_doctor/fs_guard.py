"""Refuse files that can block a reader (FIFOs, sockets, devices) by name, before anything opens them.

Opening a FIFO for reading blocks until a writer appears, so a dataset that holds one would hang a check and, under
MCP, freeze the whole server. These helpers never open a non-regular file; they only lstat/stat it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .artifacts import InputError

MAX_ENTRIES = 200_000


class DatasetLayoutError(InputError):
    """A named dataset link cannot be read under this check's containment contract."""

    def __init__(self, message, tree):
        super().__init__(message)
        self.tree = tree


def refuse_dataset_link(root, link):
    """Reuse the checked named-link copy/restore guidance without following a file body."""
    from .salvage import _StrictIntake

    root, link = Path(root), Path(link)
    relative = link.relative_to(root)
    prefix = ''
    try:
        if not link.resolve(strict=True).is_relative_to(root):
            prefix = 'The symlink escapes the dataset root. '
    except (OSError, RuntimeError):
        pass
    try:
        _StrictIntake(root, 1, 1).refuse_link(link)
    except InputError as exc:
        raise DatasetLayoutError(prefix + str(exc), relative.parts[0] if relative.parts else '') from exc


def special_kind(mode: int) -> str | None:
    """'FIFO', 'socket', 'character device' or 'block device' for a mode that is none of file/dir/link, else None."""
    if stat.S_ISFIFO(mode):
        return "FIFO"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character device"
    if stat.S_ISBLK(mode):
        return "block device"
    return None


def _refusal(label: str, named: list[tuple[str, str]], total: int) -> InputError:
    shown = "; ".join(f"{rel!r} is a {kind}" for rel, kind in named)
    more = f" (and {total - len(named)} more)" if total > len(named) else ""
    return InputError(f"The {label} holds a file that is not a regular file: {shown}{more}. RoboClue reads only regular "
                      "files and refuses this one so the check cannot hang. Remove it, or replace it with the real "
                      "file, then rerun.")


def refuse_special_file(path, label: str = "input") -> None:
    """Refuse one named path that exists and is a FIFO, socket or device (follows a final link, like a reader would)."""
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return  # missing or unreadable: the reader reports it
    kind = special_kind(mode)
    if kind is not None:
        raise InputError(f"The {label} {str(path)!r} is a {kind}, not a regular file. RoboClue reads only regular files "
                         "and refuses it so the check cannot hang. Pass the real file instead.")


def refuse_special_files(root, label: str = "dataset", *, max_entries: int = MAX_ENTRIES) -> None:
    """Walk a local artifact folder once (no symlinked directories, each real directory once) and refuse it, naming
    the relative paths, if it holds a FIFO, socket or device file. A link to one counts too."""
    root = Path(root)
    try:
        if not root.is_dir():
            refuse_special_file(root, label)
            return
    except OSError:
        return
    found, seen, stack, entries = [], set(), [root], 0
    while stack:
        directory = stack.pop()
        try:
            info = directory.stat()
            if (info.st_dev, info.st_ino) in seen:
                continue
            seen.add((info.st_dev, info.st_ino))
            iterator = os.scandir(directory)
        except OSError:
            continue
        with iterator:
            for entry in iterator:
                entries += 1
                if entries > max_entries:
                    raise InputError(f"The {label} contains more than {max_entries} entries; the bounded regular-file "
                                     "check is incomplete. Pass a smaller explicit dataset copy before rerunning.")
                try:
                    if entry.is_symlink():
                        try:
                            target = os.stat(entry.path)
                        except OSError:
                            continue  # dangling: the readers' own containment and existence rules apply
                        kind = special_kind(target.st_mode)
                    elif entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    else:
                        kind = special_kind(entry.stat(follow_symlinks=False).st_mode)
                except OSError:
                    continue
                if kind is not None:
                    found.append((Path(entry.path).relative_to(root).as_posix(), kind))
    if found:
        found.sort()
        raise _refusal(label, found[:3], len(found))


def open_regular(path, label: str = "file"):
    """Open a file for binary reading only if it is a regular file; a FIFO is never waited on (O_NONBLOCK)."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        mode = os.fstat(descriptor).st_mode
        kind = special_kind(mode)
        if not stat.S_ISREG(mode):
            kind = kind or ('directory' if stat.S_ISDIR(mode) else 'non-regular file')
            raise InputError(f"The {label} {str(path)!r} is a {kind}, not a regular file; RoboClue refuses it so the "
                             "check cannot hang. Pass the real file instead.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
