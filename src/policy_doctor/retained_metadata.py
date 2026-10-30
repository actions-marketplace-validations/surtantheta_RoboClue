"""Select original, bounded local metadata for portable offline checks.

This is a snapshot of the existing auditor's inputs, not a dataset conversion.
Unselected model weights, recordings and dataset shards are never opened.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path

from .artifacts import MAX_JSON_BYTES, MAX_STATS_BYTES, InputError, load_checkpoint, load_dataset, read_json
from .hub import check_checkpoint

MAX_RETAINED_BYTES = 128 * 1024 * 1024
MAX_RETAINED_FILES = 205
MAX_OMITTED_ENTRIES = 128
_WEIGHT_NAMES = {"model.safetensors", "pytorch_model.bin", "model.bin"}


def validate_metadata_parameters(parameters=None) -> dict:
    """Accept only an explicit assessment scope; networking is unavailable."""
    if parameters is None:
        parameters = {}
    if type(parameters) is not dict or set(parameters) - {"scope"}:
        raise InputError("Metadata parameters must be an object containing only scope")
    scope = parameters.get("scope", "metadata")
    if type(scope) is not str or scope not in {"metadata", "deployment"}:
        raise InputError("Metadata scope must be metadata or deployment")
    return {"scope": scope}


def _local(value) -> Path:
    if not isinstance(value, (str, Path)) or not str(value) or "://" in str(value):
        raise InputError("Retained metadata requires a nonempty local path")
    path = Path(value).expanduser().absolute()
    if ".." in path.parts:
        raise InputError("Retained metadata paths must not contain parent traversal")
    _no_symlinks(path)
    return path


def _no_symlinks(path: Path) -> None:
    for item in (path, *path.parents):
        if item.is_symlink():
            raise InputError(f"Retained metadata does not follow symlinks: {item}")


def _root(source, role) -> Path:
    path = _local(source)
    if path.is_file():
        if role == "checkpoint":
            if path.name != "config.json":
                raise InputError("Checkpoint input must be a directory or config.json")
            path = path.parent
        else:
            if path.name != "info.json" or path.parent.name != "meta":
                raise InputError("Retained dataset input must be a directory or meta/info.json")
            path = path.parent.parent
    if not path.is_dir():
        raise InputError("Retained metadata source must exist")
    return path


def _stamp(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read(path: Path, maximum: int) -> bytes:
    _no_symlinks(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise InputError(f"Retained metadata must be a bounded regular file: {path}")
            data = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
            _no_symlinks(path)
            current = path.stat(follow_symlinks=False)
            if (len(data) > maximum or _stamp(before) != _stamp(after)
                    or _stamp(after) != _stamp(current)):
                raise InputError(f"Metadata changed while being retained: {path}")
            return data
    except OSError as exc:
        raise InputError(f"Cannot safely retain metadata {path}: {exc}") from exc


def _relative(root, value):
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts or "\\" in value or "://" in value):
        raise InputError("Processor state references must stay within the checkpoint")
    path = root / value
    _no_symlinks(path)
    return path


def _normal(value, root):
    if isinstance(value, dict):
        return {_normal(key, root): _normal(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normal(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<metadata-root>")
    return value


def _semantic(snapshot):
    return {"summary": snapshot.summary(), "config": snapshot.config,
            "pre": snapshot.pre, "post": snapshot.post,
            "json_documents": snapshot.json_documents}


def evaluate_metadata(checkpoint, dataset=None, parameters=None) -> dict:
    """Run the existing auditor locally, without changing its verdict semantics."""
    options = validate_metadata_parameters(parameters)
    source = _root(checkpoint, "checkpoint")
    other = _root(dataset, "dataset") if dataset is not None else None
    return check_checkpoint(str(source), str(other) if other is not None else None,
                            fetch=False, **options)


def snapshot_metadata(source, destination, *, role="checkpoint") -> dict:
    """Copy only native auditor metadata and prove the copied loader view agrees.

    The caller should additionally compare the complete paired audit before and
    after snapshotting. Original paths are provenance, not portable entry points.
    """
    if type(role) is not str or role not in {"checkpoint", "dataset"}:
        raise InputError("Metadata role must be checkpoint or dataset")
    root = _root(source, role)
    target = _local(destination)
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise InputError("Retained destination must be separate from its source")
    if target.exists():
        raise InputError("Retained metadata destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    selected = {}
    identities = {}
    missing = []
    try:
        initial = ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"] if role == "checkpoint" else ["meta/info.json", "meta/stats.json"]
        required = initial[0]
        pending = [(name, MAX_JSON_BYTES) for name in initial]
        for name, maximum in pending:
            path = _relative(root, name)
            if not path.exists():
                if name == required:
                    raise InputError(f"Required metadata is absent: {name}")
                missing.append(name)
                continue
            content = _read(path, maximum)
            if len(selected) >= MAX_RETAINED_FILES or sum(len(x) for x in selected.values()) + len(content) > MAX_RETAINED_BYTES:
                raise InputError("Retained metadata exceeds the bounded snapshot budget")
            selected[name] = content
            identity = _stamp(path.stat(follow_symlinks=False))
            if any(other[:2] == identity[:2] for other in identities.values()):
                raise InputError("Distinct metadata paths alias the same native file")
            identities[name] = identity
            copied = target / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(content)
            if role == "checkpoint" and name in initial[1:]:
                data = read_json(copied)
                steps = data.get("steps")
                if not isinstance(steps, list) or len(steps) > 100:
                    raise InputError("Processor steps must be a list with at most 100 entries")
                for entry in steps:
                    if not isinstance(entry, dict):
                        raise InputError("Processor steps must be objects")
                    reference = entry.get("state_file")
                    if reference:
                        state = _relative(root, reference)
                        if state.name in _WEIGHT_NAMES:
                            raise InputError("Model weight files cannot be retained as processor stats")
                        if state.exists() and state.suffix != ".safetensors":
                            raise InputError("Processor state must be bounded safetensors")
                        if reference not in {item[0] for item in pending}:
                            pending.append((reference, MAX_STATS_BYTES))
        loader = load_checkpoint if role == "checkpoint" else load_dataset
        copied_view = loader(target)
        # Each selected byte must be an input actually recorded by the native
        # loader. Future loader changes therefore fail closed, not silently drift.
        recorded = {Path(item.path).relative_to(target).as_posix(): item.sha256 for item in copied_view.artifacts}
        expected = {name: hashlib.sha256(data).hexdigest() for name, data in selected.items()}
        if recorded != expected:
            raise InputError("Native metadata read set differs from the retained selection")
        original_view = loader(root)
        if _normal(_semantic(original_view), root) != _normal(_semantic(copied_view), target):
            raise InputError("Copied metadata changes the native loader interpretation")
        for name, content in selected.items():
            if (_read(root / name, MAX_STATS_BYTES) != content
                    or _stamp((root / name).stat(follow_symlinks=False)) != identities[name]):
                raise InputError("Source metadata changed during snapshot creation")
        for name in missing:
            if (root / name).exists() or (root / name).is_symlink():
                raise InputError("Source metadata appeared during snapshot creation")
        omitted, truncated = [], False
        with os.scandir(root) as entries:
            for item in entries:
                if item.name in {Path(name).parts[0] for name in selected}:
                    continue
                if len(omitted) >= MAX_OMITTED_ENTRIES:
                    truncated = True
                    break
                omitted.append({"name": item.name, "kind": "directory" if item.is_dir(follow_symlinks=False) else "symlink" if item.is_symlink() else "other"})
        return {"role": role, "entry": ".", "source_root": str(root),
                "retained_files": [{"path": name, "sha256": expected[name], "size_bytes": len(data)} for name, data in sorted(selected.items())],
                "missing_metadata": sorted(missing), "loader_parity": True,
                "omitted_entries": sorted(omitted, key=lambda item: item["name"]),
                "omission_scope": "top_level_only; no recursive payload inventory or payload reads",
                "omitted_entries_truncated": truncated}
    except OSError as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise InputError(f"Cannot retain metadata safely: {exc}") from exc
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
