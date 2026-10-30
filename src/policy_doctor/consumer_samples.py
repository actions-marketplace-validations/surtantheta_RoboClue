"""Selected numeric windows from ordinary GR00T/LeRobot v2.1 artifacts.

Internal data-only reader for the named G1 context. This does not decode images,
execute native code, authenticate a training run or assert full-sample parity.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

from .artifacts import InputError
from .consumer_context import (
    PROFILE,
    _authorized_root,
    _json,
    load_consumer_numeric_parameters,
)
from .native_intake import _directory_fd, _open_file, _stamp

DATA_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
MAX_METADATA = 1024 * 1024
MAX_PARQUET = 8 * 1024 * 1024
MAX_EPISODE_ROWS = 4096


def _read(root, relative, limit):
    fd, before = _open_file(root, relative)
    try:
        if before.st_size > limit:
            raise InputError("Selected consumer artifact exceeds its byte bound.")
        chunks, size = [], 0
        while size <= limit:
            body = os.read(fd, min(65536, limit + 1 - size))
            if not body:
                break
            chunks.append(body)
            size += len(body)
        if size != before.st_size or _stamp(before) != _stamp(os.fstat(fd)):
            raise InputError("Selected consumer artifact changed during intake.")
        raw = b"".join(chunks)
        return raw, {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": size,
        }, _stamp(before)
    finally:
        os.close(fd)


def _integer(value, maximum, name):
    if type(value) is not int or not 0 <= value <= maximum:
        raise InputError(name + " must be a bounded nonnegative integer.")
    return value


def _episode(raw, requested):
    lines = raw.splitlines()
    if not 1 <= len(lines) <= 4096:
        raise InputError("Episode metadata exceeds the supported bounded v2.1 layout.")
    seen, selected = set(), None
    for position, line in enumerate(lines):
        row = _json(line)
        eid = _integer(row.get("episode_index"), 999999, "Stored episode ID")
        length = _integer(row.get("length"), MAX_EPISODE_ROWS, "Episode length")
        if eid in seen or length == 0:
            raise InputError("Episode metadata contains duplicate IDs or empty episodes.")
        seen.add(eid)
        if eid == requested:
            selected = position, length
    if selected is None:
        raise InputError("The selected stored episode ID is absent from episodes.jsonl.")
    return selected


def _fields(descriptor, modality, info):
    result = []
    for field in descriptor["fields"]:
        kind, key, width = field["modality"], field["key"], field["input_dim"]
        groups = modality.get(kind)
        group = groups.get(key) if type(groups) is dict else None
        if type(group) is not dict:
            raise InputError("A selected numeric group is missing from modality.json.")
        start, end = group.get("start"), group.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= 512 or end - start != width:
            raise InputError("A selected modality slice disagrees with the supported field width.")
        column = group.get("original_key", "observation.state" if kind == "state" else "action")
        if type(column) is not str or not column or len(column) > 256:
            raise InputError("A selected original column name is unsupported.")
        feature = info.get("features", {}).get(column)
        if (type(feature) is not dict or feature.get("dtype") not in ("float32", "float64")
                or type(feature.get("shape")) is not list or len(feature["shape"]) != 1
                or type(feature["shape"][0]) is not int or not end <= feature["shape"][0] <= 512):
            raise InputError("Selected numeric columns require bounded floating vector features.")
        result.append({"modality": kind, "key": key, "column": column,
                       "start": start, "end": end, "width": feature["shape"][0]})
    return result


def load_consumer_sample(dataset_directory, context, *, episode_index, anchor_index,
                         allow_padding=False, read_roots=None):
    """Return an internal raw sample and separate selected-source evidence.

    episode_index is the stored ID in episodes.jsonl, not its list position.
    anchor_index and selection indices are physical row positions. Native training
    start eligibility is reported separately from explicit tail clamping.
    """
    root = _authorized_root(dataset_directory, read_roots)
    episode_index = _integer(episode_index, 999999, "episode_index")
    anchor_index = _integer(anchor_index, MAX_EPISODE_ROWS - 1, "anchor_index")
    if type(allow_padding) is not bool:
        raise InputError("allow_padding must be a strict boolean.")
    # Reauthorize and reread actual saved artifacts rather than trusting a pasted
    # descriptor, parameters or old file identities supplied by the caller.
    parameters = load_consumer_numeric_parameters(context, read_roots=read_roots)
    descriptor = parameters["descriptor"]
    if (type(descriptor) is not dict or descriptor.get("profile") != PROFILE
            or descriptor.get("numeric_context_complete") is not True):
        raise InputError("Only the named supported G1 context can select samples.")
    captured = {}

    def capture(relative, limit):
        raw, artifact, stamp = _read(root, relative, limit)
        captured[relative] = (artifact, stamp, limit)
        return raw

    root_fd = None
    try:
        root_fd = _directory_fd(root)
        root_stamp = _stamp(os.fstat(root_fd))
        info = _json(capture("meta/info.json", MAX_METADATA))
        modality = _json(capture("meta/modality.json", MAX_METADATA))
        if (info.get("codebase_version") != "v2.1" or info.get("data_path") != DATA_TEMPLATE
                or type(info.get("features")) is not dict):
            raise InputError("Only the explicit supported v2.1 episode-file layout is assessed.")
        chunk_size = _integer(info.get("chunks_size"), 1000000, "chunks_size")
        if chunk_size == 0:
            raise InputError("chunks_size must be positive.")
        position, length = _episode(capture("meta/episodes.jsonl", MAX_METADATA), episode_index)
        if anchor_index >= length or (not allow_padding and anchor_index + 49 >= length):
            raise InputError("Selected 50-step window exceeds the episode without explicit clamping.")
        fields = _fields(descriptor, modality, info)
        relative = DATA_TEMPLATE.format(episode_chunk=episode_index // chunk_size, episode_index=episode_index)
        raw = capture(relative, MAX_PARQUET)
        # Guard immutable captured bytes before decoding values. The private temp
        # file is removed on every exit; the source tree is never written.
        from .native_parquet_guard import guard_file

        with tempfile.TemporaryDirectory(prefix="roboclue-selected-numeric-") as temporary:
            copy = Path(temporary) / "selected.parquet"
            copy.write_bytes(raw)
            guard = guard_file(copy)
        if guard["rows"] != length or guard["rows"] > MAX_EPISODE_ROWS:
            raise InputError("Selected Parquet row count disagrees with episode metadata.")
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(io.BytesIO(raw), thrift_string_size_limit=2 * 1024 ** 2,
                                 thrift_container_size_limit=100000)
        schema = parquet.schema_arrow
        if len(set(schema.names)) != len(schema.names):
            raise InputError("Duplicate Parquet column names are unsupported.")
        columns = sorted({field["column"] for field in fields} | {"episode_index", "frame_index", "index"})
        if any(column not in schema.names for column in columns):
            raise InputError("Selected numeric or identity column is missing.")
        for field in fields:
            dtype = schema.field(field["column"]).type
            if (not (pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype))
                    or not pa.types.is_floating(dtype.value_type)):
                raise InputError("Selected Parquet numeric columns must contain floating vectors.")
        for column in ("episode_index", "frame_index", "index"):
            if not pa.types.is_integer(schema.field(column).type):
                raise InputError("Selected row identity columns require native integer scalars.")
        table = parquet.read(columns=columns, use_threads=False).to_pydict()
        if table["episode_index"] != [episode_index] * length or table["frame_index"] != list(range(length)):
            raise InputError("Selected episode/frame identities disagree with the physical episode rows.")
        if any(type(value) is not int or value < 0 for value in table["index"]):
            raise InputError("Selected global row indices must be nonnegative integer scalars.")
        action_indices = [min(anchor_index + offset, length - 1) for offset in range(50)]
        sample = {"states": {}, "actions": {}}
        for field in fields:
            indices = [anchor_index] if field["modality"] == "state" else action_indices
            values = [table[field["column"]][index] for index in indices]
            if any(type(value) is not list or len(value) != field["width"] for value in values):
                raise InputError("Selected floating vector width disagrees with feature metadata.")
            array = np.asarray([value[field["start"]:field["end"]] for value in values])
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                raise InputError("Selected numeric source values are not finite floating vectors.")
            with np.errstate(over="ignore", invalid="ignore"):
                selected = array.astype(np.float32)
            if not np.isfinite(selected).all():
                raise InputError("Selected source values overflow the native float32 boundary.")
            sample[field["modality"] + "s"][field["key"]] = selected
        # Check every captured source's current pathname and content, including
        # same-sized replacements, before returning an association claim.
        for name, (artifact, stamp, limit) in captured.items():
            _, current, current_stamp = _read(root, name, limit)
            if current != artifact or current_stamp != stamp:
                raise InputError("Selected source artifact changed while assembling its window.")
        current_root = _directory_fd(root)
        try:
            if _stamp(os.fstat(current_root)) != root_stamp or _stamp(os.fstat(root_fd)) != root_stamp:
                raise InputError("Dataset directory identity changed during sample selection.")
        finally:
            os.close(current_root)
        sample["selection"] = {
            "dataset_directory": str(root), "stored_episode_index": episode_index,
            "native_episode_position": position, "anchor_index": anchor_index,
            "episode_length": length, "state_indices": [anchor_index],
            "action_indices": action_indices,
            "action_global_indices": [table["index"][i] for i in action_indices],
            "allow_padding": allow_padding,
            "clamped_occurrences": sum(anchor_index + offset >= length for offset in range(50)),
            "native_training_start_eligible": anchor_index < max(0, length - 49),
            "fields": fields,
            "readset": [artifact for artifact, _, _ in captured.values()],
        }
        evidence = {
            "kind": "selected_numeric_source", "status": "PASS",
            "scope": "Current selected physical numeric rows and declared slices only.",
            "artifacts": sample["selection"]["readset"],
            "native_code_executed": False, "media_read": False,
            "full_sample_parity": "UNKNOWN", "recording_association": "UNKNOWN",
            "historical_execution": "UNKNOWN", "physical_timing": "UNKNOWN",
        }
        return {"raw_sample": sample, "source_evidence": evidence}
    except InputError:
        raise
    except (OSError, ValueError, TypeError, KeyError, ImportError, OverflowError) as exc:
        raise InputError("Selected numeric window could not be established: " + str(exc)[:256]) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
