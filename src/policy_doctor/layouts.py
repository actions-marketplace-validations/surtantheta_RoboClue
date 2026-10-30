"""Bounded saved source-buffer to raw-term binding; never a live-origin attestation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .publication import publish_capture
from .rollouts import _npz, _npz_preflight

PIN = "32a1f330511130ae75ffeda96304c1bf569b5acb"
RULE = "body_incoming_wrench.force_torque_per_body.v1"
TERM = (
    "isaaclab.envs.mdp.observations",
    "body_incoming_wrench",
    "d77307eca9d62b3605a5612052942777f668720eeb1ba03f762a676feae4768a",
)
BASE = (
    "isaaclab.sensors.joint_wrench.base_joint_wrench_sensor",
    "BaseJointWrenchSensor",
    "6528a4ae4e6e215b99f6294843616f7b2a8d44be1d7d9168c38c481382469fee",
)
SENSORS = {
    (
        "isaaclab_newton.sensors.joint_wrench.joint_wrench_sensor",
        "JointWrenchSensor",
        "54ca9f0fea640fdbeb6c9895ebde135a4fcdcaa285f44f0b18427ac8a237485a",
    ),
    (
        "isaaclab_ov.sensors.joint_wrench.joint_wrench_sensor",
        "JointWrenchSensor",
        "8efe1011f4fdb46ee4ca632f0be8da7d0444e914dedb7247bff417fa5a8d3697",
    ),
    (
        "isaaclab_physx.sensors.joint_wrench.joint_wrench_sensor",
        "JointWrenchSensor",
        "fbfe0fabec7cdb878e5e6ae1a13398129cd3145b4c8c60c698a5e6c46b25ad16",
    ),
}
MAX_VALUES = 262144
MAX_ARRAY_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FILES = 8
MAX_NAMES = 64
SUFFIXES = ("force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z")
SCOPE = (
    "Saved source-buffer/raw-term consistency and explicit raw-order acceptance only. "
    "An expected order is a caller contract, not proof of a backend bug. "
    "Source hashes do not authenticate origin; physical sampling and policy concatenation remain unknown."
)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    nodes, text_bytes = 0, 0

    def walk(node, depth=0):
        nonlocal nodes, text_bytes
        nodes += 1
        if nodes > 8192:
            raise InputError("Layout metadata exceeds bounded JSON node count")
        if depth > 12:
            raise InputError("Layout metadata nesting exceeds 12")
        if type(node) is dict:
            if len(node) > 128 or any(type(key) is not str for key in node):
                raise InputError("Layout metadata requires bounded string-key mappings")
            for key, item in node.items():
                if len(key) > 256:
                    raise InputError("Layout metadata key is too long")
                text_bytes += len(key.encode("utf-8"))
                walk(item, depth + 1)
        elif type(node) is list:
            if len(node) > 512:
                raise InputError("Layout metadata list is too long")
            for item in node:
                walk(item, depth + 1)
        elif type(node) is str:
            if len(node) > 4096:
                raise InputError("Layout metadata string is too long")
            text_bytes += len(node.encode("utf-8"))
        elif node is not None and type(node) not in (int, float, bool):
            raise InputError("Layout metadata must be finite declarative JSON")
        elif type(node) in (int, float) and (not math.isfinite(node) or abs(node) > 2**63):
            raise InputError("Layout metadata numbers are nonfinite or too large")
        if text_bytes > MAX_MANIFEST_BYTES:
            raise InputError("Layout metadata text exceeds the bounded JSON budget")

    try:
        walk(value)
        raw = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"Invalid layout JSON: {exc}") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InputError("Layout JSON exceeds 128 KiB")
    return raw


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError("Duplicate layout JSON keys are refused")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
        _json(value)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"Invalid layout JSON: {exc}") from exc
    if type(value) is not dict:
        raise InputError("Layout JSON must be an object")
    return value


def _text(value):
    return type(value) is str and bool(value) and value.isprintable() and len(value.encode("utf-8")) <= 128


def _names(value, label):
    if type(value) is not list or not 1 <= len(value) <= MAX_NAMES or any(not _text(v) for v in value):
        raise InputError(f"{label} requires 1–64 bounded nonempty native names")
    if len(set(value)) != len(value):
        raise InputError(f"{label} must have unique native names")
    return list(value)


def _identity(value, *, required=False):
    if value is None and not required:
        return None
    if (
        type(value) is not dict
        or not value
        or set(value) - {"prim_path", "asset_name"}
        or not _text(value.get("prim_path"))
        or any(not _text(v) for v in value.values())
    ):
        raise InputError("Sensor identity requires a bounded prim_path and optional asset_name")
    return copy.deepcopy(value)


def _tolerances(atol, rtol):
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (atol, rtol)
    ):
        raise InputError("Layout tolerances must be finite nonnegative numbers, not booleans")


def _environment_selection(runtime, saved_rows):
    present = {"source_num_envs", "env_indices"} & set(runtime)
    if not present and not ({"omitted_env_count", "environment_scope"} & set(runtime)):
        return {
            "source_num_envs": None,
            "env_indices": None,
            "saved_rows": saved_rows,
            "omitted_envs": None,
            "scope": "Saved row positions only; shared environment association is unknown. "
            "Rows are not temporal sample IDs or evidence of a physical same-read.",
        }
    count, indices = runtime.get("source_num_envs"), runtime.get("env_indices")
    if (
        len(present) != 2
        or type(count) is not int
        or not 1 <= count <= 1000000
        or type(indices) is not list
        or len(indices) != saved_rows
        or any(type(index) is not int or not 0 <= index < count for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise InputError(
            "Environment selection requires source_num_envs in [1,1000000] and unique integer "
            "env_indices matching all saved rows; booleans are refused"
        )
    omitted = count - saved_rows
    if "omitted_env_count" in runtime and (
        type(runtime["omitted_env_count"]) is not int or runtime["omitted_env_count"] != omitted
    ):
        raise InputError("Captured omitted environment count disagrees with source count and saved rows")
    return {
        "source_num_envs": count,
        "env_indices": list(indices),
        "saved_rows": saved_rows,
        "omitted_envs": omitted,
        "scope": f"Explicitly saved {saved_rows} of {count} source environments; {omitted} omitted. "
        "Indices are environment positions, not temporal sample IDs or evidence of a physical same-read.",
    }


def _source_key(key):
    return key == "LICENSE.txt" or (
        type(key) is str
        and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,96}\.py", key) is not None
        and ".." not in key
    )


def _safe_root(path):
    if not isinstance(path, (str, Path)):
        raise InputError("Layout capture path must be a path string")
    root = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise InputError("Layout paths must not contain symlinks")
    return root.resolve()


def _read(path, limit):
    try:
        if path.is_symlink():
            raise InputError("Layout symlink artifacts are refused")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise InputError("Layout artifact is not a bounded regular file")
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise InputError("Layout artifact exceeds its byte limit")
        return raw
    except OSError as exc:
        raise InputError(f"Cannot read layout artifact: {exc}") from exc


def _tree(root):
    if not root.is_dir() or root.is_symlink():
        raise InputError("Layout capture must be a native directory")
    found = set()
    for count, path in enumerate(root.iterdir(), 1):
        if count > 4 or path.is_symlink():
            raise InputError("Layout capture has unexpected entries or symlinks")
        if path.name == "sources":
            if not path.is_dir():
                raise InputError("Layout sources must be a directory")
            for index, item in enumerate(path.iterdir(), 1):
                if (
                    index > MAX_SOURCE_FILES
                    or not _source_key(item.name)
                    or item.is_symlink()
                    or not item.is_file()
                ):
                    raise InputError("Layout sources contain unsupported entries")
                found.add("sources/" + item.name)
        elif path.name in {"capture.json", "arrays.npz", "READY.json"} and path.is_file():
            found.add(path.name)
        else:
            raise InputError("Layout capture contains an unlisted artifact")
    return found


def _validate(metadata, arrays, sources):
    _json(metadata)
    if (
        type(metadata) is not dict
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or metadata.get("kind") != "isaaclab_joint_wrench_layout"
    ):
        raise InputError("Unsupported layout metadata schema")
    for key in ("source", "runtime", "processing", "provenance"):
        if key in metadata and type(metadata[key]) is not dict:
            raise InputError(f"Layout metadata {key} must be a mapping")
    runtime, source = metadata.get("runtime", {}), metadata.get("source", {})
    for key in ("repository", "revision", "rule"):
        if key in source and not _text(source[key]):
            raise InputError("Native source labels must be bounded nonempty strings")
    if "origin" in metadata.get("provenance", {}) and not _text(metadata["provenance"]["origin"]):
        raise InputError("Captured provenance origin must be a bounded string")
    if "component_names" in metadata or "component_names" in runtime:
        raise InputError("Hand-authored component_names are refused; bindings must be derived")
    files = source.get("files", {})
    if type(files) is not dict or set(files) != set(sources) or len(files) > MAX_SOURCE_FILES:
        raise InputError("Layout source files must cover exactly the bundled module bytes")
    for key, raw in sources.items():
        if (
            not _source_key(key)
            or type(raw) is not bytes
            or not 1 <= len(raw) <= MAX_SOURCE_BYTES
            or type(files[key]) is not dict
            or set(files[key]) != {"sha256"}
            or files[key]["sha256"] != _digest(raw)
        ):
            raise InputError("Layout source module bytes/hash are malformed or stale")
    identities = [source.get("term"), source.get("sensor_class"), source.get("sensor_config_class")]
    mro = source.get("sensor_mro", [])
    if type(mro) is not list or len(mro) > 16:
        raise InputError("Layout sensor MRO must be a bounded identity list")
    for identity in identities + mro:
        if identity is None:
            continue
        if type(identity) is not dict:
            raise InputError("Malformed native source identity")
        for label in ("module", "qualname", "source_file"):
            if label in identity and not _text(identity[label]):
                raise InputError("Native source identity labels must be bounded strings")
        if "sha256" in identity and (
            not re.fullmatch("[0-9a-f]{64}", str(identity["sha256"]))
            or identity.get("source_file") not in files
            or identity["sha256"] != files[identity["source_file"]]["sha256"]
        ):
            raise InputError("Native source identity hash is malformed or does not bind bundled bytes")
    required = {"source_force", "source_torque", "raw_term_output"}
    if (
        type(arrays) is not dict
        or not required <= set(arrays)
        or set(arrays) - required - {"processed_term_output"}
    ):
        raise InputError(
            "Layout arrays require exactly native force, torque, raw term and optional processed term"
        )
    total = byte_count = 0
    for array in arrays.values():
        if type(array) is not np.ndarray or array.dtype.kind != "f" or array.dtype.itemsize not in (2, 4, 8):
            raise InputError("Layout buffers must be real floating NumPy arrays without object/pickle data")
        total += array.size
        byte_count += array.nbytes
        if total > MAX_VALUES or byte_count > MAX_ARRAY_BYTES or not np.isfinite(array).all():
            raise InputError("Layout values exceed bounds or contain nonfinite data")
    force, torque, raw = (arrays[key] for key in ("source_force", "source_torque", "raw_term_output"))
    if (
        force.ndim != 3
        or force.shape != torque.shape
        or force.shape[2] != 3
        or not 1 <= force.shape[0] <= 32
        or not 1 <= force.shape[1] <= 256
        or raw.ndim != 2
        or raw.shape[0] != force.shape[0]
        or not 1 <= raw.shape[1] <= MAX_NAMES * 6
        or raw.shape[1] % 6
    ):
        raise InputError("Layout native buffer/term shapes are invalid")
    _environment_selection(runtime, force.shape[0])
    names = runtime.get("sensor_body_names")
    if names is not None:
        # A source can expose more bodies than the selected term, but remains bounded.
        if (
            type(names) is not list
            or len(names) != force.shape[1]
            or len(names) > 256
            or any(not _text(name) for name in names)
            or len(set(names)) != len(names)
        ):
            raise InputError("Native sensor names must uniquely match the source body dimension")
    ids = runtime.get("resolved_body_ids")
    if ids is not None and (
        type(ids) is not list
        or not 1 <= len(ids) <= MAX_NAMES
        or any(type(index) is not int or not 0 <= index < force.shape[1] for index in ids)
        or len(set(ids)) != len(ids)
        or len(ids) * 6 != raw.shape[1]
    ):
        raise InputError("Native resolved body IDs must be unique bounded integers matching the raw width")
    if "preserve_order" in runtime and type(runtime["preserve_order"]) is not bool:
        raise InputError("Captured preserve_order must be a boolean")
    _identity(runtime.get("sensor_identity"))
    processing = metadata.get("processing", {})
    scale = processing.get("scale")
    if scale is not None:
        scales = scale if type(scale) is list else [scale]
        if len(scales) not in (1, raw.shape[1]) or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in scales
        ):
            raise InputError("Captured scale must be a finite scalar or one scale per raw component")
    processed = arrays.get("processed_term_output")
    if processed is not None and processed.shape != raw.shape:
        raise InputError("Processed term shape must match the captured raw term")


def _load(path):
    root = _safe_root(path)
    tree = _tree(root)
    manifest_raw = _read(root / "capture.json", MAX_MANIFEST_BYTES)
    ready_raw = _read(root / "READY.json", 4096)
    manifest, ready = _parse(manifest_raw), _parse(ready_raw)
    if (
        set(ready) != {"schema_version", "manifest_sha256"}
        or type(ready["schema_version"]) is not int
        or ready["schema_version"] != 1
        or ready["manifest_sha256"] != _digest(manifest_raw)
    ):
        raise InputError("Layout commit marker does not match the manifest")
    if (
        set(manifest) != {"schema_version", "kind", "metadata", "files"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["kind"] != "resolved_layout_capture"
        or type(manifest["files"]) is not list
        or not 1 <= len(manifest["files"]) <= MAX_SOURCE_FILES + 1
    ):
        raise InputError("Unsupported layout capture manifest")
    payloads, snapshots = {}, {"capture.json": _digest(manifest_raw), "READY.json": _digest(ready_raw)}
    total = len(manifest_raw) + len(ready_raw)
    for entry in manifest["files"]:
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise InputError("Malformed layout payload entry")
        relative = entry["path"]
        if (
            type(relative) is not str
            or relative in payloads
            or not (
                relative == "arrays.npz" or (relative.startswith("sources/") and _source_key(relative[8:]))
            )
            or not re.fullmatch("[0-9a-f]{64}", str(entry["sha256"]))
        ):
            raise InputError("Invalid or duplicate layout payload path/hash")
        raw = _read(root / relative, MAX_ARRAY_BYTES if relative == "arrays.npz" else MAX_SOURCE_BYTES)
        total += len(raw)
        if total > MAX_CAPTURE_BYTES or _digest(raw) != entry["sha256"]:
            raise InputError("Layout payload is oversized or changed from its recorded hash")
        payloads[relative], snapshots[relative] = raw, _digest(raw)
    if set(payloads) | {"capture.json", "READY.json"} != tree or "arrays.npz" not in payloads:
        raise InputError("Layout manifest must cover exactly all captured payloads")
    _npz_preflight(payloads["arrays.npz"], MAX_ARRAY_BYTES)
    _, arrays = _npz(payloads["arrays.npz"], True)
    sources = {key[8:]: raw for key, raw in payloads.items() if key.startswith("sources/")}
    _validate(manifest["metadata"], arrays, sources)
    return root, manifest["metadata"], arrays, snapshots


def _recheck(root, snapshots):
    if _tree(root) != set(snapshots):
        raise InputError("Layout capture file set changed during verification")
    for key, digest in snapshots.items():
        limit = MAX_ARRAY_BYTES if key == "arrays.npz" else MAX_SOURCE_BYTES
        if _digest(_read(root / key, limit)) != digest:
            raise InputError("Layout capture changed during verification")


def _symbol(identity):
    return (
        tuple(identity.get(key) for key in ("module", "qualname", "sha256"))
        if type(identity) is dict
        else None
    )


def _difference(left, right, atol, rtol):
    mask = ~np.isclose(left, right, atol=atol, rtol=rtol)
    if not mask.any():
        return None
    coordinate = tuple(int(index) for index in np.unravel_index(int(np.argmax(mask)), mask.shape))
    return {
        "index": list(coordinate),
        "observed": float(left[coordinate]),
        "expected": float(right[coordinate]),
    }


def _report(
    root,
    metadata,
    arrays,
    snapshots,
    *,
    expected_names=None,
    expected_frame=None,
    expected_quantity=None,
    expected_identity=None,
    atol=1e-6,
    rtol=1e-5,
):
    source, runtime = metadata.get("source", {}), metadata.get("runtime", {})
    pinned = (
        source.get("revision") == PIN
        and source.get("repository") in {"https://github.com/isaac-sim/IsaacLab", "isaac-sim/IsaacLab"}
        and source.get("rule") == RULE
    )
    term_known = pinned and _symbol(source.get("term")) == TERM
    sensor_known = (
        pinned
        and _symbol(source.get("sensor_class")) in SENSORS
        and any(_symbol(item) == BASE for item in source.get("sensor_mro", []))
    )
    provenance = metadata.get("provenance", {})
    context_known = provenance.get("origin") in {"in_process_cached_read", "generated_native_rule_control"}
    names, ids = runtime.get("sensor_body_names"), runtime.get("resolved_body_ids")
    bodies = [names[index] for index in ids] if names is not None and ids is not None else None
    components = (
        [f"{name}.{suffix}" for name in bodies for suffix in SUFFIXES] if bodies is not None else None
    )
    semantic_known = bool(sensor_known and context_known)
    layout = {
        "body_names": bodies,
        "body_ids": ids,
        "component_names": components,
        "quantity": "incoming_joint_wrench" if semantic_known else None,
        "frame": "incoming_joint_frame" if semantic_known else None,
        "reference_point": "child_joint_anchor" if semantic_known else None,
        "units": {"force": "N", "torque": "N*m"} if semantic_known else None,
        "sensor_identity": copy.deepcopy(runtime.get("sensor_identity")),
        "source_pin_supported": bool(term_known and sensor_known),
        "provenance_origin": provenance.get("origin"),
        "environment_selection": _environment_selection(runtime, arrays["source_force"].shape[0]),
    }
    findings = []

    def finding(code, status, message, *, mismatch=None, array=None, pointer=None):
        evidence = []
        if array:
            evidence.append(
                {
                    "path": str(root / "arrays.npz"),
                    "sha256": snapshots["arrays.npz"],
                    "array": array,
                    **(mismatch or {}),
                }
            )
            if mismatch and array == "raw_term_output":
                row, channel = mismatch["index"]
                body_slot, axis = divmod(channel, 6)
                source_array = "source_force" if axis < 3 else "source_torque"
                evidence.append(
                    {
                        "path": str(root / "arrays.npz"),
                        "sha256": snapshots["arrays.npz"],
                        "array": source_array,
                        "index": [row, ids[body_slot], axis % 3],
                        "value": float(arrays[source_array][row, ids[body_slot], axis % 3]),
                    }
                )
        if pointer:
            evidence.append(
                {"path": str(root / "capture.json"), "sha256": snapshots["capture.json"], "pointer": pointer}
            )
        findings.append({"code": code, "status": status, "message": message, "evidence": evidence})

    raw = arrays["raw_term_output"]
    raw_binding = {
        "status": "unknown",
        "replay_equal": None,
        "source_shape": list(arrays["source_force"].shape),
        "raw_shape": list(raw.shape),
        "rule": RULE,
        "first_mismatch": None,
    }
    if term_known and bodies is not None and context_known:
        replay = np.concatenate(
            (arrays["source_force"][:, ids, :], arrays["source_torque"][:, ids, :]), axis=-1
        )
        replay = replay.reshape(raw.shape)
        mismatch = _difference(raw, replay, atol, rtol)
        raw_binding.update(
            status="error" if mismatch else "pass", replay_equal=mismatch is None, first_mismatch=mismatch
        )
        finding(
            "layout.raw_replay",
            raw_binding["status"],
            "Raw term compared with the pinned saved-buffer gather/flatten rule.",
            mismatch=mismatch,
            array="raw_term_output",
        )
    else:
        finding(
            "layout.raw_replay",
            "unknown",
            "Missing supported term source, native names/IDs or captured-read context.",
        )
    if not semantic_known or not term_known:
        finding(
            "layout.native_semantics",
            "unknown",
            "Quantity/frame require matched native term, sensor/base sources and context.",
        )
    processed = {
        "status": "unknown",
        "reason": "Processed term/scale or supported manager stages are unavailable.",
    }
    processing = metadata.get("processing", {})
    no_stages = (
        {"modifiers", "clip", "history", "noise", "delay", "scale"} <= set(processing)
        and processing.get("modifiers") in (None, [])
        and processing.get("clip") is None
        and processing.get("noise") is None
        and (
            processing.get("delay") is None
            or type(processing.get("delay")) is int
            and processing["delay"] == 0
        )
        and (
            processing.get("history") is None
            or type(processing.get("history")) is int
            and processing["history"] == 0
        )
    )
    if (
        "processed_term_output" in arrays
        and processing.get("scale") is not None
        and no_stages
        and raw_binding["status"] != "unknown"
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            replay = raw * np.asarray(processing["scale"])
        if not np.isfinite(replay).all():
            processed["reason"] = "Captured scale arithmetic is nonfinite; processed binding is unsupported."
        else:
            mismatch = _difference(arrays["processed_term_output"], replay, atol, rtol)
            processed = {
                "status": "error" if mismatch else "pass",
                "reason": "Saved raw term times captured scale only.",
                "scale": processing["scale"],
                "first_mismatch": mismatch,
            }
            finding(
                "layout.processed_replay",
                processed["status"],
                processed["reason"],
                mismatch=mismatch,
                array="processed_term_output",
            )
    processed["native_manager_semantics"] = "unknown"
    processed["arithmetic_scope"] = (
        "Saved raw term times captured scale only; no native manager implementation or policy-input binding is verified."
    )
    contract = {
        "expected_names": expected_names,
        "expected_frame": expected_frame,
        "expected_quantity": expected_quantity,
        "expected_identity": expected_identity,
        "status": "pass",
        "first_mismatch": None,
    }
    for key, expected, observed in (
        ("order", expected_names, bodies),
        ("frame", expected_frame, layout["frame"]),
        ("quantity", expected_quantity, layout["quantity"]),
        ("identity", expected_identity, layout["sensor_identity"]),
    ):
        if expected is None:
            continue
        verified_binding = raw_binding["status"] == "pass" and semantic_known and term_known
        verdict = (
            "unknown"
            if not verified_binding or observed is None
            else "pass"
            if expected == observed
            else "error"
        )
        if verdict == "error" or verdict == "unknown" and contract["status"] != "error":
            contract["status"] = verdict
        finding(
            "layout.expected_" + key,
            verdict,
            f"Explicit raw-term {key} acceptance contract.",
            pointer="/metadata/runtime" if key in ("order", "identity") else "/metadata/source",
        )
        findings[-1].update(observed=observed, expected=expected)
        if key == "order" and verdict == "error":
            slot = next(
                (
                    index
                    for index in range(min(len(expected), len(observed)))
                    if expected[index] != observed[index]
                ),
                min(len(expected), len(observed)),
            )
            contract["first_mismatch"] = {
                "slot": slot,
                "expected_body": expected[slot] if slot < len(expected) else None,
                "observed_body": observed[slot] if slot < len(observed) else None,
                "native_resolved_body_id": ids[slot] if slot < len(ids) else None,
            }
    statuses = [raw_binding["status"], contract["status"]]
    if not semantic_known or not term_known:
        statuses.append("unknown")
    if processed["status"] == "error":
        statuses.append("error")
    status = "error" if "error" in statuses else "unknown" if "unknown" in statuses else "pass"
    report = {
        "schema_version": 1,
        "kind": "resolved_layout_audit",
        "status": status,
        "raw_binding": raw_binding,
        "processed_binding": processed,
        "layout": layout,
        "acceptance_contract": contract,
        "findings": findings,
        "counts": {
            key: sum(item["status"] == key for item in findings) for key in ("pass", "error", "unknown")
        },
        "artifacts": [{"path": str(root / key), "sha256": digest} for key, digest in snapshots.items()],
        "unverified": {
            "physical_sample_identity": "unknown",
            "policy_concatenation": "unknown",
            "authenticity": "not_attested",
            "same_physical_read": "unknown",
        },
        "scope": SCOPE + " " + layout["environment_selection"]["scope"],
        "tolerances": {"atol": atol, "rtol": rtol},
    }
    if len(json.dumps(report, allow_nan=False).encode()) > 192 * 1024:
        raise InputError("Layout report exceeds 192 KiB; reduce the captured layout")
    return report


def audit_layout_capture(
    path,
    *,
    expected_names=None,
    expected_frame=None,
    expected_quantity=None,
    expected_identity=None,
    atol=1e-6,
    rtol=1e-5,
):
    """Verify saved bytes and raw-term acceptance; never execute bundled sources."""
    _tolerances(atol, rtol)
    if expected_names is not None:
        expected_names = _names(expected_names, "expected_names")
    for label, value in (("expected_frame", expected_frame), ("expected_quantity", expected_quantity)):
        if value is not None and not _text(value):
            raise InputError(label + " must be a bounded nonempty string")
    expected_identity = _identity(expected_identity)
    root, metadata, arrays, snapshots = _load(path)
    report = _report(
        root,
        metadata,
        arrays,
        snapshots,
        expected_names=expected_names,
        expected_frame=expected_frame,
        expected_quantity=expected_quantity,
        expected_identity=expected_identity,
        atol=atol,
        rtol=rtol,
    )
    _recheck(root, snapshots)
    return report


def write_layout_capture(output, metadata, arrays, *, sources):
    """Publish a new bounded snapshot from already captured CPU arrays and module bytes."""
    destination = _safe_root(output)
    if destination.exists() or destination.is_symlink():
        raise InputError("Layout capture destination already exists")
    try:
        _json(metadata)
        metadata = copy.deepcopy(metadata)
        if type(arrays) is not dict or type(sources) is not dict:
            raise InputError("Layout arrays and sources must be mappings")
        _validate(metadata, arrays, sources)
        original_arrays = arrays
        original_hashes = {key: _digest(value.tobytes()) for key, value in arrays.items()}
        arrays = {key: value.copy() for key, value in arrays.items()}
        if any(_digest(value.tobytes()) != original_hashes[key] for key, value in arrays.items()):
            raise InputError("Layout input arrays changed during snapshot")
        sources = dict(sources)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        if any(_digest(value.tobytes()) != original_hashes[key] for key, value in original_arrays.items()):
            raise InputError("Layout input arrays changed during snapshot")
        array_bytes = buffer.getvalue()
        if (
            len(array_bytes) > MAX_ARRAY_BYTES
            or sum(map(len, sources.values())) + len(array_bytes) > MAX_CAPTURE_BYTES - MAX_MANIFEST_BYTES
        ):
            raise InputError("Layout capture exceeds bounded array/total byte budgets")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".layout-capture-", dir=destination.parent))
    except (TypeError, ValueError, RecursionError, OSError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"Cannot prepare layout capture: {exc}") from exc
    try:
        (staging / "arrays.npz").write_bytes(array_bytes)
        if sources:
            (staging / "sources").mkdir()
        for key, raw in sources.items():
            (staging / "sources" / key).write_bytes(raw)
        files = [{"path": "arrays.npz", "sha256": _digest(array_bytes)}] + [
            {"path": "sources/" + key, "sha256": _digest(raw)} for key, raw in sorted(sources.items())
        ]
        manifest = {
            "schema_version": 1,
            "kind": "resolved_layout_capture",
            "metadata": metadata,
            "files": files,
        }
        manifest_raw = _json(manifest)
        (staging / "capture.json").write_bytes(manifest_raw)
        (staging / "READY.json").write_bytes(
            _json({"schema_version": 1, "manifest_sha256": _digest(manifest_raw)})
        )
        audit_layout_capture(staging)
        publish_capture(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit_layout_capture(destination)


def compare_layout_captures(reference, candidate, *, atol=1e-6, rtol=1e-5):
    """Separate ordered layout changes from named-value differences and unknown identity."""
    _tolerances(atol, rtol)
    loaded = [_load(path) for path in (reference, candidate)]
    reports = [_report(*items, atol=atol, rtol=rtol) for items in loaded]
    left, right = (report["layout"] for report in reports)
    raw_left, raw_right = (items[2]["raw_term_output"] for items in loaded)
    raw_equal = bool(
        raw_left.shape == raw_right.shape and _difference(raw_left, raw_right, atol, rtol) is None
    )
    supported = all(report["status"] == "pass" for report in reports)
    identity_match = (
        left["sensor_identity"] is not None and left["sensor_identity"] == right["sensor_identity"]
    )
    semantics_match = all(
        left[key] is not None and left[key] == right[key] for key in ("quantity", "frame", "reference_point")
    )
    a_env, b_env = left["environment_selection"], right["environment_selection"]
    a_explicit, b_explicit = a_env["env_indices"] is not None, b_env["env_indices"] is not None
    environment_match = (
        (
            a_env["env_indices"] == b_env["env_indices"]
            and a_env["source_num_envs"] == b_env["source_num_envs"]
        )
        if a_explicit and b_explicit
        else None
    )
    row_compatible = not a_explicit and not b_explicit or environment_match is True
    ordered_equal = (
        left["component_names"] == right["component_names"]
        if all(layout["component_names"] is not None for layout in (left, right))
        else None
    )
    layout_difference = None
    if ordered_equal is False:
        a_names, b_names = left["body_names"], right["body_names"]
        slot = next(
            (index for index in range(min(len(a_names), len(b_names))) if a_names[index] != b_names[index]),
            min(len(a_names), len(b_names)),
        )
        layout_difference = {
            "slot": slot,
            "reference_body": a_names[slot] if slot < len(a_names) else None,
            "candidate_body": b_names[slot] if slot < len(b_names) else None,
            "reference_resolved_body_id": left["body_ids"][slot] if slot < len(a_names) else None,
            "candidate_resolved_body_id": right["body_ids"][slot] if slot < len(b_names) else None,
            "evidence": [
                {
                    "path": str(items[0] / "capture.json"),
                    "sha256": items[3]["capture.json"],
                    "pointer": f"/metadata/runtime/resolved_body_ids/{slot}",
                }
                for items in loaded
            ],
        }
    named_equal, first = None, None
    comparable = bool(
        supported
        and identity_match
        and semantics_match
        and row_compatible
        and set(left["component_names"] or []) == set(right["component_names"] or [])
        and raw_left.shape == raw_right.shape
    )
    if comparable:
        indices = [right["component_names"].index(key) for key in left["component_names"]]
        first = _difference(raw_left, raw_right[:, indices], atol, rtol)
        named_equal = first is None
        if first:
            first["component"] = left["component_names"][first["index"][1]]
            row, channel = first["index"]
            candidate_channel = indices[channel]
            first.update(
                reference_index=[row, channel],
                candidate_index=[row, candidate_channel],
                reference_value=float(raw_left[row, channel]),
                candidate_value=float(raw_right[row, candidate_channel]),
            )
            first["evidence"] = [
                {
                    "path": str(items[0] / "arrays.npz"),
                    "sha256": items[3]["arrays.npz"],
                    "array": "raw_term_output",
                    "index": coordinate,
                    "value": value,
                }
                for items, coordinate, value in (
                    (loaded[0], first["reference_index"], first["reference_value"]),
                    (loaded[1], first["candidate_index"], first["candidate_value"]),
                )
            ]
    if not row_compatible:
        status, classification = (
            "unknown",
            "environment_selection_mismatch"
            if a_explicit and b_explicit
            else "environment_selection_unknown",
        )
    elif not comparable:
        status, classification = "unknown", "insufficient_shared_identity_semantics_or_binding"
    elif ordered_equal is False:
        status = "different"
        classification = (
            "permutation_with_equal_named_values" if named_equal else "layout_and_named_values_changed"
        )
    else:
        status, classification = (
            ("pass", "same_saved_layout_and_values") if named_equal else ("different", "named_values_changed")
        )
    for root, _, _, snapshots in loaded:
        _recheck(root, snapshots)
    report = {
        "schema_version": 1,
        "kind": "resolved_layout_comparison",
        "status": status,
        "ordered_layout_equal": ordered_equal,
        "raw_values_equal": raw_equal,
        "named_values_equal": named_equal,
        "observed_named_value_difference": named_equal is False,
        "classification": classification,
        "first_named_difference": first,
        "first_layout_difference": layout_difference,
        "sensor_identity_match": bool(identity_match),
        "environment_selection_match": environment_match,
        "reference": {"status": reports[0]["status"], "layout": left},
        "candidate": {"status": reports[1]["status"], "layout": right},
        "artifacts": reports[0]["artifacts"] + reports[1]["artifacts"],
        "scope": "Saved namespace/layout and aligned named values only. A permutation is not a numeric fault. "
        "Different or absent namespace identity makes named comparison unknown. "
        + SCOPE
        + (
            " Matching explicit environment selections are compared; these are not temporal sample IDs."
            if environment_match is True
            else " Legacy rows cover saved positions only; shared environment association is unknown."
            if row_compatible
            else " Environment selection differs or is missing; named-value comparison is unknown."
        ),
        "unverified": reports[0]["unverified"],
        "tolerances": {"atol": atol, "rtol": rtol},
    }
    if len(json.dumps(report, allow_nan=False).encode()) > 192 * 1024:
        raise InputError("Layout comparison report exceeds 192 KiB; reduce the captured layout")
    return report
