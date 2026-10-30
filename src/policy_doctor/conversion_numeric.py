"""Internal, projection-only Arena43 numeric association proposal; no artifact I/O.

This module cannot establish HDF5/Parquet membership or native consumption. Only
the exact retained converter/config closure is modeled. No code is imported from
that closure and no config constructor, converter or native pipeline is executed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .artifacts import InputError

MAX_SOURCE_BYTES = 512 * 1024
MAX_ROWS = 4096
MAX_EPISODES = 256
MAX_VALUES_BYTES = 64 * 1024**2
MAX_FINDINGS = 64
ROOT = "isaaclab_arena_gr00t/"
SOURCE_HASHES = {
    ROOT
    + "lerobot/convert_hdf5_to_lerobot.py": "0c837c3176a4ab0ed32fa71081daabc04e1c777963d65b828e3599a1b80ac8dc",
    ROOT
    + "lerobot/config/dataset_config.py": "400faef86090824ac7f334b199916fa7544cfd8d1fc2575245a73a78fe834ec1",
    ROOT + "utils/joints_conversion.py": "a9e1f3638be222221484129a2ee8c2a776275c685bd8aa07269c13618dd66386",
    ROOT + "utils/robot_joints.py": "510bf5507ed45e021e01a3fe8f26b5fd5ab6a31e035c1d5a16a72c72a7673206",
    ROOT + "utils/io_utils.py": "1e72d6b8ad862388180e253203d359d4abd6bcc75000dc37fb00d4404e6716a5",
    ROOT
    + "lerobot/config/g1_static_apple_config.yaml": "3dc9168a54d30f5d86c259b8620af075fd648d2236d8bb6c3646c72f51782c33",
    ROOT
    + "embodiments/g1/43dof_joint_space.yaml": "ff7db8a6e4da8cb6ad40329bf0f5175de00d60f804e39c7183dc7d3c2e7d24a7",
    ROOT
    + "embodiments/g1/gr00t_43dof_joint_space.yaml": "28226939039bd29fce7bf93d7f3fccce66bd3aa74a4c49e89d8229cd946357ae",
    ROOT + "embodiments/g1/modality.json": "36dcaef30bb34a4c0337775932d83ea7b79b36f19e3a715239bab6852a09d9cc",
}
# Derived from the exact YAML joint dictionaries and modality insertion order.
# Source from_numpy(...).to(cpu) -> numpy -> stack/concatenate has no float cast.
PERMUTATION = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    29,
    35,
    30,
    36,
    31,
    37,
    41,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    32,
    38,
    33,
    39,
    34,
    40,
    42,
)


@dataclass(frozen=True)
class Selection:
    source_key: str
    native_episode_index: int
    source_frames: tuple[int, ...]


@dataclass(frozen=True)
class NumericCriterion:
    selections: tuple[Selection, ...]
    closure: tuple[tuple[str, str], ...]
    digest: str


@dataclass
class EpisodeProjection:
    source_key: str
    native_episode_index: int
    source_state: np.ndarray | None
    source_action: np.ndarray | None
    exported_state: np.ndarray | None
    exported_action: np.ndarray | None


def _payload(selections, closure):
    return {
        "schema": "arena43_projection_numeric_v1",
        "closure": closure,
        "selection": [(s.source_key, s.native_episode_index, s.source_frames) for s in selections],
        "source_paths": ["data/{source_key}/obs/robot_joint_pos", "data/{source_key}/processed_actions"],
        "export_columns": ["observation.state", "action"],
        "permutation": PERMUTATION,
        "terminal_drop": 1,
        "comparison": "exact finite values and preserved native float32/float64 dtype",
        "atol": 0,
        "rtol": 0,
        "membership_assurance": "projection_only",
    }


def _digest(selections, closure):
    return hashlib.sha256(json.dumps(_payload(selections, closure), sort_keys=True).encode()).hexdigest()


def _selections(values):
    if type(values) is not tuple or not 1 <= len(values) <= MAX_EPISODES:
        raise InputError("Require a bounded immutable selection tuple; actual selection is not inferred")
    keys, ids = set(), set()
    for item in values:
        if (
            type(item) is not Selection
            or type(item.source_key) is not str
            or not item.source_key
            or item.source_key in {".", ".."}
            or len(item.source_key) > 256
            or "/" in item.source_key
            or any(ord(c) < 32 for c in item.source_key)
            or type(item.native_episode_index) is not int
            or not 0 <= item.native_episode_index <= 1_000_000
        ):
            raise InputError("Selection requires bounded original group keys and strict native episode IDs")
        if item.source_key in keys or item.native_episode_index in ids:
            raise InputError("Duplicate or ambiguous episode correspondences are unsupported")
        if (
            type(item.source_frames) is not tuple
            or not 1 <= len(item.source_frames) <= MAX_ROWS
            or any(type(n) is not int or not 0 <= n < MAX_ROWS for n in item.source_frames)
            or tuple(sorted(set(item.source_frames))) != item.source_frames
        ):
            raise InputError("Selection frames must be unique, increasing strict integers within the bound")
        keys.add(item.source_key)
        ids.add(item.native_episode_index)
    return values


def compile_criterion(source_files, selections):
    """Fingerprint bytes supplied by a future authorized reader, without opening them.

    Only the original example configuration is currently modeled. A path-edited
    actual converter YAML needs a separately reviewed semantic binding before use.
    """
    selections = _selections(selections)
    if type(source_files) is not dict or set(source_files) != set(SOURCE_HASHES):
        raise InputError("Require the exact complete converter/helper/joint/config/modality closure")
    if any(type(data) is not bytes for data in source_files.values()):
        raise InputError("Source closure must contain bounded bytes, never paths or caller objects")
    if sum(len(data) for data in source_files.values()) > MAX_SOURCE_BYTES:
        raise InputError("Converter source-data closure exceeds its explicit byte bound")
    closure = tuple(sorted((name, hashlib.sha256(data).hexdigest()) for name, data in source_files.items()))
    if dict(closure) != SOURCE_HASHES:
        raise InputError("Unsupported or drifted converter/config semantics; no fallback interpretation")
    return NumericCriterion(selections, closure, _digest(selections, closure))


def evaluate_numeric(criterion, episodes):
    """Compare provided observations; never claim original-file membership from them."""
    if (
        type(criterion) is not NumericCriterion
        or dict(criterion.closure) != SOURCE_HASHES
        or criterion.digest != _digest(_selections(criterion.selections), criterion.closure)
    ):
        raise InputError("Require the original unmodified numeric criterion")
    if type(episodes) not in (list, tuple) or len(episodes) > MAX_EPISODES:
        raise InputError("Require a bounded episode projection list")
    if any(type(e) is not EpisodeProjection for e in episodes):
        raise InputError("Episode inputs must be explicit internal numeric projections")
    allocated = 0
    for e in episodes:
        if type(e.source_key) is not str or type(e.native_episode_index) is not int:
            raise InputError("Projection identities require original text keys and strict integer IDs")
        for value in (e.source_state, e.source_action, e.exported_state, e.exported_action):
            if value is not None:
                if type(value) is not np.ndarray:
                    raise InputError("Array-like caller conversion is not supported")
                if (
                    value.ndim != 2
                    or value.shape[1] != 43
                    or not 1 <= value.shape[0] <= MAX_ROWS
                    or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
                    or not value.dtype.isnative
                ):
                    raise InputError("Require bounded native float32/float64 matrices with43 columns")
                allocated += value.nbytes
    if allocated > MAX_VALUES_BYTES:
        raise InputError("Numeric projections exceed the explicit aggregate value-byte bound")
    findings, counts = [], {"error": 0, "unknown": 0}
    expected_rows = sum(len(s.source_frames) for s in criterion.selections)
    compared_rows, represented_episodes = 0, 0

    def finding(status, code, selection, **detail):
        counts[status] += 1
        if len(findings) < MAX_FINDINGS:
            findings.append(
                {
                    "status": status,
                    "code": code,
                    "source_key": selection.source_key,
                    "native_episode_index": selection.native_episode_index,
                    **detail,
                }
            )

    selected_pairs = {(s.source_key, s.native_episode_index) for s in criterion.selections}
    if any((e.source_key, e.native_episode_index) not in selected_pairs for e in episodes):
        raise InputError("Unexpected episode projections cannot silently extend a frozen selection")
    for selection in criterion.selections:
        matches = [
            e
            for e in episodes
            if (e.source_key, e.native_episode_index)
            == (selection.source_key, selection.native_episode_index)
        ]
        if len(matches) != 1:
            finding(
                "unknown",
                "conversion.correspondence_unavailable",
                selection,
                reason="Missing or duplicate supplied source/export correspondence",
            )
            continue
        e = matches[0]
        if any(x is None for x in (e.source_state, e.source_action, e.exported_state, e.exported_action)):
            finding("unknown", "conversion.projection_missing", selection)
            continue
        source_finite = np.isfinite(e.source_state).all() and np.isfinite(e.source_action).all()
        export_finite = np.isfinite(e.exported_state).all() and np.isfinite(e.exported_action).all()
        if not source_finite:
            finding("unknown", "conversion.source_values_invalid", selection)
        if not export_finite:
            finding("error", "conversion.export_values_nonfinite", selection)
        if not source_finite or not export_finite:
            continue
        source_length = len(e.source_action)
        if len(e.source_state) != source_length or source_length < 2:
            finding("unknown", "conversion.source_lengths_unsupported", selection)
            continue
        expected_length = source_length - 1
        for role, source, exported in [
            ("state", e.source_state, e.exported_state),
            ("action", e.source_action, e.exported_action),
        ]:
            if len(exported) != expected_length:
                finding(
                    "error",
                    "conversion.terminal_trim_or_coverage",
                    selection,
                    role=role,
                    expected_export_rows=expected_length,
                    observed_export_rows=len(exported),
                )
            if exported.dtype != source.dtype:
                finding(
                    "error",
                    "conversion.dtype_changed",
                    selection,
                    role=role,
                    expected_dtype=source.dtype.str,
                    observed_dtype=exported.dtype.str,
                )
        represented_episodes += 1
        for frame in selection.source_frames:
            if frame >= expected_length:
                finding("unknown", "conversion.selection_outside_emitted_source", selection, frame=frame)
                continue
            if frame >= min(len(e.exported_state), len(e.exported_action)):
                finding("unknown", "conversion.selected_row_unavailable", selection, frame=frame)
                continue
            compared_rows += 1
            for role, source, exported in [
                ("state", e.source_state, e.exported_state),
                ("action", e.source_action, e.exported_action),
            ]:
                expected = source[frame, PERMUTATION]
                different = np.flatnonzero(expected != exported[frame])
                if len(different):
                    column = int(different[0])
                    finding(
                        "error",
                        "conversion.numeric_joint_association",
                        selection,
                        role=role,
                        source_frame=frame,
                        export_frame=frame,
                        export_column=column,
                        source_column=PERMUTATION[column],
                        expected=float(expected[column]),
                        observed=float(exported[frame, column]),
                        differing_columns=len(different),
                    )
    numeric_status = "error" if counts["error"] else "unknown" if counts["unknown"] else "pass"
    return {
        "schema_version": 1,
        "kind": "arena43_numeric_projection_comparison",
        "status": "error" if counts["error"] else "unknown",
        "numeric_association_status": numeric_status,
        "criterion_sha256": criterion.digest,
        "assurance": "projection_only",
        "source_membership": {
            "status": "unknown",
            "reason": "No original HDF5 or exported Parquet reader was run",
        },
        "native_consumption": {"status": "unknown", "reason": "No native loader/profile was executed"},
        "retention": {
            "complete_retained_readset": False,
            "original_membership_requires": "Actual original input revalidation or a dedicated budgeted original copy",
            "projection_replay_proves": "Only the provided projected values, not original-file membership",
        },
        "coverage": {
            "expected_selected_rows": expected_rows,
            "compared_selected_rows": compared_rows,
            "expected_selected_episodes": len(criterion.selections),
            "represented_selected_episodes": represented_episodes,
            "complete_selected_numeric_projection": numeric_status == "pass"
            and compared_rows == expected_rows,
            "projection_value_bytes": allocated,
            "whole_dataset_verified": False,
        },
        "counts": counts,
        "findings": findings,
        "findings_total": sum(counts.values()),
        "findings_truncated": sum(counts.values()) > len(findings),
        "unassessed": [
            "original/source identity and native membership",
            "actual converter execution/history",
            "teleop/eef columns",
            "camera/time associations",
            "relative/normalized actions",
            "native sample/horizon selection",
            "physical units and policy behavior",
        ],
        "scope": "Exact selected provided-array values under the fixed pinned remap; no source-fidelity claim.",
    }
