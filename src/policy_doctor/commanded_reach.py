"""Recorded command against recorded observation, joint by joint: where do they disagree while the arm is still?

In a LeRobot recording, action and observation.state hold the same `<joint>.pos` names. On frames where a joint's
command and reading both hold still, a joint that tracks its command shows the two equal. This module reports the
residual where they do not: a joint whose reading sits at one end of its travel while the recorded command goes on
past it, or a steady offset across its travel.

What the files can and cannot show: LeRobot records the teleoperator's action before the robot's own processing and
`max_relative_target` clipping (LeRobot 146d847 src/lerobot/scripts/lerobot_record.py:325-327, 362-367), so whether a
motor ever received a recorded command is not observed. A residual is triage evidence, not a calibration diagnosis;
every finding names possible causes rather than asserting one. Training on such data teaches commands the recorded
robot was not seen to reach (lerobot#3758 reports a ~17 degree offset with "training metrics fine; deployment broken").
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

from .artifacts import InputError, contained, dataset_root, read_json

LEROBOT = "LeRobot 146d847"
WINDOW = 5              # consecutive frames in which a joint's command and reading both hold still
STILL = 0.005           # per-frame change under 0.5% of the joint's range (or ABS_STILL) counts as still
ABS_STILL = 0.15          # about 1.5 Feetech steps (0.088 degree each)
LIMIT_BAND = 0.01       # readings within 1% of range (or ABS_BAND) of the joint's robust extreme are "at that end"
ABS_BAND = 1.0
FLAG = 0.05             # a residual over 5% of the joint's range is reported
AT_END_SECONDS, STILL_SECONDS = 6.7, 16.7   # minimum still time at one end, and away from the ends
# Whole-name tokens of joints that grasping blocks by design (grippers; hand joints such as Hope Jr index/middle/ring).
NOT_JUDGED = {"gripper", "finger", "thumb", "hand", "pinky", "palm", "index", "middle", "ring"}
SHARE = 0.25            # a residual on at least a quarter of still frames is reported even mid-range
COLUMNS = ("action", "observation.state", "episode_index")
MAX_FOLDERS = 10_000    # distinct data directories walked


def _names(feature: dict) -> list | None:
    names = feature.get("names")
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    return names if isinstance(names, list) and all(isinstance(n, str) for n in names) else None


def _data_files(root: Path, max_files: int) -> list[Path]:
    from .fs_guard import DatasetLayoutError, refuse_dataset_link

    metadata = (root / 'meta').resolve()

    def refuse_numeric_alias(path, target):
        if target.is_relative_to(root):
            kind = 'link' if path.is_symlink() else 'directory'
            where = 'inside the reserved metadata tree' if target.is_relative_to(metadata) else 'outside the data subtree'
            raise DatasetLayoutError(
                f'Numeric data {kind} {path.relative_to(root)} points to {target.relative_to(root)}, {where}. '
                'Restore the original regular numeric file or directory under data/, or remove an unused '
                'alias from a separate copy, then rerun. Metadata Parquet is not numeric data.', 'data')
        refuse_dataset_link(root, path)

    named_data = root / 'data'
    if named_data.is_symlink():
        try:
            target = named_data.resolve(strict=True)
        except (OSError, RuntimeError):
            refuse_dataset_link(root, named_data)
        if not target.is_relative_to(root) or root.is_relative_to(target):
            refuse_dataset_link(root, named_data)
        # The data fence must never be reset to the metadata namespace, including
        # when meta/ itself is an internal alias. No Parquet body is opened here.
        if target.is_relative_to(metadata):
            refuse_numeric_alias(named_data, target)
    data = contained(root, "data")
    if data.is_relative_to(metadata):
        refuse_numeric_alias(named_data, data)
    files, seen = [], set()
    if data.is_dir():
        for folder, folders, names in os.walk(data, followlinks=True):
            if Path(folder).resolve().is_relative_to(metadata):
                folders[:] = []
                continue  # canonical metadata is never a numeric directory, even as an ordinary descendant
            # A directory link back into the tree (data/l0 -> .) is entered once: without this a few such links
            # make the followed walk grow without bound.
            status = os.stat(folder)
            if (status.st_dev, status.st_ino) in seen:
                folders[:] = []
                continue
            seen.add((status.st_dev, status.st_ino))
            if len(seen) > MAX_FOLDERS:
                raise InputError(f"More than {MAX_FOLDERS} data directories; not a bounded dataset layout")
            for name in folders[:]:
                path = Path(folder) / name
                try:
                    target = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    if path.is_symlink():
                        refuse_dataset_link(root, path)
                    raise
                if target.is_relative_to(metadata):
                    if path.is_symlink():
                        refuse_numeric_alias(path, target)
                    folders.remove(name)
                    continue
                if not path.is_symlink():
                    continue
                if not target.is_relative_to(data):
                    if target.is_relative_to(root) and data.is_relative_to(target):
                        folders.remove(name)  # ancestor alias contributes no numeric data directory
                        continue
                    refuse_numeric_alias(path, target)
            for name in names:
                if name.endswith(".parquet"):
                    path = Path(folder) / name
                    try:
                        target = path.resolve(strict=True)
                    except (OSError, RuntimeError):
                        if path.is_symlink():
                            refuse_dataset_link(root, path)
                        raise
                    if target.is_relative_to(metadata):
                        if path.is_symlink():
                            refuse_numeric_alias(path, target)
                        continue
                    if path.is_symlink():
                        if not target.is_relative_to(data):
                            refuse_numeric_alias(path, target)
                    real = contained(root, str(path.relative_to(root)))
                    if real in files:  # a symlinked folder can reach the same file twice
                        continue
                    files.append(real)
                    if len(files) > max_files:
                        raise InputError(f"More than {max_files} data files; raise max_files")
    if not files:
        raise InputError(f"No Parquet data files under {data}")
    return sorted(files)


def _read(root: Path, max_rows: int, max_files: int):
    import pyarrow.parquet as pq

    files, rows, parts, receipts = _data_files(root, max_files), 0, [], []
    for path in files:
        handle = pq.ParquetFile(path)
        if handle.metadata.num_rows == 0:
            continue
        missing = [c for c in COLUMNS if c not in handle.schema_arrow.names]
        if missing:
            raise InputError(f"{path.relative_to(root)} has no column(s) {missing}")
        rows += handle.metadata.num_rows
        if rows > max_rows:
            raise InputError(f"More than {max_rows} rows; raise max_rows for a complete check")
        columns = list(COLUMNS) + (["frame_index"] if "frame_index" in handle.schema_arrow.names else [])
        table = handle.read(columns=columns)
        part = {c: table.column(c).to_pylist() for c in columns}
        part["frame_index"] = part.get("frame_index") or list(range(table.num_rows))
        parts.append(part)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        receipts.append({"path": str(path.relative_to(root)), "sha256": digest.hexdigest(), "rows": table.num_rows})
    if not parts:
        raise InputError("The data files hold no rows")

    def stack(key, dtype):
        values = [v for p in parts for v in p[key]]
        try:
            return np.asarray([np.nan if v is None else v for v in values] if dtype is float and np.ndim(values[0]) == 0
                              else [[np.nan if x is None else x for x in (v or [])] for v in values], dtype=dtype)
        except (ValueError, TypeError) as exc:
            raise InputError(f"Column {key} is not a fixed-width numeric vector: {exc}") from exc
    return (stack("action", float), stack("observation.state", float),
            np.asarray([v for p in parts for v in p["episode_index"]]),
            np.asarray([v for p in parts for v in p["frame_index"]]), receipts)


def _segments(episode: np.ndarray, frame: np.ndarray) -> list[np.ndarray]:
    """Rows of one recording run: same episode, frame_index rising by one (orphan files reuse episode indices)."""
    out = []
    for e in np.unique(episode):
        idx = np.flatnonzero(episode == e)
        breaks = np.flatnonzero(np.diff(frame[idx]) != 1) + 1
        out.extend(part for part in np.split(idx, breaks) if len(part))
    return out


def _still(series: list[np.ndarray], segments: list[np.ndarray], limit: float) -> np.ndarray:
    still = np.zeros(len(series[0]), dtype=bool)
    for idx in segments:
        if len(idx) <= WINDOW:
            continue
        moving = np.zeros(len(idx) - 1, dtype=bool)
        for values in series:
            step = np.abs(np.diff(values[idx]))
            moving |= ~(step <= limit)  # NaN counts as moving
        calm = np.convolve((~moving).astype(int), np.ones(WINDOW, dtype=int), mode="valid") == WINDOW
        for start in np.flatnonzero(calm):
            still[idx[start + 1:start + 1 + WINDOW]] = True
    return still


def check_commanded_reach(dataset: str | Path, *, max_rows: int = 2_000_000, max_files: int = 1024) -> dict:
    """Report per-joint residuals between recorded action and observation.state on still frames."""
    root = dataset_root(dataset)
    info_path = contained(root, "meta/info.json")
    info = read_json(info_path)
    base = {"path": "meta/info.json", "sha256": hashlib.sha256(info_path.read_bytes()).hexdigest()}
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    labels = _names(features.get("action", {}))
    findings, joints = [], {}

    def add(code, status, message, pointer="/features", value=None, remedy="", extra=()):
        findings.append({"code": code, "status": status, "message": message, "remedy": remedy,
                         "evidence": [dict(base, pointer=pointer, value=value), *extra]})

    if not labels or labels != _names(features.get("observation.state", {})) or not all(n.endswith(".pos") for n in labels):
        add("reach.not_applicable", "unknown",
            "action and observation.state are not the same LeRobot motor positions (identical `<joint>.pos` names), "
            "so a recorded command cannot be compared with the recorded reading.",
            value={"action": labels, "observation.state": _names(features.get("observation.state", {}))})
        return _finish(root, findings, joints, None)
    action, state, episode, frame, receipts = _read(root, max_rows, max_files)
    if len(action) == 0:
        raise InputError("The data files hold no rows")
    if action.ndim != 2 or action.shape != state.shape or action.shape[1] != len(labels):
        raise InputError(f"action {action.shape} and observation.state {state.shape} do not match {len(labels)} names")
    fps = info.get("fps") if isinstance(info.get("fps"), (int, float)) and info.get("fps") > 0 else 30
    at_end_min, still_min = max(30, round(AT_END_SECONDS * fps)), max(60, round(STILL_SECONDS * fps))
    segments = _segments(episode, frame)
    coverage = {"rows": int(len(action)), "episodes": int(len(np.unique(episode))), "runs": len(segments),
                "fps": fps, "min_still_frames_at_an_end": at_end_min, "min_still_frames_away": still_min,
                "data_files": receipts}
    declared = info.get("total_frames")
    if isinstance(declared, int) and declared != len(action):
        add("reach.rows_disagree_with_metadata", "unknown",
            f"meta/info.json declares {declared} frames but the data files hold {len(action)} rows; rows outside the "
            "declared episodes are compared too. Run the episode checks (preflight) to see which rows are orphaned.",
            pointer="/total_frames", value=declared)
    data_evidence = tuple({"path": r["path"], "sha256": r["sha256"], "pointer": "", "value": r["rows"]} for r in receipts[:3])
    for j, name in enumerate(labels):
        a, s = action[:, j], state[:, j]
        finite = np.isfinite(a) & np.isfinite(s)
        row = {"nonfinite_frames": int((~finite).sum())}
        joints[name] = row
        pointer = f"/features/action/names/{j}"
        if row["nonfinite_frames"]:
            add("reach.nonfinite_values", "unknown", f"{name}: {row['nonfinite_frames']} frame(s) hold NaN or infinite "
                "values; those frames are excluded.", pointer=pointer, value=name)
        if finite.sum() < still_min or np.ptp(s[finite]) == 0:
            add("reach.joint_undecidable", "unknown", f"{name}: the reading " + ("never changes (a dead, disconnected or "
                "unused motor)" if finite.sum() >= still_min else "has too few finite frames") + ", so its command "
                "cannot be compared.", pointer=pointer, value=name)
            row["judged"] = False
            continue
        both = np.concatenate([a[finite], s[finite]])
        span = float(np.percentile(both, 99) - np.percentile(both, 1))
        row["range_p1_p99"] = round(span, 4)
        if "gripper" in name.lower() or NOT_JUDGED & set(re.split(r"[^a-z0-9]+", name.lower().removesuffix(".pos"))):
            row["judged"] = False  # grasping blocks grippers and fingers by design
            continue
        # Position commands follow the reading (r near 1 on real teleop data); deltas or velocities stored under
        # `.pos` names do not. A joint held at one end for most of its commanded range also lowers r, so the held-end
        # test runs first and this guard only decides joints that show no held end.
        correlation = float(np.corrcoef(a[finite], s[finite])[0, 1]) if np.std(a[finite]) > 0 else 0.0
        row["action_state_correlation"] = round(correlation, 4)
        # The absolute floors are in motor-position units. Feetech degree mode leaves a signature: every reading times
        # 4095/360 lands on a whole or half step (LeRobot 146d847 motors_bus.py:965-968 maps steps around a mid that is
        # whole or half). Otherwise a range of at least 10 units is used as a numerical heuristic, not proof of units;
        # radians or metres then get relative thresholds only.
        sample = s[finite][:20000] * 4095.0 / 360.0
        steps = np.abs(sample * 2 - np.round(sample * 2))
        degree_steps = bool(sample.size) and float(np.mean(steps < 2e-3)) > 0.99
        floors = degree_steps or span >= 10.0
        row["units_basis"] = ("Feetech degree steps" if degree_steps else
                              "range of at least 10 units (heuristic)" if floors else "relative thresholds only")
        still = _still([a, s], segments, max(STILL * span, ABS_STILL if floors else 0.0)) & finite
        gap = a - s
        top, bottom = float(np.percentile(s[finite], 98)), float(np.percentile(s[finite], 2))
        band = max(LIMIT_BAND * span, ABS_BAND if floors else 0.0)
        away = still & (s > bottom + band) & (s < top - band)
        tracks = float(np.median(gap[away])) if away.sum() >= at_end_min else None
        row.update(still_frames=int(still.sum()), median_gap_away_from_ends=None if tracks is None else round(tracks, 4))
        flagged = False
        for side, end, sign in (("highest", top, 1), ("lowest", bottom, -1)):
            at = still & ((s >= end - band) if sign > 0 else (s <= end + band))
            if at.sum() < at_end_min or (tracks is None and correlation < 0.5):
                continue
            excess = float(np.median(gap[at])) * sign
            # Compare with the signed residual elsewhere: an offset away from the end in either direction neither
            # creates nor hides a held end.
            # The command must actually go beyond the reading at that end; a rest pose where both agree is not a held
            # end even when the joint sits off its command elsewhere (that is a steady offset).
            if excess <= 0 or (excess - sign * tracks if tracks is not None else excess) <= FLAG * span:
                continue
            reached = float(np.median(s[at]))
            beyond = float(np.percentile(a[at], 99 if sign > 0 else 1))
            runs = sum(1 for idx in segments if (sel := idx[at[idx]]).size and float(np.median(gap[sel])) * sign > FLAG * span)
            if runs == 0:
                continue
            flagged = True
            row["held_at_an_end"] = {"side": side, "reading": round(reached, 4), "median_excess": round(excess, 4),
                                     "command_p99": round(beyond, 4), "still_frames": int(at.sum()), "runs": runs}
            add("reach.held_at_an_end", "warning",
                f"{name}: on {int(at.sum())} still frames where the reading sat at its {side} values (about {reached:.2f}), "
                f"the recorded action was a median {excess:.2f} beyond it ({excess / span:.0%} of the joint's range; "
                f"commands reached {beyond:.2f} at the 99th percentile), in {runs} of {len(segments)} recording runs"
                + ("" if tracks is None else f"; away from that end the median residual was {tracks:+.2f}")
                + ". This is a residual between the recorded action and the recorded observation: the pattern of a "
                  "joint held at one end of its travel while its command continues. Whether the motor received these "
                  f"commands is not recorded ({LEROBOT} stores the teleoperator action before robot-side processing and "
                  "clipping, src/lerobot/scripts/lerobot_record.py:325-327, 362-367). Possible causes: a range limit on "
                  "this robot (calibration writes ranges as servo limits, src/lerobot/motors/motors_bus.py:780-781, "
                  "except full-turn joints such as SO-101 wrist_roll, src/lerobot/robots/so_follower/so_follower.py:"
                  "155-163), a software joint limit (OpenArm clips goals to its configured limits, src/lerobot/robots/"
                  "openarm_follower/openarm_follower.py:289-296), a mechanical stop or obstruction, a stall under load, a "
                  "leader with a wider range, or slow tracking.",
                pointer=pointer, value=name, extra=data_evidence,
                remedy="Check this joint's travel and calibration on the recording robot. If it cannot reach these "
                       "commands, consider clipping the recorded actions to the reached range before training, then "
                       "recheck.")
        if not flagged and correlation < 0.5:
            add("reach.not_comparable", "unknown", f"{name}: the recorded actions do not follow the reading "
                f"(correlation {correlation:.2f}) and show no held end. They may be changes or velocities rather than "
                "positions, or the joint may be blocked or idle, so they are not compared.", pointer=pointer, value=name)
            row["judged"] = False
            continue
        if not flagged and tracks is not None and abs(tracks) > FLAG * span and away.sum() >= still_min:
            flagged = True
            add("reach.steady_offset", "warning",
                f"{name}: on {int(away.sum())} still frames away from its ends, the recorded action and the reading "
                f"differed by a median {tracks:+.2f} ({tracks / span:+.0%} of the joint's range). This is a residual "
                "between recorded values. Possible causes: a zero (homing) mismatch between leader and follower, load "
                "the motor cannot hold, or a control-gain issue.", pointer=pointer, value=name, extra=data_evidence,
                remedy="Check this motor's homing offset on both arms and its tracking under load, then recheck.")
        large = still & (np.abs(gap) > FLAG * span)
        share = float(large.sum() / still.sum()) if still.sum() else 0.0
        row["share_of_still_frames_beyond_threshold"] = round(share, 4)
        if not flagged and share >= SHARE and large.sum() >= at_end_min:
            # A mixed pattern (no held end, no steady offset) is not reliable enough to warn on, e.g. policy outputs in
            # evaluation recordings; it is reported as undecided rather than as agreement.
            add("reach.mixed_residual", "unknown",
                f"{name}: on {share:.0%} of its {int(still.sum())} still frames the recorded action and the reading "
                f"differ by more than {FLAG:.0%} of the joint's range (median {float(np.median(np.abs(gap[large]))):.2f}), "
                "without a held end or a steady offset, so this joint is not judged. Possible causes include stalls, "
                "obstructions, actions that are policy outputs rather than teleoperator positions, or tracking lag.",
                pointer=pointer, value=name)
            continue
        if not flagged:
            if still.sum() < at_end_min:
                add("reach.joint_undecidable", "unknown", f"{name}: only {int(still.sum())} still frames (at least "
                    f"{at_end_min} are needed), so its command cannot be compared.", pointer=pointer, value=name)
            else:
                add("reach.joint", "pass", f"{name}: on {int(still.sum())} still frames the median residual between the "
                    f"recorded command and reading is {float(np.median(gap[still])):+.2f}; {share:.0%} of them exceed "
                    f"{FLAG:.0%} of the joint's range, below the reporting rules.", pointer=pointer, value=name)
    return _finish(root, findings, joints, coverage)


def _finish(root: Path, findings: list, joints: dict, coverage: dict | None) -> dict:
    counts = {s: sum(f["status"] == s for f in findings) for s in ("error", "warning", "unknown", "pass")}
    status = next((s for s in ("error", "warning", "unknown") if counts[s]), "pass")
    return {"schema_version": 1, "kind": "commanded_reach", "status": status, "counts": counts, "dataset": str(root),
            "coverage": coverage, "joints": joints, "findings": findings,
            "scope": "Recorded action against recorded observation.state on still frames. LeRobot records the "
                     "teleoperator action before robot-side processing, so delivery to the motor is not observed. "
                     "Grippers and hand joints are measured but not judged. A residual is triage evidence, not a "
                     "calibration diagnosis; a pass does not establish calibration."}
