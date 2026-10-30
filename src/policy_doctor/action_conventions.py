"""Action-convention fingerprint (I12): what a dataset's numbers say, against what a consumer assumes.

LeRobot's relative-action step, openpi's DeltaActions and GR00T's RELATIVE joint representation all convert
positionally: ``action[..., i] - state[..., i]`` for every masked coordinate i
(LeRobot 146d847 src/lerobot/processor/relative_action_processor.py:44-70; openpi 215abfb src/openpi/transforms.py:
204-222; Isaac-GR00T 944fe70 gr00t/data/state_action/state_action_processor.py:639-664 with JointPose differences).
That is meaningful only when action[i] is an absolute target of the same quantity, in the same units, as state[i].
The identity-level tracking statistic below tests exactly that precondition on the recorded data, without fitting a
scale: rho = std(action_t - state_{t+k}) / std(state) at the best small lag k.

Coordinates are reported as ``absolute`` (rho small and the level fit's slope ~1), ``incremental`` (rho large and the
action explains the state's per-step change, unless the state also pulls toward the action like a follower of a
jittery absolute target; delta and velocity differ only by a scale the data cannot attribute) or ``unknown``. Names are
used only as declarations to compare against the numbers. No model, consumer code or network is touched.

Consumer findings keep (a) the data fingerprint, (b) the configured transform and (c) declared or documented intent
apart: an inferred convention plus a transform warns; an error needs a declaration the data agrees with.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from pathlib import Path

import numpy as np

from .artifacts import InputError, contained, load_checkpoint, load_dataset, read_json, sha256, step_kind
from .diagnostics import MODES, normal_mode, normalizer

SCHEMA_VERSION = 1
KIND = "action_conventions"
MAX_ROWS = 200_000
MAX_FILES = 1000
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_LAG_SECONDS, MAX_LAG = 0.5, 25
RATE_LAG_SECONDS, RATE_MIN_LAG = 0.1, 3
MIN_PAIRS = 100
COPY_PROBE_ROWS = 2000
MAX_COORDINATES = 4096
# Decision thresholds, calibrated on 50 real LeRobot datasets plus 5 GR00T demo datasets (I12 DIAGNOSIS.md).
ABSOLUTE_RHO = 0.5          # residual of action - state below half the state's spread: same quantity, tracked
INCREMENTAL_RHO = 0.9       # the action does not track the state's level ...
INCREMENTAL_R2 = 0.5        # ... but explains at least half the variance of the state's per-step change
HIGH_ABSOLUTE_RHO = 0.25
HIGH_INCREMENTAL_R2 = 0.8
CONFLICT_R2 = 0.8          # a level fit that is also a strong rate fit (periodic motion) decides nothing
# Same quantity, same units: the level fit's slope is ~1 (held-out leader-follower data 0.97-1.05; census absolute p05-p95
# 0.91-1.13). Narrower than rho alone, which admits 0.6x-1.45x scaled targets (I12 validator V3).
ABSOLUTE_SLOPE = (0.85, 1.15)
SATURATION_FRACTION = 0.05  # actions outside the state's observed range; called saturation only if they track it inside
# A held joint with leader jitter and a lagging follower mimics an incremental command (rho >> 1, the jittery target
# explains the state's change). Its follower pulls the state toward the action: ds = b_a a + b_s s + c with -b_s/b_a ~ 1
# (~c for a target recorded at scale c, sign-flipped included), and the state level explains much of what the action
# leaves (I12 validator V1/V3/recheck: |ratio| 1.0-62, partial R^2 >= 0.33). Census incremental: the gate turns 4 of 45
# into unknown (xarm 009[0], 009[2], 022[0], 024[2]: mean reversion of the end effector, partial 0.30-0.38).
FOLLOWER_RATIO_MIN, FOLLOWER_PARTIAL_R2 = 0.5, 0.2
UNIT_NORM_TOL, BLOCK_FRACTION = 1e-3, 0.99
STATS_MIN_SEPARATION = 2.0  # absolute vs relative statistics distance needed before saved statistics are attributed

LEROBOT_RELATIVE = ("LeRobot 146d847 src/lerobot/processor/relative_action_processor.py:44-70 (action - state[..., :d] "
                    "for masked d), 125-142 (mask from action_names/exclude_joints)")
LEROBOT_DOCS = "LeRobot 146d847 docs/source/action_representations.mdx:59-93 (absolute vs relative), 128 (relative stats required)"
OPENPI_DELTA = "openpi 215abfb src/openpi/transforms.py:204-245 (DeltaActions/AbsoluteActions), 433-452 (make_bool_mask)"
GROOT_RELATIVE = ("Isaac-GR00T 944fe70 gr00t/data/state_action/state_action_processor.py:363-396 (only RELATIVE with "
                  "use_relative_action is converted), 639-664 (relative to the group's reference state)")
GROOT_MODALITY = "Isaac-GR00T 944fe70 gr00t/data/dataset/lerobot_episode_loader.py:60-63, 325-338 (modality.json start/end/original_key)"
GROOT_POSE = "Isaac-GR00T 944fe70 gr00t/data/state_action/pose.py:266-273 (joint widths must match)"
LEROBOT_INVERSE = "LeRobot 146d847 src/lerobot/processor/relative_action_processor.py:195-237 (AbsoluteActionsProcessorStep)"
OPENPI_INVERSE = "openpi 215abfb src/openpi/transforms.py:225-244 (AbsoluteActions)"
GROOT_INVERSE = "Isaac-GR00T 944fe70 gr00t/data/state_action/state_action_processor.py:473-526 (unapply_action, RELATIVE)"
STATUS_ORDER = ("error", "warning", "unknown", "pass")

_POSITION = re.compile(r"(\.pos$|(^|[._])(pos|position|positions|qpos)($|[._]))")
# Bare wx/wy/wz are not velocity words: LeRobot's ee.wx/wy/wz are an absolute rotation vector (LeRobot 146d847
# src/lerobot/robots/so_follower/robot_kinematic_processor.py:150-156, 359); a velocity word must be present.
_VELOCITY = re.compile(r"(^|[._])(vel|velocity|velocities|qvel|vx|vy|vz)($|[._])")
_DELTA = re.compile(r"(^|[._])(delta|deltas|diff)($|[._])")
_GRIPPER = re.compile(r"gripper")
_QUAT = [("qx", "qy", "qz", "qw"), ("qw", "qx", "qy", "qz"), ("rx", "ry", "rz", "rw")]
_QUAT_XYZW = ("x", "y", "z", "w")  # also a position plus a width: a quaternion only with a rotation word in every name
_ROTATION_WORD = re.compile(r"quat|orient|rot")
_EULER = ("roll", "pitch", "yaw")


class _Report:
    def __init__(self):
        self.findings: list[dict] = []

    def add(self, code, status, message, *evidence, remedy=""):
        self.findings.append({"code": code, "status": status, "message": message, "remedy": remedy,
                              "evidence": [e for e in evidence if e]})


def _ref(path, digest, pointer, value):
    return {"path": str(path), "sha256": digest, "pointer": pointer, "value": value}


def _short(name):
    return str(name).lower().rsplit(".", 1)[0] if str(name).lower().endswith(".pos") else str(name).lower()


# --------------------------------------------------------------------------------------------- data intake


def _numeric_vector(feature: dict) -> bool:
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        return False
    try:
        return np.dtype(feature.get("dtype", "float32")).kind == "f"
    except TypeError:
        return False


def _vector_keys(features: dict) -> list[str]:
    """Every 1-D float action or state feature (consumers such as GR00T may slice any of them)."""
    return [k for k, f in features.items() if (k == "action" or k.startswith("action.") or k == "observation.state"
                                                or k.startswith("observation.state.")) and _numeric_vector(f)]


def _group_pairs(features: dict) -> list[tuple[str, str | None]]:
    """RoboClue's default pairing: action <-> observation.state, action.<g> <-> observation.state.<g>."""
    actions = [k for k, f in features.items() if (k == "action" or k.startswith("action.")) and _numeric_vector(f)]
    states = {k for k, f in features.items()
              if (k == "observation.state" or k.startswith("observation.state.")) and _numeric_vector(f)}
    pairs = []
    for key in actions:
        if key == "action":
            pairs.append((key, "observation.state" if "observation.state" in states else None))
            continue
        suffix = key[len("action."):]
        candidates = [suffix, suffix.replace("action", "state").replace("cmd", "state")]
        pairs.append((key, next((f"observation.state.{c}" for c in candidates if f"observation.state.{c}" in states), None)))
    return pairs


def _matrix(column, width: int) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if array.null_count:
        raise InputError("Vector column contains nulls")
    kind = array.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        if not (pa.types.is_floating(kind.value_type) or pa.types.is_integer(kind.value_type)):
            raise InputError(f"Unsupported vector element type {kind.value_type}")
        lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
        if len(lengths) and (lengths != width).any():
            raise InputError(f"Vector rows do not all have the declared width {width}")
        values = array.flatten()
        if values.null_count:
            raise InputError("Vector column contains null elements")
        try:
            matrix = values.to_numpy(zero_copy_only=False).astype(float).reshape(len(array), width)
        except (TypeError, ValueError) as exc:
            raise InputError(f"Vector column cannot be read as {len(array)} x {width} numbers: {exc}") from exc
    elif width == 1 and (pa.types.is_floating(kind) or pa.types.is_integer(kind)):
        matrix = array.to_numpy(zero_copy_only=False).astype(float).reshape(-1, 1)
    else:
        raise InputError(f"Unsupported vector column type {kind}")
    if not np.isfinite(matrix).all():
        raise InputError("Vector column contains non-finite values")
    return matrix


def _read(ds, keys: list[str], max_rows: int, max_files: int) -> tuple[dict | None, dict]:
    coverage = {"rows_read": 0, "files_read": 0, "files_total": 0, "complete": False, "files": []}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        coverage["reason"] = "pyarrow is not installed (install policy-deployment-doctor[dataset]); numbers were not read"
        return None, coverage
    data_dir = contained(ds.root, "data")
    paths = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    coverage["files_total"] = len(paths)
    if not paths:
        coverage["reason"] = "the dataset has no data/**/*.parquet files"
        return None, coverage
    wanted = list(dict.fromkeys([*keys, "episode_index", "frame_index"]))
    widths = {k: ds.features[k]["shape"][0] for k in keys}
    widths.update({"episode_index": 1, "frame_index": 1})
    blocks: dict[str, list[np.ndarray]] = {k: [] for k in wanted}
    digests = []
    for path in paths[:max_files]:
        path = contained(ds.root, str(path.relative_to(ds.root)))
        if path.stat().st_size > MAX_FILE_BYTES:
            raise InputError(f"Parquet file exceeds {MAX_FILE_BYTES} bytes: {path}")
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        missing = [k for k in wanted if k not in names]
        if missing:
            coverage["reason"] = f"{path.name} lacks columns {missing[:4]}"
            return None, coverage
        table = parquet.read(columns=wanted)
        take = min(table.num_rows, max_rows - coverage["rows_read"])
        table = table.slice(0, take)
        for key in wanted:
            blocks[key].append(_matrix(table.column(key), widths[key]))
        digest = sha256(path)
        digests.append(digest)
        coverage["files"].append({"path": str(path), "sha256": digest})
        coverage["rows_read"] += take
        coverage["files_read"] += 1
        if coverage["rows_read"] >= max_rows:
            break
    coverage["complete"] = coverage["files_read"] == len(paths) and coverage["rows_read"] < max_rows
    if not coverage["rows_read"]:
        coverage["reason"] = "the data files contain no rows"
        return None, coverage
    data = {k: np.concatenate(v) for k, v in blocks.items()}
    episode, frame = data["episode_index"][:, 0], data["frame_index"][:, 0]
    order = np.lexsort((frame, episode))
    data = {k: v[order] for k, v in data.items()}
    coverage["data_path"] = str(data_dir)
    coverage["data_sha256"] = hashlib.sha256("\n".join(digests).encode()).hexdigest()
    return data, coverage


class _Series:
    """Episode segmentation and within-episode lagged index pairs."""

    def __init__(self, episode: np.ndarray, frame: np.ndarray):
        n = len(episode)
        breaks = np.flatnonzero((np.diff(episode) != 0) | (np.diff(frame) != 1)) + 1
        starts = np.concatenate([[0], breaks]).astype(int)
        lengths = np.diff(np.concatenate([starts, [n]]))
        self.segment = np.repeat(np.arange(len(starts)), lengths)
        self.position = np.arange(n) - starts[self.segment]
        self.length = lengths[self.segment]
        self.episodes = int(len(np.unique(episode)))
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def pairs(self, k: int, extra: int = 0) -> np.ndarray:
        """Rows t such that t+k (and t+k+extra) lie in t's segment."""
        if (k, extra) not in self._cache:
            valid = (self.position + k >= 0) & (self.position + k + extra < self.length)
            self._cache[(k, extra)] = np.flatnonzero(valid)
        return self._cache[(k, extra)]


def _pair_statistics(a: np.ndarray, s: np.ndarray, series: _Series, fps: float) -> dict:
    """Identity-level, rate and copy statistics of one action coordinate against one state coordinate."""
    lag = int(min(MAX_LAG, math.ceil(MAX_LAG_SECONDS * fps)))
    a_std, s_std = float(a.std()), float(s.std())
    step = series.pairs(0, 1)
    ds = s[step + 1] - s[step]
    out = {"pairs": int(len(step)), "action_std": a_std, "state_std": s_std,
           "state_step_std": float(ds.std()) if len(ds) else 0.0}
    if len(step) < MIN_PAIRS or a_std <= 0 or s_std <= 0 or out["state_step_std"] <= 0:
        return out
    best = None
    copy = None
    for k in range(-2, lag + 1):
        rows = series.pairs(k)
        if len(rows) < MIN_PAIRS:
            continue
        residual = a[rows] - s[rows + k]
        rho = float(residual.std() / s_std)
        if best is None or rho < best[0]:
            best = (rho, k, float(residual.mean() / s_std))
        if k in (-1, 0, 1) and copy is None and float(np.abs(residual).max()) == 0.0:
            copy = k
    rate = None
    # An incremental command changes the state within a few frames; longer lags only add spurious correlation
    # between absolute targets and later motion (seen on ALOHA, D1 census).
    for k in range(-1, max(RATE_MIN_LAG, math.ceil(RATE_LAG_SECONDS * fps)) + 1):
        rows = series.pairs(k, 1)
        if len(rows) < MIN_PAIRS:
            continue
        x = s[rows + k + 1] - s[rows + k]
        y = a[rows]
        x, y = x - x.mean(), y - y.mean()
        vx, vy = float(x @ x), float(y @ y)
        if vx <= 0 or vy <= 0:
            continue
        cxy = float(x @ y)
        r2 = cxy * cxy / (vx * vy)
        if rate is None or r2 > rate[0]:
            rate = (r2, k, cxy / vx)
    if best:
        out.update({"level_rho": round(best[0], 6), "level_lag": best[1], "offset_over_state_std": round(best[2], 6)})
        rows = series.pairs(best[1])
        out.update(_affine(s[rows + best[1]], a[rows]))  # a level fit outside ABSOLUTE_SLOPE is never called absolute
        # Actions beyond the state's observed range (the follower stops at a limit) distort a full-range scale fit.
        lo, hi = float(s.min()), float(s.max())
        beyond = (a[rows] > hi + 0.01 * (hi - lo)) | (a[rows] < lo - 0.01 * (hi - lo))
        out.update({"state_range": [lo, hi], "action_beyond_state_range": round(float(beyond.mean()), 6)})
        if beyond.mean() >= SATURATION_FRACTION and (~beyond).sum() >= MIN_PAIRS:
            inside = _affine(s[rows + best[1]][~beyond], a[rows][~beyond])
            out.update({f"inside_range_{k}": v for k, v in inside.items()})
    if rate:
        out.update({"rate_r2": round(rate[0], 6), "rate_lag": rate[1], "action_per_state_step": round(rate[2], 6),
                    "action_per_state_unit_per_second": round(rate[2] / fps, 6)})
        out.update(_follower(a, s, series, rate[1], rate[0]))
    if copy is not None:
        out["exact_copy_lag"] = copy
    return out


def _follower(a: np.ndarray, s: np.ndarray, series: _Series, k: int, rate_r2: float) -> dict:
    """ds_{t+k} = b_a a_t + b_s s_{t+k} + c: does the state's change also pull toward the action, as a follower of an
    absolute target does (-b_s/b_a ~ 1), and how much of what the action leaves does the state level explain?"""
    rows = series.pairs(k, 1)
    y, xa, xs = s[rows + k + 1] - s[rows + k], a[rows], s[rows + k]
    if len(rows) < MIN_PAIRS or y.std() <= 0 or xa.std() <= 0 or xs.std() <= 0 or rate_r2 >= 1:
        return {}
    design = np.column_stack([xa, xs, np.ones(len(rows))])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    if beta[0] == 0:
        return {}
    full = 1 - float(((y - design @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {"follower_ratio": round(float(-beta[1] / beta[0]), 6),
            "follower_partial_r2": round(max(0.0, (full - rate_r2) / (1 - rate_r2)), 6)}


def _affine(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = x - x.mean(), y - y.mean()
    vx, vy = float(x @ x), float(y @ y)
    if vx <= 0 or vy <= 0:
        return {}
    cxy = float(x @ y)
    return {"affine_r2": round(cxy * cxy / (vx * vy), 6), "affine_slope": round(cxy / vx, 6)}


def _scale_note(stats: dict, scaled: bool) -> str:
    """Why a level fit is not identity: actions beyond the state's range (saturation) or a different scale."""
    slope, affine = stats.get("affine_slope"), stats.get("affine_r2", 0.0)
    beyond, inside = stats.get("action_beyond_state_range", 0.0), stats.get("inside_range_affine_slope")
    note = ""
    if beyond >= SATURATION_FRACTION:
        lo, hi = stats["state_range"]
        note = f"; {beyond:.0%} of actions lie outside the state's observed range [{lo:.4g}, {hi:.4g}]"
        if (inside is not None and ABSOLUTE_SLOPE[0] <= inside <= ABSOLUTE_SLOPE[1]
                and stats["inside_range_affine_r2"] >= 0.7):
            return note + (f", while inside it the action follows the state's level at slope {inside:.3g} (R^2="
                           f"{stats['inside_range_affine_r2']:.2f}): the follower stops at the range edge (a limit or "
                           "saturation), so a full-range scale fit reflects the saturation, not the units")
    if scaled or (slope is not None and affine >= 0.7 and not ABSOLUTE_SLOPE[0] <= slope <= ABSOLUTE_SLOPE[1]):
        note = (f"; the action follows the state's level at scale {slope if slope is None else format(slope, '.3g')} "
                f"(R^2={affine:.2f}), so units, calibration or sign may differ") + note
    return note


def _classify(stats: dict) -> tuple[str, str | None, str]:
    if stats.get("pairs", 0) < MIN_PAIRS:
        return "unknown", None, f"fewer than {MIN_PAIRS} within-episode pairs"
    if stats["action_std"] <= 0 or stats["state_std"] <= 0 or stats["state_step_std"] <= 0:
        return "unknown", None, "the action or its state partner does not move in the scanned rows"
    rho, r2 = stats.get("level_rho"), stats.get("rate_r2", 0.0)
    if rho is None:
        return "unknown", None, "no lag had enough pairs"
    if rho <= ABSOLUTE_RHO and r2 >= CONFLICT_R2:
        # A scaled increment of a periodic motion matches the state level a quarter period later.
        return "unknown", None, (f"both the state's level (rho={rho:.3g}) and its per-step change (R^2={r2:.2f}) fit; "
                                 "periodic motion cannot separate absolute from incremental")
    slope = stats.get("affine_slope")
    if rho <= ABSOLUTE_RHO and (slope is None or not ABSOLUTE_SLOPE[0] <= slope <= ABSOLUTE_SLOPE[1]):
        return "unknown", None, (f"the action is close to the state's level (rho={rho:.3g}) but its slope is outside "
                                 f"{list(ABSOLUTE_SLOPE)}" + _scale_note(stats, True))
    if rho <= ABSOLUTE_RHO:
        confidence = "high" if rho <= HIGH_ABSOLUTE_RHO else "medium"
        copy = stats.get("exact_copy_lag")
        reason = (f"action equals state at lag {copy} exactly" if copy is not None else
                  f"action tracks the state's level (rho={rho:.3g} at lag {stats['level_lag']})")
        return "absolute", confidence, reason
    ratio, partial = stats.get("follower_ratio"), stats.get("follower_partial_r2", 0.0)
    if (rho >= INCREMENTAL_RHO and r2 >= INCREMENTAL_R2 and ratio is not None
            and abs(ratio) >= FOLLOWER_RATIO_MIN and partial >= FOLLOWER_PARTIAL_R2):
        return "unknown", None, (f"the action does not track the state's level (rho={rho:.3g}) and explains {r2:.0%} of "
                                 f"its per-step change, but the change also pulls the state toward the action (ds = b_a "
                                 f"action + b_s state with b_s/b_a = {-ratio:.3g}: -1 for a first-order follower of the "
                                 f"action itself, -c for a target recorded at scale c; the state level explains "
                                 f"{partial:.0%} of what the action leaves), as a follower of a jittery held target does; "
                                 "absolute (possibly scaled) and incremental are not separable")
    if rho >= INCREMENTAL_RHO and r2 >= INCREMENTAL_R2:
        confidence = "high" if r2 >= HIGH_INCREMENTAL_R2 and rho >= 1.0 else "medium"
        return "incremental", confidence, (f"action does not track the state's level (rho={rho:.3g}) but explains "
                                           f"{r2:.0%} of the state's per-step change (lag {stats['rate_lag']})")
    return "unknown", None, (f"neither level tracking (rho={rho:.3g}) nor rate tracking (R^2={r2:.2f}) is decisive"
                             + _scale_note(stats, False))


def _declared(name: str | None, group: str) -> str | None:
    text = " ".join(x for x in (str(name or "").lower(), group.lower()) if x)
    if _VELOCITY.search(text):
        return "velocity"
    if _DELTA.search(text):
        return "delta"
    if name is not None and _POSITION.search(str(name).lower()):
        return "position"
    return None


def _blocks(matrix: np.ndarray) -> list[dict]:
    found = []
    width = matrix.shape[1]
    for i in range(width - 3):
        norm = np.linalg.norm(matrix[:, i:i + 4], axis=1)
        fraction = float(np.mean(np.abs(norm - 1) < UNIT_NORM_TOL))
        if fraction >= BLOCK_FRACTION:
            found.append({"kind": "unit_norm_4", "start": i, "fraction": round(fraction, 4)})
    for i in range(width - 5):
        u, v = matrix[:, i:i + 3], matrix[:, i + 3:i + 6]
        ok = ((np.abs(np.linalg.norm(u, axis=1) - 1) < UNIT_NORM_TOL) & (np.abs(np.linalg.norm(v, axis=1) - 1) < UNIT_NORM_TOL)
              & (np.abs(np.sum(u * v, axis=1)) < UNIT_NORM_TOL))
        if float(np.mean(ok)) >= BLOCK_FRACTION:
            found.append({"kind": "rot6d", "start": i, "fraction": round(float(np.mean(ok)), 4)})
    return found


def _named_rotations(names: list[str] | None) -> list[dict]:
    if not names:
        return []
    short = [_short(n).split(".")[-1].split("_")[-1] for n in names]
    found = []
    for i in range(len(short) - 3):
        window = tuple(short[i:i + 4])
        if window in _QUAT or (window == _QUAT_XYZW and all(_ROTATION_WORD.search(str(n).lower()) for n in names[i:i + 4])):
            found.append({"declared": "quaternion", "start": i, "names": names[i:i + 4]})
    for i in range(len(short) - 2):
        if tuple(short[i:i + 3]) == _EULER:
            found.append({"declared": "euler_rpy", "start": i, "names": names[i:i + 3]})
    for i, name in enumerate(names):
        lower = str(name).lower()
        if "rot6d" in lower or "axis_angle" in lower or "rotvec" in lower:
            found.append({"declared": "rot6d" if "rot6d" in lower else "axis_angle", "start": i, "names": [name]})
    return found


def _rotation(names, matrix, role) -> list[dict]:
    numeric = _blocks(matrix) if matrix is not None else []
    unit = {b["start"] for b in numeric if b["kind"] == "unit_norm_4"}
    out = []
    for item in _named_rotations(names):
        entry = dict(item, role=role)
        if item["declared"] == "quaternion":
            entry["values"] = ("unit-norm (quaternion-consistent; component order follows the names only)"
                               if item["start"] in unit else "not unit-norm" if matrix is not None else "not read")
        else:
            entry["values"] = "not decidable from values (Euler and axis-angle are both three angles)"
        out.append(entry)
    named = {i["start"] for i in out if i["declared"] == "quaternion"}
    for block in numeric:
        if block["start"] not in named:
            out.append({"declared": None, "role": role, "start": block["start"], "values": block["kind"],
                        "fraction": block["fraction"]})
    return out


def _gripper(names, a, s, coords) -> list[dict]:
    out = []
    for coord in coords:
        name = coord.get("name")
        if not name or not _GRIPPER.search(str(name).lower()):
            continue
        column = a[:, coord["index"]]
        if not len(column):
            out.append({"index": coord["index"], "name": name, "note": "No rows were read."})
            continue
        levels = np.unique(column)
        entry = {"index": coord["index"], "name": name, "range": [float(column.min()), float(column.max())],
                 "distinct_values": int(min(len(levels), 1000))}
        if len(levels) <= 3:
            entry["levels"] = [float(x) for x in levels]
        if s is not None and coord.get("state_index") is not None:
            partner = s[:, coord["state_index"]]
            entry["state_range"] = [float(partner.min()), float(partner.max())]
            if column.std() > 0 and partner.std() > 0:
                corr = float(np.corrcoef(column, partner)[0, 1])
                entry["correlation_with_state"] = round(corr, 4)
                entry["polarity_vs_state"] = "same" if corr >= 0.5 else "inverted" if corr <= -0.5 else "unknown"
        entry["note"] = "Open/close meaning is not decidable from values; polarity is reported relative to the state only."
        out.append(entry)
    return out


# --------------------------------------------------------------------------------------------- fingerprint


class _Fingerprint:
    def __init__(self, ds, data, coverage):
        self.ds, self.data, self.coverage = ds, data, coverage
        self.fps = float(ds.config["fps"])
        self.series = _Series(data["episode_index"][:, 0], data["frame_index"][:, 0]) if data is not None else None
        self.cache: dict[tuple, dict] = {}
        self.state_keys = [k for k in _vector_keys(ds.features) if k.startswith("observation.state")
                           and data is not None and k in data]
        self.groups_cache: list[dict] = []

    def names(self, key):
        """One name per coordinate, or None (grouped or unresolved encodings are never expanded)."""
        feature = self.ds.features.get(key, {})
        value = feature.get("names")
        if isinstance(value, dict) and len(value) == 1:
            value = next(iter(value.values()))
        width = (feature.get("shape") or [None])[0]
        if isinstance(value, list) and len(value) == width and all(isinstance(x, str) and x for x in value):
            return value
        return None

    def coordinate(self, akey, i, skey, j) -> dict:
        cache_key = (akey, i, skey, j)
        if cache_key in self.cache:
            return self.cache[cache_key]
        anames, snames = self.names(akey), self.names(skey) if skey else None
        entry = {"action_key": akey, "index": i, "name": anames[i] if anames and i < len(anames) else None,
                 "state_key": skey, "state_index": j,
                 "state_name": snames[j] if snames and j is not None and j < len(snames) else None}
        entry["declared"] = _declared(entry["name"], akey[len("action"):])
        entry["state_declared"] = _declared(entry["state_name"], (skey or "")[len("observation.state"):]) if skey else None
        if self.data is None:
            entry.update(convention="unknown", confidence=None, reason=self.coverage.get("reason", "numbers were not read"))
        elif akey not in self.data or (skey is not None and skey not in self.data):
            entry.update(convention="unknown", confidence=None, reason="the column was not read")
        elif skey is None or j is None:
            entry.update(convention="unknown", confidence=None, reason="no positional state partner")
        else:
            stats = _pair_statistics(self.data[akey][:, i], self.data[skey][:, j], self.series, self.fps)
            convention, confidence, reason = _classify(stats)
            entry.update(convention=convention, confidence=confidence, reason=reason, statistics=stats)
        if self.data is not None and akey in self.data:
            copies = self._copies(akey, i)
            if copies:
                entry["exact_copies"] = copies
        self.cache[cache_key] = entry
        return entry

    def _copies(self, akey, i) -> list[dict]:
        """State coordinates (any state feature) that this action coordinate equals exactly at lag -1, 0 or 1."""
        a = self.data[akey][:, i]
        if a.std() <= 0:
            return []
        found = []
        for k in (-1, 0, 1):
            rows = self.series.pairs(k)
            if len(rows) < MIN_PAIRS:
                continue
            head = rows[:COPY_PROBE_ROWS]
            for skey in self.state_keys:
                matrix = self.data[skey]
                for j in np.flatnonzero(np.all(matrix[head + k] == a[head, None], axis=0)):
                    if np.array_equal(matrix[rows + k, j], a[rows]):
                        names = self.names(skey)
                        found.append({"state_key": skey, "state_index": int(j), "lag": k,
                                      "state_name": names[j] if names and j < len(names) else None})
        return found[:8]

    def groups(self) -> list[dict]:
        result = []
        for akey, skey in _group_pairs(self.ds.features):
            width = self.ds.features[akey]["shape"][0]
            swidth = self.ds.features[skey]["shape"][0] if skey else 0
            coords = [self.coordinate(akey, i, skey, i if i < swidth else None) for i in range(width)]
            decided = {c["convention"] for c in coords if c["convention"] != "unknown"}
            convention = (decided.pop() if len(decided) == 1 and all(c["convention"] != "unknown" for c in coords)
                          else "mixed" if len(decided) > 1 else "partly_" + next(iter(decided)) if decided else "unknown")
            a = self.data[akey] if self.data is not None else None
            s = self.data[skey] if self.data is not None and skey else None
            group = {"action_key": akey, "state_key": skey, "action_width": width, "state_width": swidth or None,
                     "convention": convention, "coordinates": coords,
                     "rotation": _rotation(self.names(akey), a, "action") + (_rotation(self.names(skey), s, "state") if skey else []),
                     "gripper": _gripper(self.names(akey), a, s, coords) if a is not None else []}
            result.append(group)
        return result


def _data_ref(fp: _Fingerprint, pointer: str, value) -> dict | None:
    if fp.data is None:
        return None
    return _ref(fp.coverage["data_path"], fp.coverage["data_sha256"], pointer, value)


def _info_ref(ds, pointer, value):
    artifact = ds.artifacts[0]
    return _ref(artifact.path, artifact.sha256, pointer, value)


def _compact(coord: dict) -> dict:
    keep = ("level_rho", "level_lag", "rate_r2", "rate_lag", "action_per_state_step", "exact_copy_lag")
    return {"action_key": coord.get("action_key"), "index": coord["index"], "name": coord.get("name"),
            "state_key": coord.get("state_key"), "state_index": coord.get("state_index"),
            "state_name": coord.get("state_name"), "convention": coord["convention"],
            **{k: coord.get("statistics", {}).get(k) for k in keep if k in coord.get("statistics", {})}}


# --------------------------------------------------------------------------------------------- consumers


def _compare_masked(report, fp, consumer, masked, consumer_ref, remedy_relative, restore, documented=None):
    """masked: list of (akey, i, skey, j) the consumer converts relative to the state. restore: what the consumer's own
    artifacts show about the inverse at inference; documented: the consumer's own statement that the transform is
    deliberate on incremental data."""
    coords = [fp.coordinate(*m) for m in masked]
    incremental = [c for c in coords if c["convention"] == "incremental"]
    unknown = [c for c in coords if c["convention"] == "unknown"]
    pointer = "/fingerprint"
    if incremental:
        # A documented compatibility transform covers the undecided coordinates too, once the data shows the documented
        # kind where it can decide; asking to confirm absolute targets would contradict the consumer's own statement.
        _relative_on_incremental(report, fp, consumer, incremental, consumer_ref, remedy_relative, restore, documented,
                                 unknown if documented else [])
    if unknown and documented and not incremental:
        report.add("convention.relative_undecided", "unknown",
                   f"{consumer} converts coordinate(s) {_listed(unknown)} relative to the state; the data cannot decide "
                   f"their convention (first: {unknown[0]['reason']}). {documented}",
                   consumer_ref, _data_ref(fp, pointer, [_compact(c) for c in unknown[:8]]))
    elif unknown and not documented:
        report.add("convention.relative_undecided", "unknown",
                   f"{consumer} converts coordinate(s) {_listed(unknown)} relative to the state, but the data cannot decide "
                   f"whether they are absolute targets of the same quantity (first: {unknown[0]['reason']}).",
                   consumer_ref, _data_ref(fp, pointer, [_compact(c) for c in unknown[:8]]),
                   remedy="Confirm from the collection code that these actions are absolute targets in the state's units.")
    absolute = [c for c in coords if c["convention"] == "absolute"]
    if absolute and not incremental:
        report.add("convention.relative_on_absolute", "pass",
                   f"{consumer} converts {len(absolute)} coordinate(s) that the data shows are absolute targets tracking "
                   "their positional state partner, which is the precondition for relative conversion.",
                   consumer_ref, _data_ref(fp, pointer, [_compact(c) for c in absolute[:8]]))


def _listed(coords):
    return ", ".join(f"{c['index']} ({c['name']})" for c in coords[:6])


def _relative_on_incremental(report, fp, consumer, incremental, consumer_ref, remedy, restore, documented, undecided=()):
    """Severity keeps (a) the data fingerprint, (b) the configured transform and (c) declared or documented intent apart:
    (a)+(b) alone warn; names declaring delta/velocity (the dataset's own declaration) escalate to error; names declaring
    position, which (a) contradicts, leave it unknown; the consumer's own documentation of the transform passes."""
    def evidence(coords):
        return _data_ref(fp, "/fingerprint", [_compact(c) for c in coords[:8]])

    if documented:
        report.add("convention.relative_on_incremental", "pass",
                   f"{consumer} subtracts the current state from coordinate(s) {_listed(incremental)}, which the data "
                   "shows are already incremental"
                   + (f", and from {_listed(undecided)}, whose convention the data cannot decide" if undecided else "")
                   + f". {documented}", consumer_ref, evidence([*incremental, *undecided]))
        return
    declared = [c for c in incremental if c["declared"] in ("delta", "velocity")
                and c.get("state_declared") not in ("delta", "velocity")]
    position = [c for c in incremental if c["declared"] == "position"]
    inferred = [c for c in incremental if c not in declared and c not in position]
    target = ("The training target becomes (command - state): a reparameterised, harder target that carries the state's "
              f"level, so the model must reproduce the state to recover a small command; {restore}")
    if declared:
        key = declared[0]["action_key"]
        report.add("convention.relative_on_incremental", "error",
                   f"{consumer} subtracts the current state from coordinate(s) {_listed(declared)}, whose names declare "
                   f"{'/'.join(sorted({c['declared'] for c in declared}))} commands and whose numbers agree (they do not "
                   f"track the state's level), while relative conversion presupposes absolute targets of the state's "
                   f"quantity. {target}.", consumer_ref,
                   _info_ref(fp.ds, f"/features/{key}/names", fp.ds.config["features"][key].get("names")),
                   evidence(declared), remedy=remedy[:1].upper() + remedy[1:])
    if inferred:
        report.add("convention.relative_on_incremental", "warning",
                   f"{consumer} subtracts the current state from coordinate(s) {_listed(inferred)}, which the data "
                   "fingerprints as incremental (they do not track the state's level; no name or config declares their "
                   f"kind). {target}. Nothing raises.", consumer_ref, evidence(inferred),
                   remedy="If the collection code confirms incremental commands: " + remedy)
    if position:
        key = position[0]["action_key"]
        report.add("convention.relative_on_incremental", "unknown",
                   f"{consumer} subtracts the current state from coordinate(s) {_listed(position)}, whose names declare "
                   "positions (the absolute targets the conversion presupposes) but whose numbers fingerprint as "
                   "incremental. An inferred convention does not overrule a declaration, so the data does not settle it.",
                   consumer_ref, _info_ref(fp.ds, f"/features/{key}/names", fp.ds.config["features"][key].get("names")),
                   evidence(position),
                   remedy="Confirm from the collection code whether these are absolute targets in the state's units.")


def _lerobot(report, fp, checkpoint):
    cp = load_checkpoint(checkpoint)
    steps = [s for s in cp.pre if step_kind(s["name"]) == "relative" and s["config"].get("enabled")]
    flag = cp.config.get("use_relative_actions")
    config_ref = _ref(cp.artifacts[0].path, cp.artifacts[0].sha256, "/use_relative_actions", flag)
    flagged = [s["name"] for s in cp.pre + cp.post if s["config"].get("use_relative_action") is True
               or s["config"].get("use_relative_actions") is True]
    unmodeled = [s["name"] for s in cp.pre if step_kind(s["name"]) == "unsupported"]
    if not steps and (flag is True or flagged or unmodeled):
        report.add("convention.relative_unmodeled", "unknown",
                   f"The checkpoint saves no enabled generic RelativeActionsProcessorStep ({LEROBOT_RELATIVE}), but "
                   + ("declares use_relative_actions and " if flag is True else "")
                   + f"has unmodeled step(s) {list(dict.fromkeys(flagged + unmodeled))[:4]} that may convert actions "
                   "natively, so which coordinates are made relative cannot be compared with the data.", config_ref,
                   remedy="Confirm from the step's pinned source which coordinates it converts, then compare them.")
        return
    if not steps:
        group = next((g for g in fp.groups_cache if g["action_key"] == "action"), None)
        convention = group["convention"] if group else "unknown"
        report.add("convention.lerobot_passthrough", "pass",
                   "The checkpoint saves no enabled relative-action step: it is trained on the dataset's stored actions "
                   f"unchanged, so it outputs the dataset's convention ({convention}); the robot interface must "
                   "interpret commands that way.", config_ref)
        return
    step = steps[0]
    artifact = next(a for a in cp.artifacts if a.path == step["source"])
    step_ref = _ref(artifact.path, artifact.sha256, step["pointer"] + "/config", step["config"])
    dim = cp.features.get("action", {}).get("shape", [None])[-1] or fp.ds.features.get("action", {}).get("shape", [0])[0]
    mask = _lerobot_mask(step["config"], dim)
    state_width = fp.ds.features.get("observation.state", {}).get("shape", [0])[0]
    consumer = f"LeRobot's RelativeActionsProcessorStep ({LEROBOT_RELATIVE})"
    if "action" not in fp.ds.features or not state_width:
        report.add("convention.relative_undecided", "unknown",
                   "The dataset lacks a single action or observation.state vector, which the LeRobot relative step reads.",
                   step_ref, _info_ref(fp.ds, "/features", sorted(fp.ds.features)))
        return
    action_width = fp.ds.features["action"]["shape"][0]
    if dim != action_width:
        report.add("convention.relative_undecided", "unknown",
                   f"The checkpoint's action has {dim} coordinates and the dataset's {action_width}; the positional mask "
                   "cannot be mapped onto this dataset (dimension checks belong to the compatibility audit).",
                   step_ref, _info_ref(fp.ds, "/features/action/shape", [action_width]))
        return
    if dim > state_width:
        report.add("convention.relative_width", "error",
                   f"The relative step subtracts state[..., :{dim}] from {dim} action coordinates, but observation.state "
                   f"has {state_width}; the native step raises a shape error at the first chunk ({LEROBOT_RELATIVE}).",
                   step_ref, _info_ref(fp.ds, "/features/observation.state/shape", [state_width]),
                   remedy="Train with a state whose first action_dim coordinates are the action's absolute counterparts.")
        return
    masked = [("action", i, "observation.state", i) for i, on in enumerate(mask) if on and i < dim]
    inverse = any(step_kind(s["name"]) == "absolute" and s["config"].get("enabled") for s in cp.post)
    restore = (f"the saved AbsoluteActionsProcessorStep ({LEROBOT_INVERSE}) adds the state back at inference, so deployed "
               "commands keep the dataset's convention" if inverse else
               f"the checkpoint saves no enabled AbsoluteActionsProcessorStep ({LEROBOT_INVERSE}) to add the state back "
               "at inference")
    _compare_masked(report, fp, consumer, masked, step_ref,
                    "train without use_relative_actions, or record absolute targets; exclude coordinates that are not "
                    "absolute through relative_exclude_joints.", restore)
    _lerobot_statistics(report, fp, cp, mask)


def _lerobot_mask(config: dict, dim: int) -> list[bool]:
    """Exact replica of RelativeActionsProcessorStep._build_mask (relative_action_processor.py:125-142)."""
    excluded, names = config.get("exclude_joints") or [], config.get("action_names")
    if not excluded or names is None:
        return [True] * dim
    tokens = [str(n).lower() for n in excluded if n]
    if not tokens:
        return [True] * dim
    mask = [not any(t == str(n).lower() or t in str(n).lower() for t in tokens) for n in names[:dim]]
    return mask + [True] * (dim - len(mask))


def _lerobot_statistics(report, fp, cp, mask):
    """Saved action statistics describe absolute or relative actions? (LeRobot docs: relative stats are required.)"""
    step = normalizer(cp, "pre")
    if (fp.data is None or not step or not step["stats"].get("action") or "action" not in fp.data
            or "observation.state" not in fp.data):
        return
    mode = normal_mode(step, "action", cp.features.get("action", {}))
    if mode not in MODES or mode == "IDENTITY":
        return
    chunk = cp.config.get("chunk_size")
    if type(chunk) is not int or chunk <= 0:
        return
    a, s = fp.data["action"], fp.data["observation.state"]
    saved = {k: np.asarray(step["stats"]["action"].get(k, []), dtype=float) for k in MODES[mode]}
    artifact = next(x for x in cp.artifacts if x.path == step.get("state_file"))
    stats_ref = _ref(artifact.path, artifact.sha256, "action." + "/".join(MODES[mode]), {"mode": mode})
    if any(v.shape != (a.shape[1],) for v in saved.values()):
        report.add("convention.relative_statistics", "unknown",
                   f"The checkpoint converts actions to relative before normalization, but its saved action {mode} "
                   f"statistics have shapes { {k: list(v.shape) for k, v in saved.items()} }, not [{a.shape[1]}] per "
                   "statistic; per-horizon (H x D) or other layouts are not compared, so whether they describe absolute "
                   "or relative actions is not decided.", stats_ref)
        return
    starts = fp.series.pairs(0, chunk - 1)
    if len(starts) == 0:
        return
    verdicts = []
    for i, on in enumerate(mask):
        if not on or i >= s.shape[1]:
            continue
        coord = fp.coordinate("action", i, "observation.state", i)
        if coord["convention"] != "absolute":
            continue
        absolute = a[:, i]
        relative = np.concatenate([a[starts + h, i] - s[starts, i] for h in range(chunk)])
        centre, spread = _centre_spread(saved, mode, i)
        absolute_cs = _centre_spread(_stats(absolute, mode), mode, None)
        relative_cs = _centre_spread(_stats(relative, mode), mode, None)
        d_abs, d_rel = _distance(centre, spread, *absolute_cs), _distance(centre, spread, *relative_cs)
        separation = _distance(*absolute_cs, *relative_cs)
        verdicts.append({"index": i, "name": coord["name"], "distance_to_absolute": round(d_abs, 4),
                         "distance_to_relative": round(d_rel, 4), "absolute_relative_separation": round(separation, 4)})
    if not verdicts:
        return
    # A coordinate votes only when its absolute and relative statistics are themselves well apart, and the saved
    # statistics sit clearly on one side (saved quantiles are approximations, so fixed small tolerances are fragile).
    decisive = [v for v in verdicts if v["absolute_relative_separation"] >= STATS_MIN_SEPARATION]
    absolute_like = [v for v in decisive if v["distance_to_absolute"] <= 0.25 * v["absolute_relative_separation"]
                     and v["distance_to_relative"] >= 0.75 * v["absolute_relative_separation"]]
    relative_like = [v for v in decisive if v["distance_to_relative"] <= 0.25 * v["absolute_relative_separation"]
                     and v["distance_to_absolute"] >= 0.75 * v["absolute_relative_separation"]]
    if absolute_like and not relative_like:
        report.add("convention.relative_statistics", "warning",
                   f"The checkpoint converts actions to relative before normalization, but its saved action {mode} "
                   "statistics match this dataset's absolute actions, not its relative actions (chunk_size "
                   f"{cp.config.get('chunk_size')}). Normalized training targets are then off-centre and mis-scaled; "
                   f"{LEROBOT_DOCS} requires relative statistics.", stats_ref,
                   _data_ref(fp, "/fingerprint", verdicts[:8]),
                   remedy="Recompute statistics with lerobot-edit-dataset --operation.type recompute_stats "
                          "--operation.relative_action true (same chunk_size and exclude_joints), then retrain.")
    elif relative_like and not absolute_like:
        report.add("convention.relative_statistics", "pass",
                   f"The saved action {mode} statistics match this dataset's relative actions.", stats_ref,
                   _data_ref(fp, "/fingerprint", verdicts[:8]))
    else:
        report.add("convention.relative_statistics", "unknown",
                   "The saved action statistics match neither this dataset's absolute nor its relative actions "
                   "decisively; they may come from other data.", stats_ref, _data_ref(fp, "/fingerprint", verdicts[:8]))


def _stats(values, mode):
    if mode == "MEAN_STD":
        return {"mean": np.array([values.mean()]), "std": np.array([values.std()])}
    q = {"MIN_MAX": (0.0, 1.0), "QUANTILES": (0.01, 0.99), "QUANTILE10": (0.10, 0.90)}[mode]
    lo, hi = MODES[mode]
    return {lo: np.array([np.quantile(values, q[0])]), hi: np.array([np.quantile(values, q[1])])}


def _centre_spread(stats, mode, i):
    pick = (lambda v: float(v[i])) if i is not None else (lambda v: float(v[0]))
    if mode == "MEAN_STD":
        return pick(stats["mean"]), pick(stats["std"])
    lo, hi = (pick(stats[n]) for n in MODES[mode])
    return (lo + hi) / 2, (hi - lo) / 2


def _distance(c1, w1, c2, w2):
    w1, w2 = max(abs(w1), 1e-12), max(abs(w2), 1e-12)
    return abs(c1 - c2) / max(w1, w2) + abs(math.log(w1 / w2))


def _groot(report, fp, config_path, embodiment):
    path = Path(config_path).expanduser().resolve()
    document = read_json(path)
    digest = sha256(path)
    kwargs = document.get("processor_kwargs", document)
    configs = kwargs.get("modality_configs") if isinstance(kwargs, dict) else None
    if not isinstance(configs, dict) or not configs:
        raise InputError("GR00T config needs processor_kwargs.modality_configs")
    tags = [t for t, v in configs.items() if isinstance(v, dict) and isinstance(v.get("action"), dict)]
    if embodiment is None and len(tags) == 1:
        embodiment = tags[0]
    if embodiment not in tags:
        report.add("convention.groot_embodiment", "unknown",
                   f"Select the embodiment tag to compare (groot_embodiment); the config declares {tags[:8]}.",
                   _ref(path, digest, "/processor_kwargs/modality_configs", tags))
        return
    base = "/processor_kwargs" if "processor_kwargs" in document else ""
    flag = kwargs.get("use_relative_action")
    action = configs[embodiment]["action"]
    keys = action.get("modality_keys") or []
    if not isinstance(keys, list) or any(not isinstance(k, str) or not k for k in keys):
        raise InputError("GR00T action modality_keys must be a list of nonempty strings")
    reps = action.get("action_configs")
    pointer = f"{base}/modality_configs/{embodiment}/action/action_configs"
    if flag is None and reps:
        report.add("convention.groot_relative_flag", "unknown",
                   "The config does not record use_relative_action, which decides whether RELATIVE groups are converted "
                   f"({GROOT_RELATIVE}).", _ref(path, digest, f"{base}/use_relative_action", None))
        return
    try:
        modality_path = contained(fp.ds.root, "meta/modality.json")
    except InputError:
        modality_path = None
    modality = read_json(modality_path) if modality_path and modality_path.is_file() else None
    if reps is None:
        reps = [None] * len(keys)
    if not isinstance(reps, list) or len(reps) != len(keys) or any(r is not None and not isinstance(r, dict) for r in reps):
        raise InputError("GR00T action_configs must have one object per action modality key")
    passthrough = []
    for index, (key, rep) in enumerate(zip(keys, reps, strict=True)):
        name = (rep or {}).get("rep", "ABSOLUTE")
        rep_ref = _ref(path, digest, f"{pointer}/{index}" if rep else f"{base}/modality_configs/{embodiment}/action", rep)
        consumer = f"GR00T action group '{key}' ({name}, use_relative_action={flag})"
        if name not in ("ABSOLUTE", "RELATIVE", "DELTA"):
            raise InputError(f"Unsupported GR00T ActionRepresentation {name!r}")
        if name == "ABSOLUTE" or (name == "RELATIVE" and not flag):
            passthrough.append(key)
            if name == "RELATIVE":
                report.add("convention.groot_relative_inactive", "warning",
                           f"{consumer}: RELATIVE is declared but use_relative_action is false, so GR00T converts nothing "
                           f"and trains on the stored actions ({GROOT_RELATIVE}).", rep_ref,
                           _ref(path, digest, f"{base}/use_relative_action", flag),
                           remedy="Set use_relative_action consistently with the declared representation.")
            continue
        if modality is None:
            report.add("convention.groot_modality", "unknown",
                       f"{consumer}: the dataset has no meta/modality.json, which GR00T uses to slice groups "
                       f"({GROOT_MODALITY}).", rep_ref)
            continue
        state_key = (rep or {}).get("state_key") or key
        aslice, sslice = _slice(modality, "action", key), _slice(modality, "state", state_key)
        if aslice is None or sslice is None or aslice[0] not in fp.ds.features or sslice[0] not in fp.ds.features:
            report.add("convention.groot_modality", "unknown",
                       f"{consumer}: meta/modality.json does not map action '{key}' and state '{state_key}' to dataset columns.",
                       rep_ref, _ref(modality_path, sha256(modality_path), "/action", list(modality.get("action", {}))))
            continue
        mod_ref = _ref(modality_path, sha256(modality_path), f"/action/{key}",
                       {"action": aslice, "state": sslice})
        (acol, a0, a1), (scol, s0, s1) = aslice, sslice
        if a1 > fp.ds.features[acol]["shape"][0] or s1 > fp.ds.features[scol]["shape"][0]:
            report.add("convention.groot_modality", "unknown",
                       f"{consumer}: meta/modality.json slices beyond the declared width of {acol} or {scol}.", rep_ref, mod_ref)
            continue
        eef = (rep or {}).get("type") == "EEF"
        if name == "DELTA":
            coords = [fp.coordinate(acol, a0 + j, scol, s0 + j) for j in range(min(a1 - a0, s1 - s0))]
            absolute = [c for c in coords if c["convention"] == "absolute"]
            report.add("convention.groot_delta_passthrough", "warning" if absolute else "pass",
                       f"{consumer}: pinned GR00T applies no conversion for DELTA (only RELATIVE is converted, {GROOT_RELATIVE})"
                       + (f"; the data shows coordinate(s) {[c['index'] for c in absolute][:6]} are absolute targets, so the "
                          "model learns absolute actions although the config declares deltas." if absolute else
                          "; the model learns the stored actions."), rep_ref, mod_ref,
                       _data_ref(fp, "/fingerprint", [_compact(c) for c in absolute[:8]]) if absolute else None,
                       remedy="Declare ABSOLUTE (or RELATIVE with use_relative_action) to match what GR00T does.")
            continue
        if not eef and a1 - a0 != s1 - s0:
            report.add("convention.relative_width", "error",
                       f"{consumer}: the action group has {a1 - a0} coordinates and its reference state group "
                       f"'{state_key}' has {s1 - s0}; GR00T's joint-space relative conversion raises ValueError "
                       f"({GROOT_POSE}).", rep_ref, mod_ref,
                       remedy="Reference a state group with the same joints, or declare the group ABSOLUTE.")
            continue
        width = 3 if eef else a1 - a0
        masked = [(acol, a0 + j, scol, s0 + j) for j in range(width)]
        label = consumer + (" (EEF: position coordinates compared; rotation composition is not modeled)" if eef else "")
        _compare_masked(report, fp, f"{label} ({GROOT_RELATIVE})", masked, rep_ref,
                        "declare the group ABSOLUTE (GR00T's default, which trains on stored actions), or record "
                        "absolute targets.",
                        f"GR00T's unapply_action ({GROOT_INVERSE}) adds the reference state back at inference, so "
                        "deployed commands keep the dataset's convention")
    if passthrough:
        report.add("convention.groot_passthrough", "pass",
                   f"GR00T trains on the stored actions of group(s) {passthrough[:8]} unchanged (ABSOLUTE, or RELATIVE "
                   f"without use_relative_action; {GROOT_RELATIVE}), so the model outputs the dataset's own convention.",
                   _ref(path, digest, pointer, [(r or {}).get("rep", "ABSOLUTE") for r in reps]))


def _slice(modality, role, key):
    entry = (modality.get(role) or {}).get(key)
    if not isinstance(entry, dict) or type(entry.get("start")) is not int or type(entry.get("end")) is not int:
        return None
    column = entry.get("original_key", "observation.state" if role == "state" else "action")
    if not 0 <= entry["start"] < entry["end"]:
        return None
    return column, entry["start"], entry["end"]


def _openpi(report, fp, config):
    source, digest = "openpi_config (in-memory object)", None
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
        config = read_json(path)
        source, digest = str(path), sha256(path)
    if not isinstance(config, dict):
        raise InputError("openpi_config must be an object or a JSON file")
    action_key, state_key = config.get("action_key", "action"), config.get("state_key", "observation.state")
    if not isinstance(action_key, str) or not isinstance(state_key, str):
        raise InputError("openpi_config action_key/state_key must be strings")
    if "delta_action_mask" in config:
        mask = config["delta_action_mask"]
        if not isinstance(mask, list) or not mask or len(mask) > 4096 or any(type(x) is not bool for x in mask):
            raise InputError("delta_action_mask must be a nonempty list of booleans")
        ref = _ref(source, digest, "/delta_action_mask", mask)
        label = "the declared openpi DeltaActions mask"
        restore = f"whether an AbsoluteActions output ({OPENPI_INVERSE}) adds the state back at inference is not declared here"
        documented = None
    else:
        resolved = _openpi_builtin(config)
        if isinstance(resolved, str):
            report.add("convention.openpi_config", "unknown", resolved,
                       {"path": str(config.get("source_root")), "sha256": None, "pointer": "/config_name",
                        "value": config.get("config_name")})
            return
        mask, ref, inverse, documented = resolved
        label = f"openpi config '{config['config_name']}'"
        restore = (f"its AbsoluteActions output (config.py:{inverse}; {OPENPI_INVERSE}) adds the state back at inference, "
                   "so deployed commands keep the dataset's convention" if inverse else
                   f"the pinned source pairs no AbsoluteActions output ({OPENPI_INVERSE}) with it")
    if mask is None:
        report.add("convention.openpi_passthrough", "pass",
                   f"{label} applies no DeltaActions transform: the model is trained on the stored actions.", ref)
        return
    consumer = f"{label} DeltaActions ({OPENPI_DELTA})"
    awidth = fp.ds.features.get(action_key, {}).get("shape", [0])[0]
    swidth = fp.ds.features.get(state_key, {}).get("shape", [0])[0]
    if not awidth or not swidth:
        report.add("convention.relative_undecided", "unknown",
                   f"The dataset lacks '{action_key}' or '{state_key}' to compare with the mask.", ref)
        return
    if len(mask) > swidth or len(mask) > awidth:
        report.add("convention.relative_width", "error",
                   f"The mask has {len(mask)} entries but the action has {awidth} and the state {swidth} coordinates; "
                   f"DeltaActions broadcasts state[..., :{len(mask)}] against the mask and raises ValueError ({OPENPI_DELTA}).",
                   ref, _info_ref(fp.ds, f"/features/{state_key}/shape", [swidth]))
        return
    masked = [(action_key, i, state_key, i) for i, on in enumerate(mask) if on]
    _compare_masked(report, fp, consumer + " (compared positionally on the dataset columns; openpi applies it after "
                    "its repack and input transforms)", masked, ref,
                    "do not apply DeltaActions to them; keep it for absolute targets only.", restore, documented)


def _openpi_builtin(config) -> tuple | str:
    """Resolve a builtin TrainConfig's DeltaActions mask from fingerprint-verified pinned source (AST only)."""
    from .openpi_interface import MANIFEST, Unsupported, source_data

    name, root = config.get("config_name"), config.get("source_root")
    if not isinstance(name, str) or not isinstance(root, (str, Path)):
        return "openpi_config needs config_name and source_root (a pinned openpi checkout), or delta_action_mask."
    try:
        trees, evidence, _ = source_data(Path(root).expanduser().resolve())
    except (Unsupported, OSError) as exc:
        return f"openpi source cannot be verified against pinned revision {MANIFEST['revision'][:7]}: {exc}"
    try:
        return _openpi_resolve(trees, evidence, name, root)
    except Unsupported as exc:
        return f"openpi config '{name}' uses an unmodeled construct: {exc}"


def _openpi_resolve(trees, evidence, name, root) -> tuple | str:
    from .openpi_interface import Unsupported, definition, dotted

    rel = "src/openpi/training/config.py"
    tree = trees[rel]
    configs = next((n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_CONFIGS" for t in n.targets)), None)
    calls = [c for c in getattr(configs, "elts", []) if isinstance(c, ast.Call) and any(
        k.arg == "name" and isinstance(k.value, ast.Constant) and k.value.value == name for k in c.keywords)]
    if len(calls) > 1:
        return f"openpi config '{name}' is defined {len(calls)} times in _CONFIGS of the pinned config.py."
    if not calls:
        spread = [f"{ast.unparse(e)} (config.py:{e.lineno})" for e in getattr(configs, "elts", []) if isinstance(e, ast.Starred)]
        return (f"openpi config '{name}' is not written out in _CONFIGS of the pinned config.py"
                + (f"; builtins added by {', '.join(spread)} are defined in modules outside the fingerprint-verified "
                   "source manifest, so their transforms are not resolved." if spread else "."))
    data = next((k.value for k in calls[0].keywords if k.arg == "data"), None)
    if not isinstance(data, ast.Call):
        return f"openpi config '{name}' has no data config call."
    try:
        class_name = dotted(data.func)
        cls = definition(tree, class_name.split(".")[-1])
    except Unsupported:
        return f"openpi config '{name}' uses an unmodeled data config."
    kwargs = {k.arg: k.value for k in data.keywords if k.arg}
    defaults = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None}
    path = Path(root).expanduser().resolve() / rel
    source_ref = {"path": str(path), "sha256": evidence[rel]["sha256"], "pointer": f"L{calls[0].lineno}",
                  "value": ast.unparse(data)[:400]}
    if any("DeltaActions" in ast.unparse(v) for v in kwargs.values()):
        return f"openpi config '{name}' passes custom transforms containing DeltaActions; not modeled."
    create = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create"), None)
    masks = []
    for node in ast.walk(create) if create else []:
        if isinstance(node, ast.If):
            for inner in node.body:
                if (isinstance(inner, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "delta_action_mask" for t in inner.targets)
                        and isinstance(inner.value, ast.Call) and dotted(inner.value.func).endswith("make_bool_mask")):
                    masks.append((node, inner))
    if not masks:
        if create and "DeltaActions" in ast.unparse(create):
            return f"openpi data config {class_name} applies DeltaActions in an unmodeled form."
        return None, source_ref, None, None
    if len(masks) != 1:
        return f"openpi data config {class_name} has several delta masks; not modeled."
    condition, assign = masks[0]
    test = condition.test

    def value_of(attr):
        node = kwargs.get(attr, defaults.get(attr))
        return node

    active = None
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) and test.value.id == "self":
        node = value_of(test.attr)
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            active = node.value
    elif (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
          and isinstance(test.left, ast.Attribute) and isinstance(test.left.value, ast.Name) and test.left.value.id == "self"):
        node = value_of(test.left.attr)
        if isinstance(node, ast.Constant) and node.value is None:
            active = False
        elif isinstance(node, (ast.Attribute, ast.Name)):
            active = dotted(node).split(".")[-1] == dotted(test.comparators[0]).split(".")[-1]
    if active is None:
        return f"openpi config '{name}': the DeltaActions condition '{ast.unparse(test)}' is not a literal."
    args = assign.value.args
    if not args or not all(isinstance(a, ast.Constant) and type(a.value) is int or isinstance(a, ast.UnaryOp)
               and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant) and type(a.operand.value) is int
               for a in args):
        return f"openpi config '{name}': make_bool_mask arguments are not literal."
    dims = [ast.literal_eval(a) for a in args]
    mask = [d > 0 for d in dims for _ in range(abs(d))]
    inverse = next((n.lineno for n in ast.walk(condition) if isinstance(n, ast.Call)
                    and ast.unparse(n.func).split(".")[-1] == "AbsoluteActions"), None)
    documented = (_openpi_documented(path, evidence[rel]["sha256"], configs, ast.unparse(data.func), condition,
                                     test.attr, inverse) if active and isinstance(test, ast.Attribute) else None)
    source_ref["value"] = {"data_config": ast.unparse(data)[:300], "condition": ast.unparse(test),
                           "condition_value": active, "mask_line": assign.lineno, "make_bool_mask": dims,
                           "absolute_actions_line": inverse}
    return (mask if active else None), source_ref, inverse, documented


def _openpi_documented(path, digest, configs, data_class, condition, flag, inverse) -> str | None:
    """The pinned source's own comment directly above the delta branch, when it says the data are already deltas
    (config.py:336-337 for LeRobotLiberoDataConfig, with the data statement at 332-333 in the block above), the inverse
    and the builtins that switch the flag off."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != digest:
        return None
    lines = raw.decode("utf-8").splitlines()
    end = start = condition.lineno - 1
    while start > 0 and lines[start - 1].strip().startswith("#"):
        start -= 1
    comment = " ".join(line.strip().lstrip("#").strip() for line in lines[start:end])
    if not re.search(r"\balready\b.*\bdeltas?\b", comment, re.IGNORECASE):
        return None
    # The comment block above it (after blank lines) states the data's own convention (config.py:332-333 for LIBERO).
    above = start
    while above > 0 and not lines[above - 1].strip():
        above -= 1
    first = above
    while first > 0 and lines[first - 1].strip().startswith("#"):
        first -= 1
    joined, offsets = "", []
    for line in lines[first:above]:
        offsets.append(len(joined))
        joined += line.strip().lstrip("#").strip() + " "
    hit = re.search(r"[^.]*\balready\b[^.]*\bdelta[^.]*\.", joined, re.IGNORECASE)
    data_note = ""
    if hit and above > first:
        lead = len(hit.group()) - len(hit.group().lstrip())
        a, b = (first + 1 + max(i for i, o in enumerate(offsets) if o <= pos) for pos in (hit.start() + lead, hit.end() - 1))
        data_note = f"the data as deltas (config.py:{a}-{b}: \"{hit.group().strip()}\") and "
    off = []
    for call in getattr(configs, "elts", []):
        cdata = next((k.value for k in getattr(call, "keywords", []) if k.arg == "data"), None)
        cname = next((k.value.value for k in getattr(call, "keywords", []) if k.arg == "name"
                      and isinstance(k.value, ast.Constant)), None)
        if isinstance(cdata, ast.Call) and ast.unparse(cdata.func) == data_class:
            off += [f"{cname} sets {flag}=False (config.py:{k.lineno})" for k in cdata.keywords
                    if k.arg == flag and isinstance(k.value, ast.Constant) and k.value.value is False]
    return (f"openpi documents {data_note}this transform as deliberate (config.py:{start + 1}-{end}: \"{comment}\"): the "
            "transform matches the checkpoints trained with it"
            + (f", and AbsoluteActions (config.py:{inverse}) adds the state back at inference, so deployed commands keep "
               "the dataset's convention" if inverse else "")
            + (f". The same data config without it: {'; '.join(off[:4])}." if off else "."))


# --------------------------------------------------------------------------------------------- entry point


def action_conventions(dataset, *, checkpoint=None, groot_config=None, groot_embodiment=None, openpi_config=None,
                       max_rows: int = MAX_ROWS, max_files: int = MAX_FILES) -> dict:
    """Fingerprint a dataset's action convention and compare it with declared consumer conventions."""
    if type(max_rows) is not int or not 1 <= max_rows <= 2_000_000:
        raise InputError("max_rows must be in [1, 2000000]")
    if type(max_files) is not int or not 1 <= max_files <= 10_000:
        raise InputError("max_files must be in [1, 10000]")
    ds = load_dataset(dataset)
    keys = _vector_keys(ds.features)
    if sum(ds.features[k]["shape"][0] for k in keys) > MAX_COORDINATES:
        raise InputError(f"More than {MAX_COORDINATES} action/state coordinates; refusing an unbounded scan")
    if not _group_pairs(ds.features):
        data, coverage = None, {"reason": "the dataset declares no 1-D float action feature"}
    else:
        data, coverage = _read(ds, keys, max_rows, max_files)
    fp = _Fingerprint(ds, data, coverage)
    fp.groups_cache = fp.groups()
    grouped = {(c["action_key"], c["index"], c["state_key"], c["state_index"]) for g in fp.groups_cache
               for c in g["coordinates"]}
    report = _Report()
    for group in fp.groups_cache:
        _fingerprint_findings(report, fp, group)
    def consumer(label, operation, *arguments):
        start = len(report.findings)
        operation(report, fp, *arguments)
        for finding in report.findings[start:]:
            finding['consumer'] = label

    if checkpoint is not None:
        consumer('LeRobot relative-action processor', _lerobot, checkpoint)
    if groot_config is not None:
        consumer('GR00T processor_config.json', _groot, groot_config, groot_embodiment)
    if openpi_config is not None:
        label = openpi_config.get('config_name') if isinstance(openpi_config, dict) else str(openpi_config)
        consumer(f"openpi config '{label}'", _openpi, openpi_config)
    counts = {s: sum(f["status"] == s for f in report.findings) for s in STATUS_ORDER}
    status = next((s for s in STATUS_ORDER if counts[s]), "pass")
    coverage = {k: v for k, v in coverage.items() if k != "files"} | {
        "files": coverage.get("files", [])[:50], "episodes": fp.series.episodes if fp.series else 0,
        "groups": len(fp.groups_cache)}
    return {"schema_version": SCHEMA_VERSION, "kind": KIND, "status": status, "counts": counts, "coverage": coverage,
            "fingerprint": {"method": {"absolute_rho_max": ABSOLUTE_RHO, "incremental_rho_min": INCREMENTAL_RHO,
                                       "incremental_rate_r2_min": INCREMENTAL_R2, "max_lag_seconds": MAX_LAG_SECONDS,
                                       "max_lag_frames": MAX_LAG, "rate_lag_seconds": RATE_LAG_SECONDS,
                                       "min_pairs": MIN_PAIRS, "absolute_slope": list(ABSOLUTE_SLOPE),
                                       "follower_abs_ratio_min": FOLLOWER_RATIO_MIN,
                                       "follower_partial_r2_min": FOLLOWER_PARTIAL_R2,
                                       "saturation_fraction": SATURATION_FRACTION,
                                       "pairing": "positional (action[i] vs state[i]), as every modeled consumer converts"},
                            "groups": fp.groups_cache,
                            "consumer_pairs": [c for key, c in fp.cache.items() if key not in grouped]},
            "findings": report.findings,
            "scope": ("Recorded action/state numbers and names of one LeRobot-format dataset, compared with declared "
                      "LeRobot relative-action steps, GR00T ActionRepresentation and openpi DeltaActions masks. No model, "
                      "consumer code or robot is run."),
            "limitations": [
                "Delta and velocity differ only by a scale; the data reports the scale, names may declare the kind.",
                "Frames (base vs end-effector) and Euler vs axis-angle are not decidable from values.",
                "Gripper open/close meaning is not decidable; polarity is reported relative to the state.",
                "A coordinate whose state partner barely moves, or whose follower does not track its command, stays unknown.",
                "Consumer masks are compared on the dataset's columns; transforms applied before them are not modeled.",
                "One NaN or infinite value in a scanned action, state or index column refuses the run; nothing is imputed.",
                f"'absolute' allows the level fit's slope within {list(ABSOLUTE_SLOPE)} (a +-15% scale tolerance): a "
                "calibration or unit mismatch inside it passes, so it does not prove identical units.",
                f"A jittery held target recorded at under {FOLLOWER_RATIO_MIN}x the state's scale (|b_s/b_a| < "
                f"{FOLLOWER_RATIO_MIN}) escapes the follower check and can still fingerprint as incremental.",
            ]}


def _fingerprint_findings(report, fp, group):
    akey = group["action_key"]
    pointer = f"/fingerprint/groups/{fp.groups_cache.index(group)}"
    coords = group["coordinates"]
    if fp.data is None:
        report.add("convention.fingerprint", "unknown", f"{akey}: {fp.coverage.get('reason', 'numbers were not read')}.",
                   _info_ref(fp.ds, f"/features/{akey}", fp.ds.features[akey].get("shape")))
        return
    moving = [c for c in coords if c["convention"] != "unknown"]
    undecided = [c for c in coords if c["convention"] == "unknown"]
    summary = {c: sum(x["convention"] == c for x in coords) for c in ("absolute", "incremental", "unknown")}
    status = "pass" if moving and not undecided else "unknown"
    report.add("convention.fingerprint", status,
               f"{akey}: {summary['absolute']} absolute, {summary['incremental']} incremental, {summary['unknown']} "
               f"undecided coordinate(s) against {group['state_key'] or 'no state'}"
               + (f"; undecided: {', '.join(str(c['name'] or c['index']) for c in undecided[:6])}." if undecided else "."),
               _data_ref(fp, pointer, summary))
    # A velocity command tracking a measured velocity is the same quantity; only a position partner conflicts.
    conflicts = [c for c in coords if (c["declared"] == "position" and c["convention"] == "incremental")
                 or (c["declared"] in ("velocity", "delta") and c["convention"] == "absolute"
                     and c.get("state_declared") not in ("velocity", "delta"))]
    copies = [c for c in coords if c.get("exact_copies")]
    if copies:
        report.add("convention.action_copies_state", "pass",
                   f"{akey}: {len(copies)} coordinate(s) equal recorded state values exactly, so they record an "
                   "observation rather than an independent command: "
                   + "; ".join(f"{c['name'] or c['index']} = {x['state_key']}[{x['state_index']}] ({x['state_name']}) at lag "
                               f"{x['lag']}" for c in copies[:4] for x in c["exact_copies"][:1]) + ".",
                   _data_ref(fp, pointer, [{"index": c["index"], "name": c["name"], "exact_copies": c["exact_copies"]}
                                           for c in copies[:8]]),
                   remedy="Confirm that recording the observation as the action is intended (lag 1 = next-state targets).")
    if conflicts:
        report.add("convention.declared_conflict", "warning",
                   f"{akey}: names and numbers disagree: "
                   + "; ".join(f"{c['name']} is named {c['declared']} but its numbers fingerprint as {c['convention']}" for c in conflicts[:6]) + ".",
                   _info_ref(fp.ds, f"/features/{akey}/names", fp.ds.config["features"][akey].get("names")),
                   _data_ref(fp, pointer, [_compact(c) for c in conflicts[:8]]),
                   remedy="Correct the feature names or the recorded values so documentation matches data.")
    for rotation in group["rotation"]:
        if rotation.get("declared") == "quaternion" and rotation.get("values") == "not unit-norm":
            key = akey if rotation["role"] == "action" else group["state_key"]
            report.add("convention.rotation_declared_conflict", "warning",
                       f"{key}: coordinates {rotation['names']} are named as a quaternion, but fewer than "
                       f"{BLOCK_FRACTION:.0%} of rows have unit norm, so they are not a quaternion.",
                       _info_ref(fp.ds, f"/features/{key}/names", fp.ds.config["features"][key].get("names")),
                       _data_ref(fp, pointer + "/rotation", rotation),
                       remedy="Name the coordinates by their actual representation (e.g. axis-angle) before any consumer "
                              "converts them.")
