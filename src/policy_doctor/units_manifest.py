"""Units, frames and timebase manifest of a LeRobot dataset, compared with the robot that will consume it.

LeRobot writes a vector feature's dtype, shape and coordinate names to ``meta/info.json``, never its unit or the motor
normalization it was recorded in (LeRobot 146d847 src/lerobot/utils/feature_utils.py:78-90). The robot decides the
unit at run time: SO, Koch, OMX and LeKiwi arms read and command body joints in DEGREES when ``use_degrees`` is set and
in RANGE_M100_100 otherwise, the gripper always in RANGE_0_100 (src/lerobot/robots/so_follower/so_follower.py:68-77).
The same coordinate names then carry different physical angles, and nothing in LeRobot compares the two.

The recorded values still carry the mode. Each mode maps integer encoder ticks to values on its own grid
(src/lerobot/motors/motors_bus.py:915-942):

* RANGE_M100_100: ``(b - min) / R * 200 - 100`` with ``b`` clamped into ``[min, max]``, ``R = max - min``: every value
  lies in [-100, 100] on a grid of step 200/R;
* DEGREES: ``(raw - (min + max) / 2) * 360 / 4095`` for 4096-step encoders: a grid of step 360/4095 (or its half);
* LeRobot before PR #777 recorded ``main_*`` joints in degrees on a 180/2048 grid from another zero
  (docs/source/backwardcomp.mdx:61-116).

A unit is stated only when the values decide it: enough distinct values sit on exactly one grid, or a value beyond
±100 rules RANGE_M100_100 out. Everything else is reported as unknown. Nothing is imported from LeRobot, no network or
hardware is touched, and every read stays inside the dataset root.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from . import artifacts, robot_interface
from .artifacts import LEROBOT_REVISION, InputError, contained, dataset_root, load_dataset, read_json, sha256

LEROBOT = f"LeRobot {LEROBOT_REVISION[:7]}"
SCHEMA_VERSION = 1
KIND = "units_manifest"
STATUSES = ("error", "warning", "unknown", "pass")

NORMALIZE = "src/lerobot/motors/motors_bus.py:915-942"
BOUNDED = "src/lerobot/motors/motors_bus.py:928-931"
UNNORMALIZE = "src/lerobot/motors/motors_bus.py:944-972"
NO_UNITS_RECORDED = "src/lerobot/utils/feature_utils.py:78-90"
BACKWARDCOMP = "docs/source/backwardcomp.mdx:61-116"
TIMESTAMPS = "src/lerobot/datasets/dataset_writer.py:223-227"
TELEOP_LOOP = "src/lerobot/scripts/lerobot_record.py:320-327, 362"
# Leaders whose body joints are always RANGE_M100_100 (their configs have no use_degrees).
FIXED_LEADERS = {"koch": "src/lerobot/teleoperators/koch_leader/koch_leader.py:45-50",
                 "omx": "src/lerobot/teleoperators/omx_leader/omx_leader.py:44-49"}
ROTATION_TOKENS = {"quat", "quats", "quaternion", "quaternions", "rot", "rotation", "rotations", "orient", "orientation",
                   "orientations"}

DEGREES, RANGE_M100_100, RANGE_0_100, PRE_PR777 = "DEGREES", "RANGE_M100_100", "RANGE_0_100", "PRE_PR777_DEGREES"
NOT_M100 = "NOT_RANGE_M100_100"
UNIT_TEXT = {DEGREES: "degrees from the middle of the calibrated range (use_degrees=True)",
             RANGE_M100_100: "-100..100 across the calibrated range (use_degrees=False)",
             RANGE_0_100: "0..100 across the calibrated range",
             PRE_PR777: "degrees from LeRobot's calibration before PR #777 (another zero and direction)"}

# Grid tests. A float32 value carries at most ~2e-4 grid steps of rounding error here, so TOL is ten times that, and
# a grid is evidence only once MIN_POINTS distinct grid points are occupied (a few values fit some grid by chance).
TOL = 2e-3
MIN_POINTS = 20
DEG_STEPS = 4095 / 360  # 4096-step encoders (sts3215, xl330, xl430): motors_bus.py:936-938
OLD_STEPS = 2048 / 180
MAX_SEARCH_VALUES = 1024
MAX_DISTINCT = 50_000  # a 4096-step encoder gives at most ~16k DEGREES half-steps; far more is not a motor reading
INDEX_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MODERN = re.compile(r"^(?P<arm>left_|right_|arm_)?(?P<motor>" + "|".join(MOTORS) + r")\.pos$")
OLD = re.compile(r"^(?P<arm>main_)(?P<motor>" + "|".join(MOTORS) + r")$")
MOTOR_FEATURES = ("observation.state", "action")  # where LeRobot puts motor readings (feature_utils.py:78-90)

# use_degrees default per robot type at 146d847, with the line that builds each arm's norm modes.
USE_DEGREES_DEFAULT = {
    "so100_follower": (True, "src/lerobot/robots/so_follower/config_so_follower.py:42"),
    "so101_follower": (True, "src/lerobot/robots/so_follower/config_so_follower.py:42"),
    "bi_so_follower": (True, "src/lerobot/robots/so_follower/config_so_follower.py:42 (per arm, "
                             "src/lerobot/robots/bi_so_follower/bi_so_follower.py:64, 78)"),
    "koch_follower": (False, "src/lerobot/robots/koch_follower/config_koch_follower.py:39"),
    "omx_follower": (False, "src/lerobot/robots/omx_follower/config_omx_follower.py:39"),
    "lekiwi": (True, "src/lerobot/robots/lekiwi/config_lekiwi.py:60"),
}
MODE_SOURCE = {
    "so100_follower": "src/lerobot/robots/so_follower/so_follower.py:68-77",
    "so101_follower": "src/lerobot/robots/so_follower/so_follower.py:68-77",
    "bi_so_follower": "src/lerobot/robots/so_follower/so_follower.py:68-77",
    "koch_follower": "src/lerobot/robots/koch_follower/koch_follower.py:47-56",
    "omx_follower": "src/lerobot/robots/omx_follower/omx_follower.py:47-56",
    "lekiwi": "src/lerobot/robots/lekiwi/lekiwi.py:51-61",
}
SCOPE = ("Units are inferred from the recorded values and the coordinate names only; LeRobot does not record them "
         f"({LEROBOT} {NO_UNITS_RECORDED}). A unit is stated when the values decide it, otherwise it is unknown. The "
         "deployment side is read from the command or config text (robot_interface); calibration files, the arm and "
         "the achieved loop rate are not inspected. A use_degrees left unset takes the default of LeRobot "
         f"{LEROBOT_REVISION[:7]} (e.g. src/lerobot/robots/so_follower/config_so_follower.py:42); other LeRobot "
         "versions may default differently. Remedies and effect estimates assume the policy was trained on this "
         "dataset's values as recorded; the policy (--policy.path) and its processors are not read. Radians, metres "
         "and other non-motor units are not decidable from values and are never asserted; quaternion component order "
         "comes from the coordinate names only. Values requantized to a step of 1/n (1/3, 1/6, 1/7, ...) sit on the "
         "RANGE_M100_100 grid of a 200*n-tick range and cannot be told from a genuine calibration; degrees written as "
         "ticks * 360/4096 (not LeRobot's 360/4095) sit on the 180/2048 grid and read as LeRobot's pre-PR #777 "
         "calibration. The timebase check compares the median within-episode timestamp step "
         f"with 1/fps; LeRobot writes timestamp = frame_index / fps ({LEROBOT} {TIMESTAMPS}), so LeRobot-written data "
         "passes it by construction. The comparison of the training and deployment loop rates is the readiness "
         "report's declared_rate axis.")


class _Report:
    def __init__(self):
        self.findings: list[dict] = []

    def add(self, code: str, status: str, message: str, evidence: list[dict], remedy: str | None = None) -> None:
        self.findings.append({"code": code, "status": status, "message": message, "remedy": remedy,
                              "evidence": evidence})


def _ev(path: str | Path | None, digest: str | None, pointer: str, value: Any) -> dict:
    return {"path": str(path) if path is not None else None, "sha256": digest, "pointer": pointer, "value": value}


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


# ----------------------------------------------------------------------------------------------------------- values

class _Column:
    """Running facts about one vector coordinate."""

    def __init__(self, motor: bool):
        self.motor = motor
        self.count = 0
        self.nonfinite = 0
        self.lo = math.inf
        self.hi = -math.inf
        self.unique: np.ndarray | None = np.empty(0) if motor else None
        self.too_many = False

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.nonfinite += int(values.size - np.count_nonzero(finite))
        values = values[finite]
        if not values.size:
            return
        self.count += int(values.size)
        self.lo = min(self.lo, float(values.min()))
        self.hi = max(self.hi, float(values.max()))
        if self.unique is not None and not self.too_many:
            self.unique = np.union1d(self.unique, values)
            if self.unique.size > MAX_DISTINCT:
                self.too_many, self.unique = True, np.empty(0)


def _grid(values: np.ndarray, scale: float, offset: float = 0.0) -> tuple[float, int]:
    """Largest distance (in grid steps) from the grid ``(v + offset) * scale`` in Z, and the grid points occupied."""
    if not values.size:
        return math.inf, 0
    k = (values + offset) * scale
    nearest = np.round(k)
    return float(np.max(np.abs(k - nearest))), int(np.unique(nearest).size)


def _range_grid(values: np.ndarray, low: float, span: float) -> tuple[int | None, int]:
    """Smallest calibrated range R (ticks, 2..4095) putting every value on ``(v - low) * R / span`` in Z."""
    if not values.size:
        return None, 0
    probe = values if values.size <= MAX_SEARCH_VALUES else values[np.linspace(0, values.size - 1, MAX_SEARCH_VALUES).astype(int)]
    for start in range(2, 4096, 256):
        ranges = np.arange(start, min(start + 256, 4096), dtype=np.float64)
        k = (probe[:, None] - low) * ranges[None, :] / span
        fits = np.nonzero(np.max(np.abs(k - np.round(k)), axis=0) < TOL)[0]
        for index in fits:
            R = int(ranges[index])
            deviation, points = _grid(values, R / span, -low)
            if deviation < TOL:
                return R, points
    return None, 0


def _rounded(values: np.ndarray) -> float | None:
    """1 or 0.1 when every value is a multiple of it: such values sit on coarse range grids by construction."""
    if not values.size:
        return None
    scale = max(1.0, float(np.max(np.abs(values))))
    for step, factor in ((1.0, 1), (0.1, 10)):
        k = values * factor
        if float(np.max(np.abs(k - np.round(k)))) < factor * scale * 2.4e-7 + 1e-9:
            return step
    return None


def _requantized(R: int, span: int) -> float | None:
    """The step span/R of a fitted range grid when it is a short decimal (at most 3 places) or a multiple of 1/16.

    Values rounded to such a step, or stored at reduced precision (bfloat16 or float16 carry multiples of 1/16 and
    coarser above 8 or 64), sit on the range grid of R = span/step by construction (0.05 -> 4000, 0.25 -> 800, 1/16 ->
    3200 ticks), so the fit shows the requantization, not an encoder. Calibrated ranges such as 2505 or 4095 ticks give
    no such step.
    """
    return span / R if (span * 1000) % R == 0 or (span * 16) % R == 0 else None


def _verdict(column: _Column, gripper: bool, reading: bool) -> dict:
    """Grid evidence for one motor-bus coordinate and the mode it decides, if any.

    ``reading`` is True for observation.state, which is always a normalized motor reading. A recorded action may be a
    policy output, stored before the bus clamps it (src/lerobot/rollout/strategies/core.py:528-533,
    src/lerobot/rollout/strategies/episodic.py:283-287), so its range alone rules no mode out.
    """
    values = column.unique if column.unique is not None else np.empty(0)
    facts: dict[str, Any] = {"distinct": int(values.size) if not column.too_many else f">{MAX_DISTINCT}"}
    # fits: the body-joint grids that hold every value, however few; an arm lends its mode only through them (_arm)
    fits: list[str] = []
    if not column.count:
        return {"mode": None, "reason": "no values read", "facts": facts, "fits": fits}
    peak = max(abs(column.lo), abs(column.hi))
    beyond = peak > 100 + 1e-3

    def undecided(reason: str) -> dict:
        """No grid decides: a state value beyond ±100 still rules RANGE_M100_100 out; nothing else does."""
        if beyond and reading and not gripper:
            return {"mode": NOT_M100, "facts": facts, "fits": fits,
                    "basis": f"a value reaches {peak:.2f}, beyond ±100, which RANGE_M100_100 readings cannot produce "
                             f"({LEROBOT} {BOUNDED}); {reason}"}
        if beyond and not reading:
            reason += ("; values beyond ±100 rule nothing out here, since recorded actions can be policy outputs "
                       "stored before the bus clamps them")
        return {"mode": None, "facts": facts, "fits": fits, "reason": reason}

    def requantized(R: int, span: int, mode: str) -> dict | None:
        step = _requantized(R, span)
        if step is None:
            return None
        facts["requantized_step"] = step
        return undecided(f"the values look requantized: they sit on the {mode} grid of a {R}-tick range, whose step "
                         f"{span}/{R} = {step:g} is a short decimal or a multiple of 1/16, which rounding or a "
                         "low-precision float (e.g. bfloat16) produces by construction, so the grid does not show "
                         "the encoder")

    if column.too_many:
        return undecided(f"more than {MAX_DISTINCT} distinct values, more than a 4096-step encoder records: not raw "
                         "motor readings")
    deg_dev, deg_points = _grid(values, 2 * DEG_STEPS)
    old_dev, old_points = _grid(values, OLD_STEPS)
    facts.update(degrees_grid_deviation=round(deg_dev, 6), degrees_grid_points=deg_points,
                 pre_pr777_grid_deviation=round(old_dev, 6), pre_pr777_grid_points=old_points)
    if not gripper:
        R, points = _range_grid(values, -100.0, 200.0) if not beyond else (None, 0)
        facts.update(range_m100_100_ticks=R, range_m100_100_points=points)
        fits += [m for m, ok in ((DEGREES, deg_dev < TOL), (RANGE_M100_100, R is not None),
                                 (PRE_PR777, old_dev < TOL)) if ok]
    rounded = _rounded(values)
    if rounded is not None:
        facts["rounded_to"] = rounded
        return undecided(f"every value is a multiple of {rounded:g}, so the encoder grids cannot be told apart")
    if gripper:
        inside = column.lo >= -1e-3 and column.hi <= 100 + 1e-3
        R, points = _range_grid(values, 0.0, 100.0) if inside else (None, 0)
        facts.update(range_0_100_ticks=R, range_0_100_points=points)
        if R is not None and points >= MIN_POINTS:
            return requantized(R, 100, RANGE_0_100) or {
                "mode": RANGE_0_100, "facts": facts, "fits": fits,
                "basis": f"all values lie in 0..100 on the RANGE_0_100 grid of a {R}-tick calibrated range "
                         f"({points} grid points; {LEROBOT} {NORMALIZE})"}
        return {"mode": None, "facts": facts, "fits": fits, "reason": "values do not sit on the RANGE_0_100 grid "
                                                                      "with enough distinct points"}
    deg = deg_dev < TOL and deg_points >= MIN_POINTS
    m100 = R is not None and points >= MIN_POINTS
    old = old_dev < TOL and old_points >= MIN_POINTS
    if deg and m100:
        return {"mode": None, "facts": facts, "fits": fits, "ambiguous": True,
                "reason": f"values fit both the DEGREES grid and the RANGE_M100_100 grid of a {R}-tick range "
                          "(the two coincide when the range divides 4550 ticks, e.g. 2275 or 910)"}
    if deg:
        return {"mode": DEGREES, "facts": facts, "fits": fits,
                "basis": f"all values lie on the DEGREES grid (raw - mid) * 360/4095 ({deg_points} half-step grid "
                         f"points, largest deviation {deg_dev:.1e} steps; {LEROBOT} {NORMALIZE})"}
    if m100:
        return requantized(R, 200, RANGE_M100_100) or {
            "mode": RANGE_M100_100, "facts": facts, "fits": fits,
            "basis": f"all values lie in -100..100 on the RANGE_M100_100 grid of a {R}-tick calibrated range "
                     f"({points} grid points) and off the DEGREES grid ({LEROBOT} {NORMALIZE})"}
    if old:
        return {"mode": PRE_PR777, "facts": facts, "fits": fits,
                "basis": f"all values lie on the 180/2048-degree grid of LeRobot's calibration before PR #777 "
                         f"({old_points} grid points; {LEROBOT} {BACKWARDCOMP})"}
    if values.size < MIN_POINTS:
        return undecided(f"only {values.size} distinct value(s)")
    return undecided("the values sit on no motor-bus grid (not raw motor readings, e.g. policy outputs or processed "
                     "values, or calibrations that changed between sessions)")


def _arm(verdicts: dict[str, dict]) -> dict:
    """One norm mode per arm: every body joint of an arm shares use_degrees (so_follower.py:68-77)."""
    decided = {v["mode"] for v in verdicts.values() if v["mode"] in (DEGREES, RANGE_M100_100, PRE_PR777)}
    beyond = sorted(j for j, v in verdicts.items() if v["mode"] == NOT_M100)
    if len(decided) > 1:
        return {"mode": None, "status": "conflict", "joints": sorted(verdicts),
                "reason": "body joints of one arm decide different modes: " +
                          ", ".join(f"{j}: {v['mode']}" for j, v in sorted(verdicts.items()) if v["mode"])}
    if decided:
        mode = decided.pop()
        if mode == RANGE_M100_100 and beyond:
            return {"mode": None, "status": "conflict", "joints": sorted(verdicts),
                    "reason": f"some joints sit on the RANGE_M100_100 grid but {beyond} exceed ±100"}
        by = sorted(j for j, v in verdicts.items() if v["mode"] == mode)
        # A joint whose values (read, however few) sit off the arm's grid is not a reading of this arm in that mode.
        off = sorted(j for j, v in verdicts.items() if v["mode"] != mode and mode not in v.get("fits", ())
                     and v.get("reason") != "no values read")
        return {"mode": mode, "status": "decided", "decided_by": by, "off_grid": off, "joints": sorted(verdicts),
                "basis": f"{', '.join(by)} decide {mode}; the arm's body joints share one mode "
                         f"({LEROBOT} src/lerobot/robots/so_follower/so_follower.py:68-77)"}
    if beyond:
        return {"mode": NOT_M100, "status": "partial", "decided_by": beyond, "joints": sorted(verdicts),
                "basis": f"{', '.join(beyond)} exceed ±100, which RANGE_M100_100 cannot produce ({LEROBOT} {BOUNDED})"}
    reasons = sorted({v.get("reason", "") for v in verdicts.values()})
    return {"mode": None, "status": "undecided", "joints": sorted(verdicts), "reason": "; ".join(r for r in reasons if r)}


# -------------------------------------------------------------------------------------------------------- reading

def _vector_features(snapshot) -> dict[str, dict]:
    result = {}
    for key, feature in snapshot.features.items():
        if key in INDEX_COLUMNS or len(feature.get("shape", [])) != 1:
            continue
        dtype = feature.get("dtype")
        try:
            numeric = isinstance(dtype, str) and np.dtype(dtype).kind in "biuf"
        except TypeError:
            numeric = False
        if not numeric:
            continue
        # Names that do not label each coordinate once (artifacts._features marks them) map no coordinate to a motor.
        names = feature.get("names")
        if isinstance(names, dict) and len(names) == 1:  # single-axis mapping, as artifacts.names reads it
            names = next(iter(names.values()))
        if (not isinstance(names, list) or len(names) != feature["shape"][0]
                or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names)):
            names = None
        result[key] = {"width": feature["shape"][0], "names": names}
    return result


def _motor(key: str, name: str | None) -> tuple[str, str, str] | None:
    """(arm prefix, motor, era) for a LeRobot motor-bus position coordinate."""
    if key not in MOTOR_FEATURES or not isinstance(name, str):
        return None
    match = MODERN.match(name)
    if match:
        return match.group("arm") or "", match.group("motor"), "current"
    match = OLD.match(name)
    if match:
        return "main_", match.group("motor"), "pre_pr777"
    return None


def _quaternions(key: str, names: list[str] | None) -> list[dict]:
    """Quaternions declared by coordinate names: q-prefixed names, or w/x/y/z in a feature whose last key part has a
    rotation token (``observation.state.orientation``, ``ee_quat``; not ``observation.protocol``)."""
    if not names:
        return []
    found = []
    rotation = bool(ROTATION_TOKENS & set(re.split(r"[^a-z0-9]+", key.split(".")[-1].lower())))
    for components in (("qx", "qy", "qz", "qw"), ("w", "x", "y", "z")):
        if components[0] == "w" and not rotation:
            continue
        if all(names.count(c) == 1 for c in components):
            index = [names.index(c) for c in components]
            order = "".join(c[-1] for c in sorted(components, key=names.index))
            found.append({"indices": index, "order": order, "names": [names[i] for i in sorted(index)]})
    return found


def _read(root: Path, features: dict[str, dict], motor_map: dict, quats: dict, max_rows: int, max_files: int):
    data = root / "data"
    paths = sorted(data.rglob("*.parquet")) if data.is_dir() else []
    coverage: dict[str, Any] = {"values_read": False, "data_files_found": len(paths), "data_files_read": 0,
                                "rows_read": 0, "complete": False, "files": []}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        coverage["reason"] = "pyarrow is not installed (pip install policy-deployment-doctor[dataset]); no values were read"
        return None, coverage
    if not paths:
        coverage["reason"] = "the dataset has no data/**/*.parquet files"
        return None, coverage
    coverage["complete"] = True
    columns = {key: [_Column(motor=(key, j) in motor_map) for j in range(spec["width"])] for key, spec in features.items()}
    norms = {(key, q["order"]): [math.inf, -math.inf, 0, 0] for key, qs in quats.items() for q in qs}
    steps: list[np.ndarray] = []
    skipped: dict[str, dict[str, int]] = {"null_rows": {}, "unread_rows": {}}
    for number, path in enumerate(paths):
        if number >= max_files or coverage["rows_read"] >= max_rows:
            coverage["complete"] = False
            coverage["stopped"] = f"max_files={max_files}" if number >= max_files else f"max_rows={max_rows}"
            break
        path = contained(root, str(path.relative_to(root)))
        if not path.is_file():
            raise InputError(f"Data file is not a readable file: {path}")
        if path.stat().st_size > 512 * 1024 * 1024:
            raise InputError(f"Parquet file exceeds the 512 MiB read limit: {path}")
        rows, last = 0, None
        try:
            parquet = pq.ParquetFile(path)
            present = set(parquet.schema_arrow.names)
            wanted = [k for k in features if k in present] + [c for c in ("timestamp", "episode_index") if c in present]
            for batch in parquet.iter_batches(batch_size=65536, columns=wanted):
                take = min(batch.num_rows, max_rows - coverage["rows_read"] - rows)
                batch = batch.slice(0, take) if take < batch.num_rows else batch
                rows += take
                _accumulate(batch, features, columns, quats, norms, skipped)
                if "timestamp" in present and "episode_index" in present:
                    last = _steps(batch, last, steps)
                if coverage["rows_read"] + rows >= max_rows:
                    break
            if rows < parquet.metadata.num_rows:
                coverage["complete"] = False
                coverage["stopped"] = f"max_rows={max_rows}"
        except (OSError, ValueError, TypeError, KeyError, NotImplementedError, pa.ArrowException) as exc:
            raise InputError(f"Cannot read Parquet {path}: {exc}") from exc
        coverage["files"].append({"path": str(path), "sha256": sha256(path), "rows_read": rows})
        coverage["rows_read"] += rows
        coverage["data_files_read"] += 1
    coverage["values_read"] = coverage["data_files_read"] > 0
    coverage.update(skipped)
    if skipped["unread_rows"]:  # those features' values were not read: coverage of the data is not complete
        coverage["complete"] = False
    return {"columns": columns, "norms": norms, "steps": np.concatenate(steps) if steps else np.empty(0)}, coverage


def _accumulate(batch, features: dict[str, dict], columns: dict, quats: dict, norms: dict, skipped: dict) -> None:
    """Add one batch's values. Null rows are dropped one by one and counted (skipped["null_rows"]); rows of a column
    that is absent or not a numeric vector of the declared width are counted as not read (skipped["unread_rows"])."""
    import pyarrow.compute as pc

    names = batch.schema.names
    for key, spec in features.items():
        array = batch.column(names.index(key)) if key in names else None
        if array is not None and array.null_count:
            skipped["null_rows"][key] = skipped["null_rows"].get(key, 0) + array.null_count
            array = array.filter(pc.is_valid(array))
        matrix = _matrix(array, spec["width"]) if array is not None else None
        if matrix is None:
            if batch.num_rows:
                skipped["unread_rows"][key] = skipped["unread_rows"].get(key, 0) + (
                    len(array) if array is not None else batch.num_rows)
            continue
        for j, column in enumerate(columns[key]):
            column.update(matrix[:, j])
        for q in quats.get(key, []):
            n = np.linalg.norm(matrix[:, q["indices"]], axis=1)
            n = n[np.isfinite(n)]
            record = norms[(key, q["order"])]
            if n.size:
                record[0], record[1] = min(record[0], float(n.min())), max(record[1], float(n.max()))
                record[2] += int(np.count_nonzero(np.abs(n - 1) <= 1e-2))
                record[3] += int(n.size)


def _episode_ranges(files: list[dict], key: str, width: int, joints: dict[str, int],
                    max_episodes: int = 256) -> dict[str, dict] | None:
    """Joints whose values sit, episode by episode, on one RANGE_M100_100 grid each but on two or more ranges: the
    readings of calibrations that change between sessions. A second pass over the rows the first pass read; None
    when the files have no episode_index."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    seen: dict[str, dict[int, np.ndarray]] = {motor: {} for motor in joints}
    for item in files:
        parquet = pq.ParquetFile(item["path"])
        if not {key, "episode_index"} <= set(parquet.schema_arrow.names):
            return None
        rows = 0
        for batch in parquet.iter_batches(batch_size=65536, columns=[key, "episode_index"]):
            if rows >= item["rows_read"]:
                break
            batch = batch.slice(0, min(batch.num_rows, item["rows_read"] - rows))
            rows += batch.num_rows
            batch = batch.filter(pc.is_valid(batch.column(0)))
            matrix = _matrix(batch.column(0), width)
            if matrix is None:
                continue
            episodes = np.asarray(batch.column(1).to_numpy(zero_copy_only=False))
            for episode in np.unique(episodes):
                block = matrix[episodes == episode]
                for motor, j in joints.items():
                    store = seen[motor]
                    store[int(episode)] = np.union1d(store.get(int(episode), np.empty(0)), block[:, j])
    found = {}
    for motor, by_episode in seen.items():
        ranges, checked = set(), 0
        for episode in sorted(by_episode)[:max_episodes]:
            values = by_episode[episode][np.isfinite(by_episode[episode])]
            if not values.size:
                continue
            R, _ = _range_grid(values, -100.0, 200.0) if np.max(np.abs(values)) <= 100 + 1e-3 else (None, 0)
            if R is None:
                break
            checked += 1
            if values.size >= MIN_POINTS:
                ranges.add(R)
        else:
            if checked and len(ranges) >= 2:
                found[motor] = {"episodes": checked, "of": len(by_episode), "ranges": sorted(ranges)}
    return found


def _steps(batch, last: tuple | None, steps: list[np.ndarray]) -> tuple | None:
    """Finite timestamp steps between consecutive rows of one episode, in stored order, within one file."""
    names = batch.schema.names
    stamp = np.asarray(batch.column(names.index("timestamp")).to_numpy(zero_copy_only=False), dtype=np.float64)
    episode = np.asarray(batch.column(names.index("episode_index")).to_numpy(zero_copy_only=False))
    if not stamp.size:
        return last
    if last is not None:
        stamp, episode = np.concatenate([[last[1]], stamp]), np.concatenate([[last[0]], episode])
    step = np.diff(stamp)[episode[1:] == episode[:-1]]
    steps.append(step[np.isfinite(step)])
    return episode[-1], stamp[-1]


def _matrix(array, width: int) -> np.ndarray | None:
    """(rows, width) float64 values of a numeric list or scalar column without null rows, or None for other types."""
    import pyarrow as pa

    def numeric(kind) -> bool:  # booleans (next.done, intervention) are read as 0/1
        return pa.types.is_floating(kind) or pa.types.is_integer(kind) or pa.types.is_boolean(kind)

    if array.null_count:
        return None
    kind = array.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        if not numeric(kind.value_type):
            return None
        flat = np.asarray(array.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
        if flat.size != len(array) * width:
            return None
        return flat.reshape(len(array), width)
    if width == 1 and numeric(kind):
        return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 1)
    return None


# ----------------------------------------------------------------------------------------------------- deployment

def _scan_use_degrees(text: str) -> dict[str, Any]:
    """``--robot.*use_degrees`` values given on the first lerobot command line (robot_interface does not keep them)."""
    found = [(tokens,) + robot_interface._program(tokens) for tokens in robot_interface._commands(text)]
    found = [item for item in found if item[1] is not None]
    if not found:
        return {}
    tokens, _, start = found[0]
    values: dict[str, Any] = {}
    index = start + 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--robot.") and token.split("=", 1)[0].endswith("use_degrees"):
            if "=" in token:
                key, raw = token[2:].split("=", 1)
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                key, raw = token[2:], tokens[index + 1]
                index += 1
            else:
                key, raw = token[2:], "true"
            try:
                values[key] = "$" if ("$" in raw or "`" in raw) else robot_interface._value(raw)
            except ValueError:
                values[key] = raw
        index += 1
    return values


def _config_use_degrees(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        config = read_json(path)
    else:
        import yaml  # from_config has already required PyYAML for a YAML file

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    robot = config.get("robot") if isinstance(config, dict) and isinstance(config.get("robot"), dict) else {}
    values = {}
    if "use_degrees" in robot:
        values["robot.use_degrees"] = robot["use_degrees"]
    for side in ("left", "right"):
        arm = robot.get(f"{side}_arm_config")
        if isinstance(arm, dict) and "use_degrees" in arm:
            values[f"robot.{side}_arm_config.use_degrees"] = arm["use_degrees"]
    return values


def _deployment(robot_command: str | None, robot_config: str | Path | None) -> dict | None:
    if robot_command is None and robot_config is None:
        return None
    if robot_command is not None and robot_config is not None:
        raise InputError("Pass either robot_command or robot_config, not both")
    if robot_command is not None:
        interface = robot_interface.from_command(robot_command)
        given = _scan_use_degrees(robot_command)
        source = {"path": None, "sha256": interface["source"]["sha256"], "kind": "command"}
    else:
        interface = robot_interface.from_config(robot_config)
        given = _config_use_degrees(Path(robot_config).expanduser())
        source = {"path": interface["source"]["path"], "sha256": interface["source"]["sha256"], "kind": "config"}
    robot_type = interface["robot_type"]
    via_config_path = any("--config_path" in note for note in interface.get("unknown", []))
    names = (interface.get("features", {}).get("observation.state") or {}).get("names") or []
    arms: dict[str, dict] = {}
    unknown = None
    if robot_type not in USE_DEGREES_DEFAULT:
        unknown = (f"robot type {robot_type} is not modeled here: its joint norm mode is unknown" if robot_type
                   else "the command sets no --robot.type")
        if robot_type == "lekiwi_client":
            unknown = ("lekiwi_client sends and receives joint values; the LeKiwi host's use_degrees decides their unit "
                       f"({LEROBOT} src/lerobot/robots/lekiwi/config_lekiwi.py:60, 85-100)")
    else:
        default, default_source = USE_DEGREES_DEFAULT[robot_type]
        sides = {"left_": "robot.left_arm_config.use_degrees", "right_": "robot.right_arm_config.use_degrees"} \
            if robot_type == "bi_so_follower" else {"arm_" if robot_type == "lekiwi" else "": "robot.use_degrees"}
        flags = (interface.get("flags") or {}).get("use_degrees")
        for prefix, key in sides.items():
            pointer = "/" + key.replace(".", "/")
            if key in interface.get("unresolved", []) or given.get(key) == "$":
                arms[prefix] = {"use_degrees": None, "pointer": pointer,
                                "basis": f"--{key} depends on shell expansion and is not resolved"}
                continue
            if key in given:
                value = given[key]
                origin = "set explicitly"
            elif via_config_path:
                arms[prefix] = {"use_degrees": None, "pointer": pointer,
                                "basis": f"--config_path is not read and --{key} is not on the command line"}
                continue
            else:
                value, origin = default, (f"the default of {LEROBOT} ({default_source}); other LeRobot versions may "
                                          "default differently")
            if robot_type in robot_interface.USE_DEGREES or robot_type == "bi_so_follower":
                # robot_interface resolves these flags (and their defaults), also from nested mappings such as
                # --robot.left_arm_config='{port: ..., use_degrees: false}'; its boolean is the value used.
                parsed = flags.get(prefix[:-1]) if isinstance(flags, dict) else flags
                if isinstance(parsed, bool):
                    if key not in given and parsed != default:
                        origin = "set in the command"
                    value = parsed
            if not isinstance(value, bool):
                arms[prefix] = {"use_degrees": None, "pointer": pointer, "basis": f"--{key} {value!r} is not a boolean"}
                continue
            arms[prefix] = {"use_degrees": value, "pointer": pointer, "origin": origin,
                            "mode": DEGREES if value else RANGE_M100_100,
                            "basis": f"use_degrees={value} ({origin}) makes the body joints {DEGREES if value else RANGE_M100_100} "
                                     f"and the gripper RANGE_0_100 ({LEROBOT} {MODE_SOURCE[robot_type]})"}
    return {"robot_type": robot_type, "program": interface.get("program"), "source": source, "arms": arms,
            "joint_names": list(names), "fps": interface.get("fps"), "unknown": unknown,
            "notes": list(interface.get("unknown", []))}


# --------------------------------------------------------------------------------------------------------- manifest

def units_manifest(dataset: str | Path, *, robot_command: str | None = None, robot_config: str | Path | None = None,
                   max_rows: int = 2_000_000, max_files: int = 1000) -> dict:
    """Per-feature, per-coordinate units of ``dataset`` and, given the robot command or config, their comparison.

    Returns ``{schema_version, kind, status, counts, coverage, manifest, findings, scope}``; ``status`` is the most
    severe finding (error > warning > unknown > pass).
    """
    if type(max_rows) is not int or not 1 <= max_rows <= 20_000_000:
        raise InputError("max_rows must be an integer in [1, 20000000]")
    if type(max_files) is not int or not 1 <= max_files <= 10_000:
        raise InputError("max_files must be an integer in [1, 10000]")
    build = _Build(dataset_root(dataset), _deployment(robot_command, robot_config), max_rows, max_files)
    build.features()
    build.pre_pr777()
    build.within_dataset()
    if build.deployment is not None:
        build.compare()
    build.rotations()
    build.timebase()
    return build.result()


def _load(root: Path) -> tuple[Any, str | None]:
    """The dataset snapshot. A meta/stats.json that the strict JSON reader rejects (LeRobot writes NaN into it when the
    data holds NaN) only loses its use: the snapshot is rebuilt from meta/info.json as load_dataset builds it."""
    try:
        return load_dataset(root), None
    except InputError:
        stats = contained(root, "meta/stats.json")
        if not stats.exists():
            raise
        try:
            read_json(stats)
        except InputError as exc:
            note = f"meta/stats.json is not read: {exc}"
        else:
            raise
    info = contained(root, "meta/info.json")
    config = read_json(info)  # load_dataset's info.json checks, which still raise
    fps = config.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise InputError("Dataset fps must be a positive number")
    snapshot = artifacts.Snapshot("dataset", root, config, artifacts._features(config.get("features"), dataset_names=True))
    snapshot.record(info, "dataset_info", config)
    return snapshot, note


class _Build:
    """Shared context of one manifest: the dataset, its values, the deployment and the findings."""

    def __init__(self, root: Path, deployment: dict | None, max_rows: int, max_files: int):
        self.root, self.deployment = root, deployment
        self.snapshot, stats_note = _load(root)
        self.info_path = contained(root, "meta/info.json")
        self.info_sha = next(a.sha256 for a in self.snapshot.artifacts if a.role == "dataset_info")
        self.vectors = _vector_features(self.snapshot)
        self.motor_map = {(key, j): parsed for key, spec in self.vectors.items()
                          for j, name in enumerate(spec["names"] or []) if (parsed := _motor(key, name))}
        self.quats = {key: q for key, spec in self.vectors.items() if (q := _quaternions(key, spec["names"]))}
        self.values, self.coverage = _read(root, self.vectors, self.motor_map, self.quats, max_rows, max_files)
        if stats_note:
            self.coverage["stats_not_used"] = stats_note
        files = self.coverage.get("files", [])
        self.data_digest = hashlib.sha256("".join(f["sha256"] for f in files).encode()).hexdigest() if files else None
        self.report = _Report()
        self.arm_results: dict[tuple[str, str], dict] = {}
        self.disagree: set[str] = set()  # arms whose state and action decide different modes (within_dataset)
        self.undecided: list[str] = []
        self.manifest: dict[str, Any] = {
            "dataset": {"root": str(root), "robot_type": self.snapshot.config.get("robot_type"),
                        "fps": self.snapshot.config.get("fps"),
                        "codebase_version": self.snapshot.config.get("codebase_version")},
            "features": {}, "rotations": [], "timebase": None, "deployment": deployment}

    # Evidence -------------------------------------------------------------------------------------------------

    def info_ev(self, key: str, index: int | None = None, value: Any = None) -> dict:
        pointer = f"/features/{_escape(key)}/names" + (f"/{index}" if index is not None else "")
        return _ev(self.info_path, self.info_sha, pointer, value)

    def data_ev(self, key: str, arm: str) -> dict:
        body = [c for c in self.manifest["features"][key]["coordinates"]
                if c.get("arm") == arm and c.get("motor") not in (None, "gripper")]
        joints = {c["motor"]: {"min": c.get("min"), "max": c.get("max"), **c.get("evidence", {})} for c in body}
        if any(c.get("from_stats") for c in body):  # values not read: the ranges come from meta/stats.json
            stats = next((a for a in self.snapshot.artifacts if a.role == "dataset_stats"), None)
            return _ev(stats.path if stats else None, stats.sha256 if stats else None,
                       f"/{_escape(key)}/min, /{_escape(key)}/max", joints)
        return _ev(self.root / "data", self.data_digest, f"{key}[:, {arm}*] ({', '.join(sorted(joints))})", joints)

    def dep_ev(self, pointer: str, value: Any) -> dict:
        source = self.deployment["source"]
        return _ev(source["path"] or "robot_command", source["sha256"], pointer, value)

    # Per-coordinate units -------------------------------------------------------------------------------------

    def features(self) -> None:
        stats = self.snapshot.config.get("stats") if isinstance(self.snapshot.config.get("stats"), dict) else {}
        for key, spec in self.vectors.items():
            columns = self.values["columns"][key] if self.values else None
            coordinates, verdicts = [], {}
            for j in range(spec["width"]):
                column = columns[j] if columns else None
                coordinate = self._coordinate(key, j, spec, column, stats.get(key))
                if coordinate.get("motor") not in (None, "gripper"):
                    verdicts.setdefault(coordinate["arm"], {})[coordinate["motor"]] = coordinate.pop("_verdict")
                coordinates.append(coordinate)
            entry = {"names": spec["names"], "coordinates": coordinates, "arms": {}}
            for arm, by_motor in verdicts.items():
                result = _arm(by_motor)
                if result.get("off_grid") and result["mode"] == RANGE_M100_100 and self.values:
                    index = {c["motor"]: c["index"] for c in coordinates if c.get("arm") == arm and c.get("motor")}
                    sessions = _episode_ranges(self.coverage.get("files", []), key, spec["width"],
                                               {m: index[m] for m in result["off_grid"]})
                    result["episodes_checked"] = sessions is not None
                    result["recalibrated"] = sessions or {}
                    result["off_grid"] = [m for m in result["off_grid"] if m not in result["recalibrated"]]
                entry["arms"][arm or "(single arm)"] = result
                self.arm_results[(key, arm)] = result
                for coordinate in coordinates:
                    if coordinate.get("arm") == arm and coordinate.get("motor") not in (None, "gripper"):
                        _apply_arm(coordinate, by_motor[coordinate["motor"]], result)
            for coordinate in coordinates:
                if coordinate.get("status") in ("decided", "decided_by_arm"):
                    coordinate["unit_text"] = UNIT_TEXT.get(coordinate["unit"])
                else:
                    self.undecided.append(f"{key}[{coordinate['index']}]"
                                          + (f" {coordinate['name']}" if coordinate["name"] else ""))
            self.manifest["features"][key] = entry
        total = sum(spec["width"] for spec in self.vectors.values())
        self.coverage.update(coordinates=total, decided=total - len(self.undecided), undecided=len(self.undecided))

    def _coordinate(self, key: str, j: int, spec: dict, column: _Column | None, stats: Any) -> dict:
        name = spec["names"][j] if spec["names"] else None
        coordinate: dict[str, Any] = {"index": j, "name": name}
        if column is not None and column.count:
            coordinate.update(min=column.lo, max=column.hi, count=column.count)
            if column.nonfinite:
                coordinate["nonfinite"] = column.nonfinite
        elif column is None and isinstance(stats, dict):
            low, high = stats.get("min"), stats.get("max")
            if isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == spec["width"]:
                coordinate.update(min=low[j], max=high[j], from_stats=True)
        parsed = self.motor_map.get((key, j))
        if not parsed:
            coordinate.update(unit=None, status="unknown",
                              basis=f"the unit is not recorded ({LEROBOT} {NO_UNITS_RECORDED}) and these values are "
                                    "not LeRobot motor-bus positions, so it is not decidable")
            return coordinate
        arm, motor, era = parsed
        coordinate.update(motor=motor, arm=arm, era=era)
        gripper = motor == "gripper"
        reading = key == "observation.state"  # always a bus reading; actions may be unclamped policy outputs
        low, high = coordinate.get("min"), coordinate.get("max")
        if column is not None and column.count:
            verdict = _verdict(column, gripper, reading)
        elif (reading and not gripper and isinstance(low, (int, float)) and isinstance(high, (int, float))
              and max(abs(low), abs(high)) > 100 + 1e-3):
            verdict = {"mode": NOT_M100, "facts": {"from_stats": True},
                       "basis": f"meta/stats.json gives a state value beyond ±100, which RANGE_M100_100 readings cannot "
                                f"produce ({LEROBOT} {BOUNDED}); the data files were not read"}
        else:
            verdict = {"mode": None, "facts": {}, "reason": self.coverage.get("reason", "no values read")}
        coordinate["evidence"] = verdict["facts"]
        if gripper:
            coordinate.update(unit=verdict["mode"], status="decided" if verdict["mode"] else "unknown",
                              basis=verdict.get("basis") or verdict.get("reason"))
        else:
            coordinate["_verdict"] = verdict
        return coordinate

    # Dataset-internal findings --------------------------------------------------------------------------------

    def pre_pr777(self) -> None:
        old = sorted({key for (key, arm) in self.arm_results if arm == "main_"})
        if not old:
            return
        status = "error" if self.deployment and self.deployment["arms"] else "warning"
        on_grid = [k for k in old if self.arm_results[(k, "main_")]["mode"] == PRE_PR777]
        self.report.add(
            "units.pre_pr777_calibration", status,
            f"{', '.join(old)} use the main_* joint names of LeRobot's calibration before PR #777"
            + (f", and the values of {', '.join(on_grid)} sit on that calibration's 180/2048-degree grid" if on_grid else "")
            + ": degrees from a zero with the arm extended, shoulder_lift reversed. Current LeRobot reads every joint "
            "differently in both modes, and recordings and policies from that calibration replay incorrectly "
            f"({LEROBOT} {BACKWARDCOMP})." + (" Deploying them on this robot drives the arm to other poses."
                                              if status == "error" else ""),
            [self.info_ev(k, 0, self.manifest["features"][k]["names"][0]) for k in old]
            + ([self.data_ev(k, "main_") for k in old] if self.values else []),
            remedy="Apply the documented conversion (backwardcomp.mdx: main_X -> X.pos, shoulder_lift = -(v - 90), "
                   "elbow_flex -= 90) with --robot.use_degrees=true, or re-record with the current calibration.")

    def within_dataset(self) -> None:
        for arm in sorted({a for (_, a) in self.arm_results}):
            state, action = self.arm_results.get(("observation.state", arm)), self.arm_results.get(("action", arm))
            if (state and action and action["status"] == "decided"
                    and ((state["status"] == "decided" and state["mode"] != action["mode"])
                         or (state["status"] == "partial" and action["mode"] == RANGE_M100_100))):
                self._disagree(arm, state, action)
            for key, result in (("observation.state", state), ("action", action)):
                if result and result["status"] == "conflict":
                    self.report.add(
                        "units.arm_mode_conflict", "warning",
                        f"{key} of the {_side(arm)} arm: {result['reason']}. Episodes recorded in different modes or "
                        "with different encoders may be mixed in this dataset.", [self.data_ev(key, arm)],
                        remedy="Check how the episodes were recorded and merged.")
                if result and (result.get("off_grid") or result.get("recalibrated")):
                    self._off_grid(key, arm, result)

    def _off_grid(self, key: str, arm: str, result: dict) -> None:
        mode, off, parts = result["mode"], result.get("off_grid", []), []
        for motor, sessions in sorted(result.get("recalibrated", {}).items()):
            parts.append(f"{motor} is RANGE_M100_100 under calibrations that change between sessions: "
                         f"{_sessions_text(sessions)}, so one value means different angles in different sessions")
        if off and mode == RANGE_M100_100:
            per = ", nor on one grid per episode" if result.get("episodes_checked") else ""
            parts.append(f"{', '.join(off)} sit on no single RANGE_M100_100 grid, as {', '.join(result['decided_by'])} "
                         f"do{per}; they may be RANGE_M100_100 readings under changing calibrations or processed or "
                         "converted values, which the values do not tell apart, so their unit is not stated")
        elif off:
            parts.append(f"{', '.join(off)} do not sit on the {mode} grid that {', '.join(result['decided_by'])} "
                         f"decide, which does not move with recalibration, so their values are not {mode} readings "
                         "of this arm (processed, converted or requantized values sit off it); their unit is not "
                         "stated")
        self.report.add(
            "units.joint_off_arm_grid", "warning",
            f"{key} of the {_side(arm)} arm: " + "; ".join(parts) + f" ({LEROBOT} {NORMALIZE}).",
            [self.data_ev(key, arm)],
            remedy="Check how these joints were recorded, recalibrated, merged or processed before relying on their "
                   "units.")

    def _disagree(self, arm: str, state: dict, action: dict) -> None:
        """State and action of one arm in different modes. In a teleoperated recording the action is the leader's
        reading, sent unchanged to the follower, which executes it in its own mode: the state's. A warning, not an
        error: a deployment in the state's mode reproduces the recording, and one in another mode is the state's own
        units.joint_mode_mismatch."""
        self.disagree.add(arm)
        follower = state["mode"] if state["status"] == "decided" else "not RANGE_M100_100 (values beyond ±100)"
        flag = _flag(arm, self.deployment)
        if state["mode"] in (DEGREES, RANGE_M100_100):
            remedy = (f"Deploy with --{flag}={str(state['mode'] == DEGREES).lower()} (the follower's recorded mode, "
                      "in which it executed these actions); ")
        elif state["mode"] == NOT_M100:
            remedy = f"Deploy with --{flag}=true, the follower's likely mode (its state rules RANGE_M100_100 out); "
        else:
            remedy = "Deploy in the follower's recorded mode, in which it executed these actions; "
        robot = (self.deployment or {}).get("robot_type") or self.snapshot.config.get("robot_type") or ""
        family = next((f for f in FIXED_LEADERS if f in str(robot)), None)
        fixed = ""
        if family and action["mode"] == RANGE_M100_100:
            fixed = (f" The {family}_leader always reads RANGE_M100_100 and has no use_degrees ({LEROBOT} "
                     f"{FIXED_LEADERS[family]}), so a {family} follower recorded with use_degrees=true always disagrees "
                     "with it.")
            remedy += "for actions that mirror the leader, record with --robot.use_degrees=false."
        else:
            remedy += "for actions that mirror the leader, set the same use_degrees on --robot and --teleop and re-record."
        self.report.add(
            "units.state_action_mode_disagree", "warning",
            f"observation.state of the {_side(arm)} arm was recorded in {follower} but action in {action['mode']}: "
            "the teleoperator and the robot used different modes. LeRobot records the teleoperator's reading as the "
            "action and sends the same numbers to the robot, which executes them in its own mode "
            f"({LEROBOT} {TELEOP_LOOP}; {NORMALIZE}), so the follower did not mirror the leader, and a policy trained "
            f"on these pairs reproduces the recording only on a robot in the follower's mode.{fixed} The deployment is "
            "compared with observation.state; the action grid shows the leader's mode and is not compared.",
            [self.data_ev("observation.state", arm), self.data_ev("action", arm)], remedy=remedy)

    # Deployment comparison ------------------------------------------------------------------------------------

    def compare(self) -> None:
        deployment = self.deployment
        if deployment["unknown"]:
            self.report.add("units.deployment_mode_unknown", "unknown",
                            f"{deployment['unknown']}; the joint units the robot will use cannot be compared.",
                            [self.dep_ev("/robot/type", deployment["robot_type"])])
            return
        if not self.arm_results:
            self.report.add("units.deployment_mode_unknown", "unknown",
                            "the dataset has no LeRobot motor-bus joint names (<motor>.pos), so its joints cannot be "
                            f"matched to this {deployment['robot_type']} robot by name.",
                            [self.info_ev(key) for key in self.vectors if key in MOTOR_FEATURES][:2])
            return
        for (key, arm), result in self.arm_results.items():
            if arm == "main_":
                continue  # reported once as units.pre_pr777_calibration
            label = f"{key} ({_side(arm)} arm)" if arm else key
            where = (f"the {deployment['robot_type']} in this lerobot-{deployment['program']} command"
                     if deployment["source"]["kind"] == "command" else f"the {deployment['robot_type']} in this robot config")
            if key == "action" and ("observation.state", arm) in self.arm_results:
                self._action_with_state(arm, label, where, result)
                continue
            robot_arm = deployment["arms"].get(arm)
            if robot_arm is None:
                self.report.add("units.deployment_mode_unknown", "unknown",
                                f"{label}: {where} has no '{arm}' arm, so these joints are not compared by name.",
                                [self.data_ev(key, arm), self.dep_ev("/robot/type", deployment["robot_type"])])
                continue
            if robot_arm.get("use_degrees") is None:
                self.report.add("units.deployment_mode_unknown", "unknown", f"{label}: {robot_arm['basis']}.",
                                [self.dep_ev(robot_arm["pointer"], None)])
                continue
            self._compare_arm(key, arm, label, where, result, robot_arm)

    def _action_with_state(self, arm: str, label: str, where: str, action: dict) -> None:
        """An arm with observation.state is compared through its state only: in a teleoperated recording the action
        shows the teleoperator's mode and the state the robot's (TELEOP_LOOP). When the state decides nothing and the
        action does, the comparison is unknown."""
        state = self.arm_results[("observation.state", arm)]
        if arm in self.disagree or state["status"] in ("decided", "partial") or action["status"] != "decided":
            return  # compared through the state, reported as a disagreement, or nothing decided either way
        why = state.get("reason") or state.get("basis") or "undecided"
        self.report.add(
            "units.joint_mode_undecided", "unknown",
            f"{label} sits on the {action['mode']} grid ({action['basis']}), but the action shows the teleoperator's "
            "mode: LeRobot records the teleoperator's reading as the action and the robot executes it in its own mode "
            f"({LEROBOT} {TELEOP_LOOP}). observation.state, which shows the robot's mode, decides nothing here "
            f"({why}), so the action is not compared with {where}.",
            [self.data_ev("action", arm), self.data_ev("observation.state", arm)],
            remedy="Replay one recorded episode on the robot in this mode before running a policy.")

    def _compare_arm(self, key: str, arm: str, label: str, where: str, result: dict, robot_arm: dict) -> None:
        robot_mode = robot_arm["mode"]
        evidence = [self.data_ev(key, arm), self.dep_ev(robot_arm["pointer"], robot_arm["use_degrees"])]
        flag = robot_arm["pointer"].lstrip("/").replace("/", ".")
        if result["status"] == "decided" and result["mode"] == PRE_PR777:
            # Current names on the old grid: the documented migration renames keys and shifts zeros by multiples of
            # 90 degrees, which keeps the grid (backwardcomp.mdx). Degrees either way; the zeros cannot be checked.
            if robot_mode == RANGE_M100_100:
                self.report.add(
                    "units.joint_mode_mismatch", "error",
                    f"{label} holds degrees on the 180/2048 grid of LeRobot's calibration before PR #777 "
                    f"({result['basis']}), but {where} reads and commands those joints in RANGE_M100_100: "
                    f"{robot_arm['basis']}. The same numbers mean different angles ({LEROBOT} {UNNORMALIZE}).",
                    evidence, remedy=f"Run with --{flag}=true after the documented offset conversion ({BACKWARDCOMP}), "
                                     "or re-record with the current calibration.")
            else:
                self.report.add(
                    "units.joint_mode_undecided", "unknown",
                    f"{label} holds degrees on the 180/2048 grid of LeRobot's calibration before PR #777 under current "
                    f"names: consistent with the documented migration for use_degrees=true ({LEROBOT} {BACKWARDCOMP}), "
                    "but whether its zero offsets were converted is not decidable from the values.", evidence,
                    remedy="Replay one recorded episode on the robot in this mode before running a policy.")
            return
        off = (f" {', '.join(result['off_grid'])} sit off this grid and are not covered (units.joint_off_arm_grid)."
               if result.get("off_grid") else "")
        if result["status"] == "decided" and result["mode"] != robot_mode:
            effect = _effect(self.manifest["features"][key], arm, result["mode"], robot_mode)
            self.report.add(
                "units.joint_mode_mismatch", "error",
                f"{label} was recorded in {result['mode']} ({result['basis']}), but {where} reads and commands those "
                f"joints in {robot_mode}: {robot_arm['basis']}. LeRobot stores no norm mode with the data ({LEROBOT} "
                f"{NO_UNITS_RECORDED}) and never compares the two, so the same numbers mean different angles "
                f"({LEROBOT} {UNNORMALIZE}).{effect}{off}", evidence,
                remedy=f"Run with --{flag}={str(result['mode'] == DEGREES).lower()} (the mode the data was recorded "
                       "in), or retrain on data recorded in the robot's mode. This assumes the policy was trained on "
                       "this dataset's values as recorded; the policy is not read.")
        elif result["status"] == "partial" and robot_mode == RANGE_M100_100:
            self.report.add(
                "units.joint_mode_mismatch", "error",
                f"{label}: {result['basis']}, but {where} reads and commands those joints in RANGE_M100_100 "
                f"({robot_arm['basis']}): values beyond ±100 cannot be RANGE_M100_100 positions, and commands beyond "
                f"±100 are clamped silently ({LEROBOT} src/lerobot/motors/motors_bus.py:957-960).", evidence,
                remedy=f"Find the mode the data was recorded in (likely --{flag}=true) and deploy with it.")
        elif result["status"] == "decided":
            self.report.add("units.joint_mode_match", "pass",
                            f"{label} was recorded in {result['mode']}, and {where} uses {robot_mode} "
                            f"({robot_arm['basis']}). Calibration offsets between arms are not checked.{off}", evidence)
        else:
            why = result.get("reason") or result.get("basis") or "undecided"
            self.report.add("units.joint_mode_undecided", "unknown",
                            f"{label}: the recorded joint mode is not decidable ({why}); {where} will use {robot_mode} "
                            f"({robot_arm['basis']}).", evidence,
                            remedy="Replay one recorded episode on the robot in this mode before running a policy.")

    # Rotations and timebase -----------------------------------------------------------------------------------

    def rotations(self) -> None:
        for key, found in self.quats.items():
            for q in found:
                record = self.values["norms"][(key, q["order"])] if self.values else None
                item = {"feature": key, "representation": "quaternion", "order": q["order"],
                        "scalar": "last" if q["order"].endswith("w") else "first", "names": q["names"],
                        "basis": "component names only (a unit norm holds in every component order)", "unit_norm": None}
                self.manifest["rotations"].append(item)
                if not (record and record[3]):
                    continue
                unit, rows = record[2], record[3]
                item.update(norm_min=record[0], norm_max=record[1], unit_norm_rows=unit, rows=rows,
                            unit_norm_fraction=unit / rows, unit_norm=unit == rows)
                evidence = [self.info_ev(key, None, self.vectors[key]["names"]),
                            _ev(self.root / "data", self.data_digest, f"{key}[:, {q['indices']}] norm",
                                {"min": record[0], "max": record[1], "rows_within_1e-2": unit, "rows": rows})]
                if unit == rows:
                    self.report.add("units.quaternion_unit", "pass",
                                    f"{key}: components {q['names']} name a quaternion, and every row has unit norm. The "
                                    f"names give {q['order']} order (scalar {item['scalar']}); the values cannot "
                                    "confirm the order, since the norm is the same in every order.", evidence)
                else:
                    self.report.add("units.quaternion_not_unit", "warning",
                                    f"{key}: components {q['names']} name a quaternion, but {rows - unit} of {rows} rows "
                                    f"are off unit norm by more than 1e-2 (norms {record[0]:.3g}..{record[1]:.3g}); "
                                    "those values are not unit quaternions. Component order is taken from the names "
                                    "and is not checked.", evidence,
                                    remedy="Check how the rotation was written (zero or unnormalized rows, or another "
                                           "representation under these names) and renormalize it.")

    def timebase(self) -> None:
        fps = self.snapshot.config.get("fps")
        steps = self.values["steps"] if self.values else np.empty(0)
        if not steps.size:
            self.manifest["timebase"] = {"declared_fps": fps, "status": "unknown",
                                         "basis": self.coverage.get("reason") or "no within-episode timestamp steps were read"}
            return
        median = float(np.median(steps))
        close = int(np.count_nonzero(np.abs(steps - 1 / fps) <= 0.05 / fps))
        within = close / steps.size
        agrees = abs(median - 1 / fps) <= 0.05 / fps
        rate = 1 / median if median > 0 else None
        self.manifest["timebase"] = {"declared_fps": fps, "median_step_s": median, "timestamp_fps": rate,
                                     "fraction_of_steps_within_5pct": within, "steps": int(steps.size),
                                     "status": "decided" if agrees else "contradicts_fps",
                                     "basis": "consecutive timestamps inside each episode of each data file"}
        evidence = [_ev(self.info_path, self.info_sha, "/fps", fps),
                    _ev(self.root / "data", self.data_digest, "timestamp (within-episode steps)",
                        {"median_step_s": median, "fraction_within_5pct": within})]
        if agrees:
            self.report.add("units.timebase_consistent", "pass",
                            f"The recorded timestamps step by a median {median * 1000:.2f} ms, matching the declared "
                            f"{fps} fps ({close} of {steps.size} steps within 5%). Only the median is compared, and "
                            f"LeRobot writes timestamp = frame_index / fps ({LEROBOT} {TIMESTAMPS}), so data it wrote "
                            "passes by construction: this does not show the loop rate the robot achieved.", evidence)
        else:
            self.report.add("units.timebase_contradicts_fps", "warning",
                            f"The recorded timestamps step by a median {median * 1000:.2f} ms "
                            f"({f'{rate:.3g} Hz' if rate else 'no forward step'}), but meta/info.json declares {fps} "
                            "fps; a loop run at the declared rate replays the data at another speed.", evidence,
                            remedy="Find the true recording rate and fix fps or the timestamps.")

    def result(self) -> dict:
        if self.undecided:
            total = self.coverage["coordinates"]
            self.report.add("units.undecided", "unknown",
                            f"{len(self.undecided)} of {total} vector coordinates have no decidable unit: "
                            + ", ".join(self.undecided[:12]) + (" ..." if len(self.undecided) > 12 else "")
                            + f". LeRobot does not record units ({LEROBOT} {NO_UNITS_RECORDED}).",
                            [self.info_ev(key) for key in self.vectors][:4])
        findings = self.report.findings
        counts = {s: sum(f["status"] == s for f in findings) for s in STATUSES}
        status = next((s for s in STATUSES if counts[s]), "unknown")
        return {"schema_version": SCHEMA_VERSION, "kind": KIND, "status": status, "counts": counts,
                "coverage": self.coverage, "manifest": self.manifest, "findings": findings, "scope": SCOPE}


def _sessions_text(sessions: dict) -> str:
    checked = f"{sessions['episodes']} episodes" + (f" (of {sessions['of']})" if sessions["of"] > sessions["episodes"]
                                                    else "")
    return (f"each of {checked} sits on one RANGE_M100_100 grid, of {len(sessions['ranges'])} calibrated ranges "
            f"({', '.join(map(str, sessions['ranges'][:6]))}{', ...' if len(sessions['ranges']) > 6 else ''} ticks), "
            "but no single range holds them all")


def _side(arm: str) -> str:
    return arm.rstrip("_") or "single"


def _flag(arm: str, deployment: dict | None) -> str:
    """The use_degrees flag that sets this arm's mode (the deployment's own pointer when it has this arm)."""
    robot_arm = (deployment or {}).get("arms", {}).get(arm)
    if robot_arm:
        return robot_arm["pointer"].lstrip("/").replace("/", ".")
    return f"robot.{arm}arm_config.use_degrees" if arm in ("left_", "right_") else "robot.use_degrees"


def _apply_arm(coordinate: dict, own: dict, result: dict) -> None:
    """A body joint's unit: its own grid decision, or the arm's when its own values sit on the arm's grid but do not
    decide it alone (too few distinct values, or grids that coincide)."""
    if result["status"] == "decided":
        mode = result["mode"]
        sessions = result.get("recalibrated", {}).get(coordinate["motor"])
        if own["mode"] == mode:
            coordinate.update(unit=mode, status="decided", basis=own.get("basis"))
        elif sessions:
            coordinate.update(unit=mode, status="decided_by_arm",
                              basis=f"{_sessions_text(sessions)}: RANGE_M100_100 under calibrations that change "
                                    f"between sessions; the arm decides it: {result['basis']}")
        elif mode in own.get("fits", ()):
            coordinate.update(unit=mode, status="decided_by_arm",
                              basis=f"{own.get('reason') or own.get('basis')}; every value sits on the {mode} grid, "
                                    f"and the arm decides it: {result['basis']}")
        elif own.get("reason") == "no values read":
            coordinate.update(unit=None, status="unknown", basis=own["reason"])
        else:  # reported as units.joint_off_arm_grid
            coordinate.update(unit=None, status="off_arm_grid",
                              basis=f"{own.get('reason') or own.get('basis')}; the arm decides {mode} "
                                    f"({result['basis']}), but these values do not all sit on the {mode} grid")
    elif result["status"] == "partial":
        coordinate.update(unit=None, status="partial", rules_out=RANGE_M100_100,
                          basis=own.get("basis") or f"{own.get('reason')}; {result['basis']}")
    else:
        coordinate.update(unit=None, status="unknown", basis=own.get("reason") or result.get("reason"))


def _effect(entry: dict, arm: str, data_mode: str, robot_mode: str) -> str:
    """What the mismatch does to commands, on an arm that keeps the recording calibration. An action is compared only
    when its arm has no observation.state (_action_with_state), so its own grid is the only calibration known."""
    joints = [c for c in entry["coordinates"] if c.get("arm") == arm and c.get("motor") not in (None, "gripper")]
    if data_mode == RANGE_M100_100 and robot_mode == DEGREES:
        worst = []
        for c in joints:
            R = c.get("evidence", {}).get("range_m100_100_ticks")
            if R and c.get("min") is not None:
                # intended raw = mid + v * R / 200; a DEGREES robot sends mid + v * 4095 / 360 (motors_bus.py:957-968)
                error = max(abs(v * (DEG_STEPS - R / 200)) for v in (c["min"], c["max"])) * 360 / 4096
                worst.append((error, c["motor"], R))
        if worst:
            worst.sort(reverse=True)
            listed = ", ".join(f"{m} up to {e:.1f}° (range {R} ticks)" for e, m, R in worst[:3])
            return (f" On an arm keeping the recording calibration, commands at the recorded extremes land off "
                    f"target: {listed}; observations are rescaled by the same factors.")
    if data_mode == DEGREES and robot_mode == RANGE_M100_100:
        clamped = [c["motor"] for c in joints if c.get("min") is not None and max(abs(c["min"]), abs(c["max"])) > 100]
        return (" Every command is rescaled by the arm's calibrated range"
                + (f", and {', '.join(clamped)} commands beyond ±100 are clamped (motors_bus.py:957-960)." if clamped else "."))
    return ""
