"""Simulator data provenance guards for Isaac Lab and robomimic HDF5 demonstrations.

Read-only, bounded and offline. Only HDF5 attributes and selected small numeric datasets (h5py) and LeRobot
metadata/parquet columns (pyarrow) are read; no simulator, converter, network or model is run.

Guarded conventions, each cited to pinned upstream source and reproduced natively before it was built:

* Isaac Lab quaternion order is declared only by the root ``format_version`` attribute (missing/<1 = legacy
  WXYZ, 1 = XYZW). Isaac Lab >= 3.0 reorders only datasets named ``root_pose`` when it loads or converts a
  legacy file, and its merge/augmentation tools drop the attribute.
* robomimic ``env_args`` decide the controller the robomimic env factory builds. robosuite <= 1.4 flat configs use
  ``control_delta`` (default true). robosuite >= 1.5 composite configs ignore ``control_delta``; ``input_type``
  (default "delta") and ``input_ref_frame`` (default "base") decide. A flat config fails loudly on robosuite 1.5.1.
* LeRobot feature names that declare quaternion components should hold unit quaternions.

A quaternion order is generally not inferable from values. Content evidence is used per demo and only when a robot
articulation root reads upright under exactly one order; undecidable checks are ``unknown`` or conditional, never
``pass``, and no remedy prescribes reordering data on undecidable evidence. Robots mounted on a ceiling or wall read
upright under the other order, so remedies that rest on the roots state the upright-mount condition.

A source/converted Isaac Lab pair is compared per (dataset, demo): the cyclic reorder that maps each source
quaternion to the converted one, a row and its negation counting as the same rotation, on the rows finite in both
files. Every quaternion-like dataset present in both paired demos is compared or named (changed beyond a reorder or in
shape, or not compared with the reason), and named ones block a pass. Fixes are listed per demo. The pair passes, and
its remedy is decisive, only when the source's robot roots corroborate its declared order (a WXYZ declaration read
upright only as WXYZ in every paired demo; a format_version=1 that no demo contradicts); decisive remedies state the
condition (robot mount or the declaration) under which the other branch applies. Otherwise the remedy is conditional on
the source's recording version.

Remedies that apply to many datasets reference ``report['remedy_lists']``: groups (relation, np.roll shift,
quaternion columns, demos) whose dataset and demo name lists are stored once in ``remedy_lists['names']`` and
referenced by id. ``remedy_list(report, key)`` expands a list; ``remedy_lists_digest(report)`` gives a bounded
summary with the sha256 of the full lists for size-limited views.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import InputError, contained, dataset_root, read_json

SCHEMA_VERSION = 2  # 2: remedy_lists (lists stored once), per-demo pair relations
KIND = 'sim_provenance'
ISAACLAB_FORMAT_VERSION = 1
MAX_HDF5_BYTES = 64 * 1024 ** 3
MAX_DEMOS = 4096
MAX_INDEX_DEMOS = 4096
MAX_DATASET_VALUES = 4_000_000
MAX_TOTAL_VALUES = 40_000_000
MAX_CHUNK_BYTES = 16 * 1024 ** 2
MAX_ATTR_BYTES = 1024 * 1024
MAX_MEMBERS = 4096
MAX_DEPTH = 8
MAX_PARQUET_FILES = 4096
MAX_COLUMN_CHUNK_BYTES = 256 * 1024 ** 2
MAX_BATCH_VALUES = 2_000_000
MAX_PARQUET_VALUES = 32_000_000
MAX_EVIDENCE = 8
BUILTIN_FILTERS = {1, 2, 3, 4, 5, 6, 32000}
UNIT_TOL = 1e-2
NEAR_UNIT_TOL = 0.1
ATOL = 1e-6
ORDER = {True: 'XYZW', False: 'WXYZ'}
ROBOSUITE, ISAACLAB = 'robomimic_robosuite', 'isaaclab'
POSITION_OSC = {'OSC_POSE': 6, 'OSC_POSITION': 3}
HDF5_ERRORS = (OSError, KeyError, RuntimeError, ValueError, TypeError)
_TOO_LARGE = object()

CITATIONS = {
    'isaaclab_version': (
        'IsaacLab v3.0.0 e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4 = main c7d50a61689c9e971b37f90a70fd9d4bc3ea9062 '
        'source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py L19-22, L76-77, L133-150 '
        '(int(attrs["format_version"]); missing -> 0 -> legacy WXYZ)'),
    'isaaclab_root_pose_only': (
        'same file L172-174 and L191-193 (load_episode reorders only "root_pose" with last dim 7), '
        'L298-320 (convert_dataset_to_xyzw reorders only "root_pose")'),
    'isaaclab_convert_attr_copy': 'same file L322-332 (stamps format_version=1, then copies source root attributes)',
    'isaaclab_drop_tools': (
        'IsaacLab main c7d50a6 scripts/tools/merge_hdf5_datasets.py L29-41 (copies data/<demo> and env_args only); '
        'scripts/tools/mp4_to_hdf5.py L164 (f_in.copy("data", f_out))'),
    'isaaclab_xyzw': 'IsaacLab main source/isaaclab/docs/CHANGELOG.rst isaaclab 1.0.0 (2026-01-30): XYZW ordering',
    'robomimic_env_args': (
        'robomimic v0.5.0 ae5799f robomimic/utils/file_utils.py L108-110; utils/env_utils.py L226-233; '
        'envs/env_robosuite.py L116 (robosuite.make(env_name, **env_kwargs)); envs/env_base.py L14-16'),
    'robosuite_delta': (
        'robosuite v1.4.1 c848ca8 robosuite/controllers/osc.py L129, L144, L235-255 (control_delta, default true; '
        'absolute targets in the world frame); controllers/base_controller.py L120 (delta input clipped)'),
    'robosuite15_input_type': (
        'robosuite v1.5.1 (branch head a071383; PyPI 1.5.1 osc.py byte-equal) robosuite/controllers/parts/arm/osc.py '
        'L135-139 (input_type default "delta", input_ref_frame default "base"; control_delta is swallowed by '
        '**kwargs), L154-162, L256-271'),
    'robosuite15_flat_loud': (
        'robosuite v1.5.1 robosuite/controllers/composite/__init__.py L8 (assert type in '
        'REGISTERED_COMPOSITE_CONTROLLERS_DICT: a flat <=1.4 config such as type "OSC_POSE" fails at env creation; '
        'refactor_composite_controller_config is only called from demos/demo_control.py L101)'),
    'robomimic_abs': (
        'robomimic v0.5.0 scripts/train.py L95 (env_meta_update_dict); '
        'scripts/conversion/robosuite_add_absolute_actions.py L53-58, L199-202; diffusion_policy 5ba07ac '
        'scripts/robomimic_dataset_conversion.py L56-64, env_runner/robomimic_lowdim_runner.py L108-109'),
    'lerobot_libero_state': (
        'LeRobot 146d847 src/lerobot/processor/env_processor.py L68-75 (state = eef pos, axis-angle, two finger '
        'qpos) and L94-99 (shape (8,), no names); any4lerobot c5c985e openx2lerobot/openx_rlds.py L102-103 '
        '(POS_QUAT naming x,y,z,rx,ry,rz,rw,gripper)'),
}

SCOPE = (
    'Isaac Lab content checks assume robots start upright on a floor or table. Ceiling or wall mounts can make a wrong '
    'quaternion order look correct, including an overall PASS: confirm the recording version and mount before relying '
    'on a pass or changing data. '
    'Bounded offline read of HDF5 attributes and selected numeric datasets (h5py) and LeRobot metadata/parquet '
    'columns (pyarrow). Conventions: Isaac Lab format_version/quaternion order (v3.0.0/main); robomimic env_args '
    'controller vs actions for single-arm robosuite OSC (<= 1.4 flat control_delta; >= 1.5 composite input_type and '
    'input_ref_frame); LeRobot declared quaternion names. Every demo is inspected by default (max_demos caps it, evenly '
    'spaced) and decided one by one; a check passes only when every demo of the file was inspected, decided and agrees '
    '(partial coverage is unknown); a source/converted pair passes only when the source\'s robot roots corroborate its '
    'declared order. A quaternion order is not inferable '
    'from values in general: content evidence is used only when a robot articulation root reads upright under '
    'exactly one order (robots are assumed upright at reset unless mounted tilted). No simulator, converter, network, '
    'model or robot is run; gripper sign/range and consumer contracts are out of scope.')


def _jsonable(value: Any, limit: int = 32) -> Any:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[:limit].tolist()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, limit) for v in list(value)[:limit]]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, limit) for k, v in list(value.items())[:limit]}
    if isinstance(value, bytes):
        value = value.decode('utf-8', 'replace')
    if isinstance(value, str) and len(value) > 512:
        return value[:512] + '...'
    return value


def _finding(code, status, message, remedy, evidence):
    return {'code': code, 'status': status, 'message': message, 'remedy': remedy,
            'evidence': [dict(e, value=_jsonable(e.get('value'))) for e in evidence[:MAX_EVIDENCE]]}


def _resolve(path, read_roots, label):
    if not isinstance(path, (str, os.PathLike)) or not str(path) or len(str(path)) > 4096:
        raise InputError(f'{label} must be a nonempty path')
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir() or resolved.name == 'info.json':
        resolved = dataset_root(resolved)  # LeRobot alias: authorize the dataset root actually read
    if read_roots is None:
        return resolved
    if (not isinstance(read_roots, (list, tuple)) or not read_roots
            or any(not isinstance(r, (str, os.PathLike)) or not str(r) for r in read_roots)):
        raise InputError('read_roots must be a nonempty list of directory paths')
    for root in read_roots:
        base = Path(root).expanduser().resolve()
        try:
            relative = resolved.relative_to(base)
        except ValueError:
            continue
        return base if str(relative) == '.' else contained(base, str(relative))
    raise InputError(f'{label} is outside the authorized read roots')


def _is_lerobot(path: Path) -> bool:
    return path.is_dir()


def _sample(names, k):
    """Evenly spaced selection (first and last included) so a file's tail is never skipped."""
    if len(names) <= k:
        return list(names)
    index = np.unique(np.linspace(0, len(names) - 1, k).round().astype(int))
    return [names[i] for i in index]


def _coverage_text(inspected, total):
    return f'{inspected} inspected of {total} demos' if inspected != total else f'all {total} demos'


# ----------------------------------------------------------------------------------------------- HDF5


class _HDF5:
    """One bounded, identity-checked HDF5 snapshot (opened by descriptor after a full sha256)."""

    def __init__(self, path: Path, role: str, budget: dict):
        try:
            import h5py
        except ImportError as exc:
            raise InputError('Install policy-deployment-doctor[dataset] (h5py) for HDF5 provenance checks') from exc
        self.h5py, self.path, self.role, self.budget = h5py, path, role, budget
        try:  # O_NONBLOCK: a FIFO must not block before the regular-file check
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise InputError(f'Cannot open {role} HDF5: {exc}') from exc
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_HDF5_BYTES:
            os.close(fd)
            raise InputError(f'{role} must be a regular HDF5 file no larger than {MAX_HDF5_BYTES} bytes')
        self.stream = os.fdopen(fd, 'rb')
        digest, total = hashlib.sha256(), 0
        try:
            for block in iter(lambda: self.stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
                total += len(block)
        except OSError as exc:
            self.stream.close()
            raise InputError(f'Cannot read {role} HDF5: {exc}') from exc
        self.stamp = (info.st_ino, info.st_size, info.st_mtime_ns)
        if total != info.st_size:
            self.stream.close()
            raise InputError(f'{role} HDF5 changed while hashing')
        self.sha256, self.size = digest.hexdigest(), total
        self.stream.seek(0)
        try:
            self.file = h5py.File(self.stream, 'r')
        except HDF5_ERRORS as exc:
            self.stream.close()
            raise InputError(f'{role} is not a readable HDF5 file: {type(exc).__name__}: {exc}') from exc
        self.skipped: list[str] = []

    def close(self):
        try:
            self.file.close()
        except HDF5_ERRORS as exc:
            raise InputError(f'Invalid or corrupt {self.role} HDF5: {type(exc).__name__}: {exc}') from exc
        finally:
            after = os.fstat(self.stream.fileno())
            self.stream.close()
        if (after.st_ino, after.st_size, after.st_mtime_ns) != self.stamp:
            raise InputError(f'{self.role} HDF5 changed while it was being read')

    def ev(self, pointer, value=None):
        return {'path': str(self.path), 'sha256': self.sha256, 'pointer': pointer, 'value': value}

    def member(self, group, name):
        link = group.get(name, getlink=True)
        if link is None:
            return None
        if not isinstance(link, self.h5py.HardLink):
            self.skipped.append(f'{group.name}/{name}: soft/external link not followed')
            return None
        return group[name]

    def unreadable(self, ds):
        """Why a dataset is not read, or None when it can be read."""
        h5py = self.h5py
        if not isinstance(ds, h5py.Dataset) or ds.is_virtual or ds.external:
            return 'virtual or external dataset'
        if ds.dtype.kind not in 'fiu' or ds.dtype.itemsize > 8:
            return 'not a real-number dtype'
        if ds.size > MAX_DATASET_VALUES:
            return f'more than {MAX_DATASET_VALUES} values'
        if ds.chunks is not None and math.prod(ds.chunks) * ds.dtype.itemsize > MAX_CHUNK_BYTES:
            self.skipped.append(f'{ds.name}: chunk of {math.prod(ds.chunks) * ds.dtype.itemsize} bytes exceeds '
                                f'{MAX_CHUNK_BYTES} (not decompressed)')
            return f'chunk larger than {MAX_CHUNK_BYTES} bytes'
        plist = ds.id.get_create_plist()
        try:
            if any(plist.get_filter(i)[0] not in BUILTIN_FILTERS for i in range(plist.get_nfilters())):
                self.skipped.append(f'{ds.name}: non-builtin HDF5 filter (plugins are not loaded)')
                return 'non-builtin HDF5 filter'
        finally:
            plist.close()
        return None

    def readable(self, ds) -> bool:
        return self.unreadable(ds) is None

    def read_any(self, ds):
        """(values as float64 with any non-finite values kept, None), or (None, why it was not read)."""
        if ds is None:
            return None, 'absent'
        why = self.unreadable(ds)
        if why:
            return None, why
        if self.budget['values'] + ds.size > MAX_TOTAL_VALUES:
            self.budget['exhausted'] = True
            return None, 'read budget exhausted'
        self.budget['values'] += ds.size
        try:
            return np.asarray(ds[...], dtype=np.float64), None
        except HDF5_ERRORS as exc:
            self.skipped.append(f'{ds.name}: unreadable ({type(exc).__name__})')
            return None, f'unreadable ({type(exc).__name__})'

    def read(self, ds):
        """Values as float64, or None when unreadable, over budget or holding any non-finite value."""
        value, _ = self.read_any(ds)
        return value if value is not None and np.isfinite(value).all() else None

    def attr(self, obj, name):
        if name not in obj.attrs:
            return None
        try:
            if obj.attrs.get_id(name).get_storage_size() > MAX_ATTR_BYTES:
                return _TOO_LARGE
        except HDF5_ERRORS:
            pass
        value = obj.attrs[name]
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if isinstance(value, str) and len(value) > MAX_ATTR_BYTES:
            return _TOO_LARGE
        return value

    def env_args(self):
        data = self.member(self.file, 'data')
        if data is None or not isinstance(data, self.h5py.Group):
            return None, 'absent', None
        raw = self.attr(data, 'env_args')
        if raw is None:
            return None, 'absent', data
        if not isinstance(raw, str):
            return None, 'invalid', data
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError):
            return None, 'invalid', data
        return (value, 'ok', data) if isinstance(value, dict) else (None, 'invalid', data)

    def demo_names(self, data):
        names = []
        for name in data:
            names.append(name)
            if len(names) > 100_000:
                raise InputError(f'{self.role} has more than 100000 data groups')
        if all(re.fullmatch(r'demo_\d+', n) for n in names):
            names.sort(key=lambda n: int(n.split('_')[1]))
        return names

    def groups(self, data, names):
        result = []
        for name in names:
            group = self.member(data, name)
            if isinstance(group, self.h5py.Group):
                result.append((name, group))
        return result

    def walk(self, group):
        """Dataset descriptors under one demo (bounded depth/count; links and images are never opened)."""
        found, pending, seen = [], [(group, '', 0)], 0
        while pending:
            node, path, depth = pending.pop()
            for name in node:
                seen += 1
                if seen > MAX_MEMBERS:
                    self.skipped.append(f'{group.name}: member budget {MAX_MEMBERS} reached')
                    return found
                child = self.member(node, name)
                rel = f'{path}/{name}' if path else name
                if isinstance(child, self.h5py.Group) and depth < MAX_DEPTH:
                    pending.append((child, rel, depth + 1))
                elif isinstance(child, self.h5py.Dataset):
                    found.append((rel, child))
        return sorted(found, key=lambda item: item[0])

    def anchor(self, group):
        arr = self.read(self.member(group, 'actions'))
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest() if arr is not None else None


def _format_version(h5: _HDF5):
    """Value as Isaac Lab reads it: int(attrs["format_version"]) (handler L141)."""
    raw = h5.attr(h5.file, 'format_version')
    if raw is None:
        return None, 'absent', None
    if raw is _TOO_LARGE:
        return None, 'invalid', 'attribute too large'
    candidate = raw
    if isinstance(candidate, np.ndarray):
        if candidate.size != 1:
            return None, 'invalid', candidate
        candidate = candidate.reshape(-1)[0]
    try:
        return int(candidate), 'ok', raw
    except (TypeError, ValueError, OverflowError):
        return None, 'invalid', raw


def _is_xyzw(version):
    return version is not None and version >= ISAACLAB_FORMAT_VERSION


def _unit_fraction(q):
    norms = np.linalg.norm(q.reshape(-1, 4), axis=1)
    return float(np.mean(np.abs(norms - 1.0) <= UNIT_TOL)) if norms.size else 0.0


def _up(q, order):
    q = np.asarray(q, dtype=np.float64)
    x, y = (q[1], q[2]) if order == 'WXYZ' else (q[0], q[1])
    n = float(np.dot(q, q))
    return 1.0 - 2.0 * (x * x + y * y) / n if n > 0 else float('nan')


_KIND_COLS = {'root_pose': '3:7', 'pose': '3:7', 'quaternion': '0:4'}


def _quaternion_kind(rel, shape):
    """Quaternion-bearing dataset kind by name and shape: 'root_pose' (Isaac Lab matches this name exactly, handler
    L192), 'quaternion' (4 wide) or 'pose' (7 wide: position then quaternion); None otherwise."""
    if not shape or len(shape) > 3:
        return None
    exact = rel.rsplit('/', 1)[-1]
    leaf, last = exact.lower(), shape[-1]
    if exact == 'root_pose' and last == 7:
        return 'root_pose'
    if last == 4 and any(token in leaf for token in ('quat', 'orientation', 'orient')):
        return 'quaternion'
    if last == 7 and 'pose' in leaf:
        return 'pose'
    return None


def _quaternion_rows(rows, kind):
    return rows[:, 3:7] if kind != 'quaternion' else rows


def _demo_inventory(h5: _HDF5, group):
    """root_pose datasets, other unit-quaternion datasets and 7-d pose datasets of one demo. Rows holding non-finite
    values are kept (unit norm is judged on the finite rows); quaternion-like datasets left out are kept in 'excluded'
    as (kind, reason, values or None), and 'shapes' holds the shape of every dataset of the demo."""
    entry = {'root_pose': {}, 'quaternion': {}, 'pose': {}, 'excluded': {}, 'shapes': {}}
    for rel, ds in h5.walk(group):
        shape = ds.shape
        entry['shapes'][rel] = shape
        kind = _quaternion_kind(rel, shape)
        if kind is None:
            continue
        arr, why = h5.read_any(ds)
        if arr is not None:
            rows = arr.reshape(-1, shape[-1])
            finite = np.isfinite(rows).all(axis=1)
            if not finite.any():
                why = 'no finite row'
            elif _unit_fraction(_quaternion_rows(rows[finite], kind)) < 0.99:
                why = 'not unit quaternions'
        if why:
            entry['excluded'][rel] = (kind, why, arr)
        else:
            entry[kind][rel] = arr
    return entry


def _demo_vote(entry):
    """WXYZ / XYZW when every robot articulation root reads upright under exactly that order; else undecidable."""
    votes = []
    for rel, arr in entry['root_pose'].items():
        parts = rel.split('/')
        if (len(parts) != 4 or parts[0] not in ('initial_state', 'states') or parts[1] != 'articulation'
                or 'robot' not in parts[2].lower()):
            continue
        rows = arr.reshape(-1, 7)[:, 3:7]
        q = rows[np.isfinite(arr.reshape(-1, 7)).all(axis=1)][0]  # the first finite row (inventory keeps >= 1)
        up = {o: _up(q, o) for o in ('WXYZ', 'XYZW')}
        vote = next((o for o, other in (('WXYZ', 'XYZW'), ('XYZW', 'WXYZ'))
                     if up[o] >= 0.9 and up[other] <= 0.0), None)
        votes.append((vote, rel, q, up))
    decided = {v for v, *_ in votes if v}
    if not votes:
        return 'no_robot_root', votes
    if len(decided) == 1 and all(v for v, *_ in votes):
        return decided.pop(), votes
    return ('conflicting' if len(decided) > 1 else 'undecidable'), votes


def _isaaclab_layout(h5: _HDF5, demos):
    """Isaac Lab recorder layout. env_args type 2 alone is robomimic's GYM_TYPE and is not sufficient."""
    if h5.attr(h5.file, 'format_version') is not None:
        return True
    for _, group in demos[:4]:
        for name in ('initial_state', 'states'):
            node = h5.member(group, name)
            if isinstance(node, h5.h5py.Group) and any(k in node for k in ('articulation', 'rigid_object')):
                return True
    return False


def _recorder_version(h5: _HDF5):
    for obj in (h5.file, h5.member(h5.file, 'data')):
        if obj is None:
            continue
        raw = h5.attr(obj, 'isaaclab_version')
        if isinstance(raw, str):
            match = re.match(r'\s*v?(\d+)\.', raw)
            if match:
                return int(match.group(1)), raw[:64], obj.name
    return None, None, None


def _isaaclab_single(h5: _HDF5, inventory, total, coverage, complete):
    findings, role = [], h5.role
    version, state, raw = _format_version(h5)
    inspected = len(inventory)
    cover = _coverage_text(inspected, total)
    roots = {n: e['root_pose'] for n, e in inventory.items() if e['root_pose']}
    others = {n: {**e['quaternion'], **e['pose']} for n, e in inventory.items() if e['quaternion'] or e['pose']}
    has_quaternions = bool(roots or others)
    coverage['quaternion_datasets'] = sum(len(v) for v in roots.values()) + sum(len(v) for v in others.values())
    declared = None
    if state == 'absent' or (state == 'ok' and version <= ISAACLAB_FORMAT_VERSION):
        declared = ORDER[_is_xyzw(version)]
    labels = {name: _demo_vote(entry) for name, entry in inventory.items()}
    by_label: dict[str, list[str]] = {}
    for name, (label, _) in labels.items():
        by_label.setdefault(label, []).append(name)
    wxyz, xyzw = by_label.get('WXYZ', []), by_label.get('XYZW', [])
    all_decided = bool(labels) and all(lab in ('WXYZ', 'XYZW') for lab, _ in labels.values())
    # A pass or a data-changing remedy needs every demo of the file inspected (no sampling, no exhausted budget).
    full = all_decided and complete
    partial_text = ("" if complete else f"; only {inspected} of {total} demos were inspected (max_demos or the read budget), "
                    "so the others are not verified")
    mixed = bool(wxyz and xyzw) or bool(by_label.get('conflicting'))
    inferred = None if mixed else ('WXYZ' if wxyz else 'XYZW' if xyzw else None)
    coverage['content_order_per_demo'] = {k: len(v) for k, v in by_label.items()}
    coverage['content_order_inference'] = ('mixed' if mixed else (inferred or 'undecidable')
                                           + ('' if all_decided or not inferred else ' (partial)'))
    coverage['content_order_verified'] = inferred if full and not mixed else None  # every demo inspected and agrees
    attr_ev = h5.ev('/@format_version', version if state == 'ok' else ('absent' if state == 'absent' else raw))
    if state == 'ok' and (isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer))):
        attr_ev['value'] = {'stored': raw, 'isaac_lab_reads': version}  # loader int()s str/bytes/1-element/float

    def vote_ev(names):
        rows = []
        for name in names[:3]:
            for vote, rel, q, up in labels[name][1][:1]:
                rows.append(h5.ev(f'/data/{name}/{rel}[0, 3:7]', {'quaternion': q, 'up_z_if_WXYZ': up['WXYZ'],
                                                                   'up_z_if_XYZW': up['XYZW'], 'vote': vote}))
        return rows

    decided_text = (f'robot roots decide the order in {len(wxyz) + len(xyzw)} of {inspected} inspected demos '
                    f'(of {total})')
    if mixed:
        # This gate precedes every attribute-, recorder-version- and non-root-dataset remedy below. Even a stored
        # isaaclab_version cannot establish one recording order for conflicting demos; a whole-file stamp breaks
        # one side of a mixed merge. Retain loud invalid/new-format diagnostics without prescribing a replacement.
        remedy = ('Change no data or format_version in this file on these votes. Confirm each demo\'s recording '
                  'version and robot mount, then re-merge from the per-version originals (one format_version per '
                  'file) and rerun. No whole-file reorder or re-stamp is prescribed.')
        if state == 'invalid':
            findings.append(_finding('isaaclab.format_version', 'error',
                f'[{role}] Root format_version {raw!r} is not an integer that Isaac Lab can read '
                f'({CITATIONS["isaaclab_version"]}); conflicting robot-root votes also prevent a file-wide replacement.',
                remedy, [attr_ev]))
        elif state == 'ok' and version > ISAACLAB_FORMAT_VERSION:
            findings.append(_finding('isaaclab.format_version', 'unknown',
                f'[{role}] format_version={version} is newer than reviewed; the file also has conflicting '
                'robot-root votes, so its convention is not assessed.', remedy, [attr_ev]))
        findings.append(_finding('isaaclab.mixed_order_across_demos', 'warning',
            f'[{role}] Demos disagree on quaternion order: robot roots read upright only as WXYZ in {len(wxyz)} '
            f'demos ({", ".join(wxyz[:4])}) and only as XYZW in {len(xyzw)} ({", ".join(xyzw[:4])}), of {inspected} '
            f'inspected (of {total}). Root-based inference assumes upright floor/table mounts. A mixed Isaac Lab '
            '2.x / >= 3.0 merge or different tilted mounts can produce conflicting votes; a recorder-version '
            'attribute does not establish one order for this file '
            f'({CITATIONS["isaaclab_drop_tools"]}).', remedy, [attr_ev, *vote_ev(wxyz), *vote_ev(xyzw)]))
        return findings
    if state == 'invalid':
        findings.append(_finding('isaaclab.format_version', 'error',
            f'[{role}] Root format_version {raw!r} is not an integer that Isaac Lab can read: its loader calls '
            f'int(attrs["format_version"]) and raises ({CITATIONS["isaaclab_version"]}).',
            'Store an integer: 1 if the file was recorded with Isaac Lab >= 3.0 (XYZW); delete the attribute for an '
            'Isaac Lab 2.x (WXYZ) recording. Change no data.', [attr_ev]))
    elif state == 'ok' and version > ISAACLAB_FORMAT_VERSION:
        findings.append(_finding('isaaclab.format_version', 'unknown',
            f'[{role}] format_version={version} is newer than the reviewed Isaac Lab version 1; its quaternion '
            'convention is not assessed.', 'Re-review the pinned Isaac Lab dataset handler for this version.', [attr_ev]))
    elif state == 'ok' and version < ISAACLAB_FORMAT_VERSION and has_quaternions:
        confirmed = inferred == 'WXYZ' and full
        findings.append(_finding('isaaclab.format_version', 'warning',
            f'[{role}] format_version={version} is stored explicitly (read as legacy WXYZ). Isaac Lab '
            f'convert_dataset_to_xyzw() stamps 1 and then copies this value over it '
            f'({CITATIONS["isaaclab_convert_attr_copy"]}), so its output keeps {version} and load_episode() reorders the '
            'already-converted root poses again (reproduced natively with 0: 180 deg). '
            + (f'The robot roots of {cover} read upright only as WXYZ, so this file has not been converted (for '
               'upright-mounted robots).' if confirmed
               else f'The values cannot tell whether this file was already converted ({decided_text}{partial_text}).'),
            ('Delete the attribute (absent already means legacy WXYZ) and change no data; stamp 1 only as the last step '
             'of a whole-file conversion (see isaaclab.quaternions_outside_root_pose).' if confirmed else
             'Establish whether root_pose was already converted to XYZW (for example by convert_dataset_to_xyzw, which '
             'keeps this value). If it was, stamp format_version=1'
             + (' (that tool leaves the other quaternion datasets WXYZ: see isaaclab.quaternions_outside_root_pose)'
                if others else '')
             + '; if the file is an unconverted 2.x recording, delete the attribute. Change no root_pose data.'),
            [attr_ev, *vote_ev(wxyz)]))
    elif state == 'absent' and has_quaternions:
        if inferred == 'WXYZ' and full:
            findings.append(_finding('isaaclab.format_version', 'pass',
                f'[{role}] For robots known to start upright on a floor or table only: no format_version, so Isaac Lab '
                f'>= 3.0 reads this file as legacy WXYZ; the robot roots of '
                f'{cover} read upright only as WXYZ, consistent with that reading.', 'None.',
                [attr_ev, *vote_ev(wxyz)]))
        elif inferred == 'WXYZ' and all_decided:
            findings.append(_finding('isaaclab.format_version', 'unknown',
                f'[{role}] No format_version (read as legacy WXYZ); the robot roots of the inspected demos read upright '
                f'only as WXYZ{partial_text}.', 'Inspect every demo (default max_demos) to verify the whole file.',
                [attr_ev, *vote_ev(wxyz)]))
        elif inferred != 'XYZW' and not mixed:
            findings.append(_finding('isaaclab.format_version', 'warning',
                f'[{role}] No root format_version: Isaac Lab >= 3.0 treats this file as legacy WXYZ and reorders '
                f'every root_pose on load ({CITATIONS["isaaclab_root_pose_only"]}). That is only correct for an '
                'Isaac Lab 2.x recording; merge_hdf5_datasets.py and mp4_to_hdf5.py drop the attribute from XYZW '
                f'files ({CITATIONS["isaaclab_drop_tools"]}), which then load rotated (reproduced natively on the public '
                f'Arena G1 recording: 180 deg for identity roots). The values cannot decide: {decided_text}.',
                'Establish the Isaac Lab version that recorded the file. If >= 3.0 (XYZW): stamp format_version=1 '
                "(f.attrs['format_version'] = 1) and change no data. If 2.x (WXYZ): leave it absent and change no "
                'data. Never store 0.', [attr_ev]))
    elif state == 'ok' and not mixed and inferred in (None, declared):
        verified = inferred == declared and full
        findings.append(_finding('isaaclab.format_version', 'pass' if verified else 'unknown',
            f'[{role}] For robots known to start upright on a floor or table only: format_version={version} '
            f'declares {declared}. '
            + (f'The robot roots of {cover} agree.' if verified else
               f'The values confirm it only partly or not at all ({decided_text}{partial_text}); nothing contradicts it, '
               'but the order of the undecided or uninspected demos is not verified (identity or yaw-only roots read '
               'upright under both orders).'),
            'None.' if verified else 'None needed if the recording version matches the declaration.',
            [attr_ev, *vote_ev(wxyz or xyzw)]))
    if inferred and declared and inferred != declared:
        tilt = ('robots mounted tilted (ceiling, wall) read this way in a correct file')
        if declared == 'XYZW':
            rest = '' if len(wxyz) == inspected else ' (the rest are undecidable)'
            message = (f'[{role}] format_version={version} declares XYZW, but the robot roots in {len(wxyz)} of '
                       f'{inspected} inspected demos read upright only as WXYZ{rest}. Either the '
                       'file holds WXYZ data under a version-1 stamp (for example a 2.x recording stamped by a copy '
                       f'tool), or {tilt}. Isaac Lab loads a version-1 file verbatim.')
            remedy = ('Change nothing on this evidence alone. Check the robot mount and the Isaac Lab version that '
                      'recorded the file before touching format_version.')
        else:
            message = (f'[{role}] The file has no format_version (read as legacy WXYZ), but the robot roots in '
                       f'{len(xyzw)} of {inspected} inspected demos read upright only as XYZW. Either it is an Isaac Lab '
                       '>= 3.0 recording that lost format_version (merge_hdf5_datasets.py and mp4_to_hdf5.py drop it), '
                       f'which Isaac Lab >= 3.0 would load rotated ({CITATIONS["isaaclab_root_pose_only"]}), or {tilt}.')
            remedy = ('If the recording is Isaac Lab >= 3.0, stamp format_version=1 and change no data; if it is a 2.x '
                      'recording with a tilted robot, change nothing.')
        findings.append(_finding('isaaclab.declared_order_vs_content', 'warning', message, remedy,
                                 [attr_ev, *vote_ev(wxyz if declared == 'XYZW' else xyzw)]))
    major, recorded, where = _recorder_version(h5)
    recorder_ev = [h5.ev(f'{where}@isaaclab_version', recorded)] if recorded is not None else []
    if major is not None and major >= 3 and state == 'absent' and has_quaternions and inferred != 'XYZW':
        findings.append(_finding('isaaclab.declared_order_vs_content', 'warning',
            f'[{role}] Recorded by Isaac Lab {recorded} (XYZW) but has no format_version, so Isaac Lab >= 3.0 will '
            f'read it as legacy WXYZ and reorder its root poses on load ({CITATIONS["isaaclab_root_pose_only"]}).',
            "Stamp format_version=1 (f.attrs['format_version'] = 1) and change no data.", [attr_ev, *recorder_ev]))
    sample = [(d, r, a) for d, items in others.items() for r, a in items.items()]
    paths = sorted({r for _, r, _ in sample})
    sample_ev = [h5.ev(f'/data/{d}/{r}[0]', a.reshape(-1, a.shape[-1])[0]) for d, r, a in sample[:4]]
    key = f"remedy_lists['{role}.non_root_quaternion_datasets']"
    if others:  # the complete list, stored once in report['remedy_lists'] (messages are bounded)
        cols = {}
        for items in others.values():
            for rel, arr in items.items():
                cols[rel] = '3:7' if arr.shape[-1] == 7 else '0:4'
        coverage['non_root_quaternion_dataset_count'] = len(paths)
        coverage.setdefault('_lists', {})['non_root_quaternion_datasets'] = [
            {'quaternion_columns': c, 'datasets': [rel for rel in paths if cols[rel] == c]}
            for c in ('0:4', '3:7') if any(cols[rel] == c for rel in paths)]
    named = f'{", ".join(paths[:8])}' + (f' and {len(paths) - 8} more' if len(paths) > 8 else '') + f' (all {len(paths)} in {key})'
    # Only content decides the current order: a 2.x recorder version says how a file was recorded, not whether its
    # root_pose was converted since (convert_dataset_to_xyzw keeps an explicit format_version below 1).
    legacy_known = inferred == 'WXYZ' and full
    if declared == 'WXYZ' and others and legacy_known:
        findings.append(_finding('isaaclab.quaternions_outside_root_pose', 'warning',
            f'[{role}] Legacy WXYZ file (robot roots of {cover} read upright only as WXYZ) with {len(paths)} quaternion '
            f'dataset(s) not named root_pose: {named}. Isaac Lab >= 3.0 load_episode() and '
            f'convert_dataset_to_xyzw() reorder only root_pose ({CITATIONS["isaaclab_root_pose_only"]}), so these stay '
            f'WXYZ while Isaac Lab >= 3.0 environments emit XYZW ({CITATIONS["isaaclab_xyzw"]}). Reproduced natively on '
            'the public RoboLab recording.',
            'Only if the robots are known to start upright on a floor or table and this is an Isaac Lab 2.x recording: '
            'convert the whole file consistently before using it with Isaac Lab >= 3.0. In every demo, reorder to '
            'XYZW (np.roll(q, -1, axis=-1)) the group\'s '
            f'quaternion_columns of every dataset in each group of {key} (0:4 of a quaternion; 3:7 of a 7-wide pose, '
            "never its position columns 0:3; names in remedy_lists['names'][id]) and columns 3:7 of every root_pose; "
            'then stamp format_version=1 as the last step. Isaac Lab convert_dataset_to_xyzw() does only the root_pose '
            'part. The roots decide WXYZ only for upright mounts: robots mounted on a ceiling or wall read the same way '
            'in an Isaac Lab >= 3.0 (XYZW) recording, so for such mounts first establish the recording version and '
            'change no data before.', [attr_ev, *recorder_ev, *sample_ev]))
    elif declared == 'WXYZ' and others and state == 'ok':  # explicit format_version below 1; roots do not confirm WXYZ
        roots_text = 'The robot roots read XYZW, as convert_dataset_to_xyzw() output does. ' if inferred == 'XYZW' else ''
        findings.append(_finding('isaaclab.quaternions_outside_root_pose', 'warning',
            f'[{role}] Conditional: {len(paths)} quaternion dataset(s) are not named root_pose ({named}). {roots_text}'
            f'Under an explicit format_version={version} they are still WXYZ if the file is an Isaac Lab 2.x recording, '
            'whether or not convert_dataset_to_xyzw() converted it: that tool keeps the explicit value '
            f'({CITATIONS["isaaclab_convert_attr_copy"]}) and reorders only root_pose '
            f'({CITATIONS["isaaclab_root_pose_only"]}). They are already XYZW only if the file is an Isaac Lab >= 3.0 '
            f'recording stored with {version}. The values cannot decide ({decided_text}{partial_text}).',
            'Change no data on these values alone. If this file came out of convert_dataset_to_xyzw(), pass its input '
            'as source (with this file as converted): the pair check lists the datasets still to reorder. Otherwise '
            'first establish the recording version and follow isaaclab.format_version.', [attr_ev, *sample_ev]))
    elif declared == 'WXYZ' and others:
        findings.append(_finding('isaaclab.quaternions_outside_root_pose', 'warning',
            f'[{role}] Conditional: {len(paths)} quaternion dataset(s) are not named root_pose '
            f"({named}). If this file is an Isaac Lab 2.x (WXYZ) recording, Isaac Lab >= 3.0 "
            f'load_episode()/convert_dataset_to_xyzw() leave them WXYZ ({CITATIONS["isaaclab_root_pose_only"]}) while '
            'Isaac Lab >= 3.0 environments emit XYZW. If it is an Isaac Lab >= 3.0 recording that lost format_version, '
            f'they are already XYZW. The values cannot decide ({decided_text}{partial_text}).',
            'Change no data on these values: first establish the recording version and follow isaaclab.format_version.',
            [attr_ev, *sample_ev]))
    elif declared == 'XYZW' and others and major is not None and major < 3:
        findings.append(_finding('isaaclab.quaternions_outside_root_pose', 'warning',
            f'[{role}] Conditional: recorded by Isaac Lab {recorded} (WXYZ) but declares format_version={version} '
            f'(XYZW), so it was converted afterwards. Isaac Lab convert_dataset_to_xyzw() reorders only root_pose '
            f'({CITATIONS["isaaclab_root_pose_only"]}); if that tool was used, these {len(paths)} dataset(s) are still '
            f'WXYZ: {named} (reproduced on the public RoboLab recording).',
            'Pass the pre-conversion file as source (with this file as converted) to decide; change no data before.',
            [attr_ev, *recorder_ev, *sample_ev]))
    return findings


_REORDER = {0: 'identity', 1: 'WXYZ->XYZW', 2: 'reordered twice', 3: 'XYZW->WXYZ'}
_ROLL_FOR = {1: 1, 2: 2, 3: -1}  # (current k - target k) mod 4 -> the np.roll shift that brings the data to the target


def _reorders(src, dst, pose):
    """The cyclic reorders k with dst == np.roll(src, -k, axis=-1) in every row, a row and its negation counting as the
    same rotation (q and -q); 'shape', or 'other' when positions or rotations changed beyond a reorder."""
    if src.shape != dst.shape:
        return 'shape'
    if pose:
        if not np.allclose(src[..., :3], dst[..., :3], atol=ATOL):
            return 'other'
        src, dst = src[..., 3:7], dst[..., 3:7]
    s = np.asarray(src, dtype=np.float64).reshape(-1, 4)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 4)
    found = set()
    for k in range(4):
        r = np.roll(s, -k, axis=-1)
        if (np.isclose(r, d, atol=ATOL).all(axis=1) | np.isclose(r, -d, atol=ATOL).all(axis=1)).all():
            found.add(k)
    return found or 'other'


def _relation_name(ks):
    return _REORDER[min(ks)] if len(ks) == 1 else 'ambiguous'


def _natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def _fix_groups(rows, target, all_demos):
    """Per-demo fixes that bring every compared dataset to relation `target`, grouped as (relation, roll, columns,
    demos) -> datasets. A dataset whose relation differs between demos lands in several groups, each with its demos."""
    per = {}
    for _, dname, rel, cols, ks, *_ in rows:
        if target in ks:
            continue
        roll = _ROLL_FOR[(min(ks) - target) % 4]  # any k in ks gives the same rotation (q and -q equal)
        per.setdefault((rel, cols, _relation_name(ks), roll), set()).add(dname)
    groups = {}
    for (rel, cols, relation, roll), demos in per.items():
        key = 'all' if demos == all_demos else tuple(sorted(demos, key=_natural))
        groups.setdefault((relation, roll, cols, key), []).append(rel)
    return [{'relation': relation, 'roll': roll, 'quaternion_columns': cols,
             'demos': 'all' if key == 'all' else list(key), 'datasets': sorted(rels)}
            for (relation, roll, cols, key), rels in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2],
                                                                                             str(kv[0][3])))]


def _apply_text(key, groups, final):
    if not groups:
        return f'{final} and change no data'
    return (f"in the converted file apply every group of remedy_lists['{key}']: in each of the group's demos ('all' = "
            "every demo), np.roll(q, roll, axis=-1) on the group's quaternion_columns of each of its datasets (0:4 of "
            "a quaternion, 3:7 of a 7-wide pose, never its position columns 0:3); dataset and demo lists are "
            "remedy_lists['names'][id] (policy_doctor.sim_guards.remedy_list(report, key) expands them); then "
            + (f'{final} as the last step' if final.startswith('stamp') else final))


def _compare(src, dst, pose):
    """_reorders() on the rows finite in both files: 'other' when the files differ in which rows are finite (a reorder
    keeps them), 'none' when no row is finite in both."""
    s, d = src.reshape(-1, src.shape[-1]), dst.reshape(-1, dst.shape[-1])
    finite_s, finite_d = np.isfinite(s).all(axis=1), np.isfinite(d).all(axis=1)
    if not np.array_equal(finite_s, finite_d):
        return 'other'
    if not finite_s.any():
        return 'none'
    return _reorders(s[finite_s], d[finite_d], pose)


def _candidate(entry, rel):
    """(kind, reason it was left out or None, values or None) of a quaternion-like dataset of one demo inventory."""
    for kind in ('root_pose', 'quaternion', 'pose'):
        if rel in entry[kind]:
            return kind, None, entry[kind][rel]
    return entry['excluded'].get(rel)


def _instance_groups(items, field):
    """(source demo, converted demo, dataset, columns, ..., label) instances grouped as (label, columns, demos) ->
    datasets; the label is stored under `field`."""
    per = {}
    for item in items:
        per.setdefault((item[2], item[3], item[-1]), set()).add(item[1])
    by = {}
    for (rel, cols, label), demos in per.items():
        by.setdefault((label, cols, tuple(sorted(demos, key=_natural))), []).append(rel)
    return [{field: label, 'quaternion_columns': cols, 'demos': list(demos), 'datasets': sorted(rels)}
            for (label, cols, demos), rels in sorted(by.items())]


def _listed(names, k=4):
    names = list(names)
    return ', '.join(names[:k]) + (f' and {len(names) - k} more' if len(names) > k else '')


def _first_row(arr):
    return None if arr is None or not arr.size else arr.reshape(-1, arr.shape[-1])[0]


def _conflicting_votes(*counts):
    return (any(c.get('conflicting', 0) for c in counts)
            or (any(c.get('WXYZ', 0) for c in counts) and any(c.get('XYZW', 0) for c in counts)))


def _mixed_pair_findings(coverage, evidence):
    coverage['source_declaration'] = 'conflicting robot-root votes; no whole-file remedy'
    return [_finding('isaaclab.conversion_quaternion_order', 'warning',
        'Source demos give conflicting robot-root order votes. A mixed Isaac Lab 2.x / >= 3.0 merge can cause '
        'this, and tilted mounts can also invalidate root-based inference. Neither one recording version nor a '
        'whole-file conversion branch is established; the conversion is not verified.',
        'Change no data or format_version on either file on this evidence. Confirm each demo\'s recording version '
        'and robot mount, then re-merge from the per-version originals (one format_version per file) and rerun. '
        'No whole-file reorder or source re-stamp is prescribed.', evidence)]


def _source_declaration(src_order, votes, labels):
    """Whether the source's robot roots corroborate its declared order, from the root votes of the paired source demos
    (votes) and of the source demos the single-file check inspected (labels): 'A' (an Isaac Lab 2.x WXYZ recording) for
    a WXYZ declaration whose roots read upright only as WXYZ in every paired demo and only as XYZW in no inspected demo;
    'B' (Isaac Lab >= 3.0, XYZW) for a format_version=1 declaration that no demo reads upright only as WXYZ; else None.
    The basis is 'roots' when every paired demo confirms the declared order, 'declaration' otherwise."""
    m = sum(votes.values())
    conflicting = votes.get('conflicting', 0) + labels.get('conflicting', 0)
    if src_order == 'WXYZ':
        corroborated = m and votes.get('WXYZ', 0) == m and not labels.get('XYZW') and not conflicting
        return ('A', 'roots') if corroborated else (None, None)
    if votes.get('WXYZ', 0) or labels.get('WXYZ') or conflicting:
        return None, None
    return 'B', ('roots' if m and votes.get('XYZW', 0) == m else 'declaration')


_STAMP = "stamp format_version=1 (f.attrs['format_version'] = 1)"


def _isaaclab_pair(src: _HDF5, dst: _HDF5, src_groups, dst_inv, coverage, src_labels=None, dst_total=None):
    """Relations are recorded per (dataset, demo); src_labels are the root-vote counts of the source demos the
    single-file check inspected. A pass or a decisive remedy needs the source's declared order corroborated by its
    robot roots (_source_declaration); otherwise the remedy is conditional on the source's recording version: branch A
    (an Isaac Lab 2.x recording, WXYZ data) or branch B (Isaac Lab >= 3.0, XYZW data)."""
    sv, ss, _ = _format_version(src)
    dv, ds, _ = _format_version(dst)
    attr_ev = [src.ev('/@format_version', sv if ss == 'ok' else 'absent'),
               dst.ev('/@format_version', dv if ds == 'ok' else 'absent')]
    # Source-wide conflict gates even invalid/new attribute repair advice and pairs containing only one source subset.
    if _conflicting_votes(src_labels or {}):
        return _mixed_pair_findings(coverage, attr_ev)
    if ss == 'invalid' or ds == 'invalid' or (sv or 0) > ISAACLAB_FORMAT_VERSION or (dv or 0) > ISAACLAB_FORMAT_VERSION:
        return [_finding('isaaclab.conversion_quaternion_order', 'unknown',
            'Source or converted format_version is invalid or newer than reviewed; order change not assessed.',
            'Fix the format_version attributes first.', [src.ev('/@format_version', sv), dst.ev('/@format_version', dv)])]
    src_order, dst_order = ORDER[_is_xyzw(sv)], ORDER[_is_xyzw(dv)]
    expected = 0 if src_order == dst_order else (1 if src_order == 'WXYZ' else 3)
    # Pair by byte-identical "actions" over every indexed source demo (merge tools renumber demos, so names are
    # not used); an anchor shared by several source demos is ambiguous and not paired.
    anchors: dict[str, list[str]] = {}
    for name, group in src_groups:
        anchor = src.anchor(group)
        if anchor:
            anchors.setdefault(anchor, []).append(name)
    src_by_name = dict(src_groups)
    pairs, unmatched, ambiguous, no_actions = [], [], [], []
    for name, entry in dst_inv.items():
        found = anchors.get(entry['anchor'], []) if entry.get('anchor') else None
        if found is None:
            no_actions.append(name)
        elif not found:
            unmatched.append(name)
        elif len(found) > 1:
            ambiguous.append(name)
        else:
            pairs.append((found[0], name))
    # Every quaternion-like dataset present in both paired demos is compared or named: compared instances (rows),
    # changed beyond a reorder or in shape (other), not compared with the reason (not_compared), or uninformative
    # (they read the same under every reorder, so any order loads them right).
    rows, other, not_compared, uninformative, votes = [], [], [], 0, {}
    for sname, dname in pairs:
        src_entry, dst_entry = _demo_inventory(src, src_by_name[sname]), dst_inv[dname]
        votes[sname] = _demo_vote(src_entry)[0]
        names = {rel for e in (src_entry, dst_entry) for kind in ('root_pose', 'quaternion', 'pose', 'excluded')
                 for rel in e[kind]}
        for rel in sorted(names & set(src_entry['shapes']) & set(dst_entry['shapes'])):
            s, d = _candidate(src_entry, rel), _candidate(dst_entry, rel)
            kind = (s or d)[0]
            cols = _KIND_COLS[kind]
            if s is None or d is None or src_entry['shapes'][rel] != dst_entry['shapes'][rel]:
                other.append((sname, dname, rel, cols, s and s[2], d and d[2], 'shape changed'))
            elif s[2] is None or d[2] is None:
                not_compared.append((sname, dname, rel, cols, s[1] if s[2] is None else d[1]))
            else:
                ks = _compare(s[2], d[2], kind != 'quaternion')
                if ks == 'none':
                    not_compared.append((sname, dname, rel, cols, 'no row finite in both files'))
                elif ks != 'other' and len(ks) == 4:
                    uninformative += 1
                elif s[1] and d[1]:  # unit quaternions in neither file: nothing this check can judge
                    not_compared.append((sname, dname, rel, cols, 'not unit quaternions in either file'))
                elif ks == 'other':
                    other.append((sname, dname, rel, cols, s[2], d[2], 'changed beyond a reorder'))
                else:
                    rows.append((sname, dname, rel, cols, ks, s[2], d[2]))
    total = len(dst_inv) if dst_total is None else dst_total
    not_inspected = max(total - len(dst_inv), 0)
    relations: dict[str, int] = {}
    for row in rows:
        relations[_relation_name(row[4])] = relations.get(_relation_name(row[4]), 0) + 1
    if other:
        relations['changed beyond a reorder'] = len(other)
    vote_counts: dict[str, int] = {}
    for label in votes.values():
        vote_counts[label] = vote_counts.get(label, 0) + 1
    primary, basis = _source_declaration(src_order, vote_counts, src_labels or {})
    coverage.update(source_demos_indexed=len(src_groups), converted_demos_total=total,
                    converted_demos_inspected=len(dst_inv), paired_demos=len(pairs),
                    paired_quaternion_datasets=len(rows), paired_datasets_changed_beyond_reorder=len(other),
                    paired_datasets_not_compared=len(not_compared), paired_datasets_uninformative=uninformative,
                    paired_datasets_not_comparable=len(not_compared) + uninformative,
                    relations_per_dataset_and_demo=relations, source_root_votes_in_paired_demos=vote_counts,
                    source_declaration=('corroborated by the robot roots' if basis == 'roots' else
                                        'format_version=1, not contradicted by the robot roots' if primary else
                                        'not established'))
    unpaired = {k: sorted(v, key=_natural) for k, v in (('no_source_demo_with_these_actions', unmatched),
                                                         ('actions_shared_by_several_source_demos', ambiguous),
                                                         ('no_readable_actions', no_actions)) if v}
    if unpaired:
        coverage['unpaired_converted_demos'] = unpaired
    code = 'isaaclab.conversion_quaternion_order'
    # Conflicting root votes can come from a mixed-version merge, or from tilted robots. Neither supports a
    # whole-file branch: choosing either branch or re-stamping the source can break demos that are already correct.
    if _conflicting_votes(vote_counts, src_labels or {}):
        return _mixed_pair_findings(coverage, attr_ev)
    lists = coverage.setdefault('_lists', {})
    if other:
        lists['changed_beyond_reorder'] = _instance_groups(other, 'relation')
    if not_compared:
        lists['not_compared'] = _instance_groups(not_compared, 'reason')
    named_other = nc_text = unpaired_text = ''
    if other:
        names = sorted({o[2] for o in other})
        reshaped = sum(o[6] == 'shape changed' for o in other)
        named_other = (f'{len(other)} paired dataset instance(s) (dataset x demo) changed beyond a reorder, with q and '
                       '-q counted as the same rotation' + (f', {reshaped} of them in shape' if reshaped else '')
                       + f' ({", ".join(names[:6])}{" and more" if len(names) > 6 else ""}; '
                       f"all in remedy_lists['pair.changed_beyond_reorder'])")
    if not_compared:
        nc_text = (f'{len(not_compared)} quaternion dataset instance(s) present in both files were not compared ('
                   + '; '.join(sorted({n[4] for n in not_compared})) + f': {_listed(sorted({n[2] for n in not_compared}), 6)}; '
                   "all in remedy_lists['pair.not_compared'])")
    if unpaired:
        parts = [f'{len(unpaired[k])} ({_listed(unpaired[k])}) {text}' for k, text in (
            ('no_source_demo_with_these_actions', 'match no source demo by byte-identical "actions"'),
            ('actions_shared_by_several_source_demos', 'match several source demos with identical "actions"'),
            ('no_readable_actions', 'have no readable "actions" to pair by')) if k in unpaired]
        unpaired_text = (f'{len(unmatched) + len(ambiguous) + len(no_actions)} of {len(dst_inv)} inspected converted '
                         f'demos could not be paired with a source demo: {"; ".join(parts)}')
    gaps = []
    if not_inspected:
        gaps.append(f'{not_inspected} of {total} converted demos were not inspected (max_demos caps the inspected demos; '
                    f'the default, max_demos=None, inspects every demo up to {MAX_DEMOS})')
    if dst.budget['exhausted']:
        gaps.append('the read budget was exhausted')
    m = len(votes)
    wx, xy = vote_counts.get('WXYZ', 0), vote_counts.get('XYZW', 0)
    vote_text = (f'of the {m} paired source demos, the robot roots read upright only as WXYZ in {wx}, only as XYZW in '
                 f'{xy} and do not decide the order in {m - wx - xy}')
    against = (src_labels or {}).get('WXYZ' if src_order == 'XYZW' else 'XYZW', 0)
    vote_text += (f'; {against} inspected source demo(s) read upright only as {"WXYZ" if src_order == "XYZW" else "XYZW"}'
                  if against else '')
    declared = ('format_version=1 (XYZW)' if src_order == 'XYZW' else
                'no format_version (legacy WXYZ)' if ss == 'absent' else f'format_version={sv} (legacy WXYZ)')
    if primary == 'A':
        decl = (f'the source declares {declared} and its robot roots read upright only as WXYZ in every paired demo, '
                'which establishes an Isaac Lab 2.x (WXYZ) recording for floor- or table-mounted robots only (a ceiling- '
                'or wall-mounted Isaac Lab >= 3.0 recording reads the same way)')
    elif primary == 'B' and basis == 'roots':
        decl = (f'the source declares {declared} and its robot roots read upright only as XYZW in every paired demo, '
                'which confirms it for floor- or table-mounted robots only (a ceiling- or wall-mounted Isaac Lab 2.x '
                'recording stamped 1 without conversion reads the same way)')
    elif primary == 'B':
        decl = f'the source declares {declared} and its robot roots do not contradict it ({vote_text})'
    else:
        decl = (f'the source declares {declared} and its robot roots '
                f'{"contradict" if src_order == "XYZW" else "do not establish"} it ({vote_text})')
    other_ev = [dst.ev(f'/data/{o[1]}/{o[2]}[0]', {'converted': _first_row(o[5]), 'source': _first_row(o[4]),
                       'source_demo': o[0], 'relation': o[6]}) for o in other[:2]]
    if not rows and not other:
        if not pairs:
            message = (f'No converted demo could be paired with a source demo (0 of {len(dst_inv)} inspected converted demos, '
                       f'of {total}, matched by byte-identical "actions" among {len(src_groups)} indexed source demos); '
                       'the order change is not assessed.')
        elif uninformative or not_compared:
            parts = ([f'{uninformative} shared dataset instance(s) read the same under every reorder'] if uninformative
                     else []) + ([nc_text] if nc_text else [])
            message = (f'The {len(pairs)} paired demos hold no quaternion dataset that can be compared: '
                       f'{"; ".join(parts)}; the order change is not assessed.')
        else:
            message = (f'The {len(pairs)} paired demos share no quaternion dataset with the source; the order change is '
                       'not assessed.')
        return [_finding(code, 'unknown', message, 'Pass the converted HDF5 produced from this source.' + (
            " Find out why the datasets in remedy_lists['pair.not_compared'] could not be compared." if nc_text else ''),
            attr_ev)]
    bad = [r for r in rows if expected not in r[4]]
    # Branch A: the source holds WXYZ data (an Isaac Lab 2.x recording). The converted data must be its XYZW reorder
    # when the converted file declares XYZW, or when a WXYZ -> WXYZ pair was (partly) reordered to XYZW: a
    # convert_dataset_to_xyzw() double conversion, completed and stamped 1. Branch B: the source holds XYZW data
    # (Isaac Lab >= 3.0); the converted data must be unchanged copies, stamped 1.
    converted_any = any(1 in r[4] for r in rows)
    if dst_order == 'XYZW':
        target_a, final_a = 1, 'keep format_version=1'
    elif converted_any:
        target_a, final_a = 1, _STAMP
    else:
        target_a, final_a = 0, 'keep the format_version as it is'
    target_b, final_b = 0, ('keep format_version=1' if dst_order == 'XYZW' else _STAMP)
    key_a, key_b = ('fixes_if_source_wxyz', 'fixes') if primary == 'B' else ('fixes', 'fixes_if_source_xyzw')
    every = set(dst_inv) if not not_inspected else set()  # 'all' = every demo of the converted file
    fixes_a, fixes_b = _fix_groups(rows, target_a, every), _fix_groups(rows, target_b, every)
    step_a = ('; also delete format_version from the source file, whose value 1 is wrong for an Isaac Lab 2.x recording'
              if src_order == 'XYZW' else '')
    step_b = (('; also stamp format_version=1 on the source file, whose ' + ('missing value' if ss == 'absent' else
               'value below 1') + ' is wrong for an Isaac Lab >= 3.0 recording') if src_order == 'WXYZ' else '')
    text_a = _apply_text(f'pair.{key_a}', fixes_a, final_a) + step_a
    text_b = _apply_text(f'pair.{key_b}', fixes_b, final_b) + step_b
    if src_order == 'WXYZ':
        evidence = ('its robot roots read upright only as XYZW in every paired demo, which points to an Isaac Lab >= 3.0 '
                    'recording unless the robots are mounted tilted (ceiling, wall)' if m and xy == m else
                    f'its robot roots do not decide it: {vote_text}')
        desc_b = 'an Isaac Lab >= 3.0 recording ' + ('that lost format_version' if ss == 'absent' else
                                                     f'stored with format_version={sv}')
    else:
        evidence = (f'it declares format_version=1, but its robot roots contradict that: {vote_text}; an Isaac Lab 2.x '
                    'recording stamped 1 without conversion reads this way, and so does an Isaac Lab >= 3.0 recording with '
                    'robots mounted on a ceiling or wall')
        desc_b = 'an Isaac Lab >= 3.0 recording, as its format_version=1 declares'
    conditional = (f'First establish that the source is an Isaac Lab 2.x (WXYZ) recording ({evidence}). If it is: '
                   f'{text_a}. If it is not ({desc_b}): {text_b}.')
    if primary == 'A':
        decisive = ('Only if the source is an Isaac Lab 2.x (WXYZ) recording; robot-root inference requires robots '
                    'known to start upright on a floor or table: ' + text_a + ". Caveat: the source's robot roots read upright only as WXYZ in "
                    'every paired demo, which establishes an Isaac Lab 2.x (WXYZ) recording for floor- or table-mounted '
                    'robots only; robots mounted on a ceiling or wall read the same way in an Isaac Lab >= 3.0 (XYZW) '
                    'recording. If the source is nevertheless an Isaac Lab >= 3.0 (XYZW) recording (robots mounted on a '
                    f'ceiling or wall): {text_b}.')
    elif primary == 'B':
        decisive = text_b[0].upper() + text_b[1:] + '.'
        if dst_order == 'WXYZ':
            decisive = (("Restore the attribute on the converted file (f.attrs['format_version'] = 1) and change no data"
                         if not fixes_b else decisive[:-1]) + '; never merge files with different format_version values.')
        if basis == 'roots':
            decisive += (" Caveat: this rests on the source's format_version=1, which its robot roots confirm (they read "
                         'upright only as XYZW in every paired demo) for floor- or table-mounted robots only; an Isaac Lab '
                         '2.x (WXYZ) recording stamped 1 without conversion reads the same way if its robots are mounted on '
                         'a ceiling or wall. If the source is nevertheless an Isaac Lab 2.x (WXYZ) recording stamped 1 '
                         f'without conversion (robots mounted on a ceiling or wall): {text_a}.')
        else:
            decisive += (" Caveat: this rests on the source's format_version=1, which its robot roots do not contradict "
                         f'but do not confirm in every paired demo ({vote_text}). If the source is nevertheless an Isaac Lab '
                         f'2.x (WXYZ) recording stamped 1 without conversion: {text_a}.')
        decisive = ('Only if the source is an Isaac Lab >= 3.0 (XYZW) recording with a valid version-1 declaration; '
                    'robot-root inference requires robots known to start upright on a floor or table: ' + decisive)
    consistent = (f'{len(rows)} paired quaternion dataset instance(s) (dataset x demo) in {len(pairs)} paired demos changed '
                  f'as the declared orders require ({_REORDER[expected]}: source {src_order}, converted {dst_order})')
    problems = gaps + [t for t in (unpaired_text, nc_text) if t]
    if not bad and not other:
        if problems:
            steps = ((['inspect every converted demo (default max_demos=None) within the read budget'] if gaps else [])
                     + (['pass the source of every converted demo'] if unpaired else [])
                     + (["find out why the datasets in remedy_lists['pair.not_compared'] could not be compared"]
                        if nc_text else []))
            todo = '; '.join(steps)
            return [_finding(code, 'unknown', f'{consistent}, but {"; ".join(problems)}, so the rest are not verified. '
                             f'{decl[0].upper()}{decl[1:]}.', f'{todo[0].upper()}{todo[1:]} before relying on this conversion.',
                             attr_ev)]
        if primary:
            return [_finding(code, 'pass',
                f'Only under the stated source-version and upright floor/table-mount conditions: {consistent}; {decl}.',
                'None under those conditions. A ceiling or wall mount can make an incorrect order pass this check; '
                'confirm the recording version and mount before relying on it.', attr_ev)]
        lists[key_a], lists[key_b] = fixes_a, fixes_b
        risk = ('an Isaac Lab >= 3.0 recording that lost format_version (merge_hdf5_datasets.py and mp4_to_hdf5.py drop '
                'it) and was then converted loads rotated (reproduced natively on the public Arena G1 recording: 180 deg '
                f'for identity roots; {CITATIONS["isaaclab_drop_tools"]})' if src_order == 'WXYZ' else
                'an Isaac Lab 2.x recording stamped 1 without conversion holds WXYZ data that Isaac Lab loads verbatim')
        return [_finding(code, 'warning', f'{consistent}, but {decl}, so the conversion is verified only if the source '
                         f'is {"an Isaac Lab 2.x (WXYZ) recording" if src_order == "WXYZ" else "the Isaac Lab >= 3.0 (XYZW) recording it declares"}: '
                         f'{risk}.', conditional, attr_ev)]
    if not bad:
        return [_finding(code, 'warning', f'{named_other[0].upper()}{named_other[1:]}: these values are not a quaternion '
            f'reorder of the source, so the conversion is not verified{"; " + "; ".join(problems) if problems else ""}.',
            'Find out what changed these values: a quaternion-order conversion only reorders components. If the '
            'converted file should be a reorder of this source, regenerate it from the source.', attr_ev + other_ev)]
    # Contradictions. Per-demo fixes are given only when every converted demo was inspected and every inspected one is
    # explained: changed-beyond-reorder data, demos that cannot be paired unambiguously or dataset instances that could
    # not be read -> regenerate. Converted demos matching no source demo (they plausibly come from another recording)
    # and datasets holding no unit quaternion or no finite row in either file are named in a note.
    unread = [n for n in not_compared if n[4] not in ('not unit quaternions in either file', 'no row finite in both files')]
    incomplete = bool(gaps)
    regenerate = not incomplete and bool(other or ambiguous or no_actions or unread)
    patch = not incomplete and not regenerate
    if patch:
        lists[key_a], lists[key_b] = fixes_a, fixes_b
        coverage['fix_instances'] = sum(len(g['datasets']) * (len(every) if g['demos'] == 'all' else len(g['demos']))
                                        for g in lists['fixes'])
    found = {}
    for r in bad:
        found[_relation_name(r[4])] = found.get(_relation_name(r[4]), 0) + 1
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r[2], set()).add(_relation_name(r[4]))
    varying = sorted(k for k, v in by_dataset.items() if len(v) > 1)
    why = []
    if src_order == 'XYZW' and dst_order == 'WXYZ' and 'identity' in found:
        why.append('the converted file lost format_version (source declares 1 = XYZW) but its data were copied unchanged. '
                   'The pinned Isaac Lab v3.0.0 reader interprets root_pose as legacy WXYZ and reorders it on load '
                   '(reproduced natively: 180 deg for identity roots). End-effector and other non-root quaternion '
                   f'fields need separate comparison ({CITATIONS["isaaclab_root_pose_only"]}; '
                   f'merge/copy source: {CITATIONS["isaaclab_drop_tools"]})')
    elif src_order == 'WXYZ' and dst_order == 'XYZW' and 'identity' in found:
        why.append(f'the converted file declares XYZW (format_version=1) but {found["identity"]} dataset instance(s) are '
                   'unchanged copies (equal up to sign) of the source, which declares WXYZ; Isaac Lab '
                   'convert_dataset_to_xyzw()/load_episode() reorder only root_pose '
                   f'({CITATIONS["isaaclab_root_pose_only"]}; reproduced on RoboLab)')
    if src_order == dst_order == 'WXYZ' and 'WXYZ->XYZW' in found:
        why.append(f'both files declare WXYZ but {found["WXYZ->XYZW"]} dataset instance(s) were reordered to XYZW, so '
                   'Isaac Lab will reorder them again on load. convert_dataset_to_xyzw() produces this from an input '
                   f'with an explicit format_version below 1 ({CITATIONS["isaaclab_convert_attr_copy"]}; reproduced)')
    if 'reordered twice' in found:
        why.append(f'{found["reordered twice"]} dataset instance(s) were reordered twice (np.roll by 2: a second reorder '
                   'of already reordered data, a 180 deg rotation for most quaternions)')
    rest = {k: v for k, v in found.items() if k not in ('identity', 'WXYZ->XYZW', 'reordered twice')}
    if rest or not why:
        why.append(f'observed relations {sorted(found)} do not match the declared change {_REORDER[expected]}')
    if varying:
        why.append(f'the relation differs between demos for {len(varying)} dataset(s) ({", ".join(varying[:4])}'
                   f'{" and more" if len(varying) > 4 else ""}: a partial conversion or a merge of differently converted '
                   'files)' + (', so every fix is listed per demo' if patch else ''))
    paths = sorted({r[2] for r in bad})
    stored = [f"remedy_lists['pair.{k}']" for k in ('fixes', 'fixes_if_source_xyzw', 'fixes_if_source_wxyz')
              if patch and lists.get(k)]
    named = (', '.join(paths[:6]) + (f' and {len(paths) - 6} more' if len(paths) > 6 else '')
             + (f'; per-demo fixes for all in {stored[0]}' if len(stored) == 1 else
                f'; per-demo fixes for all, one list per branch of the remedy, in {" and ".join(stored)}' if stored
                else ''))
    message = (f'Quaternion order contradiction in {len(bad)} of {len(rows)} paired dataset instance(s) (dataset x demo) '
               f'over {len(pairs)} paired demos ({named}): {"; ".join(why)}. {decl[0].upper()}{decl[1:]}.'
               + (f' Also {named_other}.' if other else '') + (f' Coverage: {"; ".join(problems)}.' if problems else ''))
    if incomplete:
        remedy = (f'Inspect every converted demo before changing data: {"; ".join(gaps)}. Rerun with the default '
                  'max_demos=None (if the read budget is exhausted, check the inputs in parts); per-demo fixes are listed '
                  'only when every converted demo is inspected.')
    elif regenerate:
        reason = '; '.join(([named_other] if other else []) + ([unpaired_text] if ambiguous or no_actions else [])
                           + ([nc_text] if unread else []))
        remedy = (f'Regenerate the converted file from the source instead of patching it: {reason}, so a per-demo '
                  'reorder cannot be verified to fix it.')
    else:
        remedy = decisive if primary else conditional
        if unmatched:
            remedy += (f' Not covered by these fixes: {len(unmatched)} converted demo(s) '
                       f'({_listed(unpaired["no_source_demo_with_these_actions"])}) match no source demo by '
                       'byte-identical "actions"; they plausibly come from another recording (check them against their own '
                       'source) unless their actions were changed, and they are not verified here.')
        if nc_text:
            remedy += f' Not covered either: {nc_text}.'
    ev = attr_ev + [dst.ev(f'/data/{b[1]}/{b[2]}[0]', {'converted': _first_row(b[6]), 'source': _first_row(b[5]),
                    'source_demo': b[0], 'relation': _relation_name(b[4])}) for b in bad[:4]] + other_ev
    return [_finding(code, 'error', message, remedy, ev)]


# ------------------------------------------------------------------------------------------- robomimic


def _controller(env_meta):
    """Single-arm OSC arm config: (config, pointer, 'flat' (robosuite <= 1.4 format) | 'composite' (>= 1.5))."""
    kwargs = env_meta.get('env_kwargs')
    if not isinstance(kwargs, dict):
        return None, None, None
    robots = kwargs.get('robots')
    if isinstance(robots, str):
        robots = [robots]
    config = kwargs.get('controller_configs')
    if not isinstance(robots, list) or len(robots) != 1 or not isinstance(config, dict):
        return None, None, None
    pointer, layout = '/env_kwargs/controller_configs', 'flat'
    if isinstance(config.get('body_parts'), dict):
        arms = [k for k in config['body_parts'] if k in ('right', 'left')]
        if len(arms) != 1 or not isinstance(config['body_parts'][arms[0]], dict):
            return None, None, None
        pointer, layout = pointer + f'/body_parts/{arms[0]}', 'composite'
        config = config['body_parts'][arms[0]]
    if config.get('type') not in POSITION_OSC or config.get('impedance_mode', 'fixed') != 'fixed':
        return None, None, None
    return config, pointer, layout


def _range(config, key, default, dims):
    value = config.get(key, default)
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size in (1, dims) and np.isfinite(array).all() else None


def _demo_fingerprint(actions, eef, dims, low, high):
    """One demo: absolute targets sit on the next end-effector position (median offset <= 5 cm on every axis);
    delta commands follow the end-effector step (correlation >= 0.5 on two axes) and sit far from the position.
    Rows outside the delta input range are supporting evidence only."""
    arm = actions[:, :dims]
    result = {'rows': len(arm), 'rows_outside_input_range': int(np.sum(np.any((arm < low - ATOL) | (arm > high + ATOL), axis=1))),
              'max_abs_arm_action': float(np.abs(arm).max()) if arm.size else 0.0, 'verdict': 'undecided'}
    if eef is None or eef.ndim != 2 or eef.shape[1] != 3 or len(eef) != len(actions) or len(eef) < 4:
        result['why'] = 'no usable robot0_eef_pos'
        return result
    target, nxt, step = arm[:-1, :3], eef[1:], eef[1:] - eef[:-1]
    offset = np.median(target - nxt, axis=0)
    corr = np.array([float(np.corrcoef(target[:, i], step[:, i])[0, 1])
                     if np.std(target[:, i]) > 0 and np.std(step[:, i]) > 0 else np.nan for i in range(3)])
    result.update(median_offset_to_next_eef_m=offset, corr_with_eef_step=corr)
    if np.all(np.abs(offset) <= 0.05):
        result['verdict'] = 'absolute'
    elif np.sum(np.nan_to_num(corr, nan=0.0) >= 0.5) >= 2:
        result['verdict'] = 'delta'
    return result


def _robomimic_env_args(h5: _HDF5, env_meta, status):
    role = h5.role
    pointer = '/data@env_args'
    if status == 'absent':
        return _finding('robomimic.env_args', 'unknown',
            f'[{role}] data.attrs["env_args"] is absent: robomimic cannot build the environment and the controller '
            f'convention of the actions is undeclared ({CITATIONS["robomimic_env_args"]}).',
            'Record env_args (env_name, type, env_kwargs including controller_configs) from the producing environment.',
            [h5.ev(pointer, 'absent')])
    if status == 'invalid' or not isinstance(env_meta.get('env_name'), str) or env_meta.get('type') not in (1, 2, 3):
        return _finding('robomimic.env_args', 'unknown',
            f'[{role}] env_args is not a JSON object with env_name and a robomimic EnvType (1 robosuite, 2 gym, '
            f'3 iGibson), or is too large; the controller convention cannot be read ({CITATIONS["robomimic_env_args"]}).',
            'Rewrite env_args as robomimic JSON metadata.', [h5.ev(pointer, env_meta if status == 'ok' else 'invalid')])
    summary = {k: env_meta.get(k) for k in ('env_name', 'type', 'env_version')}
    return _finding('robomimic.env_args', 'pass', f'[{role}] env_args parsed: {summary}.', 'None.', [h5.ev(pointer, summary)])


_ABS_REMEDY = ('Evaluate or convert with an absolute world-frame controller: robosuite <= 1.4 flat config '
               'control_delta=false (robomimic config.experiment.env_meta_update_dict; Diffusion Policy abs_action=True; '
               'natively 10/10); robosuite >= 1.5 composite config input_type="absolute" with input_ref_frame="world" '
               '(natively 10/10; the OSC default frame "base" fails silently, 0/10). Or train on the delta "actions" '
               f'stream ({CITATIONS["robomimic_abs"]}).')


def _controller_finding(h5: _HDF5, env_meta, demos, total, stream, origin):
    """Controller declared by env_args vs the actions of every inspected demo (decided one by one)."""
    role, code = h5.role, 'robomimic.controller_vs_actions'
    config, pointer, layout = _controller(env_meta)
    if config is None:
        kwargs = env_meta.get('env_kwargs') if isinstance(env_meta.get('env_kwargs'), dict) else {}
        return _finding(code, 'unknown',
            f'[{role}] Controller layout not supported (single-arm robosuite OSC_POSE/OSC_POSITION with fixed '
            'impedance only); the controller is not checked against the actions.', 'None (out of scope).',
            [h5.ev('/data@env_args/env_kwargs/controller_configs', kwargs.get('controller_configs'))])
    dims = POSITION_OSC[config['type']]
    low, high = _range(config, 'input_min', -1, dims), _range(config, 'input_max', 1, dims)
    if layout == 'flat':
        flag = config.get('control_delta', True)
        if not isinstance(flag, bool):
            return _finding(code, 'unknown', f'[{role}] control_delta is not a boolean.', 'Store control_delta as true/false.',
                            [h5.ev(f'/data@env_args{pointer}/control_delta', flag)])
        expected, frame = ('delta', None) if flag else ('absolute', 'world')
        decl_ev = h5.ev(f'/data@env_args{pointer}/control_delta', flag if 'control_delta' in config
                        else 'absent (robosuite <= 1.4 default true)')
        declares = (f'control_delta={str(flag).lower()} (robosuite <= 1.4 flat config; robosuite 1.5.1 refuses this '
                    f'config loudly, {CITATIONS["robosuite15_flat_loud"]})')
    else:
        itype, frame_cfg = config.get('input_type', 'delta'), config.get('input_ref_frame', 'base')
        if itype not in ('delta', 'absolute') or frame_cfg not in ('base', 'world'):
            return _finding(code, 'unknown', f'[{role}] input_type/input_ref_frame not recognized (robosuite asserts).',
                            'Use input_type "delta"/"absolute" and input_ref_frame "base"/"world".',
                            [h5.ev(f'/data@env_args{pointer}', {'input_type': itype, 'input_ref_frame': frame_cfg})])
        expected, frame = itype, (frame_cfg if itype == 'absolute' else None)
        decl_ev = h5.ev(f'/data@env_args{pointer}', {
            'input_type': config.get('input_type', 'absent (default "delta")'),
            'input_ref_frame': config.get('input_ref_frame', 'absent (default "base")'),
            'control_delta': config.get('control_delta', 'absent') if 'control_delta' in config else 'absent'})
        declares = (f'input_type="{itype}"' + (f', input_ref_frame="{frame_cfg}"' if itype == 'absolute' else '')
                    + f' (robosuite >= 1.5 composite config; control_delta, if present, is ignored: '
                    f'{CITATIONS["robosuite15_input_type"]})')
    if low is None or high is None:
        return _finding(code, 'unknown', f'[{role}] input_min/input_max do not match the controller dimension.',
                        'None.', [decl_ev])
    rows = {}
    for name, group in demos:
        actions = h5.read(h5.member(group, stream))
        if actions is None or actions.ndim != 2 or actions.shape[1] < dims:
            continue
        obs = h5.member(group, 'obs')
        eef = h5.read(h5.member(obs, 'robot0_eef_pos')) if isinstance(obs, h5.h5py.Group) else None
        rows[name] = _demo_fingerprint(actions, eef, dims, low, high)
    inspected = len(rows)
    via = f' ({origin})' if origin else ''
    if not rows:
        return _finding(code, 'unknown', f'[{role}] No readable {stream} arrays with the controller action layout.',
                        'None.', [decl_ev])
    verdicts: dict[str, list[str]] = {}
    for name, fp in rows.items():
        verdicts.setdefault(fp['verdict'], []).append(name)
    # world-frame tracking is what an absolute world-frame controller consumes
    good = 'absolute' if expected == 'absolute' and frame == 'world' else 'delta' if expected == 'delta' else None
    abs_bad = [n for n, fp in rows.items() if fp['verdict'] == 'absolute' and good != 'absolute']
    delta_bad = [n for n, fp in rows.items() if fp['verdict'] == 'delta' and good != 'delta']
    bad_names = abs_bad + delta_bad
    undecided = verdicts.get('undecided', [])
    summary = {'inspected': inspected, 'demos_total': total, 'verdicts': {k: len(v) for k, v in verdicts.items()},
               'rows_outside_input_range': sum(fp['rows_outside_input_range'] for fp in rows.values()),
               'rows': sum(fp['rows'] for fp in rows.values()),
               'max_abs_arm_action': max(fp['max_abs_arm_action'] for fp in rows.values())}
    first = next(iter(rows))
    demo_ev = [h5.ev(f'/data/{n}/{stream}[:, 0:{dims}]', {k: v for k, v in rows[n].items()})
               for n in (bad_names[:2] or [first])]
    evidence = [decl_ev, h5.ev(f'/data/*/{stream} ({inspected} of {total} demos)', summary), *demo_ev]
    cover = _coverage_text(inspected, total)
    if not bad_names:
        if undecided:
            return _finding(code, 'unknown',
                f'[{role}] {stream}{via}: {len(undecided)} of {inspected} inspected demos (of {total}) have no clear '
                f'delta/absolute fingerprint (e.g. {", ".join(undecided[:3])}); the {declares} cannot be confirmed for '
                'them.', 'None.', evidence)
        if inspected < total:  # never pass on partial coverage (max_demos, budget or unreadable demos)
            return _finding(code, 'unknown',
                f'[{role}] {stream}{via}: the {inspected} inspected demos look {good}, matching {declares}, but only '
                f'{inspected} of {total} demos were inspected; the others are not verified.',
                'Inspect every demo (default max_demos) to verify the whole file.', evidence)
        return _finding(code, 'pass',
            f'[{role}] {stream}{via}: {cover} look {good}, matching {declares}.', 'None.', evidence)
    consistent = inspected - len(undecided) - len(bad_names)
    mixed = (f' The file mixes conventions: {consistent} other inspected demos match the controller.'
             if consistent else '')
    parts, remedies = [f'[{role}] env_args declare {declares}.'], []
    if abs_bad and expected == 'delta':
        parts.append(f'But {stream}{via} are absolute world-frame targets in {len(abs_bad)} of {inspected} inspected '
                     f'demos (of {total}; e.g. {", ".join(abs_bad[:3])}): their positions sit on the next end-effector '
                     f'position. A delta controller clips and rescales them ({CITATIONS["robosuite_delta"]}). Reproduced '
                     'natively: Diffusion Policy low_dim_abs files with their own env_args 0/10 on robosuite 1.4.1; v1.5 '
                     'lift with world-frame targets under input_type "delta" 0/10 on robosuite 1.5.1.')
        remedies.append(_ABS_REMEDY)
    if abs_bad and expected == 'absolute':
        parts.append(f'But {stream}{via} are world-frame targets in {len(abs_bad)} of {inspected} inspected demos (of '
                     f'{total}; e.g. {", ".join(abs_bad[:3])}). With input_ref_frame "base" robosuite reads them relative '
                     f'to the robot base ({CITATIONS["robosuite15_input_type"]}); reproduced natively on robosuite 1.5.1: '
                     '0/10, 1.07 m.')
        remedies.append('Set input_ref_frame="world" for these world-frame targets (natively 10/10).')
    if delta_bad:
        parts.append(f'But {stream}{via} are delta commands in {len(delta_bad)} of {inspected} inspected demos (of '
                     f'{total}; e.g. {", ".join(delta_bad[:3])}): they follow the end-effector steps instead of its '
                     'position, and an absolute controller drives the end effector to these near-zero values as targets '
                     '(reproduced natively on robosuite 1.4.1 with control_delta=false).')
        remedies.append('For delta actions use control_delta=true (robosuite <= 1.4 flat config) or input_type="delta" '
                        '(robosuite >= 1.5 composite config).')
    if undecided:
        parts.append(f'{len(undecided)} further inspected demos are undecided.')
    status = 'error'
    if stream == 'actions_abs' and expected == 'delta' and abs_bad and not delta_bad:
        # robomimic's official layout: actions_abs is absolute by name (robosuite_add_absolute_actions.py L199-202) and
        # env_args keep describing the delta "actions"; training on it is legitimate if evaluation overrides the
        # controller (robomimic config.experiment.env_meta_update_dict, train.py L95). Not a data contradiction.
        status = 'warning'
        parts.append('The converted dataset uses the source\'s "actions_abs" stream, which is absolute by name '
                     '(robomimic\'s official absolute-action layout), while env_args describe the delta controller of '
                     '"actions": a policy trained on it needs an absolute evaluation controller.')
    return _finding(code, status, ' '.join(parts) + mixed, ' '.join(remedies), evidence)


# --------------------------------------------------------------------------------------------- LeRobot


_QUAT_STEM = re.compile(r'(quat|quaternion|orient|orientation)', re.IGNORECASE)
_QR_STEM = re.compile(r'(^|[._\- ])[qr]$')


def _names(feature):
    names = feature.get('names') if isinstance(feature, dict) else None
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        return names
    return None


def _declared_width(feature):
    shape = feature.get('shape') if isinstance(feature, dict) else None
    if isinstance(shape, list) and shape and all(type(x) is int and x > 0 for x in shape):
        return math.prod(shape)
    names = _names(feature)
    return len(names) if names else None


def _quaternion_groups(names):
    """4 consecutive names that declare quaternion components; returns (start, declared order or None)."""
    groups, i = [], 0
    while i + 4 <= len(names):
        window = [n.lower() for n in names[i:i + 4]]
        prefix = os.path.commonprefix(window)
        suffixes = [w[len(prefix):].strip('._- ') for w in window]
        stem = prefix.strip('._- ')
        order = None
        if sorted(suffixes) == ['w', 'x', 'y', 'z'] and (_QR_STEM.search(stem) or _QUAT_STEM.search(stem)):
            order = ''.join(suffixes).upper()
        elif suffixes == ['0', '1', '2', '3'] and _QUAT_STEM.search(stem):
            # exactly four indexed members: a 3- or 6-wide "orientation" vector is not a quaternion
            neighbours = [names[j].lower() for j in (i - 1, i + 4) if 0 <= j < len(names)]
            if not any(n.startswith(prefix) and n[len(prefix):].strip('._- ').isdigit() for n in neighbours):
                order = 'undeclared'
        if order is not None:
            groups.append((i, order))
            i += 4
        else:
            i += 1
    return groups


def _parquet_files(root: Path):
    data = root / 'data'
    if not data.is_dir():
        return []
    files = []
    for path in sorted(data.rglob('*.parquet')):
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            continue  # links and FIFOs/devices are never opened
        files.append(contained(root, str(path.relative_to(root))))
        if len(files) > MAX_PARQUET_FILES:
            raise InputError(f'More than {MAX_PARQUET_FILES} parquet files under {data}')
    return files


def _column_values(array, pa, pc):
    """One batch of a numeric scalar or flat fixed-width list column as 2-D float64; None if unsupported."""
    kind = array.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        if not (pa.types.is_floating(kind.value_type) or pa.types.is_integer(kind.value_type)):
            return None  # nested lists or non-numeric cells
        widths = pc.unique(pc.list_value_length(array)).to_pylist() if len(array) else [0]
        if len(widths) != 1 or widths[0] is None or array.null_count:
            return None
        values = np.asarray(array.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
        return values.reshape(len(array), widths[0])
    if pa.types.is_floating(kind) or pa.types.is_integer(kind):
        if array.null_count:
            return None
        return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 1)
    return None


def _metadata_bounds(parquet, column, width):
    """Refuse a column before decompressing it when its stored values exceed rows x declared width (list bombs)."""
    meta = parquet.metadata
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        for c in range(group.num_columns):
            chunk = group.column(c)
            path = chunk.path_in_schema
            if path != column and not path.startswith(column + '.'):
                continue
            limit = group.num_rows * max(1, width or 1) + group.num_rows  # + empty-list placeholders
            if chunk.num_values > limit:
                return f'{chunk.num_values} stored values exceed {group.num_rows} rows x declared width {width}'
            if chunk.total_uncompressed_size > MAX_COLUMN_CHUNK_BYTES:
                return f'column chunk of {chunk.total_uncompressed_size} bytes exceeds {MAX_COLUMN_CHUNK_BYTES}'
    return None


def _columns(files, specs, max_rows):
    """Bounded column read.

    specs: {column: (declared width, kept column indices or None for all)}. Returns ({column: 2-D float array of the
    kept columns}, files read, rows read, {column: why unsupported}, rows in all files, value budget exhausted).
    Decoded values (rows x declared width) are capped by MAX_PARQUET_VALUES across files and columns, and each batch
    by MAX_BATCH_VALUES, so an honestly declared but huge list column cannot amplify memory.
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputError('Install policy-deployment-doctor[dataset] (pyarrow) for LeRobot data checks') from exc
    parts = {c: [] for c in specs}
    used, rows, unsupported, total_rows, decoded, exhausted = [], 0, {}, 0, 0, False
    for path in files:
        try:
            from .native_parquet_guard import require_unencrypted_parquet

            require_unencrypted_parquet(path)
            parquet = pq.ParquetFile(path)
            present = [c for c in specs if c in parquet.schema_arrow.names and c not in unsupported]
            if present:
                total_rows += parquet.metadata.num_rows
            if rows >= max_rows or exhausted:
                continue
            for column in list(present):
                why = _metadata_bounds(parquet, column, specs[column][0])
                if why:
                    unsupported[column] = why
                    present.remove(column)
            if not present:
                continue
            used.append(path)
            width = sum(max(1, specs[c][0] or 1) for c in present)
            batch = max(1, min(8192, MAX_BATCH_VALUES // width))
            for record in parquet.iter_batches(columns=present, batch_size=batch):
                wanted = min(record.num_rows, max_rows - rows)
                take = min(wanted, (MAX_PARQUET_VALUES - decoded) // width)
                exhausted = take < wanted
                if take <= 0:
                    break
                for column in present:
                    values = _column_values(record.column(column).slice(0, take), pa, pc)
                    declared, keep = specs[column]
                    if values is None or (declared and values.shape[1] != declared):
                        unsupported.setdefault(column, 'not a numeric column of the declared width')
                    else:
                        parts[column].append(values if keep is None else values[:, keep])
                rows += take
                decoded += take * width
                if rows >= max_rows or exhausted:
                    break
        except (OSError, ValueError, TypeError, NotImplementedError, pa.ArrowException) as exc:
            raise InputError(f'Cannot read parquet {path}: {exc}') from exc
    merged = {}
    for column, chunks in parts.items():
        chunks = [c for c in chunks if c.size]
        if column not in unsupported and chunks and len({c.shape[1] for c in chunks}) == 1:
            merged[column] = np.concatenate(chunks)
    return merged, used, rows, unsupported, total_rows, exhausted


def _file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _LeRobot:
    def __init__(self, path: Path, role: str, max_rows: int):
        self.role, self.max_rows = role, max_rows
        self.root = dataset_root(path)
        self.info_path = contained(self.root, 'meta/info.json')
        before = _file_sha(self.info_path) if self.info_path.is_file() else None
        self.info = read_json(self.info_path)
        self.info_sha = _file_sha(self.info_path)
        if before != self.info_sha:
            raise InputError(f'{role} meta/info.json changed while it was being read')
        features = self.info.get('features')
        if not isinstance(features, dict):
            raise InputError(f'{role} LeRobot info.json has no features mapping')
        self.features = features
        stats_path = contained(self.root, 'meta/stats.json')
        self.stats, self.stats_sha = None, None
        if stats_path.is_file():
            try:
                self.stats, self.stats_sha = read_json(stats_path), _file_sha(stats_path)
            except InputError:
                self.stats = None
        self.stats_path = stats_path
        self.files = _parquet_files(self.root)
        self.hashes: dict[str, str] = {}

    def ev_info(self, pointer, value=None):
        return {'path': str(self.info_path), 'sha256': self.info_sha, 'pointer': pointer, 'value': value}

    def ev_file(self, path, pointer, value=None):
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = _file_sha(path)
        return {'path': key, 'sha256': self.hashes[key], 'pointer': pointer, 'value': value}


def _pointer_token(text):
    return str(text).replace('~', '~0').replace('/', '~1')


_MISLABEL_HINT = (
    'In lerobot/libero_10 the names x,y,z,rx,ry,rz,rw,gripper hold eef position, axis-angle (rx,ry,rz), the FIRST '
    'gripper finger (rw) and the second finger (gripper). LeRobot builds those values for LIBERO but does not create '
    f'these names; they come from converter naming ({CITATIONS["lerobot_libero_state"]}).')


def _lerobot_quaternions(ds: _LeRobot, coverage):
    findings, declared = [], []
    for feature, spec in ds.features.items():
        names = _names(spec)
        if names:
            declared += [(feature, start, order, names[start:start + 4]) for start, order in _quaternion_groups(names)]
    coverage['declared_quaternion_groups'] = len(declared)
    if not declared:
        return findings
    keep = {}  # only the quaternion columns are kept in memory; the whole row is decoded and budgeted
    for feature, start, _, _ in declared:
        keep.setdefault(feature, []).extend(range(start, start + 4))
    specs = {f: (_declared_width(ds.features[f]), sorted(set(k))) for f, k in keep.items()}
    columns, used, rows, unsupported, total_rows, exhausted = _columns(ds.files, specs, ds.max_rows)
    coverage['lerobot_rows_read'] = rows
    coverage['lerobot_rows_total'] = total_rows
    if exhausted:
        coverage['parquet_value_budget_exhausted'] = True
    complete = rows == total_rows and not exhausted
    if unsupported:
        coverage['unsupported_columns'] = unsupported
    code = 'lerobot.declared_quaternion'
    for feature, start, order, names in declared:
        name_ev = ds.ev_info(f'/features/{_pointer_token(feature)}/names', names)
        label = f'{feature}[{start}:{start + 4}] {names}'
        stats = (ds.stats or {}).get(feature) if isinstance(ds.stats, dict) else None
        stat_ev, bound_violation = [], None
        if isinstance(stats, dict):
            try:
                lo = np.asarray(stats.get('min'), dtype=np.float64).reshape(-1)[start:start + 4]
                hi = np.asarray(stats.get('max'), dtype=np.float64).reshape(-1)[start:start + 4]
                if lo.size == 4 and hi.size == 4:
                    bound_violation = bool(np.any(np.abs(np.concatenate([lo, hi])) > 1.0 + NEAR_UNIT_TOL))
                    stat_ev = [{'path': str(ds.stats_path), 'sha256': ds.stats_sha,
                                'pointer': f'/{_pointer_token(feature)}/min,max[{start}:{start + 4}]',
                                'value': {'min': lo, 'max': hi}}]
            except (TypeError, ValueError):
                stat_ev = []
        values = columns.get(feature)
        offset = specs[feature][1].index(start) if feature in specs else 0
        q = values[:, offset:offset + 4] if values is not None and values.shape[1] >= offset + 4 else None
        if q is not None:
            q = q[np.isfinite(q).all(axis=1)]
        if q is not None and len(q):  # rows read decide; stats are used only when no row is readable
            norms = np.linalg.norm(q, axis=1)
            zero = norms <= 1e-6
            live = norms[~zero]
            unit = float(np.mean(np.abs(live - 1.0) <= UNIT_TOL)) if live.size else 0.0
            near = float(np.mean(np.abs(live - 1.0) <= NEAR_UNIT_TOL)) if live.size else 0.0
            row_ev = ds.ev_file(used[0], f'/{_pointer_token(feature)}[:, {start}:{start + 4}] (rows={len(q)} over {len(used)} file(s))',
                                {'unit_norm_fraction': unit, 'near_unit_fraction': near, 'zero_rows': int(zero.sum()),
                                 'norm_min': float(norms.min()), 'norm_max': float(norms.max()),
                                 'first_three_norm_median': float(np.median(np.linalg.norm(q[:, :3], axis=1)))})
            if live.size and unit == 1.0 and not zero.any() and complete:
                findings.append(_finding(code, 'pass', f'[{ds.role}] {label}: all {len(q)} rows are unit quaternions '
                    f'(declared order: {order}; the order itself is not verifiable from values).', 'None.', [name_ev, row_ev]))
            elif live.size and unit == 1.0 and not zero.any():
                findings.append(_finding(code, 'unknown', f'[{ds.role}] {label}: the {len(q)} rows read are unit '
                    f'quaternions, but only {rows} of {total_rows} rows were read (max_rows or the parquet value budget); '
                    'the rest are not verified.', 'Raise max_rows or check the dataset in parts.', [name_ev, row_ev]))
            elif live.size and near >= 0.99:
                findings.append(_finding(code, 'warning',
                    f'[{ds.role}] {label}: rows are near-unit quaternions but not normalized ({unit:.1%} within '
                    f'{UNIT_TOL}, {near:.1%} within {NEAR_UNIT_TOL}; {int(zero.sum())} all-zero rows). Consumers that '
                    'convert them to rotations may assume unit norm.',
                    'Normalize the rows (q / |q|) and replace or drop all-zero rows.', [name_ev, row_ev]))
            else:
                findings.append(_finding(code, 'warning',
                    f'[{ds.role}] {label} are named as quaternion components, but their norms are far from 1 '
                    f'({norms.min():.3g}..{norms.max():.3g}; {near:.1%} of rows within {NEAR_UNIT_TOL}). A consumer that '
                    f'selects these names reads something else. {_MISLABEL_HINT}',
                    'Rename the components to what the values are (for LIBERO: axis_angle1..3 and two gripper finger '
                    'positions); do not change the values, which the LIBERO consumers expect.', [name_ev, row_ev, *stat_ev]))
        elif bound_violation:
            findings.append(_finding(code, 'warning',
                f'[{ds.role}] {label} are named as quaternion components but their recorded min/max exceed 1 in '
                f'magnitude, which no unit quaternion component can (no rows were readable). {_MISLABEL_HINT}',
                'Rename the components to what the values are.', [name_ev, *stat_ev]))
        else:
            reason = unsupported.get(feature, 'no rows were readable')
            findings.append(_finding(code, 'unknown',
                f'[{ds.role}] {label} are named as quaternion components but were not checked: {reason}.',
                'Provide the dataset data/ files with the declared column width.', [name_ev, *stat_ev]))
    return findings


def _episodes(ds: _LeRobot):
    width = _declared_width(ds.features.get('action', {}))
    columns, used, rows, _, _, _ = _columns(ds.files, {'action': (width, None), 'episode_index': (1, None)}, ds.max_rows)
    if ('action' not in columns or 'episode_index' not in columns
            or len(columns['action']) != len(columns['episode_index'])):
        return {}, used, rows
    action, episode = columns['action'], columns['episode_index'][:, 0].astype(np.int64)
    result = {}
    for value in dict.fromkeys(episode.tolist()):
        result[int(value)] = action[episode == value]
    return result, used, rows


def _trace_actions(h5: _HDF5, ds: _LeRobot, groups, coverage):
    """Which source action stream (actions / actions_abs) each converted episode equals (float32-exact)."""
    episodes, used, _ = _episodes(ds)
    index = {}
    for name, group in groups:
        for stream in ('actions', 'actions_abs'):
            arr = h5.read(h5.member(group, stream))
            if arr is None or arr.ndim != 2:
                continue
            arr32 = arr.astype(np.float32)
            for trimmed, view in ((False, arr32), (True, arr32[:-1])):
                index.setdefault((view.shape, hashlib.sha256(np.ascontiguousarray(view).tobytes()).hexdigest()),
                                 (name, stream, trimmed))
    matches = {}
    for episode, values in episodes.items():
        v32 = np.ascontiguousarray(values.astype(np.float32))
        hit = index.get((v32.shape, hashlib.sha256(v32.tobytes()).hexdigest()))
        if hit:
            matches[episode] = hit
    coverage['converted_episodes_read'] = len(episodes)
    coverage['converted_episodes_traced'] = len(matches)
    coverage['source_demos_indexed'] = len(groups)
    return matches, used


# ----------------------------------------------------------------------------------------------- entry


def _status(findings):
    statuses = {f['status'] for f in findings}
    for status in ('error', 'warning', 'unknown', 'pass'):
        if status in statuses:
            return status
    return 'unknown'


def _not_compared(source_layout, converted_kind, evidence):
    return _finding('conversion.not_compared', 'unknown',
        f'The converted {converted_kind} input was not compared with the {source_layout or "unrecognized"} source: '
        'source-to-converted checks exist only for Isaac Lab HDF5 -> Isaac Lab HDF5 (quaternion order) and robomimic '
        'robosuite HDF5 -> LeRobot (action stream and controller convention).',
        'None (out of scope); the converted input was checked on its own where applicable.', [evidence])


def _hdf5_checks(h5: _HDF5, max_demos, coverage):
    env_meta, env_state, data = h5.env_args()
    if data is None:
        raise InputError(f'{h5.role} HDF5 has no "data" group (not a robomimic/Isaac Lab demonstration file)')
    names = h5.demo_names(data)
    limit = MAX_DEMOS if max_demos is None else max_demos  # default: every demo (up to MAX_DEMOS and the budget)
    demos = h5.groups(data, _sample(names, limit))
    total = len(names)
    cov = coverage.setdefault(h5.role, {})
    cov.update(path=str(h5.path), sha256=h5.sha256, size_bytes=h5.size, demos_total=total, demos_inspected=len(demos),
               demo_selection=f'evenly spaced ({limit} of {total})' if total > limit else 'all')
    findings = []
    context = {'demos': demos, 'names': names, 'data': data, 'env_meta': env_meta, 'inventory': None, 'layout': None,
               'total': total}
    if _isaaclab_layout(h5, demos):
        context['layout'] = ISAACLAB
        context['inventory'] = {name: _demo_inventory(h5, group) for name, group in demos}
        for name, group in demos:
            context['inventory'][name]['anchor'] = h5.anchor(group)
        complete = len(demos) == total and not h5.budget['exhausted']
        findings += _isaaclab_single(h5, context['inventory'], total, cov, complete)
    elif env_state == 'ok' and env_meta.get('type') == 1:
        findings.append(_robomimic_env_args(h5, env_meta, env_state))
        if findings[-1]['status'] == 'pass':
            context['layout'] = ROBOSUITE
            findings.append(_controller_finding(h5, env_meta, demos, total, 'actions', None))
    else:
        findings.append(_robomimic_env_args(h5, env_meta or {}, env_state))
    cov['layout'] = context['layout'] or 'unrecognized'
    return findings, context


def _index_groups(h5: _HDF5, context):
    """Every demo up to MAX_INDEX_DEMOS (bounded by the value budget) for tracing and pairing."""
    return h5.groups(context['data'], context['names'][:MAX_INDEX_DEMOS])


def check_sim_provenance(source, *, converted=None, read_roots=None, max_demos=None, max_rows=200_000) -> dict:
    """Check simulator-demonstration provenance conventions and report contradictions with evidence.

    ``source`` is a robomimic/Isaac Lab HDF5 file, or a LeRobot dataset (directory or meta/info.json) when no
    HDF5 is available. ``converted`` is optional: an HDF5 derived from ``source`` (merge/convert/augment output)
    or the LeRobot dataset converted from it. ``read_roots`` restricts every input to those directories.
    By default every demo is inspected (up to MAX_DEMOS and the read budget); ``max_demos`` caps it, evenly spaced
    over the file. A check never passes on partial coverage.
    """
    if max_demos is not None and (type(max_demos) is not int or not 1 <= max_demos <= MAX_DEMOS):
        raise InputError(f'max_demos must be None (all demos) or an integer between 1 and {MAX_DEMOS}')
    if type(max_rows) is not int or not 1 <= max_rows <= 5_000_000:
        raise InputError('max_rows must be an integer between 1 and 5000000')
    source_path = _resolve(source, read_roots, 'source')
    converted_path = _resolve(converted, read_roots, 'converted') if converted is not None else None
    coverage: dict[str, Any] = {}
    budget = {'values': 0, 'exhausted': False}
    findings, opened = [], []
    try:
        if _is_lerobot(source_path):
            if converted_path is not None:
                raise InputError('A LeRobot source cannot have a converted input; pass the HDF5 as source')
            ds = _LeRobot(source_path, 'source', max_rows)
            coverage['source'] = {'path': str(ds.root), 'kind': 'lerobot', 'info_sha256': ds.info_sha,
                                  'parquet_files': len(ds.files)}
            findings += _lerobot_quaternions(ds, coverage['source'])
        else:
            src = _HDF5(source_path, 'source', budget)
            opened.append(src)
            try:
                findings += _with_hdf5(src, converted_path, max_demos, max_rows, coverage, budget, opened)
            except InputError:
                raise
            except HDF5_ERRORS as exc:  # malformed/corrupt HDF5 internals (as rollouts.py L410 maps them)
                raise InputError(f'Invalid or corrupt HDF5: {type(exc).__name__}: {exc}') from exc
        if budget['exhausted']:
            coverage['value_budget_exhausted'] = True
            findings.append(_finding('sim.read_budget_exhausted', 'unknown',
                f'The {MAX_TOTAL_VALUES}-value read budget was exhausted; some datasets were not assessed.',
                'Lower max_demos to stay within the budget, or check the inputs in parts.', []))
    finally:
        errors = []
        for handle in opened:
            try:
                handle.close()
            except InputError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
    for handle in opened:
        if handle.skipped:
            coverage[handle.role]['skipped'] = handle.skipped[:32]
    coverage['values_read'] = budget['values']
    assessed = [f for f in findings if f['code'] not in ('robomimic.env_args',) and f['status'] != 'unknown']
    if not assessed:
        findings.append(_finding('sim.no_applicable_checks', 'unknown',
            'No Isaac Lab quaternion data, supported robomimic controller or declared quaternion features were '
            'assessed; nothing here establishes the conventions of these inputs.', 'None.', []))
    counts = {s: sum(f['status'] == s for f in findings) for s in ('error', 'warning', 'unknown', 'pass')}
    return {'schema_version': SCHEMA_VERSION, 'kind': KIND, 'status': _status(findings), 'counts': counts,
            'coverage': coverage, 'findings': findings, 'remedy_lists': _hoist_lists(coverage), 'scope': SCOPE}


def _hoist_lists(coverage):
    """Move every structured remedy list out of coverage into one place. Each dataset or demo name list is stored
    once in 'names' (identical lists share an id, across roles too) and the groups refer to it by id."""
    names: dict[str, list[str]] = {}
    index: dict[tuple, str] = {}

    def ref(values):
        key = tuple(values)
        if key not in index:
            index[key] = f'L{len(index) + 1}'
            names[index[key]] = list(values)
        return index[key]

    lists: dict[str, Any] = {}
    for scope, cov in coverage.items():
        if not isinstance(cov, dict) or '_lists' not in cov:
            continue
        for name, groups in cov.pop('_lists').items():
            lists[f'{scope}.{name}'] = [dict(g, datasets=ref(g['datasets']),
                                             **({'demos': ref(g['demos'])} if isinstance(g.get('demos'), list) else {}))
                                        for g in groups]
    if lists:
        lists['names'] = names
    return lists


def remedy_list(report: dict, key: str) -> list[dict]:
    """Expand report['remedy_lists'][key] into one entry per dataset, with demo and dataset names resolved."""
    lists = report.get('remedy_lists') or {}
    names = lists.get('names', {})
    entries = []
    for group in lists.get(key, []):
        base = {k: v for k, v in group.items() if k != 'datasets'}
        if isinstance(base.get('demos'), str) and base['demos'] != 'all':
            base['demos'] = list(names[base['demos']])
        entries += [{'dataset': name, **base} for name in names[group['datasets']]]
    return entries


def remedy_lists_digest(report: dict, max_names: int = 5, max_groups: int = 20) -> dict:
    """Bounded summary of report['remedy_lists'] for size-limited views: per list, its groups with counts and the first
    names, plus the size and sha256 of the full canonical lists (to match a stored copy)."""
    lists = report.get('remedy_lists') or {}
    names = lists.get('names', {})
    blob = json.dumps(lists, sort_keys=True, separators=(',', ':')).encode()
    digest: dict[str, Any] = {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(), 'lists': {}}
    for key, groups in lists.items():
        if key == 'names':
            continue
        shown = []
        for group in groups[:max_groups]:
            row = {k: v for k, v in group.items() if k not in ('datasets', 'demos')}
            demos = group.get('demos')
            row['demos'] = demos if demos in (None, 'all') else len(names[demos])
            row['datasets'] = len(names[group['datasets']])
            row['first_datasets'] = names[group['datasets']][:max_names]
            shown.append(row)
        digest['lists'][key] = {'groups': len(groups), 'datasets': sum(len(names[g['datasets']]) for g in groups),
                                'shown': shown}
    return digest


def _with_hdf5(src, converted_path, max_demos, max_rows, coverage, budget, opened):
    findings, src_ctx = _hdf5_checks(src, max_demos, coverage)
    findings = list(findings)
    if converted_path is None:
        return findings
    if _is_lerobot(converted_path):
        ds = _LeRobot(converted_path, 'converted', max_rows)
        cov = coverage['converted'] = {'path': str(ds.root), 'kind': 'lerobot', 'info_sha256': ds.info_sha,
                                       'parquet_files': len(ds.files)}
        findings += _lerobot_quaternions(ds, cov)
        if src_ctx['layout'] != ROBOSUITE:
            findings.append(_not_compared(src_ctx['layout'], 'LeRobot', ds.ev_info('/features', sorted(ds.features))))
            return findings
        groups = _index_groups(src, src_ctx)
        matches, used = _trace_actions(src, ds, groups, cov)
        streams = sorted({m[1] for m in matches.values()})
        if not matches:
            findings.append(_finding('conversion.action_source', 'unknown',
                f'None of the {cov["converted_episodes_read"]} converted episodes read equals a float32 copy of '
                f'"actions"/"actions_abs" (optionally without the terminal row) in the {len(groups)} indexed source demos '
                f'(of {src_ctx["total"]}); the controller convention of the converted actions is not traced.',
                'If the converter transforms actions (scaling, filtering no-op steps, reordering) this check cannot '
                'trace them; otherwise check that this LeRobot dataset was converted from this HDF5.',
                [ds.ev_info('/features/action', ds.features.get('action'))]))
            return findings
        sample = sorted(matches.items())[:4]
        findings.append(_finding('conversion.action_source', 'pass' if len(streams) == 1 else 'warning',
            f'{len(matches)} of {cov["converted_episodes_read"]} converted episodes read equal source stream(s) {streams} '
            f'(of {len(groups)} indexed source demos)' + ('' if len(streams) == 1 else '; mixed streams in one dataset') + '.',
            'None.' if len(streams) == 1 else 'Convert every episode from the same action stream.',
            [ds.ev_file(used[0], f'/action (episode {e})', {'source_demo': m[0], 'stream': m[1],
             'terminal_row_dropped': m[2]}) for e, m in sample]))
        by_name = dict(groups)
        for stream in streams:
            names = sorted({m[0] for m in matches.values() if m[1] == stream})
            subset = [(n, by_name[n]) for n in names]
            findings.append(_controller_finding(src, src_ctx['env_meta'], subset, len(subset), stream,
                                                'the stream the converted LeRobot action equals'))
        return findings
    dst = _HDF5(converted_path, 'converted', budget)
    opened.append(dst)
    found, dst_ctx = _hdf5_checks(dst, max_demos, coverage)
    findings += found
    if src_ctx['layout'] == ISAACLAB and dst_ctx['layout'] == ISAACLAB:
        findings += _isaaclab_pair(src, dst, _index_groups(src, src_ctx), dst_ctx['inventory'],
                                   coverage.setdefault('pair', {}),
                                   coverage.get('source', {}).get('content_order_per_demo'), dst_ctx['total'])
    else:
        findings.append(_not_compared(src_ctx['layout'], f'HDF5 ({dst_ctx["layout"] or "unrecognized"})',
                                      dst.ev('/data', 'converted HDF5 checked on its own only')))
    return findings
