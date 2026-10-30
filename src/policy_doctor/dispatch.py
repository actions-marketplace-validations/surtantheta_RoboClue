"""Check recorded action roles against caller-captured dispatch boundaries."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import MAX_JSON_BYTES, InputError, _pairs, finite_tree
from .rollouts import _recording

ROLES = ('canonical_dispatch', 'canonical_endpoint', 'driver_input', 'driver_return')
SCOPE = ('Numeric consistency against an explicitly declared recording role. Boundary snapshots and frame associations '
         'are caller-recorded evidence, not authenticated driver telemetry. Driver return is not measured motion. '
         'Units, frames, task success and physical causation remain unverified.')


def _index(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise InputError(f'{name} must be a nonnegative signed-int64 index')
    return value


def snapshot_action(action: dict | None) -> dict | None:
    """Copy before handing a mutable action dictionary to an existing driver call."""
    if action is None:
        return None
    if (not isinstance(action, dict) or not 1 <= len(action) <= 256 or
            any(not isinstance(k, str) or not k or len(k) > 256 for k in action)):
        raise InputError('Action snapshot requires 1–256 named scalar channels')
    result = {}
    for name, value in action.items():
        if isinstance(value, np.generic):
            value = value.item()
        if (type(value) not in (int, float) or (type(value) is int and abs(value) > 2**53) or
                not math.isfinite(value)):
            raise InputError('Action channels must be finite real scalars with exactly representable integer values')
        result[name] = value
    return result


def make_dispatch_event(event_id: str, *, frame_index: int | None = None, canonical_dispatch: dict | None = None, canonical_endpoint: dict | None = None,
        driver_input: dict | None = None, driver_return: dict | None = None, recorded_role: str = 'unknown') -> dict:
    """Snapshot values supplied by trusted caller code; never invoke a driver or policy."""
    if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
        raise InputError('event_id must be a nonempty string of at most 256 characters')
    if recorded_role not in (*ROLES, 'unknown'):
        raise InputError('recorded_role must name a supported boundary role or unknown')
    if frame_index is not None:
        _index(frame_index, 'frame_index')
    return {'event_id': event_id, 'frame_index': frame_index, 'recorded_role': recorded_role,
            'canonical_dispatch': snapshot_action(canonical_dispatch), 'canonical_endpoint': snapshot_action(canonical_endpoint), 'driver_input': snapshot_action(driver_input),
            'driver_return': snapshot_action(driver_return)}


def _trace(path: str):
    source = Path(path).expanduser().resolve()
    try:
        if not source.is_file():
            raise InputError('Expected a dispatch evidence JSON file')
        with source.open('rb') as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise InputError('Dispatch evidence exceeds 8 MiB')
        data = json.loads(raw.decode(), object_pairs_hook=_pairs)
        finite_tree(data)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(f'Cannot read dispatch evidence: {exc}') from exc
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'kind', 'events', 'recording_sha256', 'episode_index'} or
            type(data['schema_version']) is not int or data['schema_version'] != 1 or data['kind'] != 'action_dispatch_evidence' or
            not isinstance(data['events'], list) or not 1 <= len(data['events']) <= 5000):
        raise InputError('Expected action_dispatch_evidence schema1 with 1–5000 events')
    digest = data['recording_sha256']
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(v not in '0123456789abcdef' for v in digest)):
        raise InputError('recording_sha256 must be a lowercase SHA-256 digest or null')
    if data['episode_index'] is not None:
        _index(data['episode_index'], 'episode_index')
    events, ids, frames = [], set(), set()
    for event in data['events']:
        if not isinstance(event, dict) or set(event) != {'event_id', 'frame_index', 'recorded_role', *ROLES}:
            raise InputError('Invalid dispatch event fields')
        event = make_dispatch_event(**event)
        if event['event_id'] in ids or (event['frame_index'] is not None and event['frame_index'] in frames):
            raise InputError('Duplicate event/frame identifiers make recording association ambiguous; omit frame_index for unrecorded ticks')
        ids.add(event['event_id'])
        if event['frame_index'] is not None:
            frames.add(event['frame_index'])
        events.append(event)
    return events, data, {'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'format': 'policy-doctor supplemental action boundary JSON', 'size_bytes': len(raw)}


def _compare(left, right, atol, rtol):
    if left is None or right is None or set(left) != set(right):
        return None, [], 0
    changed = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if key not in left or key not in right or (a != b and not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)):
            changed.append({'channel': key, 'expected': a, 'observed': b})
    return not changed, changed[:8], len(changed)


def audit_dispatch(evidence: str, recording: str | None = None, *, episode: int | None = None,
        action_names: list[str] | None = None, atol: float = 1e-6, rtol: float = 1e-5) -> dict:
    """Read a supplemental trace and optionally join a native Parquet episode by frame_index."""
    if any(type(v) not in (int, float) or (type(v) is int and abs(v) > 2**53) or
           not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise InputError('Tolerances must be finite nonnegative numbers, not booleans')
    events, trace, artifact = _trace(evidence)
    artifacts, rows, selected_rows = [artifact], {}, 0
    binding_verified = False
    if recording is not None:
        _index(episode, 'episode')
        if (not isinstance(action_names, list) or not 1 <= len(action_names) <= 256 or
                any(not isinstance(n, str) or not n or len(n) > 256 for n in action_names) or len(set(action_names)) != len(action_names)):
            raise InputError('Native action vector requires an explicit unique ordered action_names mapping')
        if Path(recording).suffix.lower() != '.parquet':
            raise InputError('Dispatch recording association currently supports native LeRobot Parquet only')
        _, arrays, source = _recording(recording, True, episode)
        if trace['recording_sha256'] is not None and trace['recording_sha256'] != source['sha256']:
            raise InputError('Supplemental trace is bound to different native recording bytes')
        if trace['episode_index'] is not None and trace['episode_index'] != episode:
            raise InputError('Supplemental trace is bound to a different episode')
        binding_verified = trace['recording_sha256'] is not None and trace['episode_index'] is not None
        frames, actions = arrays.get('frame_index'), arrays['action']
        if (frames is None or frames.ndim != 1 or frames.dtype.kind not in 'iu' or
                any(int(v) < 0 or int(v) >= 2**63 for v in frames) or len(set(map(int, frames))) != len(frames)):
            raise InputError('Native frame_index must be unique nonnegative signed-int64 values within the episode')
        if actions.ndim != 2 or actions.shape[1] != len(action_names):
            raise InputError('Native action width does not match explicit action_names')
        for row, (frame, action) in enumerate(zip(frames, actions, strict=True)):
            rows[int(frame)] = (snapshot_action(dict(zip(action_names, action.tolist(), strict=True))), row)
        selected_rows = len(rows)
        artifacts.append(source)
    elif episode is not None or action_names is not None:
        raise InputError('episode/action_names require a native recording')
    counts = {'pass': 0, 'error': 0, 'unknown': 0}
    details, first_error, changes = [], None, 0
    for index, event in enumerate(events):
        item = {'event_id': event['event_id'], 'frame_index': event['frame_index'], 'evidence_pointer': f'/events/{index}',
                'declared_recorded_role': event['recorded_role'], 'driver_input_to_return': 'unknown'}
        equal, preview, total = _compare(event['driver_input'], event['driver_return'], atol, rtol)
        if equal is not None:
            item['driver_input_to_return'] = 'equal' if equal else 'different'
            if not equal:
                changes += 1
            item['driver_change_preview'], item['driver_changed_channels'] = preview, total
        record = rows.get(event['frame_index'])
        expected = event.get(event['recorded_role']) if event['recorded_role'] in ROLES else None
        if record is None:
            item.update(status='unknown', reason='No associated native recording row')
        else:
            values, row = record
            item['selected_episode_row'] = row
            item['matches_roles'] = [role for role in ROLES if _compare(event[role], values, atol, rtol)[0] is True]
            equal, preview, total = _compare(expected, values, atol, rtol)
            item['declared_role_matches'] = equal
            if not binding_verified:
                item.update(status='unknown', reason='Recording hash or episode binding is missing')
            elif equal is None:
                item.update(status='unknown', reason='Recording role, corresponding snapshot or full channel-name alignment is missing')
            else:
                item.update(status='pass' if equal else 'error', recording_difference_preview=preview,
                            recording_changed_channels=total, difference_preview_truncated=total > len(preview))
        counts[item['status']] += 1
        if item['status'] == 'error' and first_error is None:
            first_error = item
        if len(details) < 12:
            details.append(item)
    return {'schema_version': 1, 'kind': 'action_boundary_audit',
            'status': 'error' if counts['error'] else 'unknown' if counts['unknown'] else 'pass',
            'counts': counts, 'events_checked': len(events), 'driver_change_events': changes,
            'first_recording_contradiction': first_error, 'event_preview': details, 'event_preview_truncated': len(events) > len(details),
            'native_recording_rows': selected_rows, 'recording_rows_with_boundary_evidence': len(set(rows) & {v['frame_index'] for v in events}),
            'coverage': 'Only supplied trace events are checked; untraced native rows are not certified.',
            'association_basis': 'Caller-declared episode/frame IDs and explicit ordered action_names; not clock alignment or authenticated lineage.',
            'recording_hash_and_episode_bound': binding_verified,
            'action_names': action_names, 'episode': episode, 'tolerances': {'atol': atol, 'rtol': rtol},
            'artifacts': artifacts, 'execution_status': 'unknown', 'physical_cause': 'unknown', 'scope': SCOPE}
