"""Explain saved native ACT selection arithmetic and separately evidenced origins.

No policy import, checkpoint execution or inference occurs in this reader.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile

import numpy as np

from .artifacts import InputError, _pairs, finite_tree
from .layouts import _read, _safe_root, _tolerances
from .rollouts import _npz_preflight

SUPPORTED_SOURCE = {
    'repository': 'https://github.com/huggingface/lerobot',
    'revision': '146d8474932c8b566717eb2738ba459337ab71c4',
    'modeling_sha256': '83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af',
    'config_sha256': '35a60c0dd3c811b137b5eb035bdbd40d40e2c24ba0cda61756052e6e1c829282',
}
MAX_BYTES = 32 * 1024 * 1024
MAX_VALUES = 1_000_000
MAX_EVENTS = 256
MAX_ARRAYS = 4096
SOURCE_NAMES = {'modeling_act.py', 'configuration_act.py', 'LICENSE.txt'}
SCOPE = ('Saved policy-output selection arithmetic and captured-call origins only. '
         'History-based contributors are conditional arithmetic explanations; untracked external state mutations leave query lineage unknown. '
         'Query IDs are local observer identities, not episode/time alignment or authenticated telemetry. '
         'Decoder, issued/returned/recorded actions, measured motion and task competence remain unknown.')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _parse(raw):
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
        finite_tree(value)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(f'Invalid selection JSON: {exc}') from exc
    if type(value) is not dict:
        raise InputError('Selection JSON must be an object')
    return value


def _tree(root):
    if not root.is_dir():
        raise InputError('Selection capture must be a directory')
    names = set()
    for path in root.iterdir():
        if path.is_symlink():
            raise InputError('Selection symlinks are refused')
        if path.name == 'sources' and path.is_dir():
            for child in path.iterdir():
                if child.is_symlink() or not child.is_file() or child.name not in SOURCE_NAMES:
                    raise InputError('Unexpected selection source payload')
                names.add('sources/' + child.name)
        elif path.name in {'selection.json', 'arrays.npz', 'READY.json'} and path.is_file():
            names.add(path.name)
        else:
            raise InputError('Unexpected selection capture entry')
    return names


def _arrays(raw, specs):
    if type(specs) is not dict or not 0 <= len(specs) <= MAX_ARRAYS:
        raise InputError('Selection requires at most4096 array specifications')
    if not specs:
        # np.savez with no arrays emits exactly the canonical empty ZIP record.
        if raw != b'PK\x05\x06' + b'\0' * 18:
            raise InputError('Empty selection inventory requires an empty numeric archive')
        return {}
    _npz_preflight(raw, MAX_BYTES, max_streams=MAX_ARRAYS)
    arrays, count = {}, 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len({e.filename for e in entries}) != len(entries) or len(entries) != len(specs):
                raise InputError('Selection array inventory must exactly match specifications')
            for entry in entries:
                key = entry.filename[:-4]
                if (not entry.filename.endswith('.npy') or not re.fullmatch(r'[A-Za-z0-9_]{1,96}', key)
                        or key not in specs):
                    raise InputError('Invalid selection array name')
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise InputError('Unsupported selection NPY header')
                    shape, _, dtype = reader(stream, max_header_size=10000)
                    size = math.prod(shape)
                    count += size
                    if (len(shape) > 3 or any(type(n) is not int or not 0 <= n <= 256 for n in shape)
                            or dtype.kind not in 'if' or dtype.itemsize > 8
                            or count > MAX_VALUES or size * dtype.itemsize != entry.file_size - stream.tell()):
                        raise InputError('Selection array shape/dtype/value budget or payload size invalid')
                    if specs[key] != {'shape': list(shape), 'dtype': str(dtype)}:
                        raise InputError('Selection array metadata differs from original header')
            with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=10000) as loaded:
                for key in specs:
                    value = loaded[key]
                    if not np.isfinite(value).all():
                        raise InputError('Selection arrays must be finite')
                    arrays[key] = value
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, OverflowError, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f'Invalid selection arrays: {exc}') from exc
    return arrays


def _load(path):
    root = _safe_root(path)
    tree = _tree(root)
    raw = _read(root / 'selection.json', 2 * 1024 * 1024)
    ready_raw = _read(root / 'READY.json', 4096)
    manifest, ready = _parse(raw), _parse(ready_raw)
    if (set(ready) != {'schema_version', 'manifest_sha256'} or type(ready['schema_version']) is not int
            or ready['schema_version'] != 1 or ready['manifest_sha256'] != _digest(raw)):
        raise InputError('Selection commit marker/hash mismatch')
    if (manifest.get('kind') != 'act_selection_capture' or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or manifest.get('domain') != 'policy_output'
            or type(manifest.get('events')) is not list or not 1 <= len(manifest['events']) <= MAX_EVENTS):
        raise InputError('Unsupported selection capture schema/domain/events')
    entries = manifest.get('files')
    if type(entries) is not list or len(entries) != 4:
        raise InputError('Selection payload manifest requires arrays and three source files')
    hashes = {'selection.json': _digest(raw), 'READY.json': _digest(ready_raw)}
    payloads, total = {}, len(raw) + len(ready_raw)
    permitted = {'arrays.npz'} | {'sources/' + name for name in SOURCE_NAMES}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {'path', 'sha256'}:
            raise InputError('Invalid selection payload entry')
        name = entry['path']
        if (type(name) is not str or name not in permitted or name in payloads
                or type(entry['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
            raise InputError('Invalid selection payload path/hash')
        content = _read(root / name, MAX_BYTES if name == 'arrays.npz' else 512 * 1024)
        total += len(content)
        if total > MAX_BYTES or _digest(content) != entry['sha256']:
            raise InputError('Selection payload hash mismatch or total budget exceeded')
        payloads[name], hashes[name] = content, _digest(content)
    if tree != set(hashes):
        raise InputError('Selection manifest must cover exactly all files')
    source = manifest.get('source')
    if type(source) is not dict or type(source.get('files')) is not dict:
        raise InputError('Missing selection source identity')
    if source['files'] != {name: {'sha256': hashes['sources/' + name]} for name in SOURCE_NAMES}:
        raise InputError('Selection source identity and payload hashes differ')
    supported = all(source.get(k) == v for k, v in SUPPORTED_SOURCE.items())
    supported = (supported and source.get('modeling_sha256') == hashes['sources/modeling_act.py']
                 and source.get('config_sha256') == hashes['sources/configuration_act.py'])
    arrays = _arrays(payloads['arrays.npz'], manifest.get('arrays'))
    env = manifest.get('environment_selection')
    if type(env) is not dict or set(env) != {'source_num_envs', 'env_indices', 'saved_rows', 'omitted_envs'}:
        raise InputError('Missing selection environment selection')
    rows, count = env.get('env_indices'), env.get('source_num_envs')
    if count is not None or rows is not None:
        if (type(count) is not int or not 1 <= count <= 1_000_000 or type(rows) is not list or not 1 <= len(rows) <= 32
                or any(type(n) is not int or not 0 <= n < count for n in rows) or len(rows) != len(set(rows))
                or type(env.get('saved_rows')) is not int or type(env.get('omitted_envs')) is not int
                or env['saved_rows'] != len(rows) or env['omitted_envs'] != count - len(rows)):
            raise InputError('Invalid saved batch-row selection')
    elif (env['saved_rows'] is not None or env['omitted_envs'] is not None
          or any(type(e) is dict and e.get('type') == 'select' for e in manifest['events'])):
        raise InputError('A selected action requires an explicit bounded batch-row selection')
    return root, manifest, arrays, hashes, supported


def _recheck(root, hashes):
    if _tree(root) != set(hashes):
        raise InputError('Selection file set changed during verification')
    for name, expected in hashes.items():
        if _digest(_read(root / name, MAX_BYTES)) != expected:
            raise InputError('Selection file changed during verification')


def _queue_replay(queue, chunk, steps, maxlen=None):
    if queue:
        if chunk is not None:
            raise InputError('A nonempty native queue must not query another chunk')
        return queue[0].copy(), [x.copy() for x in queue[1:]]
    if chunk is None:
        return None, None
    retained = list(range(min(steps, chunk.shape[1])))
    retained = retained[-(maxlen if maxlen is not None else steps):]
    return chunk[:, retained[0]].copy(), [chunk[:, i].copy() for i in retained[1:]]


def _ensemble_replay(aggregate, counts, actions, weights, cumsum):
    size = actions.shape[1]
    if aggregate is None:
        if counts is not None:
            raise InputError('Native initial ensemble counts must also be absent')
        updated = actions.copy()
        updated_count = np.ones((size, 1), dtype=np.int64)
    else:
        if (counts is None or aggregate.shape != (actions.shape[0], size - 1, actions.shape[2])
                or counts.shape != (size - 1, 1) or counts.dtype != np.dtype('int64')
                or np.any(counts < 1) or np.any(counts >= size)):
            raise InputError('Native retained ensemble shape/counts inconsistent with current chunk')
        updated = aggregate.copy()
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            updated *= cumsum[counts - 1]
            updated += actions[:, :-1] * weights[counts]
            updated /= cumsum[counts]
        updated = np.concatenate([updated, actions[:, -1:]], axis=1)
        updated_count = np.minimum(counts + 1, size)
        updated_count = np.concatenate([updated_count, np.ones_like(updated_count[-1:])])
    return updated[:, 0].copy(), updated[:, 1:].copy(), updated_count[1:].copy()


def _same(left, right, atol, rtol):
    if left is None or right is None:
        return left is right
    return left.shape == right.shape and left.dtype == right.dtype and bool(np.allclose(left, right, atol=atol, rtol=rtol))


def _state_equal(left, right, arrays, atol, rtol):
    if left is None or right is None or left.get('mode') != right.get('mode'):
        return False
    if left['mode'] == 'queue':
        a, b = left.get('queue'), right.get('queue')
        return (left.get('queue_maxlen') == right.get('queue_maxlen')
                and type(a) is list and type(b) is list and len(a) == len(b)
                and all(_same(arrays[x], arrays[y], atol, rtol) for x, y in zip(a, b, strict=True)))
    return all(_same(arrays.get(left.get(k)), arrays.get(right.get(k)), atol, rtol)
               for k in ('aggregate', 'counts', 'weights', 'weights_cumsum'))


def _analyze(manifest, arrays, supported, atol, rtol):
    results, origins, previous, next_query = [], None, None, 0
    query_values = {}
    row_count = manifest['environment_selection'].get('saved_rows')
    coverage = manifest.get('coverage', {})
    history_observed = (type(coverage) is dict and coverage.get('target') == 'one_policy_on_owner_thread'
                        and coverage.get('complete_target_call_history') is True)
    selections = 0
    for ordinal, event in enumerate(manifest['events']):
        if (type(event) is not dict or type(event.get('event_id')) is not int or event['event_id'] != ordinal
                or event.get('type') not in ('select', 'reset') or event.get('completion') != 'normal'):
            raise InputError('Selection events require sequential IDs and normal select/reset completion')
        settings = event.get('settings')
        if type(settings) is not dict:
            raise InputError('Selection event requires captured effective settings')
        steps, size, coefficient = (settings.get(k) for k in ('n_action_steps', 'chunk_size', 'temporal_ensemble_coeff'))
        if (type(steps) is not int or type(size) is not int or not 1 <= steps <= size <= 256
                or (coefficient is not None and (type(coefficient) not in (int, float) or not math.isfinite(coefficient)))):
            raise InputError('Invalid effective selection settings')
        mode = 'queue' if coefficient is None else 'ensemble'
        pre, post = event.get('pre_state'), event.get('post_state')
        def array(ref, shape=None, floating=True):
            if ref is None:
                return None
            if type(ref) is not str or ref not in arrays:
                raise InputError('Selection event references an absent array')
            result = arrays[ref]
            if floating and result.dtype.kind != 'f':
                raise InputError('Policy output/state must retain a real floating dtype')
            if floating and result.ndim >= 2 and not 1 <= result.shape[-1] <= 256:
                raise InputError('Selected action/state width must be between1and256')
            if shape is not None and (result.ndim != len(shape) or any(n is not None and result.shape[i] != n for i, n in enumerate(shape))):
                raise InputError('Selection event array has an inconsistent native shape')
            return result
        for state in (pre, post):
            if type(state) is not dict or state.get('mode') != mode:
                raise InputError('Effective selection mode and state disagree')
            if mode == 'queue':
                queue_refs = state.get('queue')
                maxlen = state.get('queue_maxlen')
                if type(maxlen) is not int or not 1 <= maxlen <= 4096:
                    raise InputError('Actual native deque maxlen must be retained')
                if type(queue_refs) is not list or len(queue_refs) > min(maxlen, 256):
                    raise InputError('Native queue exceeds actual deque maxlen')
                for ref in queue_refs:
                    array(ref, (row_count, None))
            else:
                array(state.get('aggregate'), (row_count, None, None))
                counts_value = array(state.get('counts'), (None, 1), floating=False)
                if counts_value is not None and counts_value.dtype != np.dtype('int64'):
                    raise InputError('Native ensemble counts must remain int64')
                for key in ('weights', 'weights_cumsum'):
                    value = array(state.get(key), (size,))
                    if value is None or np.any(value < 0) or (key == 'weights_cumsum' and np.any(value <= 0)):
                        raise InputError('Native weights must be nonnegative with positive cumulative denominators')
        predictions = event.get('predictions')
        if type(predictions) is not list or len(predictions) > 1:
            raise InputError('One selected native call has at most one query')
        prediction = None
        query_id = None
        if predictions:
            query = predictions[0]
            if (type(query) is not dict or set(query) != {'query_id', 'array'}
                    or type(query['query_id']) is not int or query['query_id'] != next_query):
                raise InputError('Query IDs must identify sequential actual captured query returns')
            next_query += 1
            query_id = query['query_id']
            prediction = array(query['array'], (row_count, size, None))
            if prediction is None:
                raise InputError('A captured prediction requires its numeric return')
            query_values[query_id] = prediction
        selected = array(event.get('selected'), (row_count, None))
        if event['type'] == 'select' and selected is None:
            raise InputError('Normal selected native return must be saved')
        if prediction is not None and selected is not None and prediction.shape[2] != selected.shape[1]:
            raise InputError('Selected action width differs from its query')
        if previous is not None and not _state_equal(previous, pre, arrays, 0, 0):
            origins = None
        replay = {'status': 'unknown', 'reason': 'Native source revision unsupported'}
        origin = {'status': 'unknown', 'contributors': [], 'reason': 'Original contributor history is absent'}
        if event['type'] == 'reset':
            if prediction is not None or selected is not None:
                raise InputError('Reset events must not contain prediction/selected outputs')
            empty = (not post['queue'] and post['queue_maxlen'] == steps if mode == 'queue'
                     else post.get('aggregate') is None and post.get('counts') is None)
            if supported:
                replay = {'status': 'pass' if empty else 'error', 'state_cleared': empty}
                origins = [] if empty else None
            previous = post
            results.append({'event_id': ordinal, 'type': 'reset', 'numeric_replay': replay, 'origin': {'status': 'pass' if supported and empty else 'unknown', 'contributors': []}})
            continue
        selections += 1
        if supported:
            try:
                if mode == 'queue':
                    queue = [arrays[ref] for ref in pre['queue']]
                    expected, remaining = _queue_replay(queue, prediction, steps, pre['queue_maxlen'])
                    matches = expected is not None and _same(expected, selected, atol, rtol)
                    actual_post = [arrays[ref] for ref in post['queue']]
                    post_matches = (pre['queue_maxlen'] == post['queue_maxlen']
                                    and remaining is not None and len(remaining) == len(actual_post)
                                    and all(_same(a, b, atol, rtol) for a, b in zip(remaining, actual_post, strict=True)))
                    if expected is None:
                        replay = {'status': 'unknown', 'reason': 'Empty queue but original query return was not captured'}
                        origins = None
                    else:
                        replay = {'status': 'pass' if matches and post_matches else 'error', 'selected_matches': matches, 'post_state_matches': post_matches}
                    if not queue and prediction is not None:
                        offsets = list(range(min(steps, size)))[-pre['queue_maxlen']:]
                        origins = [[{'query_id': query_id, 'query_event_id': ordinal, 'offset': i, 'weight': 1.0}] for i in offsets]
                    if origins:
                        origin = {'status': 'pass', 'contributors': origins[0]}
                        origins = origins[1:]
                else:
                    if prediction is None:
                        replay = {'status': 'unknown', 'reason': 'Actual ensemble query return absent'}
                        origins = None
                    else:
                        weights, cumsum = (arrays[pre[k]] for k in ('weights', 'weights_cumsum'))
                        with np.errstate(over='ignore', under='ignore'):
                            weight_rule = np.allclose(weights, np.exp(-coefficient * np.arange(size)), atol=atol, rtol=rtol)
                        cumulative_rule = bool(np.allclose(cumsum, np.cumsum(weights), atol=atol, rtol=rtol))
                        expected, remaining, counts = _ensemble_replay(array(pre.get('aggregate')), array(pre.get('counts'), floating=False), prediction, weights, cumsum)
                        matches = _same(expected, selected, atol, rtol)
                        post_matches = (_same(remaining, array(post.get('aggregate')), atol, rtol)
                                        and _same(counts, array(post.get('counts'), floating=False), 0, 0)
                                        and _same(weights, arrays[post['weights']], 0, 0)
                                        and _same(cumsum, arrays[post['weights_cumsum']], 0, 0))
                        replay = {'status': 'pass' if matches and post_matches else 'error', 'selected_matches': matches, 'post_state_matches': post_matches,
                                  'declared_coefficient_matches_stored_weights': bool(weight_rule),
                                  'stored_cumulative_sums_match_weights': cumulative_rule,
                                  'weight_basis': 'Actual captured helper weights/cumulative sums, not reconstructed from the configuration coefficient'}
                        if pre.get('aggregate') is None:
                            origins = [[] for _ in range(size - 1)]
                        if origins is not None and len(origins) == size - 1:
                            full = []
                            for i in range(size):
                                contributors = list(origins[i]) if i < len(origins) else []
                                contributors.append({'query_id': query_id, 'query_event_id': ordinal, 'offset': i})
                                full.append(contributors)
                            contributors = full[0]
                            denominator = sum(float(weights[i]) for i in range(len(contributors)))
                            if denominator > 0 and math.isfinite(denominator) and cumulative_rule:
                                origin = {'status': 'pass', 'contributors': [{**item, 'weight': float(weights[i]) / denominator} for i, item in enumerate(contributors)]}
                                origins = full[1:]
                            else:
                                origins = None
                        else:
                            origins = None
            except (FloatingPointError, ValueError, IndexError, InputError) as exc:
                replay = {'status': 'error', 'reason': f'Captured state contradicts supported native selection: {str(exc)[:256]}'}
                origins = None
        if replay['status'] != 'pass':
            origins = None
            if origin['status'] == 'pass':
                origin = {'status': 'unknown', 'contributors': [],
                          'expected_contributors': origin['contributors'],
                          'reason': 'Captured output/state does not verify this contributor explanation'}
        arithmetic = origin
        if arithmetic['status'] == 'pass':
            reconstructed = np.zeros(selected.shape, dtype=np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                for item in arithmetic['contributors']:
                    reconstructed += query_values[item['query_id']][:, item['offset']] * item['weight']
            if not np.allclose(reconstructed, selected, atol=atol, rtol=rtol):
                arithmetic = {'status': 'unknown', 'contributors': [],
                              'reason': 'Contribution sum does not reproduce saved output within the supplied tolerances'}
            else:
                arithmetic = dict(arithmetic, reconstruction_status='pass')
        if not history_observed:
            arithmetic = {'status': 'unknown', 'contributors': [], 'reason': 'Continuous target-call coverage is absent or incomplete'}
            origins = None
        elif arithmetic['status'] == 'pass':
            arithmetic = dict(arithmetic, basis='Captured queries and replayed native transitions, conditional on no external state mutation')
        origin = {'status': 'unknown', 'contributors': [],
                  'reason': 'State values and continuous call coverage do not witness query identity; external state mutations are untracked'}
        previous = post
        results.append({'event_id': ordinal, 'type': 'select', 'selection_ordinal': selections - 1,
                        'mode': mode, 'settings': settings, 'numeric_replay': replay, 'origin': origin,
                        'arithmetic_contributors': arithmetic,
                        'selected_array': event['selected'], 'evidence_pointer': f'/events/{ordinal}'})
    return results


def audit_selection_capture(capture, *, atol=1e-6, rtol=1e-5):
    """Replay saved selections and explain origins without importing native code."""
    _tolerances(atol, rtol)
    root, manifest, arrays, hashes, supported = _load(capture)
    events = _analyze(manifest, arrays, supported, atol, rtol)
    from .selection_bindings import validate_local_binding

    for original, result in zip(manifest['events'], events, strict=True):
        if result['type'] == 'select':
            result['local_binding'] = validate_local_binding(
                original, manifest, arrays, source_supported=supported,
                numeric_status=result['numeric_replay']['status'])
    errors = [e for e in events if e['numeric_replay']['status'] == 'error']
    unknown = [e for e in events if e['numeric_replay']['status'] == 'unknown' or e['origin']['status'] == 'unknown']
    selection_count = sum(e['type'] == 'select' for e in events)
    numeric_status = 'error' if errors else 'unknown' if not selection_count or any(e['numeric_replay']['status'] == 'unknown' for e in events) else 'pass'
    arithmetic_status = ('pass' if selection_count and all(e.get('arithmetic_contributors', {}).get('status') == 'pass'
                                                         for e in events if e['type'] == 'select') else 'unknown')
    report = {'kind': 'action_selection_audit', 'status': 'error' if errors else 'unknown' if unknown or not selection_count else 'pass',
              'local_binding_status': ('pass' if selection_count and all(e['local_binding']['status'] == 'pass'
                                                                       for e in events if e['type'] == 'select') else 'unknown'),
              'numeric_replay_status': numeric_status, 'arithmetic_contribution_status': arithmetic_status,
              'query_lineage_status': 'unknown',
              'source_supported': supported, 'domain': manifest['domain'], 'source': manifest['source'],
              'events': events, 'selection_count': selection_count,
              'first_contradiction': errors[0] if errors else None,
              'environment_selection': manifest['environment_selection'],
              'downstream_association': {'status': 'unknown', 'reason': 'No decoder/issued/returned/recorded boundary association captured by this observer'},
              'tolerances': {'atol': atol, 'rtol': rtol}, 'scope': SCOPE,
              'artifacts': [{'path': str(root / name), 'sha256': digest} for name, digest in sorted(hashes.items())]}
    _recheck(root, hashes)
    return report


def compare_selection_captures(reference, candidate, *, atol=1e-6, rtol=1e-5):
    """Compare captured selection-call order; a lawful difference is not a fault."""
    _tolerances(atol, rtol)
    left, right = (audit_selection_capture(p, atol=atol, rtol=rtol) for p in (reference, candidate))
    loaded = [_load(path) for path in (reference, candidate)]
    for report, item in zip((left, right), loaded, strict=True):
        audited_hashes = {entry['path']: entry['sha256'] for entry in report['artifacts']}
        reloaded_hashes = {str(item[0] / name): digest for name, digest in item[3].items()}
        if audited_hashes != reloaded_hashes:
            raise InputError('Selection capture changed between audit and comparison load')
    a, b = ([e for e in report['events'] if e['type'] == 'select'] for report in (left, right))
    first = None
    original_events = [[e for e in item[1]['events'] if e['type'] == 'select'] for item in loaded]
    latest_queries = [None, None]
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        for side in (0, 1):
            predictions = original_events[side][index]['predictions']
            if predictions:
                latest_queries[side] = loaded[side][2][predictions[0]['array']]
        if not _same(loaded[0][2][x['selected_array']], loaded[1][2][y['selected_array']], atol, rtol):
            first = {'selection_ordinal': index, 'reference_event_id': x['event_id'], 'candidate_event_id': y['event_id'],
                     'reference_origin': x['origin'], 'candidate_origin': y['origin'],
                     'reference_local_binding': x['local_binding'], 'candidate_local_binding': y['local_binding'],
                     'reference_arithmetic_contributors': x['arithmetic_contributors'],
                     'candidate_arithmetic_contributors': y['arithmetic_contributors'],
                     'latest_captured_prediction_values_equal': (None if any(v is None for v in latest_queries)
                                                                 else _same(*latest_queries, atol, rtol)),
                     'effective_settings_differ': x['settings'] != y['settings'],
                     'stage': 'selected_policy_output', 'causal_explanation': 'unknown'}
            break
    for item in loaded:
        _recheck(item[0], item[3])
    return {'kind': 'action_selection_comparison', 'status': 'error' if 'error' in (left['status'], right['status']) else 'unknown' if 'unknown' in (left['status'], right['status']) or len(a) != len(b) else 'pass',
            'first_selected_difference': first, 'difference_is_fault': False,
            'selection_counts': {'reference': len(a), 'candidate': len(b)},
            'alignment_basis': 'Captured selection-call ordinal only; no episode/time alignment asserted.',
            'downstream_association': {'status': 'unknown', 'reason': 'No cross-stage relation captured'},
            'reference': left, 'candidate': right, 'scope': SCOPE}
