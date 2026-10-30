"""Read-only consumer requirements over saved native ACT query-call coverage.

Cadence is an explicit acceptance criterion, not a diagnosis of an ACT bug or
proof of the age/identity of the selected action's contributing query.
"""
from .artifacts import InputError
from .selection import _analyze, _load, _recheck, _tolerances

SCOPE = (
    'Capture-wide observed selection-call cadence at policy_output only. A query selection counts as 1; '
    'resets are neither queries nor selections and preserve observed counts. Mid-run missing query '
    'history supplies only a lower bound. No guarantee outside this saved window, no environment '
    'episode, physical time, query-contributor identity, issued command, or hardware claim. '
    'The maximum gap is an explicit consumer requirement, not a universal ACT configuration rule.'
)


def audit_query_cadence(capture, *, max_query_gap, atol=1e-6, rtol=1e-5):
    """Check an explicit maximum selections-per-observed-query requirement.

    Read one bounded immutable selection snapshot and replay its native numeric
    transitions. Call coverage and numeric replay must both qualify a cadence
    verdict; local identity witnesses are independent and are not prerequisites.
    A missing initial query remains unknown unless a lower-bound violation is
    already provable. Resets do not hide observed selection-call distance.
    """
    if type(max_query_gap) is not int or not 1 <= max_query_gap <= 1_000_000:
        raise InputError('max_query_gap must be an integer between 1 and 1000000, excluding bool')
    _tolerances(atol, rtol)
    root, manifest, arrays, hashes, supported = _load(capture)
    analyzed = _analyze(manifest, arrays, supported, atol, rtol)
    coverage = manifest.get('coverage')
    coverage_supported = (
        type(coverage) is dict
        and coverage.get('target') == 'one_policy_on_owner_thread'
        and coverage.get('complete_target_call_history') is True
        and coverage.get('external_state_mutations') == 'untracked'
    )
    selection_count = sum(e['type'] == 'select' for e in analyzed)
    errors = [e for e in analyzed if e['numeric_replay']['status'] == 'error']
    numeric_status = (
        'error' if errors else 'unknown'
        if not selection_count or any(e['numeric_replay']['status'] != 'pass' for e in analyzed)
        else 'pass'
    )
    qualified = supported and coverage_supported and numeric_status == 'pass'
    if not supported:
        qualification_reason = 'Native source is unsupported; query-call interpretation is unqualified.'
    elif not coverage_supported:
        qualification_reason = 'Complete target-policy owner-thread query-call coverage is absent or incomplete.'
    elif numeric_status != 'pass':
        qualification_reason = 'Native numeric replay does not pass; cadence cannot be certified from this packet.'
    else:
        qualification_reason = None

    events = []
    gap = None
    prefix_streak = 0
    unknown_prefix = 0
    last_query_id = None
    last_query_event = None
    first = None
    query_count = 0
    reset_count = 0
    for original, analyzed_event in zip(manifest['events'], analyzed, strict=True):
        event_id = original['event_id']
        if original['type'] == 'reset':
            reset_count += 1
            events.append({'event_id': event_id, 'type': 'reset',
                           'observed_query_count': 0, 'status': 'pass' if qualified else 'unknown',
                           'reason': qualification_reason or 'Reset preserves capture-wide selection/query distance.'})
            continue
        predictions = original['predictions']  # Strict shape/count/IDs already validated by _analyze.
        if predictions:
            query_count += 1
            last_query_id = predictions[0]['query_id']
            last_query_event = event_id
            gap = 1
        elif gap is not None:
            gap += 1
        else:
            prefix_streak += 1
            unknown_prefix += 1
        lower = gap if gap is not None else prefix_streak + 1
        violates = lower > max_query_gap
        verdict = ('unknown' if not qualified else 'error' if violates else
                   'unknown' if gap is None else 'pass')
        if qualification_reason:
            reason = qualification_reason
        elif violates:
            reason = 'Observed selection distance exceeds the consumer maximum.'
        elif gap is None:
            reason = 'Earlier query is outside this capture; only a lower bound is known.'
        else:
            reason = 'Observed query distance satisfies the consumer maximum.'
        item = {'event_id': event_id, 'type': 'select',
                'selection_ordinal': analyzed_event['selection_ordinal'],
                'observed_query_count': len(predictions), 'query_id': last_query_id,
                'query_event_id': last_query_event, 'gap': gap, 'gap_lower_bound': lower,
                'status': verdict, 'reason': reason}
        events.append(item)
        if qualified and violates and first is None:
            evidence = [
                {'path': str(root / 'selection.json'), 'sha256': hashes['selection.json'],
                 'pointer': f'/events/{event_id}/predictions'},
                {'path': str(root / 'selection.json'), 'sha256': hashes['selection.json'],
                 'pointer': '/coverage'},
                {'path': str(root / 'selection.json'), 'sha256': hashes['selection.json'],
                 'pointer': '/source'},
                {'path': str(root / 'arrays.npz'), 'sha256': hashes['arrays.npz'],
                 'array': original['selected']},
            ]
            if last_query_event is not None:
                evidence.append({'path': str(root / 'selection.json'), 'sha256': hashes['selection.json'],
                                 'pointer': f'/events/{last_query_event}/predictions/0'})
            first = {key: item[key] for key in ('event_id', 'selection_ordinal', 'query_id', 'query_event_id', 'gap', 'gap_lower_bound')}
            first.update(limit=max_query_gap, evidence=evidence)
    criterion_status = ('unknown' if not qualified or not selection_count else
                        'error' if first else 'unknown' if unknown_prefix else 'pass')
    if qualification_reason:
        reason = qualification_reason
    elif not selection_count:
        reason = 'No selection calls were captured.'
    elif first:
        reason = 'An observed exact distance or lower bound exceeds the explicit maximum.'
    elif unknown_prefix:
        reason = 'Initial query history is absent for part of the captured window.'
    else:
        reason = 'Every captured selection meets the explicit maximum.'
    # Expose only recognized, bounded source/coverage fields, not arbitrary caller metadata.
    source = {key: value[:256] for key in ('repository', 'revision', 'modeling_sha256', 'config_sha256')
              if isinstance((value := manifest['source'].get(key)), str)}
    report = {
        'kind': 'query_cadence_audit',
        'status': 'error' if numeric_status == 'error' else criterion_status,
        'consumer_requirement': {'max_query_gap': max_query_gap, 'domain': 'policy_output', 'unit': 'selection_calls'},
        'cadence_criterion': {'status': criterion_status, 'reason': reason, 'selection_count': selection_count,
                              'query_count': query_count, 'reset_count': reset_count,
                              'unknown_prefix_selections': unknown_prefix, 'first_violation': first},
        'numeric_replay_status': numeric_status,
        'first_numeric_contradiction': ({'event_id': errors[0]['event_id'],
                                         'path': str(root / 'selection.json'),
                                         'pointer': f"/events/{errors[0]['event_id']}",
                                         'sha256': hashes['selection.json']} if errors else None),
        'query_lineage_status': 'unknown',
        'downstream_association': {'status': 'unknown', 'reason': 'No issued-command boundary association is supplied by query cadence.'},
        'source_supported': supported, 'coverage_supported': coverage_supported,
        'source': source, 'coverage': {
            'target': 'one_policy_on_owner_thread' if coverage_supported else 'unsupported_or_missing',
            'complete_target_call_history': coverage_supported,
            'external_state_mutations': 'untracked',
        },
        'events': events, 'selection_count': selection_count,
        'tolerances': {'atol': atol, 'rtol': rtol}, 'scope': SCOPE,
        'artifacts': [{'path': str(root / name), 'sha256': digest} for name, digest in sorted(hashes.items())],
    }
    _recheck(root, hashes)
    return report
