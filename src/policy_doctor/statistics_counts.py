"""Check native numeric summary cohort sizes without reading dataset rows.

Native image/video summaries may sample frames, so their counts are deliberately
excluded. Missing counts are not invented, and no row-statistic agreement is
implied by matching metadata counts.
"""
from __future__ import annotations

import math

_NUMERIC_DTYPES = frozenset({'float', 'float16', 'float32', 'float64', 'bfloat16',
    'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'})


def statistics_count_findings(info, stats, *, info_artifact=None, stats_artifact=None):
    """Compare supplied one-cohort numeric counts with native total_frames.

    This is a metadata contradiction check, not a claim that saved means/stds
    agree with observed rows. Unsupported/missing count representations remain
    unassessed; callers retain their normal statistics coverage.
    """
    if not isinstance(info, dict) or not isinstance(stats, dict):
        return []
    total = info.get('total_frames')
    if type(total) is not int or total < 0:
        return []
    features = info.get('features')
    if not isinstance(features, dict):
        return []
    findings = []
    for key, spec in sorted(features.items()):
        if (not isinstance(spec, dict) or str(spec.get('dtype', '')).lower() not in _NUMERIC_DTYPES
                or key.startswith('observation.images.')):
            continue
        entry = stats.get(key)
        if not isinstance(entry, dict) or 'count' not in entry:
            continue
        count = entry['count']
        if not isinstance(count, list) or len(count) != 1 or type(count[0]) not in (int, float):
            continue
        value = count[0]
        if (type(value) is float and not math.isfinite(value)) or value < 0 or value != int(value):
            mismatch, reason = True, 'is not a finite non-negative integer'
        else:
            mismatch, reason = int(value) != total, f'describes {int(value)} rows, but meta/info.json declares {total}'
        if not mismatch:
            continue
        pointer = '/' + key.replace('~', '~0').replace('/', '~1') + '/count'
        def evidence(artifact, pointer, value):
            artifact = artifact or {}
            return {'path': artifact.get('path'), 'sha256': artifact.get('sha256'),
                    'pointer': pointer, 'value': value}
        findings.append({'code': 'dataset.statistics_count_mismatch', 'status': 'error',
            'feature': key, 'saved_count': count, 'declared_total_frames': total,
            'message': f'{key}: saved statistics count {reason}. These numeric summaries do not match '
                       'the declared recording size; they may still include excluded episodes.',
            'remedy': 'Recompute statistics for the retained rows in a separate owned copy, or use '
                      'roboclue check DATASET --write-copy NEWDIR for a supported recovery. '
                      'Do not change the count alone: saved means and variances may also be stale.',
            'evidence': [evidence(stats_artifact, pointer, count),
                         evidence(info_artifact, '/total_frames', total)],
            'scope': 'Native numeric cohort size only. Sampled image/video counts are excluded; '
                     'matching counts do not verify saved statistic values against rows.'})
    return findings
