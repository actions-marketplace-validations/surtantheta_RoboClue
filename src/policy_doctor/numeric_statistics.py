"""Streaming comparison of saved native numeric summaries with complete row cohorts.

Only mean/std/min/max are compared. Native image sampling and approximate
quantile definitions are not replaced by a guessed exact-quantile calculation.
"""
from __future__ import annotations

import numpy as np

METRICS = ('mean', 'std', 'min', 'max')
RTOL, ATOL = 1e-4, 1e-6
MAX_FEATURES, MAX_DIMENSIONS = 256, 65536


class NumericStatistics:
    def __init__(self, info, features, *, enabled=True):
        self.enabled = enabled
        self.rows = 0
        self.states = {}
        self.skipped = {}
        self.unassessed_metrics = {}
        self.stats = info.get('stats') if isinstance(info.get('stats'), dict) else {}
        self.supported = str(info.get('codebase_version', '')).startswith(('v2.', 'v3.'))
        dimensions = 0
        if not enabled or not self.supported:
            return
        for key, feature in features.items():
            if feature.get('type') == 'VISUAL' or key.startswith('observation.images.'):
                continue
            entry = self.stats.get(key)
            if not isinstance(entry, dict):
                continue
            dtype = feature.get('dtype')
            try:
                numeric = np.dtype(dtype).kind in 'biuf' if dtype is not None else feature.get('type') in ('STATE', 'ACTION')
            except TypeError:
                numeric = False
            if not numeric:
                continue
            shape = feature.get('shape')
            if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
                self.skipped[key] = 'Only native numeric scalar/vector feature summaries are compared.'
                continue
            width = shape[0]
            if len(self.states) >= MAX_FEATURES or dimensions + width > MAX_DIMENSIONS:
                self.skipped[key] = 'The additional streaming statistics dimension budget was exceeded.'
                continue
            metrics = {}
            for name in METRICS:
                if name not in entry:
                    continue
                try:
                    if isinstance(entry[name], list) and any(type(v) not in (int, float) for v in entry[name]):
                        raise ValueError('not a native numeric summary vector')
                    raw = np.asarray(entry[name])
                    if raw.dtype.kind not in 'iuf':
                        raise ValueError('not a native numeric summary vector')
                    value = raw.astype(np.float64)
                except (ValueError, TypeError, OverflowError):
                    self.unassessed_metrics.setdefault(key, []).append(name)
                    continue
                if value.shape == (width,) and np.isfinite(value).all():
                    metrics[name] = value
                else:
                    self.unassessed_metrics.setdefault(key, []).append(name)
            if not metrics:
                self.skipped[key] = 'No supported finite saved mean/std/min/max vector was available.'
                continue
            dimensions += width
            precision = np.dtype(dtype) if dtype is not None else np.dtype('float64')
            float32 = precision.kind == 'f' and precision.itemsize <= 4
            self.states[key] = {'count': 0, 'width': width, 'saved': metrics, 'mean': np.zeros(width),
                                'm2': np.zeros(width), 'min': np.full(width, np.inf), 'max': np.full(width, -np.inf),
                                'native_float32': float32}

    @property
    def columns(self):
        return tuple(self.states)

    def update(self, record):
        self.rows += 1
        for key, state in list(self.states.items()):
            if key in self.skipped:
                continue
            try:
                values = np.asarray(record[key], dtype=np.float64)
                if values.ndim == 0 and state['width'] == 1:
                    values = values.reshape(1)
                if values.shape != (state['width'],) or not np.isfinite(values).all():
                    raise ValueError('not a finite matching vector')
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    n = state['count'] + 1
                    delta = values - state['mean']
                    mean = state['mean'] + delta / n
                    state['m2'] += delta * (values - mean)
                    state['mean'] = mean
                    state['min'] = np.minimum(state['min'], values)
                    state['max'] = np.maximum(state['max'], values)
                    state['count'] = n
            except (KeyError, ValueError, TypeError, OverflowError, FloatingPointError):
                self.skipped[key] = 'Not every scanned row supplied a finite supported value, or accumulation overflowed.'

    def finish(self, *, complete, declared_rows):
        compared, mismatches, metrics = [], [], set()
        precision_limits, precision_count = [], 0
        mismatch_count = 0
        complete = complete and self.rows > 0 and (type(declared_rows) is not int or declared_rows == self.rows)
        if self.enabled and self.supported and complete:
            for key, state in self.states.items():
                if key in self.skipped or state['count'] != self.rows:
                    continue
                observed = {'mean': state['mean'], 'std': np.sqrt(np.maximum(0., state['m2'] / state['count'])),
                            'min': state['min'], 'max': state['max']}
                compared.append(key)
                for name, saved in state['saved'].items():
                    metrics.add(name)
                    bad = np.flatnonzero(~np.isclose(saved, observed[name], rtol=RTOL, atol=ATOL))
                    if len(bad) and state['native_float32'] and name in ('mean', 'std'):
                        # Native RunningQuantileStats uses float32 E[x²] - E[x]².
                        # A conservative standard summation envelope avoids
                        # labelling that arithmetic error as a stale cohort.
                        # This produces UNKNOWN, never a widened-tolerance PASS.
                        u = np.finfo(np.float32).eps / 2
                        gamma = (state['count'] + 2) * u / (1 - (state['count'] + 2) * u)
                        magnitude = np.maximum(np.abs(state['min']), np.abs(state['max']))
                        with np.errstate(over='ignore', invalid='ignore'):
                            bound = gamma * magnitude if name == 'mean' else 8 * gamma * magnitude ** 2
                            difference = (np.abs(saved - observed[name]) if name == 'mean' else
                                          np.abs(saved ** 2 - observed[name] ** 2))
                        possible = np.isfinite(bound) & np.isfinite(difference) & (difference <= bound)
                        if name == 'std':
                            possible &= saved >= 0
                        uncertain = bad[possible[bad]]
                        if len(uncertain):
                            precision_count += 1
                            self.unassessed_metrics.setdefault(key, [])
                            if name not in self.unassessed_metrics[key]:
                                self.unassessed_metrics[key].append(name)
                            if len(precision_limits) < 32:
                                precision_limits.append({'feature': key, 'metric': name,
                                    'indices': uncertain[:16].tolist(), 'indices_total': len(uncertain),
                                    'reason': 'Native float32 summary rounding can explain this difference. '
                                              'Recompute numeric statistics with higher precision to resolve it.'})
                            bad = bad[~possible[bad]]
                    if len(bad):
                        mismatch_count += 1
                        selected = bad[:16]
                        if len(mismatches) >= 32:
                            continue
                        mismatches.append({'feature': key, 'metric': name, 'indices': selected.tolist(),
                            'indices_total': len(bad), 'saved': saved[selected].tolist(),
                            'observed': observed[name][selected].tolist(),
                            'message': f'{key}: saved {name} disagrees with the complete {self.rows}-row cohort '
                                       f'on {len(bad)} coordinate(s). Recompute statistics for the intended retained rows; '
                                       'changing only the saved count does not repair these values.'})
        status = 'error' if mismatch_count else 'pass' if compared and not self.skipped and not self.unassessed_metrics else 'unknown'
        reason = ('Explicitly disabled.' if not self.enabled else 'Only native LeRobot v2/v3 numeric summaries are supported.'
                  if not self.supported else 'The full declared numeric row cohort was not read.' if not complete else
                  'Some numeric summaries were unsupported or unresolved at native float32 precision.'
                  if self.skipped or self.unassessed_metrics else
                  'No supported saved numeric summaries were available.' if not compared else None)
        return {'status': status, 'rows': self.rows, 'complete_cohort': complete,
                'compared_features': sorted(compared), 'compared_metrics': sorted(metrics),
                'skipped': self.skipped, 'unassessed_metrics': self.unassessed_metrics,
                'precision_unassessed': precision_limits, 'precision_unassessed_count': precision_count,
                'reason': reason, 'mismatches': mismatches,
                'mismatch_count': mismatch_count, 'mismatches_truncated': mismatch_count > len(mismatches),
                'tolerance': {'rtol': RTOL, 'atol': ATOL},
                'scope': 'Complete native numeric rows versus supported saved mean/std/min/max only. '
                         'Population standard deviation; tolerance allows native float32 aggregation rounding. '
                         'No image/video pixels, quantiles, robot identity or training correctness are established.'}
