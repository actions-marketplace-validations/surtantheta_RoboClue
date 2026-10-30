"""Plain terminal labels and copyable commands; machine evidence stays unchanged."""
from __future__ import annotations

import shlex


def unit_text(value):
    return {'RANGE_M100_100': '-100..100 across the calibrated range',
            'RANGE_0_100': '0..100 gripper scale', 'DEGREES': 'degrees',
            'RADIANS': 'radians', None: 'not determined'}.get(value, str(value))


def display_command(arguments):
    rendered = []
    previous = None
    for argument in arguments:
        value = str(argument)
        if previous == '--robot-command' and "'" in value and not any(c in value for c in '$`\\'):
            rendered.append('"' + value.replace('"', '\\"') + '"')
        else:
            rendered.append(shlex.quote(value))
        previous = str(argument)
    return ' '.join(rendered)


def statistics_scope(comparison, *, saved_self_consistent=False):
    """Describe measured comparisons without promoting partial evidence to a pass."""
    status = comparison.get('status')
    if status == 'error':
        return 'Saved numeric statistics disagree with the checked rows.'
    if status == 'pass' and comparison.get('compared_features'):
        metrics = ', '.join(str(value) for value in comparison.get('compared_metrics', [])) or 'supported values'
        return (f'Saved numeric statistics ({metrics}) match {comparison.get("rows", "the checked")} rows; '
                'image statistics were not recomputed.')
    if comparison.get('compared_features'):
        if any(item.get('metric') == 'std' for item in comparison.get('precision_unassessed', [])):
            return ('Some numeric statistics match the rows; source rounding leaves standard deviation unverified. '
                    'Image statistics were not recomputed.')
        return ('Some saved numeric statistics were compared with rows; other values remain unverified. '
                'Image statistics were not recomputed.')
    if saved_self_consistent:
        return 'Saved statistics are self-consistent; their values were not compared with dataset rows.'
    return 'Saved statistic values were not compared with dataset rows.'

