"""Mechanistic controls from LeRobot #4006, not original model reproduction."""
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from policy_doctor.diagnostics import audit


@pytest.mark.parametrize('stats_dim,expected', [(4, 'unknown'), (2, 'error')])
def test_custom_padding_distinguishes_unknown_transform_from_bad_statistics(case, stats_dim, expected):
    for side in ('pre', 'post'):
        path = case['checkpoint'] / f'policy_{side}processor.json'
        config = json.loads(path.read_text())
        config['steps'] = [s for s in config['steps'] if 'normalizer' in s['registry_name']]
        norm = config['steps'][0]
        for key in ('action', 'observation.state'):
            norm['config']['features'][key]['shape'] = [4]
        custom = {'registry_name': 'custom_pad' if side == 'pre' else 'custom_unpad', 'config': {}}
        config['steps'].insert(0 if side == 'pre' else 1, custom)
        case['put'](path, config)
        save_file({f'{key}.{stat}': np.full(stats_dim, value)
                   for key in ('action', 'observation.state') for stat, value in [('mean', 0.), ('std', 1.)]},
                  str(case['checkpoint'] / norm['state_file']))
    report = audit(str(case['checkpoint']), str(case['dataset']), str(case['deployment']))
    assert report['status'] == expected
    shape_findings = [f for f in report['findings'] if f['code'] == 'processor.feature_shape']
    assert shape_findings and all(f['status'] == 'unknown' for f in shape_findings)
    stats_errors = [f for f in report['findings'] if f['code'] == 'normalization.statistics' and f['status'] == 'error']
    assert bool(stats_errors) == (expected == 'error')
