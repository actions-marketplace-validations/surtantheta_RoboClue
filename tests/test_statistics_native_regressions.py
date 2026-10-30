"""Production artifact checks for independently demonstrated native normalization counterexamples."""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_statistics_evidence import SAVED, dataset_stats, found, matching, run, save

from policy_doctor.artifacts import load_dataset
from policy_doctor.diagnostics import dataset_statistics_findings


def configure(case, mode, eps=1e-8):
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        p = case['checkpoint'] / name
        obj = json.loads(p.read_text())
        for step in obj['steps']:
            if 'normalization' in step.get('registry_name', '') or 'normaliz' in step.get('registry_name', ''):
                step['config']['norm_map'] = {'STATE': mode, 'ACTION': mode}
                step['config']['eps'] = eps
        p.write_text(json.dumps(obj))


@pytest.mark.parametrize('mode', ['MEAN_STD', 'MIN_MAX', 'QUANTILES', 'QUANTILE10'])
@pytest.mark.parametrize('spread', [0., 5e-7])
def test_action_inverse_reports_actual_affine_scale_not_impossibility(case, mode, spread):
    configure(case, mode)
    arrays = {k: np.array(v) for k, v in SAVED.items()}
    pairs = {'MEAN_STD': ('mean','std'), 'MIN_MAX': ('min','max'), 'QUANTILES': ('q01','q99'), 'QUANTILE10': ('q10','q90')}
    lo, hi = pairs[mode]
    arrays.update({f'action.{lo}': np.array([0., 0.]), f'action.{hi}': np.array([spread, spread])})
    save_file(arrays, case['checkpoint'] / 'post.safetensors')
    result = found(run(case), 'normalization.constant_dimension')
    assert result
    hit = result[0]
    assert 'never move' not in hit['message'] and 'whatever the policy outputs' not in hit['message']
    expected = spread if mode == 'MEAN_STD' else (spread or 1e-8) / 2
    assert hit['evidence'][0]['inverse_scales'] == [expected, expected]
    assert 'No model-output bound' in hit['message']


@pytest.mark.parametrize('policy_type', ['evo1', 'my_plugin_policy'])
def test_unmodeled_factory_never_certifies_saved_degenerate_runtime_effect(case, policy_type):
    config = case['checkpoint'] / 'config.json'
    obj = json.loads(config.read_text())
    obj['type'] = policy_type
    config.write_text(json.dumps(obj))
    save(case, 'pre', observation_state_std=[0., .25])
    dataset_stats(case, matching())
    report = run(case)
    assert not found(report, 'normalization.constant_input')
    assert not found(report, 'normalization.constant_dimension')
    hits = found(report, 'normalization.dataset_statistics')
    assert hits and all(f['status'] == 'unknown' for f in hits)
    assert all('does not model' in f['message'] for f in hits)


def test_custom_eps_one_zero_input_has_unit_gain_without_forced_retraining(case):
    configure(case, 'MEAN_STD', eps=1.)
    save(case, 'pre', observation_state_std=[0., .25])
    hit = found(run(case), 'normalization.constant_input')[0]
    assert hit['status'] == 'pass' and hit['evidence'][0]['forward_gains'] == [1.]
    assert 'exactly' not in hit['message'] and 'otherwise retrain' not in hit['remedy']


def test_dataset_only_stats_never_assume_active_mode_or_epsilon(case):
    stats = matching()
    stats['action'].update(std=[0., .25], min=[1., 0.], max=[1., 1.])
    stats['observation.state'].update(std=[0., .25], min=[10., 0.], max=[10., 1.], q01=[10.,0.], q99=[10.,1.])
    dataset_stats(case, stats)
    hits = dataset_statistics_findings(load_dataset(case['dataset']))
    assert hits
    for hit in hits:
        assert 'active normalization mode and epsilon are not established' in hit['message']
        assert 'million' not in hit['message'] and 'always commands' not in hit['message']
    assert any('IDENTITY' in h['message'] for h in hits)


def test_dataset_small_positive_spread_is_distinct_from_exact_zero(case):
    stats = matching()
    stats['action'].update(mean=[2.5e-7, .5], std=[5e-7, .25], min=[0.,0.], max=[5e-7,1.])
    dataset_stats(case, stats)
    hits = dataset_statistics_findings(load_dataset(case['dataset']))
    assert not [f for f in hits if f['code'] == 'dataset.constant_dimension']
    assert [f for f in hits if f['code'] == 'dataset.small_spread'][0]['indices'] == [0]


@pytest.mark.parametrize('change', ['malformed', 'width', 'reversed_quantiles'])
def test_direct_statistics_helper_never_silently_drops_uninterpretable_vectors(case, change):
    stats = matching()
    if change == 'malformed':
        stats['action']['std'] = ['bad', 1.]
    elif change == 'width':
        stats['action']['std'] = [1., 2., 3.]
    else:
        stats['action'].update(q01=[2., 0.], q99=[1., 1.])
    dataset_stats(case, stats)
    hits = dataset_statistics_findings(load_dataset(case['dataset']))
    assert any(f['status'] in ('unknown','warning') for f in hits)
    assert any(f['code'] in ('dataset.statistics_unassessed','dataset.statistics_invalid') for f in hits)


def test_tiny_real_drift_is_not_masked_by_absolute_allclose_tolerance(case):
    save(case, 'post', action_mean=[0., .5], action_std=[0., .25])
    stats = matching()
    stats['action']['mean'] = [5e-7, .5]
    stats['action']['std'] = [5e-7, .25]
    dataset_stats(case, stats)
    hit = next(f for f in found(run(case), 'normalization.dataset_statistics') if f['message'].startswith('post:'))
    assert hit['status'] == 'warning' and hit['evidence'][0]['constant_in_checkpoint_but_not_dataset'] == [0]


@pytest.mark.parametrize('dataset_std,expected_status', [(3., 'pass'), (7., 'warning')])
def test_custom_epsilon_comparison_describes_effective_normalized_coordinates(case, dataset_std, expected_status):
    configure(case, 'MEAN_STD', eps=1.)
    stats = matching()
    stats['observation.state']['std'] = [dataset_std, .25]
    dataset_stats(case, stats)
    hit = next(f for f in found(run(case), 'normalization.dataset_statistics') if f['message'].startswith('pre:'))
    assert hit['status'] == expected_status
    assert hit['evidence'][0]['scales'] == pytest.approx([dataset_std / 3., .25 / 1.25], abs=1e-4)
    assert "own data sits at 0 and x1" not in hit['message'] and 'wider than the saved' not in hit['message']
    assert 'nominal coordinates' in hit['message'] and 'normalized scale' in hit['message']
