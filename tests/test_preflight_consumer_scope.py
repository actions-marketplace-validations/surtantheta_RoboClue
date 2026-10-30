"""The full units manifest is qualified before any report presentation bound."""
from __future__ import annotations

import importlib
import json

import pytest
import test_mcp_r3_frontdoor as fixtures

from policy_doctor.artifacts import InputError

engine = importlib.import_module('policy_doctor.preflight')


def call(root, **kwargs):
    return engine.preflight(str(root), robot_command=fixtures.command(), video_headers=False,
                            groot_contract=False, **kwargs)


def test_raw_engine_omission_retains_legacy_units_and_no_added_cameras(tmp_path):
    root = fixtures.dataset(tmp_path)
    report = call(root)
    assert report['components']['units']['status'] == 'pass'
    assert 'consumer_version' not in report['components']['units']
    assert 'consumer_version' not in report
    assert 'camera_keys' not in report['components']


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_required_version_flag_is_strict_before_io(tmp_path, value):
    with pytest.raises(InputError, match='strict boolean'):
        engine.preflight(str(tmp_path / 'missing'), require_consumer_version=value)


def test_version_requires_actual_consumer_selector(tmp_path):
    with pytest.raises(InputError, match='actual robot_command or robot_config'):
        engine.preflight(str(tmp_path / 'missing'), lerobot_version='0.6.2')


def test_required_missing_version_is_unknown_but_recorded_units_are_retained(tmp_path):
    root = fixtures.dataset(tmp_path)
    report = call(root, require_consumer_version=True)
    units = report['components']['units']
    assert units['status'] == 'unknown'
    assert units['manifest']['features']
    assert units['consumer_version']['tested_profile'] is None
    assert units['consumer_version']['comparison_allowed'] is False
    assert report['consumer_version'] == units['consumer_version']
    assert report['components']['camera_keys']['status'] == 'unknown'
    assert report['components']['camera_keys']['compared'] is False


def test_explicit_version_qualifies_original_manifest_before_bound(tmp_path, monkeypatch):
    root = fixtures.dataset(tmp_path)
    bound = engine._bound_units
    observed = []

    def record(report):
        observed.append((report['consumer_version']['declared_version'],
                         report['consumer_version']['comparison_allowed'],
                         report['manifest']['features']['action']['coordinates']))
        return bound(report)

    monkeypatch.setattr(engine, '_bound_units', record)
    report = call(root, lerobot_version='0.6.2', require_consumer_version=True)
    assert observed and observed[0][0:2] == ('0.6.2', True)
    assert observed[0][2]
    assert report['components']['units']['status'] == 'pass'
    assert report['components']['camera_keys']['status'] == 'warning'
    assert report['coverage']['camera_keys_requested'] is True
    assert report['interface_scope']['joint order'] == 'not_assessed'


def test_unknown_consumer_does_not_hide_independent_bad_statistics(tmp_path):
    root = fixtures.dataset(tmp_path)
    (root / 'meta/stats.json').write_text(json.dumps({'action': {
        'mean': [1000.] * 6, 'std': [1.] * 6, 'min': [-100.] * 6, 'max': [100.] * 6}}))
    report = call(root, lerobot_version='0.6.3', require_consumer_version=True)
    assert report['status'] == 'error'
    assert report['components']['units']['status'] == 'unknown'
    assert report['components']['dataset_statistics']['status'] == 'error'


def test_required_mapping_is_version_aware_and_not_joined_twice(tmp_path):
    root = fixtures.dataset(tmp_path)
    report = engine.preflight(str(root), robot_command=fixtures.command(('up', 'side')),
        lerobot_version='0.6.2', require_consumer_version=True, groot_contract=False, video_headers=False)
    assert report['components']['camera_keys']['status'] == 'pass'
    codes = [finding['code'] for finding in report['findings'] if finding['component'] == 'camera_keys']
    assert codes == ['camera.keys_consistent']
    assert sum(report['counts'].values()) == len(report['findings'])
