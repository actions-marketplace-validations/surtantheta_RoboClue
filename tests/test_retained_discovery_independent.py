"""Recipient-first discovery must preserve case boundaries before explicit replay."""
import json
import shutil

import pytest

from policy_doctor import investigation
from policy_doctor.demo import create_policy_demo
from policy_doctor.retained import retain_check


def _case(root, name='shared-case', *, metadata=False):
    source = root / (name + '-inputs')
    source.mkdir()
    case = root / name
    if metadata:
        create_policy_demo(source / 'demo')
        retain_check('policy_metadata', str(source / 'demo/checkpoint'), str(case),
                     dataset=str(source / 'demo/dataset'))
    else:
        (source / 'reference.csv').write_text('sample_id,action\n0,0.1\n1,0.2\n')
        (source / 'candidate.csv').write_text('sample_id,action\n0,0.1\n1,0.9\n')
        retain_check('recording_equivalence', str(source / 'candidate.csv'), str(case),
                     reference=str(source / 'reference.csv'), parameters={'alignment_key': 'sample_id'})
    # The recipient gets this folder, not its original caller's inputs.
    shutil.rmtree(source)
    return case


def _forbid_implicit_checks(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Discovery inspected or audited case-internal native inputs')
    for name in ('audit', 'audit_capture', 'inspect_rollout'):
        monkeypatch.setattr(investigation, name, forbidden)


@pytest.mark.parametrize('metadata', [False, True])
def test_recipient_given_case_root_is_offered_replay_without_internal_audit(tmp_path, monkeypatch, metadata):
    case = _case(tmp_path, metadata=metadata)
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(case))
    assert report['status'] == 'unknown'
    assert report['inventory']['retained_checks'] == [str(case)]
    assert not report['audits']
    for key in ('checkpoints', 'datasets', 'recordings'):
        assert report['inventory'][key] == []
    actions = [a for a in report['next_actions'] if a['tool'] == 'reproduce_incident_case']
    assert len(actions) == 1
    assert actions[0]['arguments'] == {'directory': str(case)}
    assert 'not' in actions[0]['purpose'].lower() or 'unverified' in actions[0]['purpose'].lower()


def test_nested_cases_remain_independent_without_invented_pairing(tmp_path, monkeypatch):
    shared = tmp_path / 'shared'
    shared.mkdir()
    one = _case(shared, 'one', metadata=True)
    two = _case(shared, 'two')
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(shared))
    assert report['status'] == 'unknown' and not report['audits']
    assert sorted(report['inventory']['retained_checks']) == sorted([str(one), str(two)])
    actions = [a for a in report['next_actions'] if a['tool'] == 'reproduce_incident_case']
    assert sorted(a['arguments']['directory'] for a in actions) == sorted([str(one), str(two)])
    assert not report['inventory']['checkpoints'] and not report['inventory']['recordings']
    assert report['recording_comparison'] is None


def test_symlink_nested_case_is_not_followed(tmp_path, monkeypatch):
    actual = _case(tmp_path, 'outside')
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'alias').symlink_to(actual, target_is_directory=True)
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(shared))
    assert report['status'] == 'unknown'
    assert not report['inventory']['retained_checks']
    assert report['coverage']['symlinks_skipped'] >= 1
    assert not report['audits']


@pytest.mark.parametrize('kind', ['unknown_case_kind', None])
def test_malformed_case_marker_does_not_expose_copied_inputs(tmp_path, monkeypatch, kind):
    case = _case(tmp_path, metadata=True)
    manifest = json.loads((case / 'manifest.json').read_text())
    manifest['kind'] = kind
    (case / 'manifest.json').write_text(json.dumps(manifest))
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(case))
    assert report['status'] == 'unknown' and not report['audits']
    assert not report['inventory']['retained_checks']
    assert not report['inventory']['checkpoints'] and not report['inventory']['datasets']
    assert report['intake_problems'] or report['coverage']['truncated']


def test_oversize_case_manifest_stays_opaque_and_unverified(tmp_path, monkeypatch):
    case = _case(tmp_path, metadata=True)
    original = (case / 'manifest.json').read_text()
    (case / 'manifest.json').write_text(original + ' ' * (2 * 1024 * 1024))
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(case))
    assert report['status'] == 'unknown' and not report['audits']
    assert not report['inventory']['checkpoints'] and not report['inventory']['datasets']
    assert report['coverage']['truncated'] or report['intake_problems']


def test_bounded_parent_inventory_never_claims_discovery_complete(tmp_path, monkeypatch):
    shared = tmp_path / 'shared'
    shared.mkdir()
    _case(shared, 'one')
    _case(shared, 'two')
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(shared), max_files=1, max_depth=0)
    assert report['status'] == 'unknown'
    assert report['coverage']['truncated'] is True
    assert not report['audits']


def test_symlink_manifest_is_not_read_and_internal_inputs_stay_opaque(tmp_path, monkeypatch):
    case = _case(tmp_path, metadata=True)
    original_manifest = tmp_path / 'external-manifest.json'
    (case / 'manifest.json').replace(original_manifest)
    (case / 'manifest.json').symlink_to(original_manifest)
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(case))
    assert report['status'] == 'unknown' and not report['audits']
    assert not report['inventory']['retained_checks']
    assert not report['inventory']['checkpoints'] and not report['inventory']['datasets']
    assert report['intake_problems'] or report['coverage']['symlinks_skipped']


@pytest.mark.parametrize('metadata', [False, True])
def test_missing_ready_is_damaged_case_not_an_ordinary_run(tmp_path, monkeypatch, metadata):
    case = _case(tmp_path, metadata=metadata)
    (case / 'READY.json').unlink()
    _forbid_implicit_checks(monkeypatch)
    report = investigation.investigate_run(str(case))
    assert report['status'] == 'unknown' and not report['audits']
    assert not report['inventory']['checkpoints'] and not report['inventory']['datasets']
    assert not report['inventory']['recordings']
    assert report['intake_problems'] or any(
        action['tool'] == 'reproduce_incident_case' for action in report['next_actions'])
