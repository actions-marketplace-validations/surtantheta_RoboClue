"""Retain a declared runtime requirement without promoting unknown lineage."""

# Imported pytest fixtures are intentionally injected by name.
# ruff: noqa: F811
import hashlib
import json
import shutil

import numpy as np
import pytest
from test_selection import capture, event, rewrite_manifest  # noqa: F401

from policy_doctor.artifacts import InputError
from policy_doctor.capture import audit_capture
from policy_doctor.cli import human, main
from policy_doctor.incidents import check_incident_candidate, export_incident_case, reproduce_incident_case
from policy_doctor.reports import markdown_report


def native_fixture(capture, name, *, steps=2, offset=0.):
    chunk = np.arange(6, dtype=np.float32).reshape(1, 3, 2) + offset
    arrays = {'q0': chunk, 'q1': chunk + 10, 'a': chunk[:, 0],
              'b': chunk[:, 1] if steps == 2 else chunk[:, 0] + 10}
    events = [event(0, [], 'a', ['b'] if steps == 2 else [], [{'query_id': 0, 'array': 'q0'}]),
              event(1, ['b'] if steps == 2 else [], 'b', [],
                    [] if steps == 2 else [{'query_id': 1, 'array': 'q1'}])]
    for e in events:
        e['settings']['n_action_steps'] = steps
        e['pre_state']['queue_maxlen'] = e['post_state']['queue_maxlen'] = steps
    return capture(events, arrays, name)


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def test_retained_cadence_relocates_and_applies_same_requirement_to_new_values(capture, tmp_path):
    before = native_fixture(capture, 'before')
    corrected = native_fixture(capture, 'corrected', steps=1)
    healthy = native_fixture(capture, 'healthy', steps=1, offset=40.)
    recurrence = native_fixture(capture, 'recurrence', offset=40.)
    unknown = native_fixture(capture, 'unknown', steps=1, offset=80.)
    rewrite_manifest(unknown, lambda d: d['coverage'].update(complete_target_call_history=False))
    originals = {str(p): snapshot(p) for p in (before, corrected, healthy, recurrence, unknown)}
    case = tmp_path / 'retained'
    result = export_incident_case('act_query_cadence', before, case, {'max_query_gap': 1}, corrected=corrected,
        controls=[{'name': 'incomplete', 'source': unknown, 'parameters': {'max_query_gap': 1}, 'expected_status': 'unknown'}])
    assert result['status'] == 'pass'
    assert result['checks']['before']['cadence_criterion']['first_violation']['selection_ordinal'] == 1
    assert result['checks']['after']['cadence_criterion']['status'] == 'pass'
    assert result['checks']['after']['query_lineage_status'] == 'unknown'
    for p in (before, corrected, healthy, recurrence, unknown):
        assert snapshot(p) == originals[str(p)]
    manifest = json.loads((case / 'manifest.json').read_text())
    assert manifest['parameters'] == {'max_query_gap': 1, 'atol': 1e-6, 'rtol': 1e-5}
    assert 'cadence.py' in manifest['verifier']['module_sha256']
    moved = tmp_path / 'colleague'
    shutil.move(case, moved)
    shutil.rmtree(before)
    shutil.rmtree(corrected)
    shutil.rmtree(unknown)
    assert reproduce_incident_case(moved)['status'] == 'pass'
    retained = snapshot(moved)
    for candidate, status in ((healthy, 'pass'), (recurrence, 'error')):
        result = check_incident_candidate(moved, candidate)
        assert result['status'] == status and result['candidate_evaluated']
        assert result['parameters']['max_query_gap'] == 1
    assert snapshot(moved) == retained


def test_case_rejects_weakened_controls_and_numeric_fault_as_consumer_violation(capture, tmp_path):
    before = native_fixture(capture, 'before')
    after = native_fixture(capture, 'after', steps=1)
    with pytest.raises(InputError, match='same explicit requirement'):
        export_incident_case('act_query_cadence', before, tmp_path / 'weak', {'max_query_gap': 1}, corrected=after,
            controls=[{'name': 'weak', 'source': before, 'parameters': {'max_query_gap': 2}, 'expected_status': 'pass'}])
    assert not (tmp_path / 'weak').exists()
    rewrite_manifest(before, lambda d: d['events'][0].update(selected='b'))
    with pytest.raises(InputError, match='observed requirement violation'):
        export_incident_case('act_query_cadence', before, tmp_path / 'numeric', {'max_query_gap': 1}, corrected=after)
    assert not (tmp_path / 'numeric').exists()


def test_cli_and_readable_reports_for_explicit_requirement(capture, tmp_path, capsys):
    before = native_fixture(capture, 'before')
    after = native_fixture(capture, 'after', steps=1)
    assert main(['audit-capture', str(before), '--max-query-gap', '1', '--json']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['cadence_criterion']['status'] == 'error'
    assert 'Maximum selection calls per query: 1' in human(report)
    assert 'Observed native query cadence' in markdown_report(report)
    case = tmp_path / 'case'
    assert main(['create-case', 'act-query-cadence', str(before), str(case), '--corrected', str(after),
                 '--max-query-gap', '1', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'pass'
    assert main(['check-case', str(case), str(after), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['parameters']['max_query_gap'] == 1
    # Existing broad capture report keeps lineage unknown without explicit criterion.
    assert audit_capture(str(after))['status'] == 'unknown'


@pytest.mark.parametrize('value', [True, False, 0, -1, 1.0, '1', 1000001, None])
def test_case_requires_explicit_strict_query_gap(capture, tmp_path, value):
    with pytest.raises(InputError, match='explicit integer'):
        export_incident_case('act_query_cadence', tmp_path, tmp_path / 'case', {'max_query_gap': value}, corrected=tmp_path)


def test_legal_queue_setting_is_not_universally_a_fault(capture, tmp_path):
    before = native_fixture(capture, 'before')
    after = native_fixture(capture, 'after', steps=1)
    assert audit_capture(str(before), max_query_gap=2)['status'] == 'pass'
    with pytest.raises(InputError, match='observed requirement violation'):
        export_incident_case('act_query_cadence', before, tmp_path / 'not-incident', {'max_query_gap': 2}, corrected=after)
    with pytest.raises(InputError, match='Corrected evidence must pass'):
        export_incident_case('act_query_cadence', before, tmp_path / 'uncorrected', {'max_query_gap': 1}, corrected=before)
    assert not (tmp_path / 'not-incident').exists() and not (tmp_path / 'uncorrected').exists()


def test_verifier_drift_blocks_candidate_read_and_incomplete_later_is_unknown(capture, tmp_path):
    before = native_fixture(capture, 'before')
    after = native_fixture(capture, 'after', steps=1)
    late = native_fixture(capture, 'late', steps=1, offset=90.)
    rewrite_manifest(late, lambda d: d['coverage'].update(complete_target_call_history=False))
    case = tmp_path / 'case'
    export_incident_case('act_query_cadence', before, case, {'max_query_gap': 1}, corrected=after)
    result = check_incident_candidate(case, late)
    assert result['candidate_evaluated'] and result['status'] == 'unknown'
    manifest_path = case / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['verifier']['engine_version'] = 'changed-engine'
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    (case / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))
    result = check_incident_candidate(case, tmp_path / 'not-existing-candidate')
    assert result['status'] == 'unknown' and not result['candidate_evaluated']


def test_other_capture_family_cannot_silently_ignore_query_requirement(tmp_path):
    with pytest.raises(InputError, match='requires an ACT selection capture'):
        audit_capture(str(tmp_path), max_query_gap=1)
