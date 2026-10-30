"""Interface ordering and verdict preservation; these reports are generated controls."""

import importlib
import json
import sys
from types import ModuleType

import pytest
from test_preflight import dataset

from policy_doctor.artifacts import InputError
from policy_doctor.cli import human, main
from policy_doctor.server import _summary

engine = importlib.import_module('policy_doctor.preflight')


@pytest.mark.parametrize('inputs', [
    {'original_recording': 'recording.hdf5'},
    {'converter_source': 'source'},
    {'converter_config': 'actual.yaml'},
    {'converter_directory': '/explicit-native-cwd'},
])
def test_incomplete_conversion_inputs_refused_before_optional_public_policy(tmp_path, monkeypatch, inputs):
    calls = []
    monkeypatch.setattr(engine.hub, 'resolve_hub_snapshot', lambda *a, **k: calls.append((a, k)))
    with pytest.raises(InputError, match='requires original_recording'):
        engine.preflight(str(tmp_path), checkpoint='hf://org/policy', fetch=True, **inputs)
    assert calls == []


def test_public_dataset_conversion_refused_before_dataset_intake(monkeypatch):
    calls = []
    monkeypatch.setattr(engine.hub, 'resolve_dataset_content', lambda *a, **k: calls.append((a, k)))
    with pytest.raises(InputError, match='explicit local'):
        engine.preflight('hf://org/dataset', fetch=True, original_recording='recording.hdf5',
                         converter_source='source', converter_config='actual.yaml')
    assert calls == []


def test_derived_path_authorization_precedes_all_public_policy_intake(tmp_path, monkeypatch):
    # The artifact reader has its own path tests. This control verifies the
    # preflight orchestrator cannot catch/refuse authorization and fetch anyway.
    calls = []
    intake = ModuleType('policy_doctor.conversion_fidelity')

    def deny(*args, **kwargs):
        calls.append('prepare')
        raise InputError('Generated derived-reference outside-root refusal')

    intake.prepare_conversion_inputs = deny
    monkeypatch.setitem(sys.modules, 'policy_doctor.conversion_fidelity', intake)
    monkeypatch.setattr(engine.hub, 'resolve_hub_snapshot', lambda *a, **k: calls.append('network'))
    with pytest.raises(InputError, match='outside-root'):
        engine.preflight(str(tmp_path), checkpoint='hf://org/policy', fetch=True,
                         original_recording='recording.hdf5', converter_source='source',
                         converter_config='actual.yaml')
    assert calls == ['prepare']


def generated_component():
    return {'status': 'error', 'numeric_association_status': 'error', 'criterion_sha256': 'a' * 64,
            'source_membership': {'status': 'pass'}, 'native_consumption': {'status': 'unknown'},
            'coverage': {'compared_selected_rows': 3, 'whole_numeric_source_to_export': True},
            'findings': [{'status': 'error', 'code': 'conversion.numeric_joint_association',
                          'message': 'GENERATED source demo_0 action frame 1 joint 3 differs from export row 1.'}]}


def test_conversion_error_survives_other_row_error_and_native_unknown(tmp_path, monkeypatch):
    root = dataset(tmp_path, bad_vector=True)
    intake = ModuleType('policy_doctor.conversion_fidelity')
    plan = object()
    intake.prepare_conversion_inputs = lambda *args, **kwargs: plan

    def check(*args, **kwargs):
        assert kwargs['_prepared'] is plan
        return generated_component()

    intake.check_conversion_fidelity = check
    monkeypatch.setitem(sys.modules, 'policy_doctor.conversion_fidelity', intake)
    result = engine.preflight(str(root), original_recording='recording.hdf5',
                              converter_source='source', converter_config='actual.yaml')
    assert result['status'] == 'error'
    assert result['components']['numeric_rows']['status'] == 'error'
    assert result['components']['conversion_fidelity']['native_consumption']['status'] == 'unknown'
    assert {'conversion.numeric_joint_association', 'dataset.vector_invalid'} <= {
        finding['code'] for finding in result['findings']}
    assert result['coverage']['conversion_fidelity_requested'] is True
    assert any('separate export' in action for action in result['next_actions'])


def test_cli_preserves_explicit_converter_cwd_and_does_not_invent_one(tmp_path, monkeypatch, capsys):
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return {'kind': 'dataset_preflight', 'status': 'unknown', 'findings': [], 'components': {},
                'scope': 'GENERATED interface-only response; no artifact reader executed.'}

    monkeypatch.setattr(engine, 'preflight', observed)
    base = ['preflight', str(tmp_path), '--original-recording', 'recording.hdf5',
            '--converter-source', 'source', '--converter-config', 'actual.yaml', '--json']
    assert main(base) == 3
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'
    assert calls[-1]['converter_directory'] is None
    assert main(base + ['--converter-directory', '/explicit-native-cwd']) == 3
    capsys.readouterr()
    assert calls[-1]['converter_directory'] == '/explicit-native-cwd'


def test_concise_and_human_keep_numeric_error_and_unassessed_native_axis():
    component = generated_component()
    report = {'kind': 'dataset_preflight', 'status': 'error', 'scope': 'GENERATED response test only',
              'components': {'conversion_fidelity': component}, 'findings': component['findings']}
    summary = _summary(report)
    concise = summary['components']['conversion_fidelity']
    assert concise['numeric_association_status'] == 'error'
    assert concise['native_consumption_status'] == 'unknown'
    assert concise['criterion_sha256'] == component['criterion_sha256']
    assert concise['compared_numeric_rows'] == 3
    assert 'source demo_0 action frame 1' in summary['findings_shown'][0]['message']
    assert len(json.dumps(summary).encode()) <= 8192
    text = human(report)
    assert 'numeric association: error' in text and 'native consumption: unknown' in text


def test_full_conversion_totals_survive_bounded_findings_and_concise_rendering(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    component = generated_component()
    component.update(counts={'error': 35066, 'unknown': 0}, findings_total=35066, findings_truncated=True)
    component['findings'] *= 64
    intake = ModuleType('policy_doctor.conversion_fidelity')
    intake.prepare_conversion_inputs = lambda *args, **kwargs: object()
    intake.check_conversion_fidelity = lambda *args, **kwargs: component
    monkeypatch.setitem(sys.modules, 'policy_doctor.conversion_fidelity', intake)
    report = engine.preflight(str(root), original_recording='recording.hdf5',
                              converter_source='source', converter_config='actual.yaml')
    assert report['total_findings'] >= 35066 and report['findings_truncated']
    assert report['counts_scope'] == 'available_findings'
    summary = _summary(report)
    assert summary['findings_total'] == report['total_findings']
    assert summary['counts_scope'] == 'available_findings'
    conversion = summary['components']['conversion_fidelity']
    assert conversion['counts'] == {'error': 35066, 'unknown': 0}
    assert conversion['findings_total'] == 35066 and conversion['findings_truncated']
    assert len(json.dumps(summary).encode()) <= 8192
    assert 'conversion findings: 35066 (64 retained in this report)' in human(report)


def test_auxiliary_unknown_and_error_are_visible_beside_passing_joints():
    component = generated_component()
    component.update(joint_numeric_association_status='pass', auxiliary_numeric_association_status='error',
                     requested_scope='state_action_and_teleop_numeric_conversion', auxiliary_fields={
        'teleop.base_height_command': {'status': 'pass', 'width': 1, 'expected_rows': 35066, 'compared_rows': 35066},
        'teleop.navigate_command': {'status': 'error', 'width': 3, 'expected_rows': 35066, 'compared_rows': 35066, 'error_count': 1},
        'teleop.torso_orientation_rpy_command': {'status': 'unknown', 'width': 3, 'expected_rows': 35066, 'compared_rows': 0}})
    report = {'kind': 'dataset_preflight', 'status': 'error', 'components': {'conversion_fidelity': component},
              'findings': component['findings']}
    compact = _summary(report)['components']['conversion_fidelity']
    assert compact['joint_numeric_association_status'] == 'pass'
    assert compact['auxiliary_numeric_association_status'] == 'error'
    assert compact['auxiliary_fields']['teleop.navigate_command']['error_count'] == 1
    assert compact['auxiliary_fields']['teleop.torso_orientation_rpy_command']['status'] == 'unknown'
    assert 'native_consumption_status' in compact and compact['native_consumption_status'] == 'unknown'
    text = human(report)
    assert 'joint association: pass' in text and 'auxiliary commands: error' in text
    assert 'teleop.torso_orientation_rpy_command: unknown (0/35066 rows)' in text
