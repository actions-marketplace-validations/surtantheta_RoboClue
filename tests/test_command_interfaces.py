"""Saved generated command packets through existing CLI/MCP interfaces."""
import json
import shutil

import pytest
from test_command_path import change, contract
from test_command_path import packet as packet_fixture

from policy_doctor.cli import human, main
from policy_doctor.reports import markdown_report


@pytest.fixture(name='packet')
def command_packet(tmp_path, monkeypatch):
    return packet_fixture.__wrapped__(tmp_path, monkeypatch)


def write_contract(tmp_path, value=None):
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(contract() if value is None else value))
    return path


def test_cli_audit_contract_and_unknown_without_it(packet, tmp_path, capsys):
    capture = packet()
    assert main(['audit-capture', str(capture), '--json']) == 3
    assert json.loads(capsys.readouterr().out)['kind'] == 'command_path_audit'
    path = write_contract(tmp_path)
    assert main(['audit-capture', str(capture), '--command-contract', str(path), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['acceptance_contract']['status'] == 'pass'
    assert 'Software command path' in human(result)
    assert 'Declared tensor order' in human(result)
    assert 'recorded actions' in human(result)
    assert main(['audit-capture', str(capture), '--command-contract', str(path), '--markdown']) == 0
    report = capsys.readouterr().out
    assert 'Observed software command path' in report
    assert 'Physical execution' in report and 'unknown' in report


def test_cli_processor_fault_exit_and_report_evidence(packet, tmp_path, capsys):
    capture = packet(mean=(13., -20.))
    path = write_contract(tmp_path)
    assert main(['audit-capture', str(capture), '--command-contract', str(path), '--json']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['acceptance_contract']['violations'][0]['field'] == 'effective_processor'
    rendered = markdown_report(result)
    assert 'effective' in rendered.lower() and 'SHA' in rendered
    assert 'capture.json' in rendered


def test_cli_compare_reports_first_stage_without_automatic_fault(packet, capsys):
    reference = packet('reference')
    candidate = packet('candidate', mean=(13., -20.))
    assert main(['compare-captures', str(reference), str(candidate), '--json']) == 3
    result = json.loads(capsys.readouterr().out)
    assert result['first_difference']['stage'] == 'postprocessor'
    assert result['difference_is_fault'] is False
    assert 'First differing stage' in human(result)
    assert 'lawful' in markdown_report(result)


@pytest.mark.parametrize('value', [[], {'unexpected': 'value'}, {'expected_processor': True}, {'atol': 0.}])
def test_cli_bad_contract_refused_without_output(packet, tmp_path, capsys, value):
    capture = packet()
    path = write_contract(tmp_path, value)
    output = tmp_path / 'report.json'
    assert main(['audit-capture', str(capture), '--command-contract', str(path), '--output', str(output)]) == 2
    assert not output.exists()
    assert 'invalid_input' in capsys.readouterr().err


def test_cli_create_passes_same_contract_and_controls(packet, tmp_path, capsys, monkeypatch):
    from policy_doctor import incidents
    before, after = packet('before', mean=(13., -20.)), packet('after')
    config = write_contract(tmp_path)
    controls = tmp_path / 'controls.json'
    controls.write_text(json.dumps({'controls': [{'name': 'healthy', 'source': 'after', 'parameters': {}, 'expected_status': 'pass'}]}))
    calls = []
    def create(check, source, destination, parameters, **kwargs):
        calls.append((check, source, destination, parameters, kwargs))
        return {'kind': 'exported_incident_case', 'status': 'pass'}
    monkeypatch.setattr(incidents, 'export_incident_case', create)
    arguments = ['create-case', 'act-command-path', str(before), str(tmp_path / 'case'), '--corrected', str(after)]
    assert main([*arguments, '--command-contract', str(config), '--controls', str(controls), '--json']) == 0
    capsys.readouterr()
    check, source, _, parameters, extra = calls[0]
    assert check == 'act_command_path' and source == str(before) and parameters == contract()
    assert extra['corrected'] == str(after)
    assert extra['controls'][0]['source'] == str(after.resolve())
    assert main([*arguments, '--json']) == 0
    capsys.readouterr()
    assert calls[1][3] == {}  # Core derives normative criteria from supplied corrected full chain.


def test_existing_mcp_tools_forward_contract_and_keep_catalog(packet, tmp_path):
    pytest.importorskip('mcp')
    from policy_doctor.server import create_server
    capture = packet()
    server = create_server([str(tmp_path)])
    tools = server._tool_manager._tools
    assert len(tools) == 29
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    assert 'create_incident_case' not in tools
    result = tools['audit_runtime_capture'].fn(str(capture), command_contract=contract())
    assert result['kind'] == 'command_path_audit' and result['status'] == 'pass'
    with pytest.raises(Exception, match='outside'):
        tools['audit_runtime_capture'].fn('/outside', command_contract=contract())
    output = tmp_path / 'output'
    output.mkdir()
    writable = create_server([str(tmp_path)], case_root=str(output))
    assert len(writable._tool_manager._tools) == 31
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(writable._tool_manager._tools)


def test_command_human_and_markdown_escape_artifact_text():
    report = {'kind': 'command_path_comparison', 'status': 'unknown', 'comparison_supported': True,
              'equal_over_compared_ticks': False, 'difference_is_fault': False,
              'first_difference': {'tick_ordinal': 0, 'stage': '<script>processor</script>', 'reference': [1.], 'candidate': [2.]},
              'context_differences': [], 'scope': 'No physical execution evidence.'}
    assert '<script>' not in markdown_report(report)
    assert 'First differing stage' in human(report)


def test_real_cli_case_handoff_and_later_values(packet, tmp_path, capsys):
    before = packet('before', mean=(13., -20.))
    after = packet('after')
    healthy = packet('later', selected=(5., 7.))
    fault = packet('recurrence', selected=(5., 7.), mean=(13., -20.))
    incomplete = packet('incomplete', selected=(5., 7.))
    change(incomplete, lambda d: d['coverage'].update(complete_target_call_history=False))
    case = tmp_path / 'case'
    assert main(['create-case', 'act-command-path', str(before), str(case), '--corrected', str(after), '--json']) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported['checks']['before']['acceptance_contract']['status'] == 'error'
    assert exported['checks']['after']['acceptance_contract']['status'] == 'pass'
    assert 'command-path' in human(exported)
    moved = tmp_path / 'moved'
    case.rename(moved)
    shutil.rmtree(before)
    shutil.rmtree(after)
    assert main(['reproduce-case', str(moved), '--json']) == 0
    capsys.readouterr()
    for candidate, exit_code, status in [(healthy, 0, 'pass'), (fault, 1, 'error'), (incomplete, 3, 'unknown')]:
        assert main(['check-case', str(moved), str(candidate), '--json']) == exit_code
        result = json.loads(capsys.readouterr().out)
        assert result['status'] == status
        assert result['candidate']['kind'] == 'command_path_audit'
        assert 'Software command contract' in human(result)


def test_opt_in_mcp_case_and_retained_candidate_path(packet, tmp_path):
    pytest.importorskip('mcp')
    from policy_doctor.artifacts import InputError
    from policy_doctor.server import create_server
    before, after = packet('before', mean=(13., -20.)), packet('after')
    output = tmp_path / 'output'
    output.mkdir()
    tools = create_server([str(tmp_path)], case_root=str(output))._tool_manager._tools
    created = tools['create_incident_case'].fn('act_command_path', str(before), str(output / 'case'), {}, corrected=str(after))
    assert created['status'] == 'pass'
    result = tools['check_incident_candidate'].fn(str(output / 'case'), str(after))
    assert result['status'] == 'pass' and result['candidate_evaluated']
    with pytest.raises(InputError, match='strictly below'):
        tools['create_incident_case'].fn('act_command_path', str(before), str(tmp_path / 'not-output'), {}, corrected=str(after))
    with pytest.raises(InputError, match='outside'):
        tools['create_incident_case'].fn('act_command_path', str(before), str(output / 'other'), {}, corrected='/outside')
