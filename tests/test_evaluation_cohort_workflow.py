"""Observed native cohort handoff and fixed-criterion regression boundaries."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.incidents import check_incident_candidate, export_incident_case, reproduce_incident_case
from policy_doctor.investigation import investigate_run
from policy_doctor.reports import markdown_report

PARAMETERS = {'table': {'groups': '/tasks', 'group_id': '/task', 'records': '/episodes',
                         'record_id': '/episode_id', 'outcome': '/metrics/success'},
              'equal_fields': ['/config/max_steps']}
REQUIREMENT = {'kind': 'evaluation-cohort', 'parameters': PARAMETERS}


def native(root, cap=100, outcome=False, missing=False):
    root.mkdir()
    payload = {'config': {} if missing else {'max_steps': cap},
               'tasks': [{'task': 'radio', 'episodes': [{'episode_id': 0, 'metrics': {'success': outcome}}]}]}
    (root / 'original-eval.json').write_text(json.dumps(payload))
    (root / 'LICENSE').write_text('Example generated fixture license')
    return root


def test_portable_case_and_later_outcome_changes_keep_original_criteria(tmp_path):
    originals = tmp_path / 'originals'
    originals.mkdir()
    before, after = native(originals / 'before'), native(originals / 'after', 2000, True)
    unknown = native(originals / 'missing', missing=True)
    evidence = {p: p.read_bytes() for p in originals.rglob('*') if p.is_file()}
    report = investigate_run(str(before), reference=str(after), requirements=REQUIREMENT)
    assert report['status'] == 'error'
    case = tmp_path / 'case'
    result = export_incident_case(**report['case_recipe']['arguments'], destination=case, controls=[{
        'name': 'missing recorded cap', 'source': str(unknown), 'parameters': PARAMETERS, 'expected_status': 'unknown'}])
    assert result['status'] == 'pass'
    assert all(path.read_bytes() == raw for path, raw in evidence.items())
    assert (case / 'before/original-eval.json').read_bytes() == (before / 'original-eval.json').read_bytes()
    shutil.rmtree(originals)
    moved = tmp_path / 'colleague packet'
    case.rename(moved)
    replay = reproduce_incident_case(moved)
    assert replay['status'] == 'pass' and replay['reproduction_matches']
    assert replay['checks']['controls'][0]['check']['status'] == 'unknown'
    later = native(tmp_path / 'later', 2000, False)
    reused = check_incident_candidate(moved, later)
    assert reused['status'] == 'pass' and reused['candidate']['candidate']['true_count'] == 0
    assert reused['candidate']['completeness']['declared_counts'] == 'unknown'
    assert len(markdown_report(replay)) < 14000
    assert 'physical' in markdown_report(reused).lower()


def test_healthy_comparison_does_not_manufacture_incident(tmp_path):
    left, right = native(tmp_path / 'left'), native(tmp_path / 'right', outcome=True)
    report = investigate_run(str(left), reference=str(right), requirements=REQUIREMENT)
    assert report['status'] == 'pass' and report['case_recipe'] is None
    with pytest.raises(InputError, match='primary incident'):
        export_incident_case('evaluation_cohort', left, tmp_path / 'case', PARAMETERS, corrected=right)
    assert not (tmp_path / 'case').exists()


def test_control_cannot_silently_weaken_selected_context(tmp_path):
    left, right = native(tmp_path / 'left'), native(tmp_path / 'right', 2000)
    with pytest.raises(InputError, match='identical'):
        export_incident_case('evaluation_cohort', left, tmp_path / 'case', PARAMETERS, corrected=right,
            controls=[{'name': 'weakened', 'source': str(right), 'parameters': dict(PARAMETERS, equal_fields=[]),
                       'expected_status': 'pass'}])
    assert not (tmp_path / 'case').exists()


def test_cli_shared_requirement_and_reports(tmp_path, capsys):
    left, right = native(tmp_path / 'left'), native(tmp_path / 'right', 2000)
    criterion = tmp_path / 'requirement.json'
    criterion.write_text(json.dumps(REQUIREMENT))
    for format_args in ([], ['--json'], ['--markdown']):
        assert main(['investigate', str(left), '--reference', str(right), '--requirements', str(criterion), *format_args]) == 1
        output = capsys.readouterr()
        assert not output.err and ('max\\_steps' if '--markdown' in format_args else 'max_steps') in output.out
        assert 'criterion not supplied' not in output.out
        if '--json' not in format_args:
            assert len(output.out.splitlines()) <= 40
    case = tmp_path / 'case'
    assert main(['create-case', 'evaluation-cohort', str(left), str(case), '--corrected', str(right),
                 '--requirements', str(criterion), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'pass'
    assert main(['check-case', str(case), str(right), '--markdown']) == 0
    assert 'Declared counts' in capsys.readouterr().out


def test_missing_reference_is_unknown_and_no_recipe(tmp_path):
    root = native(tmp_path / 'run')
    report = investigate_run(str(root), requirements=REQUIREMENT)
    assert report['status'] == 'unknown' and report['case_recipe'] is None


def test_tampered_retained_native_json_cannot_be_reused(tmp_path):
    left, right = native(tmp_path / 'left'), native(tmp_path / 'right', 2000)
    case = tmp_path / 'case'
    export_incident_case('evaluation_cohort', left, case, PARAMETERS, corrected=right)
    (case / 'after/original-eval.json').write_text('{}')
    with pytest.raises(InputError, match='hash'):
        check_incident_candidate(case, right)


def test_actual_mcp_transport_checks_and_root_guards(tmp_path):
    before, after = native(tmp_path / 'before'), native(tmp_path / 'after', 2000, True)
    outputs = tmp_path / 'outputs'
    outputs.mkdir()
    case = outputs / 'case'

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path), '--case-root', str(outputs)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src')))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('investigate_run_directory',
                {'directory': str(before), 'reference': str(after), 'requirements': REQUIREMENT})
            assert not result.isError
            report = json.loads(result.content[0].text)
            assert report['status'] == 'error'
            result = await session.call_tool('create_incident_case',
                dict(report['case_recipe']['arguments'], destination=str(case)))
            assert not result.isError and json.loads(result.content[0].text)['status'] == 'pass'
            for tool, args in [('reproduce_incident_case', {'directory': str(case)}),
                               ('check_incident_candidate', {'directory': str(case), 'candidate': str(after)})]:
                result = await session.call_tool(tool, args)
                assert not result.isError and json.loads(result.content[0].text)['status'] == 'pass'
            for tool, args in [('investigate_run_directory', {'directory': str(before), 'reference': '/etc',
                                                             'requirements': REQUIREMENT}),
                               ('create_incident_case', dict(report['case_recipe']['arguments'],
                                                            destination=str(tmp_path / 'outside-output')))]:
                assert (await session.call_tool(tool, args)).isError

    asyncio.run(run())
