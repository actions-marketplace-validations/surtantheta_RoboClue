"""Generated-source integration controls; these are not practitioner trials."""
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_conversion_fidelity_reader_proposal import Generated
from test_conversion_preflight_artifacts import corrupt_action

from policy_doctor.artifacts import InputError
from policy_doctor.cli import human, main
from policy_doctor.preflight import preflight
from policy_doctor.server import _summary


def saved_report(generated, target):
    report = preflight(str(generated.export), **generated.kwargs())
    target.write_text(json.dumps(report, allow_nan=False))
    return report['components']['conversion_fidelity']


@pytest.mark.parametrize('reference,expected', [
    ('/outside/rejected.json', None), (None, 'NOT-A-SHA256'), ('missing.json', 'b' * 64)])
def test_bad_reuse_input_precedes_optional_policy_network(tmp_path, monkeypatch, reference, expected):
    import importlib
    engine = importlib.import_module('policy_doctor.preflight')
    calls = []
    monkeypatch.setattr(engine.hub, 'resolve_hub_snapshot', lambda *a, **k: calls.append('network'))
    with pytest.raises(InputError):
        preflight(str(tmp_path), checkpoint='hf://org/policy', fetch=True,
                  conversion_reference=reference, expected_conversion_criterion=expected,
                  read_roots=[str(tmp_path)])
    assert not calls


@pytest.mark.parametrize('alias', ['same', 'hardlink', 'symlink'])
def test_cli_report_output_preserves_existing_baseline_aliases(tmp_path, monkeypatch, capsys, alias):
    import importlib
    engine = importlib.import_module('policy_doctor.preflight')
    reference = tmp_path / 'before.json'
    original = b'{"scope":"GENERATED output-write control, no source comparison"}\n'
    reference.write_bytes(original)
    output = tmp_path / 'after.json'
    if alias == 'same':
        output = reference
    elif alias == 'hardlink':
        os.link(reference, output)
    else:
        output.symlink_to(reference)
    monkeypatch.setattr(engine, 'preflight', lambda *a, **k: dict(
        kind='dataset_preflight', status='error', components={},
        findings=[dict(status='error', code='conversion.criterion_drift', message='GENERATED report')]))
    assert main(['preflight', str(tmp_path), '--conversion-reference', str(reference),
                 '--output', str(output), '--json']) == 2
    assert json.loads(capsys.readouterr().err)['error']['code'] == 'invalid_input'
    assert reference.read_bytes() == output.read_bytes() == original


def test_known_bad_baseline_reuses_source_criterion_and_candidate_can_move(tmp_path):
    generated = Generated(tmp_path)
    corrupt_action(generated)
    reference = tmp_path / 'bad-before.json'
    before = saved_report(generated, reference)
    assert before['status'] == 'error'
    assert not before['coverage']['whole_numeric_source_to_export']
    assert before['conversion_reuse']['criterion_basis_complete']
    generated.restore()
    relocated = tmp_path / 'export-after'
    shutil.copytree(generated.export, relocated)
    after = preflight(str(relocated), conversion_reference=str(reference), read_roots=[str(tmp_path)])
    component = after['components']['conversion_fidelity']
    assert component['numeric_association_status'] == 'pass'
    assert component['criterion_reuse']['status'] == 'pass'
    assert component['criterion_reuse']['reference_original_status'] == 'error'
    assert component['criterion_sha256'] == before['criterion_sha256']
    assert component['native_consumption']['status'] == 'unknown'


def test_real_cli_reference_restores_cwd_and_pins_healthy_error_and_drift(tmp_path, capsys):
    generated = Generated(tmp_path)
    reference = tmp_path / 'before.json'
    before = saved_report(generated, reference)
    criterion = before['criterion_sha256']
    corrupt_action(generated)
    arguments = ['preflight', str(generated.export), '--conversion-reference', str(reference),
                 '--expected-conversion-criterion', criterion, '--json']
    assert main(arguments) == 1
    changed = json.loads(capsys.readouterr().out)['components']['conversion_fidelity']
    assert changed['numeric_association_status'] == 'error'
    assert changed['criterion_reuse']['matches']
    assert changed['criterion_reuse']['candidate_evaluated_under_expected_criterion']
    assert any(f['code'] == 'conversion.numeric_joint_association' for f in changed['findings'])
    generated.restore()
    assert main(arguments) == 3  # Other unassessed preflight components remain UNKNOWN.
    corrected = json.loads(capsys.readouterr().out)['components']['conversion_fidelity']
    assert corrected['numeric_association_status'] == 'pass'
    assert corrected['criterion_reuse']['matches']
    generated.config_path.write_text(generated.config_path.read_text() + '\n# deliberate recipe-byte change\n')
    assert main(arguments) == 1
    drifted = json.loads(capsys.readouterr().out)
    component = drifted['components']['conversion_fidelity']
    assert component['status'] == 'error' and component['numeric_association_status'] == 'unknown'
    assert component['criterion_reuse']['status'] == 'error'
    assert not component['criterion_reuse']['candidate_evaluated_under_expected_criterion']
    assert any(f['code'] == 'conversion.criterion_drift' for f in component['findings'])
    compact = _summary(drifted)['components']['conversion_fidelity']['criterion_reuse']
    assert compact['expected_sha256'] == criterion
    assert compact['current_sha256'] != criterion and compact['matches'] is False
    assert compact['first_drift_field'] and 'keep the old report' in compact['next']
    assert 'saved criterion: error; matches: False' in human(drifted)


def test_compact_drift_binding_survives_many_other_findings():
    binding = dict(requested=True, status='error', expected_sha256='a' * 64,
                   current_sha256='b' * 64, matches=False,
                   candidate_evaluated_under_expected_criterion=False,
                   drift_fields=['/original_sha256', '/evaluator/conversion_fidelity.py'],
                   drift_fields_omitted=4)
    component = dict(status='error', numeric_association_status='unknown', criterion_reuse=binding)
    report = dict(kind='dataset_preflight', status='error', components={'conversion_fidelity': component},
                  findings=[dict(status='error', code='conversion.numeric_joint_association',
                                 message='GENERATED independent current-criterion error ' * 20)] * 64)
    result = _summary(report)
    compact = result['components']['conversion_fidelity']['criterion_reuse']
    assert compact['matches'] is False
    assert compact['first_drift_field'] == '/original_sha256' and compact['drift_fields_omitted'] == 5
    assert 'keep the old report' in compact['next']
    assert not compact['candidate_evaluated_under_expected_criterion']
    assert len(json.dumps(result).encode()) <= 8192


def test_actual_stdio_reuse_detailed_reference_concise_later_candidate(tmp_path):
    generated = Generated(tmp_path)
    source = str(Path(__file__).parents[1] / 'src')
    bootstrap = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("policy_doctor",run_name="__main__")'

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-I', '-B', '-c', bootstrap, source, 'serve', '--root', str(tmp_path)],
            cwd=str(tmp_path), env=dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1'))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                schema = next(tool for tool in (await client.list_tools()).tools
                              if tool.name == 'preflight_check').inputSchema
                assert {'conversion_reference', 'expected_conversion_criterion'} <= set(schema['properties'])
                assert 'conversion_reference' in schema['properties']['response_format']['enum']
                first = await client.call_tool('preflight_check', dict(
                    source=str(generated.export), response_format='conversion_reference',
                    **{k: v for k, v in generated.kwargs().items() if k != 'read_roots'}))
                assert not first.isError
                full = first.structuredContent
                assert full['kind'] == 'conversion_preflight_reference'
                assert full['reference_available'] is True
                assert len(json.dumps(full).encode()) <= 192 * 1024
                criterion = full['components']['conversion_fidelity']['criterion_sha256']
                reference = tmp_path / 'sdk-before.json'
                reference.write_text(json.dumps(full, allow_nan=False))
                corrupt_action(generated)
                arguments = dict(source=str(generated.export), conversion_reference=str(reference),
                                 expected_conversion_criterion=criterion)
                changed = await client.call_tool('preflight_check', arguments)
                assert not changed.isError
                compact = changed.structuredContent['components']['conversion_fidelity']
                assert compact['numeric_association_status'] == 'error'
                assert compact['criterion_reuse']['matches']
                assert compact['criterion_reuse']['candidate_evaluated_under_expected_criterion']
                assert len(json.dumps(changed.structuredContent).encode()) <= 8192
                generated.restore()
                corrected = await client.call_tool('preflight_check', arguments)
                assert not corrected.isError
                compact = corrected.structuredContent['components']['conversion_fidelity']
                assert compact['numeric_association_status'] == 'pass'
                assert compact['criterion_reuse']['matches']
                refused = await client.call_tool('preflight_check', dict(arguments,
                    conversion_reference='/outside/rejected.json'))
                assert refused.isError

    asyncio.run(asyncio.wait_for(run(), timeout=90))
