"""Exercise raw-layout evidence, native-case retention and the actual transport."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from layout_test_support import layout_input
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor.artifacts import InputError
from policy_doctor.capture import audit_capture
from policy_doctor.incidents import check_incident_candidate, export_incident_case, reproduce_incident_case
from policy_doctor.layouts import write_layout_capture
from policy_doctor.runtime_diff import compare_runtime_captures

SRC = str(Path(__file__).parents[1] / 'src')


@pytest.fixture
def layouts(tmp_path):
    result = {}
    for name, order, identity in [('fault', ('right_foot', 'left_foot'), True),
                                  ('healthy', ('left_foot', 'right_foot'), True),
                                  ('unknown', ('left_foot', 'right_foot'), False)]:
        metadata, arrays, sources = layout_input(order=order, identity=identity)
        path = tmp_path / name
        write_layout_capture(path, metadata, arrays, sources=sources)
        result[name] = path
    return result


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'policy_doctor', *map(str, args)],
                          env=dict(os.environ, PYTHONPATH=SRC), capture_output=True,
                          text=True, timeout=30)


def test_existing_capture_dispatch_preserves_raw_and_named_distinctions(layouts):
    report = audit_capture(str(layouts['fault']), expected_names=['left_foot', 'right_foot'])
    assert report['kind'] == 'resolved_layout_audit' and report['status'] == 'error'
    assert report['raw_binding']['replay_equal']
    assert report['layout']['body_names'] == ['right_foot', 'left_foot']
    compared = compare_runtime_captures(str(layouts['fault']), str(layouts['healthy']))
    assert compared['kind'] == 'resolved_layout_comparison'
    assert compared['ordered_layout_equal'] is False
    assert compared['named_values_equal'] is True
    assert compared['raw_values_equal'] is False
    with pytest.raises(InputError):
        audit_capture(str(layouts['healthy']), dataset='/unused')


def test_retained_layout_case_moves_and_allows_later_values(layouts, tmp_path):
    contract = {'expected_names': ['left_foot', 'right_foot']}
    controls = [{'name': 'healthy', 'source': layouts['healthy'], 'parameters': contract,
                 'expected_status': 'pass'},
                {'name': 'missing identity', 'source': layouts['unknown'],
                 'parameters': dict(contract, expected_identity={'prim_path': '/World/Robot/.*'}),
                 'expected_status': 'unknown'}]
    case = tmp_path / 'case'
    report = export_incident_case('resolved_layout', layouts['fault'], case, contract,
                                 corrected=layouts['healthy'], controls=controls)
    assert report['status'] == 'pass'
    manifest = json.loads((case / 'manifest.json').read_text())
    assert manifest['parameters']['expected_identity']
    assert manifest['parameters']['expected_frame'] == 'incoming_joint_frame'
    assert manifest['parameters']['expected_quantity'] == 'incoming_joint_wrench'
    moved = tmp_path / 'moved'
    case.rename(moved)
    for path in layouts.values():
        shutil.rmtree(path)
    assert reproduce_incident_case(moved)['status'] == 'pass'
    later = {}
    for name, order, identity in [('healthy', ('left_foot', 'right_foot'), True),
                                  ('fault', ('right_foot', 'left_foot'), True),
                                  ('unknown', ('left_foot', 'right_foot'), False)]:
        metadata, arrays, sources = layout_input(order=order, identity=identity)
        for key in arrays:
            arrays[key] *= 2
        path = tmp_path / ('later-' + name)
        write_layout_capture(path, metadata, arrays, sources=sources)
        later[name] = check_incident_candidate(moved, path)
    assert [later[k]['status'] for k in ('healthy', 'fault', 'unknown')] == ['pass', 'error', 'unknown']
    assert all(report['case_verified'] and report['candidate_evaluated'] for report in later.values())
    assert all(report['parameters'] == manifest['parameters'] for report in later.values())


def test_cli_layout_contract_case_and_readable_reports(layouts, tmp_path):
    for format_option in ('--json', '--markdown', None):
        args = ['audit-capture', layouts['fault'], '--expected-names', 'left_foot', 'right_foot']
        result = cli(*args, *([format_option] if format_option else []))
        assert result.returncode == 1, result.stderr
        assert 'right_foot' in result.stdout.replace('\\_', '_')
    result = cli('compare-captures', layouts['fault'], layouts['healthy'], '--json')
    assert json.loads(result.stdout)['named_values_equal'] is True
    case = tmp_path / 'cli-case'
    result = cli('create-case', 'resolved-layout', layouts['fault'], case, '--corrected', layouts['healthy'],
                 '--expected-names', 'left_foot', 'right_foot', '--json')
    assert result.returncode == 0, result.stderr
    result = cli('check-case', case, layouts['unknown'], '--json')
    assert result.returncode == 3 and json.loads(result.stdout)['candidate_evaluated']


def test_corrected_namespace_cannot_silently_change_the_subject(layouts, tmp_path):
    metadata, arrays, sources = layout_input(order=('left_foot', 'right_foot'))
    metadata['runtime']['sensor_identity'] = {'prim_path': '/World/OtherRobot'}
    unrelated = tmp_path / 'unrelated'
    write_layout_capture(unrelated, metadata, arrays, sources=sources)
    destination = tmp_path / 'invalid-case'
    with pytest.raises(InputError, match='same evidenced sensor namespace'):
        export_incident_case('resolved_layout', layouts['fault'], destination,
                             {'expected_names': ['left_foot', 'right_foot']}, corrected=unrelated)
    assert not destination.exists()
    assert not list(tmp_path.glob('.policy-doctor-incident-*'))


def test_retained_layout_manifest_cannot_omit_semantic_or_numeric_criteria(layouts, tmp_path):
    import hashlib

    case = tmp_path / 'retained'
    export_incident_case('resolved_layout', layouts['fault'], case,
                         {'expected_names': ['left_foot', 'right_foot']}, corrected=layouts['healthy'])
    manifest = json.loads((case / 'manifest.json').read_text())
    assert manifest['parameters']['atol'] == 1e-6 and manifest['parameters']['rtol'] == 1e-5
    for key in ('expected_identity', 'expected_frame', 'expected_quantity', 'atol', 'rtol'):
        changed = json.loads(json.dumps(manifest))
        del changed['parameters'][key]
        encoded = json.dumps(changed).encode()
        (case / 'manifest.json').write_bytes(encoded)
        (case / 'READY.json').write_text(json.dumps({'schema_version': 1,
            'manifest_sha256': hashlib.sha256(encoded).hexdigest()}))
        with pytest.raises(InputError, match='requires'):
            check_incident_candidate(case, layouts['healthy'])


def test_actual_mcp_layout_audit_compare_and_case_reuse(layouts, tmp_path):
    case_root = tmp_path / 'cases'
    case_root.mkdir()

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path), '--case-root', str(case_root)],
            env=dict(os.environ, PYTHONPATH=SRC))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert len(tools) == 31
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                result = await client.call_tool('audit_runtime_capture', {'capture': str(layouts['fault']),
                    'expected_names': ['left_foot', 'right_foot']})
                assert not result.isError and result.structuredContent['status'] == 'error'
                result = await client.call_tool('compare_runtime_captures', {
                    'reference': str(layouts['fault']), 'candidate': str(layouts['healthy'])})
                assert not result.isError and result.structuredContent['named_values_equal'] is True
                case = case_root / 'retained'
                result = await client.call_tool('create_incident_case', {'check': 'resolved_layout',
                    'source': str(layouts['fault']), 'destination': str(case),
                    'corrected': str(layouts['healthy']),
                    'parameters': {'expected_names': ['left_foot', 'right_foot']}})
                assert not result.isError and result.structuredContent['status'] == 'pass'
                for name, status in [('healthy', 'pass'), ('fault', 'error'), ('unknown', 'unknown')]:
                    result = await client.call_tool('check_incident_candidate', {
                        'directory': str(case), 'candidate': str(layouts[name])})
                    assert not result.isError and result.structuredContent['status'] == status
                for payload in ({'capture': '/etc'}, {'capture': str(layouts['healthy']), 'expected_names': True}):
                    result = await client.call_tool('audit_runtime_capture', payload)
                    assert result.isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=60))
