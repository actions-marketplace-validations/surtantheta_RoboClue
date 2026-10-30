"""Incomplete and unverified portable evidence cannot become a discovery PASS."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import investigation
from policy_doctor.demo import create_policy_demo
from policy_doctor.retained import retain_check

SRC = str(Path(__file__).resolve().parents[1] / 'src')


def _metadata_case(root):
    inputs = root / 'caller-inputs'
    create_policy_demo(inputs)
    output = root / 'capsule'
    retain_check('policy_metadata', inputs / 'checkpoint', output,
                 dataset=inputs / 'corrected_dataset')
    shutil.rmtree(inputs)
    return output


def _mixed_case(case, *, tampered=True):
    root = case['checkpoint'].parent
    # The ordinary colocated checkpoint, dataset and contract really do pass.
    healthy = investigation.investigate_run(str(root))
    assert healthy['status'] == 'pass' and healthy['audits'][0]['status'] == 'pass'
    output = root / 'portable-check'
    retain_check('policy_metadata', case['checkpoint'], output, dataset=case['dataset'])
    if tampered:
        # Keep a recognizable envelope but invalidate a real retained payload.
        expected = output / 'expected.json'
        expected.write_bytes(expected.read_bytes() + b' ')
    return root, output


def _forbid_internal_audit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Discovery inspected or audited incomplete case internals')
    for name in ('audit', 'audit_capture', 'inspect_rollout'):
        monkeypatch.setattr(investigation, name, forbidden)


def _assert_unknown_opaque(report, directory):
    assert report['status'] == 'unknown'
    assert not report['audits']
    for key in ('checkpoints', 'datasets', 'recordings'):
        assert not report['inventory'][key]
    assert report['intake_problems']
    assert any(item['path'] in (str(directory), str(directory / 'manifest.json'))
               for item in report['intake_problems'])


def test_missing_manifest_keeps_real_metadata_case_opaque(tmp_path, monkeypatch):
    output = _metadata_case(tmp_path)
    (output / 'manifest.json').unlink()
    _forbid_internal_audit(monkeypatch)
    report = investigation.investigate_run(str(output))
    _assert_unknown_opaque(report, output)
    assert report['inventory']['incomplete_cases'] == [str(output)]
    assert report['coverage']['files_seen'] == 1
    action = next(a for a in report['next_actions'] if a['tool'] == 'reproduce_incident_case')
    assert action['arguments'] == {'directory': str(output)}
    assert 'incomplete' in action['purpose'].lower()


@pytest.mark.parametrize('marker', ['symlink', 'directory', 'oversized'])
def test_unreadable_ready_without_manifest_never_exposes_case_inputs(tmp_path, monkeypatch, marker):
    output = _metadata_case(tmp_path)
    (output / 'manifest.json').unlink()
    ready = output / 'READY.json'
    ready.unlink()
    if marker == 'symlink':
        outside = tmp_path / 'outside.json'
        outside.write_text('{}')
        ready.symlink_to(outside)
    elif marker == 'directory':
        ready.mkdir()
    else:
        ready.write_text(' ' * (1024 * 1024 + 1))
    _forbid_internal_audit(monkeypatch)
    _assert_unknown_opaque(investigation.investigate_run(str(output)), output)


@pytest.mark.parametrize('tampered', [False, True])
def test_unverified_colocated_case_keeps_real_pass_audit_separate(case, tampered):
    root, output = _mixed_case(case, tampered=tampered)
    report = investigation.investigate_run(str(root))
    assert report['status'] == 'unknown'
    assert len(report['audits']) == 1 and report['audits'][0]['status'] == 'pass'
    assert report['inventory']['retained_checks'] == [str(output)]
    action = next(a for a in report['next_actions'] if a['tool'] == 'reproduce_incident_case')
    assert action['arguments'] == {'directory': str(output)}
    assert 'not verified' in action['purpose']


@pytest.mark.parametrize('kind,tool', [
    ('robot_incident_case', 'reproduce_incident_case'),
    ('robot_video_reference_recovery', 'reproduce_incident_case'),
    ('trajectory_regression_case', 'reproduce_trajectory_case'),
    ('native_trajectory_sample_case', 'reproduce_trajectory_case'),
])
def test_other_unverified_envelope_kinds_also_prevent_mixed_folder_pass(case, kind, tool):
    root = case['checkpoint'].parent
    output = root / 'unverified-envelope'
    output.mkdir()
    # Discovery may recognize this discriminator; it has not verified a case.
    (output / 'manifest.json').write_text(json.dumps({'kind': kind, 'schema_version': 1}))
    (output / 'READY.json').write_text('{}')
    report = investigation.investigate_run(str(root))
    assert report['status'] == 'unknown'
    assert report['audits'][0]['status'] == 'pass'
    assert any(a['tool'] == tool and a['arguments'] == {'directory': str(output)}
               for a in report['next_actions'])


def test_real_audit_error_is_not_hidden_by_unverified_case(case):
    root, _ = _mixed_case(case, tampered=False)
    deployment = json.loads(case['deployment'].read_text())
    deployment['units'] = ['deg', 'normalized']
    case['put'](case['deployment'], deployment)
    report = investigation.investigate_run(str(root))
    assert report['status'] == 'error' and report['audits'][0]['status'] == 'error'


def test_cli_mixed_tampered_case_is_unknown_and_explicit_replay_refuses(case):
    root, output = _mixed_case(case)
    env = dict(os.environ, PYTHONPATH=SRC)
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'investigate', str(root), '--json'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 3 and not result.stderr
    assert json.loads(result.stdout)['status'] == 'unknown'
    replay = subprocess.run([sys.executable, '-m', 'policy_doctor', 'reproduce-case', str(output), '--json'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert replay.returncode == 2


def test_default_mcp_discovery_does_not_verify_mixed_or_incomplete_case(case):
    root, output = _mixed_case(case)
    incoming = root / 'incoming'
    incoming.mkdir()
    incomplete = _metadata_case(incoming)
    (incomplete / 'manifest.json').unlink()

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(root)],
            env=dict(os.environ, PYTHONPATH=SRC))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                # Isolate D4 from the separately broken incomplete transfer.
                (incomplete / 'READY.json').rename(incoming / 'temporarily-held-marker.json')
                shutil.rmtree(incomplete)
                mixed = await client.call_tool('investigate_run_directory', {'directory': str(root)})
                assert not mixed.isError and mixed.structuredContent['status'] == 'unknown'
                assert mixed.structuredContent['audits'][0]['status'] == 'pass'
                replay = await client.call_tool('reproduce_incident_case', {'directory': str(output)})
                assert replay.isError
                rebuilt = _metadata_case(incoming)
                (rebuilt / 'manifest.json').unlink()
                transfer = await client.call_tool('investigate_run_directory', {'directory': str(rebuilt)})
                assert not transfer.isError and transfer.structuredContent['status'] == 'unknown'
                assert not transfer.structuredContent['audits']
                assert not transfer.structuredContent['inventory']['checkpoints']

    asyncio.run(exercise())
