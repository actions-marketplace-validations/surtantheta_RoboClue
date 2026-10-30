"""Missing retained artifacts must identify the input across public front doors."""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from test_preflight_groot_reach import plain_dataset

from policy_doctor.artifacts import InputError
from policy_doctor.bulk_video_recovery import _seal_read
from policy_doctor.cli import human, main
from policy_doctor.conversion_fidelity import _read_small
from policy_doctor.rollouts import _source


def test_missing_conversion_reference_and_no_follow_alias_name_the_input(tmp_path):
    missing = tmp_path / 'absent-report.json'
    with pytest.raises(InputError, match='does not exist') as error:
        _read_small(str(missing))
    assert str(missing) in str(error.value) and 'rerun' in str(error.value)
    original = tmp_path / 'reference.json'
    original.write_text('{}')
    alias = tmp_path / 'alias-report.json'
    alias.symlink_to(original)
    with pytest.raises(InputError, match='without symlink ancestors') as error:
        _read_small(str(alias))
    assert str(alias) in str(error.value)
    raw, evidence = _read_small(str(original))
    assert raw == b'{}' and evidence['sha256'] == hashlib.sha256(raw).hexdigest()


def test_batch_missing_and_special_seals_name_the_exact_input(tmp_path):
    missing = tmp_path / 'absent-patch-case'
    with pytest.raises(InputError, match='does not exist') as error:
        _seal_read(missing, 'BATCH_MANIFEST.json', 4096)
    assert str(missing / 'BATCH_MANIFEST.json') in str(error.value)
    assert 'case directory' in str(error.value)
    case = tmp_path / 'case'
    case.mkdir()
    seal = case / 'BATCH_MANIFEST.json'
    os.mkfifo(seal)
    with pytest.raises(InputError, match='not a bounded regular file') as error:
        _seal_read(case, seal.name, 4096)
    assert str(seal) in str(error.value)


def test_actual_cli_missing_artifacts_name_the_path(tmp_path, capsys):
    missing_case = tmp_path / 'absent-patch-case'
    assert main(['reproduce-video-references', str(missing_case), '--patch', '--json']) == 2
    error = json.loads(capsys.readouterr().err)['error']['message']
    assert str(missing_case) in error and 'case directory' in error
    dataset = plain_dataset(tmp_path)
    missing_reference = tmp_path / 'absent-report.json'
    assert main(['preflight', str(dataset), '--conversion-reference', str(missing_reference), '--json']) == 2
    error = json.loads(capsys.readouterr().err)['error']['message']
    assert str(missing_reference) in error and 'does not exist' in error


def test_missing_recording_names_the_original_path_in_api_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InputError, match='does not exist') as error:
        _source('reference.npz')
    assert "'reference.npz'" in str(error.value) and 'rerun' in str(error.value)
    assert main(['compare-rollouts', 'reference.npz', 'candidate.npz', '--json']) == 2
    message = json.loads(capsys.readouterr().err)['error']['message']
    assert 'reference.npz' in message and 'does not exist' in message


@pytest.mark.parametrize('finding', [
    {'field': 'state.layout', 'status': 'error', 'reason': 'Declared source layout differs.'},
    {'quantity': 'camera_mapping', 'reference': 'wrist', 'candidate': 'agentview'},
])
def test_human_retained_candidate_keeps_native_findings_without_a_code(finding):
    report = {'kind': 'incident_candidate_check', 'status': 'error', 'case_verified': True,
              'candidate_evaluated': True, 'expected_status': 'pass',
              'candidate': {'kind': 'native_observation_source', 'status': 'error', 'findings': [finding]},
              'scope': 'Retained native input criterion.'}
    text = human(report)
    assert json.dumps(finding, ensure_ascii=False) in text
    assert 'candidate evaluated: True' in text and 'expected pass' in text


def test_human_retained_candidate_uses_literal_observed_and_expected_roles():
    report = {'kind': 'incident_candidate_check', 'status': 'error', 'case_verified': True,
              'candidate_evaluated': True, 'expected_status': 'pass',
              'candidate': {'kind': 'episode_windows', 'status': 'error',
                            'findings': [{'code': 'target', 'observed': 17, 'expected': 19}]}, 'scope': ''}
    assert 'First finding: target; observed 17; expected 19.' in human(report)


def test_actual_stdio_missing_artifacts_name_the_path(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    dataset = plain_dataset(tmp_path)
    absent_case = tmp_path / 'absent-patch-case'
    absent_reference = tmp_path / 'absent-report.json'
    case_root = tmp_path / 'cases'
    case_root.mkdir()

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'policy_doctor', 'serve', '--root', str(tmp_path),
                  '--case-root', str(case_root), '--allow-hub-media'],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                for tool, arguments, name in [
                    ('reproduce_dataset_video_references', {'directory': str(absent_case), 'include_patch': True},
                     str(absent_case)),
                    ('preflight_check', {'source': str(dataset), 'conversion_reference': str(absent_reference)},
                     str(absent_reference)),
                ]:
                    result = await client.call_tool(tool, arguments)
                    assert result.isError
                    assert name in '\n'.join(item.text for item in result.content if hasattr(item, 'text'))
    asyncio.run(asyncio.wait_for(run(), timeout=30))
