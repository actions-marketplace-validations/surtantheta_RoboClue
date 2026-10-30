"""Public entry points on typed generated evidence; native execution is separate."""
import asyncio
import copy
import json

import pytest
from test_native_observation_lifecycle import pair

from policy_doctor import cli
from policy_doctor.native_observation_source import CONTRACT
from policy_doctor.server import create_server


def test_cli_complete_retained_lifecycle_and_human_explanation(tmp_path, capsys):
    before, after = pair(tmp_path)
    assert cli.main(['investigate', str(before), '--reference', str(after)]) == 1
    text = capsys.readouterr().out
    assert 'Input preservation criterion: error' in text
    assert 'roll' in text
    assert 'pre-normalization' in text
    case = tmp_path / 'case'
    assert cli.main(['create-case', 'native-observation-source', str(before), str(case),
                     '--corrected', str(after), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'pass'
    assert cli.main(['reproduce-case', str(case), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'pass'
    assert cli.main(['check-case', str(case), str(before), '--json']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'error'
    assert cli.main(['check-case', str(case), str(after), '--json']) == 0
    capsys.readouterr()


def test_cli_native_probe_passes_explicit_contract(tmp_path, monkeypatch, capsys):
    from policy_doctor import native_processing
    requirement = tmp_path / 'requirement.json'
    requirement.write_text(json.dumps({'kind': 'native-observation-source', 'parameters': {'contract': CONTRACT}}))
    calls = []
    monkeypatch.setattr(native_processing, 'load_native_runtimes', lambda _: {'owned': True})
    def probe(*args, **kwargs):
        calls.append((args, kwargs))
        return {'kind': 'native_observation_source_observation', 'status': 'error', 'execution_status': 'complete'}
    monkeypatch.setattr(native_processing, 'probe_native_processing', probe)
    argv = ['probe-native', 'groot_adapter', 'model', 'dataset', '--sample', '0:0',
            '--native-runtime', 'operator.json', '--requirements', str(requirement), '--json']
    assert cli.main(argv) == 1
    capsys.readouterr()
    assert calls[0][1]['contract'] == CONTRACT
    assert calls[0][1]['samples'] == [{'episode_index': 0, 'frame_index': 0}]
    bound = {'contract': copy.deepcopy(CONTRACT), 'model_identity': 'a' * 64}
    requirement.write_text(json.dumps({'kind': 'native-observation-source', 'parameters': bound}))
    assert cli.main(argv) == 2
    assert 'frozen model identity' in capsys.readouterr().err
    assert len(calls) == 1


def test_cli_requirement_case_uses_same_model_and_contract(tmp_path, capsys):
    from policy_doctor.native_observation_source import bind_native_observation_source_parameters
    before, after = pair(tmp_path)
    requirement = tmp_path / 'requirements.json'
    requirement.write_text(json.dumps({'kind': 'native-observation-source',
                                       'parameters': bind_native_observation_source_parameters(after)}))
    assert cli.main(['create-case', 'native-observation-source', str(before), str(tmp_path / 'case'),
                     '--corrected', str(after), '--requirements', str(requirement), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['retained_context']['model_identity']


def test_mcp_existing_case_tools_support_input_preservation(tmp_path):
    pytest.importorskip('mcp')
    before, after = pair(tmp_path)
    output = tmp_path / 'outputs'
    output.mkdir()
    server = create_server([str(tmp_path)], case_root=str(output))
    async def exercise():
        from test_native_processing_mcp import structured
        result = structured(await server.call_tool('create_incident_case', {
            'check': 'native_observation_source', 'source': str(before),
            'destination': str(output / 'case'), 'parameters': {}, 'corrected': str(after)}))
        assert result['status'] == 'pass'
        result = structured(await server.call_tool('check_incident_candidate', {
            'directory': str(output / 'case'), 'candidate': str(before)}))
        assert result['status'] == 'error'
    asyncio.run(exercise())
