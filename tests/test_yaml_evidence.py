import hashlib
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.evidence import inspect_evidence


def test_yaml_finding_resolves_original_wrapper_and_raw_hash(tmp_path):
    path = tmp_path / 'trajectory.yaml'
    raw = b'# original capture\ntrajectory:\n  joint_names: [j]\n  points:\n  - positions: [0.125]\n    time_from_start: {sec: 0, nanosec: 0}\n'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    result = inspect_evidence(str(path), '/trajectory/points/0/positions/0', digest)
    assert result['status'] == 'pass' and result['value'] == .125
    assert result['sha256'] == digest and result['source_format'] == 'YAML'
    assert path.read_bytes() == raw


def test_yaml_comment_change_never_returns_value_for_old_finding(tmp_path):
    path = tmp_path / 'capture.yml'
    raw = b'value: 3\n'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(b'# comment-only change\n' + raw)
    result = inspect_evidence(str(path), '/value', digest)
    assert result['status'] == 'error' and result['code'] == 'evidence.changed'
    assert 'value' not in result


def test_yaml_pointer_escapes_and_missing_coordinate(tmp_path):
    path = tmp_path / 'mapping.yaml'
    path.write_text('"a/b": {"~joint": [3, 4]}\n')
    assert inspect_evidence(str(path), '/a~1b/~0joint/1')['value'] == 4
    assert inspect_evidence(str(path), '/a~1b/~0joint/01')['status'] == 'unknown'
    assert inspect_evidence(str(path), '/a~1b/~0joint/2')['status'] == 'unknown'


@pytest.mark.parametrize('raw', [
    'value: 1\nvalue: 2\n',
    'value: &a [1]\ncopy: *a\n',
    'value: !!python/object/apply:os.system [echo]\n',
    'value: .nan\n',
    'value: 1\n---\nvalue: 2\n',
])
def test_yaml_evidence_reuses_strict_native_parser(tmp_path, raw):
    path = tmp_path / 'unsafe.yaml'
    path.write_text(raw)
    with pytest.raises(InputError):
        inspect_evidence(str(path), '/value')


def test_yaml_during_read_change_is_refused(tmp_path, monkeypatch):
    from policy_doctor import evidence

    path = tmp_path / 'changing.yaml'
    path.write_text('value: 1\n')
    original = evidence._read_document

    def change_after_parse(source):
        result = original(source)
        path.write_text('value: 2\n')
        return result

    monkeypatch.setattr(evidence, '_read_document', change_after_parse)
    with pytest.raises(InputError, match='changed during inspection'):
        inspect_evidence(str(path), '/value')


def test_json_evidence_retains_exact_original_value(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'custom': {'value': [1, 2]}}))
    result = inspect_evidence(str(path), '/custom/value/1')
    assert result['status'] == 'pass' and result['value'] == 2


def test_yaml_finding_through_real_read_only_mcp(tmp_path):
    import asyncio
    import os
    import sys
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    source = allowed / 'source.yaml'
    source.write_text('trajectory: {joint_names: [j], points: [{positions: [0.125]}]}\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outside = tmp_path / 'outside.yaml'
    outside.write_text('value: secret\n')

    async def run():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        params = StdioServerParameters(command=sys.executable, args=['-m', 'policy_doctor', 'serve', '--root', str(allowed)], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                catalog = await session.list_tools()
                tool = next(t for t in catalog.tools if t.name == 'verify_finding_evidence')
                assert tool.annotations.readOnlyHint
                result = await session.call_tool('verify_finding_evidence',
                    {'path': str(source), 'pointer': '/trajectory/points/0/positions/0', 'expected_sha256': digest})
                assert not result.isError
                assert json.loads(result.content[0].text)['value'] == .125
                denied = await session.call_tool('verify_finding_evidence', {'path': str(outside), 'pointer': '/value'})
                assert denied.isError
                source.write_text('# changed\ntrajectory: {}\n')
                stale = await session.call_tool('verify_finding_evidence',
                    {'path': str(source), 'pointer': '/trajectory', 'expected_sha256': digest})
                data = json.loads(stale.content[0].text)
                assert data['status'] == 'error' and 'value' not in data

    asyncio.run(run())
