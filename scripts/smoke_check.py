"""Verify the installed metadata front door outside the checkout, without network."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import policy_doctor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mcp', action='store_true')
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    package = Path(policy_doctor.__file__).resolve()
    assert not package.is_relative_to(checkout / 'src'), 'Use a noneditable installed wheel'
    distribution = importlib.metadata.distribution('roboclue')
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    assert not direct.get('dir_info', {}).get('editable'), 'Editable installations are not wheel qualification'
    assert package == Path(distribution.locate_file('policy_doctor/__init__.py')).resolve()
    assert distribution.version == policy_doctor.__version__
    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME')}
    calls = []
    with tempfile.TemporaryDirectory(prefix='roboclue-installed-check-') as temporary:
        work = Path(temporary)
        identity = subprocess.run([sys.executable, '-I', '-c',
            'import json,policy_doctor; print(json.dumps({"path":policy_doctor.__file__,"version":policy_doctor.__version__}))'],
            cwd=work, env=env, capture_output=True, text=True, timeout=30, check=True)
        child_identity = json.loads(identity.stdout)
        assert Path(child_identity['path']).resolve() == package
        assert child_identity['version'] == policy_doctor.__version__

        def cli(arguments, expected):
            result = subprocess.run([sys.executable, '-I', '-m', 'policy_doctor', *arguments],
                                    cwd=work, env=env, capture_output=True, text=True, timeout=30)
            calls.append({'arguments': arguments, 'exit_code': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr})
            assert result.returncode == expected, calls[-1]
            return result.stdout

        cli(['demo', './robot-example', '--kind', 'policy'], 0)
        command = ['check', './robot-example/checkpoint', '--dataset', './robot-example/dataset']
        short = cli(command, 1)
        assert len(short.encode()) < 2048 and '/steps/0/config/action_names' in short
        assert 'Next:' in short and 'action.processor_order_names' in short
        body = json.loads(cli(command + ['--json'], 1))
        errors = [f for f in body['findings'] if f['status'] == 'error']
        assert len(errors) == 2 and all(f['code'] == 'action.processor_order_names' for f in errors)
        for finding in errors:
            for evidence in finding['evidence']:
                assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
        corrected = json.loads(cli(['check', './robot-example/checkpoint', '--dataset',
                                   './robot-example/corrected_dataset', '--json'], 3))
        assert corrected['counts']['error'] == 0 and corrected['counts']['unknown'] > 0
        scoped = json.loads(cli(['check', './robot-example/checkpoint', '--dataset',
                                 './robot-example/corrected_dataset', '--scope', 'metadata', '--json'], 0))
        assert scoped['assessment_scope'] == 'metadata' and scoped['deployment_status'] == 'not_assessed'
        assert scoped['counts']['unknown'] == 0 and scoped['unassessed_findings']
        assert all(f['code'] == 'deployment.missing' for f in scoped['unassessed_findings'])
        scoped_error = json.loads(cli(command + ['--scope', 'metadata', '--json'], 1))
        assert scoped_error['counts']['error'] == 2 and scoped_error['deployment_status'] == 'not_assessed'
        cli(['check', './robot-example/checkpoint', '--dataset', '', '--scope', 'metadata'], 2)
        assert 'nonempty' in calls[-1]['stderr']
        cli(['check', 'hf://org/policy'], 2)
        assert 'Hub network is disabled' in calls[-1]['stderr']
        cli(['demo', './robot-example', '--kind', 'policy'], 2)
        empty_checkpoint = work / 'empty-checkpoint'
        empty_checkpoint.mkdir()
        (empty_checkpoint / 'config.json').write_text(json.dumps({
            'type': 'act', 'input_features': {}, 'output_features': {}}))
        for label, registry in [('pre', 'normalizer_processor'), ('post', 'unnormalizer_processor')]:
            (empty_checkpoint / f'policy_{label}processor.json').write_text(json.dumps({
                'steps': [{'registry_name': registry, 'config': {'features': {}, 'norm_map': {}}}]}))
        empty_assessment = json.loads(cli(['check', str(empty_checkpoint), '--dataset',
            './robot-example/corrected_dataset', '--scope', 'metadata', '--json'], 3))
        assert empty_assessment['status'] == 'unknown' and empty_assessment['counts']['pass'] == 0
        assert [f['code'] for f in empty_assessment['findings']] == ['metadata.not_assessed']

        async def protocol(enabled):
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=sys.executable,
                args=['-I', '-m', 'policy_doctor', 'serve', '--root', str(work)]
                     + (['--allow-hub'] if enabled else []), env=env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    catalog = (await session.list_tools()).tools
                    assert len(catalog) == (30 if enabled else 29)
                    assert {'dataset_health_card', 'triage_flight_recording',
                            'assess_upgrade_impact', 'check_training_targets'} <= {tool.name for tool in catalog}
                    calls.append({'mcp': 'list_tools', 'allow_hub': enabled,
                                  'tool_names': [tool.name for tool in catalog]})
                    if not enabled:
                        assert all(tool.name != 'check_policy_metadata' for tool in catalog)
                        return
                    for dataset, status in [('dataset', 'error'), ('corrected_dataset', 'unknown')]:
                        response = await session.call_tool('check_policy_metadata', {
                            'source': str(work / 'robot-example/checkpoint'),
                            'dataset': str(work / 'robot-example' / dataset)})
                        assert not response.isError and response.structuredContent['status'] == status
                        calls.append({'mcp': 'check_policy_metadata', 'status': status,
                                      'result': response.structuredContent})
                    compact = await session.call_tool('check_policy_metadata', {
                        'source': str(work / 'robot-example/checkpoint'),
                        'dataset': str(work / 'robot-example/corrected_dataset'),
                        'scope': 'metadata', 'response_format': 'concise'})
                    assert not compact.isError
                    summary = compact.structuredContent
                    assert summary['kind'] == 'checkpoint_check_summary' and summary['status'] == 'pass'
                    assert summary['assessment_scope'] == 'metadata' and summary['deployment_status'] == 'not_assessed'
                    assert len(json.dumps(summary).encode('utf-8')) <= 8192
                    assert summary['unassessed_findings_count'] == 1
                    request = summary['detailed_request']
                    assert request['arguments']['scope'] == 'metadata'
                    detailed = await session.call_tool(request['tool'], request['arguments'])
                    assert not detailed.isError and detailed.structuredContent['status'] == 'pass'
                    canonical = json.dumps(detailed.structuredContent, sort_keys=True, separators=(',', ':'),
                                           ensure_ascii=False, allow_nan=False).encode('utf-8')
                    assert hashlib.sha256(canonical).hexdigest() == summary['full_report']['sha256']
                    calls.extend([{'mcp': 'check_policy_metadata', 'response_format': 'concise', 'result': summary},
                                  {'mcp': 'check_policy_metadata', 'response_format': 'detailed',
                                   'status': detailed.structuredContent['status']}])
                    for response_format in ('concise', 'detailed'):
                        unassessed = await session.call_tool('check_policy_metadata', {
                            'source': str(empty_checkpoint), 'dataset': str(work / 'robot-example/corrected_dataset'),
                            'scope': 'metadata', 'response_format': response_format})
                        assert not unassessed.isError and unassessed.structuredContent['status'] == 'unknown'
                        assert unassessed.structuredContent['counts']['pass'] == 0
                        calls.append({'mcp': 'check_policy_metadata', 'response_format': response_format,
                                      'empty_assessment_status': 'unknown', 'result': unassessed.structuredContent})
                        empty = await session.call_tool('check_policy_metadata', {
                            'source': str(work / 'robot-example/checkpoint'), 'dataset': '',
                            'scope': 'metadata', 'response_format': response_format})
                        assert empty.isError and 'nonempty' in empty.content[0].text
                        calls.append({'mcp': 'check_policy_metadata', 'response_format': response_format,
                                      'empty_dataset_refused': True})
                    denied = await session.call_tool('check_policy_metadata', {
                        'source': 'hf://org/policy', 'dataset': str(checkout)})
                    assert denied.isError and 'outside' in denied.content[0].text
                    calls.append({'mcp': 'check_policy_metadata', 'denied_local_path': True})

        if args.mcp:
            async def run():
                await protocol(False)
                await protocol(True)

            asyncio.run(asyncio.wait_for(run(), timeout=30))
    record = {'status': 'pass', 'version': policy_doctor.__version__,
              'package_path': str(package), 'python': sys.version,
              'network': 'No requests intended; socket activity is not instrumented',
              'distribution_version': distribution.version, 'editable': False,
              'isolated_child_identity': child_identity,
              'synthetic_inputs': True, 'calls': calls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Installed front door passed: {len(calls)} CLI/protocol operations; {args.output}')


if __name__ == '__main__':
    main()
