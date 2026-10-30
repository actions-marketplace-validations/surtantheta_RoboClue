"""Exercise installed portable checks outside a checkout; synthetic controls only."""
import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mcp', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.find_spec('policy_doctor')
    assert spec and spec.origin and 'site-packages' in Path(spec.origin).parts, spec
    version = importlib.metadata.version('roboclue')
    commands, calls = [], []
    with tempfile.TemporaryDirectory(prefix='roboclue-retained-installed-') as temporary:
        work = Path(temporary)

        def run(label, argv, expected):
            process = subprocess.run([sys.executable, '-I', '-B', '-m', 'policy_doctor', *map(str, argv)],
                cwd=work, text=True, capture_output=True, timeout=40)
            commands.append({'label': label, 'exit_code': process.returncode, 'expected_exit': expected})
            assert process.returncode == expected, (label, process.returncode, process.stderr, process.stdout)
            assert 'Traceback' not in process.stderr
            return json.loads(process.stdout) if process.stdout.strip().startswith('{') else process.stdout

        native, case = work / 'native', work / 'metadata-case'
        run('demo', ['demo', native, '--kind', 'policy', '--json'], 0)
        saved = run('metadata-save-error', ['check', native / 'checkpoint', '--dataset', native / 'dataset',
            '--scope', 'metadata', '--case', case, '--json'], 1)
        assert saved['retained_status'] == 'pass' and saved['original_status'] == 'error'
        healthy = work / 'healthy'
        shutil.copytree(native / 'corrected_dataset', healthy)
        shutil.rmtree(native)
        moved = work / 'moved metadata case'
        case.rename(moved)
        discovered = run('recipient-folder-discovery', ['investigate', moved, '--json'], 3)
        assert discovered['inventory']['retained_checks'] == [str(moved)] and not discovered['audits']
        assert not discovered['inventory']['checkpoints'] and not discovered['inventory']['datasets']
        replay = run('metadata-replay-relocated', ['reproduce-case', moved, '--json'], 0)
        assert replay['reproduction_matches'] and replay['original_status'] == 'error'
        checked = run('metadata-later-pass', ['check-case', moved, healthy, '--json'], 0)
        assert checked['candidate_status'] == 'pass'
        reference, changed, missing = [work / name for name in ('reference.csv', 'changed.csv', 'missing.csv')]
        reference.write_text('index,action\n0,0\n1,1\n2,2\n')
        changed.write_text('index,action\n0,0\n1,5\n2,2\n')
        missing.write_text('index,other\n0,0\n1,1\n2,2\n')
        numeric = work / 'numeric'
        saved = run('recording-save-difference', ['compare-rollouts', reference, changed, '--alignment-key', 'index',
            '--case', numeric, '--json'], 1)
        assert saved['original_status'] == 'different'
        run('recording-replay', ['reproduce-case', numeric, '--json'], 0)
        run('recording-later-pass', ['check-case', numeric, reference, '--json'], 0)
        run('recording-later-recurrence', ['check-case', numeric, changed, '--json'], 1)
        run('recording-later-unknown', ['check-case', numeric, missing, '--json'], 3)
        self_case = work / 'self-case'
        run('same-file-reference', ['retain-check', 'recording-equivalence', reference, self_case,
            '--reference', reference, '--alignment-key', 'index', '--json'], 0)
        run('same-file-replay', ['reproduce-case', self_case, '--json'], 0)
        (self_case / 'expected.json').write_text('{}')
        run('tamper-refusal', ['reproduce-case', self_case, '--json'], 2)
        run('empty-candidate-refusal', ['check-case', numeric, '', '--json'], 2)
        if args.mcp:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            async def exercise():
                parameters = StdioServerParameters(command=sys.executable, args=['-I', '-B', '-m', 'policy_doctor',
                    'serve', '--root', str(work), '--case-root', str(work)], cwd=str(work))
                async with stdio_client(parameters) as (read, write):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        catalog = {t.name: t for t in (await client.list_tools()).tools}
                        assert len(catalog) == 31 and not catalog['retain_artifact_check'].annotations.readOnlyHint
                        assert {'dataset_health_card', 'triage_flight_recording',
                                'assess_upgrade_impact', 'check_training_targets'} <= catalog.keys()
                        replay_action = None
                        for tool, arguments, expected in [
                            ('retain_artifact_check', {'check': 'recording_equivalence', 'source': str(changed),
                                'reference': str(reference), 'destination': str(work / 'mcp-case'),
                                'parameters': {'alignment_key': 'index'}}, 'pass'),
                            ('investigate_run_directory', {'directory': str(work / 'mcp-case')}, 'unknown'),
                            ('reproduce_incident_case', {'directory': str(work / 'mcp-case')}, 'pass'),
                            ('check_incident_candidate', {'directory': str(work / 'mcp-case'), 'candidate': str(reference)}, 'pass'),
                            ('check_incident_candidate', {'directory': str(work / 'mcp-case'), 'candidate': str(changed)}, 'different'),
                            ('check_incident_candidate', {'directory': str(work / 'mcp-case'), 'candidate': str(missing)}, 'unknown')]:
                            if tool == 'reproduce_incident_case':
                                assert replay_action is not None
                                tool, arguments = replay_action['tool'], replay_action['arguments']
                            response = await client.call_tool(tool, arguments)
                            assert not response.isError, response.content
                            assert response.structuredContent['status'] == expected
                            if tool == 'investigate_run_directory':
                                report = response.structuredContent
                                assert not report['audits'] and not report['inventory']['recordings']
                                replay_action = next(action for action in report['next_actions'] if action['tool'] == 'reproduce_incident_case')
                            calls.append({'tool': tool, 'status': expected})

            asyncio.run(exercise())
    output = {'status': 'pass', 'version': version, 'kind': 'installed-retained-check-smoke',
              'package_origin': spec.origin, 'commands': commands, 'mcp_calls': calls,
              'scope': 'Synthetic installed CLI/SDK controls only; no practitioner, native robot, Hub request or human benefit measurement.'}
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'pass', 'version': version, 'commands': len(commands), 'mcp_calls': len(calls)}))


if __name__ == '__main__':
    main()
