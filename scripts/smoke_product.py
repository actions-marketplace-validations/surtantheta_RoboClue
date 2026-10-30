"""Exercise the installed public RoboClue workflow outside the source tree.

This is automated technical evidence, not an independent human trial. The demo
and LeRobot capture factory use explicitly generated data, not a native trainer.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mcp', action='store_true')
    args = parser.parse_args()
    import roboclue

    assert Path(roboclue.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.metadata.version('roboclue') == roboclue.__version__
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    calls = []
    with tempfile.TemporaryDirectory(prefix='roboclue-public-smoke-') as folder:
        work = Path(folder)

        def cli(arguments, expected=0):
            started = time.monotonic()
            command = [sys.executable, '-I', '-B', '-m', 'roboclue', *map(str, arguments)]
            result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=60)
            calls.append({'arguments': list(map(str, arguments)), 'exit': result.returncode,
                          'seconds': time.monotonic() - started})
            assert result.returncode == expected, (arguments, result.stdout, result.stderr)
            return result

        assert cli(['--version']).stdout.startswith('RoboClue ')
        help_text = cli(['--help']).stdout
        assert '{check,demo,guard,mcp,advanced}' in help_text
        demo = cli(['demo', '--output', work / 'demo'])
        assert len(demo.stdout.strip().splitlines()) == 3
        assert 'synthetic' in demo.stdout.lower() and calls[-1]['seconds'] < 60
        summary = json.loads((work / 'demo/summary.json').read_text())
        assert summary['original_candidate_unchanged'] and summary['source_unchanged']
        reference = work / 'demo/reference-contract.json'
        pin = work / 'demo/criterion.sha256'
        original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (work / 'demo').iterdir()}
        for name, code, status in [('broken', 1, 'error'), ('corrected', 0, 'pass'), ('unknown', 3, 'unknown')]:
            result = cli(['check', work / f'demo/{name}.json', '--reference', reference, '--pin-file', pin, '--json'], code)
            assert json.loads(result.stdout)['status'] == status
        next_argv = shlex.split(demo.stdout.strip().splitlines()[2].removeprefix('Next: '))
        replay = cli(next_argv[1:])
        assert replay.stdout.startswith('Verdict: PASS') and len(replay.stdout.strip().splitlines()) == 3
        assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (work / 'demo').iterdir()}
        for framework in ('groot', 'lerobot', 'openpi'):
            packet = cli(['guard', 'init', '--framework', framework, '--output', work / framework, '--json'])
            result = json.loads(packet.stdout)
            assert result['assessment_status'] == 'not_run' and result['reference_status'] == 'not_captured'
            assert 'from roboclue import' in (work / framework / 'attach_guard.py').read_text()
            assert not (work / framework / 'startup-reference.json').exists()
        # A genuinely imported application factory invokes the real LeRobot
        # capture helper on generated preprocessed arrays. It is not native training.
        (work / 'example_reference_factory.py').write_text('''import hashlib,json
import numpy as np
from contextlib import contextmanager
from pathlib import Path
@contextmanager
def reviewed():
    try:
        raw=(Path(__file__).parent/'demo/source.json').read_bytes()
        row=json.loads(raw)['rows'][0]
        target=row['next_position']-row['position']
        yield {'batches':[{'index':np.array([0]), 'action':np.array([[[target]]],dtype=np.float32),
                          'action_is_pad':np.array([[False]])}],
               'dataset_id':'sha256:'+hashlib.sha256(raw).hexdigest()}
    finally:
        Path(__file__).with_suffix('.closed').write_text('closed')
''')
        # Execute exactly the public command. The product must resolve the
        # explicitly trusted application factory; the harness adds no import path.
        capture = cli(['guard', 'init', '--framework', 'lerobot', '--factory',
                       'example_reference_factory:reviewed', '--intent',
                       'Preserve generated reviewed relative targets',
                       '--reviewed-intended-recipe', '--output', work / 'captured', '--json'])
        captured = json.loads(capture.stdout)
        assert captured['reference_status'] == 'captured' and captured['assessment_status'] == 'not_run'
        assert (work / 'example_reference_factory.closed').read_text() == 'closed'
        assert (work / 'captured/startup-reference.json').is_file()
        setup = json.loads(cli(['mcp', 'setup', '--root', work, '--json']).stdout)
        server = setup['mcpServers']['roboclue']
        assert Path(server['command']).resolve() == Path(sys.executable).resolve()
        assert server['args'] == ['-I', '-B', '-m', 'roboclue', 'mcp', 'serve', '--root', str(work)]
        mcp_result = None
        if args.mcp:
            mcp_result = asyncio.run(mcp_smoke(work, env, reference, pin, server))
        result = {'status': 'pass', 'scope': 'Installed public interfaces with generated software evidence only; no human, native training or physical test.',
                  'version': roboclue.__version__, 'calls': calls,
                  'demo_fault_corrected_unknown': ['error', 'pass', 'unknown'],
                  'guard_factory': 'real LeRobot capture helper on generated preprocessed arrays; no training',
                  'mcp': mcp_result}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


async def mcp_smoke(work, env, reference, pin, server):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    expected = {'choose_workflow', 'dataset_health_card', 'check_training_targets',
                'triage_flight_recording', 'assess_upgrade_impact', 'preflight_check',
                'check_sim_provenance', 'check_recording_salvage'}
    results = []
    for advanced in (False, True):
        params = StdioServerParameters(command=server['command'],
            args=[*server['args'], *(['--advanced'] if advanced else [])], env=env)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.serverInfo.name == 'RoboClue'
            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            assert names == expected if not advanced else len(names) == 29 and expected <= names
            assert all(t.annotations.readOnlyHint and not t.annotations.destructiveHint for t in tools)
            guide = await session.read_resource('roboclue://start')
            assert 'check_training_targets' in guide.contents[0].text
            plan = await session.call_tool('choose_workflow', {'goal': 'training', 'framework': 'groot'})
            assert not plan.isError
            for name, code in [('broken', 'error'), ('corrected', 'pass'), ('unknown', 'unknown')]:
                response = await session.call_tool('check_training_targets', {
                    'contract': str(reference), 'criterion_sha256': pin.read_text().strip(),
                    'capture': str(work / f'demo/{name}.json')})
                assert not response.isError
                payload = json.loads(response.content[0].text)
                assert payload['status'] == code
            refused = await session.call_tool('check_training_targets', {
                'contract': '/outside-permitted-root/reference.json', 'criterion_sha256': pin.read_text().strip(),
                'capture': str(work / 'demo/corrected.json')})
            assert refused.isError
            results.append({'advanced': advanced, 'tool_count': len(names), 'native_saved_controls': ['error', 'pass', 'unknown'],
                            'outside_root_refused': True})
    return results


if __name__ == '__main__':
    main()
