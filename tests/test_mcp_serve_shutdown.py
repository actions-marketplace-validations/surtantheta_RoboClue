"""A real stdio MCP session that ends normally exits zero without a shutdown traceback."""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip('mcp')

SRC = Path(__file__).parents[1] / 'src'
INITIALIZE = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
              'params': {'protocolVersion': '2025-06-18', 'capabilities': {},
                         'clientInfo': {'name': 'shutdown-test', 'version': '0'}}}


def _send(process, message):
    process.stdin.write((json.dumps(message) + '\n').encode())
    process.stdin.flush()


def _reply(lines, wanted, timeout=60):
    while True:
        line = lines.get(timeout=timeout)
        assert line, 'server closed stdout before replying'
        message = json.loads(line)
        if message.get('id') == wanted:
            return message


@pytest.mark.parametrize('arguments', [['mcp', 'serve'], ['mcp', 'serve', '--advanced']])
def test_client_disconnect_after_a_tool_call_is_a_clean_zero_exit(tmp_path, arguments):
    # The transport closes stdout only after a served tool call; initialize alone
    # never reached the shutdown flush that used to raise on a closed stream.
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'schema_version': 1, 'kind': 'provenance_register_input', 'stacks': ['robosuite']}))
    env = dict(os.environ, PYTHONPATH=str(SRC))
    process = subprocess.Popen([sys.executable, '-B', '-m', 'roboclue', *arguments, '--root', str(tmp_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    lines = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line) for line in iter(process.stdout.readline, b'')] + [lines.put(b'')],
                              daemon=True)
    reader.start()
    try:
        _send(process, INITIALIZE)
        assert _reply(lines, 1)['result']['serverInfo']['name'] == 'RoboClue'
        _send(process, {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        _send(process, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
                        'params': {'name': 'preflight_check', 'arguments': {'source': str(source)}}})
        reply = _reply(lines, 2)
        assert 'error' not in reply and not reply['result'].get('isError'), reply
        process.stdin.close()
        process.wait(timeout=60)
        reader.join(timeout=10)
        stderr = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 0, stderr.decode(errors='replace')[-2000:]
    assert b'Traceback' not in stderr and b'I/O operation on closed file' not in stderr
