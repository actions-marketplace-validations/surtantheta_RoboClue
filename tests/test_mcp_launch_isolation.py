"""Generated launchers import the installed entry point from an untrusted cwd."""
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from policy_doctor import product_mcp


@pytest.mark.parametrize('shadow', ['module', 'package', 'pythonpath'])
def test_generated_launcher_ignores_consumer_modules_and_injected_pythonpath(tmp_path, monkeypatch, shadow):
    # A stdlib-only installed stand-in makes this test independent of optional
    # MCP dependencies. Actual installed-server negotiation is checked separately.
    environment = tmp_path / 'environment'
    venv.create(environment, with_pip=False)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    site = Path(subprocess.check_output([str(python), '-I', '-B', '-c',
        'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip())
    installed = site / 'roboclue'
    installed.mkdir()
    (installed / '__init__.py').write_text('')
    (installed / '__main__.py').write_text('import json,sys; print(json.dumps({"origin":"installed fixture",'
        '"isolated":sys.flags.isolated,"bytecode_disabled":sys.dont_write_bytecode,"arguments":sys.argv[1:]}))')
    consumer = tmp_path / 'consumer'
    consumer.mkdir()
    injected = tmp_path / 'injected'
    injected.mkdir()
    hostile = 'print("CONSUMER_MODULE_EXECUTED")\n'
    if shadow == 'module':
        (consumer / 'roboclue.py').write_text(hostile)
    elif shadow == 'package':
        (consumer / 'roboclue').mkdir()
        (consumer / 'roboclue/__init__.py').write_text(hostile)
        (consumer / 'roboclue/__main__.py').write_text(hostile)
    else:
        (injected / 'roboclue.py').write_text(hostile)
    monkeypatch.setattr(sys, 'executable', str(python))
    config, command = product_mcp.configuration([str(consumer)])
    configured = config['mcpServers']['roboclue']
    assert command == [configured['command'], *configured['args']]
    result = subprocess.run(command, cwd=consumer, env=dict(os.environ, PYTHONPATH=str(injected)),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and 'CONSUMER_MODULE_EXECUTED' not in result.stdout
    actual = json.loads(result.stdout)
    assert actual['origin'] == 'installed fixture' and actual['isolated'] == 1 and actual['bytecode_disabled']
    assert actual['arguments'] == ['mcp', 'serve', '--root', str(consumer)]
