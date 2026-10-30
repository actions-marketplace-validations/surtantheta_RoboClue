#!/usr/bin/env python3
"""Run source tests with an explicitly selected, already prepared interpreter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    interpreter = parser.add_mutually_exclusive_group(required=True)
    interpreter.add_argument('--venv', type=Path, help='Prepared virtual environment directory')
    interpreter.add_argument('--python', type=Path, help='Explicit Python executable path')
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER, help='Arguments after -- go to pytest')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = (args.venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
              if args.venv else args.python).absolute()
    if not python.is_file():
        parser.error(f'Python executable does not exist: {python}. Prepare .venv-dev as described in DEVELOPMENT.md.')
    probe = subprocess.run(
        [str(python), '-c',
         'import json, sys; import pytest, numpy, safetensors; '
         'print(json.dumps({"python": sys.executable, "version": sys.version.split()[0], '
         '"pytest": pytest.__version__}))'],
        cwd=root, capture_output=True, text=True,
    )
    if probe.returncode:
        print(f'Test environment is incomplete: {python}\n{probe.stderr.strip()}\n'
              'Create a fresh .venv-dev and install ".[dev]" as described in DEVELOPMENT.md. '
              'This command does not change existing environments.', file=sys.stderr)
        return 2
    print(f'Test environment: {json.loads(probe.stdout)}', flush=True)
    env = os.environ.copy()
    # Subprocess CLI checks must import this checkout, even when an older wheel is installed.
    env['PYTHONPATH'] = str(root / 'src')
    pytest_args = args.pytest_args
    if pytest_args[:1] == ['--']:
        pytest_args = pytest_args[1:]
    return subprocess.run([str(python), '-m', 'pytest', *(pytest_args or ['-q'])], cwd=root, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
