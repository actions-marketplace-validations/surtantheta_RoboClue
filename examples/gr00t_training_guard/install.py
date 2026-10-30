"""Install the private pinned GR00T preview into a new Linux CPython3.12 venv."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import venv
from itertools import islice
from pathlib import Path


def verify_packet(root):
    """Check all delivered files with fixed bounds before opening large bodies."""
    root = Path(root)
    path = root / 'manifest.json'
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 4 * 1024**2:
            raise ValueError('Preview manifest must be a regular file within4MiB')
        manifest = json.loads(stream.read(4 * 1024**2 + 1))
    if (type(manifest) is not dict or manifest.get('schema_version') != 1
            or type(manifest.get('schema_version')) is not int
            or type(manifest.get('profile')) is not str or not manifest['profile']
            or type(manifest.get('files')) is not dict or not 1 <= len(manifest['files']) <= 8192):
        raise ValueError('Unsupported preview manifest')
    entries = list(islice(root.rglob('*'), 16385))
    if len(entries) > 16384:
        raise ValueError('Preview has too many filesystem entries')
    actual, total = set(), 0
    for path in entries:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f'Preview contains a link or special file: {path.relative_to(root)}')
        if stat.S_ISREG(mode):
            actual.add(path.relative_to(root).as_posix())
            total += path.stat().st_size
            if path.stat().st_size > 8 * 1024**3 or total > 32 * 1024**3:
                raise ValueError('Preview exceeds its fixed file or total byte budget')
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Preview has missing or extra files; use the unchanged delivered packet')
    for relative, record in manifest['files'].items():
        if (Path(relative).is_absolute() or '..' in Path(relative).parts or type(record) is not dict
                or type(record.get('size_bytes')) is not int or not 0 <= record['size_bytes'] <= 8 * 1024**3
                or type(record.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
            raise ValueError('Preview contains an invalid relative file identity')
        path = root / relative
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != record['size_bytes']:
                raise ValueError(f'Preview file size/type changed: {relative}')
            if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
                raise ValueError(f'Preview file contents changed: {relative}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, type=Path, help='new environment outside this packet')
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 12) or sys.platform != 'linux' or platform.machine() != 'x86_64':
            raise ValueError('This pinned preview requires Linux x86_64 and CPython3.12')
        root = Path(__file__).resolve().parent
        environment = args.environment.absolute()
        if environment.exists() or environment.is_symlink() or environment.is_relative_to(root):
            raise ValueError('--environment must name a new directory outside the packet')
        if any(path.is_symlink() for path in environment.parents):
            raise ValueError('--environment must not traverse a symlink')
        manifest = verify_packet(root)
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / 'bin/python'
        command = [str(python), '-I', '-B', '-m', 'pip', 'install', '--no-cache-dir', '--no-index',
                   '--no-deps', '--require-hashes', '--no-compile', '--find-links', str(root / 'wheels'),
                   '-r', str(root / 'requirements.lock')]
        subprocess.run(command, check=True)
        subprocess.run([str(python), '-I', '-B', '-m', 'pip', 'check'], check=True)
        subprocess.run([str(python), '-I', '-B', '-c',
                        'import numpy,torch;from torchcodec.decoders import VideoDecoder;'
                        'from policy_doctor.groot_training_guard import Gr00tTrainingGuard;'
                        'assert numpy.__version__=="1.26.4";assert torch.__version__=="2.9.0+cu126";'
                        'print("Installed native-training dependencies and guard import successfully")'], check=True)
        verify_packet(root)
        result = {'status': 'installed', 'profile': manifest['profile'], 'python': str(python),
                  'existing_site_packages_used': False, 'network_required': False,
                  'scope': 'Pinned Linux CPython3.12/CUDA12.6 dependencies; GPU execution is checked separately.'}
        with (environment / 'roboclue-install.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, RecursionError, subprocess.CalledProcessError) as error:
        print(json.dumps({'status': 'install_error', 'message': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
