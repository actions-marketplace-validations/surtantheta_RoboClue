"""Install the bounded offline preview into a NEW CPython 3.12 Linux environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import venv
from itertools import islice
from pathlib import Path


def verify_packet(root):
    path = root / 'manifest.json'
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 2 * 1024**2:
            raise ValueError('Packet manifest must be a bounded regular file')
        manifest = json.loads(stream.read(2 * 1024**2 + 1))
    if (type(manifest) is not dict or set(manifest) != {'schema', 'profile', 'files'}
            or type(manifest.get('schema')) is not int or manifest['schema'] != 1
            or type(manifest.get('profile')) is not str or not 1 <= len(manifest['profile']) <= 256
            or type(manifest.get('files')) is not dict or not 1 <= len(manifest['files']) <= 8192):
        raise ValueError('Invalid preview manifest')
    entries = list(islice(root.rglob('*'), 16385))
    if len(entries) > 16384:
        raise ValueError('Packet has too many entries')
    total = 0
    for path in entries:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f'Packet contains a special file or link: {path.relative_to(root)}')
        if stat.S_ISREG(mode):
            total += path.stat().st_size
            if total > 2 * 1024**3:
                raise ValueError('Packet exceeds the preview byte budget')
    for relative, expected in manifest['files'].items():
        path = root / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Manifest contains an unsafe relative path')
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f'Packet file traverses a symlink: {relative}')
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f'Packet input must be a regular file: {relative}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError(f'Packet input changed: {relative}')
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Packet contains extra or missing files; use an unchanged extracted packet')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True, help='new environment outside the read-only packet')
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 12) or sys.platform != 'linux' or platform.machine() != 'x86_64':
            raise ValueError('This preview supports Linux x86_64 / CPython 3.12 only')
        root = Path(__file__).resolve().parent
        environment = args.environment.absolute()
        if environment.exists() or environment.is_symlink():
            raise ValueError('--environment must name an absent directory')
        if environment.is_relative_to(root):
            raise ValueError('--environment must be outside the delivered packet')
        if any(part.is_symlink() for part in environment.parents):
            raise ValueError('--environment must not traverse a symlink')
        manifest = verify_packet(root)
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / 'bin/python'
        command = [str(python), '-I', '-B', '-m', 'pip', 'install', '--no-cache-dir', '--no-index',
                   '--no-deps', '--require-hashes', '--no-compile', '--find-links', str(root / 'wheels'),
                   '-r', str(root / 'requirements.lock')]
        subprocess.run(command, check=True)
        subprocess.run([str(python), '-I', '-B', '-m', 'pip', 'check'], check=True)
        subprocess.run([str(python), '-I', '-B', str(root / 'runtime_check.py')], check=True)
        verify_packet(root)
        receipt = {'status': 'installed', 'profile': manifest['profile'],
                   'python': str(python), 'network_required': False, 'used_existing_site_packages': False}
        with (environment / 'roboclue-install.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
        print(json.dumps(receipt))
        return 0
    except (OSError, ValueError, RecursionError, subprocess.CalledProcessError) as error:
        print(json.dumps({'status': 'install_error', 'message': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
