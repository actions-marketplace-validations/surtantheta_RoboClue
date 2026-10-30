"""First-use refusal cases for the independently delivered private packet."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / 'examples/training_guard/portable/install.py'


@pytest.fixture
def installer():
    spec = importlib.util.spec_from_file_location('portable_installer_test', INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packet(root, **files):
    for name, content in files.items():
        (root / name).write_bytes(content)
    manifest = {'schema': 1, 'profile': 'test',
                'files': {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_clean_packet_and_changed_input(installer, tmp_path):
    packet(tmp_path, sample=b'original')
    assert installer.verify_packet(tmp_path)['profile'] == 'test'
    (tmp_path / 'sample').write_bytes(b'changed')
    with pytest.raises(ValueError, match='input changed: sample'):
        installer.verify_packet(tmp_path)


def test_unexpected_file_is_not_silently_delivered(installer, tmp_path):
    packet(tmp_path, sample=b'original')
    (tmp_path / 'unreviewed').write_bytes(b'extra')
    with pytest.raises(ValueError, match='extra or missing'):
        installer.verify_packet(tmp_path)


def test_packet_symlink_does_not_read_outside(installer, tmp_path):
    packet(tmp_path, sample=b'original')
    (tmp_path / 'sample').unlink()
    (tmp_path / 'sample').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='special file or link'):
        installer.verify_packet(tmp_path)


def test_manifest_traversal_refused(installer, tmp_path):
    packet(tmp_path, sample=b'original')
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['files'] = {'../outside': '0' * 64}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe relative path'):
        installer.verify_packet(tmp_path)


@pytest.mark.parametrize('manifest', [[], {'schema': 1, 'profile': 'test', 'files': []},
                                     {'schema': True, 'profile': 'test', 'files': {'sample': '0' * 64}}])
def test_malformed_manifest_has_a_controlled_error(installer, tmp_path, manifest):
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Invalid preview manifest'):
        installer.verify_packet(tmp_path)


@pytest.mark.parametrize('fifo_name', ['manifest.json', 'sample'])
def test_fifo_input_refused_without_blocking(tmp_path, fifo_name):
    packet(tmp_path, sample=b'original')
    (tmp_path / fifo_name).unlink()
    os.mkfifo(tmp_path / fifo_name)
    code = ('import importlib.util;from pathlib import Path;'
            f's=importlib.util.spec_from_file_location("installer",{str(INSTALLER)!r});'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            f'root=Path({str(tmp_path)!r});\n'
            'try: m.verify_packet(root)\n'
            'except ValueError: raise SystemExit(0)\n'
            'raise SystemExit("FIFO unexpectedly accepted")')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
