"""Independent controls for native CLI scalar decoding at the resume boundary.

Actual Draccus0.11.6 Path decoding reproduced the hash-comment discrepancy in a
separate retained probe. These generated controls use the dataset extra's YAML
parser and a recorder sentinel; no native recorder or robot is executed.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import yaml
from test_salvage import hashes, healthy

from policy_doctor.artifacts import InputError
from policy_doctor.product_cli import main
from policy_doctor.resume_guard import assess_resume, bind_recording_command


def move_healthy(tmp_path, name):
    source = healthy(tmp_path / 'fixture')
    target = tmp_path / name
    source.rename(target)
    return target


def recorder(tmp_path):
    program = tmp_path / 'bin/lerobot-record'
    program.parent.mkdir()
    program.write_text('#!' + sys.executable + '\nfrom pathlib import Path\n'
                       + 'Path(' + repr(str(tmp_path / 'started')) + ').write_text("started")\n')
    program.chmod(0o755)
    return str(program)


def test_realistic_hash_in_folder_name_cannot_clear_a_different_native_root(tmp_path, capsys):
    checked = healthy(tmp_path / 'recording #1')
    decoded = yaml.safe_load(str(checked))
    assert decoded != str(checked)
    native_root = tmp_path / 'recording'
    assert decoded == str(native_root)
    native_root.mkdir()
    (native_root / 'retained-original').write_text('must not be touched')
    before = hashes(checked)
    argv = [recorder(tmp_path), '--dataset.root=' + str(checked), '--resume=true']
    assert main(['guard', 'resume', str(checked), '--dry-run', '--json', '--', *argv]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error['launched'] is False and 'YAML' in error['error']
    assert not (tmp_path / 'started').exists() and hashes(checked) == before
    assert (native_root / 'retained-original').read_text() == 'must not be touched'


@pytest.mark.parametrize('raw', ["'target'", '"target"', ' target', 'target ',
                                  'true', 'null', '123', '[one,two]', '{root: other}', '!untrusted value'])
def test_quotes_whitespace_typed_values_and_tags_do_not_bind_literal_folders(tmp_path, monkeypatch, raw):
    checked = move_healthy(tmp_path, raw)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'target').mkdir(exist_ok=True)
    before = hashes(checked)
    with pytest.raises(InputError, match='YAML'):
        bind_recording_command(checked, ['lerobot-record', '--dataset.root=' + raw, '--resume=true'])
    assert hashes(checked) == before


@pytest.mark.parametrize('raw', ['include settings.yml', 'includeWhatever'])
def test_draccus_include_keyword_cannot_bypass_equal_yaml_scalar_binding(tmp_path, monkeypatch, raw):
    checked = move_healthy(tmp_path, raw)
    monkeypatch.chdir(tmp_path)
    assert yaml.safe_load(raw) == raw  # Equality alone cannot detect this native include operation.
    with pytest.raises(InputError, match='YAML|include'):
        bind_recording_command(checked, ['lerobot-record', '--dataset.root=' + raw, '--resume=true'])


def test_normal_internal_spaces_preserve_argument_and_declared_root(tmp_path, monkeypatch):
    checked = healthy(tmp_path / 'normal recording')
    monkeypatch.chdir(tmp_path)
    raw = os.path.relpath(checked, tmp_path)
    argv = ['lerobot-record', '--dataset.root=' + raw, '--resume=true', '--dataset.repo_id=me/recording']
    root, preserved = bind_recording_command(checked, argv)
    assert root == checked and preserved == argv
    assert assess_resume(checked, argv)['exit_code'] == 0
