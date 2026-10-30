import hashlib
import json
import os
import shlex

import numpy as np
import pytest

from policy_doctor import product_demo
from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import TargetContract
from policy_doctor.training_guard_capture import check_capture, load_capture


def test_actual_source_derived_copy_rechecks_against_original_criterion(tmp_path):
    output = tmp_path / 'demo'
    result = product_demo.run_demo(output)
    pin = (output / 'criterion.sha256').read_text().strip()
    assert result['criterion_sha256'] == pin
    assert {key: value['status'] for key, value in result['checks'].items()} == {
        'fault': 'error', 'corrected': 'pass', 'healthy': 'pass', 'unknown': 'unknown',
    }
    source = json.loads((output / 'source.json').read_text())
    broken, source_keys = load_capture(output / 'broken.json')
    corrected, corrected_keys = load_capture(output / 'corrected.json')
    assert source_keys == corrected_keys
    np.testing.assert_array_equal(
        corrected['inputs']['action'].flatten(),
        np.asarray([row['next_position'] - row['position'] for row in source['rows']], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        broken['inputs']['action'].flatten(),
        np.asarray([row['next_position'] for row in source['rows']], dtype=np.float32),
    )
    for name, expected in [('broken', 'error'), ('corrected', 'pass'), ('healthy', 'pass'), ('unknown', 'unknown')]:
        replay = check_capture(output / 'reference-contract.json', pin, output / (name + '.json'))
        assert replay['status'] == expected
        assert replay['criterion_sha256'] == pin
    contract = TargetContract.load(output / 'reference-contract.json', expected_sha256=pin)
    assert contract.document['intent']['source_sha256'] == hashlib.sha256((output / 'source.json').read_bytes()).hexdigest()
    assert result['model_forward_calls'] == result['optimizer_steps'] == 0
    assert result['data_origin'] == 'synthetic'
    assert result['references_regenerated_after_fault'] is False
    assert result['original_candidate_unchanged'] is True
    assert result['unassessed'] == ['your intended recipe', 'other source rows', 'training', 'model quality', 'robot behavior']


def test_pin_and_original_remain_unchanged_before_and_through_candidate_checks(tmp_path, monkeypatch):
    output = tmp_path / 'demo'
    original_check = product_demo.check_capture
    calls = []

    def observe(contract_path, pin, capture_path):
        assert (output / 'criterion.sha256').read_text().strip() == pin
        raw = contract_path.read_bytes()
        if calls:
            assert raw == calls[0]['reference']
            assert (output / 'broken.json').read_bytes() == calls[0]['broken']
        calls.append({'reference': raw, 'broken': (output / 'broken.json').read_bytes()})
        return original_check(contract_path, pin, capture_path)

    monkeypatch.setattr(product_demo, 'check_capture', observe)
    product_demo.run_demo(output)
    assert len(calls) == 4


def test_three_line_default_is_plain_words_and_exact_quoted_replay_command(tmp_path, capsys):
    output = tmp_path / "new demo's evidence"
    assert product_demo.main(['--output', str(output)]) == 0
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith('PASS: synthetic demo')
    assert 'absolute positions' in lines[1]
    assert len((output / 'criterion.sha256').read_text().strip()) == 64
    assert (output / 'criterion.sha256').read_text().strip() not in out.out
    command = shlex.split(lines[2].removeprefix('Next: '))
    assert command == ['roboclue', 'check', str(output / 'corrected.json'), '--reference',
                       str(output / 'reference-contract.json'), '--pin-file', str(output / 'criterion.sha256')]
    assert not out.err


def test_json_output_is_full_saved_report(tmp_path, capsys):
    output = tmp_path / 'demo'
    assert product_demo.main(['--output', str(output), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / 'summary.json').read_text())
    assert result['checks']['fault']['differing_target_values'] == 3
    assert result['checks']['unknown']['status'] == 'unknown'


def test_help_names_the_public_command(capsys):
    with pytest.raises(SystemExit) as result:
        product_demo.main(['--help'])
    assert result.value.code == 0
    assert 'usage: roboclue demo' in capsys.readouterr().out


def test_explain_keeps_wider_scope_and_unknown_visible(tmp_path, capsys):
    assert product_demo.main(['--output', str(tmp_path / 'demo'), '--explain']) == 0
    out = capsys.readouterr().out
    assert 'UNKNOWN' in out
    assert 'training and robot behavior are unassessed' in out
    assert 'not regenerated' in out


@pytest.mark.parametrize('kind', ['directory', 'file', 'fifo', 'symlink'])
def test_existing_destinations_never_open_or_overwrite(tmp_path, kind):
    output = tmp_path / 'evidence'
    if kind == 'directory':
        output.mkdir()
    elif kind == 'file':
        output.write_text('original evidence')
    elif kind == 'fifo':
        os.mkfifo(output)
    else:
        output.symlink_to(tmp_path / 'missing')
    before = output.lstat()
    with pytest.raises(InputError):
        product_demo.run_demo(output)
    assert output.lstat().st_ino == before.st_ino
    assert output.lstat().st_mode == before.st_mode
    if kind == 'file':
        assert output.read_text() == 'original evidence'


def test_no_evidence_is_created_through_symlink_parent(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        product_demo.run_demo(alias / 'demo')
    assert not (real / 'demo').exists()


def test_symlink_cannot_be_hidden_by_parent_traversal(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        product_demo.run_demo(alias / '..' / 'demo')
    assert not (tmp_path / 'demo').exists()


def test_missing_parent_refuses_without_creating_it(tmp_path):
    with pytest.raises(InputError, match='parent'):
        product_demo.run_demo(tmp_path / 'missing' / 'demo')
    assert not (tmp_path / 'missing').exists()


@pytest.mark.parametrize('character', ['\n', '\r', '\x1b', '\x7f'])
def test_output_path_control_characters_refused_before_creation(tmp_path, character):
    output = tmp_path / ('demo' + character + 'x')
    with pytest.raises(InputError, match='control characters'):
        product_demo.run_demo(output)
    assert not output.exists()


def test_repeated_run_refuses_without_replacing_any_existing_evidence(tmp_path, capsys):
    output = tmp_path / 'demo'
    assert product_demo.main(['--output', str(output)]) == 0
    capsys.readouterr()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert product_demo.main(['--output', str(output)]) == 2
    assert 'choose another --output' in capsys.readouterr().err
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_input_errors_remain_structured_only_when_json_requested(tmp_path, capsys):
    assert product_demo.main(['--output', str(tmp_path), '--json']) == 2
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out)['kind'] == 'demo_input_error'


def test_control_failure_is_not_printed_as_demo_success(tmp_path, monkeypatch, capsys):
    def invalid(*_args):
        return {'status': 'pass'}

    monkeypatch.setattr(product_demo, 'check_capture', invalid)
    assert product_demo.main(['--output', str(tmp_path / 'demo')]) == 2
    result = capsys.readouterr()
    assert not result.out
    assert 'control did not reproduce' in result.err


def test_reference_tampering_during_check_is_not_accepted(tmp_path, monkeypatch):
    original = product_demo.check_capture

    def tamper(contract_path, pin, capture_path):
        result = original(contract_path, pin, capture_path)
        contract_path.write_bytes(contract_path.read_bytes() + b' ')
        return result

    monkeypatch.setattr(product_demo, 'check_capture', tamper)
    with pytest.raises(InputError, match='criterion|hash|SHA|pin'):
        product_demo.run_demo(tmp_path / 'demo')
