"""Genuine retained target/mask replay and first-use refusal controls."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from policy_doctor import native_training_demo as demo
from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import TargetContract


def test_saved_native_fault_correction_and_unknown_keep_one_pin(tmp_path):
    result = demo.run_demo(tmp_path / 'replay')
    checks = result['checks']
    assert {phase: check['status'] for phase, check in checks.items()} == {
        'healthy': 'pass', 'fault': 'error', 'corrected': 'pass', 'unknown': 'unknown'}
    assert checks['fault']['differing_target_values'] == 1400
    mask_findings = [f for f in checks['fault']['findings'] if f['field'] == 'inputs.action_mask']
    assert len(mask_findings) == 2
    assert all(f['expected_dtype'] == 'float64' and f['observed_dtype'] == 'float32' for f in mask_findings)
    assert checks['unknown']['code'] == 'uncovered_source_sample'
    assert len(checks['unknown']['uncovered_source_keys']) == 2
    pin = result['criterion_sha256']
    assert {report['criterion_sha256'] for report in checks.values()} == {pin}
    assert pin != demo.ORIGINAL_CRITERION_SHA256
    assert result['overall_status'] == 'unknown'
    assert result['model_forward_calls'] == result['optimizer_steps'] == 0
    assert result['references_regenerated_after_fault'] is False
    saved = Path(result['artifacts']['reference_contract'])
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == pin
    assert Path(result['artifacts']['criterion_pin']).read_text().strip() == pin
    assert json.loads(Path(result['artifacts']['summary']).read_text()) == result
    for phase, filename in result['artifacts']['reports'].items():
        assert json.loads(Path(filename).read_text()) == checks[phase]


def test_reference_pin_is_retained_before_first_candidate_check(tmp_path, monkeypatch):
    check = TargetContract.check
    seen = []
    output = tmp_path / 'replay'

    def observe(self, batch, keys):
        pin = (output / 'criterion.sha256').read_text().strip()
        assert pin == self.criterion_sha256
        assert hashlib.sha256((output / 'reference-contract.json').read_bytes()).hexdigest() == pin
        seen.append(pin)
        return check(self, batch, keys)

    monkeypatch.setattr(TargetContract, 'check', observe)
    demo.run_demo(output)
    assert len(seen) == 4 and len(set(seen)) == 1


def test_capture_identity_reordering_and_uncovered_keys_are_not_guessed(tmp_path):
    result = demo.run_demo(tmp_path / 'replay')
    contract = TargetContract.load(result['artifacts']['reference_contract'],
                                   expected_sha256=result['criterion_sha256'])
    files = demo._verified_fixtures()
    capture, batch = demo._capture(files['healthy.json'])
    keys = capture['source_keys']
    # Same rows incorrectly paired with the other real source is a detectable error.
    assert contract.check(batch, list(reversed(keys)))['status'] == 'error'
    unknown, _ = demo._capture(files['unknown.json'])
    report = contract.check(batch, unknown['source_keys'])
    assert report['status'] == 'unknown' and report['samples_checked'] == 0
    assert report['code'] == 'uncovered_source_sample'


def test_packaged_receipts_preserve_real_dtypes_and_source_hashes():
    files = demo._verified_fixtures()
    provenance = json.loads(files['provenance.json'])
    ref, reference = demo._capture(files['reference.json'], reference=True)
    assert provenance['original_criterion_sha256'] == demo.ORIGINAL_CRITERION_SHA256
    assert provenance['original_verifier']['numpy'] == '1.26.4'
    assert provenance['dataset']['license'] == 'CC-BY-4.0'
    for key, p in zip(ref['source_keys'], ref['provenance'], strict=True):
        source = json.dumps(p['source'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert key == 'sha256:' + hashlib.sha256(source).hexdigest()
    assert reference['inputs']['action'].dtype == np.float32
    assert reference['inputs']['action_mask'].dtype == np.float64
    assert files['healthy.json'] == files['corrected.json']
    for phase in ('healthy', 'fault', 'corrected', 'unknown'):
        name = phase + '.json'
        assert hashlib.sha256(files[name]).hexdigest() == provenance['fixtures'][name]['original_raw_sha256']
    assert provenance['fixtures']['reference.json']['original_raw_sha256'] == demo.ORIGINAL_CRITERION_SHA256
    assert b'Attribution 4.0 International' in files['LICENSE-CC-BY-4.0.txt']
    assert b'/home/' not in b''.join(files.values())


@pytest.mark.parametrize('kind', ['directory', 'file', 'fifo', 'dangling-symlink'])
def test_existing_or_special_destination_is_refused(tmp_path, kind):
    output = tmp_path / 'output'
    if kind == 'directory':
        output.mkdir()
    elif kind == 'file':
        output.write_bytes(b'retain me')
    elif kind == 'fifo':
        os.mkfifo(output)
    else:
        output.symlink_to(tmp_path / 'absent')
    with pytest.raises(InputError, match='new output directory|symlink'):
        demo.run_demo(output)
    if kind == 'file':
        assert output.read_bytes() == b'retain me'


def test_symlink_parent_and_fixture_overlap_are_refused(tmp_path):
    (tmp_path / 'alias').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        demo.run_demo(tmp_path / 'alias' / 'output')
    with pytest.raises(InputError, match='overlaps'):
        demo.run_demo(Path(demo._fixtures()) / 'new-output')
    with pytest.raises(InputError, match='parent must already exist'):
        demo.run_demo(tmp_path / 'missing-parent' / 'output')


@pytest.mark.parametrize('name', ['fault.json', 'reference.json', 'manifest.json', 'NOTICE.txt'])
def test_tampered_fixture_is_rejected_before_reference_or_output(tmp_path, monkeypatch, name):
    copied = tmp_path / 'fixtures'
    copied.mkdir()
    source = demo._fixtures()
    for entry in source.iterdir():
        if entry.is_file():
            (copied / entry.name).write_bytes(entry.read_bytes())
    (copied / name).write_bytes((copied / name).read_bytes() + b' ')
    monkeypatch.setattr(demo, '_fixtures', lambda: copied)
    output = tmp_path / 'output'
    with pytest.raises(InputError, match='integrity|manifest differs'):
        demo.run_demo(output)
    assert not output.exists()


def test_replay_never_accepts_replaced_contract_or_runtime(tmp_path, monkeypatch):
    result = demo.run_demo(tmp_path / 'replay')
    path = Path(result['artifacts']['reference_contract'])
    raw = path.read_bytes()
    value = json.loads(raw)
    value['intent']['objective'] = 'replace after fault'
    path.write_text(json.dumps(value))
    with pytest.raises(InputError, match='retained pin'):
        TargetContract.load(path, expected_sha256=result['criterion_sha256'])
    path.write_bytes(raw)
    import policy_doctor.training_guard as guard
    verifier = guard._verifier()
    monkeypatch.setattr(guard, '_verifier', lambda: {**verifier, 'numpy': 'other-runtime'})
    with pytest.raises(InputError, match='Unsupported, changed'):
        TargetContract.load(path, expected_sha256=result['criterion_sha256'])


def test_cli_default_is_concise_and_links_actual_reports(tmp_path, capsys):
    output = tmp_path / 'cli-default'
    assert demo.main(['--output', str(output)]) == 0
    captured = capsys.readouterr()
    assert not captured.err and len(captured.out.encode()) <= 4096
    saved = json.loads((output / 'summary.json').read_text())
    for phase in ('healthy', 'fault', 'corrected'):
        assert f"{phase.capitalize()} {saved['checks'][phase]['status'].upper()}" in captured.out
    assert '1,400 changed target values; 2 mask dtype findings' in captured.out
    assert 'Uncovered UNKNOWN: 2 source keys outside the reference; 0 samples paired' in captured.out
    assert 'same retained pin' in captured.out
    assert 'Overall research UNKNOWN' in captured.out
    assert 'expected controls reproduced (exit 0)' in captured.out
    assert 'no new native/model execution, training or physical correctness claim' in captured.out
    assert saved['criterion_sha256'] in captured.out
    for field in ('reference_contract', 'criterion_pin', 'summary', 'inputs'):
        assert saved['artifacts'][field] in captured.out
    for phase, path in saved['artifacts']['reports'].items():
        assert path in captured.out
        assert json.loads(Path(path).read_text()) == saved['checks'][phase]
    # Provenance is retained in the saved summary instead of overwhelming stdout.
    assert saved['source_provenance']['qualification']['raw_sha256'] not in captured.out


def test_cli_json_is_full_saved_summary_and_actual_check_reports(tmp_path, capsys):
    output = tmp_path / 'cli-json'
    assert demo.main(['--output', str(output), '--json']) == 0
    captured = capsys.readouterr()
    assert not captured.err
    printed = json.loads(captured.out)
    saved = json.loads((output / 'summary.json').read_text())
    assert printed == saved
    assert printed['source_provenance']['original_criterion_sha256'] == demo.ORIGINAL_CRITERION_SHA256
    assert printed['overall_status'] == 'unknown'
    for phase, path in printed['artifacts']['reports'].items():
        assert printed['checks'][phase] == json.loads(Path(path).read_text())


@pytest.mark.parametrize('flags', [[], ['--json']])
def test_cli_refusal_exit_retains_existing_evidence(tmp_path, capsys, flags):
    output = tmp_path / 'previous-replay'
    output.mkdir()
    retained = output / 'summary.json'
    retained.write_bytes(b'previous evidence must remain')
    assert demo.main(['--output', str(output), *flags]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert 'Saved native demo refused' in captured.err
    assert 'new output directory' in captured.err
    assert retained.read_bytes() == b'previous evidence must remain'
    assert set(output.iterdir()) == {retained}
