"""Synthetic onboarding must show a checkable finding without claiming deployment success."""

import json
import shlex
from pathlib import Path

import pytest

from policy_doctor import hub
from policy_doctor.demo import create_controller_demo, create_policy_demo


def test_policy_demo_exposes_order_fault_and_keeps_corrected_unknown(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('Local generated demo must not use the network')
    monkeypatch.setattr(hub, '_request', no_network)
    destination = tmp_path / 'policy example'
    result = create_policy_demo(destination)
    assert result['kind'] == 'generated_policy_demo' and result['status'] == 'pass'
    assert result['synthetic'] is True
    assert result['destination'] == str(destination)
    assert shlex.split(result['next_command']) == [
        'policy-doctor', 'check', result['checkpoint'], '--dataset', result['dataset']]
    before = hub.check_checkpoint(result['checkpoint'], dataset=result['dataset'])
    after = hub.check_checkpoint(result['checkpoint'], dataset=result['corrected_dataset'])
    errors = [finding for finding in before['findings']
              if finding['code'] == 'action.processor_order_names' and finding['status'] == 'error']
    assert errors and before['status'] == 'error'
    assert not any(finding['code'] == 'action.processor_order_names' and finding['status'] == 'error'
                   for finding in after['findings'])
    assert after['status'] == 'unknown'
    assert any(finding['status'] == 'unknown' for finding in after['findings'])
    assert json.loads((destination / 'summary.json').read_text()) == result
    assert 'synthetic' in (destination / 'README.md').read_text().lower()
    assert not list(destination.rglob('*.bin'))


@pytest.mark.parametrize('kind', ['directory', 'populated_directory', 'file', 'symlink', 'dangling_symlink'])
def test_policy_demo_refuses_existing_leaf_without_mutation(tmp_path, kind):
    destination, target = tmp_path / 'existing', tmp_path / 'target'
    if kind in ('directory', 'populated_directory'):
        destination.mkdir()
        if kind == 'populated_directory':
            (destination / 'untouched').write_bytes(b'keep me')
    elif kind == 'file':
        destination.write_bytes(b'keep me')
    else:
        if kind == 'symlink':
            target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*'))
    with pytest.raises(FileExistsError):
        create_policy_demo(destination)
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*')) == before
    if kind == 'file':
        assert destination.read_bytes() == b'keep me'
    if kind == 'populated_directory':
        assert (destination / 'untouched').read_bytes() == b'keep me'


def test_trajectory_demo_remains_available_and_unchanged(tmp_path):
    result = create_controller_demo(tmp_path / 'trajectory')
    assert result['kind'] == 'generated_trajectory_demo'
    assert result['candidate_position_at_750ms'] == [1.03125]
    assert result['candidate_bounds_status'] == 'error'
    assert result['corrected_status'] == 'pass'
    assert Path(result['destination'], 'reference.json').is_file()
