import json

import numpy as np

from policy_doctor.investigation import investigate_run


def test_one_folder_detects_fault_and_gives_callable_next_action(case):
    deployment = json.loads(case['deployment'].read_text())
    deployment['units'] = ['deg', 'normalized']
    case['put'](case['deployment'], deployment)
    report = investigate_run(str(case['checkpoint'].parent))
    assert report['status'] == 'error'
    assert any(f['code'] == 'semantics.units' for f in report['findings'])
    action = next(a for a in report['next_actions'] if a['tool'] == 'audit_deployment')
    assert action['arguments']['checkpoint'] == str(case['checkpoint'])
    assert action['arguments']['deployment'] == str(case['deployment'])
    assert report['coverage']['executes_code'] is False


def test_multiple_checkpoints_never_silently_selected(case):
    extra = case['checkpoint'].parent / 'other'
    extra.mkdir()
    case['put'](extra / 'config.json', json.loads((case['checkpoint'] / 'config.json').read_text()))
    result = investigate_run(str(extra.parent))
    assert result['status'] == 'unknown'
    assert len(result['inventory']['checkpoints']) == 2
    assert not result['audits']


def test_no_files_is_unknown_not_clean(tmp_path):
    result = investigate_run(str(tmp_path))
    assert result['status'] == 'unknown' and not result['audits']


def test_symlinked_outside_checkpoint_is_not_followed(case, tmp_path):
    root = tmp_path / 'visible'
    root.mkdir()
    (root / 'outside').symlink_to(case['checkpoint'], target_is_directory=True)
    result = investigate_run(str(root))
    assert not result['inventory']['checkpoints']
    assert result['coverage']['symlinks_skipped'] == 1


def test_discovery_limit_discloses_truncation(tmp_path):
    for index in range(4):
        (tmp_path / f'{index}.txt').write_text('data')
    result = investigate_run(str(tmp_path), max_files=2)
    assert result['status'] == 'unknown'
    assert result['coverage']['truncated']
    assert result['coverage']['files_seen'] == 2


def test_recordings_discovered_without_inventing_pairing(tmp_path):
    for name in ('first', 'second'):
        np.savez(tmp_path / f'{name}.npz', action=np.zeros((3, 2)))
    result = investigate_run(str(tmp_path))
    assert len(result['inventory']['recordings']) == 2
    assert result['status'] == 'unknown'
    assert any(a['tool'] == 'compare_rollout_recordings' for a in result['next_actions'])
    assert result['recording_comparison'] is None
