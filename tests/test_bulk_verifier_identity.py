"""Batch adjudication must use the exact retained verifier and native parser identity."""

import json
import shutil
from pathlib import Path

import pytest
from recovery_fixtures import ORIGINAL, REPO, REVISION
from test_bulk_video_recovery import source

from policy_doctor import bulk_video_recovery as bulk
from policy_doctor import video_recovery as vr
from policy_doctor.artifacts import InputError


def automatic(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True)
    return batch


def reseal_manifest(batch, manifest):
    raw = vr._encode(manifest)
    (batch / 'BATCH_MANIFEST.json').write_bytes(raw)
    (batch / 'BATCH_READY.json').write_bytes(vr._encode({'manifest_sha256': vr._sha(raw)}))


@pytest.mark.parametrize('change', ['missing', 'module', 'dependency', 'schema', 'native_reference'])
def test_changed_or_missing_verifier_refuses_before_inventory_or_adjudication(monkeypatch, tmp_path, change):
    batch = automatic(monkeypatch, tmp_path)
    manifest = json.loads((batch / 'BATCH_MANIFEST.json').read_text())
    if change == 'missing':
        del manifest['verifier']
    elif change == 'module':
        manifest['verifier']['module_sha256']['bulk_video_recovery.py'] = '0' * 64
    elif change == 'dependency':
        manifest['verifier']['dependency_versions']['pyarrow'] = '0.0.0'
    elif change == 'schema':
        manifest['verifier']['detector_schema_version'] = 999
    else:
        manifest['verifier']['native_reference_revision'] = '0' * 40
    reseal_manifest(batch, manifest)
    monkeypatch.setattr(bulk, '_inventory', lambda *a: pytest.fail('inventory before verifier gate'))
    monkeypatch.setattr(bulk, '_retained_detection_consistency', lambda *a: pytest.fail('adjudication before verifier gate'))
    with pytest.raises(InputError, match='versioned seal|verifier identity'):
        bulk.reproduce_bulk_proposal(batch)


@pytest.mark.parametrize('module', ['bulk_video_recovery.py', 'bulk_workflow.py', 'video_index.py', 'windows.py'])
def test_actual_source_byte_drift_refuses_before_detector_replay(monkeypatch, tmp_path, module):
    batch = automatic(monkeypatch, tmp_path)
    original_read = Path.read_bytes
    trusted_root = Path(bulk.__file__).parent
    def changed(path):
        raw = original_read(path)
        return raw + b'\n# simulated installed source drift\n' if path == trusted_root / module else raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    monkeypatch.setattr(bulk, '_retained_detection_consistency', lambda *a: pytest.fail('detector replay before identity'))
    with pytest.raises(InputError, match='verifier identity'):
        bulk.reproduce_bulk_proposal(batch)


def test_exact_version_batch_replays_after_relocation(monkeypatch, tmp_path):
    batch = automatic(monkeypatch, tmp_path)
    manifest = json.loads((batch / 'BATCH_MANIFEST.json').read_text())
    identity = manifest['verifier']
    assert manifest['kind'] == 'robot_video_reference_batch_v2' and manifest['schema_version'] == 2
    assert {'bulk_video_recovery.py', 'bulk_workflow.py', 'windows.py', 'video_index.py', 'artifacts.py'} <= set(identity['module_sha256'])
    assert identity['dependency_versions']['pyarrow'] and identity['dependency_versions']['numpy']
    moved = tmp_path / 'colleague' / 'batch'
    moved.parent.mkdir()
    shutil.move(batch, moved)
    monkeypatch.setattr(vr, '_opener', lambda: pytest.fail('network during replay'))
    replay = bulk.reproduce_bulk_proposal(moved)
    assert replay['verifier_matches'] and replay['batch_schema_version'] == 2
    assert replay['proposal_complete'] and replay['reproduction_matches']
    assert not replay['origin_authenticated'] and not replay['detector_coverage_replayed_from_raw_headers']


def test_acquisition_verifier_change_never_publishes_batch(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    original = bulk._batch_fingerprint
    calls = 0
    def changed():
        nonlocal calls
        calls += 1
        result = original()
        if calls > 1:
            result['detector_schema_version'] = 999
        return result
    monkeypatch.setattr(bulk, '_batch_fingerprint', changed)
    batch = tmp_path / 'batch'
    with pytest.raises(InputError, match='changed during acquisition'):
        bulk.recover_all_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True)
    assert not batch.exists() and not list(tmp_path.glob('.bulk-recovery-*'))
