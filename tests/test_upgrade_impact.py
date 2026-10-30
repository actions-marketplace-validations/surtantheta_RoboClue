import copy
import hashlib
import json

import numpy as np
import pytest

from policy_doctor import upgrade_impact as ui
from policy_doctor.artifacts import InputError
from policy_doctor.training_guard import TargetContract
from policy_doctor.training_guard_capture import save_capture


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    # Generated unit controls use their own explicitly mocked source coverage.
    # Actual pinned native source qualification is a separate recorded PBS run.
    source = tmp_path / 'native.py'
    source.write_text('original source fixture\n')
    source_pin = identity(source)
    revisions = list(ui.ISAACLAB_SOURCES)
    monkeypatch.setattr(ui, 'ISAACLAB_SOURCES', {rev: {'loader.py': source_pin['sha256']} for rev in revisions})
    before_config, after_config = tmp_path / 'before.json', tmp_path / 'after.json'
    before_config.write_text('{"active":true,"inactive":1}')
    after_config.write_text('{"active":true,"inactive":2}')
    fields = {'/data/demo_0/robot/root_pose': np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
              '/data/demo_0/ee_pose/orientation': np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)}
    contract = ui.BoundaryContract.from_reference(fields, intent={
        'all_quaternions': 'XYZW as explicitly declared',
        'reference_adapter': 'Caller-supplied canonical reference snapshot; raw old/new values are not compared',
    })
    contract_file = tmp_path / 'criterion.json'
    contract.save(contract_file)
    before_capture, after_capture = tmp_path / 'before-capture.json', tmp_path / 'after-capture.json'
    ui.save_boundary_capture(fields, before_capture)
    ui.save_boundary_capture(fields, after_capture)

    def snapshot(revision, config, capture):
        return {'consumer': 'isaaclab-hdf5-xyzw', 'revision': revision,
                'source_files': {'loader.py': source_pin}, 'configs': {'recipe': identity(config)},
                'capture': identity(capture)}

    manifest = {'schema_version': 1, 'kind': 'native_upgrade_impact', 'contract': identity(contract_file),
                'assets': [{'asset_id': 'my-recording',
                            'reference': snapshot(revisions[0], before_config, before_capture),
                            'candidate': snapshot(revisions[1], after_config, after_capture)}]}
    return manifest, fields, tmp_path


def assess(pilot, mutate=None):
    manifest, fields, root = pilot
    if mutate:
        mutate(manifest, fields, root)
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return ui.assess_upgrade_impact(path)


def test_real_revision_names_alone_and_inactive_config_change_stay_healthy(pilot):
    report = assess(pilot)
    asset = report['assets'][0]
    assert report['status'] == 'healthy' and not report['affected_asset_ids']
    assert asset['change_kind'] == 'upstream_revision_change'
    assert asset['associated_config_changes'][0]['name'] == 'recipe'
    assert asset['native_boundary_delta']['status'] == 'equal'
    assert asset['candidate_check']['criterion_sha256'] == report['criterion_sha256']
    assert asset['criterion_intent'] == report['criterion_intent']
    assert asset['comparison_basis'] == report['comparison_basis']
    assert report['comparison_basis']['reference_adapter'] == report['criterion_intent']['reference_adapter']
    assert report['source_input_postguard'] == 'pass'


def fault(manifest, fields, root):
    changed = {key: value.copy() for key, value in fields.items()}
    changed['/data/demo_0/ee_pose/orientation'][0] = [1, 0, 0, 0]
    path = root / 'fault.json'
    ui.save_boundary_capture(changed, path)
    manifest['assets'][0]['candidate']['capture'] = identity(path)


def test_affected_asset_cites_observed_pointer_and_fixed_rule(pilot):
    report = assess(pilot, fault)
    assert report['status'] == 'changed'
    assert report['affected_asset_ids'] == ['my-recording']
    asset = report['assets'][0]
    assert asset['candidate_check']['status'] == 'error'
    assert asset['code'] == 'retained_boundary_criterion_violated'
    assert asset['reference_check']['status'] == 'pass'
    assert asset['native_boundary_delta']['changes'][0]['field'] == '/data/demo_0/ee_pose/orientation'
    assert asset['candidate_check']['differing_values'] == 2
    assert asset['native_boundary_delta']['criterion_intent'] == report['criterion_intent']
    assert 'adapted' in asset['native_boundary_delta']['basis']


def test_separately_frozen_legal_field_specific_intent_does_not_clear_original_case(pilot):
    manifest, fields, root = pilot
    original_pin = manifest['contract']['sha256']
    legal_fields = {key: value.copy() for key, value in fields.items()}
    legal_fields['/data/demo_0/ee_pose/orientation'][0] = [1, 0, 0, 0]
    legal = ui.BoundaryContract.from_reference(legal_fields, intent={
        'root_order': 'XYZW', 'ee_order': 'WXYZ',
        'reference_adapter': 'Explicit root-only adapter; retain end-effector WXYZ',
    })
    legal_path, legal_reference = root / 'legal-criterion.json', root / 'legal-reference.json'
    # Freeze this distinct legal intent before producing the candidate control.
    legal.save(legal_path)
    ui.save_boundary_capture(legal_fields, legal_reference)
    violated = assess(pilot, fault)
    legal_manifest = copy.deepcopy(manifest)
    legal_manifest['contract'] = identity(legal_path)
    legal_manifest['assets'][0]['reference']['capture'] = identity(legal_reference)
    legal_manifest_path = root / 'legal-manifest.json'
    legal_manifest_path.write_text(json.dumps(legal_manifest))
    legal_report = ui.assess_upgrade_impact(legal_manifest_path)
    assert legal_report['status'] == 'healthy'
    assert legal_report['criterion_sha256'] != original_pin
    assert legal_report['criterion_intent']['ee_order'] == 'WXYZ'
    assert violated['status'] == assess(pilot)['status'] == 'changed'
    assert identity(root / 'criterion.json')['sha256'] == original_pin


@pytest.mark.parametrize('case', [
    'version', 'consumer', 'source', 'missing_provenance', 'missing_field', 'new_field', 'bad_reference',
])
def test_unsupported_coverage_is_unknown(pilot, case):
    def mutate(manifest, fields, root):
        candidate = manifest['assets'][0]['candidate']
        if case == 'version':
            candidate['revision'] = 'future-version'
        elif case == 'consumer':
            candidate['consumer'] = 'another-simulator'
        elif case == 'source':
            candidate['source_files'] = {'other.py': candidate['source_files']['loader.py']}
        elif case == 'missing_provenance':
            candidate['source_files'] = {}
        else:
            modified = copy.deepcopy(fields)
            if case == 'missing_field':
                modified.pop('/data/demo_0/ee_pose/orientation')
            elif case == 'new_field':
                modified['/unknown'] = np.ones((1, 4))
            else:
                modified['/data/demo_0/ee_pose/orientation'][0, 0] = 1
            path = root / 'incomplete.json'
            ui.save_boundary_capture(modified, path)
            manifest['assets'][0]['reference' if case == 'bad_reference' else 'candidate']['capture'] = identity(path)
    report = assess(pilot, mutate)
    assert report['status'] == 'unknown' and not report['affected_asset_ids']


def test_same_rule_repair_replay_never_replaces_criterion(pilot):
    manifest, _, root = pilot
    criterion_path = root / 'criterion.json'
    original = criterion_path.read_bytes()
    failed = assess(pilot, fault)
    manifest['assets'][0]['candidate']['capture'] = manifest['assets'][0]['reference']['capture']
    repaired = assess(pilot)
    replay = assess(pilot)
    assert failed['status'] == 'changed' and repaired['status'] == replay['status'] == 'healthy'
    assert failed['criterion_sha256'] == repaired['criterion_sha256'] == replay['criterion_sha256']
    assert criterion_path.read_bytes() == original


@pytest.mark.parametrize('case', ['pin', 'capture_bytes', 'duplicate_asset', 'extra_key', 'symlink', 'fifo'])
def test_malformed_or_changed_inputs_cannot_produce_healthy(pilot, case):
    def mutate(manifest, fields, root):
        if case == 'pin':
            manifest['contract']['sha256'] = '0' * 64
        elif case == 'capture_bytes':
            (root / 'after-capture.json').write_text('{}')
        elif case == 'duplicate_asset':
            manifest['assets'].append(copy.deepcopy(manifest['assets'][0]))
        elif case == 'extra_key':
            manifest['assets'][0]['candidate']['unverified_default'] = True
        else:
            import os

            path = root / 'special'
            if case == 'symlink':
                path.symlink_to(root / 'after-capture.json')
            else:
                os.mkfifo(path)
            manifest['assets'][0]['candidate']['capture']['path'] = str(path)
    with pytest.raises(InputError):
        assess(pilot, mutate)


def test_duplicate_json_keys_and_nonfinite_metadata_refused(pilot):
    _, _, root = pilot
    for text in ('{"kind":1,"kind":2}', '{"x":NaN}'):
        path = root / 'duplicate.json'
        path.write_text(text)
        with pytest.raises(InputError):
            ui.assess_upgrade_impact(path)


def test_input_postguard_refuses_config_change_during_comparison(pilot, monkeypatch):
    original = ui._load_boundary

    def load_then_change(path):
        result = original(path)
        if path.endswith('after-capture.json'):
            (pilot[2] / 'after.json').write_text('{"changed_during_assessment":true}')
        return result

    monkeypatch.setattr(ui, '_load_boundary', load_then_change)
    with pytest.raises(InputError, match='during impact assessment'):
        assess(pilot)


def test_reference_fields_are_immutable_snapshots(pilot):
    fields = pilot[1]
    contract = ui.BoundaryContract.from_reference(fields, intent={'declared': 'original'})
    pin = contract.criterion_sha256
    exposed = contract.fields
    exposed['/data/demo_0/ee_pose/orientation'][0, 0] = 99
    assert contract.check(fields)['status'] == 'pass' and contract.criterion_sha256 == pin


def test_relative_paths_resolve_beside_manifest(pilot, monkeypatch):
    manifest, _, root = pilot
    manifest['contract']['path'] = 'criterion.json'
    monkeypatch.chdir('/')
    assert assess(pilot)['status'] == 'healthy'


def test_alerts_are_deterministic_local_and_do_not_overwrite(pilot):
    report = assess(pilot, fault)
    root = pilot[2]
    first = ui.write_upgrade_alerts(report, root / 'alerts-one')
    second = ui.write_upgrade_alerts(report, root / 'alerts-two')
    assert first['exit_code'] == second['exit_code'] == 1
    for name in ('impact.json', 'alerts.md'):
        assert (root / 'alerts-one' / name).read_bytes() == (root / 'alerts-two' / name).read_bytes()
    alerts = (root / 'alerts-one/alerts.md').read_text()
    assert 'Criterion intent:' in alerts and 'Comparison basis:' in alerts
    assert report['criterion_intent']['reference_adapter'] in alerts
    with pytest.raises(InputError):
        ui.write_upgrade_alerts(report, root / 'alerts-one')


def test_unknown_and_healthy_ci_outcomes(pilot):
    report = assess(pilot)
    assert ui.write_upgrade_alerts(report, pilot[2] / 'healthy-alert')['exit_code'] == 0
    pilot[0]['assets'][0]['candidate']['revision'] = 'unreviewed'
    report = assess(pilot)
    assert ui.write_upgrade_alerts(report, pilot[2] / 'unknown-alert')['exit_code'] == 3


def test_gr00t_target_profile_reuses_original_contract(tmp_path, monkeypatch):
    source = tmp_path / 'native.py'
    source.write_text('native source control')
    pin = identity(source)
    monkeypatch.setattr(ui, 'QUALIFIED_SOURCES', {'native.py': pin['sha256']})
    arrays = {'inputs': {'action': np.ones((1, 2, 3), dtype=np.float32),
                         'action_mask': np.ones((1, 2, 3), dtype=np.float64)}}
    contract = TargetContract.from_references([(arrays, ['row1'], [{'original': 'public-row1'}])],
                                               intent={'recipe': 'explicit reference'})
    contract_path, capture = tmp_path / 'contract.json', tmp_path / 'capture.json'
    contract.save(contract_path)
    save_capture(arrays, ['row1'], capture)
    snapshot = {'consumer': ui.QUALIFIED_CONSUMER, 'revision': ui.QUALIFIED_REVISION,
                'source_files': {'native.py': pin}, 'configs': {'recipe': pin}, 'capture': identity(capture)}
    manifest = {'schema_version': 1, 'kind': 'native_upgrade_impact', 'contract': identity(contract_path),
                'assets': [{'asset_id': 'my-training-recipe', 'reference': snapshot, 'candidate': snapshot}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    report = ui.assess_upgrade_impact(path)
    assert report['status'] == 'healthy'
    assert report['assets'][0]['change_kind'] == 'configuration_comparison'
    assert report['criterion_sha256'] == contract.criterion_sha256
