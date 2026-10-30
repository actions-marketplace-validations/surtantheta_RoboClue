"""Independent misuse, unknown-preservation and relocation review of retained checks."""
import hashlib
import json
import shutil

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.demo import create_policy_demo
from policy_doctor.retained import check_retained_candidate, reproduce_retained_check, retain_check


def _recording(root, name, values=(1, 2, 3), ids=(0, 1, 2)):
    path = root / (name + '.csv')
    path.write_text('sample_id,action\n' + ''.join(f'{i},{v}\n' for i, v in zip(ids, values, strict=True)))
    return path


def _numeric_case(root, *, aligned=True):
    reference = _recording(root, 'reference')
    original = _recording(root, 'original', (1, 9, 3))
    case = root / 'case'
    parameters = {'alignment_key': 'sample_id', 'atol': 0, 'rtol': 0} if aligned else {}
    result = retain_check('recording_equivalence', str(original), str(case), reference=str(reference), parameters=parameters)
    return case, reference, original, result


def _rewrite_ready(case, manifest):
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    (case / 'manifest.json').write_bytes(raw)
    (case / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


@pytest.mark.parametrize(('values', 'ids', 'status'), [
    ((1, 2, 3), (0, 1, 2), 'pass'),
    ((1, 9, 3), (0, 1, 2), 'different'),
    ((1, 2, 3), (0, 2, 3), 'unknown'),
])
def test_unchanged_reference_survives_relocation_and_later_controls(tmp_path, values, ids, status):
    case, reference, original, exported = _numeric_case(tmp_path)
    assert exported['status'] == 'pass' and exported['original_status'] == 'different'
    reference.unlink()
    original.unlink()
    moved = tmp_path / 'colleague case with spaces'
    shutil.move(case, moved)
    before = (moved / 'reference/recording.csv').read_bytes()
    replay = reproduce_retained_check(str(moved))
    assert replay['status'] == 'pass' and replay['original_status'] == 'different'
    later = _recording(tmp_path, 'later', values, ids)
    checked = check_retained_candidate(str(moved), str(later))
    assert checked['candidate_evaluated'] is True and checked['candidate_status'] == status
    assert checked['status'] == status
    assert (moved / 'reference/recording.csv').read_bytes() == before


def test_original_unknown_reproduces_without_becoming_known_evidence(tmp_path):
    case, _, _, result = _numeric_case(tmp_path, aligned=False)
    assert result['original_status'] == 'unknown'
    replay = reproduce_retained_check(str(case))
    assert replay['status'] == 'pass'
    assert replay['check_result']['status'] == replay['original_status'] == 'unknown'
    candidate = _recording(tmp_path, 'later')
    checked = check_retained_candidate(str(case), str(candidate))
    assert checked['status'] == 'unknown' and checked['check_result']['comparison_complete'] is False


@pytest.mark.parametrize('name', ['reference/recording.csv', 'expected.json', 'provenance.json'])
def test_tampering_is_refused_before_reproduction(tmp_path, name):
    case, _, _, _ = _numeric_case(tmp_path)
    path = case / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(InputError):
        reproduce_retained_check(str(case))
    with pytest.raises(InputError):
        check_retained_candidate(str(case), str(tmp_path / 'does-not-exist'))


def test_bound_manifest_traversal_is_refused(tmp_path):
    case, _, _, _ = _numeric_case(tmp_path)
    manifest = json.loads((case / 'manifest.json').read_text())
    manifest['roles']['reference'] = '../reference.csv'
    _rewrite_ready(case, manifest)
    with pytest.raises(InputError):
        reproduce_retained_check(str(case))


def test_verifier_drift_never_opens_later_input(tmp_path):
    case, _, _, _ = _numeric_case(tmp_path)
    manifest = json.loads((case / 'manifest.json').read_text())
    manifest['verifier']['engine_version'] = 'unavailable-verifier'
    _rewrite_ready(case, manifest)
    result = check_retained_candidate(str(case), str(tmp_path / 'does-not-exist'))
    assert result['status'] == 'unknown'
    assert result['candidate_evaluated'] is False
    assert result['candidate_status'] is None


def test_later_input_cannot_alias_the_frozen_reference(tmp_path):
    case, _, _, _ = _numeric_case(tmp_path)
    with pytest.raises(InputError):
        check_retained_candidate(str(case), str(case / 'reference/recording.csv'))


def test_symlink_later_input_is_refused(tmp_path):
    case, reference, _, _ = _numeric_case(tmp_path)
    alias = tmp_path / 'alias.csv'
    alias.symlink_to(reference)
    with pytest.raises(InputError):
        check_retained_candidate(str(case), str(alias))


@pytest.mark.parametrize('scope,status', [('metadata', 'pass'), ('deployment', 'unknown')])
def test_metadata_scope_and_deployment_unknown_survive_portable_replay(tmp_path, scope, status):
    demo = tmp_path / 'demo'
    create_policy_demo(demo)
    case = tmp_path / 'metadata-case'
    result = retain_check('policy_metadata', str(demo / 'checkpoint'), str(case),
                          dataset=str(demo / 'corrected_dataset'), parameters={'scope': scope})
    assert result['original_status'] == status
    shutil.rmtree(demo)
    replay = reproduce_retained_check(str(case))
    assert replay['status'] == 'pass' and replay['original_status'] == status
    assert replay['check_result']['assessment_scope'] == scope
    assert replay['check_result']['deployment_status'] == 'not_assessed'


@pytest.mark.parametrize('role', ['checkpoint', 'dataset'])
def test_metadata_replacement_role_is_frozen(tmp_path, role):
    demo = tmp_path / 'demo'
    create_policy_demo(demo)
    case = tmp_path / 'metadata-case'
    retain_check('policy_metadata', str(demo / 'checkpoint'), str(case),
                 dataset=str(demo / 'dataset'), candidate_role=role)
    replacement = demo / ('checkpoint' if role == 'checkpoint' else 'corrected_dataset')
    result = check_retained_candidate(str(case), str(replacement))
    assert result['candidate_role'] == role
    assert result['candidate_status'] == ('error' if role == 'checkpoint' else 'pass')


def test_invalid_alignment_criterion_is_refused_before_copying(tmp_path):
    source = tmp_path / 'nonexistent.csv'
    with pytest.raises(InputError):
        retain_check('recording_equivalence', str(source), str(tmp_path / 'case'),
                     reference=str(source), parameters={'alignment_key': 'id', 'streams': ['action']})
    assert not (tmp_path / 'case').exists()
