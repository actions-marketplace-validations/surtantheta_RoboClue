"""Entry routing checks; source-evidence semantics are tested by their own engine."""
import json
import sys
from types import SimpleNamespace

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.investigation import investigate_run


@pytest.fixture
def source_entry(tmp_path, monkeypatch):
    calls = []
    roots = {}
    for name in ('suspect', 'corrected', 'unknown'):
        root = tmp_path / name
        root.mkdir()
        (root / 'capture.json').write_text(json.dumps({
            'kind': 'native_processor_source_capture', 'schema_version': 1}))
        roots[name] = root
    frozen = {'identity': 'example/checkpoint@pin', 'bundle_sha256': 'a' * 64}

    def validate(parameters):
        if set(parameters) - {'expected_source', 'require_same_input_native_reference'}:
            raise InputError('Unexpected source parameter')
        if parameters.get('require_same_input_native_reference', True) is not True:
            raise InputError('Same-input native reference is required')
        return {'expected_source': parameters.get('expected_source', 'selected_checkpoint'),
                'require_same_input_native_reference': True}

    def bind(directory, parameters=None):
        calls.append(('bind', str(directory)))
        return {**validate(parameters or {}), 'expected_source': frozen}

    def audit(directory, **parameters):
        calls.append(('audit', str(directory), parameters))
        name = str(directory).rsplit('/', 1)[-1]
        status = {'suspect': 'error', 'corrected': 'pass', 'unknown': 'unknown'}[name]
        return {'status': status, 'scope': 'Recorded source evidence only; no native replay.',
                'source_criterion': {'status': status,
                    'code': 'source_contradiction' if status == 'error' else status},
                'findings': [], 'finding_count': 0, 'findings_omitted': 0,
                'input_digest': 'b' * 64, 'expected_source': parameters.get('expected_source'),
                'native_identity': {'episode_index': 0, 'frame_index': 0, 'index': 0},
                'selected_source': frozen, 'native_processing_status': 'complete',
                'native_replay_status': 'unknown'}

    monkeypatch.setitem(sys.modules, 'policy_doctor.native_processor_source', SimpleNamespace(
        PARAMETER_KEYS={'expected_source', 'require_same_input_native_reference'},
        validate_parameters=validate,
        bind_native_processor_source_parameters=bind,
        audit_native_processor_source=audit))
    return roots, calls, frozen


def test_explicit_source_requirement_freezes_reference_for_both_audits(source_entry):
    roots, calls, frozen = source_entry
    report = investigate_run(str(roots['suspect']), reference=str(roots['corrected']),
        requirements={'kind': 'native-processor-source', 'parameters': {}})
    assert report['status'] == 'error'
    audits = [entry for entry in calls if entry[0] == 'audit']
    assert len(audits) == 2
    assert all(entry[2]['expected_source'] == frozen for entry in audits)
    recipe = report['case_recipe']['arguments']
    assert recipe['check'] == 'native_processor_source'
    assert recipe['parameters']['expected_source'] == frozen
    assert recipe['source'] == str(roots['suspect'])
    assert recipe['corrected'] == str(roots['corrected'])


def test_source_capture_is_discovered_without_tool_recipe(source_entry):
    roots, calls, _ = source_entry
    report = investigate_run(str(roots['suspect']), reference=str(roots['corrected']))
    assert report['status'] == 'error'
    assert report['case_recipe']['arguments']['check'] == 'native_processor_source'
    assert report['coverage']['executes_code'] is False
    assert report['inventory']['captures'] == [str(roots['suspect'])]
    assert any(entry[0] == 'bind' for entry in calls)


def test_unknown_source_does_not_offer_a_correction_case(source_entry):
    roots, _, _ = source_entry
    report = investigate_run(str(roots['unknown']), reference=str(roots['corrected']))
    assert report['status'] == 'unknown'
    assert report['case_recipe'] is None


def test_source_without_reference_reports_evidence_without_case(source_entry):
    roots, calls, _ = source_entry
    report = investigate_run(str(roots['suspect']))
    assert report['status'] == 'error'
    assert report['case_recipe'] is None
    assert not any(entry[0] == 'bind' for entry in calls)


def test_identical_values_on_different_native_frame_do_not_offer_correction(source_entry, monkeypatch):
    roots, _, _ = source_entry
    engine = sys.modules['policy_doctor.native_processor_source']
    original = engine.audit_native_processor_source

    def changed_identity(directory, **parameters):
        result = original(directory, **parameters)
        if str(directory).endswith('corrected'):
            result['native_identity'] = {'episode_index': 0, 'frame_index': 1, 'index': 1}
        return result

    monkeypatch.setattr(engine, 'audit_native_processor_source', changed_identity)
    result = investigate_run(str(roots['suspect']), reference=str(roots['corrected']))
    assert result['case_recipe'] is None
    assert result['comparison']['same_correction_input'] is False


@pytest.mark.parametrize('value', [False, 0, 1, 'true', None])
def test_source_requirement_cannot_weaken_same_input_reference(source_entry, value):
    roots, _, _ = source_entry
    with pytest.raises(InputError):
        investigate_run(str(roots['suspect']), requirements={
            'kind': 'native-processor-source',
            'parameters': {'require_same_input_native_reference': value}})


def test_source_capture_refuses_unrelated_named_criteria(source_entry):
    roots, _, _ = source_entry
    with pytest.raises(InputError):
        investigate_run(str(roots['suspect']), named_contract={'expected_action_names': ['x']})


def test_source_capture_root_symlink_is_refused(source_entry, tmp_path):
    roots, _, _ = source_entry
    link = tmp_path / 'alias'
    link.symlink_to(roots['suspect'], target_is_directory=True)
    with pytest.raises(InputError):
        investigate_run(str(link), requirements={
            'kind': 'native-processor-source', 'parameters': {}})
