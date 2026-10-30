"""Typed generated evidence verifies incident/investigation integration, not native execution."""
import copy
import hashlib
import json
import shutil

import numpy as np
import pytest
from test_native_observation_source import fixture, save

from policy_doctor.artifacts import InputError
from policy_doctor.incidents import check_incident_candidate, export_incident_case, reproduce_incident_case
from policy_doctor.investigation import investigate_run


def pair(tmp_path):
    before, after = tmp_path / 'before-view', tmp_path / 'after-view'
    capture, raw = fixture(before)
    leaf = raw['captured']['state']['roll']
    leaf['values'] = [0.0]
    leaf['sha256'] = hashlib.sha256(np.asarray([0.0], dtype='<f4').tobytes()).hexdigest()
    save(before, capture, raw)
    fixture(after)
    return before, after


def test_incident_retains_same_contract_and_reproduces_after_inputs_move(tmp_path):
    before, after = pair(tmp_path)
    result = export_incident_case('native_observation_source', before, tmp_path / 'case', {}, corrected=after)
    assert result['status'] == 'pass'
    shutil.rmtree(before)
    shutil.rmtree(after)
    assert reproduce_incident_case(tmp_path / 'case')['status'] == 'pass'
    healthy = tmp_path / 'later-healthy'
    fixture(healthy)
    assert check_incident_candidate(tmp_path / 'case', healthy)['status'] == 'pass'


def test_investigation_recognizes_capture_and_offers_existing_case_tool(tmp_path):
    before, after = pair(tmp_path)
    report = investigate_run(str(before), reference=str(after))
    assert report['status'] == 'error'
    assert report['applied_requirements']['kind'] == 'native-observation-source'
    assert report['case_recipe']['tool'] == 'create_incident_case'
    assert report['case_recipe']['arguments']['check'] == 'native_observation_source'
    assert report['comparison']['same_correction_input'] is True


def test_same_input_is_required_for_recipe_and_case_lifecycle(tmp_path):
    before, after = pair(tmp_path)
    # A fully well-formed other original input gets another metadata/array digest.
    # The existing unit fixture supports this through its generated identity only.
    cap = json.loads((after / 'capture.json').read_text())
    raw = json.loads((after / 'raw/result.json').read_text())
    metadata = json.loads((after / 'samples/metadata.json').read_text())
    raw['input_identity']['frame_index'] = metadata['frame_index'] = 1
    raw['input_identity']['global_index'] = metadata['global_index'] = 1
    raw['input_identity']['timestamp'] = metadata['timestamp'] = 0.05
    for camera in metadata['cameras'].values():
        camera['ordinal'], camera['pts'], camera['time_seconds'] = 1, 1, 0.05
    (after / 'samples/metadata.json').write_text(json.dumps(metadata))
    cap['sample']['metadata']['sha256'] = hashlib.sha256((after / 'samples/metadata.json').read_bytes()).hexdigest()
    save(after, cap, raw)
    report = investigate_run(str(before), reference=str(after))
    assert report['case_recipe'] is None
    with pytest.raises(InputError):
        export_incident_case('native_observation_source', before, tmp_path / 'case', {}, corrected=after)


def test_unknown_does_not_become_incident(tmp_path):
    before, after = pair(tmp_path)
    cap = json.loads((before / 'capture.json').read_text())
    raw = json.loads((before / 'raw/result.json').read_text())
    raw.pop('state_supplied')
    save(before, cap, raw)
    assert investigate_run(str(before), reference=str(after))['status'] == 'unknown'
    with pytest.raises(InputError):
        export_incident_case('native_observation_source', before, tmp_path / 'case', {}, corrected=after)


def test_explicit_requirement_validates_identically_for_both_views(tmp_path):
    from policy_doctor.native_observation_source import bind_native_observation_source_parameters
    before, after = pair(tmp_path)
    requirement = {'kind': 'native-observation-source', 'parameters': bind_native_observation_source_parameters(after)}
    assert investigate_run(str(before), reference=str(after), requirements=copy.deepcopy(requirement))['status'] == 'error'
    assert investigate_run(str(after), requirements=requirement)['status'] == 'pass'
