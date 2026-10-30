import hashlib
import json
import shutil

import pytest
import yaml

from policy_doctor.artifacts import InputError
from policy_doctor.cases import export_trajectory_case, reproduce_trajectory_case


def test_native_yaml_and_json_case_moves_without_conversion(tmp_path):
    native = {'joint_names': ['j'], 'points': [
        {'positions': [0.], 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1.], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}
    original = tmp_path / 'original'
    original.mkdir()
    reference = original / 'moveit.yml'
    raw = ('# Retain original comment and wrapper\n' + yaml.safe_dump({'joint_trajectory': native, 'multi_dof_joint_trajectory': {'joint_names': [], 'points': []}})).encode()
    reference.write_bytes(raw)
    candidate = original / 'candidate.json'
    candidate.write_text(json.dumps({'trajectory': native}))
    exported = export_trajectory_case(str(reference), str(candidate), str(tmp_path / 'case'), [.25, .75], control=str(reference))
    assert exported['checks'] == {'candidate': 'pass', 'control': 'pass'}
    case = tmp_path / 'case'
    assert (case / 'reference.yml').read_bytes() == raw
    assert exported['files']['reference.yml'] == hashlib.sha256(raw).hexdigest()
    assert exported['verifier']['pyyaml_version'] == yaml.__version__
    shutil.rmtree(original)
    moved = tmp_path / 'colleague'
    case.rename(moved)
    result = reproduce_trajectory_case(str(moved))
    assert result['reproduction_matches'] and result['verifier_matches']
    assert result['checks']['candidate']['physical_cause'] == 'unknown'
    artifact = result['checks']['candidate']['sampling']['reference']['artifacts'][0]
    assert artifact['message_pointer'] == '/joint_trajectory'
    assert artifact['path'] == 'reference.yml'


def test_case_refuses_two_formats_for_same_role(tmp_path):
    native = {'joint_names': ['j'], 'points': [{'positions': [0], 'time_from_start': {'sec': 0, 'nanosec': 0}}]}
    source = tmp_path / 'native.json'
    source.write_text(json.dumps(native))
    case = tmp_path / 'case'
    export_trajectory_case(str(source), str(source), str(case), [0.])
    manifest = json.loads((case / 'manifest.json').read_text())
    (case / 'reference.yaml').write_text(yaml.safe_dump(native))
    manifest['files']['reference.yaml'] = hashlib.sha256((case / 'reference.yaml').read_bytes()).hexdigest()
    raw = json.dumps(manifest).encode()
    (case / 'manifest.json').write_bytes(raw)
    (case / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))
    with pytest.raises(InputError, match='exactly one'):
        reproduce_trajectory_case(str(case))
