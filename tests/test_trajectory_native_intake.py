"""Native ROS text intake must preserve evidence and refuse ambiguous documents."""
import builtins
import hashlib
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.trajectory import sample_joint_trajectory


def message():
    return {'joint_names': ['shoulder'], 'points': [
        {'positions': [0.0], 'velocities': [4.0], 'time_from_start': {'sec': 0, 'nanosec': 0}},
        {'positions': [1.0], 'velocities': [0.0], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}


def save(tmp_path, data, suffix):
    path = tmp_path / f'input{suffix}'
    if suffix == '.json':
        raw = json.dumps(data).encode()
    else:
        yaml = pytest.importorskip('yaml')
        raw = ('# Existing native export; preserve this comment.\n' + yaml.safe_dump(data)).encode()
    path.write_bytes(raw)
    return path, raw


@pytest.mark.parametrize('suffix', ['.json', '.yaml', '.yml'])
@pytest.mark.parametrize('wrapper,pointer', [('direct', ''), ('moveit', '/joint_trajectory'), ('action', '/trajectory')])
def test_native_formats_and_wrappers_preserve_raw_evidence(tmp_path, suffix, wrapper, pointer):
    data = message()
    if wrapper == 'moveit':
        data = {'joint_trajectory': data, 'multi_dof_joint_trajectory': {'joint_names': [], 'points': []}}
    elif wrapper == 'action':
        data = {'trajectory': data, 'path_tolerance': [{'name': 'shoulder', 'position': 0.1}]}
    path, raw = save(tmp_path, data, suffix)
    report = sample_joint_trajectory(str(path), [.75])
    assert report['samples'][0]['positions'] == [1.03125]
    assert report['artifacts'][0]['sha256'] == hashlib.sha256(raw).hexdigest()
    assert report['artifacts'][0]['size_bytes'] == len(raw)
    assert report['artifacts'][0]['message_pointer'] == pointer
    assert report['execution_status'] == report['physical_cause'] == 'unknown'
    if wrapper == 'action':
        assert report['artifacts'][0]['wrapper_fields_not_modeled'] == ['path_tolerance']


@pytest.mark.parametrize('wrapper,field', [('joint_trajectory', 'multi_dof_joint_trajectory'), ('trajectory', 'multi_dof_trajectory')])
def test_nonempty_multidof_is_not_silently_discarded(tmp_path, wrapper, field):
    path, _ = save(tmp_path, {wrapper: message(), field: {'joint_names': ['base'], 'points': [{}]}}, '.json')
    with pytest.raises(InputError, match='multi.dof'):
        sample_joint_trajectory(str(path), [.5])


def test_ambiguous_and_nonstandard_wrappers_refuse(tmp_path):
    for data in [{'trajectory': message(), 'joint_trajectory': message()},
                 {'trajectory': message(), 'command': 'ignored'},
                 {'joint_trajectory': message(), 'multi_dof_joint_trajectory': 'invalid'}]:
        path, _ = save(tmp_path, data, '.json')
        with pytest.raises(InputError):
            sample_joint_trajectory(str(path), [.5])


@pytest.mark.parametrize('text', [
    'joint_names: [shoulder]\npoints: []\n---\njoint_names: [other]\npoints: []\n',
    'joint_names: [shoulder]\njoint_names: [other]\npoints: []\n',
    '1: numeric_key\njoint_names: [shoulder]\npoints: []\n',
    'joint_names: &names [shoulder]\npoints: *names\n',
    'joint_names: &names [shoulder]\npoints: []\n',
    'trajectory: &cycle {trajectory: *cycle}\n',
    '!!python/object/apply:os.system ["touch /tmp/policy-doctor-yaml-should-never-run"]\n',
    'joint_names: [shoulder]\npoints: [{positions: [.nan], time_from_start: {sec: 0, nanosec: 0}}]\n',
    'joint_names: [shoulder]\npoints: [{positions: [.inf], time_from_start: {sec: 0, nanosec: 0}}]\n',
    'joint_names: [shoulder]\npoints: [{positions: [true], time_from_start: {sec: 0, nanosec: 0}}]\n',
    'joint_names: [shoulder]\npoints: [{positions: [1], time_from_start: {sec: 0, nanosec: 0}, positions: [2]}]\n',
    '<<: {joint_names: [shoulder]}\npoints: []\n',
    'joint_names: [shoulder]\npoints: []\nother: 2026-10-19\n',
])
def test_unsafe_ambiguous_or_nonfinite_yaml_refuses(tmp_path, text):
    pytest.importorskip('yaml')
    path = tmp_path / 'unsafe.yaml'
    path.write_text(text)
    with pytest.raises(InputError):
        sample_joint_trajectory(str(path), [.5])


@pytest.mark.parametrize('suffix', ['.yaml', '.json'])
def test_excessive_nesting_is_controlled_input_error(tmp_path, suffix):
    path = tmp_path / f'deep{suffix}'
    path.write_text('[' * 100 + '0' + ']' * 100)
    with pytest.raises(InputError, match='nest|depth'):
        sample_joint_trajectory(str(path), [.5])


def test_json_is_never_reinterpreted_as_yaml(tmp_path):
    path = tmp_path / 'invalid.json'
    path.write_text('joint_names: [shoulder]\npoints: []\n')
    with pytest.raises(InputError, match='JSON'):
        sample_joint_trajectory(str(path), [.5])


def test_unknown_extensions_do_not_select_a_parser(tmp_path):
    path = tmp_path / 'untyped.txt'
    path.write_text(json.dumps(message()))
    with pytest.raises(InputError, match='suffix|extension'):
        sample_joint_trajectory(str(path), [.5])


def test_missing_yaml_extra_has_install_guidance_and_json_still_works(tmp_path, monkeypatch):
    path, _ = save(tmp_path, message(), '.json')
    yaml_path = tmp_path / 'input.yaml'
    yaml_path.write_text('joint_names: [shoulder]\npoints: []\n')
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == 'yaml':
            raise ImportError('simulated absent optional dependency')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    assert sample_joint_trajectory(str(path), [.75])['status'] == 'pass'
    with pytest.raises(InputError, match=r'\[ros\]'):
        sample_joint_trajectory(str(yaml_path), [.5])


def test_large_documents_refuse_before_parsing(tmp_path, monkeypatch):
    import policy_doctor.trajectory as module
    monkeypatch.setattr(module, 'MAX_JSON_BYTES', 128)
    path = tmp_path / 'large.yaml'
    path.write_text('#' * 129)
    with pytest.raises(InputError, match='bounded|byte|large'):
        sample_joint_trajectory(str(path), [.5])


def test_yaml_token_budget_refuses_even_valid_native_input(tmp_path, monkeypatch):
    import policy_doctor.trajectory as module
    path, _ = save(tmp_path, message(), '.yaml')
    monkeypatch.setattr(module, 'MAX_YAML_TOKENS', 8)
    with pytest.raises(InputError, match='token budget'):
        sample_joint_trajectory(str(path), [.5])


@pytest.mark.parametrize('suffix', ['.json', '.yaml'])
def test_portable_case_keeps_original_wrapper_bytes_and_locator(tmp_path, suffix):
    from policy_doctor.cases import export_trajectory_case, reproduce_trajectory_case
    wrapped = {'joint_trajectory': message(), 'multi_dof_joint_trajectory': {'joint_names': [], 'points': []}}
    source, raw = save(tmp_path, wrapped, suffix)
    target = tmp_path / 'case'
    export_trajectory_case(str(source), str(source), str(target), [.75])
    assert (target / f'reference{suffix}').read_bytes() == raw
    assert (target / f'candidate{suffix}').read_bytes() == raw
    result = reproduce_trajectory_case(str(target))
    assert result['status'] == 'pass'
    assert '/joint_trajectory' in json.dumps(result)


def test_duplicate_json_keys_still_refuse(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"trajectory":{},"trajectory":{}}')
    with pytest.raises(InputError, match='Duplicate'):
        sample_joint_trajectory(str(path), [.5])
