"""Public dispatch/config boundaries; fake parents are not native execution evidence."""
import json
import sys
import types
from pathlib import Path

import pytest

from policy_doctor import native_processing as native
from policy_doctor.artifacts import InputError


def runtime_file(tmp_path, **extra):
    for name in ('harness', 'groot', 'reference'):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / 'native.py').write_text('# generated config fixture\n')
    entry = {'python': sys.executable, 'source_root': str(tmp_path / 'harness'),
             'groot_source_root': str(tmp_path / 'groot'), **extra}
    p = tmp_path / 'runtime.json'
    p.write_text(json.dumps({'providers': {'groot_adapter': entry}}))
    return p


def test_operator_config_accepts_only_explicit_native_roots(tmp_path):
    p = runtime_file(tmp_path, reference_source_root=str(tmp_path / 'reference'))
    value = native.load_native_runtimes(p)['groot_adapter']
    assert value['groot_source_root'] == str(tmp_path / 'groot')
    assert value['reference_source_root'] == str(tmp_path / 'reference')
    data = json.loads(p.read_text())
    data['providers']['groot_adapter']['groot_source_root'] = str(tmp_path / 'missing')
    p.write_text(json.dumps(data))
    with pytest.raises(InputError):
        native.load_native_runtimes(p)


def test_auxiliary_code_roots_are_not_accepted_by_other_provider_configs(tmp_path):
    p = runtime_file(tmp_path)
    value = json.loads(p.read_text())
    value['providers']['lerobot'] = value['providers'].pop('groot_adapter')
    p.write_text(json.dumps(value))
    with pytest.raises(InputError):
        native.load_native_runtimes(p)


def test_probe_dispatches_supported_contract_without_client_code_authority(monkeypatch):
    calls = []
    def probe(*args, **kwargs):
        calls.append((args, kwargs))
        return {'kind': 'native_observation_source_observation', 'status': 'error'}
    monkeypatch.setitem(sys.modules, 'policy_doctor.native_observation_processing',
                        types.SimpleNamespace(probe_native_observation_source=probe))
    contract = {'explicit': 'unit contract; core validates independently'}
    runtimes = {'groot_adapter': {'python': '/trusted/python', 'source_root': '/trusted/harness'}}
    result = native.probe_native_processing('groot_adapter', '/model', '/data', runtimes=runtimes,
        samples=[{'episode_index': 0, 'frame_index': 55}], destination='/new', contract=contract)
    assert result['status'] == 'error'
    assert calls == [(('/model', '/data'), {'runtimes': runtimes,
        'samples': [{'episode_index': 0, 'frame_index': 55}], 'destination': '/new', 'contract': contract})]
    for provider in ('lerobot', 'groot_state'):
        with pytest.raises(InputError):
            native.probe_native_processing(provider, '/model', '/data', runtimes={}, samples=[], contract=contract)


@pytest.mark.parametrize('kind,filename', [
    ('native_observation_source_attempt', 'manifest.json'),
    ('native_observation_source_view', 'manifest.json'),
    ('native_observation_source_capture', 'capture.json'),
])
def test_replay_dispatches_attempt_or_standalone_view(tmp_path, monkeypatch, kind, filename):
    (tmp_path / filename).write_text(json.dumps({'kind': kind}))
    calls = []
    def reproduce(directory, *, runtimes):
        calls.append((directory, runtimes))
        return {'kind': 'native_observation_source_reproduction', 'status': 'pass'}
    monkeypatch.setitem(sys.modules, 'policy_doctor.native_observation_processing',
                        types.SimpleNamespace(reproduce_native_observation_source=reproduce))
    result = native.reproduce_native_processing(tmp_path, runtimes={'configured': True})
    assert result['status'] == 'pass'
    assert calls == [(tmp_path.resolve(), {'configured': True})]


def test_fingerprint_covers_both_native_source_roots_and_optional_correction(tmp_path, monkeypatch):
    p = runtime_file(tmp_path, reference_source_root=str(tmp_path / 'reference'))
    runtime = json.loads(p.read_text())['providers']['groot_adapter']
    original_hash = native._hash
    def hash_file(path):
        path = Path(path)
        if path.name == 'native_groot_adapter_worker.py' and not path.exists():
            return 'f' * 64, 1  # Dispatch unit tests need no native worker implementation.
        return original_hash(path)
    monkeypatch.setattr(native, '_hash', hash_file)
    first = native._fingerprint('groot_adapter', runtime)
    (tmp_path / 'groot/native.py').write_text('# changed original native component\n')
    assert native._fingerprint('groot_adapter', runtime) != first
    second = native._fingerprint('groot_adapter', runtime)
    (tmp_path / 'reference/native.py').write_text('# changed corrected native adapter\n')
    assert native._fingerprint('groot_adapter', runtime) != second
