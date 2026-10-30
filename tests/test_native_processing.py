"""Parent publication/runtime unit tests using an explicitly fake fixed worker.

These tests exercise genuine capsule publication and verification machinery.
The fake worker is not evidence of any native provider execution or correctness.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from policy_doctor import native_processing as native
from policy_doctor.artifacts import InputError


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def unit_runtime(tmp_path):
    source = tmp_path / 'trusted-source'
    source.mkdir()
    (source / 'native_original.py').write_text('# generated unit fixture; never executed\n')
    cache = tmp_path / 'cache'
    cache.mkdir()
    config = put(tmp_path / 'runtime.json', {'providers': {'lerobot': {
        'python': sys.executable, 'source_root': str(source), 'cache_root': str(cache)}}})
    return config, source, cache


def test_runtime_config_preserves_explicit_interpreter_and_normalizes_source(unit_runtime):
    config, source, cache = unit_runtime
    loaded = native.load_native_runtimes(config)
    assert loaded == {'lerobot': {'python': str(Path(sys.executable).absolute()),
                                 'source_root': str(source.resolve()), 'cache_root': str(cache.resolve())}}


@pytest.mark.parametrize('document', [
    {}, {'providers': {}}, {'providers': []}, {'providers': {'arbitrary_module': {}}},
    {'providers': {}, 'command': 'execute'},
    {'providers': {'lerobot': {'python': sys.executable}}},
    {'providers': {'lerobot': {'python': sys.executable, 'source_root': '.', 'args': ['-c', 'code']}}},
    {'providers': {'lerobot': {'python': False, 'source_root': '.'}}},
    {'providers': {'lerobot': {'python': '', 'source_root': '.'}}},
])
def test_runtime_config_refuses_unconfigured_authority(tmp_path, document):
    config = put(tmp_path / 'runtime.json', document)
    with pytest.raises(InputError):
        native.load_native_runtimes(config)


def test_runtime_config_requires_actual_paths(unit_runtime):
    config, source, _ = unit_runtime
    for runtime in ({'python': '/missing/python', 'source_root': str(source)},
                    {'python': sys.executable, 'source_root': '/missing/native/source'},
                    {'python': sys.executable, 'source_root': str(source), 'cache_root': '/missing/cache'}):
        put(config, {'providers': {'lerobot': runtime}})
        with pytest.raises(InputError):
            native.load_native_runtimes(config)


def test_runtime_keeps_venv_executable_symlink_spelling(unit_runtime, tmp_path):
    config, source, _ = unit_runtime
    interpreter = tmp_path / 'venv-python'
    interpreter.symlink_to(sys.executable)
    put(config, {'providers': {'lerobot': {'python': str(interpreter), 'source_root': str(source)}}})
    assert native.load_native_runtimes(config)['lerobot']['python'] == str(interpreter)


@pytest.mark.parametrize('samples', [
    None, [], [{'episode_index': 0, 'frame_index': 0}] * 3,
    [{'episode_index': 0, 'frame_index': 0}] * 2,
    [{'episode_index': True, 'frame_index': 0}], [{'episode_index': 0, 'frame_index': -1}],
    [{'episode_index': 0, 'frame_index': 1.0}], [{'episode_index': 0, 'frame_index': 10**9 + 1}],
    [{'episode_index': 0}], [{'episode_index': 0, 'frame_index': 1, 'callable': 'anything'}],
])
def test_samples_are_bounded_explicit_unique_coordinates(samples):
    with pytest.raises(InputError):
        native._samples(samples)


def test_two_valid_samples_remain_literal_and_copied():
    samples = [{'episode_index': 2, 'frame_index': 0}, {'episode_index': 2, 'frame_index': 1}]
    normalized = native._samples(samples)
    assert normalized == samples and normalized is not samples and normalized[0] is not samples[0]


@pytest.mark.parametrize('content', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '[' * 34 + '0' + ']' * 34])
def test_json_input_is_finite_unambiguous_and_bounded(tmp_path, content):
    path = tmp_path / 'metadata.json'
    path.write_text(content)
    with pytest.raises(InputError):
        native._json(path)
    path.write_text(' ' * 100)
    with pytest.raises(InputError, match='bounded'):
        native._json(path, limit=16)


def test_inventory_refuses_symlink_and_oversized_file(tmp_path, monkeypatch):
    root = tmp_path / 'inputs'
    root.mkdir()
    (root / 'link').symlink_to(tmp_path / 'outside')
    with pytest.raises((InputError, FileNotFoundError)):
        native._inventory(root)
    (root / 'link').unlink()
    (root / 'large').write_bytes(b'x' * 17)
    monkeypatch.setattr(native, 'MAX_FILE', 16)
    with pytest.raises(InputError, match='bounded'):
        native._inventory(root)


def test_worker_environment_clears_python_and_online_authority(unit_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/client/modules')
    monkeypatch.setenv('PYTHONHOME', '/client/python')
    monkeypatch.setenv('HF_ENDPOINT', 'https://client.invalid')
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    runtime = native.load_native_runtimes(unit_runtime[0])['lerobot']
    env = native._environment(runtime, tmp_path)
    assert 'PYTHONPATH' not in env and 'PYTHONHOME' not in env and 'HF_ENDPOINT' not in env
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == env['HF_DATASETS_OFFLINE'] == '1'
    assert env['CUDA_VISIBLE_DEVICES'] == '' and env['OMP_NUM_THREADS'] == '1'
    assert env['HF_HUB_CACHE'] == runtime['cache_root']


@pytest.fixture
def fake_fixed_worker(tmp_path, monkeypatch, unit_runtime):
    """Generated worker substitute for parent unit tests; never a native oracle."""
    model, dataset = tmp_path / 'model', tmp_path / 'dataset'
    put(model / 'config.json', {'type': 'unit-fixture-only'})
    put(dataset / 'meta/info.json', {'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                                     'features': {'observation.state': {'shape': [2]}}})
    calls = []
    state = {'execution_status': 'complete', 'verifier': {'unit_verifier': 1}, 'changed_replay': False}
    sample = {'identity': {'episode_index': 0, 'frame_index': 1, 'index': 1, 'timestamp': 0.1},
              'source': {'path': 'unit-fixture-only', 'sha256': 'a' * 64},
              'routes': [{'name': name, 'status': 'complete', 'steps': [{'class': 'generated_unit_fake'}],
                          'input': {'unit': {'shape': [1], 'dtype': 'float32', 'sha256': 'b' * 64}},
                          'output': {'unit': {'shape': [1], 'dtype': 'float32', 'sha256': 'c' * 64}}}
                         for name in ('saved', 'current_factory')]}
    def invoke(script, runtime, request, workspace):
        assert script.name == 'native_lerobot_worker.py'
        calls.append(copy.deepcopy(request))
        output = Path(request['output'])
        output.mkdir()
        put(output / 'samples/sample_0.json', {'explicit_unit_fixture': True})
        (output / 'samples/sample_0.npz').write_bytes(b'Explicit unit placeholder, not native array evidence')
        samples = [copy.deepcopy(sample)]
        if request['operation'] == 'replay' and state['changed_replay']:
            samples[0]['routes'][0]['output'] = {'changed': True}
        receipts = []
        for key, root, path in [('model', model, 'config.json'), ('dataset', dataset, 'meta/info.json')]:
            data = (root / path).read_bytes()
            receipts.append({'root': key, 'path': path, 'sha256': hashlib.sha256(data).hexdigest(),
                             'size_bytes': len(data), 'role': 'generated_unit_fixture_metadata'})
        report = {'provider': 'lerobot', 'execution_status': state['execution_status'],
                  'scope': 'Generated unit worker, no native execution', 'samples': samples,
                  'native_versions': {}, 'source_files': [], 'exports': [], 'unverified': ['Not native proof'],
                  'artifact_receipts': receipts}
        put(output / 'worker_manifest.json', {'provider': 'lerobot', 'schema': 'policy_doctor.native_lerobot.v1',
            'samples': [{field: {'path': 'samples/sample_0.' + suffix,
                                'sha256': hashlib.sha256((output / ('samples/sample_0.' + suffix)).read_bytes()).hexdigest()}
                         for field, suffix in [('arrays', 'npz'), ('metadata', 'json')]}], 'exports': []})
        put(output / 'result.json', report)
        return report
    monkeypatch.setattr(native, '_invoke', invoke)
    monkeypatch.setattr(native, '_fingerprint', lambda *args: copy.deepcopy(state['verifier']))
    def prepare(provider, original_model, original_dataset, samples, workspace, runtime, invoke):
        # This publication unit deliberately substitutes intake as well as the
        # worker. Separate native_intake tests exercise actual dependency rules.
        inventories = {'model': native._inventory(original_model), 'dataset': native._inventory(original_dataset)}
        receipts = [{'root': name, 'path': path, **value, 'role': 'generated_unit_fixture_metadata'}
                    for name, files in inventories.items() for path, value in files.items()]
        return {'model': original_model, 'dataset': original_dataset, 'receipts': receipts,
                'selection': inventories, 'omissions': ['Explicit fake intake; no native proof']}
    def verify(inventories):
        assert inventories == {'model': native._inventory(model), 'dataset': native._inventory(dataset)}
    monkeypatch.setattr(native, '_prepare_native_inputs', prepare)
    monkeypatch.setattr(native, '_verify_native_inputs', verify)
    runtime = native.load_native_runtimes(unit_runtime[0])
    def probe(destination=None):
        return native.probe_native_processing('lerobot', model, dataset, runtimes=runtime,
            samples=[{'episode_index': 0, 'frame_index': 1}], destination=destination)
    return {'model': model, 'dataset': dataset, 'calls': calls, 'state': state,
            'runtimes': runtime, 'probe': probe, 'destination': tmp_path / 'capsule'}


def test_genuine_parent_publication_commits_hashes_originals_and_replays_unit_fake(fake_fixed_worker):
    unit = fake_fixed_worker
    result = unit['probe'](unit['destination'])
    root = unit['destination']
    assert result['retained'] and result['status'] == 'observed'
    assert (root / 'originals/model/config.json').read_bytes() == (unit['model'] / 'config.json').read_bytes()
    assert (root / 'originals/dataset/meta/info.json').read_bytes() == (unit['dataset'] / 'meta/info.json').read_bytes()
    manifest = json.loads((root / 'manifest.json').read_text())
    ready = json.loads((root / 'READY.json').read_text())
    assert manifest['kind'] == native.KIND and manifest['provider'] == 'lerobot'
    assert ready == {'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()}
    assert {k: v for k, v in native._inventory(root).items() if k not in {'manifest.json', 'READY.json'}} == manifest['files']
    reproduced = native.reproduce_native_processing(root, runtimes=unit['runtimes'])
    assert reproduced['reproduction_matches'] and reproduced['status'] == 'pass'
    assert unit['calls'][-1]['operation'] == 'replay'
    assert 'raw_samples' not in unit['calls'][-1] and 'model' not in unit['calls'][-1]


def test_no_retention_reports_exports_unavailable(fake_fixed_worker):
    result = fake_fixed_worker['probe']()
    assert not result['retained'] and result['artifact_files_available'] is False
    assert 'removed' in result['retention_note']


def test_publication_never_overwrites_or_writes_inside_originals(fake_fixed_worker):
    unit = fake_fixed_worker
    unit['probe'](unit['destination'])
    before = native._inventory(unit['destination'])
    for target in (unit['destination'], unit['model'] / 'new-capsule', unit['dataset'] / 'new-capsule'):
        with pytest.raises(InputError, match='new native capsule'):
            unit['probe'](target)
    assert native._inventory(unit['destination']) == before
    assert len(unit['calls']) == 1


@pytest.mark.parametrize('mutation', ['changed', 'extra', 'missing', 'uncommitted'])
def test_capsule_tamper_and_membership_fail_before_worker(fake_fixed_worker, mutation):
    unit = fake_fixed_worker
    root = unit['destination']
    unit['probe'](root)
    if mutation == 'changed':
        (root / 'samples/sample_0.json').write_text('{}')
    elif mutation == 'extra':
        (root / 'extra.txt').write_text('new')
    elif mutation == 'missing':
        (root / 'expected.json').unlink()
    else:
        (root / 'READY.json').write_text('{}')
    with pytest.raises(InputError):
        native.reproduce_native_processing(root, runtimes=unit['runtimes'])
    assert len(unit['calls']) == 1


def test_verifier_drift_does_not_evaluate_candidate(fake_fixed_worker):
    unit = fake_fixed_worker
    unit['probe'](unit['destination'])
    unit['state']['verifier'] = {'unit_verifier': 2}
    result = native.reproduce_native_processing(unit['destination'], runtimes=unit['runtimes'])
    assert result['status'] == 'unknown' and result['verifier_matches'] is False
    assert result['candidate_evaluated'] is False and len(unit['calls']) == 1


def test_replay_numerical_difference_is_error(fake_fixed_worker):
    unit = fake_fixed_worker
    unit['probe'](unit['destination'])
    unit['state']['changed_replay'] = True
    result = native.reproduce_native_processing(unit['destination'], runtimes=unit['runtimes'])
    assert result['status'] == 'error' and result['reproduction_matches'] is False


@pytest.mark.parametrize('status', ['partial', 'error'])
def test_incomplete_native_observation_cannot_publish(fake_fixed_worker, status):
    unit = fake_fixed_worker
    unit['state']['execution_status'] = status
    with pytest.raises(InputError, match='incomplete'):
        unit['probe'](unit['destination'])
    assert not unit['destination'].exists()
    assert not list(unit['destination'].parent.glob('.policy-doctor-native-*'))
    assert unit['probe']()['status'] == 'unknown'


def test_provider_must_be_operator_configured(fake_fixed_worker):
    unit = fake_fixed_worker
    with pytest.raises(InputError, match='explicitly configured'):
        native.probe_native_processing('groot_state', unit['model'], unit['dataset'], runtimes=unit['runtimes'],
            samples=[{'episode_index': 0, 'frame_index': 0}])
    assert not unit['calls']
