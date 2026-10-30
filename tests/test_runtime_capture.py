import copy
import json
import os

import numpy as np
import pytest
from safetensors.numpy import load_file

from policy_doctor.artifacts import InputError, sha256
from policy_doctor.capture import audit_capture, capture_runtime


def write_manifest(path, data):
    path.write_text(json.dumps(data))
    (path.parent / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': sha256(path)}))


class LoadedPipeline:
    def __init__(self, case, side):
        self.config = json.loads((case['checkpoint'] / f'policy_{side}processor.json').read_text())
        self.states = {s['state_file'][:-12]: load_file(str(case['checkpoint'] / s['state_file']))
                       for s in self.config['steps'] if 'state_file' in s}
        from types import SimpleNamespace

        self.steps = [SimpleNamespace() for _ in self.config['steps']]
        if side == 'pre':
            case['_loaded_relative_step'] = self.steps[0]
        else:
            self.steps[-1].relative_step = case.get('_loaded_relative_step')

    def get_config(self):
        return copy.deepcopy(self.config)

    def state_dict(self):
        return {k: {n: v.copy() for n, v in state.items()} for k, state in self.states.items()}


def captured(case, tmp_path, **kwargs):
    pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
    result = capture_runtime(tmp_path / 'capture',
        policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
        preprocessor=pre, postprocessor=post,
        deployment=json.loads(case['deployment'].read_text()), **kwargs)
    return result


def test_capture_actual_loaded_override_and_roundtrip(case, tmp_path):
    pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
    for pipeline in [pre, post]:
        for state in pipeline.states.values():
            state['action.mean'] = np.array([5., .5])
    target = tmp_path / 'snapshot'
    result = capture_runtime(target,
        policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
        preprocessor=pre, postprocessor=post, deployment=json.loads(case['deployment'].read_text()))
    assert result['status'] == 'pass'
    assert result['capture']['integrity'] == 'verified'
    assert (target / 'checkpoint/pre_001.safetensors').is_file()
    assert load_file(str(target / 'checkpoint/pre_001.safetensors'))['action.mean'].tolist() == [5., .5]
    assert audit_capture(str(target), str(case['dataset']))['status'] == 'pass'
    assert not list(target.rglob('*model*'))


@pytest.mark.parametrize('stat,value', [('count', np.array([1300], dtype=np.int64)),
                                      ('q50', np.array([1., .5], dtype=np.float32))])
def test_native_auxiliary_stats_preserved_compared_and_excluded_from_decoder_shape(case, tmp_path, stat, value):
    from policy_doctor.artifacts import load_checkpoint
    from policy_doctor.runtime_diff import compare_runtime_captures
    from policy_doctor.trace import trace_actions

    for label, offset in [('reference', 0), ('changed', 1)]:
        pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
        for pipeline in (pre, post):
            for state in pipeline.states.values():
                state['action.' + stat] = value + offset
        snapshot = tmp_path / label
        result = capture_runtime(snapshot, policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
            preprocessor=pre, postprocessor=post, deployment=json.loads(case['deployment'].read_text()))
        assert result['status'] == 'pass'
        checkpoint = load_checkpoint(snapshot / 'checkpoint')
        assert checkpoint.post[0]['auxiliary_stats']['action'][stat] == (value + offset).tolist()
        assert trace_actions(str(snapshot / 'checkpoint'), str(case['sample']))['status'] == 'pass'
    compared = compare_runtime_captures(str(tmp_path / 'reference'), str(tmp_path / 'changed'))
    assert compared['status'] == 'different'
    change = next(item for item in compared['changes'] if item['pointer'].endswith('/auxiliary_statistics/action/' + stat))
    assert change['pointer_kind'] == 'normalized_auxiliary_statistic'
    assert all(item['tensor'] == 'action.' + stat for item in change['evidence'])


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('stat', ['count', 'q50'])
def test_auxiliary_group_presence_changes_retain_native_tensor_evidence(case, tmp_path, reverse, stat):
    from policy_doctor.runtime_diff import compare_runtime_captures

    for label in ('without', 'with'):
        pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
        if label == 'with':
            for pipeline in (pre, post):
                for state in pipeline.states.values():
                    state['action.' + stat] = np.array([1300], dtype=np.int64)
        capture_runtime(tmp_path / label, policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
                        preprocessor=pre, postprocessor=post, deployment=json.loads(case['deployment'].read_text()))
    names = ['without', 'with'] if not reverse else ['with', 'without']
    result = compare_runtime_captures(*(str(tmp_path / name) for name in names))
    assert result['status'] == 'different'
    auxiliary = [change for change in result['changes'] if '/auxiliary_statistics/' in change['pointer']]
    assert auxiliary and all(change['pointer_kind'] == 'normalized_auxiliary_statistic' for change in auxiliary)
    assert all(change['evidence'] and all(item['tensor'] == 'action.' + stat for item in change['evidence'])
               for change in auxiliary)


def test_changed_snapshot_rejected(case, tmp_path):
    captured(case, tmp_path)
    path = tmp_path / 'capture/checkpoint/config.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(InputError, match='digest'):
        audit_capture(str(tmp_path / 'capture'))


def test_no_contract_preserves_unknown_semantics(case, tmp_path):
    result = capture_runtime(tmp_path / 'capture',
        policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
        preprocessor=LoadedPipeline(case, 'pre'), postprocessor=LoadedPipeline(case, 'post'))
    assert result['status'] == 'unknown'
    assert any(f['code'] == 'deployment.missing' for f in result['findings'])


def test_capture_refuses_overwrite(case, tmp_path):
    captured(case, tmp_path)
    with pytest.raises(InputError, match='exists'):
        captured(case, tmp_path)


def test_mutating_pipeline_does_not_publish_partial_snapshot(case, tmp_path):
    class Changing(LoadedPipeline):
        def state_dict(self):
            result = super().state_dict()
            self.states['pre']['action.mean'] += 1
            return result
    with pytest.raises(InputError, match='changed'):
        capture_runtime(tmp_path / 'capture',
            policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
            preprocessor=Changing(case, 'pre'), postprocessor=LoadedPipeline(case, 'post'))
    assert not (tmp_path / 'capture').exists()


def test_capture_rejects_non_stats_state_without_exporting_weights(case, tmp_path):
    pre = LoadedPipeline(case, 'pre')
    pre.states['pre']['model.weight'] = np.ones((2, 2))
    with pytest.raises(InputError, match='statistics'):
        capture_runtime(tmp_path / 'capture',
            policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
            preprocessor=pre, postprocessor=LoadedPipeline(case, 'post'))
    assert not (tmp_path / 'capture').exists()


def test_capture_manifest_traversal_rejected(case, tmp_path):
    captured(case, tmp_path)
    path = tmp_path / 'capture/capture.json'
    data = json.loads(path.read_text())
    data['files'][0]['path'] = '../../outside.json'
    write_manifest(path, data)
    with pytest.raises(InputError, match='within'):
        audit_capture(str(path.parent))


def test_custom_converter_is_recorded_and_not_certified(case, tmp_path):
    pre = LoadedPipeline(case, 'pre')
    pre.to_transition = lambda batch: batch
    result = capture_runtime(tmp_path / 'capture',
        policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
        preprocessor=pre, postprocessor=LoadedPipeline(case, 'post'),
        deployment=json.loads(case['deployment'].read_text()))
    assert result['status'] == 'unknown'
    assert any(f['code'] == 'capture.converters' for f in result['findings'])
    assert '<lambda>' in result['capture']['converters']['pre']['to_transition']


def test_real_cli_and_readonly_mcp_capture(case, tmp_path):
    import asyncio
    import os
    import subprocess
    import sys
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    captured(case, tmp_path)
    result = subprocess.run([sys.executable, '-m', 'policy_doctor', 'audit-capture',
                             str(tmp_path / 'capture'), '--dataset', str(case['dataset']), '--json'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['capture']['integrity'] == 'verified'

    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                result = await client.call_tool('audit_runtime_capture', {'capture': str(tmp_path / 'capture')})
                assert not result.isError and result.structuredContent['status'] == 'pass'
                assert result.structuredContent['capture']['integrity'] == 'verified'
                result = await client.call_tool('audit_runtime_capture', {'capture': '/etc'})
                assert result.isError and 'outside' in result.content[0].text
    asyncio.run(asyncio.wait_for(exercise(), timeout=30))


@pytest.mark.parametrize('mutation', ['duplicate', 'boolean_schema', 'missing_converter', 'wrong_converter_type'])
def test_malformed_capture_manifests_fail_as_input_errors(case, tmp_path, mutation):
    captured(case, tmp_path)
    path = tmp_path / 'capture/capture.json'
    data = json.loads(path.read_text())
    if mutation == 'duplicate':
        data['files'].append(data['files'][0])
    elif mutation == 'boolean_schema':
        data['schema_version'] = True
    elif mutation == 'missing_converter':
        data['converters']['pre'] = {}
    else:
        data['converters']['pre']['to_transition'] = []
    write_manifest(path, data)
    with pytest.raises(InputError):
        audit_capture(str(path.parent))


def test_unsupported_tensor_dtype_is_a_structured_input_error(case, tmp_path):
    class UnsupportedTensor:
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            raise TypeError('unsupported BFloat16')
    class Pipeline(LoadedPipeline):
        def state_dict(self):
            data = super().state_dict()
            data['pre']['action.mean'] = UnsupportedTensor()
            return data
    with pytest.raises(InputError, match='dtype'):
        capture_runtime(tmp_path / 'capture',
            policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
            preprocessor=Pipeline(case, 'pre'), postprocessor=LoadedPipeline(case, 'post'))


@pytest.mark.parametrize('kind,status', [('missing', 'error'), ('external', 'unknown'), ('uninspected', 'unknown')])
def test_capture_records_actual_relative_binding_topology(case, tmp_path, kind, status):
    pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
    if kind == 'missing':
        post.steps[-1].relative_step = None
    elif kind == 'external':
        post.steps[-1].relative_step = object()
    else:
        del pre.steps
    result = capture_runtime(tmp_path / 'capture',
        policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
        preprocessor=pre, postprocessor=post, deployment=json.loads(case['deployment'].read_text()))
    assert result['status'] == status
    assert any(f['code'] == 'capture.relative_binding' and f['status'] == status for f in result['findings'])


def test_binding_mutation_during_capture_refuses_publication(case, tmp_path):
    pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
    original = post.state_dict
    calls = []
    def change():
        if calls:
            post.steps[-1].relative_step = None
        calls.append(True)
        return original()
    post.state_dict = change
    with pytest.raises(InputError, match='binding changed'):
        capture_runtime(tmp_path / 'capture',
            policy_config=json.loads((case['checkpoint'] / 'config.json').read_text()),
            preprocessor=pre, postprocessor=post)
    assert not (tmp_path / 'capture').exists()


def test_concurrent_empty_destination_is_not_replaced(case, tmp_path, monkeypatch):
    from policy_doctor import capture
    original = capture.publish_capture
    def concurrent(staging, destination):
        destination.mkdir(mode=0o755)
        os.chmod(destination, 0o755)  # The competing writer chooses its mode independently of umask.
        original(staging, destination)
    monkeypatch.setattr(capture, 'publish_capture', concurrent)
    with pytest.raises(InputError, match='exists'):
        captured(case, tmp_path)
    assert (tmp_path / 'capture').is_dir() and not list((tmp_path / 'capture').iterdir())
    assert (tmp_path / 'capture').stat().st_mode & 0o777 == 0o755
    assert not list(tmp_path.glob('.policy-doctor-capture-*'))


def test_omitting_relative_binding_cannot_certify_snapshot(case, tmp_path):
    captured(case, tmp_path)
    path = tmp_path / 'capture/capture.json'
    data = json.loads(path.read_text())
    data['relative_bindings'] = []
    write_manifest(path, data)
    with pytest.raises(InputError, match='every enabled'):
        audit_capture(str(path.parent))


def test_portable_publication_refuses_partial_capture_until_ready(case, tmp_path, monkeypatch):
    from policy_doctor import publication
    monkeypatch.setattr(publication.ctypes, 'CDLL', lambda *args, **kwargs: object())
    original = publication.shutil.copyfileobj
    observations = []
    def copying(source, target):
        with pytest.raises(InputError):
            audit_capture(str(tmp_path / 'capture'))
        observations.append(Path(source.name).name)
        original(source, target)
    from pathlib import Path
    monkeypatch.setattr(publication.shutil, 'copyfileobj', copying)
    result = captured(case, tmp_path)
    assert result['status'] == 'pass' and observations[-1] == 'READY.json'


def test_missing_commit_marker_is_not_a_verified_capture(case, tmp_path):
    captured(case, tmp_path)
    (tmp_path / 'capture/READY.json').unlink()
    with pytest.raises(InputError):
        audit_capture(str(tmp_path / 'capture'))


def test_runtime_snapshot_remains_verifiable_after_moving(case, tmp_path):
    import shutil
    captured(case, tmp_path)
    destination = tmp_path / 'support/capture'
    destination.parent.mkdir()
    shutil.move(str(tmp_path / 'capture'), destination)
    report = audit_capture(str(destination), str(case['dataset']))
    assert report['status'] == 'pass' and report['capture']['integrity'] == 'verified'
    assert report['capture']['relative_bindings'][0]['status'] == 'paired'
