"""Regression witnesses for the real 0.23 transient-cache retention finding."""
import hashlib
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_processing import _durable_payload


def report():
    return {'provider': 'lerobot', 'samples': [{'routes': [{'name': 'saved', 'trace': {'path': 'traces/sample_0_saved.json'}}]}],
            'exports': [{'path': 'exports/saved', 'route': 'saved', 'config_filename': 'policy_preprocessor.json',
                         'files': [{'path': 'policy_preprocessor.json'}]}]}


def payload(tmp_path):
    (tmp_path / 'traces').mkdir()
    (tmp_path / 'traces/sample_0_saved.json').write_text('{}')
    (tmp_path / 'exports/saved').mkdir(parents=True)
    (tmp_path / 'exports/saved/policy_preprocessor.json').write_text(json.dumps({
        'steps': [{'registry_name': 'device_processor', 'config': {'device': 'cpu'}}]}))
    (tmp_path / 'samples').mkdir()
    (tmp_path / 'samples/sample_0.npz').write_bytes(b'Explicit unit placeholder')
    (tmp_path / 'samples/sample_0.json').write_text('{}')
    manifest = {'provider': 'lerobot', 'schema': 'policy_doctor.native_lerobot.v1', 'exports': report()['exports'],
                'samples': [{field: {'path': 'samples/sample_0.' + suffix,
                           'sha256': hashlib.sha256((tmp_path / ('samples/sample_0.' + suffix)).read_bytes()).hexdigest()}
                             for field, suffix in [('arrays', 'npz'), ('metadata', 'json')]}]}
    (tmp_path / 'worker_manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'result.json').write_text(json.dumps(report()))
    (tmp_path / 'runtime/datasets-cache').mkdir(parents=True)
    (tmp_path / 'runtime/datasets-cache/_home_private_author_cache.lock').write_text('')
    return tmp_path


def test_observed_native_cache_is_removed_but_replay_assets_survive(tmp_path):
    stage = payload(tmp_path)
    _durable_payload(stage, report())
    assert not (stage / 'runtime').exists()
    assert json.loads((stage / 'exports/saved/policy_preprocessor.json').read_text())['steps'][0]['config'] == {'device': 'cpu'}
    assert (stage / 'traces/sample_0_saved.json').is_file()


@pytest.mark.parametrize('where', ['trace', 'array_export', 'native_export'])
def test_temporary_dependency_refused_before_cache_deletion(tmp_path, where):
    stage = payload(tmp_path)
    value = report()
    name = 'runtime/datasets-cache/_home_private_author_cache.lock'
    if where == 'trace':
        value['samples'][0]['routes'][0]['trace']['path'] = name
    elif where == 'array_export':
        value['samples'][0]['routes'][0]['output'] = {'state': {'export': name}}
    else:
        value['exports'][0]['path'] = name
        manifest = json.loads((stage / 'worker_manifest.json').read_text())
        manifest['exports'] = value['exports']
        (stage / 'worker_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(InputError):
        _durable_payload(stage, value)
    assert (stage / 'runtime').is_dir()


def test_unexpected_payload_is_not_silently_published(tmp_path):
    stage = payload(tmp_path)
    (stage / 'unapproved.py').write_text('raise AssertionError("never executed")')
    with pytest.raises(InputError, match='Unexpected'):
        _durable_payload(stage, report())


def test_trace_descriptor_must_bind_actual_retained_bytes(tmp_path):
    stage = payload(tmp_path)
    value = report()
    value['samples'][0]['routes'][0]['trace']['sha256'] = '0' * 64
    with pytest.raises(InputError, match='descriptor'):
        _durable_payload(stage, value)


def test_runtime_symlink_is_not_followed_or_deleted(tmp_path):
    stage, outside = tmp_path / 'stage', tmp_path / 'outside'
    stage.mkdir()
    outside.mkdir()
    (outside / 'keep').write_text('unchanged')
    (stage / 'runtime').symlink_to(outside, target_is_directory=True)
    with pytest.raises(InputError):
        _durable_payload(stage, {'samples': [], 'exports': []})
    assert (outside / 'keep').read_text() == 'unchanged'
