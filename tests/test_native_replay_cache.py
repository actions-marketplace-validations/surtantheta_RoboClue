"""A retained replay can load runtime config after the initial cache is removed."""
import asyncio
import json
import sys

import pytest

from policy_doctor import native_processing as native
from policy_doctor.artifacts import InputError
from policy_doctor.server import create_server


def config(tmp_path):
    source = tmp_path / 'trusted'
    source.mkdir()
    cache = tmp_path / 'removed-initial-cache'
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps({'providers': {'lerobot': {
        'python': sys.executable, 'source_root': str(source), 'cache_root': str(cache)}}}))
    return path, cache


def test_replay_config_accepts_missing_initial_cache_but_default_probe_config_refuses(tmp_path):
    path, cache = config(tmp_path)
    with pytest.raises(InputError, match='cache root must exist'):
        native.load_native_runtimes(path)
    runtime = native.load_native_runtimes(path, require_initial_cache=False)
    assert runtime['lerobot']['cache_root'] == str(cache)
    assert not cache.exists()


@pytest.mark.parametrize('mode', ['observation', 'processor_source'])
def test_initial_probe_refuses_removed_configured_cache_before_native_execution(tmp_path, monkeypatch, mode):
    path, _ = config(tmp_path)
    runtime = native.load_native_runtimes(path, require_initial_cache=False)
    def forbidden(*args, **kwargs):
        pytest.fail('An unavailable configured initial cache must be refused before native execution')
    monkeypatch.setattr(native, '_fingerprint', forbidden)
    import policy_doctor.native_source_processing as source
    monkeypatch.setattr(source, 'probe_native_processor_source', forbidden)
    with pytest.raises(InputError, match='cache root must exist'):
        native.probe_native_processing('lerobot', tmp_path, tmp_path, runtimes=runtime,
            samples=[{'episode_index': 0, 'frame_index': 0}], mode=mode)


def test_mcp_can_start_for_retained_replay_with_removed_initial_cache(tmp_path):
    path, cache = config(tmp_path)
    server = create_server([str(tmp_path)], case_root=str(tmp_path), native_runtime=str(path))
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {'reproduce_native_processing', 'probe_native_processing'} <= names
    assert not cache.exists()


def test_deferred_cache_check_does_not_defer_trusted_source_existence(tmp_path):
    path, _ = config(tmp_path)
    value = json.loads(path.read_text())
    value['providers']['lerobot']['source_root'] = str(tmp_path / 'missing-source')
    path.write_text(json.dumps(value))
    with pytest.raises(InputError, match='interpreter/source must exist'):
        native.load_native_runtimes(path, require_initial_cache=False)


def test_cli_replay_loads_real_config_after_initial_cache_removal(tmp_path, monkeypatch, capsys):
    from policy_doctor import cli
    path, cache = config(tmp_path)
    calls = []
    def retained_replay(directory, *, runtimes):
        calls.append((directory, runtimes))
        return {'kind': 'native_processing_reproduction', 'status': 'pass', 'reproduction_matches': True}
    monkeypatch.setattr(native, 'reproduce_native_processing', retained_replay)
    assert cli.main(['reproduce-native-probe', 'retained-case', '--native-runtime', str(path), '--json']) == 0
    assert calls[0][1]['lerobot']['cache_root'] == str(cache)
    assert not cache.exists()
    assert json.loads(capsys.readouterr().out)['reproduction_matches'] is True
