import copy
import hashlib
import json
import urllib.error
from pathlib import Path

import pytest

from policy_doctor import upgrade_watch as watch
from policy_doctor.artifacts import InputError


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    """Generated coordination controls; actual native/public proof is separate."""
    old, new = list(watch.ISAACLAB_SOURCES)
    owned = tmp_path / 'recording.bin'
    owned.write_bytes(b'actual-current-owned-bytes')
    version = tmp_path / 'consumer.json'
    version.write_text(json.dumps({'revision': old}))
    criterion = tmp_path / 'criterion.json'
    criterion.write_text('{"intent":"explicit retained test intent"}')
    manifest = tmp_path / 'native.json'
    manifest.write_text(json.dumps({'contract': identity(criterion)}))
    artifact = identity(owned)
    native = {'criterion_sha256': identity(criterion)['sha256'], 'assets': [{
        'asset_id': 'native-row', 'status': 'changed', 'code': 'retained_boundary_criterion_violated',
        'criterion_intent': {'field_orders': {'root': 'XYZW', 'ee': 'XYZW'},
                             'reference_adapter': 'explicit canonicalized reference'},
        'comparison_basis': {'causal_attribution': 'not established', 'reference_adapter': 'explicit canonicalized reference'},
        'reference': {'consumer': 'isaaclab-hdf5-xyzw', 'revision': old,
                      'configs': {'recording': artifact}, 'source_files': {}, 'capture': artifact},
        'candidate': {'consumer': 'isaaclab-hdf5-xyzw', 'revision': new,
                      'configs': {'recording': artifact}, 'source_files': {}, 'capture': artifact},
    }]}
    config = {'schema_version': 1, 'kind': 'owned_upgrade_watch',
              'watches': [{'watch_id': 'isaac', 'repository': 'isaac-sim/IsaacLab', 'kind': 'commit', 'ref': new}],
              'assets': [{'asset_id': 'my-recording', 'consumer': 'isaaclab-hdf5-xyzw', 'watch_id': 'isaac',
                          'files': {'recording': str(owned)},
                          'version': {'path': str(version), 'format': 'json', 'selector': '/revision'},
                          'native_comparison': {'manifest': str(manifest), 'manifest_sha256': identity(manifest)['sha256'],
                                                'asset_id': 'native-row', 'criterion_sha256': identity(criterion)['sha256']}}]}

    def request(url):
        value = {'sha': new, 'html_url': 'https://github.com/isaac-sim/IsaacLab/commit/' + new}
        raw = json.dumps(value).encode()
        return value, {'url': url, 'body_sha256': hashlib.sha256(raw).hexdigest(),
                       'body_utf8': raw.decode(), 'size_bytes': len(raw), 'etag': 'recorded-but-never-reused'}

    monkeypatch.setattr(watch, '_request_json', request)
    monkeypatch.setattr(watch, 'assess_upgrade_impact', lambda _: copy.deepcopy(native))
    return {'root': tmp_path, 'config': config, 'native': native, 'owned': owned, 'version': version,
            'manifest': manifest, 'old': old, 'new': new}


def scan(pilot, **kwargs):
    path = pilot['root'] / 'watch.json'
    path.write_text(json.dumps(pilot['config']))
    return watch.scan_once(path, pilot['root'] / 'state', **kwargs)


def test_owned_asset_observed_upstream_retained_criterion_local_alert(pilot):
    result = scan(pilot)
    report = result['report']
    assert result['exit_code'] == 1 and report['counts'] == {'affected': 1, 'unaffected': 0, 'unknown': 0}
    asset = report['assets'][0]
    assert asset['inventory']['files']['recording'] == dict(identity(pilot['owned']), size_bytes=len(pilot['owned'].read_bytes()))
    assert asset['inventory']['version_provenance']['selector'] == '/revision'
    assert asset['observed_revision'] == pilot['new']
    assert asset['criterion_intent']['reference_adapter'] == 'explicit canonicalized reference'
    assert asset['comparison_basis']['causal_attribution'] == 'not established'
    assert 'SAME criterion' in asset['remedy']
    assert report['input_postguards'] == 'pass'
    assert Path(result['report_path']).exists()
    assert 'Criterion intent:' in Path(result['alerts_path']).read_text()


def test_native_healthy_is_unaffected_with_separate_rule_intent(pilot):
    native = pilot['native']['assets'][0]
    native.update(status='healthy', code='retained_criterion_preserved')
    native['criterion_intent']['field_orders']['ee'] = 'WXYZ'
    result = scan(pilot)
    assert result['exit_code'] == 0 and result['report']['assets'][0]['status'] == 'unaffected'
    assert result['report']['assets'][0]['criterion_intent']['field_orders']['ee'] == 'WXYZ'


@pytest.mark.parametrize('control', ['missing_native', 'unsupported_consumer', 'version_range', 'upstream', 'repo'])
def test_version_changes_and_unsupported_coverage_remain_unknown(pilot, monkeypatch, control):
    asset = pilot['config']['assets'][0]
    if control == 'missing_native':
        asset.pop('native_comparison')
    elif control == 'unsupported_consumer':
        asset['consumer'] = 'unqualified-simulator'
    elif control == 'version_range':
        pilot['version'].write_text('{"revision": ">=3"}')
    elif control == 'repo':
        pilot['config']['watches'][0]['repository'] = 'other/project'
    else:
        def request(url):
            return {'sha': 'f' * 40}, {'url': url}
        monkeypatch.setattr(watch, '_request_json', request)
    result = scan(pilot)
    assert result['exit_code'] == 3
    assert result['report']['counts']['unknown'] == 1
    assert result['report']['assets'][0]['version_advisory_only']


@pytest.mark.parametrize('control', ['owned_config_changed', 'comparison_changed', 'criterion_changed', 'revision_mismatch'])
def test_stale_evidence_never_inherits_a_previous_healthy_result(pilot, control):
    pilot['native']['assets'][0].update(status='healthy', code='retained_criterion_preserved')
    first = scan(pilot)
    assert first['report']['status'] == 'unaffected'
    if control == 'owned_config_changed':
        pilot['owned'].write_bytes(b'new-current-configuration')
    elif control == 'comparison_changed':
        pilot['manifest'].write_text('{}')
    elif control == 'criterion_changed':
        (pilot['root'] / 'criterion.json').write_text('{"new_rule":true}')
    else:
        pilot['native']['assets'][0]['candidate']['revision'] = pilot['old']
    second = scan(pilot)
    assert second['report']['status'] == 'unknown'
    assert second['report']['previous_scan_id'] == first['report']['scan_id']
    assert second['report']['previous_results_reused'] is False


@pytest.mark.parametrize('offline', [True, False])
def test_offline_and_failed_refresh_do_not_promote_cached_observation(pilot, monkeypatch, offline):
    first = scan(pilot)

    def failed(_):
        if offline:
            raise AssertionError('Offline scan must not attempt any network')
        raise urllib.error.URLError('controlled network failure')

    monkeypatch.setattr(watch, '_request_json', failed)
    result = scan(pilot, offline=offline)
    assert result['exit_code'] == 3 and result['report']['status'] == 'unknown'
    observed = result['report']['observations'][0]
    assert observed['status'] == 'unknown' and 'revision' not in observed
    assert observed['previous_observations_reused'] is False
    assert first['report']['scan_id'] != result['report']['scan_id']


@pytest.mark.parametrize(('format_', 'selector', 'text'), [
    ('toml', '/tool/consumer/revision', '[tool.consumer]\nrevision = "v2.3.2"\n'),
    ('requirements', 'isaaclab', 'isaaclab==2.3.2\n'),
])
def test_owned_manifest_version_sources_have_exact_provenance(pilot, format_, selector, text):
    pilot['version'].write_text(text)
    pilot['config']['assets'][0]['version'].update(format=format_, selector=selector)
    report = scan(pilot)['report']
    assert report['assets'][0]['declared_revision'] == pilot['old']
    assert report['assets'][0]['inventory']['version_provenance']['sha256'] == identity(pilot['version'])['sha256']


def test_requirements_range_is_not_a_detected_loaded_version(pilot):
    pilot['version'].write_text('isaaclab>=2.0\n')
    pilot['config']['assets'][0]['version'].update(format='requirements', selector='isaaclab')
    asset = scan(pilot)['report']['assets'][0]
    assert asset['status'] == 'unknown' and asset['inventory']['declared_version'] is None


def test_explicit_release_observation_resolves_tag_to_immutable_commit(pilot, monkeypatch):
    calls = []

    def request(url):
        calls.append(url)
        return ({'tag_name': 'v3.0.0', 'published_at': '2026-01-01'} if '/releases/' in url else
                {'sha': pilot['new']}), {'url': url}

    monkeypatch.setattr(watch, '_request_json', request)
    pilot['config']['watches'][0].update(kind='release', ref='v3.0.0')
    observed = scan(pilot)['report']['observations'][0]
    assert len(calls) == 2 and '/releases/tags/v3.0.0' in calls[0]
    assert observed['revision'] == pilot['new'] and observed['release_tag'] == 'v3.0.0'


def test_history_is_bounded_and_report_tampering_is_not_silently_overwritten(pilot, monkeypatch):
    monkeypatch.setattr(watch, 'MAX_HISTORY', 2)
    first = scan(pilot)
    second = scan(pilot)
    third = scan(pilot)
    assert not Path(first['report_path']).exists()
    assert Path(second['report_path']).exists() and Path(third['report_path']).exists()
    state = json.loads((pilot['root'] / 'state/state.json').read_text())
    assert len(state['history']) == 2
    Path(third['report_path']).write_text('{}')
    with pytest.raises(InputError, match='report changed'):
        scan(pilot)


def test_state_symlinks_and_concurrent_scans_are_refused(pilot):
    import fcntl

    state = pilot['root'] / 'state'
    state.mkdir()
    with (state / 'scan.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(InputError, match='Another scan'):
            scan(pilot)
    (state / 'scan.lock').unlink()
    (state / 'scan.lock').symlink_to(pilot['owned'])
    with pytest.raises(OSError):
        scan(pilot)


def test_public_network_request_is_bounded_and_has_no_conditional_cache_headers(monkeypatch):
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, count):
            assert count == watch.MAX_RESPONSE + 1
            return b'x' * count

    class Opener:
        def open(self, request, timeout):
            assert timeout == watch.TIMEOUT
            headers = {key.lower(): value for key, value in request.header_items()}
            assert headers['cache-control'] == 'no-cache'
            assert 'authorization' not in headers and 'if-none-match' not in headers
            return Response()

    monkeypatch.setattr(watch.urllib.request, 'build_opener', lambda *_: Opener())
    with pytest.raises(InputError, match='256 KiB'):
        watch._request_json('https://api.github.com/repos/isaac-sim/IsaacLab/git/commits/' + 'a' * 40)


def test_public_watch_redirect_cannot_switch_hosts():
    with pytest.raises(InputError, match='redirect'):
        watch._PublicRedirect().redirect_request(None, None, 302, 'redirect', {}, 'http://127.0.0.1/private')


def test_duplicate_config_and_current_source_changes_refuse_publication(pilot, monkeypatch):
    original = watch._evaluate

    def mutate(*args):
        result = original(*args)
        pilot['version'].write_text('{"revision":"changed-during-scan"}')
        return result

    monkeypatch.setattr(watch, '_evaluate', mutate)
    with pytest.raises(InputError, match='changed during scan'):
        scan(pilot)
    assert not list((pilot['root'] / 'state').glob('scan-*.json'))


def test_cli_writes_local_outputs_and_no_product_or_message_action(pilot, capsys):
    config = pilot['root'] / 'cli.json'
    config.write_text(json.dumps(pilot['config']))
    assert watch.main(['scan-once', '--config', str(config), '--state-dir', str(pilot['root'] / 'state'), '--json']) == 1
    response = json.loads(capsys.readouterr().out)
    assert response['status'] == 'affected' and Path(response['alerts']).exists()
    assert watch.main(['catalog']) == 0
