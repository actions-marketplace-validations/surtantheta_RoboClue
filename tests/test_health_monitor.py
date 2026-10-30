"""A recurrent inventory monitor preserves current uncertainty and durable evidence."""
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError

monitor = importlib.import_module('policy_doctor.health_monitor')
cards = importlib.import_module('policy_doctor.health_cards')
SHA = 'a' * 40
OTHER = 'b' * 40
SOURCE = 'hf://org/explicit-dataset'


def inventory(**changes):
    return {'schema_version': 1, 'datasets': [{'id': 'robot', 'source': SOURCE,
            'revision': SHA, 'consumer': 'bounded-input-check', **changes}]}


def card(entry, revision, *, status='pass', partial=False):
    """An explicitly generated checker envelope, never a public-data observation."""
    request = cards._request(entry['source'], entry['consumer'], entry['config'], revision, True)
    report = {'status': status, 'scope': 'Generated checker response for monitor controls',
              'findings': [], 'artifacts': [], 'provenance': [{'source': SOURCE, 'resolved_revision': revision}],
              'components': {key: {'status': status, 'coverage': {'complete': not partial}}
                             for key in ('numeric_rows', 'dataset_statistics', 'episode_windows', 'video_index')}}
    payload = cards._payload(request, report)
    payload['card_id'] = 'sha256:' + cards._digest(payload)
    return payload


@pytest.fixture
def engine(monkeypatch):
    calls = []
    state = {'revision': SHA, 'status': 'pass', 'partial': False}

    def operation(request, directory, timeout):
        calls.append(copy.deepcopy(request))
        assert timeout > 0
        if state.get('failure'):
            raise state['failure']
        if request['operation'] == 'resolve':
            return {'resolved_revision': state['revision'], 'public_identity_revalidated': True,
                    'api_sha256': 'c' * 64}
        entry = {'source': request['source'], 'consumer': request['consumer'], 'config': request['config']}
        return card(entry, request['revision'], status=state['status'], partial=state['partial'])

    monkeypatch.setattr(monitor, '_operation', operation)
    return calls, state


def published(root):
    return json.loads((root / 'state.json').read_text())


def test_two_cycles_revalidate_revision_and_reuse_bound_assessment(tmp_path, engine):
    calls, state = engine
    root = tmp_path / 'monitor'
    first = monitor.scan_once(inventory(), root, fetch=True)
    second = monitor.scan_once(inventory(), root, fetch=True)
    assert first['status'] == second['status'] == 'pass'
    assert [request['operation'] for request in calls] == ['resolve', 'scan', 'resolve']
    assert first['entries'][0]['card_id'] == second['entries'][0]['card_id']
    assert second['entries'][0]['assessment'].startswith('cached immutable assessment')
    assert len(published(root)['history']) == 2
    for result in (first, second):
        run = root / 'runs' / result['run_id']
        assert (run / 'robot/card/health-card.json').is_file()
        assert (run / 'robot/card/badge.svg').is_file()
        assert json.loads((run / 'report.json').read_text()) == result
    text = (root / 'index.html').read_text()
    assert second['run_id'] in text and first['run_id'] in text  # historical links remain explicit
    assert 'No Hub census' in text


@pytest.mark.parametrize('change', ['revision', 'config', 'consumer', 'engine'])
def test_changed_binding_forces_a_new_scan(tmp_path, engine, monkeypatch, change):
    calls, state = engine
    root = tmp_path / 'monitor'
    first = monitor.scan_once(inventory(), root, fetch=True)
    request = inventory()
    if change == 'revision':
        state['revision'] = OTHER
        request['datasets'][0]['revision'] = OTHER
    elif change == 'config':
        request['datasets'][0]['config'] = {'max_rows': 1}
    elif change == 'consumer':
        request['datasets'][0]['consumer'] = 'another-job'
    else:
        original = monitor._engine_identity
        monkeypatch.setattr(monitor, '_engine_identity', lambda: {**original(), 'sha256': 'e' * 64})
    second = monitor.scan_once(request, root, fetch=True)
    assert len([call for call in calls if call['operation'] == 'scan']) == 2
    assert first['entries'][0]['cache_key'] != second['entries'][0]['cache_key']
    assert second['entries'][0]['assessment'] == 'new bounded assessment'


@pytest.mark.parametrize('failure', [TimeoutError('deadline'), InputError('HubNotPublic: gated'),
                                    InputError('repository identity mismatch'), OSError('unavailable')])
def test_failure_after_pass_publishes_unknown_without_historical_green(tmp_path, engine, failure):
    calls, state = engine
    root = tmp_path / 'monitor'
    first = monitor.scan_once(inventory(), root, fetch=True)
    original_card = (root / 'runs' / first['run_id'] / 'robot/card/health-card.json').read_bytes()
    state['failure'] = failure
    second = monitor.scan_once(inventory(), root, fetch=True)
    assert second['status'] == 'unknown' and second['entries'][0]['reason_type'] == type(failure).__name__
    run = root / 'runs' / second['run_id'] / 'robot'
    assert not (run / 'card').exists()
    assert '#276b37' not in (run / 'badge.svg').read_text()
    assert second['run_id'] in (root / 'index.html').read_text()
    assert (root / 'runs' / first['run_id'] / 'robot/card/health-card.json').read_bytes() == original_card
    assert published(root)['current']['status'] == 'unknown'


def test_offline_cycle_does_not_reuse_public_pass_or_make_network_call(tmp_path, engine):
    calls, state = engine
    root = tmp_path / 'monitor'
    monitor.scan_once(inventory(), root, fetch=True)
    before = len(calls)
    result = monitor.scan_once(inventory(), root)
    assert result['status'] == 'unknown' and 'Offline cycle' in result['entries'][0]['reason']
    assert len(calls) == before


@pytest.mark.parametrize('status,partial', [('unknown', True), ('error', True), ('error', False)])
def test_partial_and_known_error_outcomes_never_become_pass(tmp_path, engine, status, partial):
    calls, state = engine
    state.update(status=status, partial=partial)
    root = tmp_path / 'monitor'
    for _ in range(2):
        result = monitor.scan_once(inventory(), root, fetch=True)
        assert result['status'] != 'pass'
    scans = sum(call['operation'] == 'scan' for call in calls)
    assert scans == (2 if partial or status == 'unknown' else 1)


def test_local_consumer_paths_bypass_cache(tmp_path, engine):
    calls, state = engine
    root = tmp_path / 'monitor'
    request = inventory(config={'checkpoint': str(tmp_path / 'checkpoint')})
    monitor.scan_once(request, root, fetch=True)
    monitor.scan_once(request, root, fetch=True)
    assert sum(call['operation'] == 'scan' for call in calls) == 2
    assert not list((root / 'cache').glob('*.json'))


def test_concurrency_lock_is_nonblocking_and_does_not_change_state(tmp_path):
    root = tmp_path / 'monitor'
    with monitor._lock(root):
        with pytest.raises(monitor.MonitorBusy):
            monitor.scan_once(inventory(), root, fetch=True)
    assert not (root / 'state.json').exists()


@pytest.mark.parametrize('target', ['inventory', 'state', 'cache', 'lock'])
def test_symlinks_and_special_files_are_refused_without_reading_targets(tmp_path, engine, target):
    root = tmp_path / 'monitor'
    root.mkdir()
    secret = tmp_path / 'secret'
    secret.write_text('SECRET NOT JSON')
    if target == 'inventory':
        path = tmp_path / 'inventory.json'
        os.mkfifo(path)
        result = monitor.scan_once(path, root)
        assert result['status'] == 'unknown' and result['configuration_error']
    elif target == 'state':
        (root / 'state.json').symlink_to(secret)
        with pytest.raises(InputError):
            monitor.scan_once(inventory(), root)
    elif target == 'cache':
        (root / 'cache').symlink_to(tmp_path)
        with pytest.raises(InputError):
            monitor.scan_once(inventory(), root)
    else:
        os.mkfifo(root / 'monitor.lock')
        with pytest.raises(InputError):
            monitor.scan_once(inventory(), root)
    assert secret.read_text() == 'SECRET NOT JSON'


@pytest.mark.parametrize('change', ['duplicate', 'path_id', 'extra_key', 'empty', 'arbitrary_url'])
def test_bad_inventory_replaces_current_index_with_unknown_before_any_intake(tmp_path, engine, change):
    calls, state = engine
    root = tmp_path / 'monitor'
    monitor.scan_once(inventory(), root, fetch=True)
    request = inventory()
    if change == 'duplicate':
        request['datasets'] *= 2
    elif change == 'path_id':
        request['datasets'][0]['id'] = '../outside'
    elif change == 'extra_key':
        request['datasets'][0]['token'] = 'not-authorized'
    elif change == 'empty':
        request['datasets'] = []
    else:
        request['datasets'][0]['source'] = 'https://example.invalid/data'
    before = len(calls)
    result = monitor.scan_once(request, root, fetch=True)
    assert result['configuration_error'] and result['status'] == 'unknown'
    assert len(calls) == before
    assert 'current outcome: <b>unknown' in (root / 'index.html').read_text()


def test_interrupted_run_is_retained_and_restart_checks_again(tmp_path, engine):
    root = tmp_path / 'monitor'
    root.mkdir()
    run_id = 'd' * 32
    pending = root / ('.pending-' + run_id)
    pending.mkdir()
    (pending / 'partial.txt').write_text('preserved failed attempt')
    monitor._write(root / 'state.json', {'schema_version': 1, 'current': {'phase': 'in_progress', 'run_id': run_id},
                                       'history': []})
    result = monitor.scan_once(inventory(), root, fetch=True)
    assert result['status'] == 'pass'
    assert (root / 'runs' / run_id / 'partial.txt').read_text() == 'preserved failed attempt'
    assert (root / 'runs' / run_id / 'interrupted.json').is_file()
    assert published(root)['history'][0]['phase'] == 'interrupted'


def test_in_progress_index_is_unknown_and_engine_change_discards_cache(tmp_path, engine, monkeypatch):
    original_engine = monitor._engine_identity
    calls = []

    def changed():
        value = original_engine()
        calls.append(value)
        return {**value, 'sha256': 'a' * 64 if len(calls) == 1 else 'b' * 64}

    operation = monitor._operation
    root = tmp_path / 'monitor'

    def observed(request, directory, timeout):
        assert published(root)['current']['phase'] == 'in_progress'
        assert 'current outcome: <b>unknown' in (root / 'index.html').read_text()
        return operation(request, directory, timeout)

    monkeypatch.setattr(monitor, '_engine_identity', changed)
    monkeypatch.setattr(monitor, '_operation', observed)
    result = monitor.scan_once(inventory(), root, fetch=True)
    assert result['status'] == 'unknown' and 'engine changed' in result['entries'][0]['assessment']
    assert not list((root / 'cache').glob('*.json'))


def test_resolver_rejects_private_or_wrong_identity_without_credentials(monkeypatch):
    base = {'sha': SHA, 'id': 'org/explicit-dataset', 'private': False, 'gated': False}
    for change in ({'private': True}, {'gated': 'auto'}, {'id': 'other/repo'}, {'sha': OTHER}):
        monkeypatch.setattr(monitor.hub, '_request', lambda *a, _v={**base, **change}, **k: json.dumps(_v).encode())
        with pytest.raises(InputError):
            monitor._resolve(SOURCE, SHA)


def test_corrupted_cache_never_yields_a_green_current_result(tmp_path, engine):
    root = tmp_path / 'monitor'
    first = monitor.scan_once(inventory(), root, fetch=True)
    key = first['entries'][0]['cache_key']
    cached = root / 'cache' / (key + '.json')
    value = json.loads(cached.read_text())
    value['card']['request']['consumer'] = 'tampered'
    cached.write_text(json.dumps(value))
    result = monitor.scan_once(inventory(), root, fetch=True)
    assert result['status'] == 'unknown'
    assert 'binding verification' in result['entries'][0]['reason']


def test_real_isolated_worker_timeout_and_local_cli_artifacts(tmp_path):
    # The empty folder is genuinely unsupported by preflight, not a fake pass.
    source = tmp_path / 'unsupported'
    source.mkdir()
    request = {'schema_version': 1, 'datasets': [{'id': 'local', 'source': str(source)}]}
    root = tmp_path / 'monitor'
    result = monitor.scan_once(request, root, timeout_seconds=20)
    assert result['status'] == 'unknown'
    assert (root / 'runs' / result['run_id'] / 'local/card/health-card.json').is_file()
    path = tmp_path / 'inventory.json'
    path.write_text(json.dumps(request))
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1] / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    process = subprocess.run([sys.executable, '-B', '-m', 'policy_doctor.health_monitor', 'scan-once',
                              '--inventory', str(path), '--state', str(root), '--timeout', '20'],
                             env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 3 and json.loads(process.stdout)['status'] == 'unknown'
    timed = monitor.scan_once(request, tmp_path / 'timed', timeout_seconds=0.001)
    assert timed['status'] == 'unknown' and timed['entries'][0]['reason_type'] == 'TimeoutError'


def test_timeout_kills_a_term_ignoring_descendant_group_and_reaps_worker(tmp_path, monkeypatch):
    pid_file = tmp_path / 'descendant.pid'
    child_code = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'
    worker_code = ('import subprocess,sys,time;from pathlib import Path;'
                   f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}]);'
                   f'Path({str(pid_file)!r}).write_text(str(p.pid));time.sleep(60)')
    monkeypatch.setattr(monitor, '_worker_command', lambda *a: [sys.executable, '-c', worker_code])
    with pytest.raises(TimeoutError, match='direct worker reaped'):
        monitor._operation({'operation': 'scan'}, tmp_path, 1)
    pid = int(pid_file.read_text())
    # Non-child descendants are reaped by their normal POSIX adoptive parent.
    # They must already be absent or terminated, never a running orphan.
    for _ in range(50):
        try:
            state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == 'Z':
            break
        time.sleep(0.02)
    else:
        pytest.fail('Assessment descendant survived process-group timeout cleanup')


@pytest.mark.parametrize('quota', ['runs', 'bytes', 'cache'])
def test_retention_limits_publish_current_unknown_without_deleting_history(tmp_path, engine, quota):
    calls, state = engine
    root = tmp_path / 'monitor'
    first = monitor.scan_once(inventory(), root, fetch=True)
    retained = (root / 'runs' / first['run_id'] / 'report.json').read_bytes()
    options = {'max_runs': 1} if quota == 'runs' else {'max_data_bytes': 65536} if quota == 'bytes' else {'max_cache_entries': 1}
    request = inventory()
    if quota == 'cache':
        cached = monitor.scan_once(request, root, fetch=True, **options)
        assert cached['status'] == 'pass'  # capacity does not prevent an existing cache hit
        request = inventory(config={'max_rows': 2})
    scans_before = sum(call['operation'] == 'scan' for call in calls)
    result = monitor.scan_once(request, root, fetch=True, **options)
    assert result['status'] == 'unknown'
    assert sum(call['operation'] == 'scan' for call in calls) == scans_before
    assert (root / 'runs' / first['run_id'] / 'report.json').read_bytes() == retained
    assert 'current outcome: <b>unknown' in (root / 'index.html').read_text()


def test_valid_unresolved_health_card_keeps_its_unknown_coverage(tmp_path, engine, monkeypatch):
    original = monitor._operation

    def unresolved(request, directory, timeout):
        value = original(request, directory, timeout)
        if request['operation'] == 'scan':
            value['report']['provenance'] = []
            payload = cards._payload(value['request'], value['report'])
            payload['card_id'] = 'sha256:' + cards._digest(payload)
            return payload
        return value

    monkeypatch.setattr(monitor, '_operation', unresolved)
    result = monitor.scan_once(inventory(), tmp_path / 'monitor', fetch=True)
    assert result['status'] == 'unknown'
    assert result['entries'][0]['card_path']
    assert not list((tmp_path / 'monitor/cache').glob('*.json'))


@pytest.mark.parametrize('selection', ['same', 'nested', 'metadata_file'])
def test_output_inside_input_is_refused_before_any_files_are_created(tmp_path, engine, selection):
    source = tmp_path / 'input'
    (source / 'meta').mkdir(parents=True)
    (source / 'meta/info.json').write_text('{}')
    (source / 'recorded.txt').write_text('original input stays unchanged')
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    selected = source / 'meta/info.json' if selection == 'metadata_file' else source
    state = source if selection == 'same' else source / 'monitor-output'
    with pytest.raises(InputError, match='separate'):
        monitor.scan_once({'schema_version': 1, 'datasets': [{'id': 'input', 'source': str(selected)}]}, state)
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    assert after == before
    assert not (source / 'monitor.lock').exists() and not (source / 'monitor-output').exists()
