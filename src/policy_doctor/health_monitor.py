"""Private, bounded inventory monitoring; run scan-once from cron or CI.

Only explicitly listed inputs are checked. Immutable reports are preserved; a
current failure never borrows a historical passing badge. No service is published.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from html import escape
from pathlib import Path

# The isolated worker loads only this explicitly selected product package.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_doctor import __version__, health_cards, hub  # noqa: E402
from policy_doctor.artifacts import InputError  # noqa: E402

MAX_JSON_BYTES = 2 * 1024**2
MAX_HISTORY = 64
DEFAULT_MAX_RUNS = 128
DEFAULT_MAX_CACHE = 256
DEFAULT_MAX_DATA_BYTES = 512 * 1024**2
IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
SHA = re.compile(r'[a-f0-9]{40}\Z')
LOCAL_CONFIG = ('checkpoint', 'deployment', 'groot_modality_config', 'openpi_source_root')
SCOPE = ('Only this operator-defined inventory, reported checks and coverage. Consumer names are caller labels. '
         'No Hub census, native-training, whole-dataset quality or physical-success guarantee.')


class MonitorBusy(InputError):
    """Another scan owns this monitor's nonblocking process lock."""


class _StateInputOverlap(InputError):
    """Output setup must never write inside a declared input tree."""


def _encode(value):
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise InputError('Monitor data must be finite JSON') from error
    if len(raw) > MAX_JSON_BYTES:
        raise InputError('Monitor JSON exceeds its 2 MiB bound')
    return raw


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _directory(path):
    path = Path(path).expanduser().absolute()
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            part.mkdir(mode=0o700)
            info = part.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise InputError(f'Monitor directories must not contain symlinks or special files: {part}')
    return path


def _read(path):
    path = Path(path)
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        with os.fdopen(os.open(path, flags), 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
                raise InputError('Monitor JSON input is not a bounded regular file')
            raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES:
                raise InputError('Monitor JSON input exceeds its byte bound')
        return json.loads(raw)
    except (OSError, ValueError) as error:
        raise InputError(f'Cannot read monitor JSON {path.name}: {type(error).__name__}') from error


def _atomic(path, data):
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise InputError(f'Refuse replacing a monitor link or special file: {path.name}')
    temporary = path.with_name('.' + path.name + '-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write(path, value):
    _atomic(path, _encode(value) + b'\n')


@contextmanager
def _lock(root):
    root = _directory(root)
    descriptor = os.open(root / 'monitor.lock', os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InputError('Monitor lock must be a regular file')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MonitorBusy('Another scan owns this monitor state; overlapping runs are refused') from error
        yield root
    finally:
        os.close(descriptor)


def _engine_identity():
    package = Path(health_cards.__file__).parent
    paths = sorted([*package.glob('*.py'), *package.glob('*.json')])
    if len(paths) > 256 or sum(path.lstat().st_size for path in paths) > 8 * 1024**2:
        raise InputError('Monitor engine identity exceeds its source bound')
    files = {}
    for path in paths:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise InputError('Monitor engine source must contain regular files')
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for name in ('numpy', 'safetensors', 'pyarrow', 'h5py'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    value = {'package_version': __version__, 'python': sys.version, 'files': files, 'dependencies': versions}
    return {'sha256': _hash(value), **value}


def _inventory(value, root):
    if isinstance(value, (str, Path)):
        value = _read(value)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'datasets'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or not isinstance(value['datasets'], list) or not 1 <= len(value['datasets']) <= health_cards.MAX_SOURCES):
        raise InputError('Inventory requires schema_version=1 and 1..16 explicit datasets')
    entries, seen = [], set()
    for item in value['datasets']:
        if not isinstance(item, dict) or item.keys() - {'id', 'source', 'consumer', 'revision', 'config'}:
            raise InputError('Inventory entries accept id/source/consumer/revision/config only')
        identifier = item.get('id')
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier) or identifier in seen:
            raise InputError('Inventory IDs must be distinct bounded ASCII names without paths')
        seen.add(identifier)
        request = health_cards._request(item.get('source'), item.get('consumer', 'unspecified'),
                                        item.get('config'), item.get('revision', 'main'), True)
        for source in [request['source'], *(request['config'].get(key) for key in LOCAL_CONFIG)]:
            if source and isinstance(source, str) and '://' not in source:
                local = Path(source).expanduser().resolve()
                if local.name == 'info.json' and local.parent.name == 'meta':
                    local = local.parent.parent
                if root.is_relative_to(local) or local.is_relative_to(root):
                    raise _StateInputOverlap('Monitor state must be separate from dataset and consumer input paths')
        entries.append({'id': identifier, 'source': request['source'], 'consumer': request['consumer'],
                        'revision': request['requested_revision'] or 'main', 'config': request['config']})
    return entries


def _resolve(source, revision):
    repo, revision = hub._repo(source), hub._revision(revision)
    url = f'https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true'
    raw = hub._request(url, limit=hub.MAX_API_BYTES)
    value = hub._json(raw)
    resolved = value.get('sha')
    if (not isinstance(resolved, str) or not SHA.fullmatch(resolved)
            or (SHA.fullmatch(revision) and resolved != revision)
            or not isinstance(value.get('id'), str) or value['id'].lower() != repo.lower()):
        raise InputError('Hub revision or repository identity does not match the explicit inventory')
    if value.get('private') is not False or value.get('gated') not in (None, False):
        raise hub.HubNotPublic('Monitor accepts public ungated repositories only; no credentials used')
    return {'resolved_revision': resolved, 'api_sha256': hashlib.sha256(raw).hexdigest(),
            'public_identity_revalidated': True}


def _worker(request, output):
    try:
        if request['operation'] == 'resolve':
            result = {'ok': True, 'result': _resolve(request['source'], request['revision'])}
        else:
            result = {'ok': True, 'result': health_cards.health_card(request['source'],
                consumer=request['consumer'], config=request['config'], fetch=request['fetch'],
                revision=request['revision'])}
    except Exception as error:
        result = {'ok': False, 'type': type(error).__name__, 'message': str(error)[:1024]}
    _write(output, result)


def _worker_command(input_file, output_file):
    return [sys.executable, '-I', '-B', str(Path(__file__)), '--_worker', str(input_file), str(output_file)]


def _operation(request, directory, timeout):
    input_file = directory / (request['operation'] + '-request.json')
    output_file = directory / (request['operation'] + '-response.json')
    _write(input_file, request)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    process = subprocess.Popen(_worker_command(input_file, output_file), env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        process.wait(timeout=max(0.01, timeout))
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        # Send KILL even if the direct worker already exited: a descendant may
        # ignore TERM. Only this worker's newly created process group is targeted.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
        raise TimeoutError('Dataset monitoring deadline expired; worker group terminated and direct worker reaped') from error
    if process.returncode and not output_file.exists():
        raise InputError(f'Isolated monitor worker exited {process.returncode}; no bounded structured result was returned')
    result = _read(output_file)
    if not result.get('ok'):
        raise InputError(f"{result.get('type', 'WorkerError')}: {result.get('message', 'No completed assessment')}")
    if process.returncode:
        raise InputError(f'Isolated monitor worker exited {process.returncode}; its result cannot qualify this cycle')
    return result['result']


def _cacheable(card, entry):
    # Local/mutable consumer paths require a new assessment. Path strings alone
    # cannot prove that their contents or selected checkpoint revision are unchanged.
    return (entry['source'].startswith('hf://') and not any(entry['config'].get(key) for key in LOCAL_CONFIG)
            and card['source_revision']['kind'] == 'hub_commit'
            and card['status'] in ('pass', 'warning', 'error')
            and all(part.get('status') in ('pass', 'warning', 'error')
                    and not any(part.get('coverage', {}).get(key) is False
                                for key in ('complete', 'complete_parquet_scan'))
                    for part in card['report'].get('components', {}).values())
            and not any(finding.get('status') == 'unknown' for finding in card['report'].get('findings', [])))


def _verify_card(card, entry, revision):
    if not health_cards.verify_health_card(card):
        raise InputError('Cached or returned health card failed its binding verification')
    request = card['request']
    if any(request.get(key) != entry[key] for key in ('source', 'consumer', 'config')):
        raise InputError('Health card source, consumer or configuration differs from this inventory')
    if entry['source'].startswith('hf://'):
        unresolved = card['source_revision']['kind'] == 'unresolved' and card['status'] in ('unknown', 'error')
        if ((card['source_revision'] != {'kind': 'hub_commit', 'value': revision} and not unresolved)
                or request['requested_revision'] != revision):
            raise InputError('Health card does not bind or explicitly leave unresolved the selected Hub revision')


def _badge(status, reason):
    title = escape('Current bounded monitor outcome: ' + status + '. ' + reason + '. ' + SCOPE)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="228" height="24" role="img" '
            f'aria-label="bounded checks: {status}"><title>{title}</title>'
            '<rect width="228" height="24" fill="#59636e"/>'
            f'<text x="114" y="16" fill="white" font-family="sans-serif" font-size="11" '
            f'text-anchor="middle">bounded checks: {status}</text></svg>\n').encode()


def _check_storage(root, max_runs, max_cache_entries, max_data_bytes, reserve):
    paths = [root / 'runs', root / 'cache', *root.glob('.pending-*')]
    count, total = 0, 0
    for path in paths:
        if not path.is_dir() or path.is_symlink():
            raise InputError('Monitor storage contains a link or special directory')
        for directory, directories, files in os.walk(path, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in directories):
                raise InputError('Monitor storage contains a directory link')
            for name in files:
                info = (Path(directory) / name).lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise InputError('Monitor storage contains a link or special file')
                count += 1
                total += info.st_size
                if count > 8192 or total > max_data_bytes:
                    raise InputError('Monitor data quota exhausted; archive old runs or use a new state directory')
    runs = sum(1 for _ in (root / 'runs').iterdir())
    cached = sum(1 for _ in (root / 'cache').iterdir())
    # Reserve worst-case bounded checker requests/responses/cards before scanning.
    if (runs >= max_runs or cached > max_cache_entries
            or total + reserve * (4 * MAX_JSON_BYTES + 65536) > max_data_bytes
            or count + reserve * 12 + 1 > 8192):
        raise InputError('Monitor retention quota reached; archive old runs/cache or use a new state directory')
    return {'runs': runs, 'cache_entries': cached, 'data_bytes': total, 'data_files': count}


def _index(report, history=()):
    rows = []
    for item in report['entries']:
        base = 'runs/' + report['run_id'] + '/' + item['id'] + '/'
        badge = f'<img alt="bounded checks: {item["status"]}" src="{base}badge.svg">' if report['phase'] == 'complete' else 'unknown (' + escape(report['phase']) + ')'
        rows.append('<tr><td>' + escape(item['id']) + '</td><td>' + escape(item['source']) + '</td><td>' +
                    escape(item.get('consumer', 'unspecified')) + '</td><td>' +
                    escape(item.get('resolved_revision') or 'unresolved') + '</td><td>' + badge + '</td><td>' +
                    escape(item.get('assessment', 'unassessed')) + '</td><td>' + escape(item.get('reason', '')) + '</td></tr>')
    body = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Private dataset monitor</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:1em}td,th{padding:.6em;text-align:left;vertical-align:top}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}code{overflow-wrap:anywhere}</style>'
            '<h1>Private dataset monitor</h1><p>' + escape(SCOPE) + '</p><p>Cycle: <code>' + report['run_id'] +
            '</code>; ' + escape(report['phase']) + '; current outcome: <b>' + report['status'] + '</b>; started ' +
            escape(report['started_at']) + '</p><p>' + escape(report.get('configuration_error', report.get('operational_error', ''))) + '</p>'
            '<p>Every cycle revalidates public revision identity before cache reuse. An offline or failed current check has no passing badge. Historical artifacts do not establish current coverage.</p>'
            '<table><tr><th>Inventory ID</th><th>Source</th><th>Consumer label</th><th>Commit</th><th>Bounded outcome</th><th>Assessment</th><th>Reason</th></tr>' +
            ''.join(rows) + '</table><p><a href="' + report.get('report_path', 'runs/' + report['run_id'] + '/report.json') + '">Cycle JSON and coverage links</a></p><h2>Retained history</h2><ul>')
    for item in history[-20:]:
        if isinstance(item, dict) and re.fullmatch('[a-f0-9]{32}', str(item.get('run_id', ''))):
            filename = 'interrupted.json' if item.get('phase') == 'interrupted' else 'report.json'
            body += '<li><a href="runs/' + item['run_id'] + '/' + filename + '">' + item['run_id'] + '</a>: ' + escape(str(item.get('status', 'unknown'))) + '</li>'
    body += '</ul></html>\n'
    return body.encode()


def _publish(root, report, history):
    # Publish an unknown in-progress page before any network work. A killed process
    # therefore cannot leave the previous successful cycle looking newly assessed.
    _atomic(root / 'index.html', _index(report, history))
    _write(root / 'state.json', {'schema_version': 1, 'current': report, 'history': history[-MAX_HISTORY:]})


def scan_once(inventory: dict | str | Path, state_dir: str | Path, *, fetch: bool = False,
              timeout_seconds: float = 300, max_runs: int = DEFAULT_MAX_RUNS,
              max_cache_entries: int = DEFAULT_MAX_CACHE, max_data_bytes: int = DEFAULT_MAX_DATA_BYTES) -> dict:
    """Run one explicit bounded inventory cycle and return its durable report.

    Network is opt-in. Unknown/failed assessments are current outcomes, never
    substitutions from history. Use this function or module CLI from cron/CI.
    """
    if type(fetch) is not bool or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 900:
        raise InputError('Require boolean fetch and a finite dataset deadline in (0,900] seconds')
    if (type(max_runs) is not int or not 1 <= max_runs <= 4096
            or type(max_cache_entries) is not int or not 1 <= max_cache_entries <= 4096
            or type(max_data_bytes) is not int or not 65536 <= max_data_bytes <= 32 * 1024**3):
        raise InputError('Retention limits require 1..4096 runs/cache entries and 64 KiB..32 GiB data')
    state_root = Path(state_dir).expanduser().absolute()
    configuration_error = None
    try:
        entries = _inventory(inventory, state_root)
    except _StateInputOverlap:
        raise  # refuse before creating even a lock/output directory inside inputs
    except (InputError, OSError, ValueError) as error:
        entries, configuration_error = [], str(error)[:1024]
    with _lock(state_root) as root:
        runs, cache = _directory(root / 'runs'), _directory(root / 'cache')
        previous = _read(root / 'state.json') if (root / 'state.json').exists() or (root / 'state.json').is_symlink() else {}
        history = previous.get('history', [])
        if not isinstance(history, list) or len(history) > MAX_HISTORY:
            raise InputError('Invalid bounded monitor history')
        prior = previous.get('current', {})
        if prior.get('phase') == 'in_progress':
            old_id = prior.get('run_id', '')
            if not re.fullmatch(r'[0-9a-f]{32}', old_id):
                raise InputError('Invalid interrupted run identity')
            pending = root / ('.pending-' + old_id)
            if pending.exists():
                if pending.is_symlink() or not pending.is_dir():
                    raise InputError('Interrupted run directory is not regular')
                _write(pending / 'interrupted.json', {'status': 'unknown', 'reason': 'Previous process stopped before durable completion'})
                pending.rename(runs / old_id)
            history.append({'run_id': old_id, 'status': 'unknown', 'phase': 'interrupted'})
        run_id = uuid.uuid4().hex
        engine = _engine_identity()
        report = {'schema_version': 1, 'kind': 'dataset_health_monitor_cycle', 'run_id': run_id,
                  'started_at': _now(), 'phase': 'in_progress', 'status': 'unknown', 'scope': SCOPE,
                  'engine': engine, 'entries': [], 'fetch_enabled': fetch, 'timeout_seconds': timeout_seconds}
        if configuration_error:
            report['configuration_error'] = configuration_error
        report['inventory_sha256'] = _hash(entries)
        report['entries'] = [{'id': entry['id'], 'source': entry['source'], 'consumer': entry['consumer'],
                              'config': entry['config'], 'requested_revision': entry['revision'],
                              'status': 'unknown', 'assessment': 'pending'} for entry in entries]
        report['retention_limits'] = {'max_runs': max_runs, 'max_cache_entries': max_cache_entries,
                                      'max_data_bytes': max_data_bytes, 'max_data_files': 8192}
        try:
            report['storage_before'] = _check_storage(root, max_runs, max_cache_entries, max_data_bytes, len(entries))
        except InputError as error:
            report.update(phase='blocked', operational_error=str(error), report_path='monitor-status.json')
            _write(root / 'monitor-status.json', report)
            _publish(root, report, history)
            return report
        work = _directory(root / ('.pending-' + run_id))
        _publish(root, report, history)
        results, cache_updates = [], []
        for entry in entries:
            item = {'id': entry['id'], 'source': entry['source'], 'consumer': entry['consumer'],
                    'config': entry['config'], 'status': 'unknown', 'assessment': 'unassessed',
                    'requested_revision': entry['revision']}
            directory = _directory(work / entry['id'])
            deadline = time.monotonic() + timeout_seconds
            try:
                resolved = None
                if entry['source'].startswith('hf://'):
                    if not fetch:
                        raise InputError('Offline cycle: current public revision/identity was not revalidated')
                    identity = _operation({'operation': 'resolve', 'source': entry['source'],
                                           'revision': entry['revision']}, directory, deadline - time.monotonic())
                    resolved = identity['resolved_revision']
                    if (not isinstance(resolved, str) or not SHA.fullmatch(resolved)
                            or identity.get('public_identity_revalidated') is not True):
                        raise InputError('Worker did not return an immutable Hub commit')
                    item.update(identity)
                key = _hash({'source': entry['source'], 'revision': resolved, 'consumer': entry['consumer'],
                             'config': entry['config'], 'engine': engine['sha256']})
                item['cache_key'] = key
                cached_file = cache / (key + '.json')
                card = None
                cache_allowed = entry['source'].startswith('hf://') and not any(entry['config'].get(name) for name in LOCAL_CONFIG)
                if cache_allowed and (cached_file.exists() or cached_file.is_symlink()):
                    cached = _read(cached_file)
                    if cached.get('key') == key and cached.get('engine') == engine['sha256']:
                        _verify_card(cached.get('card'), entry, resolved)
                        card = cached['card']
                        if not _cacheable(card, entry):
                            card = None
                if card is None:
                    reserved_keys = {str(path) for path, _value in cache_updates if not path.exists()}
                    if (cache_allowed and not cached_file.exists()
                            and sum(1 for _ in cache.iterdir()) + len(reserved_keys) >= max_cache_entries):
                        raise InputError('Monitor cache quota reached; archive unused cache entries before a new assessment')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Dataset deadline expired before assessment')
                    card = _operation({'operation': 'scan', 'source': entry['source'], 'consumer': entry['consumer'],
                                       'revision': resolved or entry['revision'], 'config': entry['config'],
                                       'fetch': fetch}, directory, deadline - time.monotonic())
                    _verify_card(card, entry, resolved)
                    item['assessment'] = 'new bounded assessment'
                    if _cacheable(card, entry):
                        cache_updates.append((cached_file, {'key': key, 'engine': engine['sha256'], 'card': card,
                                                           'origin_run_id': run_id, 'created_at': _now()}))
                else:
                    item['assessment'] = 'cached immutable assessment; revision revalidated this cycle'
                item.update(status=card['status'], card_id=card['card_id'],
                            resolved_revision=card['source_revision']['value'], selected_revision=resolved)
                health_cards.write_health_card(card, directory / 'card')
                _atomic(directory / 'badge.svg', health_cards.render_badge(card).encode())
                item['card_path'] = entry['id'] + '/card/health-card.json'
            except (InputError, OSError, TimeoutError, ValueError, KeyError, TypeError) as error:
                item.update(status='unknown', assessment='no current completed assessment',
                            reason=str(error)[:1024], reason_type=type(error).__name__)
                _atomic(directory / 'badge.svg', _badge('unknown', item['reason']))
            item['finished_at'] = _now()
            _write(directory / 'monitor-result.json', item)
            results.append(item)
        if _engine_identity() != engine:
            for item in results:
                item.update(status='unknown', assessment='engine changed during cycle', reason='Rerun under one stable engine identity')
                _atomic(work / item['id'] / 'badge.svg', _badge('unknown', item['reason']))
                _write(work / item['id'] / 'monitor-result.json', item)
        else:
            for path, value in cache_updates:
                _write(path, value)
        report.update(entries=results, phase='complete', finished_at=_now())
        report['status'] = next((status for status in health_cards.STATUSES if any(item['status'] == status for item in results)), 'unknown')
        report['counts'] = {status: sum(item['status'] == status for item in results) for status in health_cards.STATUSES}
        _write(work / 'report.json', report)
        # Flush every newly created artifact before exposing this immutable run.
        for path in work.rglob('*'):
            if path.is_file() and not path.is_symlink():
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        work.rename(runs / run_id)
        history.append({'run_id': run_id, 'status': report['status'], 'finished_at': report['finished_at'],
                        'inventory_sha256': report['inventory_sha256']})
        _publish(root, report, history)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--_worker', nargs=2, metavar=('REQUEST', 'OUTPUT'), help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest='command')
    once = commands.add_parser('scan-once', help='run one private inventory cycle; suitable for cron/CI')
    once.add_argument('--inventory', required=True)
    once.add_argument('--state', required=True)
    once.add_argument('--fetch', action='store_true')
    once.add_argument('--timeout', type=float, default=300)
    once.add_argument('--max-runs', type=int, default=DEFAULT_MAX_RUNS)
    once.add_argument('--max-cache-entries', type=int, default=DEFAULT_MAX_CACHE)
    once.add_argument('--max-data-mb', type=int, default=DEFAULT_MAX_DATA_BYTES // 1024**2)
    args = parser.parse_args(argv)
    if args._worker:
        _worker(_read(args._worker[0]), args._worker[1])
        return 0
    if args.command != 'scan-once':
        parser.error('select scan-once')
    try:
        report = scan_once(args.inventory, args.state, fetch=args.fetch, timeout_seconds=args.timeout,
                           max_runs=args.max_runs, max_cache_entries=args.max_cache_entries,
                           max_data_bytes=args.max_data_mb * 1024**2)
    except MonitorBusy as error:
        print(str(error), file=sys.stderr)
        return 75
    except (InputError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    if report.get('configuration_error'):
        return 2
    return {'pass': 0, 'warning': 0, 'error': 1, 'unknown': 3}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
