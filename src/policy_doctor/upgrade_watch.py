"""Explicit owned-asset/upstream scans with bounded, durable local alerts.

An upstream version is advisory. Only a configured, pinned native comparison
bound to current owned files can establish a retained-criterion impact result.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .artifacts import InputError
from .conversion_fidelity import _read_small, _reference_json
from .upgrade_impact import (
    ISAACLAB_SOURCES,
    QUALIFIED_CONSUMER,
    QUALIFIED_REVISION,
    assess_upgrade_impact,
)

MAX_FILE = 4 * 1024**2
MAX_RESPONSE = 256 * 1024
MAX_HISTORY = 8
TIMEOUT = 5
SCOPE = ('Owned-file/version declarations and freshly observed public upstream metadata. '
         'Affected/unaffected refer only to a bound retained native criterion. Caller-adapted '
         'snapshots, including the IsaacLab end-effector ordering intent, do not establish '
         'an intrinsic upstream bug, causal attribution, physical benefit or a safe migration.')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _encode(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()
    if len(raw) > MAX_FILE:
        raise InputError('Watch document exceeds 4 MiB; narrow the inventory/watch cohort')
    return raw


def _object(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise InputError('Unsupported watch document keys: expected ' + ', '.join(sorted(required)))


def _text(value):
    if type(value) is not str or not value or len(value) > 512 or not value.isprintable():
        raise InputError('Use bounded nonempty printable text')
    return value


def _pin(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise InputError('Use an independently retained lowercase SHA256 pin')
    return value


def _path(value, base):
    value = Path(_text(value))
    return value.absolute() if value.is_absolute() else (base / value).absolute()


def _read(path, tracked, expected=None):
    raw, _ = _read_small(str(path), limit=MAX_FILE)
    digest = _digest(raw)
    if expected is not None and digest != _pin(expected):
        raise InputError('File differs from the retained pin: ' + str(path))
    if str(path) in tracked and tracked[str(path)] != digest:
        raise InputError('Input changed during scan: ' + str(path))
    tracked[str(path)] = digest
    return raw


def support_catalog():
    """Explicit qualified source coverage and retained rules; no migration table."""
    revisions = list(ISAACLAB_SOURCES)
    return {
        'isaaclab-hdf5-xyzw': {
            'repository': 'isaac-sim/IsaacLab', 'revisions': revisions,
            'version_aliases': {'2.3.2': revisions[0], 'v2.3.2': revisions[0],
                                '3.0.0': revisions[1], 'v3.0.0': revisions[1]},
            'rule': 'Exact caller-retained named fields, dtype and values under declared per-field intent',
            'caveat': 'Raw ee_pose/orientation may remain identical WXYZ; all-XYZW adapted intent is a separate criterion',
        },
        QUALIFIED_CONSUMER: {
            'repository': 'NVIDIA/Isaac-GR00T', 'revisions': [QUALIFIED_REVISION], 'version_aliases': {},
            'rule': 'Exact retained native action targets/masks for covered source identities',
            'caveat': 'Same-revision configuration differences are not upstream upgrade evidence',
        },
    }


def _pointer(value, selector):
    if not selector.startswith('/'):
        raise InputError('JSON/TOML version selector must be an explicit JSON pointer')
    try:
        for token in selector.split('/')[1:]:
            token = token.replace('~1', '/').replace('~0', '~')
            value = value[int(token)] if type(value) is list else value[token]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise InputError('Version selector does not exist') from error
    return value


def _inventory(asset, base, tracked):
    _object(asset, {'asset_id', 'consumer', 'watch_id', 'files', 'version'}, {'native_comparison'})
    name, consumer, watch = [_text(asset[key]) for key in ('asset_id', 'consumer', 'watch_id')]
    if type(asset['files']) is not dict or not 1 <= len(asset['files']) <= 16:
        raise InputError('Each asset requires one to sixteen explicitly owned files')
    files = {}
    for logical, value in asset['files'].items():
        path = _path(value, base)
        raw = _read(path, tracked)
        files[_text(logical)] = {'path': str(path), 'sha256': _digest(raw), 'size_bytes': len(raw)}
    declaration = asset['version']
    _object(declaration, {'path', 'format', 'selector'})
    path = _path(declaration['path'], base)
    raw = _read(path, tracked)
    selector = _text(declaration['selector'])
    try:
        if declaration['format'] == 'json':
            version = _pointer(_reference_json(raw), selector)
        elif declaration['format'] == 'toml':
            version = _pointer(tomllib.loads(raw.decode('utf-8')), selector)
        elif declaration['format'] == 'requirements':
            package = selector.lower().replace('_', '-')
            matches = [match.group(2) for line in raw.decode('utf-8').splitlines()
                       if (match := re.fullmatch(r'\s*([A-Za-z0-9_.-]+)==([^\s;#]+)\s*(?:#.*)?', line))
                       and match.group(1).lower().replace('_', '-') == package]
            version = matches[0] if len(matches) == 1 else None
        else:
            raise InputError('Supported version sources are json, toml and exact requirements pins')
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise InputError('Cannot parse the declared version configuration') from error
    if version is not None:
        version = _text(version)
    return {'asset_id': name, 'consumer': consumer, 'watch_id': watch, 'files': files,
            'declared_version': version,
            'version_provenance': {'path': str(path), 'sha256': _digest(raw),
                                   'format': declaration['format'], 'selector': selector,
                                   'scope': 'Saved declaration; loaded/runtime identity is not inferred'}}


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != 'https' or target.hostname != 'api.github.com' or target.port not in (None, 443):
            raise InputError('Upstream redirect outside the explicit public GitHub API is refused')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _request_json(url):
    request = urllib.request.Request(url, headers={
        'Accept': 'application/vnd.github+json', 'User-Agent': 'RoboClue-local-upgrade-watch',
        'X-GitHub-Api-Version': '2022-11-28', 'Cache-Control': 'no-cache',
    })
    # No ambient credentials, conditional-cache responses or custom hosts.
    opener = urllib.request.build_opener(_PublicRedirect())
    with opener.open(request, timeout=TIMEOUT) as response:
        raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise InputError('Public upstream response exceeds the 256 KiB bound')
        value = _reference_json(raw)
        if type(value) is not dict:
            raise InputError('Public upstream metadata must be a JSON object')
        return value, {'url': url, 'body_sha256': _digest(raw), 'size_bytes': len(raw),
                       'body_utf8': raw.decode('utf-8'), 'etag': response.headers.get('ETag')}


def _observe(watch, offline):
    _object(watch, {'watch_id', 'repository', 'kind', 'ref'})
    identifier = _text(watch['watch_id'])
    repository = _text(watch['repository'])
    ref = _text(watch['ref'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or watch['kind'] not in ('commit', 'release'):
        raise InputError('Explicit public GitHub repository and commit/release watch are required')
    result = {'watch_id': identifier, 'repository': repository, 'kind': watch['kind'], 'requested_ref': ref,
              'status': 'unknown', 'code': 'offline_upstream_unobserved', 'receipts': [],
              'previous_observations_reused': False}
    if offline:
        return result
    prefix = 'https://api.github.com/repos/' + repository
    try:
        if watch['kind'] == 'release':
            suffix = '/releases/latest' if ref == 'latest' else '/releases/tags/' + urllib.parse.quote(ref, safe='')
            release, receipt = _request_json(prefix + suffix)
            result['receipts'].append(receipt)
            ref = _text(release.get('tag_name'))
            result.update(release_tag=ref, release_url=release.get('html_url'),
                          release_published_at=release.get('published_at'))
        suffix = ('/git/commits/' + ref if re.fullmatch('[0-9a-f]{40}', ref) else
                  '/commits/' + urllib.parse.quote(ref, safe='') + '?per_page=1')
        commit, receipt = _request_json(prefix + suffix)
        result['receipts'].append(receipt)
        revision = commit.get('sha')
        if type(revision) is not str or re.fullmatch('[0-9a-f]{40}', revision) is None:
            raise InputError('Upstream response did not identify a complete immutable commit')
        if re.fullmatch('[0-9a-f]{40}', ref) and revision != ref:
            raise InputError('Upstream response differs from the explicitly requested immutable commit')
        result.update(status='observed', code='public_upstream_observed', revision=revision,
                      observed_at=dt.datetime.now(dt.timezone.utc).isoformat(), source_url=commit.get('html_url'))
    except (InputError, OSError, ValueError, urllib.error.URLError) as error:
        result.update(code='upstream_fetch_failed', message=str(error)[:512])
    return result


def _native(asset, inventory, observation, base, tracked):
    binding = asset.get('native_comparison')
    if binding is None:
        return {'status': 'unknown', 'code': 'native_comparison_not_configured'}
    _object(binding, {'manifest', 'manifest_sha256', 'asset_id', 'criterion_sha256'})
    try:
        path = _path(binding['manifest'], base)
        manifest_raw = _read(path, tracked, binding['manifest_sha256'])
        native_document = _reference_json(manifest_raw)
        report = assess_upgrade_impact(path)
        if report['criterion_sha256'] != _pin(binding['criterion_sha256']):
            raise InputError('Native comparison changed the independently retained criterion')
        candidates = [row for row in report['assets'] if row['asset_id'] == binding['asset_id']]
        if len(candidates) != 1:
            raise InputError('Native comparison must select exactly one assessed asset')
        row = candidates[0]
        contract = native_document['contract']
        _read(_path(contract['path'], path.parent), tracked, binding['criterion_sha256'])
        for side in ('reference', 'candidate'):
            for file in [row[side]['capture'], *row[side]['source_files'].values(), *row[side]['configs'].values()]:
                _read(Path(file['path']), tracked, file['sha256'])
        if (row['candidate']['consumer'] != inventory['consumer']
                or row['candidate']['revision'] != observation['revision']):
            return {'status': 'unknown', 'code': 'native_comparison_scope_mismatch'}
        # Every current owned file must belong to one complete snapshot, not a
        # mixture of unrelated before/after files. Paths AND bytes must match.
        matching = [side for side in ('reference', 'candidate') if all(
            any(file['path'] == value['path'] and file['sha256'] == value['sha256']
                for value in row[side]['configs'].values()) for file in inventory['files'].values())]
        if not matching:
            return {'status': 'unknown', 'code': 'stale_native_comparison'}
        return {'status': {'changed': 'affected', 'healthy': 'unaffected', 'unknown': 'unknown'}[row['status']],
                'code': row['code'], 'owned_files_bound_to_snapshots': matching,
                'criterion_sha256': report['criterion_sha256'], 'criterion_intent': row['criterion_intent'],
                'comparison_basis': row['comparison_basis'], 'native_evidence': row}
    except InputError as error:
        return {'status': 'unknown', 'code': 'stale_or_invalid_native_evidence', 'message': str(error)}


def _remedy(code, status):
    if status == 'affected':
        return ('Review the retained field intent and any reference adapter, then supply a corrected candidate '
                'through the ordinary native pipeline and replay the SAME criterion. No automatic migration is implied.')
    if status == 'unaffected':
        return 'Keep the retained criterion; rescan after owned inputs or the explicitly watched upstream revision changes.'
    if code in ('offline_upstream_unobserved', 'upstream_fetch_failed'):
        return 'Run a fresh online scan when the explicit public upstream can be reached; prior receipts are historical only.'
    if code.startswith(('stale', 'native_comparison_scope')):
        return ('Capture current owned inputs at the ordinary native boundary and review a new comparison manifest '
                'while retaining the original criterion pin; do not clear the case by rebuilding intent.')
    return ('Review the explicit support catalog and supply covered native evidence for this asset/revision. '
            'A declared version or a new release alone establishes no native breakage.')


def _evaluate(asset, inventory, observation, base, tracked, catalog):
    consumer = inventory['consumer']
    result = {'asset_id': inventory['asset_id'], 'status': 'unknown', 'inventory': inventory,
              'upstream_watch_id': inventory['watch_id'], 'code': 'unsupported_consumer',
              'version_advisory_only': True}
    support = catalog.get(consumer)
    if support is not None:
        result['support_rule'] = support
        version = inventory['declared_version']
        revision = support['version_aliases'].get(version, version)
        result['declared_revision'] = revision
        if support['repository'].lower() != observation['repository'].lower():
            result['code'] = 'upstream_repository_not_qualified_for_consumer'
        elif observation['status'] != 'observed':
            result['code'] = observation['code']
        elif revision not in support['revisions']:
            result['code'] = 'unsupported_declared_version'
        elif observation['revision'] not in support['revisions']:
            result.update(code='upstream_revision_unqualified', observed_revision=observation['revision'])
        else:
            result.update(declared_revision_differs_from_observed=revision != observation['revision'],
                          observed_revision=observation['revision'])
            native = _native(asset, inventory, observation, base, tracked)
            result.update(native)
            if native['status'] in ('affected', 'unaffected'):
                result['version_advisory_only'] = False
    result['remedy'] = _remedy(result['code'], result['status'])
    return result


def _atomic(path, raw):
    fd, temporary = tempfile.mkstemp(prefix='.scan-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _state_root(path):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise InputError('State directory must not have symlink ancestors')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def scan_once(config_path, state_dir, *, offline=False):
    """Inventory, observe and publish one durable bounded local scan.

    No scheduler, daemon, auto-repair, outgoing message or prior-result reuse is
    installed. Invoke this operation explicitly from your own CI/scheduler.
    """
    tracked = {}
    path = Path(config_path).absolute()
    config_raw = _read(path, tracked)
    config = _reference_json(config_raw)
    _object(config, {'schema_version', 'kind', 'watches', 'assets'})
    if (type(config['schema_version']) is not int or config['schema_version'] != 1
            or config['kind'] != 'owned_upgrade_watch'
            or type(config['watches']) is not list or not 1 <= len(config['watches']) <= 4
            or type(config['assets']) is not list or not 1 <= len(config['assets']) <= 32):
        raise InputError('Watch configuration requires one to four watches and one to32 owned assets')
    root = _state_root(state_dir)
    lock_fd = os.open(root / 'scan.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InputError('Another scan owns this state directory; retry after it finishes') from error
        state_path = root / 'state.json'
        state = {'schema_version': 1, 'kind': 'upgrade_watch_state', 'history': []}
        if state_path.exists() or state_path.is_symlink():
            raw, _ = _read_small(str(state_path), limit=MAX_FILE)
            state = _reference_json(raw)
            _object(state, {'schema_version', 'kind', 'history'})
            if state['schema_version'] != 1 or state['kind'] != 'upgrade_watch_state' or type(state['history']) is not list:
                raise InputError('Existing state is not a supported upgrade-watch history')
            if len(state['history']) > MAX_HISTORY:
                raise InputError('Existing watch history exceeds its retained bound')
            for entry in state['history']:
                _object(entry, {'scan_id', 'report_sha256'})
                if not re.fullmatch('[0-9a-f]{32}', entry['scan_id']):
                    raise InputError('Existing watch scan identifier is invalid')
                raw, _ = _read_small(str(root / ('scan-' + entry['scan_id'] + '.json')), limit=MAX_FILE)
                if _digest(raw) != _pin(entry['report_sha256']):
                    raise InputError('Retained watch report changed; preserve it and use a new state directory')
        observations = {}
        for watch in config['watches']:
            observed = _observe(watch, offline)
            if observed['watch_id'] in observations:
                raise InputError('Watch identifiers must be unique')
            observations[observed['watch_id']] = observed
        catalog, assets, names = support_catalog(), [], set()
        for asset in config['assets']:
            _object(asset, {'asset_id', 'consumer', 'watch_id', 'files', 'version'}, {'native_comparison'})
            name = _text(asset['asset_id'])
            if name in names or asset['watch_id'] not in observations:
                raise InputError('Asset identifiers must be unique and select a configured watch')
            names.add(name)
            try:
                inventory = _inventory(asset, path.parent, tracked)
                result = _evaluate(asset, inventory, observations[asset['watch_id']], path.parent, tracked, catalog)
            except InputError as error:
                result = {'asset_id': name, 'status': 'unknown', 'code': 'inventory_not_established',
                          'message': str(error), 'remedy': 'Supply accessible owned files and an explicit parseable version declaration.'}
            assets.append(result)
        for filename, expected in tracked.items():
            raw, _ = _read_small(filename, limit=MAX_FILE)
            if _digest(raw) != expected:
                raise InputError('Owned/config/evidence input changed during scan; rerun on stable inputs')
        counts = {status: sum(row['status'] == status for row in assets) for status in ('affected', 'unaffected', 'unknown')}
        status = 'affected' if counts['affected'] else 'unknown' if counts['unknown'] else 'unaffected'
        scan_id = uuid.uuid4().hex
        report = {'schema_version': 1, 'kind': 'upgrade_watch_scan', 'scan_id': scan_id,
                  'scanned_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': status,
                  'config_sha256': _digest(config_raw), 'offline': offline,
                  'previous_scan_id': state['history'][-1]['scan_id'] if state['history'] else None,
                  'previous_results_reused': False, 'observations': list(observations.values()),
                  'catalog': catalog, 'counts': counts, 'assets': assets, 'input_postguards': 'pass',
                  'network_bounds': {'max_watches': 4, 'max_requests_per_watch': 2,
                                     'socket_timeout_seconds': TIMEOUT, 'response_bytes': MAX_RESPONSE}, 'scope': SCOPE}
        raw = _encode(report)
        lines = ['# Owned upgrade watch', '', 'Status: ' + status.upper(),
                 'Counts: ' + json.dumps(counts, sort_keys=True), '', SCOPE, '']
        for observation in observations.values():
            lines.append('Upstream: ' + json.dumps({key: observation.get(key) for key in
                         ('watch_id', 'repository', 'requested_ref', 'status', 'revision', 'observed_at')}, sort_keys=True))
        lines.append('')
        for row in assets:
            lines.extend(['- ' + json.dumps(row['asset_id']) + ': ' + row['status'].upper() + ' (' + row['code'] + ')',
                          '  Remedy: ' + row['remedy']])
            if 'criterion_intent' in row:
                lines.extend(['  Criterion intent: ' + json.dumps(row['criterion_intent'], sort_keys=True),
                              '  Comparison basis: ' + json.dumps(row['comparison_basis'], sort_keys=True)])
                for change in row['native_evidence'].get('native_boundary_delta', {}).get('changes', [])[:2]:
                    lines.append('  Intended-field evidence: ' + json.dumps(change, sort_keys=True))
        filename = root / ('scan-' + scan_id + '.json')
        _atomic(filename, raw)
        _atomic(root / ('scan-' + scan_id + '.md'), '\n'.join(lines).encode())
        history = state['history'] + [{'scan_id': scan_id, 'report_sha256': _digest(raw)}]
        discarded, state['history'] = history[:-MAX_HISTORY], history[-MAX_HISTORY:]
        _atomic(state_path, _encode(state))
        for entry in discarded:
            for suffix in ('.json', '.md'):
                (root / ('scan-' + entry['scan_id'] + suffix)).unlink(missing_ok=True)
        return {'report': report, 'report_path': str(filename),
                'alerts_path': str(root / ('scan-' + scan_id + '.md')),
                'exit_code': {'unaffected': 0, 'affected': 1, 'unknown': 3}[status]}
    finally:
        os.close(lock_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inventory owned assets and scan explicit public upstream watches once.')
    commands = parser.add_subparsers(dest='command', required=True)
    scan = commands.add_parser('scan-once')
    scan.add_argument('--config', required=True)
    scan.add_argument('--state-dir', required=True)
    scan.add_argument('--offline', action='store_true')
    scan.add_argument('--json', action='store_true')
    commands.add_parser('catalog')
    args = parser.parse_args(argv)
    if args.command == 'catalog':
        print(json.dumps(support_catalog(), sort_keys=True, indent=2))
        return 0
    try:
        result = scan_once(args.config, args.state_dir, offline=args.offline)
        report = result['report']
        summary = {'status': report['status'], 'counts': report['counts'], 'scan_id': report['scan_id'],
                   'report': result['report_path'], 'alerts': result['alerts_path'], 'offline': report['offline']}
        print(json.dumps(summary, sort_keys=True) if args.json else
              report['status'].upper() + ': local alerts ' + result['alerts_path'])
        return result['exit_code']
    except (InputError, OSError) as error:
        print(json.dumps({'status': 'input_error', 'message': str(error)}, sort_keys=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
