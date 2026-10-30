"""Observe installed distribution metadata without importing a robotics framework.

This process's metadata is context, not the recorder/trainer environment, code
attestation or renewed native qualification. Versions alone never imply support.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
import re
import stat
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlsplit

from .artifacts import InputError
from .rule_catalog import rule_catalog

MAX_METADATA_BYTES = 64 * 1024
MAX_DIRECT_URL_BYTES = 8 * 1024
ENVIRONMENT = 'RoboClue Python process; child recorder/trainer environment may differ'
VERSION_POLICY = ('Exact commit metadata can name a tested source revision; it does not authenticate installed bytes '
                  'or renew native execution qualification. A release version alone never establishes support.')
_REGISTRY = (
    ('lerobot', 'huggingface/lerobot', ('lerobot',), ('LeRobot', 'LeRobot metadata')),
    ('isaaclab', 'isaac-sim/IsaacLab', ('isaaclab',), ('Isaac Lab',)),
    ('groot', 'NVIDIA/Isaac-GR00T', ('gr00t',), ('Isaac-GR00T',)),
    ('openpi', 'Physical-Intelligence/openpi', ('openpi',), ('openpi',)),
)


def supported_framework_profiles() -> dict:
    """Expose primary repositories and catalog pins for offline/CI revalidation planning."""
    rules = rule_catalog()['rules']
    frameworks = []
    for name, repository, distributions, labels in _REGISTRY:
        matching = [(rule, profile) for rule in rules for profile in rule['frameworks'] if profile['name'] in labels]
        frameworks.append({'id': name, 'repository': repository, 'distribution_names': list(distributions),
                           'tested_commits': sorted({profile['revision'] for _, profile in matching}),
                           'rule_ids': sorted({rule['id'] for rule, _ in matching}),
                           'version_policy': VERSION_POLICY})
    return {'schema_version': 1, 'kind': 'framework_profiles', 'frameworks': frameworks}


def _regular_bytes(path: Path, limit: int) -> bytes | None:
    """Read bounded regular metadata only; FIFO/device/link input must not block."""
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(observed.st_mode) or observed.st_size > limit:
        raise ValueError('Metadata is not a bounded regular file.')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or current.st_size > limit or (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError('Metadata changed identity or is not a bounded regular file.')
        chunks, remaining = [], limit + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b''.join(chunks)
        final = os.fstat(fd)
        if len(raw) > limit or (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (final.st_size, final.st_mtime_ns, final.st_ctime_ns):
            raise ValueError('Metadata exceeded its bound or changed during the read.')
        return raw
    finally:
        os.close(fd)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Repeated direct_url JSON keys are ambiguous.')
        result[key] = value
    return result


def _normal_name(value):
    return re.sub('[-_.]+', '-', value).lower()


def _inspect_distribution(distribution, expected_names) -> dict:
    result = {'distribution': None, 'version': None, 'declared_commit': None,
              'source_identity': 'unknown', 'reason': 'Installed metadata could not be assessed.'}
    # The stdlib permits arbitrary metadata providers and ZIP archives. This
    # bounded reader supports filesystem PathDistribution directories only.
    if not isinstance(distribution, metadata.PathDistribution):
        result['reason'] = 'Non-filesystem distribution metadata is not modeled.'
        return result
    path = getattr(distribution, '_path', None)
    if not isinstance(path, Path):
        result['reason'] = 'ZIP or non-path distribution metadata is not modeled.'
        return result
    try:
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError('Distribution metadata directory is linked or unsupported.')
        raw = _regular_bytes(path / 'METADATA', MAX_METADATA_BYTES)
        if raw is None:
            raw = _regular_bytes(path / 'PKG-INFO', MAX_METADATA_BYTES)
        if raw is None:
            raise ValueError('Distribution metadata is missing.')
        message = BytesParser().parsebytes(raw, headersonly=True)
        names, versions = message.get_all('Name', []), message.get_all('Version', [])
        if len(names) != 1 or len(versions) != 1:
            raise ValueError('Distribution name/version fields are missing or repeated.')
        if not isinstance(names[0], str) or not isinstance(versions[0], str):
            raise ValueError('Distribution identity fields are not supported text.')
        name, version = names[0].strip(), versions[0].strip()
        if _normal_name(name) not in {_normal_name(n) for n in expected_names}:
            raise ValueError('Distribution name does not match its requested framework.')
        if not version or len(version) > 200 or any(ord(c) < 32 or ord(c) == 127 for c in version):
            raise ValueError('Distribution version is missing, oversized or non-printable.')
        result.update(distribution=name, version=version, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                      source_identity='not_revalidated', reason='Version observed; no exact VCS source revision is attested by this metadata.')
        direct = _regular_bytes(path / 'direct_url.json', MAX_DIRECT_URL_BYTES)
        if direct is None:
            return result
        document = json.loads(direct, object_pairs_hook=_unique_object)
        if not isinstance(document, dict):
            raise ValueError('direct_url metadata is not an object.')
        url = document.get('url')
        if not isinstance(url, str) or not url or len(url) > 4096 or not urlsplit(url).scheme:
            raise ValueError('direct_url metadata lacks a supported URL declaration.')
        variants = [name for name in ('vcs_info', 'dir_info', 'archive_info') if name in document]
        if len(variants) != 1 or not isinstance(document[variants[0]], dict):
            raise ValueError('direct_url source identity is ambiguous or unsupported.')
        vcs = document.get('vcs_info')
        if vcs is None:
            result['reason'] = 'Local/editable or archive metadata does not declare an exact VCS commit; checkout code was not inspected.'
            return result
        if not isinstance(vcs, dict) or vcs.get('vcs') != 'git' or not isinstance(vcs.get('commit_id'), str) or not re.fullmatch('[a-fA-F0-9]{40}', vcs['commit_id']):
            raise ValueError('VCS metadata lacks an unambiguous full Git commit.')
        result['declared_commit'] = vcs['commit_id'].lower()
        result['reason'] = 'Exact VCS commit observed in distribution metadata; installed code bytes were not attested.'
        return result
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        result['source_identity'] = 'unknown'
        result['reason'] = 'Distribution metadata is unreadable, malformed, linked, changed or outside supported size/format bounds.'
        return result


def collect_framework_versions() -> dict:
    """Read this Python environment's installed framework metadata; no imports/network.

    No path/URL supplied by a dataset or server caller is followed. Generated
    fixture tests substitute stdlib distribution discovery, not framework code.
    """
    frameworks = []
    for profile in supported_framework_profiles()['frameworks']:
        observed = []
        failed = False
        for name in profile['distribution_names']:
            try:
                observed.extend(itertools.islice(metadata.distributions(name=name), 3))
            except (OSError, ValueError, TypeError, RuntimeError):
                failed = True
        base = {'id': profile['id'], 'repository': profile['repository'], 'rule_ids': profile['rule_ids'],
                'tested_commits': profile['tested_commits'], 'distribution': None, 'version': None,
                'declared_commit': None, 'source_identity': 'not_installed',
                'reason': 'No matching distribution metadata was discovered in the RoboClue Python environment.'}
        if failed:
            base.update(source_identity='unknown', reason='Distribution metadata discovery failed; framework context is unassessed.')
        elif len(observed) > 1:
            base.update(source_identity='unknown', reason='Multiple matching distributions were discovered; active consumer identity is ambiguous.')
        elif observed:
            base.update(_inspect_distribution(observed[0], profile['distribution_names']))
            if base['source_identity'] == 'not_revalidated' and base['declared_commit'] in profile['tested_commits']:
                base['source_identity'] = 'declared_tested_commit'
                base['reason'] = 'Metadata declares an exact catalog-tested commit. This does not authenticate installed bytes or renew native execution qualification.'
        frameworks.append(base)
    return {'schema_version': 1, 'kind': 'installed_framework_context', 'environment': ENVIRONMENT,
            'runtime_revalidated': False, 'version_policy': VERSION_POLICY, 'frameworks': frameworks}


def compact_framework_context(report: dict, framework_id: str) -> dict:
    """Keep minimal identity context for a scoped MCP answer; no compatibility PASS."""
    if not isinstance(report, dict) or not isinstance(report.get('frameworks'), list):
        raise InputError('Framework context must come from collect_framework_versions().')
    matching = [row for row in report['frameworks'] if isinstance(row, dict) and row.get('id') == framework_id]
    if len(matching) != 1:
        raise InputError('Unknown or ambiguous framework context selector.')
    row = matching[0]
    result = {'framework': framework_id, 'version': row.get('version'),
              'source_identity': row.get('source_identity', 'unknown'),
              'environment': 'RoboClue Python; consumer environment unverified', 'runtime_revalidated': False}
    version = result['version']
    if isinstance(version, str) and len(json.dumps(version, ensure_ascii=True).encode()) > 160:
        result.update(version=None, version_omitted=True,
                      version_sha256=hashlib.sha256(version.encode('utf-8')).hexdigest(),
                      details='roboclue advanced versions --json')
    return result


def render_framework_versions(report: dict | None = None) -> str:
    """Render observed versions separately from declared-source identity."""
    result = collect_framework_versions() if report is None else report
    lines = ['RoboClue framework context', result['environment'], '']
    for row in result['frameworks']:
        lines.extend([f"{row['id']}: version={row['version'] or 'not established'}; source identity={row['source_identity']}",
                      '  ' + row['reason']])
    lines.extend(['', VERSION_POLICY])
    return '\n'.join(lines)
