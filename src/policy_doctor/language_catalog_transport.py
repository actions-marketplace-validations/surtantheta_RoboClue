"""Anonymous, identity-bound retrieval for two task-catalog paths only.

The private worker bounds DNS as well as HTTP work. Shared Hub transports and
their permissions are not changed. CDN ETags are not LFS content identities.
"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import re
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

MAX_BYTES = 8 * 1024 * 1024
TIMEOUT = 35
_PATHS = {'meta/tasks.parquet', 'meta/tasks.jsonl'}
_CDN = 'us.aws.cdn.hf.co'
_PART = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
_SHA = re.compile(r'[0-9a-f]{40}')
_DIGEST = re.compile(r'[0-9a-f]{64}')
_CDN_PATH = re.compile(r'/xet-bridge-us/[0-9a-f]{24}/[0-9a-f]{64}')


class CatalogTransportError(ValueError):
    """A structured UNKNOWN at the public instruction/catalog boundary."""


def _validate(repo, revision, filename, size, digest):
    if (not isinstance(repo, str) or len(repo.split('/')) != 2
            or any(not _PART.fullmatch(p) or '..' in p for p in repo.split('/'))
            or not isinstance(revision, str) or not _SHA.fullmatch(revision)
            or not isinstance(filename, str) or filename not in _PATHS
            or type(size) is not int or not 0 < size <= MAX_BYTES
            or not isinstance(digest, str) or not _DIGEST.fullmatch(digest)):
        raise CatalogTransportError('Invalid pinned task-catalog identity or byte bound')


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CatalogTransportError('Task-catalog transport exceeded its time bound')
    return min(10, remaining)


def _public(address):
    ip = ipaddress.ip_address(address)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return False
    return (ip.is_global and not ip.is_multicast and not ip.is_reserved
            and not ip.is_loopback and not ip.is_link_local and not ip.is_unspecified)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, deadline):
        if host not in ('huggingface.co', _CDN):
            raise CatalogTransportError('Task-catalog host is outside the exact allowlist')
        super().__init__(host, port=443, timeout=_remaining(deadline), context=ssl.create_default_context())
        self.deadline = deadline

    def connect(self):
        answers = socket.getaddrinfo(self.host, 443, type=socket.SOCK_STREAM)
        if not answers or len(answers) > 64:
            raise CatalogTransportError('Task-catalog DNS answer count exceeds its bound')
        for family, kind, proto, _, target in answers:
            if (family not in (socket.AF_INET, socket.AF_INET6) or kind != socket.SOCK_STREAM or proto != socket.IPPROTO_TCP
                    or target[1] != 443 or not _public(target[0])):
                raise CatalogTransportError('Task-catalog DNS resolved a non-public address')
        # Pin a vetted literal address: HTTP/TLS must not perform a second lookup.
        family, kind, proto, _, target = next((a for a in answers if a[0] == socket.AF_INET), answers[0])
        raw = socket.socket(family, kind, proto)
        try:
            raw.settimeout(_remaining(self.deadline))
            raw.connect(target)
            if ipaddress.ip_address(raw.getpeername()[0]) != ipaddress.ip_address(target[0]):
                raise CatalogTransportError('Task-catalog peer differs from its vetted public address')
            raw.settimeout(_remaining(self.deadline))
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def _cdn_location(value):
    if not isinstance(value, str) or len(value) > 8192 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise CatalogTransportError('Malformed task-catalog redirect')
    try:
        parts = urlsplit(value)
        valid = (parts.scheme == 'https' and parts.hostname == _CDN and parts.port in (None, 443)
                 and parts.username is None and parts.password is None and not parts.fragment
                 and _CDN_PATH.fullmatch(parts.path))
    except ValueError:
        valid = False
    if not valid:
        raise CatalogTransportError('Task-catalog redirect left the exact HTTPS CDN route')
    return parts


def _identity_headers(headers, revision, size, digest, *, redirect):
    expected = {'X-Repo-Commit': revision, 'X-Linked-Size': str(size), 'X-Linked-ETag': digest}
    for name, value in expected.items():
        actual = headers.get(name)
        if name == 'X-Linked-ETag' and actual == '"' + digest + '"':
            actual = digest
        if (redirect or actual is not None) and actual != value:
            raise CatalogTransportError('Task-catalog response changed its pinned revision, size or content identity')


def _fetch(repo, revision, filename, size, digest):
    _validate(repo, revision, filename, size, digest)
    deadline = time.monotonic() + 30
    host = 'huggingface.co'
    target = f'/datasets/{repo}/resolve/{revision}/{filename}'
    for hop in range(2):
        connection = _PinnedHTTPS(host, deadline)
        response = None
        try:
            connection.request('GET', target, headers={'Accept-Encoding': 'identity',
                'User-Agent': 'RoboClue-task-catalog/1', 'Connection': 'close'})
            connection.sock.settimeout(_remaining(deadline))
            response = connection.getresponse()
            _identity_headers(response.headers, revision, size, digest,
                              redirect=response.status == 302 and hop == 0)
            if response.status == 302 and hop == 0:
                redirected = _cdn_location(response.headers.get('Location'))
                host = redirected.hostname
                target = redirected.path + ('?' + redirected.query if redirected.query else '')
                continue
            if response.status != 200 or response.headers.get('Content-Range') is not None:
                raise CatalogTransportError('Task catalog requires a complete HTTP200 and at most one302 CDN hop')
            encoding = response.headers.get('Content-Encoding', 'identity')
            if encoding not in ('identity', ''):
                raise CatalogTransportError('Compressed task-catalog responses are refused')
            length = response.headers.get('Content-Length')
            transfer = response.headers.get('Transfer-Encoding')
            if length is not None and length != str(size):
                raise CatalogTransportError('Task-catalog HTTP length differs from its pinned size')
            if transfer is not None and (transfer.lower() != 'chunked' or length is not None):
                raise CatalogTransportError('Ambiguous task-catalog transfer framing')
            chunks, total, actual_digest = [], 0, hashlib.sha256()
            while True:
                if response.isclosed():
                    break
                # HTTPConnection detaches its socket for Connection: close;
                # the response's buffered reader still owns it until EOF.
                reader_socket = connection.sock or response.fp.raw._sock
                reader_socket.settimeout(_remaining(deadline))
                chunk = response.read1(min(65536, size + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise CatalogTransportError('Task-catalog response exceeds its pinned byte bound')
                actual_digest.update(chunk)
                chunks.append(chunk)
            if total != size or actual_digest.hexdigest() != digest:
                raise CatalogTransportError('Task-catalog bytes differ from the pinned LFS size or SHA256')
            return b''.join(chunks)
        finally:
            if response is not None:
                response.close()
            connection.close()
    raise CatalogTransportError('Task-catalog redirect count exceeds its bound')


def fetch_catalog(repo, revision, filename, *, size, sha256):
    """Fetch a validated API-listed LFS catalog; never accepts a free-form URL."""
    _validate(repo, revision, filename, size, sha256)
    request = json.dumps({'repo': repo, 'revision': revision, 'filename': filename,
                          'size': size, 'sha256': sha256}, allow_nan=False).encode()
    try:
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
            input=request, capture_output=True, timeout=TIMEOUT, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        raise CatalogTransportError('Task-catalog worker unavailable or exceeded its35s bound') from exc
    if result.returncode != 0:
        message = result.stderr[:512].decode('utf-8', errors='replace')
        raise CatalogTransportError('Task-catalog retrieval UNKNOWN: ' + message)
    if len(result.stdout) != size or hashlib.sha256(result.stdout).hexdigest() != sha256:
        raise CatalogTransportError('Task-catalog worker returned a wrong size or SHA256')
    return result.stdout


def _worker_main():
    try:
        raw = sys.stdin.buffer.read(4097)
        if len(raw) > 4096:
            raise CatalogTransportError('Task-catalog worker request exceeds its bound')
        args = json.loads(raw)
        if not isinstance(args, dict) or set(args) != {'repo', 'revision', 'filename', 'size', 'sha256'}:
            raise CatalogTransportError('Unsupported task-catalog worker request')
        data = _fetch(args['repo'], args['revision'], args['filename'], args['size'], args['sha256'])
        sys.stdout.buffer.write(data)
        return 0
    except Exception as exc:
        message = ''.join(c if c.isprintable() else ' ' for c in str(exc))[:360]
        sys.stderr.write(type(exc).__name__ + ': ' + message)
        return 2


if __name__ == '__main__':
    raise SystemExit(_worker_main() if sys.argv[1:] == ['--worker'] else 2)
