"""Pinned catalog transport refuses unbound content and hosts before contact."""
import hashlib
import io
import json
import socket
import subprocess
import sys
import time
from email.message import Message
from types import SimpleNamespace

import pytest
from test_language_vocab import fixture

from policy_doctor import hub
from policy_doctor import language_catalog_transport as transport
from policy_doctor.language_vocab import _remote_files, check_language_vocabulary

SHA = 'a' * 40
BODY = b'catalog'
DIGEST = hashlib.sha256(BODY).hexdigest()
LOCATION = 'https://us.aws.cdn.hf.co/xet-bridge-us/' + 'b' * 24 + '/' + 'c' * 64 + '?Signature=public'


class Response:
    def __init__(self, data=BODY, *, status=200, headers=None):
        self.data, self.status = io.BytesIO(data), status
        self.headers, self.reads, self.closed = Message(), [], False
        for k, v in (headers or {}).items():
            self.headers[k] = v

    def read1(self, count):
        self.reads.append(count)
        result = self.data.read(count)
        if not result:
            self.closed = True
        return result

    def isclosed(self):
        return self.closed

    def close(self):
        self.closed = True


def redirect(**changes):
    headers = {'Location': LOCATION, 'X-Repo-Commit': SHA,
               'X-Linked-Size': str(len(BODY)), 'X-Linked-ETag': '"' + DIGEST + '"'}
    headers.update(changes)
    return Response(status=302, headers=headers)


def fake_connections(monkeypatch, responses, *, detached=False):
    calls = []

    class Connection:
        def __init__(self, host, deadline):
            assert len(calls) < len(responses), 'Unapproved extra request'
            self.response = responses[len(calls)]
            self.sock = SimpleNamespace(settimeout=lambda _: None)
            self.reader_socket = self.sock
            calls.append({'host': host})

        def request(self, method, target, headers):
            calls[-1].update(method=method, target=target, headers=headers)

        def getresponse(self):
            if detached:
                self.response.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.reader_socket))
                self.sock = None
            return self.response

        def close(self):
            pass

    monkeypatch.setattr(transport, '_PinnedHTTPS', Connection)
    return calls


@pytest.mark.parametrize('cdn,detached', [(False, False), (False, True), (True, False), (True, True)])
def test_verified_exact_catalog_uses_only_anonymous_fixed_hops(monkeypatch, cdn, detached):
    final = Response(headers={'Content-Length': str(len(BODY))})
    calls = fake_connections(monkeypatch, [redirect(), final] if cdn else [final], detached=detached)
    assert transport._fetch('org/data', SHA, 'meta/tasks.parquet', len(BODY), DIGEST) == BODY
    assert calls[0]['target'] == '/datasets/org/data/resolve/' + SHA + '/meta/tasks.parquet'
    assert [c['host'] for c in calls] == (['huggingface.co', 'us.aws.cdn.hf.co'] if cdn else ['huggingface.co'])
    assert all(c['method'] == 'GET' and set(c['headers']) == {'Accept-Encoding', 'User-Agent', 'Connection'} for c in calls)
    assert final.closed


@pytest.mark.parametrize('url', ['http://us.aws.cdn.hf.co/x', 'https://127.0.0.1/x',
    'https://169.254.169.254/x', 'https://[::1]/x', 'https://us.aws.cdn.hf.co.evil/x',
    'https://other.hf.co/x', LOCATION.replace('https://', 'https://user:pass@'),
    LOCATION.replace('.co/', '.co:444/'), LOCATION + '#fragment',
    LOCATION.replace('/xet-bridge-us/', '/other/'), LOCATION + '\r\nX-Evil: yes', 'x' * 8193,
    LOCATION.replace('us.aws.cdn.hf.co', 'other.hf.co'),
    LOCATION.replace('us.aws.cdn.hf.co', '127.0.0.1'), LOCATION.replace('https:', 'http:')])
def test_hostile_redirect_is_refused_before_contact(monkeypatch, url):
    calls = fake_connections(monkeypatch, [redirect(Location=url)])
    with pytest.raises(transport.CatalogTransportError, match='redirect'):
        transport._fetch('org/data', SHA, 'meta/tasks.parquet', len(BODY), DIGEST)
    assert len(calls) == 1


@pytest.mark.parametrize('change', [{'X-Repo-Commit': 'b' * 40}, {'X-Linked-Size': '8'},
    {'X-Linked-Size': None}, {'X-Linked-ETag': 'f' * 64}, {'X-Linked-ETag': None}])
def test_redirect_identity_must_agree_before_cdn_request(monkeypatch, change):
    response = redirect()
    for name, value in change.items():
        del response.headers[name]
        if value is not None:
            response.headers[name] = value
    calls = fake_connections(monkeypatch, [response])
    with pytest.raises(transport.CatalogTransportError, match='identity'):
        transport._fetch('org/data', SHA, 'meta/tasks.parquet', len(BODY), DIGEST)
    assert len(calls) == 1


def test_second_redirect_is_never_followed(monkeypatch):
    calls = fake_connections(monkeypatch, [redirect(), redirect()])
    with pytest.raises(transport.CatalogTransportError, match='at most one'):
        transport._fetch('org/data', SHA, 'meta/tasks.parquet', len(BODY), DIGEST)
    assert len(calls) == 2


@pytest.mark.parametrize('status,headers,data', [(206, {}, BODY), (404, {}, BODY),
    (307, {}, BODY), (200, {'Content-Encoding': 'gzip'}, BODY),
    (200, {'Content-Range': 'bytes 0-6/7'}, BODY), (200, {'Content-Length': '8'}, BODY),
    (200, {'Content-Length': '-1'}, BODY), (200, {'Content-Length': '7,7'}, BODY),
    (200, {'Transfer-Encoding': 'gzip'}, BODY),
    (200, {'Content-Length': '7', 'Transfer-Encoding': 'chunked'}, BODY),
    (200, {}, b'wrong!!'), (200, {}, BODY[:-1]), (200, {}, BODY + b'extra')])
def test_framing_hash_and_size_refusals_are_bounded(monkeypatch, status, headers, data):
    response = Response(data, status=status, headers=headers)
    fake_connections(monkeypatch, [response])
    with pytest.raises(transport.CatalogTransportError):
        transport._fetch('org/data', SHA, 'meta/tasks.parquet', len(BODY), DIGEST)
    assert sum(response.reads) <= 2 * (len(BODY) + 1)
    assert response.closed


@pytest.mark.parametrize('address', ['127.0.0.1', '10.0.0.1', '169.254.169.254', '192.168.1.1',
    '0.0.0.0', '224.0.0.1', '240.0.0.1', '::1', 'fe80::1', 'fd00::1', '::ffff:127.0.0.1'])
def test_private_or_mixed_dns_is_refused_before_socket(monkeypatch, address):
    family = socket.AF_INET6 if ':' in address else socket.AF_INET
    answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443)),
               (family, socket.SOCK_STREAM, 6, '', (address, 443))]
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: answers)
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('private-address socket opened'))
    connection = transport._PinnedHTTPS('huggingface.co', transport.time.monotonic() + 30)
    with pytest.raises(transport.CatalogTransportError, match='non-public'):
        connection.connect()


def test_vetted_address_is_tcp_pinned_with_original_hostname_tls(monkeypatch):
    calls = []
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))])
    raw = SimpleNamespace(settimeout=lambda _: None, connect=lambda target: calls.append(('connect', target)),
                          getpeername=lambda: ('8.8.8.8', 443), close=lambda: None)
    monkeypatch.setattr(socket, 'socket', lambda *a: raw)
    connection = transport._PinnedHTTPS('huggingface.co', transport.time.monotonic() + 30)
    assert connection._context.check_hostname and connection._context.verify_mode == transport.ssl.CERT_REQUIRED
    connection._context = SimpleNamespace(wrap_socket=lambda sock, server_hostname: calls.append(('TLS', server_hostname)) or raw)
    connection.connect()
    assert calls == [('connect', ('8.8.8.8', 443)), ('TLS', 'huggingface.co')]


def test_changed_peer_is_refused_before_tls(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))])
    raw = SimpleNamespace(settimeout=lambda _: None, connect=lambda _: None,
                          getpeername=lambda: ('127.0.0.1', 443), close=lambda: None)
    monkeypatch.setattr(socket, 'socket', lambda *a: raw)
    connection = transport._PinnedHTTPS('huggingface.co', transport.time.monotonic() + 30)
    connection._context = SimpleNamespace(wrap_socket=lambda *a, **k: pytest.fail('changed-peer TLS'))
    with pytest.raises(transport.CatalogTransportError, match='peer'):
        connection.connect()


@pytest.mark.parametrize('change', [{'repo': 'org/../bad'}, {'revision': 'main'},
    {'filename': 'model.safetensors'}, {'size': True}, {'size': 0}, {'size': -1},
    {'size': transport.MAX_BYTES + 1}, {'sha256': 'bad'}, {'sha256': None}])
def test_invalid_parent_identity_never_starts_worker(monkeypatch, change):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('worker before identity validation'))
    args = dict(repo='org/data', revision=SHA, filename='meta/tasks.parquet', size=len(BODY), sha256=DIGEST)
    args.update(change)
    with pytest.raises(transport.CatalogTransportError):
        transport.fetch_catalog(**args)


def test_parent_deadline_and_worker_output_identity(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == 35
        raise subprocess.TimeoutExpired(args[0], 35)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(transport.CatalogTransportError, match='35s'):
        transport.fetch_catalog('org/data', SHA, 'meta/tasks.parquet', size=len(BODY), sha256=DIGEST)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=b'wrong!!'))
    with pytest.raises(transport.CatalogTransportError, match='wrong size or SHA256'):
        transport.fetch_catalog('org/data', SHA, 'meta/tasks.parquet', size=len(BODY), sha256=DIGEST)


def test_actual_parent_timeout_kills_a_stalled_worker_without_network(tmp_path, monkeypatch):
    worker = tmp_path / 'stalled.py'
    worker.write_text('import sys,time\nsys.stdin.buffer.read()\ntime.sleep(20)\n')
    monkeypatch.setattr(transport, '__file__', str(worker))
    monkeypatch.setattr(transport, 'TIMEOUT', 0.2)
    start = time.monotonic()
    with pytest.raises(transport.CatalogTransportError, match='35s'):
        transport.fetch_catalog('org/data', SHA, 'meta/tasks.parquet', size=len(BODY), sha256=DIGEST)
    assert time.monotonic() - start < 5


@pytest.mark.parametrize('payload', [b'not JSON', b'{}', b'x' * 4097,
    json.dumps({'repo': 'org/data', 'revision': SHA, 'filename': 'model.safetensors',
                'size': len(BODY), 'sha256': DIGEST}).encode()])
def test_real_worker_malformed_protocol_is_controlled_without_traceback(payload):
    result = subprocess.run([sys.executable, '-B', transport.__file__, '--worker'],
        input=payload, capture_output=True, timeout=5)
    assert result.returncode == 2 and result.stdout == b''
    assert len(result.stderr) <= 512 and b'Traceback' not in result.stderr


@pytest.mark.parametrize('change', [{'size': True}, {'size': 0}, {'size': 2**40}, {'blobId': None},
    {'lfs': {}}, {'lfs': {'sha256': DIGEST, 'size': True}},
    {'lfs': {'sha256': DIGEST, 'size': len(BODY) + 1}}, {'lfs': {'sha256': 'f' * 63, 'size': len(BODY)}}])
def test_api_identity_refusal_precedes_transport(monkeypatch, change):
    entry = {'rfilename': 'meta/tasks.parquet', 'size': len(BODY), 'blobId': 'c' * 40,
             'lfs': {'sha256': DIGEST, 'size': len(BODY)}}
    entry.update(change)
    monkeypatch.setattr(hub, '_request', lambda url, **kw: json.dumps({'id': 'org/data', 'sha': SHA,
        'private': False, 'siblings': [entry]}).encode() if '/api/' in url else pytest.fail('body before identity'))
    monkeypatch.setattr(transport, 'fetch_catalog', lambda *a, **kw: pytest.fail('unbound LFS request'))
    with pytest.raises(ValueError):
        _remote_files('hf://org/data', 'dataset', SHA, ['meta/tasks.parquet'], transport.MAX_BYTES)


@pytest.mark.parametrize('field', ['size', 'blobId', 'lfs.size', 'lfs.sha256'])
@pytest.mark.parametrize('bad', [None, True, 1, [], {}, float('inf'), float('nan'), 'x' * 8193])
def test_every_new_api_identity_field_is_strict_and_bounded(monkeypatch, field, bad):
    entry = {'rfilename': 'meta/tasks.parquet', 'size': len(BODY), 'blobId': 'c' * 40,
             'lfs': {'sha256': DIGEST, 'size': len(BODY)}}
    parent, key = (entry['lfs'], field[4:]) if field.startswith('lfs.') else (entry, field)
    parent[key] = bad
    monkeypatch.setattr(hub, '_request', lambda *a, **kw: json.dumps({'id': 'org/data', 'sha': SHA,
        'private': False, 'siblings': [entry]}).encode())
    monkeypatch.setattr(transport, 'fetch_catalog', lambda *a, **kw: pytest.fail('malformed identity downloaded'))
    with pytest.raises(ValueError):
        _remote_files('hf://org/data', 'dataset', SHA, ['meta/tasks.parquet'], transport.MAX_BYTES)


def test_transport_refusal_is_structured_unknown_at_public_boundary(tmp_path, monkeypatch):
    checkpoint, _ = fixture(tmp_path)
    monkeypatch.setattr(hub, '_request', lambda url, **kw: json.dumps({'id': 'org/robot', 'sha': SHA,
        'private': False, 'siblings': [{'rfilename': 'meta/tasks.parquet', 'size': len(BODY), 'blobId': 'c' * 40,
        'lfs': {'sha256': DIGEST, 'size': len(BODY)}}]}).encode())
    def refusal(*a, **kw):
        raise transport.CatalogTransportError('Untrusted redirect refused')
    monkeypatch.setattr(transport, 'fetch_catalog', refusal)
    report = check_language_vocabulary(checkpoint, instruction='CloseFridge', fetch=True)
    assert report['status'] == 'unknown' and report['verdict'] == 'UNKNOWN'
    assert all(f['status'] == 'unknown' for f in report['findings'])
    json.dumps(report, allow_nan=False)
