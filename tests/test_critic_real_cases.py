"""Regressions for the real ALOHA metadata and UR5 batch-recovery reports."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import BaseHandler, OpenerDirector, Request

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_bulk_video_recovery import source
from test_salvage import hashes, rows_for, write

from policy_doctor import bulk_video_recovery as bulk
from policy_doctor import recovery_copy, video_recovery
from policy_doctor.salvage import check_salvage

FIXTURE = Path(__file__).parent / 'data/native_aloha_pointer_spans.json'


def aloha(tmp_path, *, corrupt=None):
    recorded = json.loads(FIXTURE.read_text())
    episodes = [(m['episode_index'], m['length']) for m in recorded['metadata']]
    rows = rows_for(episodes)
    meta = recorded['metadata']
    if corrupt == 'frame':
        rows[8000]['frame_index'] = 77
    elif corrupt == 'duplicate':
        rows[-1]['episode_index'] = 18
    elif corrupt == 'index':
        rows[8000]['index'] = 7
    elif corrupt == 'length':
        meta[20]['length'] += 1
    elif corrupt == 'metadata_identity':
        meta[20]['episode_index'] = 19
    shards = [rows[s['start']:s['start'] + s['rows']] for s in recorded['data_shards']]
    if corrupt == 'split_run':
        shards = [rows[:7300], rows[7300:14400], rows[14400:]]
    # Preserve native-shaped episode summaries so copy application can recheck
    # its rebuilt aggregates. Only identities/spans/pointers come from the source.
    for m in meta:
        values = rows[m['episode_index'] * 400:(m['episode_index'] + 1) * 400]
        for key in ('action', 'index'):
            a = np.asarray([r[key] if key == 'action' else [r[key]] for r in values], dtype=float)
            stats = {'min': a.min(0), 'max': a.max(0), 'mean': a.mean(0), 'std': a.std(0),
                     'count': np.asarray([len(a)])}
            m.update({f'stats/{key}/{k}': v.tolist() for k, v in stats.items()})
    root = write(tmp_path / 'source', shards, [meta], 50, 20000)
    global_stats = {}
    for key in ('action', 'index'):
        a = np.asarray([r[key] if key == 'action' else [r[key]] for r in rows], dtype=float)
        values = {'min': a.min(0), 'max': a.max(0), 'mean': a.mean(0), 'std': a.std(0),
                  'count': np.asarray([len(a)])}
        global_stats[key] = {k: v.tolist() for k, v in values.items()}
    (root / 'meta/stats.json').write_text(json.dumps(global_stats))
    return root


def test_real_aloha_all_intact_rows_have_metadata_repair_not_drop(tmp_path):
    root = aloha(tmp_path)
    before = hashes(root)
    report = check_salvage(root)
    assert report['assessment_kind'] == 'metadata_rows_mismatch'
    assert report['episodes']['keep'] == [[0, 49]]
    assert report['episodes']['drop'] == []
    assert not any(f['code'] == 'salvage.episodes_dropped' for f in report['findings'])
    plan = report['plan']
    assert [r['episode_index'] for r in plan['pointer_corrections']] == [18, 36]
    assert [r['to']['data/file_index'] for r in plan['pointer_corrections']] == [1, 2]
    assert len(plan['row_span_corrections']) == 31
    assert plan['changes']['metadata_rows_removed'] == 0
    assert plan['kept_frames'] == 20000 and plan['files']['lists_complete']
    assert plan['files']['rewrite'][0]['row_span_corrections'] == 31
    explanation = next(f for f in report['findings'] if f['code'] == 'salvage.metadata_plan')
    assert explanation['status'] == 'error'
    assert 'reads the original differently' in explanation['message']
    assert 'already reads the kept episodes this way' not in explanation['message']
    assert hashes(root) == before


def test_real_aloha_repair_applies_only_metadata_and_preserves_all_episodes(tmp_path):
    root = aloha(tmp_path)
    before = hashes(root)
    original = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet')
    result = recovery_copy.write_recovery_copy(root, tmp_path / 'repaired')
    assert result['status'] == 'pass'
    assert result['candidate_report']['plan']['strategy'] == 'none_needed'
    assert result['candidate_report']['episodes']['keep'] == [[0, 49]]
    repaired = pq.read_table(tmp_path / 'repaired/meta/episodes/chunk-000/file-000.parquet')
    assert repaired.schema.equals(original.schema, check_metadata=True)
    allowed = {'data/chunk_index', 'data/file_index', 'dataset_from_index', 'dataset_to_index'}
    for a, b in zip(original.to_pylist(), repaired.to_pylist(), strict=True):
        assert all(a[k] == b[k] for k in a if k not in allowed)
        assert b['dataset_from_index'] == b['episode_index'] * 400
        assert b['dataset_to_index'] == (b['episode_index'] + 1) * 400
    for rel in before:
        if rel.startswith('data/'):
            assert (tmp_path / 'repaired' / rel).read_bytes() == (root / rel).read_bytes()
    assert hashes(root) == before


@pytest.mark.parametrize('corrupt', ['frame', 'duplicate', 'index', 'length', 'metadata_identity', 'split_run'])
def test_metadata_repair_refuses_ambiguous_or_incomplete_identity_runs(tmp_path, corrupt):
    report = check_salvage(aloha(tmp_path, corrupt=corrupt))
    assert report.get('assessment_kind') != 'metadata_rows_mismatch'
    assert not report['plan'].get('row_span_corrections')
    assert report['assessment_kind'] == 'metadata_rows_unresolved'
    assert report['plan']['strategy'] == 'unknown'
    assert report['counts']['drop'] == 0 and report['episodes']['drop'] == []
    assert any(f['code'] == 'salvage.metadata_row_conflict' for f in report['findings'])


def test_metadata_repair_plain_answer_prioritizes_intact_rows(tmp_path):
    from policy_doctor import product_cli

    root = aloha(tmp_path)
    text = product_cli.render(check_salvage(root), source=str(root), kind='recovery')
    assert 'identity columns support a repair' in text
    assert 'for episodes 0–49' in text or 'for episodes 0-49' in text
    assert 'do not resume this directory' in text
    assert 'All episode rows are intact' not in text
    assert '--explain' in text


def test_batch_body_budget_uses_real_urllib_get_without_integer_body(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.command, self.headers.get('Content-Length')))
            self.send_response(200)
            self.send_header('Content-Length', '4')
            self.end_headers()
            self.wfile.write(b'body')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        budget = bulk.BodyBudget(8)
        transport = budget.transport(tmp_path)
        with transport.opener.open(Request(f'http://127.0.0.1:{server.server_port}/'), timeout=3) as response:
            assert response.read(8) == b'body'
            assert response.read(8) == b''
        assert requests == [('GET', None)]
        assert budget.used == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_full_batch_recovery_through_actual_urllib_opener_contract(monkeypatch, tmp_path):
    public, _ = source(monkeypatch, tmp_path)

    class RecordedHTTPS(BaseHandler):
        def https_open(self, request):
            assert request.data is None, 'Timeout was supplied as an HTTP request body'
            assert request.timeout == 15
            return public.open(request, timeout=request.timeout)

    opener = OpenerDirector()
    opener.add_handler(RecordedHTTPS())
    monkeypatch.setattr(video_recovery, '_opener', lambda: opener)
    report = bulk.recover_all_video_references(
        'community/recorded', '2' * 40, '1' * 40, tmp_path / 'batch', fetch=True)
    assert report['status'] == 'pass' and report['proposal_complete']
    assert report['verified_associations'] == 2 and report['proposed_cells'] == 6
    assert bulk.reproduce_bulk_proposal(tmp_path / 'batch')['status'] == 'pass'
