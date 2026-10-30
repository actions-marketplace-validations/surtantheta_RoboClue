"""Position evidence and safe operations remain separate under uncertain repair."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_compound_metadata_recovery import META, make_variant
from test_salvage import hashes

from policy_doctor import product_cli
from policy_doctor.artifacts import InputError
from policy_doctor.recovery_copy import write_recovery_copy
from policy_doctor.salvage import check_salvage
from policy_doctor.windows import audit_episode_windows


def mutate(root, kind):
    path = root / META
    table = pq.read_table(path)
    meta = table.to_pylist()
    if kind in ('swap_spans', 'swap_spans_split'):
        for key in ('dataset_from_index', 'dataset_to_index'):
            meta[3][key], meta[4][key] = meta[4][key], meta[3][key]
    elif kind in ('shift', 'negative', 'reverse'):
        start, end = {'shift': (10001, 10401), 'negative': (-400, 0), 'reverse': (10400, 10000)}[kind]
        meta[25].update(dataset_from_index=start, dataset_to_index=end)
    pq.write_table(pa.Table.from_pylist(meta, schema=table.schema), path)
    if kind == 'swap_rows':
        path = root / 'data/chunk-000/file-000.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[1200:2000] = rows[1600:2000] + rows[1200:1600]
        for i, row in enumerate(rows):
            row['index'] = i
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    if kind == 'swap_spans_split':
        first = root / 'data/chunk-000/file-001.parquet'
        second = root / 'data/chunk-000/file-002.parquet'
        a, b = pq.read_table(first), pq.read_table(second)
        pq.write_table(pa.concat_tables([a, b.slice(0, 100)]), first)
        pq.write_table(b.slice(100), second)
    if kind in ('index_scale', 'index_gap'):
        path = root / 'data/chunk-000/file-002.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            if kind == 'index_scale':
                row['index'] *= 2
            elif row['episode_index'] == 40 and row['frame_index'] == 1:
                row['index'] += 1000
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


@pytest.mark.parametrize('kind', ['swap_rows', 'swap_spans_split', 'shift', 'negative', 'reverse'])
@pytest.mark.parametrize('headers', [False, True])
def test_proven_physical_contradiction_is_error_even_with_unknown_plan(tmp_path, capsys, kind, headers):
    root = make_variant(tmp_path, corrected=True)
    mutate(root, kind)
    before = hashes(root)
    report = check_salvage(root, video_headers=headers)
    assert report['status'] == 'error'
    if not headers or kind in ('swap_rows', 'swap_spans_split'):
        assert report['plan']['strategy'] == 'unknown'
        assert report['episodes']['drop'] == []
        conflict = next(f for f in report['findings'] if f['code'] == 'salvage.metadata_row_conflict')
        assert conflict['status'] == 'error' and conflict['evidence']
        assert report['metadata_row_conflicts']['physical_span_conflicts'] > 0
        assert report['plan']['validity']['current_directory_resume_status'] == 'unsafe'
    flags = [] if headers else ['--no-video-headers']
    for fail_on in ('error', 'warning'):
        for prefix in (['check', str(root)], ['check', str(root), '--question', 'recovery'],
                       ['advanced', 'check-salvage', str(root)]):
            assert product_cli.main([*prefix, *flags, '--fail-on', fail_on, '--json']) == 1
            assert json.loads(capsys.readouterr().out)['status'] == 'error'
    assert hashes(root) == before


@pytest.mark.parametrize('kind,prefix', [('index_scale', 36), ('index_gap', 40)])
def test_noncontiguous_index_alone_cannot_authorize_exclusion_or_copy(tmp_path, kind, prefix):
    root = make_variant(tmp_path, corrected=True, videos=False)
    mutate(root, kind)
    before = hashes(root)
    report = check_salvage(root)
    assert report['status'] == 'error'
    assert report['metadata_row_conflicts']['physical_span_conflicts'] == 0
    assert report['metadata_row_conflicts']['row_index_conflicts'] > 0
    assert report['plan']['strategy'] == 'unknown'
    assert report['episodes']['drop'] == []
    assert report['episodes']['keep'] == [[0, prefix - 1]]
    assert report['episodes']['unresolved'] == [[prefix, 49]]
    assert any(f['code'] == 'salvage.row_identity' and f['status'] == 'error' for f in report['findings'])
    assert not any(f['code'] == 'salvage.episodes_dropped' for f in report['findings'])
    # Native positional selection still serves all fifty intact episode runs.
    data = pa.concat_tables([pq.read_table(p) for p in sorted((root / 'data').rglob('*.parquet'))]).to_pydict()
    for ep in range(50):
        assert data['episode_index'][ep * 400:(ep + 1) * 400] == [ep] * 400
        assert data['frame_index'][ep * 400:(ep + 1) * 400] == list(range(400))
    with pytest.raises(InputError, match='complete supported plan'):
        write_recovery_copy(root, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists() and hashes(root) == before


@pytest.mark.parametrize('videos', [False, True])
def test_healthy_ledgers_keep_all_episodes_without_conflict(tmp_path, videos):
    report = check_salvage(make_variant(tmp_path, corrected=True, videos=videos))
    assert report['status'] != 'error'
    assert report['episodes']['keep'] == [[0, 49]] and report['episodes']['drop'] == []
    assert not any(f['code'] == 'salvage.metadata_row_conflict' for f in report['findings'])


def test_unreadable_identity_is_unknown_without_invented_error_or_plan(tmp_path):
    root = make_variant(tmp_path, corrected=True, videos=False)
    report = check_salvage(root, max_rows=100)
    assert report['status'] == 'unknown' and report['plan']['strategy'] == 'unknown'
    assert not report['episodes'].get('drop')


def test_pointer_examples_and_precheck_statistics_reason_are_accurate(tmp_path, capsys):
    root = make_variant(tmp_path)
    assert product_cli.main(['check', str(root), '--no-video-headers', '--json']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['statistics_comparison']['reason'] != 'Explicitly disabled.'
    conflict = report['metadata_row_conflicts']
    assert conflict['span_examples'] == [19, 20, 21]
    assert conflict['observations'][0]['episode_index'] == 18
    assert conflict['observations'][0]['physical_span_conflict'] is False
    assert sum(f['code'] == 'salvage.video_unchecked' for f in report['findings']) == 1
class _TC:
    def __init__(self, b, p):
        self.b, self.p = b, p

    def byte(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self):
        shift = result = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return result

    def zz(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def skip(self, t):
        if t in (1, 2):
            return
        if t == 3:
            self.p += 1
        elif t in (4, 5, 6):
            self.varint()
        elif t == 7:
            self.p += 8
        elif t == 8:
            n = self.varint()  # read the length first: `self.p += self.varint()` would use the stale self.p
            self.p += n
        elif t in (9, 10):
            size, et = self.list_header()
            for _ in range(size):
                if et in (1, 2):
                    self.p += 1
                else:
                    self.skip(et)
        elif t == 11:
            size = self.varint()
            if size:
                kv = self.byte()
                for _ in range(size):
                    for tt in (kv >> 4, kv & 0x0F):
                        if tt in (1, 2):
                            self.p += 1
                        else:
                            self.skip(tt)
        elif t == 12:
            self.struct(lambda fid, tt: False)
        else:
            raise ValueError(f'unknown thrift type {t}')

    def list_header(self):
        h = self.byte()
        size, et = h >> 4, h & 0x0F
        if size == 15:
            size = self.varint()
        return size, et

    def struct(self, visit):
        """visit(fid, type) -> True if it consumed the value itself."""
        last = 0
        while True:
            h = self.byte()
            if h == 0:
                return
            t, d = h & 0x0F, h >> 4
            fid = self.zz() if d == 0 else last + d
            last = fid
            if not visit(fid, t):
                self.skip(t)


def _enc_varint_i64(v):
    z = (v << 1) ^ (v >> 63)
    out = bytearray()
    while True:
        b = z & 0x7F
        z >>= 7
        if z:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def patch_num_rows(path, file_rows=None, rg_rows=None, rg_index=0):
    raw = bytearray(Path(path).read_bytes())
    flen = int.from_bytes(raw[-8:-4], 'little')
    start = len(raw) - 8 - flen
    tc = _TC(raw, start)
    spans = []  # (begin, end, new_value)

    def rowgroup_visit(fid, t):
        if fid == 3 and t == 6 and rg_rows is not None and rg_visit.idx == rg_index:
            b = tc.p
            tc.varint()
            spans.append((b, tc.p, rg_rows))
            return True
        return False

    def rg_visit():
        pass

    def top(fid, t):
        if fid == 3 and t == 6 and file_rows is not None:
            b = tc.p
            tc.varint()
            spans.append((b, tc.p, file_rows))
            return True
        if fid == 4 and t == 9:
            size, et = tc.list_header()
            assert et == 12
            for i in range(size):
                rg_visit.idx = i
                tc.struct(rowgroup_visit)
            return True
        return False
    tc.struct(top)
    assert tc.p == len(raw) - 8, (tc.p, len(raw) - 8)
    for b, e, v in sorted(spans, reverse=True):
        raw[b:e] = _enc_varint_i64(v)
    newlen = len(raw) - 8 - start
    raw[-8:-4] = newlen.to_bytes(4, 'little')
    Path(path).write_bytes(bytes(raw))
    md = pq.ParquetFile(path).metadata
    return {'file_num_rows': md.num_rows, 'rg0_num_rows': md.row_group(0).num_rows, 'row_groups': md.num_row_groups,
            'patched_fields': len(spans)}



@pytest.mark.parametrize('relative,count', [(META, 10 ** 9), ('data/chunk-000/file-002.parquet', 10 ** 12),
                                            (META, -1), (META, 49)])
def test_lying_row_group_footer_is_refused_before_arrow_read(tmp_path, monkeypatch, relative, count):
    root = make_variant(tmp_path, corrected=True, videos=False)
    path = root / relative
    observed = patch_num_rows(path, rg_rows=count)
    assert observed['rg0_num_rows'] == count
    before = hashes(root)
    original = pq.ParquetFile

    class Guarded:
        def __init__(self, source, *args, **kwargs):
            self.inner, self.source = original(source, *args, **kwargs), Path(source)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read(self, *args, **kwargs):
            assert self.source != path, 'Arrow read must not see the lying row count'
            return self.inner.read(*args, **kwargs)

    monkeypatch.setattr(pq, 'ParquetFile', Guarded)
    report = check_salvage(root)
    expected = 'error' if count >= 0 else 'unknown'
    assert report['status'] == expected and report['plan']['strategy'] == 'unknown'
    assert not report['episodes'].get('drop')
    assert any('row-group' in f['message'] for f in report['findings'])
    assert hashes(root) == before


@pytest.mark.parametrize('exception', [MemoryError, pa.ArrowMemoryError])
def test_arrow_allocation_failure_is_a_named_refusal_without_cli_traceback(tmp_path, monkeypatch, capsys, exception):
    root = make_variant(tmp_path, corrected=True, videos=False)
    from policy_doctor import salvage

    def fail(*args, **kwargs):
        raise exception('allocation refused')

    monkeypatch.setattr(salvage._Work, 'table', fail)
    with pytest.raises(InputError, match='bounded salvage evidence.*allocation refused'):
        check_salvage(root)
    assert product_cli.main(['check', str(root), '--json']) == 2
    output = capsys.readouterr()
    assert 'Traceback' not in output.out + output.err


@pytest.mark.parametrize('kind,code', [('negative', 'episode.invalid_span'), ('reverse', 'episode.range_contradiction'),
                                     ('index_scale', 'episode.row_index_mismatch'),
                                     ('index_gap', 'episode.row_index_mismatch')])
def test_default_preflight_retains_independently_proved_span_and_index_errors(tmp_path, kind, code):
    from policy_doctor.preflight import preflight

    root = make_variant(tmp_path, corrected=True, videos=False)
    mutate(root, kind)
    report = preflight(str(root))
    assert report['status'] == 'error'
    assert any(f['code'] == code and f['status'] == 'error' for f in report['findings'])


@pytest.mark.parametrize('relative,count', [(META, 10 ** 9), ('data/chunk-000/file-002.parquet', 10 ** 12)])
def test_episode_consistency_fallback_cannot_bypass_footer_allocation_guard(tmp_path, monkeypatch, relative, count):
    root = make_variant(tmp_path, corrected=True, videos=False)
    path = root / relative
    patch_num_rows(path, rg_rows=count)
    original = pq.ParquetFile

    class Guarded:
        def __init__(self, source, *args, **kwargs):
            self.inner, self.source = original(source, *args, **kwargs), Path(source)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read(self, *args, **kwargs):
            assert self.source != path, 'The consistency fallback must not read the lying row group'
            return self.inner.read(*args, **kwargs)

    monkeypatch.setattr(pq, 'ParquetFile', Guarded)
    report = audit_episode_windows(root)
    assert report['status'] == 'error'
    assert any(f['code'] == 'episode.parquet_row_counts' and f['status'] == 'error' for f in report['findings'])
    assert report['metadata_patches'] == []
