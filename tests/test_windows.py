"""Native episode windows must follow actual row identity before suggesting metadata changes."""
import hashlib
import importlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError


def audit(*args, **kwargs):
    return importlib.import_module('policy_doctor.windows').audit_episode_windows(*args, **kwargs)


def make_dataset(tmp_path, *, bad=False, split=False, mutate=None):
    root = tmp_path / 'dataset'
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'total_frames': 60, 'total_episodes': 2,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {'action': {'dtype': 'float32', 'shape': [2]}, 'state': {'dtype': 'float32', 'shape': [2]}}}
    rows = [{'index': i, 'episode_index': i // 30, 'frame_index': i % 30,
             'action': [float(i), -float(i)], 'state': [1000. + i, 0.]} for i in range(60)]
    meta = [{'episode_index': e, 'length': 30, 'dataset_from_index': e * 30,
             'dataset_to_index': 33 if bad and e == 0 else (e + 1) * 30,
             'data/chunk_index': 0, 'data/file_index': 1 if split and e == 1 else 0} for e in range(2)]
    if mutate:
        mutate(info, rows, meta)
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist(meta), root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    chunks = [rows[:20], rows[20:]] if split else [rows]
    for i, chunk in enumerate(chunks):
        pq.write_table(pa.Table.from_pylist(chunk), root / 'data' / 'chunk-000' / f'file-{i:03d}.parquet')
    return root


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def test_bad_metadata_is_localized_with_exact_metadata_only_whatif(tmp_path):
    root = make_dataset(tmp_path, bad=True)
    before = hashes(root)
    report = audit(str(root), episode_index=0, anchor_index=29, delta_indices=[0, 1, 2, 3, 4])
    assert report['status'] == 'error'
    patch = next(p for p in report['metadata_patches'] if p['column'] == 'dataset_to_index')
    assert patch['row'] == 0 and patch['observed'] == 33 and patch['expected'] == 30
    assert patch['sha256'] == before['meta/episodes/chunk-000/file-000.parquet']
    declared, corrected = report['query']['declared'], report['query']['corrected_what_if']
    assert declared['indices'] == [29, 30, 31, 32, 32]
    assert declared['action_is_pad'] == [False, False, False, False, True]
    assert declared['selected_episode_indices'] == [0, 1, 1, 1, 1]
    assert len(declared['cross_episode_witnesses']) == 3
    # Unpadded rows of episode 1 inside episode 0's window: the declared window itself is the error.
    assert declared['status'] == 'error' and 'utils.py:146-162' in declared['reason'] and '[1]' in declared['reason']
    assert corrected['status'] == 'pass' and corrected['reason'] is None
    assert corrected['indices'] == [29] * 5 and corrected['action_is_pad'] == [False, True, True, True, True]
    assert corrected['values'] == [[29., -29.]] * 5 and corrected['cross_episode_witnesses'] == []
    assert hashes(root) == before


@pytest.mark.parametrize('episode,anchor,deltas,indices,pad', [
    (0, 29, [0, 1, 2, 3, 4], [29] * 5, [False, True, True, True, True]),
    (0, 10, [0, 1, 2], [10, 11, 12], [False] * 3),
    (1, 59, [0, 1, 2], [59] * 3, [False, True, True]),
    (0, 0, [-2, -1, 0, 1], [0, 0, 0, 1], [True, True, False, False]),
])
def test_healthy_native_windows_match_without_correction(tmp_path, episode, anchor, deltas, indices, pad):
    root = make_dataset(tmp_path)
    report = audit(str(root), episode_index=episode, anchor_index=anchor, delta_indices=deltas)
    assert report['status'] == 'pass' and not report['metadata_patches']
    assert report['query']['declared']['indices'] == indices
    assert report['query']['declared']['action_is_pad'] == pad
    assert report['query']['declared']['cross_episode_witnesses'] == []
    assert report['query']['declared']['status'] == 'pass' and report['query']['declared']['reason'] is None


def test_window_rule_cites_the_pinned_lerobot_revision(tmp_path):
    from policy_doctor.artifacts import LEROBOT_REVISION

    reference = audit(str(make_dataset(tmp_path)))['upstream_reference']
    assert reference['revision'] == LEROBOT_REVISION
    assert reference['locations'] == ['src/lerobot/datasets/dataset_reader.py:298-311', 'src/lerobot/datasets/utils.py:146-162']


def test_episode_continuation_shard_is_discovered_even_without_its_own_start_pointer(tmp_path):
    root = make_dataset(tmp_path, split=True)
    report = audit(str(root), episode_index=0, anchor_index=19, delta_indices=[0, 1, 2])
    assert report['status'] == 'pass' and report['coverage']['data_files_read'] == 2
    assert report['coverage']['data_rows_read'] == 60
    assert report['query']['declared']['values'] == [[19., -19.], [20., -20.], [21., -21.]]


@pytest.mark.parametrize('mutation', ['duplicate', 'gap', 'noncontiguous', 'missing_index', 'missing_shard'])
def test_ambiguous_or_incomplete_row_identity_never_derives_a_patch(tmp_path, mutation):
    def alter(info, rows, meta):
        if mutation == 'duplicate':
            rows[31]['index'] = 30
        elif mutation == 'gap':
            rows[31]['index'] = 61
        elif mutation == 'noncontiguous':
            rows[20]['episode_index'] = 1
        elif mutation == 'missing_index':
            for row in rows:
                row.pop('index')
        elif mutation == 'missing_shard':
            meta[1]['data/file_index'] = 9

    root = make_dataset(tmp_path, bad=True, mutate=alter)
    report = audit(str(root))
    # Ambiguous identity never derives a patch or a pass. Certain per-episode count contradictions
    # (for example a relabelled row) are still reported by the bounded consistency pass.
    assert report['status'] != 'pass' and not report['metadata_patches']
    if report['status'] == 'error':
        assert {f['code'] for f in report['findings'] if f['status'] == 'error'} <= {
            'episode.length_mismatch', 'episode.metadata_without_rows', 'episode.rows_without_metadata',
            'episode.declared_totals', 'episode.duplicate_global_index',
            'episode.invalid_span', 'episode.row_index_mismatch'}
    assert not report['coverage']['identity_verified']


@pytest.mark.parametrize('mutation', ['custom_template', 'missing_metadata', 'missing_total', 'wrong_version'])
def test_unsupported_or_partial_intake_is_unknown(tmp_path, mutation):
    def alter(info, rows, meta):
        if mutation == 'custom_template':
            info['data_path'] = 'custom/{arbitrary_code}.parquet'
        elif mutation == 'missing_total':
            info.pop('total_frames')
        elif mutation == 'wrong_version':
            info['codebase_version'] = 'v2.1'

    root = make_dataset(tmp_path, mutate=alter)
    if mutation == 'missing_metadata':
        (root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet').unlink()
    result = audit(str(root))
    assert result['status'] == 'unknown' and not result['metadata_patches']


@pytest.mark.parametrize('kwargs', [
    {'episode_index': True, 'anchor_index': 29, 'delta_indices': [0]},
    {'episode_index': 0, 'anchor_index': True, 'delta_indices': [0]},
    {'episode_index': 0, 'anchor_index': 29, 'delta_indices': [True]},
    {'episode_index': 0, 'anchor_index': 2**63, 'delta_indices': [0]},
    {'episode_index': 0, 'anchor_index': 29, 'delta_indices': [2**63]},
    {'episode_index': 0}, {'max_rows': True}, {'max_files': False},
])
def test_strict_query_and_budget_inputs_refuse(tmp_path, kwargs):
    with pytest.raises(InputError):
        audit(str(make_dataset(tmp_path)), **kwargs)


def test_anchor_is_absolute_global_index_and_must_match_episode(tmp_path):
    with pytest.raises(InputError, match='anchor|episode'):
        audit(str(make_dataset(tmp_path)), episode_index=0, anchor_index=30, delta_indices=[0])


def test_budget_exhaustion_is_unknown_with_no_partial_correction(tmp_path):
    root = make_dataset(tmp_path, bad=True)
    result = audit(str(root), max_rows=20)
    assert result['status'] == 'unknown' and not result['metadata_patches']


def test_symlink_escape_is_refused_before_read(tmp_path):
    root = make_dataset(tmp_path)
    shard = root / 'data' / 'chunk-000' / 'file-000.parquet'
    outside = tmp_path / 'outside.parquet'
    shard.rename(outside)
    shard.symlink_to(outside)
    with pytest.raises(InputError, match='symlink|contain|root'):
        audit(str(root))


def test_missing_action_values_do_not_guess_a_state_alias(tmp_path):
    def alter(info, rows, meta):
        for row in rows:
            row.pop('action')

    root = make_dataset(tmp_path, mutate=alter)
    before = hashes(root)
    result = audit(str(root), episode_index=0, anchor_index=29, delta_indices=[0, 1])
    assert result['query']['declared']['value_status'] == 'unknown'
    assert result['query']['declared']['values'] is None
    assert hashes(root) == before


def test_lexicographically_loaded_physical_rows_are_not_sorted_into_apparent_health(tmp_path):
    def alter(info, rows, meta):
        rows[1], rows[2] = rows[2], rows[1]

    result = audit(str(make_dataset(tmp_path, mutate=alter)))
    assert result['status'] == 'error' and not result['metadata_patches']
    assert any(f['code'] == 'episode.row_index_mismatch' for f in result['findings'])


@pytest.mark.parametrize('budget', ['footer', 'expanded', 'files'])
def test_parquet_resource_limits_stop_before_certification(tmp_path, monkeypatch, budget):
    module = importlib.import_module('policy_doctor.windows')
    root = make_dataset(tmp_path, bad=True)
    kwargs = {}
    if budget == 'footer':
        monkeypatch.setattr(module, 'MAX_FOOTER_BYTES', 8)
    elif budget == 'expanded':
        monkeypatch.setattr(module, 'MAX_EXPANDED_BYTES', 8)
    else:
        kwargs['max_files'] = 1
    result = audit(str(root), **kwargs)
    # The independently bounded consistency pass still reads the two small
    # files and proves the 33-row span contradicts its declared length of 30.
    assert result['status'] == ('error' if budget == 'files' else 'unknown')
    assert not result['metadata_patches']


def test_artifact_changed_during_read_refuses_report(tmp_path, monkeypatch):
    module = importlib.import_module('policy_doctor.windows')
    root = make_dataset(tmp_path)
    original = module._Intake.parquet

    def changing(self, path, columns, role, **kwargs):
        result = original(self, path, columns, role, **kwargs)
        if role == 'dataset_parquet':
            with path.open('ab') as stream:
                stream.write(b' ')
        return result

    monkeypatch.setattr(module._Intake, 'parquet', changing)
    with pytest.raises(InputError, match='changed'):
        audit(str(root))


def test_nonfloating_action_never_becomes_a_state_alias_or_healthy_value(tmp_path):
    def alter(info, rows, meta):
        for row in rows:
            row['action'] = [int(row['index']), 0]

    root = make_dataset(tmp_path, mutate=alter)
    result = audit(str(root), episode_index=0, anchor_index=29, delta_indices=[0, 1])
    assert result['status'] == 'unknown'
    assert result['query']['declared']['value_status'] == 'unknown'
    assert result['query']['declared']['values'] is None


def test_findings_and_patches_are_bounded_with_exact_totals(tmp_path):
    def alter(info, rows, meta):
        info['total_episodes'] = 30
        for i, row in enumerate(rows):
            row['episode_index'], row['frame_index'] = i // 2, i % 2
        meta[:] = [{'episode_index': e, 'length': 2, 'dataset_from_index': 2 * e,
                    'dataset_to_index': 2 * e + 3, 'data/chunk_index': 0, 'data/file_index': 0}
                   for e in range(30)]

    result = audit(str(make_dataset(tmp_path, mutate=alter)))
    assert result['status'] == 'error'
    assert result['counts'] == {'findings': 30, 'metadata_patches': 30, 'episodes': 30}
    assert len(result['findings']) == len(result['metadata_patches']) == len(result['episodes']) == 16
    assert result['findings_truncated'] and result['metadata_patches_truncated'] and result['episodes_truncated']
    assert len(json.dumps(result).encode()) < 192 * 1024


def test_largest_valid_action_cell_window_stays_within_transport_budget(tmp_path):
    def alter(info, rows, meta):
        info['features']['action']['shape'] = [32]
        for row in rows:
            row['action'] = [1.7976931348623157e308] * 32

    root = make_dataset(tmp_path, bad=True, mutate=alter)
    result = audit(str(root), episode_index=0, anchor_index=29, delta_indices=list(range(64)))
    assert result['status'] == 'error'
    assert len(result['query']['declared']['values']) == 64
    assert len(result['query']['corrected_what_if']['values'][0]) == 32
    assert len(json.dumps(result, allow_nan=False).encode()) <= 192 * 1024


def test_excessive_action_cell_window_has_explicit_reduce_window_refusal(tmp_path):
    def alter(info, rows, meta):
        info['features']['action']['shape'] = [64]
        for row in rows:
            row['action'] = [float(row['index'])] * 64

    root = make_dataset(tmp_path, mutate=alter)
    with pytest.raises(InputError, match='4096.*reduce delta_indices'):
        audit(str(root), episode_index=0, anchor_index=29, delta_indices=list(range(64)))


def test_explicit_other_backend_does_not_audit_stale_parquet_as_current(tmp_path):
    def alter(info, rows, meta):
        info['storage_format'] = 'lance'

    result = audit(str(make_dataset(tmp_path, mutate=alter)))
    assert result['status'] == 'unknown' and not result['coverage']['identity_verified']


@pytest.mark.parametrize('damage', ['length', 'envelope'])
def test_explicit_other_backend_never_reports_stale_parquet_contradictions(tmp_path, damage):
    def alter(info, rows, meta):
        info['storage_format'] = 'lance'
        if damage == 'length':
            meta[0]['length'] = 31

    root = make_dataset(tmp_path, mutate=alter)
    if damage == 'envelope':
        stale = root / 'data' / 'chunk-000' / 'file-000.parquet'
        stale.write_bytes(b'XXXX' + stale.read_bytes()[4:])
    result = audit(str(root))
    assert result['status'] == 'unknown', result['findings']
    assert not any(finding.get('status') == 'error' for finding in result['findings'])
    assert result['coverage']['episode_consistency'].startswith('not checked')


@pytest.mark.parametrize('backend', ['parquet', 'lance'])
def test_explicit_other_backend_never_reports_stale_video_index(tmp_path, backend):
    def alter(info, rows, meta):
        info.update(fps=30, storage_format=backend,
                    video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        info['features']['observation.images.front'] = {'dtype': 'video', 'shape': [4, 4, 3]}

    result = audit(str(make_dataset(tmp_path, mutate=alter)), video_headers=False)
    errors = {f['code'] for f in result['findings'] if f.get('status') == 'error'}
    if backend == 'parquet':
        assert any(code.startswith('video.') for code in errors), result['findings']
    else:
        assert result['status'] == 'unknown' and not errors, result['findings']


@pytest.mark.parametrize('mutation', ['all_width', 'one_width', 'missing_declaration', 'nonfloat_declaration'])
def test_queried_action_values_require_the_native_shape_and_floating_declaration(tmp_path, mutation):
    def alter(info, rows, meta):
        if mutation == 'all_width':
            for row in rows:
                row['action'] = [float(row['index'])]
        elif mutation == 'one_width':
            rows[29]['action'] = [29.]
        elif mutation == 'missing_declaration':
            info['features'].pop('action')
        else:
            info['features']['action']['dtype'] = 'int64'

    result = audit(str(make_dataset(tmp_path, mutate=alter)), episode_index=0, anchor_index=28, delta_indices=[0, 1])
    assert result['coverage']['identity_verified']
    assert result['status'] == 'unknown'
    assert result['query']['declared']['value_status'] == 'unknown'
    assert result['query']['declared']['values'] is None


def test_metadata_physical_episode_order_is_not_silently_remapped(tmp_path):
    def alter(info, rows, meta):
        meta.reverse()

    result = audit(str(make_dataset(tmp_path, mutate=alter)), episode_index=0, anchor_index=29, delta_indices=[0, 1])
    assert result['status'] == 'unknown' and not result['coverage']['identity_verified']
    assert not result['metadata_patches']


@pytest.mark.parametrize('reversed_files', [False, True])
def test_split_metadata_files_preserve_native_physical_episode_lookup(tmp_path, reversed_files):
    root = make_dataset(tmp_path)
    directory = root / 'meta' / 'episodes' / 'chunk-000'
    original = pq.read_table(directory / 'file-000.parquet').to_pylist()
    if reversed_files:
        original.reverse()
    for index, row in enumerate(original):
        pq.write_table(pa.Table.from_pylist([row]), directory / f'file-{index:03d}.parquet')
    result = audit(str(root), episode_index=0, anchor_index=29, delta_indices=[0, 1])
    assert result['status'] == ('unknown' if reversed_files else 'pass')
    if reversed_files:
        assert result['query'] is None and not result['metadata_patches']
    else:
        assert result['coverage']['metadata_files_read'] == 2
