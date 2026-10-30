"""A second defect must not turn a metadata contradiction into episode loss."""
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from recovery_fixtures import movie
from test_critic_real_cases import aloha
from test_salvage import hashes, strip_footer

from policy_doctor import product_cli
from policy_doctor.artifacts import InputError
from policy_doctor.recovery_copy import write_recovery_copy
from policy_doctor.salvage import check_salvage
from policy_doctor.sim_salvage_frontdoor import concise_salvage

CAM = 'observation.images.top'
META = 'meta/episodes/chunk-000/file-000.parquet'


def make_variant(tmp_path, variant='clean', *, corrected=False, videos=True):
    root = aloha(tmp_path)
    path = root / META
    table = pq.read_table(path)
    meta = table.to_pylist()
    if corrected:
        for row in meta:
            ep = row['episode_index']
            row.update({'dataset_from_index': ep * 400, 'dataset_to_index': (ep + 1) * 400,
                        'data/chunk_index': 0, 'data/file_index': ep // 18})
    if videos:
        info = json.loads((root / 'meta/info.json').read_text())
        info.update(fps=5, video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        info['features'][CAM] = {'dtype': 'video', 'shape': [480, 640, 3]}
        (root / 'meta/info.json').write_text(json.dumps(info))
        for row in meta:
            ep = row['episode_index']
            row.update({f'videos/{CAM}/chunk_index': 0, f'videos/{CAM}/file_index': 0,
                        f'videos/{CAM}/from_timestamp': ep * 80.,
                        f'videos/{CAM}/to_timestamp': (ep + 1) * 80.})
        video = root / f'videos/{CAM}/chunk-000/file-000.mp4'
        video.parent.mkdir(parents=True)
        # Structurally valid header/sample tables only. No decoded-pixel claim.
        video.write_bytes(movie((b'packet',) * 20000))
    if variant == 'video':
        meta[-1][f'videos/{CAM}/from_timestamp'] = 5000.
        meta[-1][f'videos/{CAM}/to_timestamp'] = 5080.
    pq.write_table(pa.Table.from_pylist(meta), path)
    if variant == 'totals':
        info = json.loads((root / 'meta/info.json').read_text())
        info['total_frames'] = 20001
        (root / 'meta/info.json').write_text(json.dumps(info))
    elif variant == 'frame':
        path = root / 'data/chunk-000/file-001.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        # Episode30 begins at local row4800.
        rows[4800]['frame_index'], rows[4801]['frame_index'] = 1, 0
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    elif variant == 'unreadable':
        strip_footer(root / 'data/chunk-000/file-002.parquet')
    return root


@pytest.mark.parametrize('variant', ['totals', 'frame', 'video', 'unreadable'])
def test_compound_defect_never_approves_exclusion_or_automatic_copy(tmp_path, variant):
    root = make_variant(tmp_path, variant)
    before = hashes(root)
    report = check_salvage(root)
    assert report['assessment_kind'] == 'metadata_rows_unresolved'
    assert report['plan']['strategy'] == 'unknown'
    assert report['plan']['validity']['resume_safe'] is False
    assert report['episodes']['drop'] == [] and report['counts']['drop'] == 0
    assert report['episodes']['keep'] == [[0, 17]]
    assert report['episodes']['unresolved'] == [[18, 49]]
    assert not any(f['code'] == 'salvage.episodes_dropped' for f in report['findings'])
    assert any(f['code'] == 'salvage.metadata_row_conflict' and f['status'] == 'error'
               for f in report['findings'])
    observations = report['metadata_row_conflicts']['observations']
    assert report['metadata_row_conflicts']['observations_complete']
    assert report['metadata_row_conflicts']['observations_total'] == len(observations)
    if variant != 'unreadable':
        assert {o['episode_index'] for o in observations
                if o['declared_pointer'] != o['observed_pointer']} == {18, 36}
    target = tmp_path / 'new-copy'
    with pytest.raises(InputError, match='complete supported plan'):
        write_recovery_copy(root, target)
    assert not target.exists() and hashes(root) == before
    view = concise_salvage(report)
    assert view['counts_episodes']['drop'] == 0 and view['episodes']['unresolved_count'] == 32


@pytest.mark.parametrize('variant', ['totals', 'frame', 'video', 'unreadable'])
def test_default_and_recovery_cli_agree_on_compound_refusal(tmp_path, capsys, variant):
    root = make_variant(tmp_path, variant)
    for options in ([], ['--question', 'recovery'], ['--question', 'dataset']):
        status = product_cli.main(['check', str(root), *options, '--json'])
        raw = capsys.readouterr()
        assert status in (1, 3), (status, raw)
        report = json.loads(raw.out)
        assert report['assessment_kind'] == 'metadata_rows_unresolved'
        assert report['counts']['drop'] == 0 and report['plan']['strategy'] == 'unknown'
    status = product_cli.main(['check', str(root)])
    text = capsys.readouterr().out
    assert status in (1, 3)
    assert 'REVIEW REQUIRED' in text and 'do not resume' in text
    assert 'No keep/drop plan is approved' in text
    assert 'keep episodes 0-49' not in text


def test_header_optout_preserves_warning_and_next_header_check(tmp_path, capsys):
    root = make_variant(tmp_path, 'video')
    assert product_cli.main(['check', str(root), '--no-video-headers']) == 1
    text = capsys.readouterr().out
    assert 'REVIEW REQUIRED' in text and 'No keep/drop plan is approved' in text
    assert 'do not resume this directory' in text
    assert 'Video evidence was not fully checked' in text and '--video-headers' in text
    assert 'All episode rows are intact' not in text


@pytest.mark.parametrize('variant,expected', [('frame', 30), ('video', 49)])
def test_correct_spans_local_defect_is_not_confused_with_global_metadata_conflict(tmp_path, variant, expected):
    report = check_salvage(make_variant(tmp_path, variant, corrected=True))
    assert report.get('assessment_kind') != 'metadata_rows_unresolved'
    assert [e['episode_index'] for e in report['episodes']['drop']] == [expected]


def test_healthy_corrected_identity_ledger_has_no_conflict_or_drop(tmp_path):
    report = check_salvage(make_variant(tmp_path, corrected=True))
    assert report['episodes']['keep'] == [[0, 49]] and report['episodes']['drop'] == []
    assert not any(f['code'] in ('salvage.metadata_row_conflict', 'salvage.metadata_rows_repairable')
                   for f in report['findings'])


def test_partial_read_never_approves_exclusion(tmp_path):
    root = make_variant(tmp_path)
    report = check_salvage(root, max_rows=5000)
    assert report['status'] == 'unknown'
    assert report['plan']['strategy'] == 'unknown'
    assert not report['episodes'].get('drop')


def test_mcp_preview_retains_the_withheld_plan_reason_among_repeated_errors(tmp_path):
    report = check_salvage(make_variant(tmp_path, 'totals'))
    errors = [f for f in report['findings'] if f['status'] == 'error']
    other = [f for f in report['findings'] if f['status'] != 'error']
    report['findings'] = errors * 20 + other
    report['counts']['findings']['error'] = len(errors) * 20
    before = deepcopy(report)
    view = concise_salvage(report)
    assert view['findings_shown'][0]['code'] == 'salvage.metadata_row_conflict'
    assert view['counts']['error'] == len(errors) * 20 and view['status'] == 'error'
    assert view['episodes']['drop_count'] == 0 and view['episodes']['unresolved_count'] == 32
    assert view['plan']['strategy'] == 'unknown'
    assert len(json.dumps(view, indent=2).encode()) <= 8192
    assert report == before


@pytest.mark.parametrize('machine', [False, True])
def test_pointer_only_repairs_are_reachable_from_default_check(tmp_path, capsys, machine):
    root = make_variant(tmp_path, corrected=True)
    path = root / META
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[18]['data/file_index'] = 0
    rows[36]['data/file_index'] = 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    status = product_cli.main(['check', str(root), *(['--json'] if machine else [])])
    text = capsys.readouterr().out
    assert status == 1
    if machine:
        report = json.loads(text)
        assert [r['episode_index'] for r in report['plan']['pointer_corrections']] == [18, 36]
    else:
        assert '2 data-file pointer' in text
