"""Salvage decisions for interrupted native LeRobot v3 recordings, on byte-level fixtures that reproduce the
on-disk signatures observed natively with the pinned LeRobot writer (see the I4 diagnosis). Tests named
``test_d<N>_...`` are regressions for the independent validator's defects D1-D10 on b3404bbd; ``test_r3_<item>_...``
are regressions for its recheck3 findings on aa66bfb3 (B1, D2-D5)."""
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.salvage import check_salvage

CAM = 'observation.images.cam'


def rows_for(episodes, start=0, frame_start=0):
    rows, index = [], start
    for ep, length in episodes:
        for frame in range(frame_start, frame_start + length):
            rows.append({'index': index, 'episode_index': ep, 'frame_index': frame, 'timestamp': frame / 10,
                         'task_index': 0, 'action': [float(ep), float(frame)]})
            index += 1
    return rows


def meta_for(episodes, start=0, data_file=0, meta_file=0, video=False):
    out, position = [], start
    for ep, length in episodes:
        item = {'episode_index': ep, 'length': length, 'dataset_from_index': position,
                'dataset_to_index': position + length, 'data/chunk_index': 0, 'data/file_index': data_file,
                'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': meta_file}
        if video:
            item.update({f'videos/{CAM}/chunk_index': 0, f'videos/{CAM}/file_index': 0,
                         f'videos/{CAM}/from_timestamp': position / 10, f'videos/{CAM}/to_timestamp': (position + length) / 10})
        out.append(item)
        position += length
    return out


def write(root, data_files, meta_files, total_episodes, total_frames, *, video=False, version='v3.0', total_tasks=1,
          video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'):
    (root / 'meta').mkdir(parents=True)
    features = {'action': {'dtype': 'float32', 'shape': [2]}, 'index': {'dtype': 'int64', 'shape': [1]}}
    if video:
        features[CAM] = {'dtype': 'video', 'shape': [8, 8, 3], 'names': ['height', 'width', 'channels']}
    info = {'codebase_version': version, 'total_episodes': total_episodes, 'total_frames': total_frames, 'fps': 10,
            'total_tasks': total_tasks, 'chunks_size': 1000, 'splits': {'train': f'0:{total_episodes}'},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': video_path if video else None, 'features': features}
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    committed = [row for shard in data_files for row in shard][:total_frames]
    if committed:
        values = np.asarray([row['action'] for row in committed], dtype=np.float32)
        saved = {'action': {'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(),
                           'min': values.min(0).tolist(), 'max': values.max(0).tolist(),
                           'count': [len(values)]}}
    else:
        saved = {'action': {'mean': [0.0, 0.0], 'std': [1.0, 1.0]}}
    (root / 'meta' / 'stats.json').write_text(json.dumps(saved))
    pq.write_table(pa.table({'task_index': [0], 'task': ['t']}), root / 'meta' / 'tasks.parquet')
    for number, rows in enumerate(data_files):
        path = root / 'data' / 'chunk-000' / f'file-{number:03d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=max(1, len(rows)))
    for number, meta in enumerate(meta_files):
        path = root / 'meta' / 'episodes' / 'chunk-000' / f'file-{number:03d}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(meta), path)
    return root


def healthy(tmp_path, episodes=((0, 5), (1, 6), (2, 7)), **kwargs):
    total = sum(length for _, length in episodes)
    return write(tmp_path / 'ds', [rows_for(episodes)], [meta_for(episodes, video=kwargs.get('video', False))],
                 len(episodes), total, **kwargs)


def strip_footer(path):
    """What an open ParquetWriter leaves after a hard kill: row groups written, footer never written."""
    raw = path.read_bytes()
    footer = int.from_bytes(raw[-8:-4], 'little')
    path.write_bytes(raw[:len(raw) - 8 - footer])


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def codes(result):
    return {f['code']: f['status'] for f in result['findings']}


def finding(result, code):
    return next(f for f in result['findings'] if f['code'] == code)


def copy_files(plan):
    """Every file a literal application of the plan writes: copied unchanged or rewritten."""
    return set(plan['files']['copy_unchanged']) | {r['file'] for r in plan['files']['rewrite']}


def test_complete_recording_needs_no_salvage(tmp_path):
    result = check_salvage(healthy(tmp_path))
    assert result['status'] == 'pass' and result['findings'] == []
    assert result['plan']['strategy'] == 'none_needed' and result['episodes']['keep'] == [[0, 2]]
    assert result['kind'] == 'recording_salvage' and result['schema_version'] == 1
    assert result['coverage']['complete'] is True and result['counts']['keep'] == 3


def test_d1_every_plan_copies_or_rewrites_info_json(tmp_path):
    none_needed = check_salvage(healthy(tmp_path / 'a'))['plan']
    assert none_needed['strategy'] == 'none_needed' and 'meta/info.json' in none_needed['files']['copy_unchanged']
    orphan = write(tmp_path / 'b', [rows_for([(0, 5), (1, 6), (2, 7)])], [meta_for([(0, 5), (1, 6)])], 2, 11)
    keep_as_is = check_salvage(orphan)['plan']
    assert keep_as_is['strategy'] == 'keep_as_is' and 'meta/info.json' in copy_files(keep_as_is)
    meta = meta_for([(0, 5), (1, 6)])
    meta[1]['data/file_index'] = 1
    pointer_only = check_salvage(write(tmp_path / 'c', [rows_for([(0, 5), (1, 6)])], [meta], 2, 11))['plan']
    assert pointer_only['strategy'] == 'metadata_copy' and 'meta/info.json' in pointer_only['files']['copy_unchanged']
    for plan in (none_needed, keep_as_is, pointer_only):
        assert {'meta/tasks.parquet', 'meta/stats.json'} <= copy_files(plan)


def test_orphan_tail_rows_from_an_interrupted_save_are_dropped_and_resume_is_refused(tmp_path):
    # Native I2/I3/I7: rows of episode 2 were written, its metadata and totals were not.
    root = write(tmp_path / 'ds', [rows_for([(0, 5), (1, 6), (2, 7)])], [meta_for([(0, 5), (1, 6)])], 2, 11)
    before = hashes(root)
    result = check_salvage(root)
    found = codes(result)
    assert found['salvage.episodes_dropped'] == 'warning'
    assert found['salvage.metadata_plan'] == 'warning' and found['salvage.resume_hazard'] == 'error'
    assert result['status'] == 'error' and result['episodes']['keep'] == [[0, 1]]
    drop = result['episodes']['drop'][0]
    assert drop['episode_index'] == 2 and drop['reasons'] == ['rows_without_metadata'] and drop['rows'] == 7
    assert drop['evidence'][0]['pointer'] == 'row 11/episode_index' and len(drop['evidence'][0]['sha256']) == 64
    dropped = finding(result, 'salvage.episodes_dropped')
    assert 'did not finish' not in dropped['message'] and 'partial' not in dropped['remedy']
    assert 'are recoverable' in dropped['message']  # one complete run of rows; this fixture has no video feature
    plan = result['plan']
    assert plan['strategy'] == 'keep_as_is' and plan['corrected_info']['total_frames'] == 11
    assert plan['changes']['rows_left_unreferenced_in_copied_files'] == 7
    assert plan['validity']['resume_safe'] is False
    hazard = finding(result, 'salvage.resume_hazard')
    assert '7 readable data row(s) lie after the last metadata span' in hazard['message']
    assert 'dataset_writer.py:157,287' in hazard['message']
    assert hashes(root) == before


def test_hard_kill_footerless_data_and_unflushed_metadata(tmp_path):
    # Native K1: footer-less data file, no meta/episodes, info records 2 committed, staging for episode 2.
    root = write(tmp_path / 'ds', [rows_for([(0, 5), (1, 6)])], [], 2, 11)
    strip_footer(root / 'data' / 'chunk-000' / 'file-000.parquet')
    staging = root / 'images' / CAM / 'episode-000002'
    staging.mkdir(parents=True)
    for frame in range(3):
        (staging / f'frame-{frame:06d}.png').write_bytes(b'png')
    before = hashes(root)
    result = check_salvage(root)
    found = codes(result)
    assert found['salvage.file_unreadable'] == 'error'
    assert found['salvage.episode_metadata_missing'] == 'error' and found['salvage.leftovers'] == 'warning'
    assert 'salvage.resume_hazard' not in found and result['status'] == 'error'
    unreadable = finding(result, 'salvage.file_unreadable')
    assert unreadable['files'] == ['data/chunk-000/file-000.parquet']
    assert unreadable['evidence'][0]['value']['head_magic'] is True and unreadable['evidence'][0]['value']['tail_magic'] is False
    missing = finding(result, 'salvage.episode_metadata_missing')
    assert missing['committed_according_to_info'] == {'total_episodes': 2, 'total_frames': 11}
    assert 'dataset_metadata.py:144-151' in missing['message'] and 'dataset_metadata.py:637-641' in missing['message']
    leftovers = finding(result, 'salvage.leftovers')
    assert leftovers['staging_episodes'] == [2] and 'Episode(s) [2] are at or beyond' in leftovers['message']
    assert result['plan']['strategy'] == 'not_metadata_only'
    assert result['episodes']['committed_but_not_verifiable'] == [[0, 1]]
    assert hashes(root) == before


def test_wording_footer_message_does_not_claim_the_file_is_never_named(tmp_path):
    root = healthy(tmp_path)
    strip_footer(root / 'data' / 'chunk-000' / 'file-000.parquet')
    message = finding(check_salvage(root), 'salvage.file_unreadable')['message']
    assert 'fails without naming the file' not in message and 'logs at ERROR level' in message


def test_footerless_metadata_and_data_after_a_flush(tmp_path):
    # Native K4: the metadata buffer was flushed once, so both writers were open at the kill.
    root = healthy(tmp_path)
    strip_footer(root / 'data' / 'chunk-000' / 'file-000.parquet')
    strip_footer(root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    result = check_salvage(root)
    found = codes(result)
    assert found['salvage.file_unreadable'] == 'error' and found['salvage.not_metadata_only'] == 'error'
    assert finding(result, 'salvage.file_unreadable')['files'] == [
        'data/chunk-000/file-000.parquet', 'meta/episodes/chunk-000/file-000.parquet']
    assert 'salvage.episode_metadata_missing' not in found and result['plan']['strategy'] == 'not_metadata_only'


def test_footerless_only_shard_with_readable_metadata_names_the_unreadable_file(tmp_path):
    root = healthy(tmp_path)
    strip_footer(root / 'data' / 'chunk-000' / 'file-000.parquet')
    result = check_salvage(root)
    assert {d['episode_index']: d['reasons'] for d in result['episodes']['drop']} == {
        0: ['rows_in_unreadable_file'], 1: ['rows_in_unreadable_file'], 2: ['rows_in_unreadable_file']}
    assert result['plan']['strategy'] == 'not_metadata_only'
    assert codes(result)['salvage.resume_hazard'] == 'error'  # resume reads no data file, so it proceeds


def test_long_recording_keeps_closed_shards_and_omits_the_open_one(tmp_path):
    # Earlier shards were closed at rollover; the kill left only the last data shard footer-less.
    first, second = [(0, 5), (1, 6)], [(2, 4), (3, 4)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for(second, start=11)], [meta_for(first)], 4, 19)
    strip_footer(root / 'data' / 'chunk-000' / 'file-001.parquet')
    result = check_salvage(root)
    plan = result['plan']
    assert plan['strategy'] == 'metadata_copy' and result['episodes']['keep'] == [[0, 1]]
    assert plan['corrected_info'] == {'total_episodes': 2, 'total_frames': 11, 'total_tasks': 1,
                                      'splits': {'train': '0:2'}}
    assert 'data/chunk-000/file-001.parquet' in plan['files']['omit']
    assert 'data/chunk-000/file-000.parquet' in plan['files']['copy_unchanged']
    assert plan['validity']['resume_safe'] is True
    assert result['episodes']['committed_but_not_verifiable'] == [[2, 3]]
    assert codes(result)['salvage.metadata_plan'] == 'error'
    # Resume reads no data file, so it proceeds here (native K11) and would replace the existing next shard.
    hazard = finding(result, 'salvage.resume_hazard')
    assert 'footer-less data file(s) remain' in hazard['message']
    assert 'data/chunk-000/file-001.parquet, which already exists' in hazard['message']


def test_d6_committed_episodes_whose_metadata_was_not_flushed_are_recoverable_not_unfinished(tmp_path):
    # Native K11: shards 000-001 closed, info counts 4 episodes, only episodes 0-1 reached a metadata file.
    first, second = [(0, 5), (1, 6)], [(2, 4), (3, 4)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for(second, start=11)], [meta_for(first)], 4, 19)
    result = check_salvage(root)
    assert {d['episode_index']: d['reasons'] for d in result['episodes']['drop']} == {
        2: ['metadata_not_flushed'], 3: ['metadata_not_flushed']}
    dropped = finding(result, 'salvage.episodes_dropped')
    assert 'are committed' in dropped['message'] and 'did not finish' not in dropped['message']
    assert 'recoverable in a separate copy by regenerating' in dropped['remedy'] and 'lost' not in dropped['remedy']


def test_l1_recovery_from_video_files_states_its_basis_and_never_claims_header_or_file_counts(tmp_path):
    # As above with an unbatched videos/ directory: the frames of episodes 2-3 are in the video files only by the
    # recorder's write order; nothing locates them, so the message must say that, not cite header or file counts.
    first, second = [(0, 5), (1, 6)], [(2, 4), (3, 4)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for(second, start=11)], [meta_for(first, video=True)], 4, 19,
                 video=True)
    video = root / 'videos' / CAM / 'chunk-000' / 'file-000.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'\x00' * 64)
    dropped = finding(check_salvage(root), 'salvage.episodes_dropped')
    assert 'Episode(s) [[2, 3]] are recoverable' in dropped['message']
    assert f'in the video files under videos/{CAM} (not located per episode)' in dropped['message']
    assert 'That rests, for video files, on info.json counting the episode as saved' in dropped['message']
    assert 'where their frames sit in those files was not read' in dropped['message']
    assert 'dataset_writer.py:328-378' in dropped['message'] and 'dataset_writer.py:380-386' in dropped['message']
    assert 'encoder temp MP4s' not in dropped['message'] and 'count of frame files' not in dropped['message']
    assert 'look complete' not in dropped['message'] and 'no frame was decoded or opened' in dropped['message']
    assert 'record that video file and timestamps' in dropped['remedy'] and 'Verify the frames decode' in dropped['remedy']


def test_truncated_stats_json_is_an_unreadable_file(tmp_path):
    # Native D1: the disk filled during the in-place stats.json rewrite.
    root = healthy(tmp_path)
    stats = root / 'meta' / 'stats.json'
    stats.write_bytes(stats.read_bytes()[:20])
    result = check_salvage(root)
    unreadable = finding(result, 'salvage.file_unreadable')
    assert unreadable['files'] == ['meta/stats.json'] and unreadable['evidence'][0]['value']['problem'] == 'invalid_json'
    assert 'io_utils.py:48-50' in unreadable['message'] and result['status'] == 'error'


def test_d10_unreadable_stats_or_tasks_require_a_rebuild_not_a_copy(tmp_path):
    stats_root = healthy(tmp_path / 's')
    (stats_root / 'meta' / 'stats.json').write_bytes(b'{"action": {"mean": [0')
    plan = check_salvage(stats_root)['plan']
    assert plan['strategy'] == 'rebuild_required' and [r['file'] for r in plan['rebuild']] == ['meta/stats.json']
    assert 'meta/stats.json' not in copy_files(plan) and plan['validity']['resume_safe'] is False
    tasks_root = healthy(tmp_path / 't')
    strip_footer(tasks_root / 'meta' / 'tasks.parquet')
    result = check_salvage(tasks_root)
    plan = result['plan']
    assert plan['strategy'] == 'rebuild_required' and [r['file'] for r in plan['rebuild']] == ['meta/tasks.parquet']
    assert 'meta/tasks.parquet' not in copy_files(plan) and plan['validity']['resume_safe'] is False
    assert codes(result)['salvage.metadata_plan'] == 'error'
    assert 'LeRobot already reads' not in finding(result, 'salvage.metadata_plan')['message']


def test_d10_a_copy_without_tasks_or_stats_never_passes(tmp_path):
    no_tasks = healthy(tmp_path / 't')
    (no_tasks / 'meta' / 'tasks.parquet').unlink()
    result = check_salvage(no_tasks)
    assert codes(result)['salvage.tasks_unusable'] == 'error' and result['status'] == 'error'
    no_stats = healthy(tmp_path / 's')
    (no_stats / 'meta' / 'stats.json').unlink()
    assert codes(check_salvage(no_stats)) == {'salvage.stats_missing': 'warning'}


def test_truncated_info_json_stops_with_an_error_not_a_plan(tmp_path):
    root = healthy(tmp_path)
    info = root / 'meta' / 'info.json'
    info.write_bytes(info.read_bytes()[:40])
    result = check_salvage(root)
    assert codes(result) == {'salvage.file_unreadable': 'error'} and result['plan']['strategy'] == 'not_metadata_only'


def test_unflushed_metadata_with_readable_rows_cannot_be_kept_metadata_only(tmp_path):
    # Native I6/B0: finalize raised before the metadata buffer was written.
    root = write(tmp_path / 'ds', [rows_for([(0, 5), (1, 6)])], [], 2, 11)
    result = check_salvage(root)
    found = codes(result)
    assert found['salvage.episode_metadata_missing'] == 'error' and found['salvage.episodes_dropped'] == 'warning'
    assert {d['episode_index']: d['reasons'] for d in result['episodes']['drop']} == {
        0: ['metadata_not_flushed'], 1: ['metadata_not_flushed']}
    assert 'No readable episode-metadata row exists at all' in finding(result, 'salvage.episodes_dropped')['message']
    assert result['plan']['strategy'] == 'not_metadata_only'


def test_d7_d8_zero_totals_skip_metadata_and_resume_replaces_file_000(tmp_path):
    # Census 012 shape: info.json says 0/0 (and total_tasks 0) while every episode is complete.
    episodes = [(0, 5), (1, 6), (2, 7)]
    root = write(tmp_path / 'ds', [rows_for(episodes)], [meta_for(episodes)], 0, 0, total_tasks=0)
    result = check_salvage(root)
    found = codes(result)
    assert found['salvage.metadata_plan'] == 'error' and found['salvage.resume_hazard'] == 'error'
    assert found['salvage.tasks_unusable'] == 'error'
    plan = result['plan']
    assert plan['corrected_info'] == {'total_episodes': 3, 'total_frames': 18, 'total_tasks': 1,
                                      'splits': {'train': '0:3'}}
    assert plan['files']['rewrite'][0] == {'file': 'meta/info.json', 'set': plan['corrected_info']}
    hazard = finding(result, 'salvage.resume_hazard')['message']
    assert 'skips episode metadata when total_episodes is 0' in hazard and 'dataset_metadata.py:221-222' in hazard
    assert 'data/chunk-000/file-000.parquet, which already exists' in hazard
    assert 'collide' not in hazard
    assert 'serves an empty dataset' in finding(result, 'salvage.metadata_plan')['message']


def test_second_session_restarting_labels_does_not_authorize_discarding_a_history(tmp_path):
    # Census 016 shape: a last session restarted episode and global indices while its spans continued.
    first, second = [(0, 5), (1, 6), (2, 7)], [(0, 4), (1, 4)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for(second, start=0)],
                 [meta_for(first), meta_for(second, start=18, data_file=1, meta_file=1)], 0, 0)
    result = check_salvage(root)
    plan = result['plan']
    assert result['episodes']['keep'] == [[0, 2]] and result['episodes']['drop'] == []
    # Two histories are evidenced; their precedence is not. A prefix is useful
    # evidence but cannot authorize silently dropping the later intact file.
    assert plan['strategy'] == 'unknown' and plan['validity']['resume_safe'] is False
    assert 'files' not in plan
    assert codes(result)['salvage.metadata_row_conflict'] == 'error'
    assert 'salvage.metadata_plan' not in codes(result)


def test_divergent_history_preserves_physical_prefix_without_approving_conflicted_tail(tmp_path):
    # Census 020 shape: a second history re-records episodes 1-3 from a stale state.
    first, second = [(0, 5), (1, 5), (2, 5)], [(1, 5), (2, 5), (3, 5)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for(second, start=5)],
                 [meta_for(first), meta_for(second, start=5, data_file=1, meta_file=1)], 3, 15)
    result = check_salvage(root)
    assert result['episodes']['keep'] == [[0, 2]]
    assert result['episodes']['drop'] == [] and result['episodes']['unresolved'] == [[3, 3]]
    assert result['plan']['strategy'] == 'unknown'
    assert codes(result)['salvage.row_identity'] == 'error'
    assert codes(result)['salvage.resume_hazard'] == 'error'


def test_resume_onto_orphan_rows_is_ambiguous_not_a_pointer_fix(tmp_path):
    # Native I2 + --resume: the resumed episode 2 lives in file-001, but its span lands on the orphan rows of the
    # interrupted episode 2 in file-000, whose identity columns look identical.
    old = rows_for([(0, 5), (1, 6), (2, 7)])
    new = rows_for([(2, 4), (3, 4)], start=11)
    meta = meta_for([(0, 5), (1, 6)]) + meta_for([(2, 4), (3, 4)], start=11, data_file=1)
    root = write(tmp_path / 'ds', [old, new], [meta], 4, 19)
    result = check_salvage(root)
    assert result['episodes']['keep'] == [[0, 1]]
    assert result['episodes']['drop'] == [] and result['episodes']['unresolved'] == [[2, 3]]
    assert result['plan']['strategy'] == 'unknown' and not result['plan'].get('pointer_corrections')
    assert codes(result)['salvage.resume_hazard'] == 'error' and result['status'] == 'error'


@pytest.mark.parametrize('orphan_length', [7, 4])
def test_d2_aggregate_of_an_interrupted_dataset_in_one_file_is_ambiguous(tmp_path, orphan_length):
    # Validator AGG_AB: aggregate_datasets([A with orphan rows of episode 2, B]) appends B's first episode, relabelled
    # 2 with index 11.., right after A's orphan rows in the same file. Identity columns and pointer match the span.
    rows = rows_for([(0, 5), (1, 6)]) + rows_for([(2, orphan_length)], start=11) + rows_for([(2, 4)], start=11)
    meta = meta_for([(0, 5), (1, 6), (2, 4)])
    result = check_salvage(write(tmp_path / 'ds', [rows], [meta], 3, 15))
    assert result['episodes']['keep'] == [[0, 1]]
    assert {d['episode_index']: d['reasons'] for d in result['episodes']['drop']} == {2: ['ambiguous_rows']}
    assert result['plan']['strategy'] != 'keep_as_is' and result['status'] == 'error'


def test_d3_complete_episodes_after_a_dropped_one_need_renumbering_and_are_not_lost(tmp_path):
    episodes = [(0, 5), (1, 5), (2, 5), (3, 5)]
    rows = rows_for(episodes)
    rows[12]['frame_index'] = 9  # episode 2 rows out of frame order
    root = write(tmp_path / 'ds', [rows], [meta_for(episodes)], 4, 20)
    result = check_salvage(root)
    assert result['episodes']['keep'] == [[0, 1]] and result['episodes']['need_renumbering'] == [[3, 3]]
    assert {d['episode_index']: d['reasons'] for d in result['episodes']['drop']} == {2: ['span_rows_mismatch']}
    assert result['episodes']['drop'][0]['evidence'][1]['pointer'] == 'row 12/episode_index'
    renumber = finding(result, 'salvage.episodes_need_renumbering')
    assert renumber['status'] == 'warning' and 'not lost' in renumber['message']
    assert 'serves them correctly' in renumber['message']
    assert '3' not in str(finding(result, 'salvage.episodes_dropped')['episodes'])


def test_wrong_data_file_pointer_is_corrected_metadata_only(tmp_path):
    # Census 047 shape: rows present, pointer names a file that does not exist.
    episodes = [(0, 5), (1, 6), (2, 7)]
    meta = meta_for(episodes)
    meta[1]['data/file_index'] = 1
    root = write(tmp_path / 'ds', [rows_for(episodes)], [meta], 3, 18)
    result = check_salvage(root)
    assert codes(result) == {'salvage.metadata_plan': 'warning'}
    fix = result['plan']['pointer_corrections'][0]
    assert fix['episode_index'] == 1 and fix['to'] == {'data/chunk_index': 0, 'data/file_index': 0}
    assert result['plan']['files']['rewrite'] == [{'file': 'meta/episodes/chunk-000/file-000.parquet',
        'keep_physical_rows': [0, 2], 'drop_physical_rows': 0, 'pointer_corrections': 1}]


def test_d9_resume_safety_uses_the_corrected_pointers(tmp_path):
    # Validator M2: the data file is split inside episode 1; episode 2's metadata still names file-000.
    episodes = [(0, 5), (1, 6), (2, 7)]
    rows = rows_for(episodes)
    root = write(tmp_path / 'ds', [rows[:8], rows[8:]], [meta_for(episodes)], 3, 18)
    plan = check_salvage(root)['plan']
    assert [f['episode_index'] for f in plan['pointer_corrections']] == [2]
    assert plan['validity']['resume_safe'] is True and plan['validity']['resume_blockers'] == []


def test_staging_for_the_next_index_and_encoder_temp_dirs_are_reported(tmp_path):
    root = healthy(tmp_path)
    staging = root / 'images' / CAM / 'episode-000003'
    staging.mkdir(parents=True)
    (staging / 'frame-000000.png').write_bytes(b'x')
    temp = root / 'tmpab12cd_9'
    temp.mkdir()
    (temp / f'{CAM}_003.mp4').write_bytes(b'\x00\x00\x00\x08free')
    result = check_salvage(root)
    assert codes(result) == {'salvage.leftovers': 'warning'}  # served frames stay correct (native H4); no resume hazard
    leftovers = finding(result, 'salvage.leftovers')
    assert [Path(d).name for d in leftovers['encoder_temp_dirs']] == [temp.name]
    assert any(e['pointer'] == 'encoder temp file' and len(e['sha256']) == 64 for e in leftovers['evidence'])
    assert 'video_utils.py:529-532' in leftovers['message'] and 'Frames left for episode 3' in leftovers['message']
    assert result['plan']['strategy'] == 'none_needed'


def test_existing_next_shard_would_be_replaced_by_resume(tmp_path):
    episodes = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(episodes), rows_for([(2, 3)], start=11)], [meta_for(episodes)], 2, 11)
    result = check_salvage(root)
    assert 'data/chunk-000/file-001.parquet, which already exists' in finding(result, 'salvage.resume_hazard')['message']
    assert result['plan']['validity']['resume_safe'] is True  # the orphan-only shard is omitted from the copy


def test_video_not_present_locally_is_unknown_never_pass(tmp_path):
    result = check_salvage(healthy(tmp_path, video=True))
    assert codes(result) == {'salvage.video_unchecked': 'unknown'} and result['status'] == 'unknown'
    assert result['coverage']['complete'] is False
    assert result['plan']['strategy'] == 'none_needed_rows_only' and result['plan']['validity']['video_verified'] is False


def test_wording_truncated_video_is_named_and_the_plan_does_not_say_none_needed(tmp_path):
    root = healthy(tmp_path, video=True)
    video = root / 'videos' / CAM / 'chunk-000' / 'file-000.mp4'
    video.parent.mkdir(parents=True)
    ftyp = (20).to_bytes(4, 'big') + b'ftypisom' + (512).to_bytes(4, 'big') + b'isom'
    video.write_bytes(ftyp + (4000).to_bytes(4, 'big') + b'mdat' + b'\x00' * 100)  # mdat claims more than the file holds
    result = check_salvage(root)
    truncated = finding(result, 'salvage.video_file_truncated')
    assert truncated['status'] == 'error' and truncated['episodes'] == [[0, 2]]
    assert 'shorter than their own MP4 boxes declare' in truncated['message']
    assert result['plan']['strategy'] != 'none_needed' and result['plan']['validity']['video_verified'] is False
    assert 'unsupported MP4 layout' not in finding(result, 'salvage.video_unchecked')['message']


def test_camera_without_any_location_columns_is_a_feature_decision_not_episode_drops(tmp_path):
    # Census 012 shape: a second declared camera has no videos/<key>/* column at all.
    root = healthy(tmp_path, video=True)
    info = json.loads((root / 'meta' / 'info.json').read_text())
    info['features']['observation.images.side'] = dict(info['features'][CAM])
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    result = check_salvage(root)
    assert codes(result)['salvage.feature_unlocatable'] == 'error' and result['status'] == 'error'
    assert result['episodes']['keep'] == [[0, 2]] and result['episodes']['drop'] == []
    assert result['plan']['validity']['blocked_until_features_restored_or_removed'] == ['observation.images.side']


def test_missing_video_file_in_an_existing_camera_directory_drops_that_episode(tmp_path):
    episodes = [(0, 5), (1, 6), (2, 7)]
    meta = meta_for(episodes, video=True)
    meta[2][f'videos/{CAM}/file_index'] = 1
    root = write(tmp_path / 'ds', [rows_for(episodes)], [meta], 3, 18, video=True)
    (root / 'videos' / CAM / 'chunk-000').mkdir(parents=True)
    (root / 'videos' / CAM / 'chunk-000' / 'file-000.mp4').write_bytes(b'not parsed')
    result = check_salvage(root)
    assert result['episodes']['keep'] == [[0, 1]]
    assert result['episodes']['drop'][0]['reasons'] == ['video_file_missing']


@pytest.mark.parametrize('template', ['videos/{video_key}/{nope}.mp4', 'videos/{video_key}/{chunk_index:03d}/{0}.mp4'])
def test_d4_malformed_video_path_template_is_unknown_not_a_crash(tmp_path, template):
    result = check_salvage(healthy(tmp_path, video=True, video_path=template))
    assert codes(result) == {'salvage.coverage_unknown': 'unknown'} and result['plan']['strategy'] == 'unknown'
    assert 'video_path' in result['findings'][0]['message']


def test_d5_video_path_escaping_the_root_never_reaches_a_plan(tmp_path):
    template = '../../outside/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
    result = check_salvage(healthy(tmp_path, video=True, video_path=template))
    assert codes(result) == {'salvage.coverage_unknown': 'unknown'} and result['plan']['strategy'] == 'unknown'
    assert 'files' not in result['plan']  # no plan path is derived from the escaping template


def stage(root, episode, frames):
    directory = root / 'images' / CAM / f'episode-{episode:06d}'
    directory.mkdir(parents=True)
    for frame in range(frames):
        (directory / f'frame-{frame:06d}.png').write_bytes(b'\x89PNG')
    return directory


@pytest.mark.parametrize('frames, recoverable', [(7, True), (3, False)])
def test_w1_orphan_rows_with_complete_camera_frames_are_recoverable_not_partial(tmp_path, frames, recoverable):
    # Native I7 shape: rows of episode 2 remain without metadata; its camera frames survive outside videos/.
    episodes = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(episodes + [(2, 7)])], [meta_for(episodes, video=True)], 2, 11, video=True)
    stage(root, 2, frames)
    result = check_salvage(root)
    dropped, leftovers = finding(result, 'salvage.episodes_dropped'), finding(result, 'salvage.leftovers')
    assert result['episodes']['drop'][0]['reasons'] == ['rows_without_metadata']
    assert 'did not finish' not in dropped['message'] and 'partial' not in dropped['remedy']
    assert 'no readable metadata or rows describe it' not in leftovers['remedy']
    if recoverable:
        assert 'Episode(s) [[2, 2]] are recoverable' in dropped['message'] and 'staging frames' in dropped['message']
        assert 'no frame was decoded or opened' in dropped['message']  # L1/L2: basis is names, header counts, file counts
        assert 'That rests, for staging frames, on the count of frame files; no frame was decoded or opened' in dropped['message']
        assert '(7 frame files)' in dropped['message']
        assert 'encoder temp MP4s' not in dropped['message'] and 'video files' not in dropped['message']
        assert 'Verify the frames decode' in dropped['remedy'] and 'verify the frames decode' in leftovers['remedy']
        assert leftovers['remedy'].startswith('Keep images/') and 'episode(s) [[2, 2]]' in leftovers['message']
        assert 'look complete' in leftovers['message'] and 'not decoded' in leftovers['message']
    else:
        assert 'no complete encoder temp MP4 or staging frames' in dropped['message'] and 'are recoverable' not in dropped['message']
        assert 'whether a video file holds their frames cannot be read' in dropped['message']
        assert leftovers['remedy'].startswith('Leave images/')


def test_w1_rows_from_another_writer_get_no_recorder_cause(tmp_path):
    # Census 033 shape: one row group for every episode, so the pinned recorder did not write the file.
    root = write(tmp_path / 'ds', [rows_for([(0, 5), (1, 6), (2, 7)])], [meta_for([(0, 5), (1, 6)])], 2, 11)
    message = finding(check_salvage(root), 'salvage.episodes_dropped')['message']
    assert 'another tool wrote them and these causes may not apply' in message and 'did not finish' not in message


def test_w3_staging_frames_are_the_only_camera_frames_when_no_video_was_written(tmp_path):
    # Native B0/I6: batched encoding never wrote videos/, metadata was not flushed, staging PNGs are complete.
    root = write(tmp_path / 'ds', [rows_for([(0, 5), (1, 6)])], [], 2, 11, video=True)
    stage(root, 0, 5)
    stage(root, 1, 6)
    result = check_salvage(root)
    leftovers = finding(result, 'salvage.leftovers')
    assert 'these staging frames are the only camera frames' in leftovers['message']
    assert leftovers['remedy'].startswith('Keep images/') and 'Do not copy' not in leftovers['remedy']
    dropped = finding(result, 'salvage.episodes_dropped')
    assert 'encode the complete staging frames' in dropped['remedy']
    assert 'staging frames under images/ are the only camera frames' in finding(
        result, 'salvage.episode_metadata_missing')['remedy']


def test_g1_task_table_shorter_than_the_rows_task_indices_never_passes(tmp_path):
    # Validator TASKS_EMPTY: tasks.parquet has 0 rows, total_tasks 1; natively every sample fails (IndexError).
    root = healthy(tmp_path)
    pq.write_table(pa.table({'task_index': pa.array([], pa.int64()), 'task': pa.array([], pa.string())}),
                   root / 'meta' / 'tasks.parquet')
    result = check_salvage(root)
    assert codes(result)['salvage.tasks_unusable'] == 'error' and result['status'] == 'error'
    plan = result['plan']
    assert plan['strategy'] == 'rebuild_required' and 'meta/tasks.parquet' not in copy_files(plan)
    assert 'total_tasks' not in plan['corrected_info'] and plan['validity']['resume_safe'] is False


@pytest.mark.parametrize('video_headers', [True, False])
def test_g2_video_file_symlinked_outside_the_root_is_refused(tmp_path, video_headers):
    root = healthy(tmp_path, video=True)
    video = root / 'videos' / CAM / 'chunk-000' / 'file-000.mp4'
    video.parent.mkdir(parents=True)
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'\x00' * 32)
    video.symlink_to(outside)
    with pytest.raises(InputError, match='symlink|escape'):
        check_salvage(root, video_headers=video_headers)


def test_n1_headers_off_never_claims_temp_mp4s_hold_no_frames(tmp_path):
    # Validator I7_noheaders: a temp MP4 named for the orphan episode exists but is not inspected without header reads.
    episodes = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(episodes + [(2, 7)])], [meta_for(episodes, video=True)], 2, 11, video=True)
    temp = root / 'tmpab12cd_9'
    temp.mkdir()
    (temp / f'{CAM}_002.mp4').write_bytes(b'\x00' * 64)
    result = check_salvage(root, video_headers=False)
    dropped, leftovers = finding(result, 'salvage.episodes_dropped'), finding(result, 'salvage.leftovers')
    assert 'MP4 headers were not read' in dropped['message'] and 'is unknown' in dropped['message']
    assert 'were found here' not in dropped['message'] and 'rerun with MP4 header reads' in dropped['remedy']
    assert leftovers['remedy'].startswith('Keep images/ and the tmp* directories with the original')
    assert 'not inspected' in leftovers['remedy'] and 'none of them holds' not in leftovers['remedy']
    assert 'not inspected' in leftovers['message']


def outside_videos(tmp_path):
    root = healthy(tmp_path, video=True)
    disk = tmp_path / 'disk2'
    video = disk / 'videos' / CAM / 'chunk-000' / 'file-000.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'\x00' * 64)
    (root / 'videos').symlink_to(disk / 'videos')
    return root, disk


def refusal(root, **kwargs):
    with pytest.raises(InputError) as refused:
        check_salvage(root, **kwargs)
    return str(refused.value)


def link_copy(link):
    """The remedy command for one refused link: a real copy of its target replaces that link only."""
    temporary = shlex.quote(str(link) + '.tmp')
    return (f'test ! -e {temporary} && test ! -L {temporary} && '
            f'cp -rP -- {shlex.quote(str(link.resolve()))} {temporary} && '
            f'rm -- {shlex.quote(str(link))} && mv -T -- {temporary} {shlex.quote(str(link))}')


@pytest.mark.parametrize('video_headers', [True, False])
def test_u1_videos_on_another_disk_name_the_link_and_the_remedy(tmp_path, video_headers):
    root, disk = outside_videos(tmp_path)
    message = refusal(root, video_headers=video_headers)
    assert message.startswith(f'Refused before reading: videos is a symlink to {(disk / "videos").resolve()}')
    assert f'Replace the link with a real copy of its target, which LeRobot reads as the same bytes (`{link_copy(root.resolve() / "videos")}`)' in message
    assert 'extra_roots' not in message and 'authorize' not in message and '<new-directory>' not in message


def test_r4_m2_the_remedy_copies_only_the_refused_link_and_then_the_check_runs(tmp_path):
    # Validator recheck4 M2: a whole-root `cp -rL` also followed root-level links the check never reads (cp exited 1 on
    # a cyclic one and copied outside data). The suggested command touches only the refused link.
    root, disk = outside_videos(tmp_path)
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'big.bin').write_bytes(b'\x00' * 1024)
    (root / 'cache_link').symlink_to(tmp_path / 'cache')
    (root / 'up_link').symlink_to(tmp_path)
    message = refusal(root)
    command = message.split('`')[1]
    assert command == link_copy(root.resolve() / 'videos')
    subprocess.run(['bash', '-c', command], check=True)
    assert not (root / 'videos').is_symlink() and (root / 'videos' / CAM / 'chunk-000' / 'file-000.mp4').is_file()
    assert (root / 'cache_link').is_symlink() and (root / 'up_link').is_symlink()  # untouched, never followed
    assert check_salvage(root)['episodes']['keep'] == [[0, 2]]  # the check now runs on the real copy


def test_r3_b1_extra_roots_is_not_offered_and_nothing_is_followed(tmp_path):
    # Recheck3 B1: links into "authorized" roots were recorded as followed but never read. The option is removed.
    root, disk = outside_videos(tmp_path)
    with pytest.raises(TypeError):
        check_salvage(root, extra_roots=[str(disk)])
    result = check_salvage(healthy(tmp_path / 'plain'))
    assert not {'followed_links', 'authorized_extra_roots'} & set(result['coverage'])
    assert 'followed_links' not in result['plan']['files'] and result['status'] == 'pass'
    assert 'authorized' not in result['scope'] and 'No symlink is followed' in result['scope']


@pytest.mark.parametrize('relative', ['data', 'data/chunk-000', 'data/chunk-000/file-000.parquet', 'meta',
                                      'meta/episodes', 'meta/episodes/chunk-000', 'meta/tasks.parquet', 'meta/stats.json'])
def test_r3_b1_any_linked_dataset_path_is_refused_naming_the_link(tmp_path, relative):
    # Recheck3 E8-E10/E20/E21: LeRobot follows these links; the check refuses them by name, never silently.
    root = healthy(tmp_path)
    original, moved = root / relative, tmp_path / 'disk2' / relative.replace('/', '_')
    moved.parent.mkdir(parents=True)
    original.rename(moved)
    original.symlink_to(moved)
    message = refusal(root)
    assert message.startswith(f'Refused before reading: {relative} is a symlink to {moved.resolve()}, outside')
    assert link_copy(root.resolve() / relative) in message and 'extra_roots' not in message


@pytest.mark.parametrize('target', ['file', 'inside', 'inside_parent', 'dangling', 'loop', 'parent', 'slash', 'fifo',
                                    'device'])
def test_r3_d3_link_refusals_suggest_only_actions_that_work_for_that_target(tmp_path, target):
    root = healthy(tmp_path)
    stats = root / 'meta' / 'stats.json'
    real = tmp_path / 'stats_elsewhere.json'
    stats.rename(real)
    if target == 'inside':
        real.rename(root / 'meta' / 'stats_real.json')
    elif target == 'fifo':
        real.unlink()
        os.mkfifo(real)
    pointed = {'file': real, 'inside': Path('stats_real.json'), 'inside_parent': Path('.'),
               'dangling': tmp_path / 'missing.json', 'loop': Path('stats.json'), 'parent': tmp_path, 'slash': Path('/'),
               'fifo': real, 'device': Path('/dev/null')}[target]
    stats.symlink_to(pointed)
    with deadline():
        message = refusal(root)
    assert message.startswith('Refused before reading: meta/stats.json is a symlink to ')
    assert 'extra_roots' not in message and 'authorize' not in message and '<new-directory>' not in message
    # Copying the link's target in its place works only for a regular file or a directory that does not contain it.
    if target in ('file', 'inside'):
        assert link_copy(root.resolve() / 'meta' / 'stats.json') in message
    else:
        assert '`test ! -e ' not in message
    expected = {'file': 'Replace the link with a real copy of its target', 'inside': 'inside the dataset root',
                'inside_parent': 'which contains the link itself', 'dangling': 'cannot be followed',
                'loop': 'cannot be followed', 'parent': 'contains the dataset itself', 'slash': 'contains the dataset itself',
                'fifo': 'a FIFO (named pipe), not a regular file or directory',
                'device': 'a character device, not a regular file or directory'}[target]
    assert expected in message


def test_r4_m1_a_file_argument_carried_by_a_link_into_another_dataset_is_refused(tmp_path):
    # Validator recheck4 M1: P1/meta/info.json linked into P2 used to check P2 silently (dataset_root alias rule).
    other = healthy(tmp_path / 'p2')
    linked_file = tmp_path / 'p1' / 'meta' / 'info.json'
    linked_file.parent.mkdir(parents=True)
    linked_file.symlink_to(other / 'meta' / 'info.json')
    message = refusal(linked_file)
    assert message.startswith(f'Refused before reading: the argument {linked_file} is a file reached through a symlink')
    assert f'would check {other.resolve()} instead of {tmp_path / "p1"}' in message
    (tmp_path / 'p3').mkdir()
    (tmp_path / 'p3' / 'meta').symlink_to(other / 'meta')  # the link is a parent of the file argument
    assert 'is a file reached through a symlink' in refusal(tmp_path / 'p3' / 'meta' / 'info.json')
    # Still accepted: the file in place, and a dataset directory reached through a link (the dataset named is checked).
    (tmp_path / 'alias').symlink_to(other)
    for argument in (other / 'meta' / 'info.json', tmp_path / 'alias', tmp_path / 'alias' / 'meta' / 'info.json'):
        result = check_salvage(argument)
        assert result['status'] == 'pass' and result['artifacts'][0]['path'].startswith(str(other.resolve()))


@pytest.mark.parametrize('video_headers', [True, False])
@pytest.mark.parametrize('where', ['camera', 'episode', 'frame', 'tempdir', 'tempfile'])
def test_r3_d2_links_in_the_leftover_scan_stop_the_check_by_name(tmp_path, where, video_headers):
    # Recheck3 D2/E14/E15/F4/B0_EPSYM: these links were skipped silently, losing "keep the only camera frames" advice.
    episodes = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(episodes + [(2, 7)])], [meta_for(episodes, video=True)], 2, 11, video=True)
    staged = stage(root, 2, 7)
    temp = root / 'tmpab12cd_9'
    temp.mkdir()
    (temp / f'{CAM}_002.mp4').write_bytes(b'\x00' * 64)
    link = {'camera': root / 'images' / CAM, 'episode': staged, 'frame': staged / 'frame-000003.png',
            'tempdir': temp, 'tempfile': temp / f'{CAM}_002.mp4'}[where]
    moved = tmp_path / 'elsewhere' / link.name
    moved.parent.mkdir()
    link.rename(moved)
    link.symlink_to(moved)
    message = refusal(root, video_headers=video_headers)
    assert message.startswith(f'Refused before reading: {link.relative_to(root)} is a symlink to {moved.resolve()}')


def test_r3_d2_links_and_fifos_the_check_never_uses_do_not_block_it(tmp_path):
    root = healthy(tmp_path)
    (tmp_path / 'notes.txt').write_text('n')
    (root / 'notes.txt').symlink_to(tmp_path / 'notes.txt')  # LeRobot never reads it; neither does the check
    os.mkfifo(root / 'stray_pipe')
    os.mkfifo(root / 'data' / 'stray_pipe')
    with deadline():
        assert check_salvage(root)['status'] == 'pass'


class _Hang(BaseException):
    """Raised by the deadline alarm; not an OSError, so the check cannot turn a hang into an InputError."""


@contextlib.contextmanager
def deadline(seconds=20):
    def expire(signum, frame):
        raise _Hang(f'check_salvage did not return within {seconds} s')
    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize('video_headers', [True, False])
@pytest.mark.parametrize('where', ['meta/tasks.parquet', 'meta/info.json', 'meta/stats.json',
                                   'data/chunk-000/file-000.parquet', 'meta/episodes/chunk-000/file-000.parquet',
                                   'referenced video', 'video without metadata', 'temp mp4', 'staging frame'])
def test_r3_d4_fifos_never_block_and_are_refused_by_name(tmp_path, where, video_headers):
    # Recheck3 D4/F1/F2: a FIFO meta/tasks.parquet and a FIFO video in a recording without metadata used to hang.
    if where == 'video without metadata':  # builder K1 shape: info.json counts 2 episodes, meta/episodes/ never written
        root = write(tmp_path / 'ds', [rows_for([(0, 5), (1, 6)])], [], 2, 11, video=True)
    else:
        root = healthy(tmp_path, video=True)
    relative = {'referenced video': f'videos/{CAM}/chunk-000/file-000.mp4',
                'video without metadata': f'videos/{CAM}/chunk-000/file-000.mp4',
                'temp mp4': f'tmpab12cd_9/{CAM}_003.mp4',
                'staging frame': f'images/{CAM}/episode-000003/frame-000000.png'}.get(where, where)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    os.mkfifo(path)
    with deadline():
        message = refusal(root, video_headers=video_headers)
    assert message.startswith(f'Refused before reading: {relative} is a FIFO (named pipe), not a regular file')


def test_r3_d5_symlink_loops_and_unresolvable_paths_are_input_errors(tmp_path):
    # Recheck3 E12a/E12c and '~nosuchuser': these raised a raw RuntimeError.
    root = healthy(tmp_path / 'a', video=True)
    (root / 'videos').symlink_to('videos')
    assert refusal(root).startswith('Refused before reading: videos is a symlink to videos, which cannot be followed')
    root = healthy(tmp_path / 'b')
    shutil.rmtree(root / 'data' / 'chunk-000')
    (root / 'data' / 'chunk-000').symlink_to(root / 'data' / 'chunk-001')
    (root / 'data' / 'chunk-001').symlink_to(root / 'data' / 'chunk-000')
    assert 'which cannot be followed (a symlink loop or a missing target)' in refusal(root)
    loop = tmp_path / 'loop'
    loop.symlink_to(loop)
    refusal(loop)
    refusal('~nosuchuser_zz9_salvage/x')


@pytest.mark.parametrize('video_headers', [True, False])
@pytest.mark.parametrize('kind', ['socket', 'directory'])
def test_r3_d4_no_plan_copies_a_non_regular_file(tmp_path, monkeypatch, kind, video_headers):
    # A referenced video (listed in copy_unchanged when its episode is kept) that is a socket or a directory, and a
    # socket task table, are refused by name instead of reaching the plan.
    root = healthy(tmp_path, video=True)
    video = root / 'videos' / CAM / 'chunk-000' / 'file-000.mp4'
    video.parent.mkdir(parents=True)
    if kind == 'directory':
        video.mkdir()
        expected = f'videos/{CAM}/chunk-000/file-000.mp4 is a directory, not a regular file'
    else:
        import socket
        monkeypatch.chdir(video.parent)  # AF_UNIX paths are short: bind relative to the camera chunk directory
        server = socket.socket(socket.AF_UNIX)
        server.bind(video.name)
        expected = f'videos/{CAM}/chunk-000/file-000.mp4 is a socket, not a regular file'
    with deadline():
        assert refusal(root, video_headers=video_headers).startswith(f'Refused before reading: {expected}')
    if kind == 'socket':
        server.close()


def test_r3_d5_other_path_errors_are_input_errors(tmp_path):
    with pytest.raises(InputError, match='Cannot resolve the dataset path'):
        check_salvage(f'{tmp_path}/bad\x00name')
    if os.geteuid() == 0:
        return  # root reads through permission bits
    root = healthy(tmp_path)
    stage(root, 3, 2)
    (root / 'images').chmod(0)
    try:
        with pytest.raises(InputError, match='Permission denied'):
            check_salvage(root)
    finally:
        (root / 'images').chmod(0o755)


def test_w2_metadata_lost_with_its_footer_and_rows_indexed_from_totals_need_reindexing(tmp_path):
    # Validator K_RES_METACUT: a resume wrote episode 3's metadata to file-001, whose footer was then lost; the resumed
    # rows carry global indices from info totals (18..), not their row positions (11..).
    first = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for([(3, 4)], start=18)],
                 [meta_for(first), meta_for([(3, 4)], start=11, data_file=1, meta_file=1)], 4, 22)
    strip_footer(root / 'meta' / 'episodes' / 'chunk-000' / 'file-001.parquet')
    result = check_salvage(root)
    assert result['episodes']['drop'] == [{'episode_index': 3, 'reasons': ['metadata_unreadable_or_unflushed'],
                                           'rows': 4, 'metadata_rows': 0,
                                           'evidence': result['episodes']['drop'][0]['evidence']}]
    dropped = finding(result, 'salvage.episodes_dropped')
    assert 'lost their footer' in dropped['message'] and 'still buffered' not in dropped['message']
    assert 're-indexed' in dropped['remedy'] and 'index 18 at row position 11' in dropped['remedy']


def test_empty_recording_has_nothing_to_salvage(tmp_path):
    root = tmp_path / 'ds'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'total_episodes': 0,
        'total_frames': 0, 'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet', 'features': {}}))
    result = check_salvage(root)
    assert codes(result) == {'salvage.nothing_committed': 'warning'}


@pytest.mark.parametrize('mutation', ['budget', 'version'])
def test_unsupported_or_over_budget_input_is_unknown(tmp_path, mutation):
    root = healthy(tmp_path, version='v2.1' if mutation == 'version' else 'v3.0')
    result = check_salvage(root, max_rows=4 if mutation == 'budget' else 200000)
    assert codes(result) == {'salvage.coverage_unknown': 'unknown'} and result['status'] == 'unknown'
    assert result['plan']['strategy'] == 'unknown'


@pytest.mark.parametrize('kwargs', [{'max_rows': 0}, {'max_rows': True}, {'max_files': 5000}, {'video_headers': 1}])
def test_strict_budget_inputs(tmp_path, kwargs):
    with pytest.raises(InputError):
        check_salvage(healthy(tmp_path), **kwargs)


def test_network_and_missing_inputs_are_refused(tmp_path):
    with pytest.raises(InputError, match='local'):
        check_salvage('hf://lerobot/pusht')
    with pytest.raises(InputError, match='directory'):
        check_salvage(tmp_path / 'absent')


def test_symlink_escape_is_refused_before_read(tmp_path):
    root = healthy(tmp_path)
    shard = root / 'data' / 'chunk-000' / 'file-000.parquet'
    outside = tmp_path / 'outside.parquet'
    shard.rename(outside)
    shard.symlink_to(outside)
    with pytest.raises(InputError, match='symlink|escape'):
        check_salvage(root)


def test_many_dropped_episodes_stay_within_the_transport_budget(tmp_path):
    kept = [(e, 2) for e in range(50)]
    orphans = [(e, 1) for e in range(51, 4000, 2)]
    root = write(tmp_path / 'ds', [rows_for(kept + orphans)], [meta_for(kept)], 50, 100)
    result = check_salvage(root)
    assert len(json.dumps(result).encode()) <= 192 * 1024
    assert result['counts']['drop'] == len(orphans) and result['episodes']['drop_truncated'] is True
    assert len(result['findings'][0]['episodes']) == 64


def test_dataset_aliases_resolve_to_the_same_root(tmp_path):
    root = healthy(tmp_path)
    assert check_salvage(root / 'meta' / 'info.json')['status'] == 'pass'


@pytest.mark.parametrize('temporary_kind', ['file', 'directory', 'dangling_link'])
def test_p2_link_copy_retry_refuses_existing_temporary_without_touching_input(tmp_path, temporary_kind):
    root, disk = outside_videos(tmp_path)
    link = root / 'videos'
    temporary = root / 'videos.tmp'
    if temporary_kind == 'file':
        temporary.write_bytes(b'user-owned partial copy')
    elif temporary_kind == 'directory':
        temporary.mkdir()
        (temporary / 'partial.mp4').write_bytes(b'partial')
    else:
        temporary.symlink_to(tmp_path / 'missing')
    before = hashes(disk)
    command = refusal(root).split('`')[1]
    result = subprocess.run(['bash', '-c', command], capture_output=True, timeout=10)
    assert result.returncode != 0 and link.is_symlink()
    assert hashes(disk) == before
    if temporary_kind == 'file':
        assert temporary.read_bytes() == b'user-owned partial copy'
    elif temporary_kind == 'directory':
        assert list(temporary.iterdir()) == [temporary / 'partial.mp4']
    else:
        assert temporary.is_symlink() and os.readlink(temporary) == str(tmp_path / 'missing')


def test_p3_copy_preserves_inner_links_and_reports_all_links_without_following_them(tmp_path):
    root, disk = outside_videos(tmp_path)
    (disk / 'videos' / 'up').symlink_to(tmp_path)
    before = hashes(disk)
    command = refusal(root).split('`')[1]
    subprocess.run(['bash', '-c', command], check=True, timeout=10)
    assert not (root / 'videos').is_symlink()
    assert (root / 'videos' / 'up').is_symlink() and hashes(disk) == before
    # No recursive copy of tmp_path, no partial-copy retry, and no acceptance of the preserved inner link.
    message = refusal(root)
    assert 'videos/up' in message and 'contains the dataset itself' in message
    stats = root / 'meta' / 'stats.json'
    stats.rename(root / 'meta' / 'stats.real.json')
    stats.symlink_to('stats.real.json')
    message = refusal(root)
    assert 'All 2 link(s)' in message and 'videos/up -> ' in message and 'meta/stats.json -> stats.real.json' in message


@pytest.mark.parametrize('argument', ['info.json', './info.json', 'stats.json'])
def test_p3_relative_metadata_argument_inside_meta_is_accepted(tmp_path, monkeypatch, argument):
    root = healthy(tmp_path)
    monkeypatch.chdir(root / 'meta')
    assert check_salvage(argument)['status'] == 'pass'


def test_p3_nonregular_and_dangling_arguments_name_the_actual_kind(tmp_path):
    pipe = tmp_path / 'pipe'
    os.mkfifo(pipe)
    assert 'a FIFO (named pipe)' in refusal(pipe)
    assert 'a character device' in refusal('/dev/null')
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'absent')
    assert 'is a symlink to' in refusal(dangling) and 'target is missing or loops' in refusal(dangling)


@pytest.mark.parametrize('backend', [None, 'lance'])
def test_explicit_other_backend_never_salvages_stale_parquet(tmp_path, backend):
    root = healthy(tmp_path, video=True)
    strip_footer(root / 'data' / 'chunk-000' / 'file-000.parquet')
    if backend:
        info = json.loads((root / 'meta' / 'info.json').read_text())
        (root / 'meta' / 'info.json').write_text(json.dumps(dict(info, storage_format=backend)))
    before = hashes(root)
    result = check_salvage(str(root), video_headers=False)
    if backend is None:
        assert result['status'] == 'error'
    else:
        assert result['status'] == 'unknown' and result['plan']['strategy'] == 'unknown'
        assert not any(f['status'] == 'error' for f in result['findings']), result['findings']
    assert hashes(root) == before
