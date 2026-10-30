"""Plan/execution boundaries and complete video accounting after practitioner feedback."""
import json
import os
import shutil

import pyarrow.parquet as pq
import pytest
from test_salvage import CAM, hashes, healthy, meta_for, rows_for, strip_footer, write

from policy_doctor.artifacts import InputError
from policy_doctor.salvage import check_salvage
from policy_doctor.sim_salvage_frontdoor import concise_salvage, human_salvage, markdown_salvage


def test_original_hazard_and_predicted_copy_are_distinct(tmp_path):
    first = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(first), rows_for([(2, 3)], start=11)],
                 [meta_for(first)], 3, 14)
    strip_footer(root / 'data/chunk-000/file-001.parquet')
    before = hashes(root)
    report = check_salvage(root)
    validity = report['plan']['validity']
    assert validity['current_directory_resume_status'] == 'unsafe'
    assert validity['resume_safe'] is True  # prediction after omitting the damaged shard
    assert validity['copy_verification_required'] is True
    assert 'never the current directory' in validity['resume_safe_applies_to']
    text = human_salvage(report)
    assert 'Current-directory resume: UNSAFE; do not resume here.' in text
    assert 'resume safe: yes' not in text
    assert 'prediction only, not applied or verified' in text
    view = concise_salvage(report)
    assert view['plan']['current_directory_resume_status'] == 'unsafe'
    assert view['plan']['applied'] is False and view['plan']['copy_verification_required'] is True
    assert hashes(root) == before


def test_excluded_does_not_mean_delete_recoverable_source(tmp_path):
    root = write(tmp_path / 'ds', [rows_for([(0, 5)]), rows_for([(1, 4)], start=5)],
                 [meta_for([(0, 5)])], 2, 9)
    report = check_salvage(root)
    assert report['episodes']['recovery_candidates'] == [[1, 1]]
    assert 'Not an instruction to delete' in report['episodes']['excluded_from_plan_meaning']
    text = human_salvage(report)
    assert 'Episodes to drop:' not in text
    assert 'Episodes excluded from this metadata-only copy: 1' in text
    assert 'Recovery candidates outside this plan: 1' in text
    assert 'not a deletion instruction' in text


def test_orphan_videos_are_all_accounted_for(tmp_path):
    root = healthy(tmp_path)
    folder = root / 'videos' / CAM / 'chunk-000'
    folder.mkdir(parents=True)
    videos = []
    for index in range(4):
        name = f'videos/{CAM}/chunk-000/file-{index:03d}.mp4'
        (root / name).write_bytes(b'original bytes; not a referenced video')
        videos.append(name)
    before = hashes(root)
    report = check_salvage(root)
    files = report['plan']['files']
    assert files['lists_complete'] is True
    assert set(videos) <= set(files['omit'])
    assert not set(videos) & set(files['copy_unchanged'])
    assert files['omit_count'] == 4
    assert 'fresh directory' in files['inventory_scope']
    assert hashes(root) == before


def test_unreferenced_video_inventory_obeys_file_bound(tmp_path):
    root = healthy(tmp_path)
    folder = root / 'videos/cam/chunk-000'
    folder.mkdir(parents=True)
    for i in range(5):
        (folder / f'file-{i:03d}.mp4').write_bytes(b'x')
    report = check_salvage(root, max_files=4)
    assert report['status'] == 'unknown'
    assert report['plan']['strategy'] == 'unknown'
    assert report['episodes'] == {}
    assert 'Video copy-plan inventory file budget' in report['plan']['reason']


def test_unreferenced_special_video_file_is_refused_before_open(tmp_path):
    root = healthy(tmp_path)
    folder = root / 'videos/cam/chunk-000'
    folder.mkdir(parents=True)
    path = folder / 'file-999.mp4'
    os.mkfifo(path)
    with pytest.raises(InputError, match='FIFO'):
        check_salvage(root)


def test_unreferenced_video_symlink_never_reads_target(tmp_path):
    root = healthy(tmp_path)
    folder = root / 'videos/cam/chunk-000'
    folder.mkdir(parents=True)
    (folder / 'file-999.mp4').symlink_to(tmp_path / 'absent')
    with pytest.raises(InputError, match='symlink'):
        check_salvage(root)


def test_ctrlc_plan_names_required_data_trim_and_same_source_remains(tmp_path):
    kept = [(0, 5), (1, 6)]
    root = write(tmp_path / 'ds', [rows_for(kept + [(2, 14)])], [meta_for(kept)], 2, 11)
    before = hashes(root)
    original = check_salvage(root)
    assert original['plan']['validity']['resume_safe'] is False
    descriptor, = original['plan']['required_data_rewrite']
    assert descriptor['file'] == 'data/chunk-000/file-000.parquet'
    assert descriptor['keep_physical_rows'] == [0, 11]
    assert descriptor['range_end'] == 'exclusive' and descriptor['rows_to_exclude'] == 14
    assert 'Separate-copy data rewrite needed:' in human_salvage(original)
    candidate = tmp_path / 'candidate'
    shutil.copytree(root, candidate)
    path = candidate / descriptor['file']
    table = pq.read_table(path)
    start, end = descriptor['keep_physical_rows']
    pq.write_table(table.slice(start, end - start), path)
    corrected = check_salvage(candidate)
    assert corrected['status'] == 'pass'
    assert corrected['plan']['validity']['resume_safe'] is True
    assert corrected['plan']['validity']['current_directory_resume_status'] == 'no_hazard_detected'
    assert pq.read_table(path).schema.metadata == table.schema.metadata
    assert hashes(root) == before


def test_partial_plan_markdown_does_not_claim_complete_listing():
    report = {'kind': 'recording_salvage', 'status': 'unknown', 'coverage': {}, 'findings': [],
              'episodes': {}, 'plan': {'strategy': 'metadata_copy', 'files': {'lists_complete': False}}}
    text = markdown_salvage(report)
    assert '## Plan (complete)' not in text
    assert 'inspect files.lists_complete' in text
    assert 'Plan file lists are truncated' in text


def test_staging_preview_marks_omissions():
    report = {'kind': 'recording_salvage', 'status': 'warning', 'coverage': {}, 'findings': [],
              'episodes': {'staging_only': list(range(20))}, 'plan': None}
    assert '... (+4 more in --json)' in human_salvage(report)


def test_plan_remains_finite_json(tmp_path):
    report = check_salvage(healthy(tmp_path))
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize(('video', 'episode', 'expected'), [
    (True, 3, 'unsafe'), (True, 4, 'no_hazard_detected'), (False, 3, 'no_hazard_detected'),
])
def test_current_resume_accounts_for_existing_staging_contamination(tmp_path, video, episode, expected):
    root = healthy(tmp_path, video=video)
    frames = root / 'images' / CAM / f'episode-{episode:06d}'
    frames.mkdir(parents=True)
    (frames / 'frame-000000.png').write_bytes(b'original staging bytes')
    before = hashes(root)
    report = check_salvage(root)
    validity = report['plan']['validity']
    assert validity['current_directory_resume_status'] == expected
    assert ('salvage.leftovers' in validity['current_directory_resume_hazard_codes']) == (expected == 'unsafe')
    assert validity['resume_safe'] is True  # separate predicted copy excludes staging
    assert hashes(root) == before
