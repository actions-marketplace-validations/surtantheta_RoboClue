"""Native row checks preserve data semantics around aliases and name unread input."""

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_commanded_reach as reach_fixtures
import test_video_index as video_fixtures
import test_windows as window_fixtures
from test_summary_composition_budget import client_for

from policy_doctor.artifacts import MAX_JSON_BYTES, InputError
from policy_doctor.commanded_reach import check_commanded_reach
from policy_doctor.windows import DATA_PATH, audit_episode_windows

engine = importlib.import_module('policy_doctor.preflight')


def native_reach(path):
    root = reach_fixtures._dataset(path)
    file = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(file)
    episodes = table['episode_index'].to_pylist()
    counters, metadata, frames = {}, [], []
    for index, episode in enumerate(episodes):
        if episode not in counters:
            counters[episode] = 0
            metadata.append({'episode_index': episode, 'length': 0, 'dataset_from_index': index,
                             'dataset_to_index': index, 'data/chunk_index': 0, 'data/file_index': 0})
        frames.append(counters[episode])
        counters[episode] += 1
        metadata[-1]['length'] += 1
        metadata[-1]['dataset_to_index'] = index + 1
    for name, values in (('index', list(range(len(episodes)))), ('frame_index', frames),
                         ('timestamp', [frame / 30 for frame in frames])):
        table = table.append_column(name, pa.array(values))
    pq.write_table(table, file)
    meta = root / 'meta/episodes/chunk-000/file-000.parquet'
    meta.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(metadata), meta)
    info = root / 'meta/info.json'
    value = json.loads(info.read_text())
    value.update(total_frames=len(episodes), total_episodes=len(counters), data_path=DATA_PATH)
    info.write_text(json.dumps(value))
    return root


def add_loops(root, tree='data'):
    for name, target in (('l0', '.'), ('l1', '.'), ('up', '..')):
        (root / tree / name).symlink_to(target)


def cli(*args, cwd):
    source = str(Path(__file__).resolve().parents[1] / 'src')
    bootstrap = 'import sys,runpy;sys.path.insert(0,' + repr(source) + ');runpy.run_module("policy_doctor",run_name="__main__")'
    return subprocess.run([sys.executable, '-I', '-B', '-c', bootstrap, *map(str, args)], cwd=cwd,
                          env={**os.environ, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                          capture_output=True, text=True, timeout=90)


def test_ancestor_and_self_aliases_preserve_native_rows_and_residual(tmp_path):
    plain = native_reach(tmp_path / 'plain')
    loop = Path(shutil.copytree(plain, tmp_path / 'loop'))
    add_loops(loop)
    reports = [engine.preflight(str(root), commanded_reach=True, units=True) for root in (plain, loop)]
    for component in ('episode_windows', 'video_index', 'numeric_rows', 'commanded_reach', 'units'):
        assert reports[1]['components'][component]['status'] == reports[0]['components'][component]['status']
    before, after = [report['components']['commanded_reach'] for report in reports]
    assert before['status'] == after['status'] == 'pass'
    assert before['joints'] == after['joints']
    assert before['coverage']['rows'] == after['coverage']['rows'] > 1000
    assert before['coverage']['data_files_read'] == after['coverage']['data_files_read'] == 1
    for root in (plain, loop):
        window = audit_episode_windows(str(root), episode_index=0, anchor_index=3, delta_indices=[-3, 0, 3])
        assert window['status'] == 'pass'
        values = window['query']['declared']
        if root == plain:
            original = values
        else:
            for key in ('indices', 'selected_episode_indices', 'values', 'action_is_pad'):
                assert values[key] == original[key]


def test_looped_video_directory_preserves_header_and_episode_coverage(tmp_path):
    plain = video_fixtures.dataset(tmp_path / 'plain', [(10, 0, 0.0)], {0: 10}, with_data=True)
    loop = Path(shutil.copytree(plain, tmp_path / 'loop'))
    add_loops(loop)
    add_loops(loop, 'videos')
    before, after = [audit_episode_windows(str(root)) for root in (plain, loop)]
    assert before['status'] == after['status'] == 'pass'
    assert before['coverage'] == after['coverage']
    assert before['counts'] == after['counts']


def test_nonnative_directory_names_required_info_and_discovery_command(tmp_path):
    root = tmp_path / 'ordinary_run'
    root.mkdir()
    (root / 'observations.csv').write_text('value\n1\n')
    report = audit_episode_windows(str(root))
    assert report['status'] == 'unknown' and report['coverage']['data_files_read'] == 0
    reason = report['findings'][0]['reason']
    assert str(root / 'meta/info.json') in reason and 'not a supported native LeRobot dataset' in reason
    assert 'investigate DIRECTORY' in reason
    result = cli('audit-windows', root, '--json', cwd=tmp_path)
    assert result.returncode == 3 and 'meta/info.json' in result.stdout


def test_oversized_metadata_is_named_unread_coverage_not_missing(tmp_path):
    root = tmp_path / 'large'
    (root / 'meta').mkdir(parents=True)
    info = root / 'meta/info.json'
    info.write_bytes(b'{"padding":"' + b'x' * MAX_JSON_BYTES + b'"}')
    report = audit_episode_windows(str(root))
    assert report['status'] == 'unknown' and report['coverage']['identity_verified'] is False
    reason = report['findings'][0]['reason']
    assert 'meta/info.json' in reason and str(MAX_JSON_BYTES) in reason and 'read cap' in reason
    assert 'contents were not checked' in reason and 'Missing' not in reason
    assert report['artifacts'] == []


def test_unsafe_directory_link_names_plain_copy_without_reading_target(tmp_path):
    root = window_fixtures.make_dataset(tmp_path / 'inside')
    outside = tmp_path / 'other-disk'
    outside.mkdir()
    (outside / 'DO_NOT_READ.parquet').write_bytes(b'not parquet')
    (root / 'data/linked').symlink_to(outside)
    for check in (audit_episode_windows, lambda source: engine.preflight(source)):
        with pytest.raises(InputError) as caught:
            check(str(root))
        message = str(caught.value)
        assert 'data/linked' in message and str(outside) in message
        assert 'cp -rP --' in message and 'test ! -e' in message and 'test ! -L' in message
        assert 'rm --' in message and 'mv -T --' in message
        assert 'DO_NOT_READ' not in message  # no target listing or body was inspected


def test_data_alias_to_metadata_cannot_be_read_as_numeric_rows(tmp_path):
    root = native_reach(tmp_path / 'root')
    (root / 'data/meta-alias').symlink_to('../meta')
    with pytest.raises(InputError) as caught:
        check_commanded_reach(str(root))
    assert 'data/meta-alias' in str(caught.value)
    assert 'Restore the original regular numeric' in str(caught.value) and 'has no column' not in str(caught.value)


@pytest.mark.parametrize('internal_meta_alias', [False, True])
def test_top_level_data_alias_cannot_set_fence_to_metadata(tmp_path, monkeypatch, internal_meta_alias):
    root = native_reach(tmp_path / 'root')
    if internal_meta_alias:
        (root / 'meta').rename(root / 'metadata_store')
        (root / 'meta').symlink_to('metadata_store')
    shutil.rmtree(root / 'data')
    (root / 'data').symlink_to('meta')

    def no_parquet_body(*args, **kwargs):
        pytest.fail('metadata Parquet must not be opened as numeric data')
    monkeypatch.setattr(pq, 'ParquetFile', no_parquet_body)
    with pytest.raises(InputError) as caught:
        check_commanded_reach(str(root))
    message = str(caught.value)
    assert 'Numeric data link data' in message and 'Metadata Parquet is not numeric data' in message
    assert 'Restore the original regular numeric' in message


def test_top_level_alias_to_separate_numeric_directory_keeps_rows(tmp_path):
    root = native_reach(tmp_path / 'root')
    before = check_commanded_reach(str(root))
    (root / 'data').rename(root / 'numeric_store')
    (root / 'data').symlink_to('numeric_store')
    after = check_commanded_reach(str(root))
    assert before['status'] == after['status'] == 'pass'
    assert before['joints'] == after['joints']
    assert before['coverage']['rows'] == after['coverage']['rows']
    assert [item['sha256'] for item in before['coverage']['data_files']] == \
        [item['sha256'] for item in after['coverage']['data_files']]


@pytest.mark.parametrize('data_alias', [False, True])
def test_canonical_metadata_descendant_is_excluded_from_numeric_walk(tmp_path, monkeypatch, data_alias):
    root = native_reach(tmp_path / 'root')
    before = check_commanded_reach(str(root))
    data = root / 'data'
    if data_alias:
        data.rename(root / 'numeric_store')
        data.symlink_to('numeric_store')
    physical_data = data.resolve()
    (root / 'meta').rename(physical_data / 'nested_metadata')
    (root / 'meta').symlink_to(physical_data / 'nested_metadata')
    metadata = (root / 'meta').resolve()
    original_reader, opened = pq.ParquetFile, []

    def only_numeric(path, *args, **kwargs):
        resolved = Path(path).resolve()
        assert not resolved.is_relative_to(metadata), 'metadata Parquet body must never be opened as numeric data'
        opened.append(resolved)
        return original_reader(path, *args, **kwargs)
    monkeypatch.setattr(pq, 'ParquetFile', only_numeric)
    after = check_commanded_reach(str(root))
    assert before['status'] == after['status'] == 'pass'
    assert before['joints'] == after['joints']
    assert before['coverage']['rows'] == after['coverage']['rows']
    assert len(opened) == len(after['coverage']['data_files']) == 1
    assert [item['sha256'] for item in before['coverage']['data_files']] == \
        [item['sha256'] for item in after['coverage']['data_files']]


def test_numeric_file_alias_into_metadata_is_named_before_parquet_read(tmp_path, monkeypatch):
    root = native_reach(tmp_path / 'root')
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    (root / 'data/metadata-as-data.parquet').symlink_to(metadata)

    def no_parquet_body(*args, **kwargs):
        pytest.fail('metadata file alias must be refused before any Parquet body is opened')
    monkeypatch.setattr(pq, 'ParquetFile', no_parquet_body)
    with pytest.raises(InputError) as caught:
        check_commanded_reach(str(root))
    assert 'data/metadata-as-data.parquet' in str(caught.value)
    assert 'reserved metadata tree' in str(caught.value) and 'Restore' in str(caught.value)


def test_actual_stdio_loop_parity_and_named_refusal_remains_responsive(tmp_path):
    plain = native_reach(tmp_path / 'plain')
    loop = Path(shutil.copytree(plain, tmp_path / 'loop'))
    add_loops(loop)
    unsafe = window_fixtures.make_dataset(tmp_path / 'unsafe')
    outside = tmp_path / 'external'
    outside.mkdir()
    (unsafe / 'data/linked').symlink_to(outside)
    original_hash = hashlib.sha256((plain / 'data/chunk-000/file-000.parquet').read_bytes()).hexdigest()

    async def run():
        async with client_for(tmp_path) as client:
            for root in (plain, loop):
                response = await client.call_tool('preflight_check', {'source': str(root), 'commanded_reach': True,
                                                                      'response_format': 'detailed'})
                assert not response.isError, response.content
                parts = response.structuredContent['components']
                assert parts['episode_windows']['status'] == parts['commanded_reach']['status'] == 'pass'
            refused = await client.call_tool('audit_dataset_windows', {'dataset': str(unsafe)})
            assert refused.isError
            assert 'data/linked' in refused.content[0].text and 'cp -rP --' in refused.content[0].text
            following = await client.call_tool('audit_dataset_windows', {'dataset': str(loop)})
            assert not following.isError and following.structuredContent['status'] == 'pass'
    asyncio.run(asyncio.wait_for(run(), timeout=120))
    assert hashlib.sha256((plain / 'data/chunk-000/file-000.parquet').read_bytes()).hexdigest() == original_hash
