"""Planner unit controls with generated metadata and a declared native-API double.

Actual pinned native metadata qualification is separately frozen and launched.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from policy_doctor import native_lerobot_plan_worker as worker

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')


class MetadataDouble:
    def __init__(self, dataset):
        self.root = dataset
        self.episodes = [
            {'episode_index': 0, 'length': 3, 'data/chunk_index': 7, 'data/file_index': 11,
             'videos/observation.images.unusual/chunk_index': 9,
             'videos/observation.images.unusual/file_index': 14},
            {'episode_index': 1, 'length': 8, 'data/chunk_index': 12, 'data/file_index': 18,
             'videos/observation.images.unusual/chunk_index': 15,
             'videos/observation.images.unusual/file_index': 27},
        ]
        self.video_keys = ['observation.images.unusual']
        self.calls = []

    def get_data_file_path(self, episode):
        self.calls.append(('data', episode))
        row = self.episodes[episode]
        return Path(f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet')

    def get_video_file_path(self, episode, key):
        self.calls.append(('video', episode, key))
        row = self.episodes[episode]
        return Path(f'videos/{key}/chunk-{row[f"videos/{key}/chunk_index"]:03d}/file-{row[f"videos/{key}/file_index"]:03d}.mp4')


@pytest.fixture
def setup_plan(tmp_path, monkeypatch):
    dataset = tmp_path / 'metadata'
    episodes = dataset / 'meta/episodes/chunk-005'
    episodes.mkdir(parents=True)
    info = {'total_episodes': 2, 'total_tasks': 1,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'features': {'observation.images.unusual': {'dtype': 'video', 'shape': [32, 32, 3]},
                         'observation.state': {'dtype': 'float32', 'shape': [8]}}}
    (dataset / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.table({'task': ['ordinary task text']}), dataset / 'meta/tasks.parquet')
    pq.write_table(pa.table({'episode_index': [0, 1], 'length': [3, 8]}), episodes / 'file-019.parquet')
    native = MetadataDouble(dataset)
    loaded = []

    def load(source, root, output):
        loaded.append((source, root, output))
        return native, [MetadataDouble]

    monkeypatch.setattr(worker, '_load_native_metadata', load)
    request = {'source_root': str(Path(__file__).parent), 'dataset': str(dataset),
               'samples': [{'episode_index': 1, 'frame_index': 5}, {'episode_index': 0, 'frame_index': 2}],
               'output': str(tmp_path / 'output')}
    return SimpleNamespace(request=request, dataset=dataset, info=info, native=native, loaded=loaded,
                           episode_file=episodes / 'file-019.parquet')


def test_native_methods_select_changed_paths_and_request_order(setup_plan):
    context = setup_plan
    result = worker.run(context.request)
    assert result['execution_status'] == 'complete'
    assert result['provider'] == 'lerobot' and result['operation'] == 'plan'
    assert [sample['episode_index'] for sample in result['samples']] == [1, 0]
    assert [sample['episode_length'] for sample in result['samples']] == [8, 3]
    assert result['files'] == [
        {'path': 'data/chunk-007/file-011.parquet', 'role': 'native_data'},
        {'path': 'data/chunk-012/file-018.parquet', 'role': 'native_data'},
        {'path': 'videos/observation.images.unusual/chunk-009/file-014.mp4', 'role': 'native_video'},
        {'path': 'videos/observation.images.unusual/chunk-015/file-027.mp4', 'role': 'native_video'}]
    assert context.native.calls == [('data', 1), ('video', 1, 'observation.images.unusual'),
                                     ('data', 0), ('video', 0, 'observation.images.unusual')]
    assert result['metadata_files'] == ['meta/episodes/chunk-005/file-019.parquet',
                                         'meta/info.json', 'meta/tasks.parquet']
    assert result['source_files'][0]['path'] == Path(__file__).name


def test_shared_paths_are_deduplicated_without_losing_sample_bindings(setup_plan):
    context = setup_plan
    for key in ('data/chunk_index', 'data/file_index', 'videos/observation.images.unusual/chunk_index',
                'videos/observation.images.unusual/file_index'):
        context.native.episodes[1][key] = context.native.episodes[0][key]
    result = worker.run(context.request)
    assert len(result['files']) == 2
    assert len(result['samples']) == 2


def test_guard_precedes_native_import_on_dictionary_expansion(setup_plan):
    context = setup_plan
    pq.write_table(pa.table({'task': ['x' * 16_384] * 2500}), context.dataset / 'meta/tasks.parquet')
    with pytest.raises(ValueError, match='decoded allocation'):
        worker.run(context.request)
    assert context.loaded == []


def test_every_episode_metadata_shard_is_guarded_before_native_loading(setup_plan):
    context = setup_plan
    pq.write_table(pa.table({'binary': [b'unsafe']}), context.episode_file.with_name('file-020.parquet'))
    with pytest.raises(ValueError, match='Unsupported Parquet schema'):
        worker.run(context.request)
    assert context.loaded == []


def test_optional_stats_absent_is_lawful_present_is_retained(setup_plan):
    context = setup_plan
    assert 'meta/stats.json' not in worker.run(context.request)['metadata_files']
    (context.dataset / 'meta/stats.json').write_text('{"observation.state":{"mean":[0]}}')
    assert 'meta/stats.json' in worker.run(context.request)['metadata_files']


@pytest.mark.parametrize('field,value', [('data_path', '../{chunk_index}.parquet'),
                                        ('video_path', 'https://example.test/a.mp4'),
                                        ('storage_format', 'lance'), ('total_tasks', True),
                                        ('total_episodes', 100_001)])
def test_metadata_preflight_blocks_native_loader(setup_plan, field, value):
    context = setup_plan
    context.info[field] = value
    (context.dataset / 'meta/info.json').write_text(json.dumps(context.info))
    with pytest.raises(ValueError):
        worker.run(context.request)
    assert context.loaded == []


@pytest.mark.parametrize('samples', [[{'episode_index': True, 'frame_index': 0}],
                                     [{'episode_index': 0, 'physical_row': 0}],
                                     [{'episode_index': 0, 'frame_index': 0}] * 2,
                                     [{'episode_index': 2, 'frame_index': 0}], []])
def test_typed_selectors_are_checked_before_native_loading(setup_plan, samples):
    context = setup_plan
    context.request['samples'] = samples
    with pytest.raises(ValueError):
        worker.run(context.request)
    assert context.loaded == []


def test_native_positional_identity_must_match_request_before_path_call(setup_plan):
    context = setup_plan
    context.native.episodes.reverse()
    with pytest.raises(ValueError, match='positional episode identity'):
        worker.run(context.request)
    assert context.native.calls == []


def test_native_table_cannot_be_selected_only_reindexed_metadata(setup_plan):
    context = setup_plan
    context.native.episodes.pop()
    with pytest.raises(ValueError, match='incomplete'):
        worker.run(context.request)


@pytest.mark.parametrize('frame', [8, 999])
def test_frame_bound_comes_from_native_episode_length(setup_plan, frame):
    context = setup_plan
    context.request['samples'] = [{'episode_index': 1, 'frame_index': frame}]
    with pytest.raises(ValueError, match='outside its original episode'):
        worker.run(context.request)
    assert context.native.calls == []


def test_malformed_native_file_index_is_not_formatted_as_integer(setup_plan):
    context = setup_plan
    context.native.episodes[1]['data/chunk_index'] = True
    with pytest.raises(ValueError, match='data file index'):
        worker.run(context.request)


@pytest.mark.parametrize('path', ['../escape.parquet', '/tmp/x.parquet',
                                 'data/../escape.parquet', 'https://host/a.parquet'])
def test_native_returned_paths_are_still_inert_and_validated(setup_plan, path):
    context = setup_plan
    context.native.get_data_file_path = lambda _: path
    with pytest.raises(ValueError, match='dependency'):
        worker.run(context.request)


def test_required_task_file_missing_fails_before_native_loading(setup_plan):
    context = setup_plan
    (context.dataset / 'meta/tasks.parquet').unlink()
    with pytest.raises(ValueError, match='task metadata'):
        worker.run(context.request)
    assert context.loaded == []


def test_no_task_file_is_required_when_native_declares_zero_tasks(setup_plan):
    context = setup_plan
    context.info['total_tasks'] = 0
    (context.dataset / 'meta/info.json').write_text(json.dumps(context.info))
    (context.dataset / 'meta/tasks.parquet').unlink()
    assert 'meta/tasks.parquet' not in worker.run(context.request)['metadata_files']


def test_symlink_episode_metadata_fails_before_native_loading(setup_plan, tmp_path):
    context = setup_plan
    original = tmp_path / 'original.parquet'
    context.episode_file.rename(original)
    context.episode_file.symlink_to(original)
    with pytest.raises(ValueError, match='safe entry bounds'):
        worker.run(context.request)
    assert context.loaded == []


def test_code_reference_and_duplicate_json_never_reach_native_loader(setup_plan):
    context = setup_plan
    for raw in ['{"auto_map":{}}', '{"total_episodes":2,"total_episodes":3}']:
        (context.dataset / 'meta/info.json').write_text(raw)
        with pytest.raises(ValueError):
            worker.run(context.request)
    assert context.loaded == []


def test_plan_does_not_discover_unrelated_data_or_model_payloads(setup_plan, monkeypatch):
    context = setup_plan
    original = worker.os.scandir
    visited = []

    def metadata_only(path):
        visited.append(Path(path))
        assert Path(path).is_relative_to(context.dataset / 'meta/episodes')
        return original(path)

    monkeypatch.setattr(worker.os, 'scandir', metadata_only)
    worker.run(context.request)
    assert visited


def test_fixed_sibling_guard_is_used_under_isolated_import(setup_plan):
    assert worker._guard([setup_plan.episode_file])['execution_status'] == 'complete'
