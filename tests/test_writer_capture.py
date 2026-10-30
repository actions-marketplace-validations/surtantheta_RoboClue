"""Generated writer mechanics only; original-source/type gates explicitly replaced."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_named_capture import actors, send_next_action  # noqa: F401

from policy_doctor import named_capture as nc
from policy_doctor import writer_capture as wc


class DatasetWriter:
    def __init__(self, root):
        self.root = root
        self._meta = SimpleNamespace(total_episodes=0, total_frames=0)
        self.episode_buffer = None
        self._finalized = False

    def add_frame(self, frame):
        if self.episode_buffer is None:
            self.episode_buffer = {'size': 0, 'episode_index': 0, 'frame_index': [], 'action': []}
        buffer = self.episode_buffer
        buffer['frame_index'].append(buffer['size'])
        frame.pop('task')
        buffer['action'].append(frame['action'])
        buffer['size'] += 1

    def save_episode(self, episode_data=None, parallel_encoding=True):
        for name in wc.DATASET_PATHS:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'Generated persistence mechanics; not native JSON or Parquet')
        self._meta.total_frames = self.episode_buffer['size']
        self._meta.total_episodes = 1
        self.episode_buffer = None

    def finalize(self):
        self._finalized = True


class LeRobotDataset:
    def __init__(self, root):
        self.root = root
        self.root.mkdir()
        self.features = {'action': {'names': ['a.pos', 'b.pos'], 'dtype': 'float32', 'shape': (2,)}}
        self.writer = DatasetWriter(root)
        self._is_finalized = False

    def add_frame(self, frame):
        self.writer.add_frame(frame)

    def save_episode(self, episode_data=None, parallel_encoding=True):
        self.writer.save_episode(episode_data, parallel_encoding)

    def finalize(self):
        self.writer.finalize()
        self._is_finalized = True


def build_dataset_frame(ds_features, values, prefix):
    return {'action': np.array([values[n] for n in ds_features['action']['names']], dtype=np.float32)}


class EpisodicStrategy:
    def _policy_loop(self, ctx, robot, events, features, timer, control_time_s, dataset):
        interpolator = self.interpolator
        for _ in range(2):
            action_dict = send_next_action({}, {}, ctx, interpolator)
            action_frame = build_dataset_frame(features, action_dict, prefix='action')
            dataset.add_frame({**action_frame, 'task': 'generated writer mechanics'})


@pytest.fixture
def recording(actors, tmp_path, monkeypatch):  # noqa: F811 - imported pytest fixture injection
    ctx, interp, backend, processor, engine = actors
    dataset = LeRobotDataset(tmp_path / 'native')
    ctx.data.dataset = dataset
    ctx.data.dataset_features = dataset.features
    strategy = EpisodicStrategy()
    strategy.interpolator = interp
    definition = nc._definition(ctx, interp)
    definition['loaded_bindings'] = []
    names = ('episodic.py', 'feature_utils.py', 'lerobot_dataset.py', 'dataset_writer.py')
    raw = open(__file__, 'rb').read()
    monkeypatch.setattr(wc, 'RECORDING_SOURCE_MODULES', dict.fromkeys(names, __name__))
    monkeypatch.setattr(wc, 'RECORDING_SOURCE_HASHES', dict.fromkeys(names, hashlib.sha256(raw).hexdigest()))
    monkeypatch.setattr(nc, '_code', lambda function, raw: None)
    return ctx, interp, dataset, strategy


def call_loop(recording):
    ctx, _, dataset, strategy = recording
    strategy._policy_loop(ctx, ctx.hardware.robot_wrapper, {}, ctx.data.dataset_features, None, 1, dataset)


def context(recording, path):
    ctx, interp, _, _ = recording
    return nc.capture_native_actions(ctx, interp, path, driver_namespace='generated/test', observe_recording=True)


def test_normal_writer_identity_values_lifecycle_and_inert_native_copy(recording, tmp_path):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        recording[2].save_episode()
        recording[2].finalize()
    assert cap.receipt['published'], cap.receipt
    data = json.loads((tmp_path / 'capture/capture.json').read_text())
    assert data['schema_version'] == 2 and len(data['recording']['rows']) == 2
    for i, row in enumerate(data['recording']['rows']):
        assert row['index'] == row['frame_index'] == row['tick_id'] == i
        assert row['source_role'] == 'canonical_return'
        assert row['frame_action'] == row['action'] == row['buffer_action'] == [2., -3.]
        assert all(row['relations'].values()) and row['facade_completed'] and row['writer_completed']
    lifecycle = data['recording']['lifecycle']
    assert lifecycle['save_episode_count'] == lifecycle['finalize_count'] == 1
    assert all(v is True for k, v in lifecycle.items() if not k.endswith('_count'))
    assert len(data['recording']['dataset_files']) == 5
    for entry in data['recording']['dataset_files']:
        relative = entry['path'].removeprefix('dataset/')
        assert (tmp_path / 'capture' / entry['path']).read_bytes() == (recording[2].root / relative).read_bytes()
    assert sys.getprofile() is None


def test_buffer_only_never_invents_save_or_persistence(recording, tmp_path):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
    assert cap.receipt['published']
    data = json.loads((tmp_path / 'capture/capture.json').read_text())
    assert len(data['recording']['rows']) == 2
    assert not data['recording']['lifecycle']['writer_save_completed']
    assert data['recording']['dataset_files'] == []


def test_same_array_value_mutation_before_save_refuses_without_changing_caller(recording, tmp_path):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        recording[2].writer.episode_buffer['action'][0][0] = 99
        recording[2].save_episode()
        recording[2].finalize()
    assert not cap.receipt['published']
    assert 'values changed before saving' in cap.receipt['reason']
    assert recording[2]._is_finalized


def test_equal_values_different_array_before_save_does_not_preserve_origin(recording, tmp_path):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        buffer = recording[2].writer.episode_buffer
        buffer['action'][0] = buffer['action'][0].copy()
        recording[2].save_episode()
        recording[2].finalize()
    assert not cap.receipt['published']


def test_derived_episode_override_not_joined_to_original_buffer(recording, tmp_path):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        recording[2].save_episode(episode_data={'unrelated': True})
        recording[2].finalize()
    assert not cap.receipt['published'] and 'supplied episode' in cap.receipt['reason']


@pytest.mark.parametrize('change', ['extra', 'symlink', 'missing', 'empty-directory'])
def test_full_native_tree_is_preserved_or_refused(recording, tmp_path, change):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        recording[2].save_episode()
        recording[2].finalize()
        path = recording[2].root / 'meta/stats.json'
        if change == 'extra':
            (recording[2].root / 'unexpected.bin').write_bytes(b'not silently discarded')
        elif change == 'empty-directory':
            (recording[2].root / 'unmodeled-empty-tree').mkdir()
        elif change == 'missing':
            path.unlink()
        else:
            path.unlink()
            path.symlink_to(recording[2].root / 'meta/info.json')
    assert not cap.receipt['published'] and not (tmp_path / 'capture').exists()


def test_new_source_file_during_copy_is_not_silently_omitted(recording, tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.write_bytes

    def write(path, data):
        result = original(path, data)
        if '/.named-action-' in str(path) and '/dataset/' in str(path):
            original(recording[2].root / 'late-extra.bin', b'late source file')
        return result

    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        recording[2].save_episode()
        recording[2].finalize()
        monkeypatch.setattr(Path, 'write_bytes', write)
    assert not cap.receipt['published'] and not (tmp_path / 'capture').exists()


def test_existing_episode_refused_before_observation(recording, tmp_path):
    recording[2].writer._meta.total_episodes = 1
    with pytest.raises(Exception, match='fresh empty'):
        with context(recording, tmp_path / 'capture'):
            pass
    assert sys.getprofile() is None


def test_writer_replaced_after_observed_loop_refuses(recording, tmp_path):
    with context(recording, tmp_path / 'capture') as cap:
        call_loop(recording)
        recording[2].writer = DatasetWriter(recording[2].root)
    assert not cap.receipt['published']
