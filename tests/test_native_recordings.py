"""Keep native LeRobot/robomimic keys and episode boundaries."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.rollouts import compare_rollouts, inspect_rollout


def test_native_lerobot_parquet_keys_and_episode_selection(tmp_path):
    path = tmp_path / 'data.parquet'
    pq.write_table(pa.table({'episode_index': [0, 0, 1, 1], 'frame_index': [0, 1, 0, 1],
                            'timestamp': [0., .1, 0., .1], 'action': [[1., 2.]] * 4,
                            'observation.state': [[3., 4.]] * 4}), path)
    with pytest.raises(InputError, match='episode'):
        compare_rollouts(str(path), str(path), alignment_key='frame_index')
    result = compare_rollouts(str(path), str(path), alignment_key='frame_index',
                              reference_episode=0, candidate_episode=0)
    assert result['status'] == 'pass'
    assert {'action', 'observation.state'} <= {s['key'] for s in result['streams']}
    assert result['artifacts'][0]['episode'] == 0
    cross_episode = compare_rollouts(str(path), str(path), alignment_key='frame_index',
                                    reference_episode=0, candidate_episode=1)
    assert cross_episode['status'] == 'pass' and not cross_episode['observed_difference']
    assert cross_episode['bookkeeping_difference']
    assert cross_episode['first_bookkeeping_difference']['stream'] == 'episode_index'
    assert cross_episode['first_observed_difference'] is None
    with pytest.raises(InputError, match='episode'):
        compare_rollouts(str(path), str(path), reference_episode=7, candidate_episode=0)


def test_native_robomimic_hdf5_keeps_phase_named_keys(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'demo.hdf5'
    with h5py.File(path, 'w') as file:
        for name in ['demo_0', 'demo_1']:
            group = file.create_group('data/' + name)
            group['actions'] = np.ones((3, 7))
            group['states'] = np.ones((3, 20))
            group['obs/robot0_joint_pos'] = np.ones((3, 7))
            group['next_obs/robot0_joint_pos'] = np.ones((3, 7))
    with pytest.raises(InputError, match='episode'):
        inspect_rollout(str(path))
    report = compare_rollouts(str(path), str(path), reference_episode='demo_0', candidate_episode='demo_0')
    assert report['status'] == 'unknown' and not report['observed_difference']
    assert {'actions', 'states', 'obs/robot0_joint_pos', 'next_obs/robot0_joint_pos'} <= {s['key'] for s in report['streams']}
    assert report['alignment']['sample_phase'] == 'unknown'


def test_hdf5_external_links_are_never_followed(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'external.hdf5'
    with h5py.File(path, 'w') as file:
        file['data'] = h5py.ExternalLink('/etc/passwd', '/')
    with pytest.raises(InputError, match='link'):
        inspect_rollout(str(path))


def test_hdf5_virtual_dataset_is_refused(tmp_path):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'virtual.hdf5'
    with h5py.File(path, 'w', libver='latest') as file:
        layout = h5py.VirtualLayout(shape=(3, 2), dtype='f8')
        layout[:] = h5py.VirtualSource(str(tmp_path / 'outside.h5'), 'secret', shape=(3, 2))
        file.create_virtual_dataset('data/demo_0/actions', layout)
    with pytest.raises(InputError, match='virtual'):
        inspect_rollout(str(path), episode='demo_0')


def test_parquet_numeric_budget_checked_before_python_row_conversion(tmp_path, monkeypatch):
    from policy_doctor import rollouts
    path = tmp_path / 'large_vector.parquet'
    pq.write_table(pa.table({'episode_index': [0], 'action': [[1.] * 100]}), path)
    monkeypatch.setattr(rollouts, 'MAX_VALUES', 50)
    with pytest.raises(InputError, match='budget'):
        inspect_rollout(str(path), episode=0)


def test_native_action_cannot_be_excluded_as_an_unsupported_column(tmp_path):
    path = tmp_path / 'missing_action.parquet'
    pq.write_table(pa.table({'episode_index': [0], 'action': ['unsupported']}), path)
    with pytest.raises(InputError, match='numeric action'):
        inspect_rollout(str(path))


def test_parquet_vector_metadata_budget_precedes_arrow_decoding(tmp_path, monkeypatch):
    from policy_doctor import rollouts
    path = tmp_path / 'compressed_vector.parquet'
    pq.write_table(pa.table({'episode_index': [0], 'action': [[0.]],
                            'observation.state': [[1.] * 100]}), path)
    monkeypatch.setattr(rollouts, 'MAX_VALUES', 50)
    def never_decode(*args, **kwargs):
        pytest.fail('Oversized metadata must refuse before Arrow batch decoding')
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', never_decode)
    with pytest.raises(InputError, match='scan budget'):
        inspect_rollout(str(path))
