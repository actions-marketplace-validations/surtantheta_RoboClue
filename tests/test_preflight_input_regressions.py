"""Regressions for observed 0.40 review and first-use failures."""
import json
import os

import pytest

from policy_doctor.artifacts import InputError, sha256
from policy_doctor.preflight import _bound_units, preflight
from policy_doctor.sim_salvage_frontdoor import ReportCache, concise_sim


def _dataset(root, *, rows=3):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    (root / 'meta').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({
        'fps': 10, 'total_frames': rows, 'total_episodes': 1,
        'features': {'action': {'dtype': 'float32', 'shape': [1]},
                     'observation.state': {'dtype': 'float32', 'shape': [1]}},
        'codebase_version': 'v2.1',
    }))
    table = pa.table({
        'episode_index': pa.array([0] * rows, type=pa.int64()),
        'frame_index': pa.array(range(rows), type=pa.int64()),
        'timestamp': pa.array([i / 10 for i in range(rows)], type=pa.float64()),
        'action': pa.array([[float(i)] for i in range(rows)], type=pa.list_(pa.float32())),
        'observation.state': pa.array([[float(i)] for i in range(rows)], type=pa.list_(pa.float32())),
    })
    pq.write_table(table, root / 'data/chunk-000/episode_000000.parquet')
    return root


def test_nonexistent_preflight_names_input(tmp_path):
    missing = tmp_path / 'misspelled-dataset'
    with pytest.raises(InputError, match='does not exist.*misspelled-dataset'):
        preflight(str(missing))


def test_incident_missing_directory_names_original_relative_source(tmp_path, monkeypatch):
    from policy_doctor.incidents import _native_source

    monkeypatch.chdir(tmp_path)
    with pytest.raises(InputError, match="native evidence directory.*'./dataset'"):
        _native_source('./dataset', 'episode_windows')


def test_human_preflight_names_the_requested_hub_source():
    from policy_doctor.cli import human

    text = human({'kind': 'dataset_preflight', 'status': 'unknown', 'source': 'hf://ORG/REPO',
                  'summary': 'Dataset intake could not be completed.', 'components': {},
                  'findings': [], 'coverage': {}, 'next_actions': [], 'scope': 'No training was tested.'})
    assert 'Source: hf://ORG/REPO' in text


def test_healthy_numeric_checks_stay_distinct_from_unsupported_windows(tmp_path):
    root = _dataset(tmp_path / 'data')
    report = preflight(str(root))
    assert report['status'] == 'unknown'
    assert report['components']['numeric_rows']['status'] == 'pass'
    assert 'Passed checks: numeric_rows' in report['summary']
    assert 'episode_windows' in report['summary']
    assert any('native LeRobot v3' in step and 'Do not relabel' in step for step in report['next_actions'])


def test_window_missing_path_names_the_input(tmp_path):
    from policy_doctor.windows import audit_episode_windows

    with pytest.raises(InputError, match='native dataset directory.*misspelled-recording'):
        audit_episode_windows(str(tmp_path / 'misspelled-recording'))


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO control')
def test_sim_frontdoor_names_fifo_before_native_reader(tmp_path):
    from policy_doctor.sim_salvage_frontdoor import _sim_input

    path = tmp_path / 'blocked.hdf5'
    os.mkfifo(path)
    with pytest.raises(InputError, match='blocked.hdf5.*FIFO'):
        _sim_input(str(path), 'source')


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO control')
def test_checksum_refuses_fifo_before_read(tmp_path):
    path = tmp_path / 'blocked.parquet'
    os.mkfifo(path)
    with pytest.raises(InputError, match='blocked.parquet.*FIFO'):
        sha256(path)


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO control')
def test_preflight_refuses_fifo_before_component_readers(tmp_path):
    root = _dataset(tmp_path / 'data')
    os.mkfifo(root / 'data/chunk-000/extra.parquet')
    with pytest.raises(InputError, match='extra.parquet.*FIFO'):
        preflight(str(root))


def test_dataset_info_alias_is_same_input(tmp_path):
    root = _dataset(tmp_path / 'data')
    direct = preflight(str(root))
    alias = preflight(str(root / 'meta/info.json'))
    for key in ('numeric_rows', 'episode_windows', 'video_index'):
        assert alias['components'][key]['status'] == direct['components'][key]['status']
        assert alias['components'][key].get('coverage') == direct['components'][key].get('coverage')


def test_unrelated_file_beside_dataset_never_selects_its_parent(tmp_path):
    root = _dataset(tmp_path / 'data')
    path = root / 'unrelated.csv'
    path.write_text('action\n0\n')
    with pytest.raises(InputError, match='unrelated.csv.*names a file.*dataset directory'):
        preflight(str(path))


def test_empty_row_scan_is_not_a_pass(tmp_path):
    from policy_doctor.dataset import scan_dataset

    report = scan_dataset(str(_dataset(tmp_path / 'empty', rows=0)))
    assert report['status'] == 'error'
    assert report['coverage']['rows_scanned'] == 0
    assert any(item['code'] == 'dataset.empty' for item in report['findings'])


def test_lerobot_alias_cannot_reuse_one_file_cache_identity(tmp_path):
    info = _dataset(tmp_path / 'data') / 'meta/info.json'
    cache = ReportCache()
    identity = cache.identity([str(info)])
    cache.put('selected-input', identity, {'status': 'pass'})
    assert cache.get('selected-input', identity) is None


def test_complete_hdf5_file_cache_still_reuses_unchanged_input(tmp_path):
    path = tmp_path / 'input.hdf5'
    path.write_bytes(b'complete-file-control')
    cache = ReportCache()
    identity = cache.identity([str(path)])
    cache.put('selected-input', identity, {'status': 'warning'})
    assert cache.get('selected-input', identity) == {'status': 'warning'}
    path.write_bytes(b'changed-file-control')
    assert cache.get('selected-input', cache.identity([str(path)])) is None


def test_units_second_reduction_preserves_full_coordinate_count(monkeypatch):
    import policy_doctor.preflight as module

    entries = {
        f'observation.extra_{index}': {
            'names': [f'coordinate_{i}' for i in range(64)],
            'coordinates': [{'index': i, 'unit': 'unknown', 'status': 'unknown', 'motor': None,
                             'evidence': 'bounded control'} for i in range(64)],
        } for index in range(40)
    }
    monkeypatch.setattr(module, 'UNITS_MAX_BYTES', 32 * 1024)
    report = _bound_units({'findings': [], 'manifest': {'features': entries}, 'coverage': {}})
    for entry in report['manifest']['features'].values():
        summary = entry['coordinates_summary']
        assert summary['total'] == 64
        assert summary['listed'] + summary['not_listed'] == 64
        assert entry['names_total'] == 64


def test_small_units_report_keeps_structured_evidence():
    value = {'per_joint': [{'joint': i, 'range': [0, 90]} for i in range(40)]}
    report = {'findings': [{'code': 'units.mismatch', 'status': 'error', 'evidence': [{'value': value}]}],
              'manifest': {'features': {}}, 'coverage': {}}
    actual = _bound_units(report)
    assert actual['findings'][0]['evidence'][0]['value'] == value
    assert isinstance(actual['findings'][0]['evidence'][0]['value'], dict)


def test_concise_sim_never_cuts_remedy_condition():
    condition = 'Condition: establish the source convention before applying this correction.'
    remedy = 'Inspect the selected inputs. ' * 30 + condition
    report = {'kind': 'sim_provenance', 'status': 'warning', 'counts': {'warning': 20},
              'scope': condition + ' Scope: selected saved data only.',
              'findings': [{'code': f'control.{i}', 'status': 'warning', 'message': 'Observed mismatch. ' * 12,
                            'remedy': remedy, 'evidence': []} for i in range(20)]}
    result = concise_sim(report)
    assert condition in result['scope']
    assert result['findings_shown']
    for finding in result['findings_shown']:
        assert finding['remedy'] == remedy
        assert 'remedy' not in finding.get('text_truncated', [])
    assert len(json.dumps(result).encode()) <= 8192


def test_fail_on_warning_counts_a_component_left_out_at_bound(monkeypatch, capsys):
    import policy_doctor.preflight as module
    from policy_doctor import cli

    report = {'kind': 'dataset_preflight', 'status': 'unknown', 'findings': [], 'components': {},
              'coverage': {'components_left_out_at_size_bound': {'action_conventions': {'status': 'warning'}}}}
    monkeypatch.setattr(module, 'preflight', lambda *args, **kwargs: report)
    assert cli.main(['preflight', 'test-input', '--fail-on', 'warning', '--json']) == 1
    capsys.readouterr()


def test_large_conditional_remedy_is_withheld_instead_of_partially_actionable():
    remedy = 'Inspect the input. ' * 400 + 'Only apply after establishing the source declaration.'
    report = {'kind': 'sim_provenance', 'status': 'warning', 'counts': {'warning': 1},
              'scope': 'A result is conditional on the original source declaration. ' * 40,
              'findings': [{'code': 'control.large', 'status': 'warning', 'message': 'Supported conflict.',
                            'remedy': remedy, 'evidence': []}]}
    result = concise_sim(report)
    item = result['findings_shown'][0]
    assert item['status'] == 'warning' and item['remedy_omitted'] is True
    assert item['remedy'].startswith('Full remedy omitted')
    assert 'Inspect the input.' not in item['remedy']
    assert len(item['full_remedy_sha256']) == 64
    assert result['scope'] == report['scope']
