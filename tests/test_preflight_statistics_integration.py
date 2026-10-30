"""Statistics coverage and independent failure containment through the actual preflight workflow."""

import asyncio
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_preflight import dataset
from test_statistics_evidence import dataset_stats, matching, save
from test_windows import audit, make_dataset

from policy_doctor.cli import human, main

engine = importlib.import_module('policy_doctor.preflight')


def statistics(root, value):
    path = root / 'meta/stats.json'
    path.write_text(json.dumps(value))
    return path


def rows_statistics(root, transform):
    """Change actual source rows before deriving a claimed healthy summary."""
    path = root / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    for i, row in enumerate(rows):
        transform(i, row)
    pq.write_table(pa.Table.from_pylist(rows), path)
    result = {}
    for key in ('action', 'observation.state'):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {name: value.tolist() for name, value in {
            'mean': values.mean(0), 'std': values.std(0), 'min': values.min(0),
            'max': values.max(0), 'count': np.asarray([len(rows)])}.items()}
    return result


def test_healthy_saved_stats_are_separate_scoped_component(tmp_path):
    root = dataset(tmp_path)
    result = engine.preflight(str(root))
    part = result['components']['dataset_statistics']
    assert result['status'] == part['status'] == 'pass'
    assert part['coverage']['complete'] and not part['coverage']['observed_row_statistics_compared']
    assert set(part['coverage']['features_considered']) == {'action', 'observation.state'}
    assert len(part['artifacts']) == 2
    stat = next(a for a in result['artifacts'] if a['role'] == 'dataset_stats')
    assert stat['sha256'] == hashlib.sha256((root / 'meta/stats.json').read_bytes()).hexdigest()
    assert 'conditional' in part['coverage']['normalization_mode']


def test_missing_statistics_are_unknown_without_hiding_independent_error(tmp_path):
    root = dataset(tmp_path, bad_span=True)
    (root / 'meta/stats.json').unlink()
    result = engine.preflight(str(root))
    assert result['status'] == result['components']['episode_windows']['status'] == 'error'
    part = result['components']['dataset_statistics']
    assert part['status'] == 'unknown' and not part['coverage']['complete']
    assert 'absent' in part['findings'][0]['message'] and part['artifacts'][0]['role'] == 'dataset_info'


@pytest.mark.parametrize('raw', ['{bad json', '{"action":NaN}', '[1,2]', '{"action":{}}'])
def test_bad_statistics_preserve_raw_identity_and_row_error(tmp_path, raw):
    root = dataset(tmp_path, bad_vector=True, bad_span=True)
    path = root / 'meta/stats.json'
    path.write_text(raw)
    result = engine.preflight(str(root))
    part = result['components']['dataset_statistics']
    assert result['status'] == result['components']['episode_windows']['status'] == 'error'
    assert result['components']['numeric_rows']['status'] == ('error' if raw == '{"action":{}}' else 'unknown')
    assert part['status'] == 'unknown' and not part['coverage']['complete']
    artifact = next(a for a in part['artifacts'] if a['role'] == 'dataset_stats')
    assert artifact['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', [['a', 'b'], [10**400, 1.], [True, 1.], [[1., 2.]], [1., 2., 3.]])
def test_unsupported_stat_vectors_are_unknown_not_silent_pass(tmp_path, bad):
    root = dataset(tmp_path)
    stats = json.loads((root / 'meta/stats.json').read_text())
    stats['action']['std'] = bad
    statistics(root, stats)
    report = engine.preflight(str(root))
    part = report['components']['dataset_statistics']
    assert part['status'] == 'unknown' and part['coverage']['unassessed']
    assert report['components']['numeric_rows']['status'] == 'unknown'
    assert report['components']['numeric_rows']['statistics_comparison']['unassessed_metrics'] == {'action': ['std']}
    assert any(f['code'] == 'dataset.statistics_values_unassessed'
               for f in report['components']['numeric_rows']['findings'])
    json.dumps(report, allow_nan=False)


def test_constants_collapsed_quantiles_and_invalid_spreads_preserve_remedies(tmp_path):
    root = dataset(tmp_path)
    stats = rows_statistics(root, lambda i, row: row['action'].__setitem__(0, 1.))
    stats['observation.state'].update(q01=[1., 2.], q99=[1., 21.])
    statistics(root, stats)
    result = engine.preflight(str(root))
    part = result['components']['dataset_statistics']
    assert part['status'] == result['status'] == 'warning'
    hits = {f['code']: f for f in result['findings'] if f['component'] == 'dataset_statistics'}
    assert hits['dataset.constant_dimension']['indices'] == [0]
    assert 'unused' in hits['dataset.constant_dimension']['message']
    assert hits['dataset.collapsed_quantiles']['indices'] == [0]
    assert 'Other modes are unaffected' in hits['dataset.collapsed_quantiles']['message']
    assert all(f['remedy'] and f['evidence'][0]['path'].endswith('meta/stats.json') for f in hits.values())
    text = human(result)
    assert 'Next step: Check these dimensions' in text and 'Next step: Before training a quantile-normalized' in text
    stats = json.loads((root / 'meta/stats.json').read_text())
    stats['action']['std'] = [-1., 1.]
    statistics(root, stats)
    result = engine.preflight(str(root))
    assert any(f['code'] == 'dataset.statistics_invalid' for f in result['findings'])


@pytest.mark.parametrize('name,component', [('scan_dataset', 'numeric_rows'),
    ('audit_episode_windows', 'episode_windows'), ('_dataset_statistics', 'dataset_statistics'),
    ('_video_component', 'video_index'), ('_checkpoint', 'checkpoint_metadata')])
@pytest.mark.parametrize('exception', [TypeError('reader type failure'), RuntimeError('reader runtime failure')])
def test_reader_exception_is_local_unknown_and_preserves_independent_error(tmp_path, monkeypatch, name, component, exception):
    root = dataset(tmp_path, bad_span=True, bad_vector=True)
    def broken(*a, **k):
        raise exception
    monkeypatch.setattr(engine, name, broken)
    report = engine.preflight(str(root), checkpoint=str(tmp_path / 'missing-checkpoint'))
    assert report['components'][component]['status'] == 'unknown'
    assert 'reader ' in report['components'][component]['findings'][0]['message']
    assert report['status'] == 'error'
    assert any(f['status'] == 'error' for f in report['findings'])


@pytest.mark.parametrize('name', ['scan_dataset', 'audit_episode_windows', '_dataset_statistics', '_video_component'])
def test_keyboard_interrupt_is_not_converted_to_unknown(tmp_path, monkeypatch, name):
    root = dataset(tmp_path)
    def interrupted(*a, **k):
        raise KeyboardInterrupt('stop requested')
    monkeypatch.setattr(engine, name, interrupted)
    with pytest.raises(KeyboardInterrupt):
        engine.preflight(str(root))


def test_checkpoint_statistics_comparison_is_exposed_by_preflight(tmp_path, case):
    root = dataset(tmp_path)
    # Match actual declarations to the existing checkpoint fixture, then plant a saved-state shift.
    info = json.loads((root / 'meta/info.json').read_text())
    fixture_info = json.loads((case['dataset'] / 'meta/info.json').read_text())
    info['features'] = fixture_info['features']
    (root / 'meta/info.json').write_text(json.dumps(info))
    dataset_stats({'dataset': root}, matching())
    save(case, 'pre', observation_state_mean=[20., .5], observation_state_std=[6., .25])
    result = engine.preflight(str(root), checkpoint=str(case['checkpoint']))
    hits = [f for f in result['findings'] if f['code'] == 'normalization.dataset_statistics' and f['message'].startswith('pre:')]
    assert hits and hits[0]['status'] == 'warning' and hits[0]['component'] == 'checkpoint_metadata'
    assert hits[0]['remedy'] in human(result)


def test_window_query_human_declared_error_and_healthy_status(tmp_path):
    bad = tmp_path / 'bad'
    bad.mkdir()
    report = audit(str(make_dataset(bad, bad=True)), episode_index=0, anchor_index=29, delta_indices=[0,1,2])
    text = human(report)
    assert 'declared: status ERROR' in text and 'Reason: ' + report['query']['declared']['reason'] in text
    assert 'corrected_what_if: status PASS' in text
    healthy = tmp_path / 'healthy'
    healthy.mkdir()
    report = audit(str(make_dataset(healthy)), episode_index=0, anchor_index=29, delta_indices=[0,1,2])
    assert 'declared: status PASS' in human(report) and 'Reason: None' not in human(report)


def test_cli_and_actual_stdio_statistics_report_remain_unknown_on_missing_input(tmp_path, capsys):
    root = dataset(tmp_path)
    (root / 'meta/stats.json').unlink()
    assert main(['preflight', str(root), '--json']) == 3
    assert json.loads(capsys.readouterr().out)['components']['dataset_statistics']['status'] == 'unknown'
    async def run():
        params = StdioServerParameters(command=sys.executable, args=['-B', '-m', 'policy_doctor', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                for format in ('concise', 'detailed'):
                    report = await client.call_tool('preflight_check', {'source': str(root), 'response_format': format})
                    assert not report.isError and report.structuredContent['status'] == 'unknown'
                    assert report.structuredContent['components']['dataset_statistics']['status'] == 'unknown'
    asyncio.run(asyncio.wait_for(run(), 30))


def test_actual_default_concise_sdk_keeps_statistics_mode_and_epsilon_uncertainty(tmp_path):
    cases = []
    for name, expected in [('constant', 'dataset.constant_dimension'),
                           ('small', 'dataset.small_spread'), ('quantiles', 'dataset.collapsed_quantiles')]:
        root = dataset(tmp_path / name)
        if name == 'constant':
            stats = rows_statistics(root, lambda i, row: row['action'].__setitem__(0, 1.))
        elif name == 'small':
            stats = rows_statistics(root, lambda i, row: row['action'].__setitem__(0, i * 5e-7 / 19))
        else:
            stats = json.loads((root / 'meta/stats.json').read_text())
            stats['action'].update(q01=[1., 1.], q99=[1., 20.])
        statistics(root, stats)
        cases.append((root, expected))
    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'policy_doctor', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                for root, code in cases:
                    # Omit response_format to exercise the actual default concise response.
                    short = await client.call_tool('preflight_check', {'source': str(root)})
                    full = await client.call_tool('preflight_check', {'source': str(root), 'response_format': 'detailed'})
                    assert not short.isError and not full.isError
                    body = short.structuredContent
                    assert body['summary'] and body['status'] == full.structuredContent['status'] == 'warning'
                    concise = next(f for f in body['findings_shown'] if f['code'] == code)
                    detailed = next(f for f in full.structuredContent['findings'] if f['code'] == code)
                    for finding in (concise, detailed):
                        assert finding['message'].startswith('The active normalization mode and epsilon are not established')
                        assert 'IDENTITY leaves values unchanged.' in finding['message']
                        assert finding['status'] == 'warning' and finding['remedy']
                    assert len(concise['message']) <= 400
                    assert len(json.dumps(body, allow_nan=False).encode()) <= 8192
    asyncio.run(asyncio.wait_for(run(), 40))
