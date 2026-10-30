"""The public entry point must reach the checks and preserve refusals across CLI/MCP."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_preflight import dataset

from policy_doctor import __version__
from policy_doctor.cli import human, main
from policy_doctor.server import create_server


@pytest.mark.parametrize('form', ['--json', '--markdown', None])
def test_preflight_cli_reports_actual_native_defect(tmp_path, capsys, form):
    root = dataset(tmp_path, bad_vector=True)
    code = main(['preflight', str(root)] + ([form] if form else []))
    output = capsys.readouterr().out
    assert code == 1
    assert ('dataset.vector\\_invalid' if form == '--markdown' else 'dataset.vector_invalid') in output
    if form == '--json':
        report = json.loads(output)
        assert report['components']['numeric_rows']['status'] == 'error'
    elif form == '--markdown':
        assert output.startswith('# RoboClue dataset preflight')
    else:
        assert output.startswith('RoboClue preflight: ERROR')


def test_cli_incomplete_preflight_is_unknown_and_fail_on_is_explicit(tmp_path, capsys):
    root = dataset(tmp_path)
    assert main(['preflight', str(root), '--max-rows', '1', '--json']) == 3
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'
    assert main(['preflight', str(root), '--max-rows', '1', '--fail-on', 'unknown', '--json']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'


def test_human_scan_handles_dict_evidence_without_crashing():
    result = {'status': 'error', 'findings': [{'code': 'dataset.total_frames', 'detail': 'declared count differs',
        'evidence': {'path': '/tmp/meta/info.json', 'pointer': '/total_frames', 'sha256': 'a' * 64}}]}
    output = human(result)
    assert 'dataset.total_frames' in output and '/tmp/meta/info.json#/total_frames' in output


def test_template_default_output_is_valid_json(case, capsys):
    assert main(['template', str(case['checkpoint'])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['schema_version'] == 1 and result['features']


def test_hub_dataset_preflight_requires_distinct_network_permission(tmp_path, monkeypatch):
    import policy_doctor.preflight as engine

    calls = []
    monkeypatch.setattr(engine, 'preflight', lambda *args, **kw: calls.append((args, kw)))

    async def run():
        for options in [{}, {'allow_hub': True}]:
            server = create_server([str(tmp_path)], **options)
            with pytest.raises(Exception, match='allow-hub-datasets'):
                await server.call_tool('preflight_check', {'source': 'hf://org/data'})
        server = create_server([str(tmp_path)], allow_hub_datasets=True)
        with pytest.raises(Exception, match='outside'):
            await server.call_tool('preflight_check', {'source': 'hf://org/data', 'checkpoint': '/etc'})
    asyncio.run(run())
    assert calls == []


def test_actual_stdio_preflight_and_unknown_arguments(tmp_path):
    root = dataset(tmp_path, bad_span=True)

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                assert initialized.serverInfo.version == __version__
                tools = (await client.list_tools()).tools
                assert len(tools) == 29
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                tool = next(t for t in tools if t.name == 'preflight_check')
                assert tool.inputSchema['additionalProperties'] is False
                full = await client.call_tool('preflight_check', {'source': str(root), 'response_format': 'detailed'})
                assert not full.isError and full.structuredContent['status'] == 'error'
                concise = await client.call_tool('preflight_check', {'source': str(root)})
                assert not concise.isError and concise.structuredContent['status'] == 'error'
                assert concise.structuredContent['next_actions']
                assert concise.structuredContent['components']['episode_windows']['status'] == 'error'
                typo = await client.call_tool('preflight_check', {'source': str(root), 'max_rowz': 1})
                assert typo.isError and 'max_rowz' in typo.content[0].text
                boolean = await client.call_tool('preflight_check', {'source': str(root), 'max_download_mb': True})
                assert boolean.isError
                outside = await client.call_tool('preflight_check', {'source': '/etc'})
                assert outside.isError
    asyncio.run(asyncio.wait_for(run(), timeout=40))


def test_retained_window_preview_preserves_cross_episode_reason(tmp_path):
    from test_windows import make_dataset

    from policy_doctor.incidents import _summary
    from policy_doctor.windows import audit_episode_windows

    root = make_dataset(tmp_path, bad=True)
    result = audit_episode_windows(str(root), episode_index=0, anchor_index=29, delta_indices=[0, 1, 2])
    assert result['query']['declared']['status'] == 'error'
    preview = _summary(result)
    assert preview['query']['declared']['reason'] == result['query']['declared']['reason']
    assert preview['query']['corrected_what_if']['status'] == 'pass'


def test_concise_summary_handles_original_native_dictionary_evidence():
    from policy_doctor.server import _summary

    report = {'kind': 'dataset_preflight', 'status': 'error', 'findings': [
        {'code': 'episode.declared_totals', 'status': 'error', 'message': 'Count differs',
         'evidence': {'path': 'hf://org/data@' + 'a' * 40 + '/meta/info.json',
                      'pointer': '/total_frames', 'sha256': 'b' * 64, 'value': 0}}],
        'next_actions': ['Inspect the cited totals.']}
    concise = _summary(report)
    assert concise['status'] == 'error'
    assert concise['findings_shown'][0]['evidence'][0]['pointer'] == '/total_frames'
    assert concise['next_actions'] == report['next_actions']


def test_known_episode_error_survives_refused_video_component(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_video_index import KEY
    from test_video_index import dataset as video_dataset

    from policy_doctor.preflight import preflight

    root = video_dataset(tmp_path, [(3, 0, 0.0), (2, 0, .6)], {0: 5}, with_data=True)
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    rows = pq.read_table(metadata).to_pylist()
    rows[0]['dataset_to_index'] += 1
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    data = root / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(data).to_pylist()
    for row in rows:
        row['timestamp'] = row['frame_index'] / 5
    pq.write_table(pa.Table.from_pylist(rows), data)
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'must not be read')
    video = root / 'videos' / KEY / 'chunk-000/file-000.mp4'
    video.unlink()
    video.symlink_to(outside)
    report = preflight(str(root))
    assert report['components']['numeric_rows']['status'] == 'pass'
    assert report['components']['episode_windows']['status'] == 'error'
    assert report['components']['video_index']['status'] == 'unknown'
    assert report['status'] == 'error'
    assert any(f['code'] == 'episode.range_contradiction' for f in report['findings'])
    assert any(f['code'] == 'video.coverage_unknown' for f in report['findings'])
