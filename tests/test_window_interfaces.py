import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def native_dataset(root):
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'fps': 10, 'total_episodes': 2, 'total_frames': 3,
        'features': {'action': {'dtype': 'float32', 'shape': [2], 'names': None}},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'}))
    pq.write_table(pa.Table.from_pylist([
        {'index': 0, 'episode_index': 0, 'frame_index': 0, 'action': [0., 0.]},
        {'index': 1, 'episode_index': 0, 'frame_index': 1, 'action': [0., 1.]},
        {'index': 2, 'episode_index': 1, 'frame_index': 0, 'action': [1., 0.]},
    ]), root / 'data/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist([
        {'episode_index': 0, 'dataset_from_index': 0, 'dataset_to_index': 3, 'length': 2,
         'data/chunk_index': 0, 'data/file_index': 0},
        {'episode_index': 1, 'dataset_from_index': 2, 'dataset_to_index': 3, 'length': 1,
         'data/chunk_index': 0, 'data/file_index': 0},
    ]), root / 'meta/episodes/chunk-000/file-000.parquet')


def test_native_window_cli_and_markdown(tmp_path):
    root = tmp_path / 'native'
    native_dataset(root)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    args = [sys.executable, '-m', 'policy_doctor', 'audit-windows', str(root),
            '--episode-index', '0', '--anchor-index', '1', '--delta-indices', '0', '1', '2']
    result = subprocess.run(args, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1, result.stderr
    assert 'episode windows: error' in result.stdout
    assert 'Metadata-only candidate changes' in result.stdout
    assert 'action_is_pad' in result.stdout
    report = subprocess.run([*args, '--markdown'], env=env, text=True, capture_output=True, timeout=30)
    assert report.returncode == 1, report.stderr
    assert 'Native episode-window investigation' in report.stdout
    assert 'dataset\\_to\\_index' in report.stdout


def test_native_window_actual_read_only_sdk_and_strict_arguments(tmp_path):
    root = tmp_path / 'allowed'
    native_dataset(root)
    outside = tmp_path / 'outside'
    native_dataset(outside)

    async def run():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        server = StdioServerParameters(command=sys.executable,
            args=['-m', 'policy_doctor', 'serve', '--root', str(root)], env=env)
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool = next(t for t in tools.tools if t.name == 'audit_dataset_windows')
                assert tool.annotations.readOnlyHint
                args = {'dataset': str(root), 'episode_index': 0, 'anchor_index': 1, 'delta_indices': [0, 1, 2]}
                result = await session.call_tool(tool.name, args)
                assert not result.isError
                data = json.loads(result.content[0].text)
                assert data['status'] == 'error'
                assert data['query']['declared']['selected_episode_indices'] == [0, 1, 1]
                assert data['query']['corrected_what_if']['action_is_pad'] == [False, True, True]
                for changed in ({'episode_index': True}, {'anchor_index': False},
                                {'delta_indices': [0, True]}, {'max_rows': True}, {'max_files': False}):
                    refused = await session.call_tool(tool.name, dict(args, **changed))
                    assert refused.isError, changed
                denied = await session.call_tool(tool.name, dict(args, dataset=str(outside)))
                assert denied.isError

    asyncio.run(run())
