"""Front doors for the simulator provenance guards and recording salvage: the CLI through ``python -m policy_doctor``
(every output mode and exit code), the real stdio MCP server (schemas, refusals, bounds, concurrency) and complete
remedy-list pages."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import sim_salvage_frontdoor as fd
from policy_doctor.salvage import check_salvage
from policy_doctor.server import create_server
from policy_doctor.sim_guards import check_sim_provenance, remedy_list

SRC = Path(__file__).parents[1] / 'src'
ENV = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1')
CAM = 'observation.images.cam'
DELTA_OSC = {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
             'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'control_delta': True}


def cli(*args, timeout=600):
    return subprocess.run([sys.executable, '-m', 'policy_doctor', *map(str, args)], capture_output=True, text=True,
                          env=ENV, timeout=timeout)


def refused(run, *needles):
    assert run.returncode == 2, (run.returncode, run.stdout[-500:], run.stderr[-2000:])
    assert 'Traceback' not in run.stderr and run.stdout == ''
    error = json.loads(run.stderr.strip().splitlines()[-1])['error']
    assert error['code'] == 'invalid_input', error
    for needle in needles:
        assert needle in error['message'], error['message']
    return error['message']


# ------------------------------------------------------------------------------------------------ fixtures


def yaw(deg):
    h = np.radians(deg) / 2
    return [np.cos(h), 0, 0, np.sin(h)]  # WXYZ


def legacy_isaac(path, *, demos=2, cams=3):
    """A legacy (WXYZ) Isaac Lab recording: identity robot roots decide WXYZ; camera orientations are not root_pose."""
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({'env_name': 'Isaac-Front-v0', 'type': 2})
        for i in range(demos):
            g = data.create_group(f'demo_{i}')
            g.create_dataset('actions', data=np.full((2, 7), float(i)))
            g.create_dataset('initial_state/articulation/robot/root_pose', data=np.array([[0, 0, 0, 1.0, 0, 0, 0]]))
            for c in range(cams):
                g.create_dataset(f'initial_state/cameras/camera_{c:04d}/orientation', data=np.array([yaw(3 + 0.7 * c)]))
    return path


def partial_pair(directory, demos, cams, seed=11):
    """Source legacy file and a converted copy stamped 1 where root_pose and a random half of the cameras of each demo
    were reordered to XYZW: per-demo fixes for every camera (the validator's worst case for list size)."""
    directory.mkdir(parents=True, exist_ok=True)
    src = legacy_isaac(directory / f'src_{demos}x{cams}.hdf5', demos=demos, cams=cams)
    dst = directory / f'dst_{demos}x{cams}.hdf5'
    dst.write_bytes(src.read_bytes())
    rng = np.random.default_rng(seed)
    with h5py.File(dst, 'r+') as f:
        f.attrs['format_version'] = 1
        for i in range(demos):
            g = f[f'data/demo_{i}']
            root = g['initial_state/articulation/robot/root_pose']
            value = root[...]
            value[..., 3:7] = np.roll(value[..., 3:7], -1, axis=-1)
            root[...] = value
            for c in range(cams):
                if rng.random() < 0.5:
                    ds = g[f'initial_state/cameras/camera_{c:04d}/orientation']
                    ds[...] = np.roll(ds[...], -1, axis=-1)
    return src, dst


def robomimic(path, *, kind='delta', env_args=True, demos=3, steps=60):
    rng = np.random.default_rng(1)
    meta = {'env_name': 'Lift', 'type': 1, 'env_version': '1.5.1',
            'env_kwargs': {'robots': ['Panda'], 'control_freq': 20, 'controller_configs': dict(DELTA_OSC)}}
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        if env_args:
            data.attrs['env_args'] = json.dumps(meta)
        for d in range(demos):
            g = data.create_group(f'demo_{d}')
            p = np.zeros((steps, 3))
            p[0] = [-0.1, 0.0, 1.0]
            delta = np.clip(np.cumsum(rng.normal(scale=0.3, size=(steps, 3)), 0) / 4, -1, 1)
            for t in range(steps - 1):
                p[t + 1] = p[t] + 0.005 * delta[t] + rng.normal(scale=1e-5, size=3)
            rot = rng.uniform(-0.2, 0.2, size=(steps, 3))
            grip = np.where(np.arange(steps) > steps // 2, 1.0, -1.0)[:, None]
            target = np.vstack([p[1:], p[-1:]]) + rng.normal(scale=2e-3, size=(steps, 3))
            absolute = np.concatenate([target, np.tile([3.0, 0.1, 0.0], (steps, 1)), grip], 1)
            g.create_dataset('actions', data=np.concatenate([delta, rot, grip], 1) if kind == 'delta' else absolute)
            g.create_dataset('obs/robot0_eef_pos', data=p)
            g.create_dataset('obs/robot0_eef_quat', data=np.tile([1.0, 0, 0, 0], (steps, 1)))
            g.create_dataset('states', data=rng.normal(size=(steps, 10)))
    return path


def rows_for(episodes):
    rows, index = [], 0
    for ep, length in episodes:
        for frame in range(length):
            rows.append({'index': index, 'episode_index': ep, 'frame_index': frame, 'timestamp': frame / 10,
                         'task_index': 0, 'action': [float(ep), float(frame)]})
            index += 1
    return rows


def meta_for(episodes, video=False):
    out, position = [], 0
    for ep, length in episodes:
        item = {'episode_index': ep, 'length': length, 'dataset_from_index': position,
                'dataset_to_index': position + length, 'data/chunk_index': 0, 'data/file_index': 0,
                'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0}
        if video:
            item.update({f'videos/{CAM}/chunk_index': 0, f'videos/{CAM}/file_index': 0,
                         f'videos/{CAM}/from_timestamp': position / 10,
                         f'videos/{CAM}/to_timestamp': (position + length) / 10})
        out.append(item)
        position += length
    return out


def lerobot_recording(root, rows, meta, total_episodes, total_frames, *, video=False):
    (root / 'meta').mkdir(parents=True)
    features = {'action': {'dtype': 'float32', 'shape': [2]}, 'index': {'dtype': 'int64', 'shape': [1]}}
    if video:
        features[CAM] = {'dtype': 'video', 'shape': [8, 8, 3], 'names': ['height', 'width', 'channels']}
    info = {'codebase_version': 'v3.0', 'total_episodes': total_episodes, 'total_frames': total_frames, 'fps': 10,
            'total_tasks': 1, 'chunks_size': 1000, 'splits': {'train': f'0:{total_episodes}'},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4' if video else None,
            'features': features}
    (root / 'meta' / 'info.json').write_text(json.dumps(info))
    committed = rows[:total_frames]
    statistics = {}
    for key, width in [('action', 2), ('index', 1)]:
        values = np.asarray([row[key] for row in committed], dtype=np.float64).reshape(len(committed), width)
        if len(values):
            statistics[key] = {name: value.tolist() for name, value in {
                'mean': values.mean(0), 'std': values.std(0), 'min': values.min(0),
                'max': values.max(0), 'count': np.asarray([len(values)])}.items()}
    (root / 'meta' / 'stats.json').write_text(json.dumps(statistics))
    pq.write_table(pa.table({'task_index': [0], 'task': ['t']}), root / 'meta' / 'tasks.parquet')
    data = root / 'data' / 'chunk-000' / 'file-000.parquet'
    data.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data, row_group_size=max(1, len(rows)))
    episodes = root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
    episodes.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(meta), episodes)
    return root


def healthy_recording(root, *, video=False):
    episodes = [(0, 5), (1, 6), (2, 7)]
    return lerobot_recording(root, rows_for(episodes), meta_for(episodes, video), 3, 18, video=video)


def interrupted_recording(root):
    """Native I2/I3/I7 signature: rows of episode 2 written, its metadata and the totals not."""
    return lerobot_recording(root, rows_for([(0, 5), (1, 6), (2, 7)]), meta_for([(0, 5), (1, 6)]), 2, 11)


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


@pytest.fixture
def sim_inputs(tmp_path):
    return {'pass': robomimic(tmp_path / 'delta.hdf5'), 'error': robomimic(tmp_path / 'abs.hdf5', kind='abs'),
            'unknown': robomimic(tmp_path / 'noenv.hdf5', env_args=False),
            'warning': legacy_isaac(tmp_path / 'legacy.hdf5', cams=60)}


# ------------------------------------------------------------------------------------------------ CLI


def test_cli_help_lists_both_commands_and_every_option():
    root = cli('--help')
    assert root.returncode == 0 and 'check-sim-data' in root.stdout and 'check-salvage' in root.stdout
    sim = cli('check-sim-data', '--help')
    assert sim.returncode == 0
    for option in ('--converted', '--max-demos', '--max-rows', '--json', '--markdown', '--output', '--fail-on'):
        assert option in sim.stdout
    salvage = cli('check-salvage', '--help')
    assert salvage.returncode == 0
    for option in ('--video-headers', '--no-video-headers', '--max-rows', '--max-files', '--json', '--markdown',
                   '--output', '--fail-on'):
        assert option in salvage.stdout
    assert 'writes nothing' in ' '.join(salvage.stdout.split()) and 'offline' in ' '.join(sim.stdout.split())


@pytest.mark.parametrize('status, code', [('pass', 0), ('warning', 0), ('error', 1), ('unknown', 3)])
def test_cli_sim_exit_codes_follow_the_cli_conventions(sim_inputs, status, code):
    run = cli('check-sim-data', sim_inputs[status])
    assert run.returncode == code, run.stderr[-2000:]
    assert run.stdout.startswith(f'RoboClue simulator data provenance: {status.upper()}') and run.stderr == ''
    gates = {'error': {'error'}, 'warning': {'error', 'warning'}, 'unknown': {'error', 'warning', 'unknown'}}
    statuses = {f['status'] for f in check_sim_provenance(str(sim_inputs[status]))['findings']}
    for gate, failing in gates.items():
        assert cli('check-sim-data', sim_inputs[status], '--fail-on', gate).returncode == int(bool(statuses & failing))


def test_cli_sim_output_modes_carry_the_complete_report(sim_inputs, tmp_path):
    source = sim_inputs['warning']
    report = check_sim_provenance(str(source))
    entries = remedy_list(report, 'source.non_root_quaternion_datasets')
    names = [e['dataset'] for e in entries]
    assert len(names) == 60 > fd.TERMINAL_LIST_ENTRIES
    human = cli('check-sim-data', source)
    assert 'This terminal view abbreviates' in human.stdout and names[-1] not in human.stdout
    assert "remedy_lists['source.non_root_quaternion_datasets']" in human.stdout  # remedies refer to the list by key
    as_json = cli('check-sim-data', source, '--json')
    assert as_json.returncode == 0 and json.loads(as_json.stdout) == report
    markdown = cli('check-sim-data', source, '--markdown')
    assert markdown.stdout.startswith('# RoboClue simulator data provenance')
    assert all(name.replace('_', '\\_') in markdown.stdout for name in names)
    written = tmp_path / 'out' / 'report.txt'
    run = cli('check-sim-data', source, '--output', written)
    assert run.returncode == 0 and run.stdout == '' and 'abbreviates' not in written.read_text()
    assert all(name in written.read_text() for name in names)
    as_file = tmp_path / 'report.json'
    assert cli('check-sim-data', source, '--json', '--output', as_file).returncode == 0
    saved = json.loads(as_file.read_text())
    assert saved == report and remedy_list(saved, 'source.non_root_quaternion_datasets') == entries
    refused(cli('check-sim-data', source, '--json', '--output', as_file), 'already exists')
    assert json.loads(as_file.read_text()) == report  # never overwritten
    refused(cli('check-sim-data', source, '--json', '--max-rows', '0'), 'max_rows')
    both = cli('check-sim-data', source, '--json', '--markdown')
    assert both.returncode == 2 and 'not allowed with argument' in both.stderr


def test_cli_sim_refusals_are_named_and_never_tracebacks(sim_inputs, tmp_path):
    text = tmp_path / 'notes.txt'
    text.write_text('not hdf5')
    folder = tmp_path / 'plain_folder'
    (folder / 'sub').mkdir(parents=True)
    fifo = tmp_path / 'demo.fifo'
    os.mkfifo(fifo)
    lerobot = healthy_recording(tmp_path / 'lerobot')
    refused(cli('check-sim-data', tmp_path / 'absent.hdf5'), 'absent.hdf5', 'path does not exist')
    refused(cli('check-sim-data', text), 'not a readable HDF5 file')
    refused(cli('check-sim-data', folder), 'not a LeRobot dataset', 'HDF5 file')
    refused(cli('check-sim-data', 'hf://lerobot/pusht'), 'local path')
    refused(cli('check-sim-data', sim_inputs['pass'], '--converted', 'https://example.com/x.hdf5'), 'local path')
    started = time.monotonic()
    refused(cli('check-sim-data', fifo, timeout=60), fifo.name, 'FIFO', 'regular file')
    assert time.monotonic() - started < 30
    refused(cli('check-sim-data', sim_inputs['pass'], '--max-demos', '0'), 'max_demos')
    refused(cli('check-sim-data', sim_inputs['pass'], '--max-demos', '4097'), 'max_demos')
    refused(cli('check-sim-data', sim_inputs['pass'], '--max-rows', '5000001'), 'max_rows')
    refused(cli('check-sim-data', lerobot, '--converted', sim_inputs['pass']), 'LeRobot source cannot have a converted')
    refused(cli('check-sim-data', lerobot, '--output', lerobot / 'report.txt'), 'inside the checked input')
    refused(cli('check-sim-data', lerobot / 'meta' / 'info.json', '--output', lerobot / 'r.txt'), 'inside the checked')
    assert not (lerobot / 'report.txt').exists() and not (lerobot / 'r.txt').exists()
    usage = cli('check-sim-data', sim_inputs['pass'], '--max-demos', 'many')
    assert usage.returncode == 2 and 'invalid int value' in usage.stderr and 'Traceback' not in usage.stderr
    assert cli('check-sim-data').returncode == 2


def test_cli_without_the_dataset_extra_names_it(sim_inputs, tmp_path):
    """Core install (no h5py, no pyarrow): an explicit refusal naming the extra, never a crash."""
    recording = healthy_recording(tmp_path / 'ds')
    program = ("import sys; sys.modules['h5py'] = None; sys.modules['pyarrow'] = None; "
               "sys.modules['pyarrow.parquet'] = None; from policy_doctor.cli import main; "
               "raise SystemExit(main(sys.argv[1:]))")
    for args, needle in (([str(sim_inputs['pass'])], 'policy-deployment-doctor[dataset] (h5py)'),
                         (['--json', str(recording)], 'policy-deployment-doctor[dataset]')):
        command = 'check-sim-data' if needle.endswith('(h5py)') else 'check-salvage'
        run = subprocess.run([sys.executable, '-c', program, command, *args], capture_output=True, text=True,
                             env=ENV, timeout=120)
        refused(run, needle)


@pytest.mark.parametrize('case, status, code', [('healthy', 'pass', 0), ('interrupted', 'error', 1),
                                                ('video_absent', 'unknown', 3)])
def test_cli_salvage_exit_codes_and_output_modes(tmp_path, case, status, code):
    root = {'healthy': lambda p: healthy_recording(p), 'interrupted': interrupted_recording,
            'video_absent': lambda p: healthy_recording(p, video=True)}[case](tmp_path / 'ds')
    before = tree_hashes(root)
    report = check_salvage(root)
    assert report['status'] == status
    human = cli('check-salvage', root)
    assert human.returncode == code and human.stdout.startswith(f'RoboClue recording salvage: {status.upper()}')
    assert 'nothing was written' in human.stdout and 'Episodes to keep:' in human.stdout
    as_json = cli('check-salvage', root, '--json')
    assert json.loads(as_json.stdout) == report and as_json.returncode == code
    markdown = cli('check-salvage', root / 'meta' / 'info.json', '--markdown')
    assert markdown.returncode == code and markdown.stdout.startswith('# RoboClue recording salvage')
    assert '## Plan (complete)' in markdown.stdout
    out = tmp_path / 'salvage.json'
    assert cli('check-salvage', root, '--json', '--output', out).returncode == code
    assert json.loads(out.read_text()) == report
    for gate, failing in {'error': {'error'}, 'warning': {'error', 'warning'},
                          'unknown': {'error', 'warning', 'unknown'}}.items():
        expected = int(any(f['status'] in failing for f in report['findings']))
        assert cli('check-salvage', root, '--fail-on', gate).returncode == expected
    if case == 'interrupted':
        assert 'episode 2: rows_without_metadata' in human.stdout and 'resume safe: no' in human.stdout
    assert tree_hashes(root) == before  # the recording is never changed


def test_cli_salvage_refusals_are_named(tmp_path):
    root = healthy_recording(tmp_path / 'ds', video=True)
    plain = tmp_path / 'folder'
    plain.mkdir()
    fifo = tmp_path / 'ds.fifo'
    os.mkfifo(fifo)
    refused(cli('check-salvage', plain), 'Not a LeRobot dataset', 'no meta/info.json')
    refused(cli('check-salvage', root / 'data' / 'chunk-000' / 'file-000.parquet'), 'is a file, not a LeRobot dataset')
    refused(cli('check-salvage', robomimic(tmp_path / 'demo.hdf5')), 'check-sim-data')
    refused(cli('check-salvage', tmp_path / 'absent'), 'does not exist')
    refused(cli('check-salvage', 'hf://lerobot/pusht'), 'local path')
    refused(cli('check-salvage', fifo, timeout=60), 'not a regular directory')
    refused(cli('check-salvage', root, '--max-rows', '0'), 'max_rows')
    refused(cli('check-salvage', root, '--max-files', '4097'), 'max_files')
    refused(cli('check-salvage', root, '--output', root / 'report.txt'), 'inside the checked input')
    assert not (root / 'report.txt').exists()
    outside = tmp_path / 'elsewhere'
    (outside / CAM / 'chunk-000').mkdir(parents=True)
    (outside / CAM / 'chunk-000' / 'file-000.mp4').write_bytes(b'\x00' * 64)
    (root / 'videos').symlink_to(outside, target_is_directory=True)
    for flag in ('--video-headers', '--no-video-headers'):
        message = refused(cli('check-salvage', root, flag), 'videos', 'symlink')
        assert 'cp -rP' in message and 'test ! -e' in message and 'test ! -L' in message and 'mv -T' in message
    linked_info = tmp_path / 'linked'
    (linked_info / 'meta').mkdir(parents=True)
    (linked_info / 'meta' / 'info.json').symlink_to(root / 'meta' / 'info.json')
    refused(cli('check-salvage', linked_info / 'meta' / 'info.json'), 'symlink')
    for relative in ('meta/tasks.parquet', 'data/chunk-000/file-000.parquet', 'meta/stats.json'):
        piped = healthy_recording(tmp_path / ('fifo_' + relative.replace('/', '_')))
        (piped / relative).unlink()
        os.mkfifo(piped / relative)
        started = time.monotonic()
        message = refused(cli('check-salvage', piped, '--json', timeout=60), 'FIFO')
        assert relative in message and time.monotonic() - started < 30


# ------------------------------------------------------------------------------------------------ MCP


def tool_body(result):
    assert not result.isError, result.content[0].text[:2000]
    return result.structuredContent or json.loads(result.content[0].text)


def server_params(root, *extra):
    return StdioServerParameters(command=sys.executable, args=['-m', 'policy_doctor', 'serve', '--root', str(root),
                                                               *extra], env=ENV)


def test_mcp_catalog_and_strict_schemas(tmp_path):
    for options, expected in (({}, 29), ({'case_root': str(tmp_path)}, 31), ({'allow_hub': True}, 30),
                              ({'case_root': str(tmp_path), 'allow_hub_media': True}, 34)):
        tools = create_server([str(tmp_path)], **options)._tool_manager._tools
        assert len(tools) == expected and {fd.SIM_TOOL, fd.SALVAGE_TOOL} <= set(tools)
        assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    tools = create_server([str(tmp_path)])._tool_manager._tools
    sim, salvage = tools[fd.SIM_TOOL], tools[fd.SALVAGE_TOOL]
    for tool in (sim, salvage):
        assert tool.annotations.readOnlyHint and not tool.annotations.destructiveHint
        assert tool.annotations.openWorldHint is False and tool.parameters['additionalProperties'] is False
    props = sim.parameters['properties']
    assert set(props) == {'source', 'converted', 'max_demos', 'max_rows', 'response_format', 'remedy_list', 'offset',
                          'limit'} and sim.parameters['required'] == ['source']
    assert props['max_demos']['anyOf'][0] == {'type': 'integer', 'minimum': 1, 'maximum': 4096}
    assert props['max_rows']['maximum'] == 5_000_000 and props['limit']['anyOf'][0]['maximum'] == fd.PAGE_MAX
    assert set(props['response_format']['enum']) == {'concise', 'detailed'}
    props = salvage.parameters['properties']
    assert set(props) == {'dataset', 'video_headers', 'max_rows', 'max_files', 'response_format'}
    assert props['video_headers']['type'] == 'boolean' and props['max_files']['maximum'] == 4096
    assert 'extra_roots' not in json.dumps(salvage.parameters)


def test_mcp_stdio_simulator_views_pages_and_refusals(tmp_path, sim_inputs):
    root = tmp_path
    source = sim_inputs['warning']
    report = check_sim_provenance(str(source))
    key = 'source.non_root_quaternion_datasets'
    folder = tmp_path / 'folder'
    folder.mkdir()
    fifo = tmp_path / 'x.fifo'
    os.mkfifo(fifo)

    async def exercise():
        async with stdio_client(server_params(root)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert len(tools) == 29 and {fd.SIM_TOOL, fd.SALVAGE_TOOL} <= {t.name for t in tools}
                assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= {t.name for t in tools}
                concise = tool_body(await session.call_tool(fd.SIM_TOOL, {'source': str(source)}))
                assert concise['summary'] is True and concise['status'] == report['status'] == 'warning'
                assert concise['counts'] == report['counts'] and len(json.dumps(concise).encode()) <= fd.SUMMARY_LIMIT
                digest = concise['remedy_lists']['lists'][key]
                assert digest['entries'] == 60 and digest['list_sha256'] == fd.canonical_sha256(remedy_list(report, key))
                assert 'remedy_list=<key>' in concise['remedy_lists_complete_in']
                warning = next(f for f in concise['findings_shown'] if f['code'] == 'isaaclab.quaternions_outside_root_pose')
                full = next(f for f in report['findings'] if f['code'] == warning['code'])
                assert warning['remedy'] == full['remedy'] and 'text_truncated' not in warning  # the action stays whole
                detailed = tool_body(await session.call_tool(fd.SIM_TOOL, {'source': str(source),
                                                                            'response_format': 'detailed'}))
                assert detailed.pop('served_from') == 'cached_report_same_file_identity' and detailed == report
                assembled, offset, sha = [], 0, None
                while offset is not None:
                    page = tool_body(await session.call_tool(fd.SIM_TOOL, {'source': str(source), 'remedy_list': key,
                                                                            'offset': offset, 'limit': 7}))
                    assert page['offset'] == offset and page['returned'] <= 7 and page['entries_total'] == 60
                    sha = sha or page['list_sha256']
                    assert page['list_sha256'] == sha and page['remedy_lists_sha256'] == digest_sha(report)
                    assembled += page['entries']
                    offset = page['next_offset']
                assert page['complete'] is True and assembled == remedy_list(report, key)
                assert fd.canonical_sha256(assembled) == sha
                end = tool_body(await session.call_tool(fd.SIM_TOOL, {'source': str(source), 'remedy_list': key,
                                                                       'offset': 60}))
                assert end['entries'] == [] and end['complete'] is True
                for args, needle in (
                        ({'source': str(source), 'unknown_key': 1}, 'unknown_key'),
                        ({'source': str(source), 'offset': 3}, 'require remedy_list'),
                        ({'source': str(source), 'limit': 3}, 'require remedy_list'),
                        ({'source': str(source), 'remedy_list': 'nope'}, key),
                        ({'source': str(source), 'remedy_list': key, 'offset': 61}, 'past the end'),
                        ({'source': str(source), 'remedy_list': key, 'limit': 0}, 'limit'),
                        ({'source': str(source), 'remedy_list': key, 'offset': True}, 'offset'),
                        ({'source': str(source), 'max_demos': True}, 'max_demos'),
                        ({'source': str(source), 'max_demos': '3'}, 'max_demos'),
                        ({'source': str(source), 'max_rows': 0}, 'max_rows'),
                        ({'source': str(source), 'response_format': 'full'}, 'response_format'),
                        ({'source': '/etc/hostname'}, 'outside the configured MCP read roots'),
                        ({'source': str(source), 'converted': '/etc'}, 'outside the configured MCP read roots'),
                        ({'source': 'hf://lerobot/pusht'}, 'local path'),
                        ({'source': ''}, 'nonempty'),
                        ({'source': str(folder)}, 'not a LeRobot dataset'),
                        ({'source': str(fifo)}, 'FIFO'),
                        ({}, 'source')):
                    result = await session.call_tool(fd.SIM_TOOL, args)
                    assert result.isError and needle in result.content[0].text, (args, result.content[0].text[:500])
                    if args.get('source') == str(fifo):
                        assert fifo.name in result.content[0].text and 'regular file' in result.content[0].text
                assert tool_body(await session.call_tool(fd.SIM_TOOL, {'source': str(sim_inputs['pass'])}))['status'] \
                    == 'pass'  # the server keeps answering after every refusal

    asyncio.run(asyncio.wait_for(exercise(), timeout=240))


def digest_sha(report):
    return hashlib.sha256(json.dumps(report['remedy_lists'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_mcp_stdio_salvage_views_and_refusals(tmp_path):
    healthy = healthy_recording(tmp_path / 'healthy')
    interrupted = interrupted_recording(tmp_path / 'interrupted')
    linked = healthy_recording(tmp_path / 'linked', video=True)
    outside = tmp_path.parent / f'{tmp_path.name}_outside'
    (outside / CAM).mkdir(parents=True)
    (linked / 'videos').symlink_to(outside, target_is_directory=True)
    plain = tmp_path / 'plain'
    plain.mkdir()
    expected = check_salvage(interrupted)
    before = tree_hashes(interrupted)

    async def exercise():
        async with stdio_client(server_params(tmp_path)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                ok = tool_body(await session.call_tool(fd.SALVAGE_TOOL, {'dataset': str(healthy)}))
                assert ok['status'] == 'pass' and ok['plan']['strategy'] == 'none_needed' and ok['summary'] is True
                concise = tool_body(await session.call_tool(fd.SALVAGE_TOOL, {'dataset': str(interrupted)}))
                assert concise['status'] == 'error' and len(json.dumps(concise).encode()) <= fd.SUMMARY_LIMIT
                assert concise['episodes']['keep'] == [[0, 1]] and concise['episodes']['drop'][0] == {
                    'episode_index': 2, 'reasons': ['rows_without_metadata']}
                assert concise['plan']['applied'] is False and concise['plan']['resume_safe'] is False
                assert concise['counts'] == expected['counts']['findings']
                detailed = tool_body(await session.call_tool(fd.SALVAGE_TOOL, {
                    'dataset': str(interrupted / 'meta' / 'info.json'), 'response_format': 'detailed'}))
                assert detailed == expected
                headers_off = tool_body(await session.call_tool(fd.SALVAGE_TOOL, {
                    'dataset': str(healthy), 'video_headers': False, 'max_rows': 100, 'max_files': 8}))
                assert headers_off['coverage']['max_rows'] == 100 and headers_off['coverage']['max_files'] == 8
                for args, needle in (
                        ({'dataset': str(linked)}, 'symlink'),
                        ({'dataset': str(linked), 'video_headers': False}, 'cp -rP'),
                        ({'dataset': str(plain)}, 'Not a LeRobot dataset'),
                        ({'dataset': str(healthy / 'data' / 'chunk-000' / 'file-000.parquet')}, 'is a file'),
                        ({'dataset': str(outside)}, 'outside the configured MCP read roots'),
                        ({'dataset': 'hf://lerobot/pusht'}, 'local path'),
                        ({'dataset': str(healthy), 'video_headers': 'yes'}, 'video_headers'),
                        ({'dataset': str(healthy), 'video_headers': 1}, 'video_headers'),
                        ({'dataset': str(healthy), 'max_files': 0}, 'max_files'),
                        ({'dataset': str(healthy), 'max_rows': 2_000_000}, 'max_rows'),
                        ({'dataset': str(healthy), 'extra_roots': [str(outside)]}, 'extra_roots'),
                        ({'dataset': str(healthy), 'response_format': 'brief'}, 'response_format')):
                    result = await session.call_tool(fd.SALVAGE_TOOL, args)
                    assert result.isError and needle in result.content[0].text, (args, result.content[0].text[:500])
                assert tool_body(await session.call_tool(fd.SALVAGE_TOOL, {'dataset': str(healthy)}))['status'] == 'pass'
                # A FIFO where a Parquet file belongs may block a read; later requests must still be answered.
                stuck = asyncio.create_task(session.call_tool(fd.SALVAGE_TOOL, {'dataset': str(fifo_dataset)}))
                await asyncio.sleep(3)
                started = time.monotonic()
                assert tool_body(await session.call_tool(fd.SALVAGE_TOOL, {'dataset': str(healthy)}))['status'] == 'pass'
                assert time.monotonic() - started < 30
                try:  # release a reader blocked on the FIFO (no-op when the check refused it without opening)
                    os.close(os.open(fifo_dataset / 'meta' / 'tasks.parquet', os.O_WRONLY | os.O_NONBLOCK))
                except OSError:
                    pass
                refusal = await asyncio.wait_for(stuck, 60)
                assert refusal.isError and 'FIFO' in refusal.content[0].text and 'tasks.parquet' in refusal.content[0].text

    fifo_dataset = healthy_recording(tmp_path / 'fifo_ds')
    (fifo_dataset / 'meta' / 'tasks.parquet').unlink()
    os.mkfifo(fifo_dataset / 'meta' / 'tasks.parquet')
    asyncio.run(asyncio.wait_for(exercise(), timeout=180))
    assert tree_hashes(interrupted) == before


def test_mcp_cache_is_used_only_while_the_file_is_unchanged(tmp_path):
    source = legacy_isaac(tmp_path / 'legacy.hdf5', cams=4)
    server = create_server([str(tmp_path)])

    def call(**args):
        content = asyncio.run(server.call_tool(fd.SIM_TOOL, dict(source=str(source), **args)))
        if isinstance(content, tuple):
            return content[1]
        return content if isinstance(content, dict) else json.loads(content[0].text)

    first = call(response_format='detailed')
    assert first['served_from'] == 'fresh_read'
    assert call()['served_from'] == 'cached_report_same_file_identity'
    assert call(max_demos=1)['served_from'] == 'fresh_read'  # other bounds are another report
    with h5py.File(source, 'r+') as f:
        f.attrs['format_version'] = 1
    changed = call(response_format='detailed')
    assert changed['served_from'] == 'fresh_read'
    assert changed['coverage']['source']['sha256'] != first['coverage']['source']['sha256']
    assert changed == dict(check_sim_provenance(str(source)), served_from='fresh_read')
    assert call(response_format='detailed')['served_from'] == 'cached_report_same_file_identity'


# ------------------------------------------------------------------------------------------------ bounded views


def test_concise_views_stay_bounded_and_mark_every_cut():
    findings = [{'code': f'isaaclab.c{i}', 'status': ('error', 'warning', 'unknown', 'pass')[i % 4],
                 'message': 'm ' * 3000, 'remedy': 'r ' * 3000,
                 'evidence': [{'path': '/p' * 200, 'pointer': '/x', 'sha256': '0' * 64, 'value': list(range(500))}]}
                for i in range(40)]
    report = {'schema_version': 2, 'kind': 'sim_provenance', 'status': 'error', 'findings': findings,
              'counts': {s: 10 for s in fd.STATUSES}, 'coverage': {}, 'remedy_lists': {}, 'scope': 's ' * 2000}
    view = fd.concise_sim(report)
    assert len(json.dumps(view).encode()) <= fd.SUMMARY_LIMIT and view['status'] == 'error'
    assert view['counts'] == report['counts'] and view['findings_total'] == 40
    assert view['findings_shown'][0]['status'] == 'error' and view['findings_omitted'] > 0
    for item in view['findings_shown']:
        for key in item.get('text_truncated', []):
            assert item[key].endswith('[...]') and len(item[f'full_{key}_sha256']) == 64


def test_detailed_view_degrades_explicitly_and_pages_reassemble_exactly(tmp_path):
    src, dst = partial_pair(tmp_path, 60, 80)
    report = check_sim_provenance(str(src), converted=str(dst))
    keys = fd.list_keys(report)
    assert keys and report['status'] in ('error', 'warning', 'unknown')
    view = fd.detailed_sim(report)
    if len(json.dumps(report).encode()) <= fd.RESULT_LIMIT - fd.ENVELOPE_MARGIN:
        assert view is report
    big = dict(report, remedy_lists=dict(report['remedy_lists']))
    big['remedy_lists']['names'] = dict(report['remedy_lists']['names'], padding=['x' * 1000] * 300)
    degraded = fd.detailed_sim(big)
    assert degraded['detailed_bounded'] is True and degraded['remedy_lists'] is None
    assert degraded['remedy_lists_omitted'] == keys and degraded['findings'] == report['findings']
    assert set(degraded['remedy_lists_digest']['lists']) == set(keys)
    assert 'remedy_list=<key>' in degraded['remedy_lists_complete_in']
    assert len(json.dumps(degraded).encode()) <= fd.RESULT_LIMIT
    for key in keys:
        entries, offset = [], 0
        while offset is not None:
            page = fd.remedy_page(report, key, offset, 3)
            assert len(json.dumps(page).encode()) <= fd.RESULT_LIMIT
            entries += page['entries']
            offset = page['next_offset']
        assert entries == remedy_list(report, key) and fd.canonical_sha256(entries) == page['list_sha256']
    with pytest.raises(fd.InputError, match='available'):
        fd.remedy_page(report, 'names', 0, 10)
    empty = dict(report, remedy_lists=dict(report['remedy_lists'], **{'pair.empty': []}))
    page = fd.remedy_page(empty, 'pair.empty', 0, 10)
    assert page['entries'] == [] and page['complete'] and page['next_offset'] is None and page['entries_total'] == 0
    assert fd.list_digest(empty)['lists']['pair.empty']['entries'] == 0
    for bad in ({'offset': -1}, {'offset': True}, {'limit': 0}, {'limit': fd.PAGE_MAX + 1}, {'limit': 2.0}):
        with pytest.raises(fd.InputError):
            fd.remedy_page(report, keys[0], bad.get('offset', 0), bad.get('limit', 10))


def test_salvage_concise_view_stays_bounded_with_thousands_of_dropped_episodes(tmp_path):
    kept = [(e, 2) for e in range(50)]
    orphans = [(e, 1) for e in range(51, 4000, 2)]
    root = lerobot_recording(tmp_path / 'ds', rows_for(kept + orphans), meta_for(kept), 50, 100)
    report = check_salvage(root)
    view = fd.concise_salvage(report)
    assert len(json.dumps(view).encode()) <= fd.SUMMARY_LIMIT and view['status'] == report['status']
    assert view['episodes']['drop_count'] == len(orphans) and view['episodes']['drop_omitted'] > 0
    assert fd.detailed_salvage(report) == report
    text = fd.human_salvage(report)
    assert 'more in --json' in text and len(text.splitlines()) < 200


# ------------------------------------------------------------------------------------------------ scale


@pytest.fixture(scope='module')
def big_pair(tmp_path_factory):
    return partial_pair(tmp_path_factory.mktemp('big'), 400, 200)


def test_400_demo_200_camera_pair_pages_equal_remedy_list_and_slow_calls_do_not_block(big_pair, tmp_path):
    """The validator's worst case: a report over RESULT_LIMIT whose complete lists must reach an MCP client. The full
    CLI report (--json --output) runs in parallel and is the reference remedy_list() is computed from."""
    src, dst = big_pair
    saved = tmp_path / 'full.json'
    cli_run = subprocess.Popen([sys.executable, '-m', 'policy_doctor', 'check-sim-data', str(src), '--converted',
                                str(dst), '--json', '--output', str(saved)], env=ENV, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    small = healthy_recording(src.parent / 'small_ds')

    async def exercise():
        async with stdio_client(server_params(src.parent)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                slow = asyncio.create_task(session.call_tool(fd.SIM_TOOL, {'source': str(src), 'converted': str(dst)}))
                await asyncio.sleep(2)
                started = time.monotonic()
                quick = tool_body(await session.call_tool(fd.SALVAGE_TOOL, {'dataset': str(small)}))
                assert quick['status'] == 'pass' and time.monotonic() - started < 30
                assert not slow.done()  # answered while the long check was still running
                concise = tool_body(await slow)
                assert len(json.dumps(concise).encode()) <= fd.SUMMARY_LIMIT
                lists = concise['remedy_lists']['lists']
                detailed = tool_body(await session.call_tool(fd.SIM_TOOL, {'source': str(src), 'converted': str(dst),
                                                                            'response_format': 'detailed'}))
                assert detailed['served_from'] == 'cached_report_same_file_identity'
                assert detailed['detailed_bounded'] is True and detailed['remedy_lists'] is None
                assert detailed['truncated_from_bytes'] > fd.RESULT_LIMIT
                assert detailed['remedy_lists_omitted'] == sorted(lists)
                pages = {}
                for key in sorted(lists):
                    entries, offset, count = [], 0, 0
                    while offset is not None:
                        page = tool_body(await session.call_tool(fd.SIM_TOOL, {
                            'source': str(src), 'converted': str(dst), 'remedy_list': key, 'offset': offset,
                            'limit': fd.PAGE_MAX}))
                        assert page['served_from'] == 'cached_report_same_file_identity'
                        assert len(json.dumps(page).encode()) <= fd.RESULT_LIMIT
                        assert page['list_sha256'] == lists[key]['list_sha256']
                        entries += page['entries']
                        offset, count = page['next_offset'], count + 1
                    pages[key] = (entries, count)
                return concise, detailed, pages

    concise, detailed, pages = asyncio.run(asyncio.wait_for(exercise(), timeout=1500))
    out, err = cli_run.communicate(timeout=1500)
    assert cli_run.returncode in (0, 1, 3) and out == '' and 'Traceback' not in err
    report = json.loads(saved.read_text())
    assert len(json.dumps(report).encode()) > fd.RESULT_LIMIT and report['status'] == concise['status']
    assert detailed['findings'] == report['findings']
    assert max(count for _, count in pages.values()) >= 2  # at least one list needed several bounded pages
    assert max(len(entries) for entries, _ in pages.values()) >= 200
    for key, (entries, _) in pages.items():
        assert entries == remedy_list(report, key)
        assert fd.canonical_sha256(entries) == concise['remedy_lists']['lists'][key]['list_sha256']
    assert digest_sha(report) == concise['remedy_lists']['sha256']
