"""Owner decision 0.40: an explicitly named top-level path (dataset, checkpoint, robot config, deployment, command
directory) is resolved once to its real path and reported; links inside artifact trees and in paths read from a
command or config stay refused; under MCP the real path must stay inside the read roots."""
import asyncio
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from test_readiness_workflow import config, healthy

from policy_doctor.artifacts import InputError
from policy_doctor.readiness_workflow import _local, check_robot_readiness, prepare_readiness

SRC = Path(__file__).resolve().parents[1] / 'src'
ENV = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1')


def assert_resolution_view(shown, complete):
    """Require full identities or an explicitly bound display preview."""
    for key in ('role', 'given', 'real'):
        full = complete[key]
        assert shown[key] == full[:240]
        if len(full) > 240:
            assert shown[key + '_truncated'] is True
            assert shown[key + '_length'] == len(full)
            assert shown['full_' + key + '_sha256'] == hashlib.sha256(full.encode('utf-8')).hexdigest()
        else:
            assert key + '_truncated' not in shown
            assert key + '_length' not in shown
            assert 'full_' + key + '_sha256' not in shown


def report_digest(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, '-B', '-m', 'policy_doctor', *map(str, args)], capture_output=True,
                          text=True, env=ENV, timeout=300, cwd=cwd)


@pytest.fixture
def linked(case):
    """~/data -> /mnt/big: the case's folder reached through a symlinked parent directory."""
    path = healthy(case)
    alias = case['dataset'].parent.parent / (case['dataset'].parent.name + '-alias')
    alias.symlink_to(case['dataset'].parent, target_is_directory=True)
    return {'alias': alias, 'config': path, **case}


def test_explicit_paths_under_a_symlinked_parent_resolve_once_and_are_reported(linked):
    alias = linked['alias']
    report = check_robot_readiness(robot_config=str(alias / 'robot.json'), dataset=str(alias / 'dataset'),
                                   checkpoint=str(alias / 'checkpoint'))
    assert report['status'] == 'pass'
    assert report['resolved_paths'] == [
        {'role': 'checkpoint', 'given': str(alias / 'checkpoint'), 'real': str(linked['checkpoint'])},
        {'role': 'dataset', 'given': str(alias / 'dataset'), 'real': str(linked['dataset'])},
        {'role': 'robot config', 'given': str(alias / 'robot.json'), 'real': str(linked['config'])}]
    plain = check_robot_readiness(robot_config=str(linked['config']), dataset=str(linked['dataset']))
    assert plain['resolved_paths'] == []
    assert {k: a['status'] for k, a in plain['axes'].items()} == {k: a['status'] for k, a in report['axes'].items()}


def test_the_named_path_itself_may_be_a_link(linked):
    named = linked['dataset'].parent / 'dataset-link'
    named.symlink_to(linked['dataset'], target_is_directory=True)
    plan = prepare_readiness(robot_config=str(linked['config']), dataset=str(named))
    assert plan['dataset'] == str(linked['dataset'])
    assert plan['resolved_paths'] == [{'role': 'dataset', 'given': str(named), 'real': str(linked['dataset'])}]


def test_cli_readiness_and_preflight_report_the_resolution(linked):
    alias = linked['alias']
    out = cli('readiness', '--robot-config', alias / 'robot.json', '--dataset', alias / 'dataset')
    assert out.returncode == 0, out.stderr
    assert f"resolved {alias / 'dataset'} → {linked['dataset']}" in out.stdout
    assert f"resolved {alias / 'robot.json'} → {linked['config']}" in out.stdout
    out = cli('preflight', alias / 'dataset', '--robot-config', alias / 'robot.json', '--json')
    assert out.returncode != 2, out.stderr
    report = json.loads(out.stdout)
    assert {'role': 'dataset', 'given': str(alias / 'dataset'), 'real': str(linked['dataset'])} in report['resolved_paths']
    assert 'symlink ancestry' not in out.stdout
    human = cli('preflight', alias / 'dataset', '--robot-config', alias / 'robot.json')
    assert f"resolved {alias / 'dataset'} → {linked['dataset']}" in human.stdout.splitlines()[2]


def test_links_read_from_a_command_or_config_stay_refused_with_the_offending_path(linked):
    alias = linked['alias']
    path = config(linked, policy=alias / 'checkpoint')  # the policy path inside the robot config goes through a link
    with pytest.raises(InputError, match='symlink ancestry is refused') as refused:
        prepare_readiness(robot_config=str(path), base_directory=str(alias.parent))
    message = str(refused.value)
    assert repr(alias.name) in message and 'Write the real path there' in message and '--policy' in message
    out = cli('readiness', '--robot-command', f'lerobot-rollout --robot.type=so100_follower --policy.path={alias / "checkpoint"}',
              cwd=alias.parent)
    assert out.returncode == 2 and 'symlink ancestry is refused' in out.stderr and alias.name in out.stderr


def test_relative_and_parent_rules_still_apply_from_the_real_path(linked):
    alias = linked['alias']
    with pytest.raises(InputError, match='parent path components'):
        _local(str(alias / 'dataset' / '..' / 'dataset'), explicit='dataset')
    with pytest.raises(InputError, match='literal tilde'):
        _local('~/dataset', explicit='dataset')
    resolved = []
    assert _local('dataset', base_directory=str(alias), explicit='dataset', resolutions=resolved) == str(linked['dataset'])
    assert resolved == [{'role': 'dataset', 'given': 'dataset', 'real': str(linked['dataset'])}]


def test_mcp_resolution_never_widens_the_read_roots(linked, tmp_path_factory):
    root = linked['dataset'].parent
    escape = tmp_path_factory.mktemp('outside') / 'elsewhere'
    shutil.copytree(linked['dataset'], escape)
    (root / 'escape').symlink_to(escape, target_is_directory=True)
    with pytest.raises(InputError, match='outside configured MCP read roots once its symlinks are resolved'):
        prepare_readiness(robot_config=str(linked['config']), dataset=str(root / 'escape'), read_roots=[str(root)])
    with pytest.raises(InputError, match='outside configured MCP read roots'):
        prepare_readiness(robot_config=str(linked['config']), dataset=str(escape), read_roots=[str(root)])


def test_real_stdio_server_resolves_inside_roots_and_refuses_outside(linked, tmp_path_factory):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    alias, root = linked['alias'], linked['dataset'].parent
    escape = tmp_path_factory.mktemp('outside') / 'elsewhere'
    shutil.copytree(linked['dataset'], escape)
    (root / 'escape').symlink_to(escape, target_is_directory=True)

    async def run():
        params = StdioServerParameters(command=sys.executable, args=['-B', '-m', 'policy_doctor', 'serve', '--root',
                                                                     str(root), '--root', str(alias)], env=ENV)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = {t.name: t for t in (await client.list_tools()).tools}
                complete = {'role': 'dataset', 'given': str(alias / 'dataset'), 'real': str(linked['dataset'])}
                detailed = None
                for format in ('detailed', 'concise'):
                    result = await client.call_tool('check_robot_readiness', {
                        'robot_config': str(alias / 'robot.json'), 'dataset': str(alias / 'dataset'),
                        'response_format': format})
                    assert not result.isError, result.content
                    body = result.structuredContent
                    jsonschema.validate(body, tools['check_robot_readiness'].outputSchema)
                    if format == 'detailed':
                        assert complete in body['resolved_paths']
                        detailed = body
                    else:
                        assert body['status'] == detailed['status']
                        shown = next(item for item in body['resolved_paths'] if item['role'] == 'dataset')
                        assert_resolution_view(shown, complete)
                        assert body['full_report_sha256'] == report_digest(detailed)
                    assert len(json.dumps(body).encode()) <= 8192 or format == 'detailed'
                full_pre = await client.call_tool('preflight_check', {
                    'source': str(alias / 'dataset'), 'robot_config': str(alias / 'robot.json'),
                    'response_format': 'detailed'})
                assert not full_pre.isError, full_pre.content
                assert complete in full_pre.structuredContent['resolved_paths']
                pre = await client.call_tool('preflight_check', {
                    'source': str(alias / 'dataset'), 'robot_config': str(alias / 'robot.json')})
                assert not pre.isError, pre.content
                jsonschema.validate(pre.structuredContent, tools['preflight_check'].outputSchema)
                brief = pre.structuredContent
                assert brief['status'] == full_pre.structuredContent['status']
                assert_resolution_view(next(item for item in brief['resolved_paths'] if item['role'] == 'dataset'), complete)
                assert brief['full_report_sha256'] == report_digest(full_pre.structuredContent)
                assert len(json.dumps(brief).encode()) <= 8192
                refused = await client.call_tool('check_robot_readiness', {
                    'robot_config': str(alias / 'robot.json'), 'dataset': str(root / 'escape')})
                assert refused.isError and 'once its symlinks are resolved' in refused.content[0].text
                assert str(escape) not in refused.content[0].text
                refused = await client.call_tool('preflight_check', {
                    'source': str(root / 'escape'), 'robot_config': str(alias / 'robot.json')})
                assert refused.isError and 'outside' in refused.content[0].text
    asyncio.run(asyncio.wait_for(run(), timeout=180))


def test_concise_summaries_bound_the_resolutions():
    from policy_doctor.server import SUMMARY_LIMIT, _summary

    many = [{'role': 'dataset', 'given': '/g/' + 'x' * 1000, 'real': '/r/' + 'y' * 1000} for _ in range(7)]
    value = {'kind': 'robot_readiness', 'status': 'pass', 'summary': 's', 'axes': {}, 'findings': [],
             'resolved_paths': many}
    concise = _summary(value)
    assert len(concise['resolved_paths']) == 4 and concise['resolved_paths_omitted'] == 3
    assert all(len(item['given']) <= 240 and len(item['real']) <= 240 for item in concise['resolved_paths'])
    for shown, full in zip(concise['resolved_paths'], many, strict=False):
        assert_resolution_view(shown, full)
    assert concise['full_report_sha256'] == report_digest(value)
    crowded = dict(value, findings=[{'code': f'c{i}', 'status': 'error', 'message': 'm' * 400, 'remedy': 'r' * 300,
                                     'evidence': [{'path': 'p' * 300}]} for i in range(40)])
    concise = _summary(crowded)
    assert len(json.dumps(concise).encode()) <= SUMMARY_LIMIT and concise['status'] == 'pass'
    assert concise['resolved_paths_omitted'] == len(many) - len(concise['resolved_paths'])
    for shown, full in zip(concise['resolved_paths'], many, strict=False):
        assert_resolution_view(shown, full)


def test_resolution_preview_boundary_and_metadata_negative_controls():
    from policy_doctor.server import _summary

    complete = {'role': 'dataset', 'given': '/' + 'é' * 239, 'real': '/' + 'é' * 240}
    value = {'kind': 'robot_readiness', 'status': 'unknown', 'summary': 's', 'axes': {}, 'findings': [],
             'resolved_paths': [complete]}
    shown = _summary(value)['resolved_paths'][0]
    assert_resolution_view(shown, complete)
    for field in ('real_truncated', 'real_length', 'full_real_sha256'):
        missing = copy.deepcopy(shown)
        missing.pop(field)
        with pytest.raises((AssertionError, KeyError)):
            assert_resolution_view(missing, complete)
        wrong = dict(shown, **{field: False if field.endswith('truncated') else 0})
        with pytest.raises(AssertionError):
            assert_resolution_view(wrong, complete)


def test_openpi_resolution_previews_preserve_digest_and_bound():
    from policy_doctor.server import SUMMARY_LIMIT, _summary

    paths = [{'role': 'source root', 'given': '/g/' + 'é' * 1000, 'real': '/r/' + '界' * 1000}] * 3
    value = {'kind': 'openpi_readiness', 'status': 'unknown', 'summary': 's', 'resolved_paths': paths}
    concise = _summary(value)
    assert concise['status'] == 'unknown' and concise['resolved_paths_omitted'] == 1
    assert len(json.dumps(concise).encode()) <= SUMMARY_LIMIT
    for shown, full in zip(concise['resolved_paths'], paths, strict=False):
        assert_resolution_view(shown, full)
    assert concise['full_report_sha256'] == hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    pressured = dict(value, unknowns=['x' * 4300])
    bounded = _summary(pressured)
    assert len(json.dumps(bounded).encode()) <= SUMMARY_LIMIT
    assert bounded['resolved_paths'] == [] and bounded['resolved_paths_omitted'] == len(paths)


def test_openpi_command_directory_and_source_root_under_a_link(tmp_path):
    from test_openpi_workflow import CURRENT, FIXTURES

    from policy_doctor.cli import human
    from policy_doctor.openpi_workflow import check_readiness
    from policy_doctor.server import _summary

    source = tmp_path / 'openpi'
    shutil.copytree(FIXTURES / 'openpi_source_profile_215abfb', source)
    for file in source.rglob('*.txt'):
        file.rename(file.with_suffix(''))
    target = source / 'checkpoint/assets/droid'  # local statistics: no public read
    target.mkdir(parents=True)
    shutil.copyfile(FIXTURES / 'openpi_original_20261024/norm_stats.json', target / 'norm_stats.json')
    command = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=checkpoint'
    alias = tmp_path / 'openpi-alias'
    alias.symlink_to(source, target_is_directory=True)
    report = check_readiness(robot_command=command, command_directory=str(alias), source_root=str(alias),
                             read_roots=[str(tmp_path)])
    assert report['statistics_intake']['current']['provenance']['source'] == str(target / 'norm_stats.json')
    assert report['kind'] == 'openpi_readiness'
    assert report['resolved_paths'] == [
        {'role': 'command directory', 'given': str(alias), 'real': str(source)},
        {'role': 'OpenPI source root', 'given': str(alias), 'real': str(source)}]
    assert f'resolved {alias} → {source}' in human(report).splitlines()[2]
    concise = _summary(report)
    for shown, complete in zip(concise['resolved_paths'], report['resolved_paths'], strict=True):
        assert_resolution_view(shown, complete)
    assert concise['full_report_sha256'] == hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with pytest.raises(InputError, match='outside configured MCP read roots once its symlinks are resolved'):
        check_readiness(robot_command=CURRENT, command_directory=str(alias), read_roots=[str(alias.parent / 'none')])
