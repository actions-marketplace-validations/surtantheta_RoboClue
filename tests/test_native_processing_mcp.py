"""Actual FastMCP opt-in/schema/root boundaries with fake parent API results.

The in-process MCP layer is real; substituted processing calls are unit fixtures,
not native execution evidence. Original native trials are qualified separately.
"""

import asyncio
import json
import sys

import pytest

from policy_doctor import native_processing
from policy_doctor.artifacts import InputError
from policy_doctor.server import create_server


def run(coroutine):
    return asyncio.run(coroutine)


def structured(result):
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and isinstance(result[-1], dict):
        return result[-1]
    return json.loads(result[0].text)


@pytest.fixture
def configured(tmp_path):
    read = tmp_path / 'read-root'
    read.mkdir()
    model, dataset, capsule = (read / name for name in ('model', 'dataset', 'capsule'))
    for path in (model, dataset, capsule):
        path.mkdir()
    outputs = tmp_path / 'case-root'
    outputs.mkdir()
    source = tmp_path / 'operator-source'
    source.mkdir()
    (source / 'native.py').write_text('# generated unit source, never executed\n')
    config = tmp_path / 'operator-runtime.json'
    config.write_text(json.dumps({'providers': {'groot_state': {'python': sys.executable, 'source_root': str(source)}}}))
    return {'read': read, 'model': model, 'dataset': dataset, 'capsule': capsule,
            'outputs': outputs, 'config': config, 'outside': tmp_path / 'outside'}


@pytest.fixture
def parent_calls(monkeypatch):
    calls = []
    def probe(provider, model, dataset, **kwargs):
        calls.append(('probe', provider, model, dataset, kwargs))
        return {'kind': 'native_processing_probe', 'status': 'observed', 'scope': 'Unit MCP route only; no native execution',
                'probe': {'provider': provider, 'execution_status': 'complete', 'samples': []}}
    def replay(directory, **kwargs):
        calls.append(('replay', directory, kwargs))
        return {'kind': 'native_processing_reproduction', 'status': 'pass', 'reproduction_matches': True,
                'scope': 'Unit MCP route only; no native execution'}
    monkeypatch.setattr(native_processing, 'probe_native_processing', probe)
    monkeypatch.setattr(native_processing, 'reproduce_native_processing', replay)
    return calls


@pytest.mark.parametrize('with_cases', [False, True])
def test_default_toolset_unchanged_and_native_optin_adds_exactly_two(configured, with_cases):
    options = {'case_root': str(configured['outputs'])} if with_cases else {}
    default = create_server([str(configured['read'])], **options)
    enabled = create_server([str(configured['read'])], native_runtime=str(configured['config']), **options)
    async def exercise():
        before = {t.name: t for t in await default.list_tools()}
        after = {t.name: t for t in await enabled.list_tools()}
        assert len(before) == 29 + 2 * int(with_cases)
        assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(before)
        assert set(after) - set(before) == {'probe_native_processing', 'reproduce_native_processing'}
        assert len(after) == 31 + 2 * int(with_cases)
        assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(after)
        assert ('create_incident_case' in before) == with_cases
        for name in ('probe_native_processing', 'reproduce_native_processing'):
            annotation = after[name].annotations
            assert annotation.readOnlyHint is False and annotation.destructiveHint is False
            assert annotation.idempotentHint is False and annotation.openWorldHint is False
        assert before['inspect_checkpoint'].annotations.readOnlyHint is True
    run(exercise())


def test_client_schema_cannot_select_runtime_module_or_callable(configured):
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        tools = {t.name: t for t in await server.list_tools()}
        probe = tools['probe_native_processing'].inputSchema
        replay = tools['reproduce_native_processing'].inputSchema
        assert set(probe['properties']) == {'provider', 'model', 'dataset', 'samples', 'destination',
                                            'mode', 'processor_source', 'intent', 'approved_sources', 'contract'}
        assert set(replay['properties']) == {'directory'}
        for schema in (probe, replay):
            assert not set(schema['properties']) & {'native_runtime', 'runtimes', 'python', 'source_root', 'module', 'callable'}
    run(exercise())


def test_alternate_source_root_is_guarded_before_parent(configured, parent_calls):
    configured['outside'].mkdir()
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match='outside'):
            await server.call_tool('probe_native_processing', {
                'provider': 'lerobot', 'model': str(configured['model']),
                'dataset': str(configured['dataset']), 'samples': [{'episode_index': 0, 'frame_index': 0}],
                'mode': 'processor_source', 'processor_source': 'serialized_pretrained_path',
                'intent': 'restore_selected_checkpoint', 'approved_sources': [{
                    'repo_id': 'example/base', 'revision': 'a' * 40,
                    'directory': str(configured['outside'])}]})
    run(exercise())
    assert not parent_calls


def test_source_options_retain_operator_runtime_authority(configured, parent_calls):
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        await server.call_tool('probe_native_processing', {
            'provider': 'lerobot', 'model': str(configured['model']),
            'dataset': str(configured['dataset']), 'samples': [{'episode_index': 0, 'frame_index': 0}],
            'mode': 'processor_source', 'processor_source': 'serialized_pretrained_path',
            'intent': 'restore_selected_checkpoint', 'approved_sources': [{
                'repo_id': 'example/base', 'revision': 'a' * 40,
                'directory': str(configured['model'])}]})
    run(exercise())
    assert parent_calls[0][-1]['approved_sources'][0]['directory'] == str(configured['model'])
    assert parent_calls[0][-1]['mode'] == 'processor_source'
    assert set(parent_calls[0][-1]['runtimes']) == {'groot_state'}


def test_alternate_source_cannot_embed_runtime_selection(configured, parent_calls):
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match='only'):
            await server.call_tool('probe_native_processing', {
                'provider': 'lerobot', 'model': str(configured['model']),
                'dataset': str(configured['dataset']), 'samples': [{'episode_index': 0, 'frame_index': 0}],
                'mode': 'processor_source', 'approved_sources': [{
                    'repo_id': 'example/base', 'revision': 'a' * 40,
                    'directory': str(configured['model']), 'python': '/untrusted/python'}]})
    run(exercise())
    assert not parent_calls


def test_strict_runtime_loaded_once_at_startup(configured, parent_calls, monkeypatch):
    actual = native_processing.load_native_runtimes
    loads = []
    def load(path, **kwargs):
        loads.append(path)
        return actual(path, **kwargs)
    monkeypatch.setattr(native_processing, 'load_native_runtimes', load)
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    configured['config'].write_text('{"untrusted_later_change": true}')
    async def exercise():
        for _ in range(2):
            result = structured(await server.call_tool('probe_native_processing', {
                'provider': 'groot_state', 'model': str(configured['model']), 'dataset': str(configured['dataset']),
                'samples': [{'episode_index': 0, 'frame_index': 1}]}))
            assert result['status'] == 'observed'
    run(exercise())
    assert loads == [str(configured['config'])]
    assert len(parent_calls) == 2
    assert parent_calls[0][-1]['runtimes'] == parent_calls[1][-1]['runtimes']
    assert set(parent_calls[0][-1]['runtimes']) == {'groot_state'}


def test_bad_runtime_config_fails_server_startup(configured):
    configured['config'].write_text('{"providers": {"arbitrary_import": {}}}')
    with pytest.raises(InputError):
        create_server([str(configured['read'])], native_runtime=str(configured['config']))


@pytest.mark.parametrize('denied_role', ['model', 'dataset', 'replay'])
def test_read_root_guard_precedes_parent_api(configured, parent_calls, denied_role):
    configured['outside'].mkdir()
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        from mcp.server.fastmcp.exceptions import ToolError

        if denied_role == 'replay':
            name, arguments = 'reproduce_native_processing', {'directory': str(configured['outside'])}
        else:
            name = 'probe_native_processing'
            arguments = {'provider': 'groot_state', 'model': str(configured['model']), 'dataset': str(configured['dataset']),
                         'samples': [{'episode_index': 0, 'frame_index': 1}], denied_role: str(configured['outside'])}
        with pytest.raises(ToolError, match='outside'):
            await server.call_tool(name, arguments)
    run(exercise())
    assert parent_calls == []


def test_symlink_to_outside_read_root_is_denied_before_parent(configured, parent_calls):
    configured['outside'].mkdir()
    link = configured['read'] / 'link-to-outside'
    link.symlink_to(configured['outside'], target_is_directory=True)
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match='outside'):
            await server.call_tool('probe_native_processing', {'provider': 'groot_state', 'model': str(link),
                'dataset': str(configured['dataset']), 'samples': [{'episode_index': 0, 'frame_index': 0}]})
    run(exercise())
    assert not parent_calls


def test_retention_requires_operator_case_root(configured, parent_calls):
    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match='case root'):
            await server.call_tool('probe_native_processing', {'provider': 'groot_state', 'model': str(configured['model']),
                'dataset': str(configured['dataset']), 'samples': [{'episode_index': 0, 'frame_index': 0}],
                'destination': str(configured['outputs'] / 'capsule')})
    run(exercise())
    assert not parent_calls


@pytest.mark.parametrize('destination_kind', ['case_root_itself', 'outside', 'symlink'])
def test_retained_destination_must_be_strictly_inside_case_root(configured, parent_calls, destination_kind):
    target = configured['outputs']
    if destination_kind == 'outside':
        target = configured['outside']
    elif destination_kind == 'symlink':
        configured['outside'].mkdir()
        target = configured['outputs'] / 'link'
        target.symlink_to(configured['outside'], target_is_directory=True)
    server = create_server([str(configured['read'])], case_root=str(configured['outputs']),
                           native_runtime=str(configured['config']))
    async def exercise():
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            await server.call_tool('probe_native_processing', {'provider': 'groot_state', 'model': str(configured['model']),
                'dataset': str(configured['dataset']), 'samples': [{'episode_index': 0, 'frame_index': 0}],
                'destination': str(target)})
    run(exercise())
    assert not parent_calls


def test_allowed_calls_use_cached_operator_runtime_and_explicit_destination(configured, parent_calls):
    server = create_server([str(configured['read'])], case_root=str(configured['outputs']),
                           native_runtime=str(configured['config']))
    destination = configured['outputs'] / 'new-capsule'
    async def exercise():
        probed = structured(await server.call_tool('probe_native_processing', {
            'provider': 'groot_state', 'model': str(configured['model']), 'dataset': str(configured['dataset']),
            'samples': [{'episode_index': 0, 'frame_index': 0}], 'destination': str(destination)}))
        assert probed['status'] == 'observed'
        replayed = structured(await server.call_tool('reproduce_native_processing', {'directory': str(configured['capsule'])}))
        assert replayed['reproduction_matches'] is True
    run(exercise())
    assert parent_calls[0][1:4] == ('groot_state', str(configured['model']), str(configured['dataset']))
    assert parent_calls[0][-1]['destination'] == str(destination)
    assert parent_calls[1][1] == str(configured['capsule'])
    assert parent_calls[0][-1]['runtimes'] == parent_calls[1][-1]['runtimes']
    assert not destination.exists()  # fake parent fixture makes no native/publication claim


def test_actual_sdk_client_extra_runtime_arguments_are_rejected_before_native_execution(configured, parent_calls):
    """Undeclared runtime overrides must be refused before any trusted worker is called."""
    from mcp.shared.memory import create_connected_server_and_client_session

    server = create_server([str(configured['read'])], native_runtime=str(configured['config']))
    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool('probe_native_processing', {
                'provider': 'groot_state', 'model': str(configured['model']), 'dataset': str(configured['dataset']),
                'samples': [{'episode_index': 0, 'frame_index': 0}],
                'python': '/client/untrusted-python', 'source_root': '/client/untrusted-source',
                'worker_path': '/client/untrusted-worker.py', 'runtimes': {'groot_state': {'python': 'anything'}}})
            assert result.isError and 'extra' in result.content[0].text.lower()
            result = await client.call_tool('reproduce_native_processing', {
                'directory': str(configured['capsule']), 'python': '/client/untrusted-python',
                'source_root': '/client/untrusted-source', 'worker_path': '/client/untrusted-worker.py'})
            assert result.isError and 'extra' in result.content[0].text.lower()
    run(exercise())
    assert parent_calls == []
