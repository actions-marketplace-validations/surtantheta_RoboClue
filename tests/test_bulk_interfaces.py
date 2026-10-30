"""Actual generated recovery engines behind CLI/MCP, plus refusal-before-I/O controls."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from recovery_fixtures import ORIGINAL, REPO, REVISION
from test_bulk_video_recovery import source

from policy_doctor import bulk_workflow as workflow
from policy_doctor import video_recovery as vr
from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.server import _summary, create_server


def arguments(destination, *extra):
    return ['recover-video-references', REPO, str(destination), '--revision', REVISION,
            '--original-revision', ORIGINAL, '--fetch', *extra]


def test_cli_generated_full_recovery_and_offline_replay(monkeypatch, tmp_path, capsys):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    assert main(arguments(batch, '--json')) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['original_status'] == 'error' and report['proposal_status'] == 'pass'
    assert report['status'] == 'pass' and report['execution_status'] == 'completed'
    assert report['proposed_cells'] == 6 and report['proposal_complete']
    assert report['body_bytes_read'] is None and not report['upstream_modified']
    monkeypatch.setattr(vr, '_opener', lambda: pytest.fail('Offline replay tried network'))
    for command in ('reproduce-video-references', 'reproduce-case'):
        assert main([command, str(batch), '--json']) == 0
        replay = json.loads(capsys.readouterr().out)
        assert replay['reproduction_matches'] and replay['original_status'] == 'error'
        assert replay['execution_status'] == 'reproduced' and replay['proposal_complete']
        assert not replay['detector_coverage_replayed_from_raw_headers']


def test_cli_partial_unknown_remains_visible_after_successful_reproduction(monkeypatch, tmp_path, capsys):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'partial'
    assert main(arguments(batch, '--max-associations', '1')) == 3
    text = capsys.readouterr().out
    assert 'Original detector: ERROR' in text and 'Derived proposal: UNKNOWN; complete: False' in text
    assert 'coverage: truncated' in text and 'Original dataset modified: false' in text
    assert main(['reproduce-video-references', str(batch)]) == 0
    text = capsys.readouterr().out
    assert 'Execution: REPRODUCED' in text and 'Derived proposal: UNKNOWN' in text


def test_cli_unresolved_associations_and_budget_accounting(monkeypatch, tmp_path, capsys):
    source(monkeypatch, tmp_path, bad_original=True)
    assert main(arguments(tmp_path / 'bad', '--json')) == 3
    result = json.loads(capsys.readouterr().out)
    concise = _summary(result)
    assert concise['original_status'] == 'error' and concise['proposal_status'] == 'unknown'
    assert concise['verified_associations'] == 0 and concise['attempted_associations'] == 2
    assert len(concise['unresolved_associations']) == 2
    assert concise['request']['max_body_bytes'] == 512 * 1024**2
    assert concise['recovery_body_bytes_read'] > 0
    assert concise['response_body_upper_bound'] <= concise['request']['max_body_bytes']
    assert not concise['proposal_complete'] and len(json.dumps(concise).encode()) <= 8192


@pytest.mark.parametrize('field,value', [
    ('max_associations', True), ('max_associations', 0), ('max_associations', 257),
    ('max_response_body_mb', False), ('max_response_body_mb', 0), ('max_response_body_mb', 8193),
    ('max_intake_mb', True), ('max_intake_mb', 301), ('max_rows', True), ('max_rows', 200001),
    ('max_files', False), ('max_files', 1001), ('fetch', 1), ('fetch', False),
    ('revision', 'main'), ('original_revision', 'main')])
def test_bad_options_refuse_before_engine_or_destination(monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(workflow.engine, 'recover_all_video_references', lambda *a, **k: pytest.fail('engine called'))
    options = dict(repo_id=REPO, revision=REVISION, original_revision=ORIGINAL,
                   destination=tmp_path / 'not-created', fetch=True)
    options[field] = value
    with pytest.raises(InputError):
        workflow.recover_video_references(**options)
    assert not options['destination'].exists()


def test_insufficient_aggregate_budget_refuses_before_network(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(workflow.engine.hub, 'resolve_dataset_content', lambda *a, **k: pytest.fail('network intake'))
    batch = tmp_path / 'not-created'
    assert main(arguments(batch, '--max-response-body-mb', '32', '--json')) == 2
    assert 'reservation' in capsys.readouterr().err and not batch.exists()


def test_explicit_large_budget_is_passed_without_silent_raise(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    result = workflow.recover_video_references(REPO, REVISION, ORIGINAL, tmp_path / 'large', fetch=True,
        max_response_body_mb=2048, max_intake_mb=8, max_rows=200000)
    assert result['request']['max_body_bytes'] == 2048 * 1024**2
    assert result['request']['max_rows'] == 200000
    assert result['response_body_upper_bound'] <= 2048 * 1024**2


def test_mcp_catalog_opt_in_and_annotations(tmp_path):
    for options, expected in [({}, 29), ({'case_root': str(tmp_path)}, 31),
                              ({'allow_hub_datasets': True}, 29),
                              ({'case_root': str(tmp_path), 'allow_hub_media': True}, 34)]:
        tools = create_server([str(tmp_path)], **options)._tool_manager._tools
        assert len(tools) == expected
        assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
        assert ('recover_dataset_video_references' in tools) == bool(options.get('allow_hub_media'))
    recover = tools['recover_dataset_video_references']
    replay = tools['reproduce_dataset_video_references']
    assert not recover.annotations.readOnlyHint and recover.annotations.openWorldHint
    assert replay.annotations.readOnlyHint and not replay.annotations.openWorldHint
    assert recover.parameters['additionalProperties'] is False
    assert recover.parameters['properties']['max_response_body_mb']['maximum'] == 8192


@pytest.mark.parametrize('destination', ['', '/etc/outside', '.'])
def test_mcp_write_scope_refuses_before_public_intake(monkeypatch, tmp_path, destination):
    monkeypatch.setattr(workflow, 'recover_video_references', lambda *a, **k: pytest.fail('public intake'))
    fn = create_server([str(tmp_path)], case_root=str(tmp_path), allow_hub_media=True)._tool_manager._tools[
        'recover_dataset_video_references'].fn
    if destination == '.':
        destination = str(tmp_path)
    with pytest.raises(InputError):
        fn(REPO, REVISION, ORIGINAL, destination)


def test_mcp_symlink_destination_escape_and_existing_refused(monkeypatch, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    (tmp_path / 'escape').symlink_to(outside, target_is_directory=True)
    (tmp_path / 'existing').mkdir()
    monkeypatch.setattr(workflow, 'recover_video_references', lambda *a, **k: pytest.fail('public intake'))
    fn = create_server([str(tmp_path)], case_root=str(tmp_path), allow_hub_media=True)._tool_manager._tools[
        'recover_dataset_video_references'].fn
    for target in (tmp_path / 'escape', tmp_path / 'escape' / 'nested', tmp_path / 'existing'):
        with pytest.raises(InputError):
            fn(REPO, REVISION, ORIGINAL, str(target))


def test_mcp_actual_engine_recovery_and_schema_refusals(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    server = create_server([str(tmp_path)], case_root=str(tmp_path), allow_hub_media=True)
    fn = server._tool_manager._tools['recover_dataset_video_references'].fn
    result = fn(REPO, REVISION, ORIGINAL, str(tmp_path / 'batch'))
    assert result['summary'] and result['original_status'] == 'error'
    assert result['proposal_status'] == 'pass' and result['proposed_cells'] == 6
    assert result['proposal_complete'] and result['request']['max_body_bytes'] == 512 * 1024**2

    async def run():
        base = dict(repo_id=REPO, revision=REVISION, original_revision=ORIGINAL, destination=str(tmp_path / 'new'))
        for extra in ({'max_response_body_mb': True}, {'max_associations': '1'}, {'max_filez': 2}):
            with pytest.raises(Exception, match="max_response_body_mb|max_associations|max_filez"):
                await server.call_tool('recover_dataset_video_references', dict(base, **extra))
    asyncio.run(run())
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('marker', ['BATCH_MANIFEST.json', 'BATCH_READY.json'])
@pytest.mark.parametrize('mode', ['missing', 'symlink', 'broken_symlink', 'both'])
def test_batch_markers_never_fall_back_to_legacy(monkeypatch, tmp_path, capsys, marker, mode):
    batch = tmp_path / 'batch'
    batch.mkdir()
    target = tmp_path / 'outside-secret'
    target.write_text('must not read')
    if mode == 'missing':
        (batch / marker).write_text('{}')
    elif mode == 'both':
        (batch / marker).write_text('{}')
        (batch / 'manifest.json').write_text('{"kind":"robot_video_reference_recovery"}')
    else:
        (batch / marker).symlink_to(target if mode == 'symlink' else tmp_path / 'absent')
    if mode == 'both':
        with pytest.raises(InputError, match='Ambiguous'):
            workflow.is_bulk_case(batch)
    else:
        assert workflow.is_bulk_case(batch)
    original_read = Path.open
    def guarded(path, *a, **k):
        assert path != target, 'outside target opened'
        return original_read(path, *a, **k)
    monkeypatch.setattr(Path, 'open', guarded)
    assert main(['reproduce-case', str(batch), '--json']) == 2
    assert capsys.readouterr().err


def test_real_stdio_offline_batch_replay_and_scope_refusal(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    workflow.recover_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True, max_associations=1)
    async def run():
        args = ['-m', 'policy_doctor', 'serve', '--root', str(tmp_path), '--case-root', str(tmp_path), '--allow-hub-media']
        params = StdioServerParameters(command=sys.executable, args=args,
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                for format in ('concise', 'detailed'):
                    result = await client.call_tool('reproduce_dataset_video_references', {
                        'directory': str(batch), 'response_format': format})
                    assert not result.isError
                    body = result.structuredContent
                    assert body['status'] == 'pass' and body['reproduction_matches']
                    assert body['original_status'] == 'error' and body['batch_status'] == 'unknown'
                    assert not body['proposal_complete'] and body['detection_coverage'] == 'truncated'
                result = await client.call_tool('reproduce_incident_case', {'directory': str(batch)})
                assert not result.isError and not result.structuredContent['proposal_complete']
                refused = await client.call_tool('reproduce_dataset_video_references', {'directory': '/etc'})
                assert refused.isError and 'outside' in refused.content[0].text
    asyncio.run(asyncio.wait_for(run(), 30))


@pytest.mark.parametrize('marker', ['BATCH_MANIFEST.json', 'BATCH_READY.json'])
def test_existing_batch_hostile_marker_is_refused_before_outside_read(monkeypatch, tmp_path, capsys, marker):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    workflow.recover_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True)
    target = tmp_path / 'outside-secret'
    target.write_bytes((batch / marker).read_bytes())
    (batch / marker).unlink()
    (batch / marker).symlink_to(target)
    original_read = os.read
    def guarded(fd, length):
        assert Path(f'/proc/self/fd/{fd}').resolve() != target, 'outside target read'
        return original_read(fd, length)
    monkeypatch.setattr(os, 'read', guarded)
    assert main(['reproduce-case', str(batch), '--json']) == 2
    assert 'no-follow' in capsys.readouterr().err


def test_adapter_does_not_display_stale_report_during_replay(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    workflow.recover_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True)
    replay = workflow.engine.reproduce_bulk_proposal
    def changed(root):
        result = replay(root)
        report = json.loads((root / 'report.json').read_text())
        report['scope'] = 'changed while preparing display'
        (root / 'report.json').write_bytes(vr._encode(report))
        workflow.engine._seal(root)
        return result
    monkeypatch.setattr(workflow.engine, 'reproduce_bulk_proposal', changed)
    with pytest.raises(InputError, match='changed during report'):
        workflow.reproduce_video_references(batch)


def test_malformed_batch_is_structured_cli_refusal(tmp_path, capsys):
    batch = tmp_path / 'batch'
    batch.mkdir()
    (batch / 'report.json').write_text('{}')
    (batch / 'BATCH_MANIFEST.json').write_text('{}')
    assert main(['reproduce-video-references', str(batch), '--json']) == 2
    assert 'invalid_input' in capsys.readouterr().err


@pytest.mark.parametrize('mode', ['existing', 'inside', 'symlink'])
def test_bulk_report_output_refuses_before_recovery(monkeypatch, tmp_path, capsys, mode):
    batch = tmp_path / 'batch'
    output = batch / 'display.json' if mode == 'inside' else tmp_path / 'report.json'
    if mode == 'existing':
        output.write_text('existing')
    elif mode == 'symlink':
        output.symlink_to(batch / 'report.json')
    monkeypatch.setattr(workflow.engine, 'recover_all_video_references', lambda *a, **k: pytest.fail('recovery dispatched'))
    assert main(arguments(batch, '--json', '--output', str(output))) == 2
    assert 'Report output' in capsys.readouterr().err and not batch.exists()


def test_replay_report_export_cannot_invalidate_retained_evidence(monkeypatch, tmp_path, capsys):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    workflow.recover_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True)
    for command in ('reproduce-video-references', 'reproduce-case'):
        assert main([command, str(batch), '--json', '--output', str(batch / 'display.json')]) == 2
        assert 'outside' in capsys.readouterr().err and not (batch / 'display.json').exists()
    output = tmp_path / 'display.json'
    assert main(['reproduce-video-references', str(batch), '--json', '--output', str(output)]) == 0
    assert json.loads(output.read_text())['reproduction_matches']
    assert workflow.reproduce_video_references(batch)['proposal_complete']


def test_invalid_response_format_refuses_before_mcp_write_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, 'recover_video_references', lambda *a, **k: pytest.fail('write dispatch'))
    fn = create_server([str(tmp_path)], case_root=str(tmp_path), allow_hub_media=True)._tool_manager._tools[
        'recover_dataset_video_references'].fn
    with pytest.raises(InputError, match='response_format'):
        fn(REPO, REVISION, ORIGINAL, str(tmp_path / 'new'), response_format='invalid')


def test_generic_replay_markdown_preserves_batch_verdicts(monkeypatch, tmp_path, capsys):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    workflow.recover_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True, max_associations=1)
    assert main(['reproduce-case', str(batch), '--markdown']) == 0
    text = capsys.readouterr().out
    assert '# RoboClue derived video-reference proposal' in text
    assert 'Original detector: ERROR' in text and 'Derived proposal: UNKNOWN' in text
