"""Recovery handoff: exact proposed metadata paths, bounded before->after cells and the complete patch on request."""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from recovery_fixtures import CAMERA, ORIGINAL, REPO, REVISION
from test_bulk_video_recovery import source

from policy_doctor import bulk_workflow as workflow
from policy_doctor import recovery_handoff as handoff_view
from policy_doctor import video_recovery as vr
from policy_doctor.artifacts import InputError
from policy_doctor.cli import main
from policy_doctor.reports import markdown_report
from policy_doctor.server import SUMMARY_LIMIT, _summary

SRC = Path(__file__).resolve().parents[1] / 'src'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(monkeypatch, tmp_path, **kwargs):
    source(monkeypatch, tmp_path)
    batch = tmp_path / 'batch'
    result = workflow.recover_video_references(REPO, REVISION, ORIGINAL, batch, fetch=True, **kwargs)
    return handoff_view.batch_handoff(result), batch


def test_batch_result_names_the_proposed_copy_and_shows_cells(monkeypatch, tmp_path):
    result, batch = recover(monkeypatch, tmp_path)
    handoff = result['handoff']
    proposals = json.loads((batch / 'proposals.json').read_text())
    assert result['proposed_cells'] == len(proposals) == 6
    [listed] = handoff['proposed_metadata_files']
    assert listed['dataset_path'] == proposals[0]['path'] and listed['cells'] == 6
    assert listed['proposed_copy'] == str(batch / 'proposed_metadata' / listed['dataset_path'])
    assert listed['proposed_sha256'] == sha(listed['proposed_copy'])
    assert listed['original_copy'] == str(batch / 'original_metadata' / listed['dataset_path'])
    assert listed['original_sha256'] == sha(listed['original_copy']) != listed['proposed_sha256']
    assert handoff['patch_preview'] == [{k: c[k] for k in ('episode_index', 'path', 'row', 'column', 'old', 'new')}
                                        for c in proposals]
    assert handoff['patch_preview_omitted'] == 0
    complete = handoff['complete_patch']
    assert complete['path'] == str(batch / 'proposals.json') and complete['sha256'] == sha(batch / 'proposals.json')
    assert complete['cells'] == 6 and complete['command'][-2:] == ['--patch', '--json']
    assert 'include_patch=true' in complete['mcp']
    assert 'not a complete dataset' in handoff['overlay_scope']
    assert set(handoff['validation_scope']) == {'replay', 'native_consumer'}
    assert handoff['validation_scope']['native_consumer'].startswith('Not run')
    assert 'patch' not in result  # the complete patch only on request


def test_preview_is_bounded_and_representative():
    proposals = [{'episode_index': e, 'path': 'meta/episodes/chunk-000/file-000.parquet', 'row': e,
                  'column': f'videos/{camera}/{name}', 'old': 0, 'new': 1}
                 for e, camera in [(i, 'observation.images.wrist') for i in range(10)]
                 + [(50, 'observation.images.image')] for name in ('file_index', 'from_timestamp', 'to_timestamp')]
    preview = handoff_view._representative(proposals, handoff_view.PATCH_PREVIEW_CELLS)
    assert len(preview) == 12 and preview == sorted(preview, key=proposals.index)
    assert {c['episode_index'] for c in preview} >= {0, 50}  # the first episode of each camera is shown whole
    assert [c for c in preview if c['episode_index'] == 50] == proposals[-3:]


def test_cli_replay_prints_paths_preview_and_complete_patch(monkeypatch, tmp_path, capsys):
    result, batch = recover(monkeypatch, tmp_path)
    monkeypatch.setattr(vr, '_opener', lambda: pytest.fail('Offline replay tried network'))
    assert main(['reproduce-video-references', str(batch)]) == 0
    text = capsys.readouterr().out
    copy = result['handoff']['proposed_metadata_files'][0]['proposed_copy']
    assert 'Proposed metadata copy (overlay; only changed files): ' in text and json.dumps(copy) in text
    proposals = json.loads((batch / 'proposals.json').read_text())
    assert 'Patch preview (6 of 6 cells):' in text
    assert f"{json.dumps(proposals[0]['old'])} -> {json.dumps(proposals[0]['new'])}" in text
    assert 'Complete patch: ' + json.dumps(str(batch / 'proposals.json')) in text
    assert 'Metadata overlay, not a complete dataset' in text
    assert 'Replay: ' in text and 'Native consumer validation: Not run' in text
    for command in ('reproduce-video-references', 'reproduce-case'):
        assert main([command, str(batch), '--patch', '--json']) == 0
        replay = json.loads(capsys.readouterr().out)
        assert replay['reproduction_matches'] and replay['patch']['truncated'] is False
        assert replay['patch']['cells'] == json.loads((batch / 'proposals.json').read_text())
        assert replay['patch']['sha256'] == sha(batch / 'proposals.json')
    assert main(['reproduce-video-references', str(batch), '--patch']) == 0
    text = capsys.readouterr().out
    assert 'Complete patch (6 cells):' in text and text.count('  episode ') >= 6 + 6
    assert main(['reproduce-case', str(batch), '--markdown']) == 0
    assert 'Patch preview (6 of 6 cells)' in capsys.readouterr().out


def test_patch_bytes_bound_marks_truncation(monkeypatch, tmp_path):
    _, batch = recover(monkeypatch, tmp_path)
    replay = handoff_view.batch_handoff(workflow.reproduce_video_references(batch), include_patch=True, patch_bytes=400)
    assert replay['patch']['truncated'] is True and 0 < len(replay['patch']['cells']) < 6
    assert replay['patch']['cell_count'] == 6 and replay['handoff']['complete_patch']['path'] == str(batch / 'proposals.json')


def test_partial_batch_keeps_preview_and_completeness_separate(monkeypatch, tmp_path):
    result, batch = recover(monkeypatch, tmp_path, max_associations=1)
    assert result['proposal_complete'] is False and result['proposed_cells'] == 3
    assert len(result['handoff']['patch_preview']) == 3
    assert result['handoff']['complete_patch']['cells'] == 3


def test_concise_summary_keeps_the_handoff_within_8_kib(monkeypatch, tmp_path):
    result, batch = recover(monkeypatch, tmp_path)
    concise = _summary(result)
    handoff = concise['handoff']
    assert handoff['proposed_metadata_files'][0]['proposed_copy'] == result['handoff']['proposed_metadata_files'][0]['proposed_copy']
    assert 1 <= len(handoff['patch_preview']) <= 4 and handoff['complete_patch']['cells'] == 6
    assert handoff['patch_preview_omitted'] == 6 - len(handoff['patch_preview'])
    assert len(json.dumps(concise).encode()) <= SUMMARY_LIMIT
    long = dict(result, handoff=dict(result['handoff'], patch_preview=[dict(c, column='x' * 5000)
                                                                       for c in result['handoff']['patch_preview']]))
    assert len(json.dumps(_summary(long)).encode()) <= SUMMARY_LIMIT


def test_single_case_replay_reports_patch_values_and_derived_copy(monkeypatch, tmp_path, capsys):
    source(monkeypatch, tmp_path)
    case = tmp_path / 'case'
    vr.recover_video_reference(REPO, REVISION, ORIGINAL, 0, CAMERA, case)
    plan = json.loads((case / 'repair_plan.json').read_text())
    replay = handoff_view.case_handoff(vr.reproduce_video_recovery(case), case)
    assert replay['status'] == 'pass' and replay['patch'] == plan and replay['patch_cells'] == len(plan) == 3
    assert replay['patch_truncated'] is False
    derived = replay['derived_metadata']
    assert derived['path'] == str(case / 'corrected_metadata.parquet') and derived['dataset_path'] == plan[0]['path']
    assert derived['sha256'] == sha(case / 'corrected_metadata.parquet')
    assert 'only this episode' in replay['overlay_scope']
    assert replay['validation_scope']['native_consumer'].startswith('Not run')
    assert main(['reproduce-case', str(case)]) == 0
    text = capsys.readouterr().out
    assert text.count('Metadata proposal: row ') == 3 and 'Derived metadata copy: ' in text
    assert json.dumps(plan[0]['path']) in text
    assert '## Proposed metadata changes' in markdown_report(replay)


def test_single_case_replay_with_verifier_drift_reports_no_values(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    case = tmp_path / 'case'
    vr.recover_video_reference(REPO, REVISION, ORIGINAL, 0, CAMERA, case)
    monkeypatch.setattr(vr, '_fingerprint', lambda: 'drifted')
    replay = handoff_view.case_handoff(vr.reproduce_video_recovery(case), case)
    assert replay['status'] == 'unknown' and replay['patch'] == [] and replay['derived_metadata'] is None


def test_real_stdio_handoff_and_complete_patch(monkeypatch, tmp_path):
    _, batch = recover(monkeypatch, tmp_path)
    case = tmp_path / 'case'
    vr.recover_video_reference(REPO, REVISION, ORIGINAL, 1, CAMERA, case)
    expected = json.loads((batch / 'proposals.json').read_text())

    async def run():
        args = ['-B', '-m', 'policy_doctor', 'serve', '--root', str(tmp_path), '--case-root', str(tmp_path),
                '--allow-hub-media']
        params = StdioServerParameters(command=sys.executable, args=args,
                                       env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1'))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = {t.name: t for t in (await client.list_tools()).tools}
                for name in ('reproduce_dataset_video_references', 'reproduce_incident_case'):
                    assert tools[name].inputSchema['properties']['include_patch']['type'] == 'boolean'
                concise = (await client.call_tool('reproduce_dataset_video_references', {'directory': str(batch)}))
                body = concise.structuredContent
                jsonschema.validate(body, tools['reproduce_dataset_video_references'].outputSchema)
                assert body['handoff']['complete_patch']['path'] == str(batch / 'proposals.json')
                assert len(json.dumps(body).encode()) <= SUMMARY_LIMIT
                for name in ('reproduce_dataset_video_references', 'reproduce_incident_case'):
                    full = (await client.call_tool(name, {'directory': str(batch), 'include_patch': True}))
                    assert not full.isError
                    jsonschema.validate(full.structuredContent, tools[name].outputSchema)
                    assert full.structuredContent['patch']['cells'] == expected
                single = (await client.call_tool('reproduce_incident_case', {'directory': str(case)})).structuredContent
                assert single['patch'] == json.loads((case / 'repair_plan.json').read_text())
                assert single['derived_metadata']['path'] == str(case / 'corrected_metadata.parquet')
    asyncio.run(asyncio.wait_for(run(), 120))


def test_patch_option_is_refused_for_other_case_kinds(tmp_path, capsys):
    case = tmp_path / 'incident'
    case.mkdir()
    (case / 'manifest.json').write_text(json.dumps({'kind': 'robot_incident_case'}))
    assert main(['reproduce-case', str(case), '--patch']) == 2
    assert '--patch applies to video-reference recovery batches and cases' in capsys.readouterr().err
    with pytest.raises(Exception, match='strict boolean'):
        handoff_view.batch_handoff({'directory': str(case)}, include_patch='yes')


def _single_replay(monkeypatch, tmp_path):
    source(monkeypatch, tmp_path)
    case = tmp_path / 'case'
    vr.recover_video_reference(REPO, REVISION, ORIGINAL, 0, CAMERA, case)
    replay = vr.reproduce_video_recovery(case)
    assert replay['status'] == 'pass' and replay['reproduction_matches'] is True
    return case, replay


def _reseal_changed_case(case, names):
    """An adversary can reseal a local manifest; the old replay's report identity must still gate presentation."""
    manifest = json.loads((case / 'manifest.json').read_text())
    for name in names:
        manifest['files'][name] = sha(case / name)
    raw = vr._encode(manifest)
    (case / 'manifest.json').write_bytes(raw)
    (case / 'READY.json').write_bytes(vr._encode({'schema_version': 1, 'manifest_sha256': vr._sha(raw)}))


@pytest.mark.parametrize('name', ['manifest.json', 'repair_plan.json', 'report.json', 'corrected_metadata.parquet'])
def test_handoff_refuses_post_replay_symlinks_before_reading_their_targets(monkeypatch, tmp_path, name):
    case, replay = _single_replay(monkeypatch, tmp_path)
    target = tmp_path / 'outside-case'
    (case / name).rename(target)
    (case / name).symlink_to(target)
    with pytest.raises(InputError):
        handoff_view.case_handoff(replay, case)


@pytest.mark.parametrize('replace_report', [False, True])
def test_handoff_rejects_regular_file_patch_substitution_even_with_a_fresh_seal(monkeypatch, tmp_path, replace_report):
    case, replay = _single_replay(monkeypatch, tmp_path)
    plan = json.loads((case / 'repair_plan.json').read_text())
    plan[0]['new'] = 31337
    (case / 'repair_plan.json').write_bytes(vr._encode(plan))
    names = ['repair_plan.json']
    if replace_report:
        report = json.loads((case / 'report.json').read_text())
        report['patch'] = plan
        (case / 'report.json').write_bytes(vr._encode(report))
        names.append('report.json')
    _reseal_changed_case(case, names)
    with pytest.raises(InputError, match='differs from the replayed content'):
        handoff_view.case_handoff(replay, case)


def test_handoff_binds_the_derived_copy_to_the_replayed_metadata_not_a_fresh_manifest(monkeypatch, tmp_path):
    case, replay = _single_replay(monkeypatch, tmp_path)
    path = case / 'corrected_metadata.parquet'
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    _reseal_changed_case(case, ['corrected_metadata.parquet'])
    with pytest.raises(InputError, match='Derived metadata differs from the replayed content'):
        handoff_view.case_handoff(replay, case)


def test_handoff_reads_remain_bounded_and_nonblocking_after_replay(monkeypatch, tmp_path):
    case, replay = _single_replay(monkeypatch, tmp_path)
    plan = case / 'repair_plan.json'
    plan.unlink()
    os.mkfifo(plan)
    with pytest.raises(InputError, match='bounded regular file'):
        handoff_view.case_handoff(replay, case)
    plan.unlink()
    plan.write_bytes(b' ' * (vr.MAX_METADATA + 1))
    with pytest.raises(InputError, match='bounded regular file'):
        handoff_view.case_handoff(replay, case)


def test_handoff_detects_changes_during_native_copy_reconstruction(monkeypatch, tmp_path):
    case, replay = _single_replay(monkeypatch, tmp_path)
    original = vr._corrected_metadata

    def change_after_read(selected, patch):
        value = original(selected, patch)
        (case / 'repair_plan.json').write_bytes(b'[]')
        return value

    monkeypatch.setattr(vr, '_corrected_metadata', change_after_read)
    with pytest.raises(InputError, match='changed while reading'):
        handoff_view.case_handoff(replay, case)


def _reseal_changed_batch(batch, names, *, update_ready=True):
    manifest = json.loads((batch / 'BATCH_MANIFEST.json').read_text())
    for name in names:
        manifest['files'][name] = sha(batch / name)
    raw = vr._encode(manifest)
    (batch / 'BATCH_MANIFEST.json').write_bytes(raw)
    if update_ready:
        (batch / 'BATCH_READY.json').write_bytes(vr._encode({'manifest_sha256': vr._sha(raw)}))


@pytest.mark.parametrize('update_ready', [False, True])
def test_batch_handoff_rejects_post_replay_proposal_reseal(monkeypatch, tmp_path, update_ready):
    _, batch = recover(monkeypatch, tmp_path)
    replay = workflow.reproduce_video_references(batch)  # raw kernel result has no content digest
    proposals = json.loads((batch / 'proposals.json').read_text())
    proposals[0]['new'] = 31337
    (batch / 'proposals.json').write_bytes(vr._encode(proposals))
    _reseal_changed_batch(batch, ['proposals.json'], update_ready=update_ready)
    with pytest.raises(InputError):
        handoff_view.batch_handoff(replay, include_patch=True)


def test_batch_handoff_rejects_resealed_native_copy_after_replay(monkeypatch, tmp_path):
    _, batch = recover(monkeypatch, tmp_path)
    replay = workflow.reproduce_video_references(batch)
    name = 'proposed_metadata/meta/episodes/chunk-000/file-000.parquet'
    path = batch / name
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    _reseal_changed_batch(batch, [name])
    with pytest.raises(InputError):
        handoff_view.batch_handoff(replay)


def test_batch_handoff_rejects_summary_substitution_with_a_fresh_manifest(monkeypatch, tmp_path):
    _, batch = recover(monkeypatch, tmp_path)
    replay = workflow.reproduce_video_references(batch)
    report = json.loads((batch / 'report.json').read_text())
    report['scope'] = 'Changed after the original replay'
    (batch / 'report.json').write_bytes(vr._encode(report))
    _reseal_changed_batch(batch, ['report.json'])
    with pytest.raises(InputError, match='differs from the replayed snapshot'):
        handoff_view.batch_handoff(replay)


def test_batch_handoff_refuses_post_replay_links_and_bounds_the_snapshot(monkeypatch, tmp_path):
    _, batch = recover(monkeypatch, tmp_path)
    replay = workflow.reproduce_video_references(batch)
    proposal = batch / 'proposals.json'
    target = tmp_path / 'outside-proposals.json'
    proposal.rename(target)
    proposal.symlink_to(target)
    with pytest.raises(InputError):
        handoff_view.batch_handoff(replay)


def test_batch_handoff_checks_snapshot_after_fresh_replay(monkeypatch, tmp_path):
    _, batch = recover(monkeypatch, tmp_path)
    replay = workflow.reproduce_video_references(batch)
    original = workflow.reproduce_video_references

    def change_after_replay(directory):
        result = original(directory)
        (batch / 'proposals.json').write_bytes(b'[]')
        return result

    monkeypatch.setattr(workflow, 'reproduce_video_references', change_after_replay)
    with pytest.raises(InputError, match='changed during replay'):
        handoff_view.batch_handoff(replay)
