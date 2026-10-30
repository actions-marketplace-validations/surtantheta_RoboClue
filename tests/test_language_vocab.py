"""Lexical catalog membership never certifies native equivalence or capability."""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from policy_doctor import hub, product_cli
from policy_doctor.language_vocab import (
    MAX_TASKS,
    augment_report,
    check_language_vocabulary,
    evaluate_vocabulary,
    instruction_inputs,
    lexical_key,
)


def fixture(tmp_path, tasks=('OpenDrawer', 'CloseFridge'), *, layout='column', policy='smolvla', revision=None):
    checkpoint, catalog = tmp_path / 'checkpoint', tmp_path / 'catalog'
    checkpoint.mkdir(parents=True)
    (catalog / 'meta').mkdir(parents=True)
    (checkpoint / 'config.json').write_text(json.dumps({'type': policy, 'input_features': {}, 'output_features': {}}))
    (checkpoint / 'train_config.json').write_text(json.dumps({'dataset': {'repo_id': 'org/robot', 'revision': revision}}))
    if layout == 'jsonl':
        (catalog / 'meta/tasks.jsonl').write_text(''.join(json.dumps({'task_index': i, 'task': t}) + '\n' for i, t in enumerate(tasks)))
    else:
        column = 'task' if layout == 'column' else '__index_level_0__'
        pq.write_table(pa.table({column: list(tasks), 'task_index': list(range(len(tasks)))}), catalog / 'meta/tasks.parquet')
    return checkpoint, catalog


@pytest.mark.parametrize('layout', ['column', 'index', 'jsonl'])
def test_closefridge_report_pattern_is_warning_only_with_lexical_suggestion(tmp_path, layout):
    checkpoint, catalog = fixture(tmp_path, layout=layout)
    r = check_language_vocabulary(checkpoint, instruction='Close the fridge doors.', catalog_directory=str(catalog))
    assert r['status'] == 'warning' and r['verdict'] == 'OUT_OF_CATALOG'
    item = r['evaluations'][0]
    assert item['nearest_task'] == 'CloseFridge' and 'doors' in item['absent_catalog_words']
    assert item['similarity'] < 1 and item['native_token_path_equivalence'] == 'unknown'
    assert not any(f['status'] == 'error' for f in r['findings'])
    assert 'never substitute' in r['findings'][0]['remedy']
    assert r['physical_capability'] == 'unassessed'


@pytest.mark.parametrize('instruction,match', [('CloseFridge', 'exact'), ('  CLOSEFRIDGE ! ', 'lexical')])
def test_matching_controls_are_silent_for_warning_but_never_pipeline_pass(tmp_path, instruction, match):
    checkpoint, catalog = fixture(tmp_path)
    r = check_language_vocabulary(checkpoint, instruction=instruction, catalog_directory=str(catalog))
    assert r['verdict'] == 'IN_CATALOG' and r['status'] == 'unknown'
    assert r['evaluations'][0]['match'] == match
    assert not any(f['status'] in ('error', 'warning') for f in r['findings'])
    assert r['native_token_path_equivalence'] == 'unknown'
    assert any(f['code'] == 'language.catalog_provenance_unknown' for f in r['findings'])
    if match == 'lexical':
        assert any(f['code'] == 'language.lexical_equivalence_unknown' for f in r['findings'])
    assert 'does not establish whether changing instructions changes behavior' in r['catalog']['few_task_note']


def test_lexical_fold_is_explicit_and_not_semantic():
    assert lexical_key('  put_the_cup_on_the_table! ') == 'put the cup on the table'
    r = evaluate_vocabulary(['put the cup on the table'], ['put the cup under the table'])
    assert r[0]['verdict'] == 'OUT_OF_CATALOG' and 'under' in r[0]['absent_catalog_words']


def test_nearest_similarity_never_certifies_negated_or_compound_goals():
    r = evaluate_vocabulary(['CloseFridge', 'OpenDrawer'], ['do not close the fridge', 'OpenDrawer then CloseFridge'])
    assert all(x['verdict'] == 'OUT_OF_CATALOG' for x in r)
    assert r[1]['compound_marker'] is True


@pytest.mark.parametrize('kind', ['no_training', 'no_instruction', 'empty', 'unsupported', 'offline', 'wrong_repo', 'invalid_revision'])
def test_unknown_inputs_never_pass(tmp_path, kind):
    checkpoint, catalog = fixture(tmp_path, policy='groot' if kind == 'unsupported' else 'smolvla')
    kwargs = {'instruction': 'CloseFridge', 'catalog_directory': str(catalog)}
    if kind == 'no_training':
        (checkpoint / 'train_config.json').unlink()
    elif kind == 'no_instruction':
        kwargs.pop('instruction')
    elif kind == 'empty':
        kwargs['instruction'] = ''
    elif kind == 'offline':
        kwargs.pop('catalog_directory')
    elif kind in ('wrong_repo', 'invalid_revision'):
        (checkpoint / 'train_config.json').write_text(json.dumps({'dataset': {
            'repo_id': 'org/../bad' if kind == 'wrong_repo' else 'org/robot',
            'revision': '../main' if kind == 'invalid_revision' else None}}))
    r = check_language_vocabulary(checkpoint, **kwargs)
    assert r['verdict'] == 'UNKNOWN' and r['status'] == 'unknown'
    assert all(f['status'] == 'unknown' for f in r['findings'])


@pytest.mark.parametrize('value', [None, False, 1, [], {}, 'x' * 4097, '\ud800', float('inf'), float('nan')])
def test_malformed_instruction_is_bounded_unknown(tmp_path, value):
    checkpoint, catalog = fixture(tmp_path)
    r = check_language_vocabulary(checkpoint, instruction=value, catalog_directory=str(catalog))
    assert r['status'] == 'unknown' and r['verdict'] == 'UNKNOWN'
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('tasks', [[0, 1], ['', 'ok'], [None, 'ok'], ['x' * 2049], ['task'] * (MAX_TASKS + 1)])
def test_malformed_catalog_never_certifies_partial_or_fabricated_labels(tmp_path, tasks):
    checkpoint, catalog = fixture(tmp_path, tasks=tasks)
    r = check_language_vocabulary(checkpoint, instruction='ok', catalog_directory=str(catalog))
    assert r['status'] == 'unknown' and r['verdict'] == 'UNKNOWN'


def test_lying_row_group_is_refused_before_read(tmp_path, monkeypatch):
    checkpoint, catalog = fixture(tmp_path)
    original = pq.ParquetFile

    def fake(*args, **kwargs):
        inner = original(*args, **kwargs)
        group = inner.metadata.row_group(0)
        meta = SimpleNamespace(num_rows=2, num_row_groups=1, num_columns=2,
            row_group=lambda _: SimpleNamespace(num_rows=10**12, num_columns=group.num_columns, column=group.column))
        return SimpleNamespace(metadata=meta, read=lambda **_: pytest.fail('unbudgeted Arrow read'))

    monkeypatch.setattr(pq, 'ParquetFile', fake)
    r = check_language_vocabulary(checkpoint, instruction='ok', catalog_directory=str(catalog))
    assert r['status'] == 'unknown' and 'allocation budget' in r['findings'][0]['message']


def test_command_selectors_and_literal_eval_inputs_are_never_executed(tmp_path):
    assert instruction_inputs(launch_command='lerobot-rollout --task="CloseFridge"') == ['CloseFridge']
    assert instruction_inputs(launch_command='lerobot-record --dataset.single_task "close fridge"') == ['close fridge']
    checkpoint, catalog = fixture(tmp_path)
    for command in ('lerobot-eval --env.task=CloseFridge', 'echo a; lerobot-rollout --task=CloseFridge',
                    'lerobot-rollout --task=$(touch /tmp/forbidden)', 'lerobot-rollout --task=A --task=B'):
        r = check_language_vocabulary(checkpoint, launch_command=command, catalog_directory=str(catalog))
        assert r['status'] == 'unknown' and r['verdict'] == 'UNKNOWN'
    path = tmp_path / 'eval.py'
    path.write_text('TASK_INSTRUCTIONS = ["CloseFridge"]\n')
    assert instruction_inputs(instruction_file=str(path)) == ['CloseFridge']
    path.write_text('raise RuntimeError("must not execute")\nTASK_INSTRUCTIONS = ["CloseFridge"]\n')
    assert check_language_vocabulary(checkpoint, instruction_file=str(path), catalog_directory=str(catalog))['status'] == 'unknown'
    path.write_text('TASK_INSTRUCTIONS = make_instructions()\n')
    assert check_language_vocabulary(checkpoint, instruction_file=str(path), catalog_directory=str(catalog))['status'] == 'unknown'


def test_additive_results_preserve_independent_errors_and_unknowns(tmp_path):
    checkpoint, catalog = fixture(tmp_path)
    language = check_language_vocabulary(checkpoint, instruction='close fridge doors', catalog_directory=str(catalog))
    original = {'status': 'error', 'findings': [{'code': 'existing', 'status': 'error'}], 'counts': {'error': 1}}
    before = json.dumps(original, sort_keys=True)
    r = augment_report(original, language)
    assert r['status'] == 'error' and r['findings'][0] == original['findings'][0]
    assert json.dumps(original, sort_keys=True) == before
    assert augment_report({'status': 'unknown', 'findings': []}, language)['status'] == 'unknown'


def test_remote_saved_revision_and_null_revision_remain_explicit(tmp_path, monkeypatch):
    checkpoint, _ = fixture(tmp_path, revision='2' * 40)
    calls, payload = [], b'{"task_index":0,"task":"CloseFridge"}\n'

    def request(url, *, limit, required_sha=None):
        calls.append((url, required_sha))
        if '/api/' in url:
            return json.dumps({'id': 'org/robot', 'private': False, 'sha': '2' * 40,
                'siblings': [{'rfilename': 'meta/tasks.jsonl', 'size': len(payload),
                             'blobId': hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()}]}).encode()
        return payload

    monkeypatch.setattr(hub, '_request', request)
    r = check_language_vocabulary(checkpoint, instruction='different', fetch=True)
    assert r['status'] == 'warning' and r['catalog']['saved_revision'] == '2' * 40
    assert all('/' + '2' * 40 in url and pin == '2' * 40 for url, pin in calls if '/resolve/' in url)
    (checkpoint / 'train_config.json').write_text(json.dumps({'dataset': {'repo_id': 'org/robot', 'revision': None}}))
    r = check_language_vocabulary(checkpoint, instruction='CloseFridge', fetch=True)
    assert r['verdict'] == 'IN_CATALOG' and r['catalog']['resolved_revision'] == '2' * 40
    assert 'current revision ' + '2' * 40 in r['catalog']['note']
    assert any(f['code'] == 'language.catalog_provenance_unknown' for f in r['findings'])


def test_cli_default_mcp_gates_and_native_unknown_scope_agree(tmp_path, capsys):
    checkpoint, catalog = fixture(tmp_path)
    args = ['check', str(checkpoint), '--instruction', 'Close the fridge doors.', '--language-catalog', str(catalog)]
    assert product_cli.main([*args, '--fail-on', 'warning', '--json']) == 1
    cli = json.loads(capsys.readouterr().out)
    assert cli['language_vocabulary']['verdict'] == 'OUT_OF_CATALOG'
    assert product_cli.main([*args, '--fail-on', 'error', '--json']) == 0
    capsys.readouterr()

    async def run():
        params = StdioServerParameters(command=sys.executable,
            args=['-B', '-m', 'roboclue', 'mcp', 'serve', '--root', str(tmp_path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src')))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                reply = await client.call_tool('preflight_check', {'source': str(checkpoint),
                    'instruction': 'Close the fridge doors.', 'language_catalog': str(catalog)})
                assert not reply.isError
                body = reply.structuredContent
                assert body['language_vocabulary']['verdict'] == 'OUT_OF_CATALOG'
                assert any(f['code'] == 'language.out_of_catalog' and f['status'] == 'warning' for f in body['findings_shown'])
                assert body['language_vocabulary']['native_token_path_equivalence'] == 'unknown'
                assert len(json.dumps(body).encode()) <= 8192
                exact = await client.call_tool('preflight_check', {'source': str(checkpoint),
                    'instruction': 'CloseFridge', 'language_catalog': str(catalog)})
                assert not exact.isError
                assert exact.content[0].text.startswith('Instruction matches linked catalog; training revision is unpinned.')
                assert exact.structuredContent['language_vocabulary']['verdict'] == 'IN_CATALOG'
                assert len(exact.content[0].text.encode()) <= 8192

    asyncio.run(run())


def test_readable_exact_control_leads_with_known_match_and_provenance(tmp_path, capsys):
    checkpoint, catalog = fixture(tmp_path)
    assert product_cli.main(['check', str(checkpoint), '--instruction', 'CloseFridge',
                             '--language-catalog', str(catalog)]) == 3
    text = capsys.readouterr().out
    assert text.startswith('Verdict: Instruction matches linked catalog; training revision is unpinned.')
    assert 'UNKNOWN' in text and 'unverified' in text


def test_language_selectors_do_not_demote_real_checkpoint_error_gate(tmp_path, monkeypatch, capsys):
    checkpoint, catalog = fixture(tmp_path)
    original = {'kind': 'checkpoint_check', 'status': 'error', 'counts': {'error': 1},
                'findings': [{'code': 'existing.error', 'status': 'error', 'message': 'independent error'}]}
    monkeypatch.setattr(hub, 'check_checkpoint', lambda *a, **kw: original)
    assert product_cli.main(['check', str(checkpoint), '--instruction', 'CloseFridge',
        '--language-catalog', str(catalog), '--fail-on', 'error', '--json']) == 1
    r = json.loads(capsys.readouterr().out)
    assert r['findings'][0] == original['findings'][0] and r['status'] == 'error'
    assert r['counts']['error'] == 1 and r['counts']['unknown'] >= 1
    assert original['counts'] == {'error': 1}


def test_mcp_authorizes_explicit_catalog_before_remote_checkpoint_request(tmp_path, monkeypatch):
    from policy_doctor.artifacts import InputError
    from policy_doctor.product_mcp import create_product_server

    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    monkeypatch.setattr(hub, '_request', lambda *a, **kw: pytest.fail('network before authorization'))
    server = create_product_server([str(allowed)], allow_hub=True, allow_hub_datasets=True)
    tool = next(t for t in server._tool_manager.list_tools() if t.name == 'preflight_check')
    with pytest.raises(InputError, match='outside'):
        tool.fn(source='hf://org/model', instruction='text', language_catalog=str(outside))


def test_mcp_requires_model_and_linked_catalog_permissions_independently(tmp_path, monkeypatch):
    from policy_doctor.artifacts import InputError
    from policy_doctor.product_mcp import create_product_server

    monkeypatch.setattr(hub, '_request', lambda *a, **kw: pytest.fail('network without model permission'))
    server = create_product_server([str(tmp_path)], allow_hub_datasets=True)
    tool = next(t for t in server._tool_manager.list_tools() if t.name == 'preflight_check')
    with pytest.raises(InputError, match='allow-hub'):
        tool.fn(source='hf://org/model', instruction='text')


def test_instruction_file_duplicate_keys_and_mutating_python_remain_unknown(tmp_path):
    checkpoint, catalog = fixture(tmp_path)
    file = tmp_path / 'instructions.json'
    file.write_text('{"instructions": ["CloseFridge"], "instructions": ["OpenDrawer"]}')
    assert check_language_vocabulary(checkpoint, instruction_file=str(file), catalog_directory=str(catalog))['status'] == 'unknown'
    script = tmp_path / 'eval.py'
    script.write_text('TASK_INSTRUCTIONS = ["CloseFridge"]\nTASK_INSTRUCTIONS.append("different task")\n')
    r = check_language_vocabulary(checkpoint, instruction_file=str(script), catalog_directory=str(catalog))
    assert r['verdict'] == 'UNKNOWN' and r['status'] == 'unknown'
