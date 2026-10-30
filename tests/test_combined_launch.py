"""Combined front-door controls; synthetic entries do not qualify public facts."""
import asyncio
import hashlib
import json

import pytest
from test_fastwam_contract import inputs as inputs
from test_provenance_register import control as control
from test_provenance_register import dump, synthetic_entry
from test_register_recipes import recipe

from policy_doctor import product_cli
from policy_doctor.product_mcp import BASIC_TOOLS, create_product_server
from policy_doctor.register.commands import select_command
from policy_doctor.register.engine import _same, assess_register, concise_register_summary
from policy_doctor.register.schema import validate_entry
from policy_doctor.rule_catalog import rule_catalog
from policy_doctor.server import _encoded_json_size

# Explicitly export shared fixtures for pytest and all supported Ruff versions.
__all__ = ['control', 'inputs']


def unpack(value):
    return ((value.structuredContent, '\n'.join(item.text for item in value.content if hasattr(item, 'text')))
            if hasattr(value, 'structuredContent') else (value[1], '\n'.join(item.text for item in value[0] if hasattr(item, 'text'))))


def test_recipe_full_wording_fix_pin_and_decisive_witness_survive_both_frontdoors(control, capsys):
    entry = recipe(control)
    published = 'published-' + 'x' * 240
    entry['recipe_field']['published_value'] = published
    entry['violation_when'][0]['config_key']['value'] = published
    entry['applies_when'] += [{'config_key': {'path': 'mode', 'op': 'exists'}} for _ in range(6)]
    dump(control['entry_path'], entry)
    report = assess_register(control['path'])
    fix = report['findings'][0]['first_fix']
    assert len(report['findings'][0]['message']) > 400
    assert product_cli.main(['check', str(control['path']), '--fail-on', 'error']) == 0
    text = capsys.readouterr().out
    assert 'Results are not comparable to the published numbers.' in text and fix in text
    assert 'example.py:7' in text and 'a' * 40 not in text

    async def run():
        server = create_product_server([str(control['path'].parent)])
        assert {tool.name for tool in server._tool_manager.list_tools()} == BASIC_TOOLS
        assert len(BASIC_TOOLS) == 8
        tool = server._tool_manager.get_tool('preflight_check')
        assert 'launch_command' in tool.parameters['properties'] and 'language_command' not in tool.parameters['properties']
        result, text = unpack(await server.call_tool('preflight_check', {'source': str(control['path'])}))
        finding = result['findings'][0]
        assert result['status'] == 'warning' and finding['first_fix'] == fix
        assert 'Results are not comparable to the published numbers.' in text and fix in text
        assert finding['evidence'][0]['observed'] == 'bad'
        assert finding['evidence'][0]['predicate']['config_key']['op'] == 'ne'
        assert finding['source_evidence'][0]['source']['sha'] == 'a' * 40
    asyncio.run(run())


@pytest.mark.parametrize('exact', [False, True])
def test_comment_literal_cap_cannot_be_evaded_with_release_metadata_label(control, exact):
    root = control['path'].parent / 'source'
    root.mkdir()
    source = root / 'implementation.py'
    source.write_text('# dangerous_behavior is mentioned in a comment only\n')
    control['entry']['applies_when'].append({'source_contains': {'path': source.name, 'literal': 'dangerous_behavior'}})
    if exact:
        control['entry']['applies_when'].append({'source_files': {'files': [
            {'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}})
    control['manifest']['runtime_source_root'] = str(root)
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    assert assess_register(control['path'])['status'] == 'warning'


@pytest.mark.parametrize('identity,status', [('synthetic/checkpoint/', 'warning'),
    ('another/checkpoint', 'unknown'), ('SYNTHETIC/checkpoint', 'unknown'),
    ('/cache/synthetic/checkpoint', 'unknown')])
def test_unknown_checkpoint_identity_never_certifies_healthy(control, identity, status):
    recipe(control)
    control['manifest']['checkpoint_id'] = identity
    dump(control['path'], control['manifest'])
    assert assess_register(control['path'])['status'] == status


def test_unknown_next_requests_evidence_and_independent_error_stays_first(control):
    entry = recipe(control)
    entry['recipe_field']['field'] = 'missing'
    entry['violation_when'][0]['config_key']['path'] = 'missing'
    dump(control['entry_path'], entry)
    unknown = assess_register(control['path'])
    assert unknown['status'] == 'unknown'
    assert unknown['findings'][0]['first_fix'].startswith('Supply the actual missing')
    dump(control['entries'] / 'REAL.json', {**synthetic_entry(), 'id': 'SYNTHETIC-INDEPENDENT'})
    report = assess_register(control['path'])
    assert report['status'] == 'error'
    assert concise_register_summary(report)['findings'][0]['code'] == 'register.SYNTHETIC-INDEPENDENT'


def test_independent_error_does_not_hide_recipe_warning_in_default_preview(control):
    entry = recipe(control)
    dump(control['entries'] / 'REAL.json', {**synthetic_entry(), 'id': 'SYNTHETIC-INDEPENDENT'})
    report = assess_register(control['path'])
    assert report['status'] == 'error'
    preview = concise_register_summary(report)
    shown = [(row['code'], row['status']) for row in preview['findings']]
    assert shown[0] == ('register.SYNTHETIC-INDEPENDENT', 'error')
    assert ('register.' + entry['id'], 'warning') in shown
    caveat = next(row for row in preview['findings'] if row['code'] == 'register.' + entry['id'])
    assert caveat['message'].endswith('Results are not comparable to the published numbers.')
    assert preview['findings_omitted'] == len(report['findings']) - len(preview['findings'])
    assert len(json.dumps(preview).encode()) <= 8192


def test_other_actionable_rows_never_expand_the_preview_bound(control):
    recipe(control)
    for index in range(6):
        dump(control['entries'] / f'REAL{index}.json',
             {**synthetic_entry(), 'id': f'SYNTHETIC-INDEPENDENT-{index}-' + 'X' * 70,
              'title': 'Long independent title ' + 'z' * 2000})
    report = assess_register(control['path'])
    actionable = sum(row['status'] in ('error', 'warning') for row in report['findings'])
    assert actionable >= 7
    preview = concise_register_summary(report)
    assert len(json.dumps(preview).encode()) <= 8192
    assert len(preview['findings']) <= 4
    assert preview['findings_omitted'] + len(preview['findings']) == len(report['findings'])
    assert preview.get('other_actionable_omitted', 0) + len(preview['findings']) == actionable


def test_legacy_contract_also_requires_complete_stack_selection_and_unknown_requests_input(control):
    control['manifest'].pop('stacks')
    dump(control['path'], control['manifest'])
    report = assess_register(control['path'])
    assert report['status'] == 'unknown'
    finding = report['findings'][0]
    assert finding['first_fix'].startswith('Supply the missing or conflicting')
    assert control['entry']['first_fix'] not in finding['first_fix']


def test_unrelated_real_error_dominates_an_external_command_conflict(control):
    control['manifest']['launch_command'] = 'python train.py --mode good'
    dump(control['path'], control['manifest'])
    report = assess_register(control['path'], launch_command='python train.py --mode bad')
    assert report['status'] == 'error'
    assert concise_register_summary(report)['findings'][0]['code'] == 'register.SYNTHETIC-ENGINE-CONTROL'


def test_external_launch_conflicts_are_unknown_without_overwriting_manifest(control, capsys):
    control['entry']['violation_when'] = [{'launch_flag': {'flag': '--mode', 'op': 'eq', 'value': 'bad'}}]
    control['manifest']['launch_command'] = 'python train.py --mode good'
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    original = control['path'].read_bytes()
    assert product_cli.main(['check', str(control['path']), '--launch-command', 'python train.py --mode bad', '--json']) == 3
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'unknown' and report['observation_conflicts'][0]['field'] == 'launch_command'
    assert control['path'].read_bytes() == original

    async def run():
        server = create_product_server([str(control['path'].parent)])
        result, _ = unpack(await server.call_tool('preflight_check', {'source': str(control['path']),
            'launch_command': 'python train.py --mode bad', 'response_format': 'detailed'}))
        assert result['status'] == 'unknown' and result['observation_conflicts'] == report['observation_conflicts']
    asyncio.run(run())


def test_supported_shell_selection_does_not_merge_commands_or_ignore_unknown_context():
    target = ['script:eval.py']
    selected = select_command('python other.py --mode bad; python eval.py --mode "good;still"', target)
    assert selected['state'] == 'selected' and selected['flags']['--mode'] == ['good;still']
    for command in ('python first.py --mode bad; python second.py', 'python eval.py --mode=$(pick)',
                    'python eval.py --mode good; python eval.py --mode bad', 'python eval.py --mode bad # later override'):
        assert select_command(command, target)['state'] == 'unknown'


@pytest.mark.parametrize('left,right', [([True], [1]), ({'x': [False]}, {'x': [0]}), (None, 'None')])
def test_nested_strict_json_equality(left, right):
    assert not _same(left, right)


@pytest.mark.parametrize('value', [None, 0, 1, 'false', [], {}])
def test_exists_expected_presence_requires_a_strict_boolean(control, value):
    control['entry']['applies_when'].append({'launch_flag': {'flag': '--pytorch_weight_path', 'op': 'exists', 'value': value}})
    with pytest.raises(ValueError, match='strict boolean'):
        validate_entry(control['entry'])


def test_cli_weight_override_absence_predicate_never_negates_unknown(control):
    entry = control['entry']
    entry['applies_when'].append({'launch_flag': {'flag': '--pytorch_weight_path', 'op': 'exists', 'value': False}})
    dump(control['entry_path'], entry)
    command = 'python scripts/train_pytorch.py --batch_size 16'
    control['manifest']['launch_command'] = command
    dump(control['path'], control['manifest'])
    positive = assess_register(control['path'])
    assert positive['status'] == 'error'
    assert positive['entries'][0]['applicability'][-1]['observed_presence'] is False
    control['manifest']['launch_command'] = command + ' --pytorch_weight_path=/owned/checkpoint'
    dump(control['path'], control['manifest'])
    healthy = assess_register(control['path'])
    assert healthy['entries'][0]['status'] == 'not_applicable'
    assert healthy['status'] == 'unknown' and healthy['assessment_state'] == 'no_applicable_entries'
    assert not healthy['findings']
    for selected in (None, 'python first.py; python scripts/train_pytorch.py'):
        if selected is None:
            control['manifest'].pop('launch_command')
        else:
            control['manifest']['launch_command'] = selected
        dump(control['path'], control['manifest'])
        unknown = assess_register(control['path'])
        assert unknown['status'] == 'unknown'
        assert unknown['entries'][0]['applicability'][-1]['result'] == 'unknown'


@pytest.mark.parametrize('document,state', [({'mode': 'bad', 'optional': {}}, 'error'),
    ({'mode': 'bad', 'optional': {'weight': None}}, 'unknown'), ({'mode': 'bad'}, 'unknown')])
def test_exists_false_config_requires_an_unambiguous_loaded_parent(control, document, state):
    control['entry']['applies_when'].append({'config_key': {'path': 'optional.weight', 'op': 'exists', 'value': False}})
    dump(control['entry_path'], control['entry'])
    dump(control['config'], document)
    assert assess_register(control['path'])['status'] == state


@pytest.mark.parametrize('actual,expected', [('bad' * 3000, 'published'), ([True] * 500, [1] * 500)])
def test_large_actual_values_keep_full_detail_and_bounded_typed_semantics(control, actual, expected):
    control['entry']['violation_when'] = [{'config_key': {'path': 'mode', 'op': 'ne', 'value': expected}}]
    dump(control['entry_path'], control['entry'])
    dump(control['config'], {'mode': actual})
    report = assess_register(control['path'])
    assert report['status'] == 'error'
    assert report['findings'][0]['evidence'][0]['observed'] == actual
    summary = concise_register_summary(report)
    witness = summary['findings'][0]['evidence'][0]
    assert witness['phase'] == 'violation' and witness['op'] == 'ne'
    assert witness['observed_type'] == ('string' if isinstance(actual, str) else 'array')
    assert witness['observed_truncated'] and 'observed' not in witness
    assert witness['observed_sha256'] and witness['witness_sha256'] and summary['full_report_sha256']
    assert summary['status'] == 'error' and summary['findings'][0]['first_fix'] == control['entry']['first_fix']
    assert len(json.dumps(summary).encode()) < 8192

    async def run():
        server = create_product_server([str(control['path'].parent)])
        result, text = unpack(await server.call_tool('preflight_check', {'source': str(control['path'])}))
        assert result['status'] == 'error' and result['findings'][0]['first_fix'] in text
        assert result['findings'][0]['evidence'][0]['observed_type'] == witness['observed_type']
        assert result['findings'][0]['evidence'][0]['observed_sha256'] == witness['observed_sha256']
    asyncio.run(run())


def test_long_selector_preserves_actual_typed_primitive_and_expected_op_phase(control):
    key = 'x' * 250
    control['entry']['violation_when'] = [{'config_key': {'path': key, 'op': 'eq', 'value': True}}]
    dump(control['entry_path'], control['entry'])
    dump(control['config'], {key: True})
    report = assess_register(control['path'])
    witness = concise_register_summary(report)['findings'][0]['evidence'][0]
    assert witness['observed'] is True and witness['expected'] is True
    assert witness['op'] == 'eq' and witness['phase'] == 'violation'
    assert witness['context_omitted'] and witness['witness_sha256']

    async def run():
        server = create_product_server([str(control['path'].parent)])
        result, _ = unpack(await server.call_tool('preflight_check', {'source': str(control['path'])}))
        actual = result['findings'][0]['evidence'][0]
        assert actual['observed'] is True and actual['expected'] is True
        assert actual['phase'] == 'violation' and actual['op'] == 'eq'
    asyncio.run(run())


def test_recipe_typed_published_value_binding_is_strict_inside_containers(control):
    entry = recipe(control)
    entry['recipe_field']['published_value'] = [1]
    entry['violation_when'][0]['config_key']['value'] = [True]
    with pytest.raises(ValueError, match='typed published_value'):
        validate_entry(entry)


def test_malformed_home_and_dynamic_catalog_frontdoors(capsys):
    assert product_cli.main(['check', '~roboclue-no-such-user-20261029/input.json']) == 2
    text = capsys.readouterr().err
    assert 'Traceback' not in text and 'home directory' in text
    catalog = rule_catalog()
    assert catalog['count'] == len(catalog['rules']) == 13
    assert [row['id'] for row in catalog['rules']] == [f'RC{number:03}' for number in range(1, 14)]


def test_streamed_report_size_remains_exact_after_crossing_a_summary_budget():
    value = {'long': 'x' * (9 * 1024 * 1024), 'later': [True, 1, None, {'unicode': '界' * 500}]}
    assert _encoded_json_size(value) == len(json.dumps(value, allow_nan=False).encode())


def test_checkpoint_image_and_language_observations_survive_cli_and_default_mcp(inputs, tmp_path, capsys):
    checkpoint, source, _ = inputs
    catalog = tmp_path / 'catalog'
    (catalog / 'meta').mkdir(parents=True)
    (catalog / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'CloseFridge'}) + '\n')
    dump(checkpoint / 'train_config.json', {'dataset': {'repo_id': 'synthetic/tasks', 'revision': None}})
    assert product_cli.main(['check', str(checkpoint), '--runtime-source-root', str(source),
        '--launch-command', 'python eval.py --task OpenDrawer', '--language-catalog', str(catalog)]) == 1
    text = capsys.readouterr().out
    assert 'image map ERROR' in text and 'Instruction catalog: UNKNOWN' in text

    async def run():
        server = create_product_server([str(tmp_path)])
        result, text = unpack(await server.call_tool('preflight_check', {'source': str(checkpoint),
            'runtime_source_root': str(source), 'launch_command': 'python eval.py --task OpenDrawer',
            'language_catalog': str(catalog)}))
        assert result['status'] == 'error' and result['image_contract']['status'] == 'error'
        # FastWAM's unsupported language profile stays explicit UNKNOWN; its
        # independent image contradiction must survive that missing evidence.
        assert result['language_vocabulary']['verdict'] == 'UNKNOWN'
        assert 'Instruction catalog: UNKNOWN' in text
        assert any(row['code'] == 'fastwam.image_map' and row['status'] == 'error' for row in result['findings'])
    asyncio.run(run())


def _hostile_independent_errors(control, errors, url_length):
    from policy_doctor.register.schema import validate_entry

    for path in control['entries'].glob('REAL*.json'):
        path.unlink()
    for index in range(errors):
        entry = {**synthetic_entry(), 'id': f'SYNTHETIC-INDEPENDENT-{index}', 'title': 'T' * 900, 'first_fix': 'F' * 2048}
        entry['applies_when'] = list(entry['applies_when']) + [
            {'config_key': {'path': 'mode', 'op': 'exists'}},
            {'config_key': {'path': 'mode', 'op': 'ne', 'value': 'zzz'}},
            {'config_key': {'path': 'mode', 'op': 'eq', 'value': 'bad'}}]
        evidence = dict(entry['evidence'][0], quote='Q' * 300, url='https://example.org/' + 'u' * url_length)
        evidence['source'] = dict(evidence['source'], repo='r' * 250, path='p' * 400 + '.py')
        entry['evidence'] = [evidence]
        validate_entry(entry)
        dump(control['entries'] / f'REAL{index}.json', entry)
    return assess_register(control['path'])


@pytest.mark.parametrize('errors', [1, 4])
def test_preview_bound_and_recipe_name_hold_for_maximum_schema_values(control, errors):
    # Found by independent review: counters added after the size check reached 8193-8223 bytes.
    entry = recipe(control)
    recipe_code = 'register.' + entry['id']
    for url_length in range(1, 2020, 37):
        report = _hostile_independent_errors(control, errors, url_length)
        preview = concise_register_summary(report)
        assert len(json.dumps(preview, allow_nan=False).encode()) <= 8192, url_length
        shown = [(row['code'], row['status']) for row in preview['findings']]
        assert shown[0][1] == 'error' and (recipe_code, 'warning') in shown, (url_length, shown)
        actionable = sum(row['status'] in ('error', 'warning') for row in report['findings'])
        assert preview.get('other_actionable_omitted', 0) + len(preview['findings']) == actionable
        assert preview['findings_omitted'] + len(preview['findings']) == len(report['findings'])


def test_text_channels_name_the_recipe_caveat_beside_an_independent_error(control, capsys):
    from policy_doctor.mcp_presentation import _readable
    from policy_doctor.register.engine import register_text

    entry = recipe(control)
    dump(control['entries'] / 'REAL.json', {**synthetic_entry(), 'id': 'SYNTHETIC-INDEPENDENT'})
    report = assess_register(control['path'])
    caveat = 'Also WARNING register.' + entry['id'] + ': A setting differs from the published evaluation recipe. ' \
             'Results are not comparable to the published numbers.'
    for text in (register_text(report), _readable(concise_register_summary(report))):
        assert text.splitlines()[1].startswith('Synthetic') and caveat in text
    assert product_cli.main(['check', str(control['path']), '--fail-on', 'error']) == 1
    assert caveat in capsys.readouterr().out
