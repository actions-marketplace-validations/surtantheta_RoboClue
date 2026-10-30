"""Synthetic successor controls. No publisher/public-incident entry qualification."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_provenance_register import control as control
from test_provenance_register import dump, synthetic_entry

from policy_doctor.register.commands import select_command
from policy_doctor.register.engine import assess_register, concise_register_summary
from policy_doctor.register.schema import validate_entry, version

# Explicitly export the shared fixture for pytest and supported Ruff versions.
__all__ = ['control']


def recipe(control, *, status='published'):
    entry = control['entry']
    entry['kind'] = 'recipe'
    entry['evidence'][0]['basis'] = 'publisher_recipe'
    entry['title'] = 'UNSAFE authored crash claim must never appear in output'
    entry['first_fix'] = 'UNSAFE authored fix must never appear'
    entry['recipe_field'] = {'field': 'mode', 'checkpoint': 'synthetic/checkpoint', 'status': status}
    entry['applies_when'].append({'checkpoint_id_in': ['synthetic/checkpoint']})
    if status == 'published':
        entry['recipe_field'].update(published_value='published', source_index=0)
        entry['violation_when'] = [{'config_key': {'path': 'mode', 'op': 'ne', 'value': 'published'}}]
    else:
        entry['recipe_field']['reason'] = 'The pinned publishers disagree.' if status == 'conflict' else 'The value was not published.'
        entry['violation_when'] = []
        if status == 'conflict':
            entry['evidence'].append(copy.deepcopy(entry['evidence'][0]))
            entry['evidence'][1]['source']['line'] = 8
            entry['recipe_field']['source_indices'] = [0, 1]
    control['manifest'].update(stacks=['synthetic'], checkpoint_id='synthetic/checkpoint')
    dump(control['path'], control['manifest'])
    dump(control['entry_path'], entry)
    return entry


def result(control):
    return assess_register(control['path'])


def test_publisher_recipe_fixed_warning_two_healthy_controls_and_unknown(control):
    entry = recipe(control)
    report = result(control)
    assert report['status'] == 'warning'  # Requested ERROR cannot evade the owner cap.
    finding = report['findings'][0]
    assert finding['message'].startswith('mode = "bad" differs from the published evaluation recipe for synthetic/checkpoint ("published", ')
    assert finding['message'].endswith('Results are not comparable to the published numbers.')
    assert finding['first_fix'] == 'set mode to "published" to reproduce the published setup'
    assert 'UNSAFE' not in json.dumps(report)
    dump(control['config'], {'mode': 'published'})
    assert result(control)['status'] == 'pass'  # Corrected setup.
    control['manifest']['stacks'] = ['another-stack']
    dump(control['path'], control['manifest'])
    dump(control['config'], {'mode': 'bad'})
    assert result(control)['entries'][0]['status'] == 'not_applicable'  # Intentional other-stack setup.
    control['manifest']['stacks'] = ['synthetic']
    dump(control['path'], control['manifest'])
    dump(control['config'], {})
    assert result(control)['status'] == 'unknown'
    assert 'no recipe difference was established' in result(control)['findings'][0]['message']
    assert entry['severity'] == 'ERROR'


@pytest.mark.parametrize('status', ['unknown', 'conflict'])
def test_unresolved_publication_fields_full_only_and_real_findings_survive(control, status):
    recipe(control, status=status)
    report = result(control)
    assert report['entries'][0]['status'] == status
    assert not report['findings'] and report['status'] == 'unknown'
    preview = concise_register_summary(report)
    assert 'recipe_field' not in json.dumps(preview)
    real = synthetic_entry()
    real['id'] = 'SYNTHETIC-REAL-CONTRADICTION'
    dump(control['entries'] / 'REAL.json', real)
    report = result(control)
    assert report['status'] == 'error'
    assert len(report['findings']) == 1 and report['findings'][0]['code'].endswith(real['id'])


def test_mixed_recipe_evidence_still_capped_and_selected_source_is_exact(control):
    entry = recipe(control)
    entry['evidence'].insert(0, copy.deepcopy(entry['evidence'][0]))
    entry['evidence'][0]['basis'] = 'user_report'
    entry['evidence'][0]['source']['line'] = 99
    entry['recipe_field']['source_index'] = 1
    dump(control['entry_path'], entry)
    report = result(control)
    assert report['status'] == 'warning'
    assert report['findings'][0]['source_evidence'][0]['source']['line'] == 7
    assert ':7).' in report['findings'][0]['message']


@pytest.mark.parametrize('bad', [None, [], {}, True, -1, 100, '0'])
def test_bad_recipe_source_index_is_structured_unknown(control, bad):
    entry = recipe(control)
    entry['recipe_field']['source_index'] = bad
    dump(control['entry_path'], entry)
    assert result(control)['status'] == 'unknown' and result(control)['catalog_errors']


@pytest.mark.parametrize('change', ['wrong_field', 'wrong_value', 'wrong_operator', 'no_basis', 'contract_publisher', 'duplicate_conflict'])
def test_recipe_schema_rejects_unbound_or_invented_comparisons(control, change):
    entry = recipe(control, status='conflict' if change == 'duplicate_conflict' else 'published')
    if change == 'wrong_field':
        entry['recipe_field']['field'] = 'other'
    elif change == 'wrong_value':
        entry['recipe_field']['published_value'] = 1
    elif change == 'wrong_operator':
        entry['violation_when'][0]['config_key']['op'] = 'eq'
    elif change == 'no_basis':
        entry['evidence'][0].pop('basis')
    elif change == 'contract_publisher':
        entry['kind'] = 'contract'
        entry.pop('recipe_field')
    else:
        entry['recipe_field']['source_indices'] = [0, 0]
    dump(control['entry_path'], entry)
    assert result(control)['status'] == 'unknown' and result(control)['catalog_errors']


def test_complete_stack_declaration_required_for_recipe(control):
    recipe(control)
    control['manifest'].pop('stacks')
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'
    for value in [None, True, 'synthetic', ['synthetic', 'synthetic'], ['Synthetic'], ['synthetic ']]:
        control['manifest']['stacks'] = value
        dump(control['path'], control['manifest'])
        assert result(control)['status'] == 'unknown'
    control['manifest']['stacks'] = []
    dump(control['path'], control['manifest'])
    assert result(control)['entries'][0]['status'] == 'not_applicable'


@pytest.mark.parametrize('command,target', [
    ('python scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('python3.12 -u -m synthetic.train --mode=bad', 'module:synthetic.train'),
    ('torchrun --standalone --nproc_per_node=2 scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('torchrun --nnodes 2 -m synthetic.train --mode bad', 'module:synthetic.train'),
    ('python -m torch.distributed.run --nproc-per-node 2 scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('uv run --frozen python -m synthetic.train --mode bad', 'module:synthetic.train'),
    ('uv run --project repo torchrun --nproc_per_node=2 scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('uv run lerobot-eval --mode bad', 'entrypoint:lerobot-eval'),
    ('source activate;python scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('prep --mode wrong&&python scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('prep --mode wrong || python scripts/train.py --mode bad', 'script:scripts/train.py'),
    ('MODE=unused env OTHER=value python scripts/train.py --mode bad', 'script:scripts/train.py'),
])
def test_exact_target_common_launchers_and_separator_selection(command, target):
    selected = select_command(command, [target])
    assert selected['state'] == 'selected', selected
    assert selected['selector'] == target and selected['flags']['--mode'] == ['bad']


@pytest.mark.parametrize('command', [
    'python scripts/train.py --mode "a;b"', "python scripts/train.py --mode 'a;b'",
    'python scripts/train.py --mode a\\;b',
])
def test_quoted_and_escaped_semicolons_are_value_content(command):
    selected = select_command(command, ['script:scripts/train.py'])
    assert selected['state'] == 'selected' and selected['flags']['--mode'] == ['a;b']


@pytest.mark.parametrize('command', [None, '', 'python -c "anything"',
    'python scripts/train.py; python scripts/train.py', 'python scripts/train.py --mode $(anything)',
    'python scripts/train.py --mode "$VALUE"', 'python scripts/train.py | other',
    'python scripts/train.py > result.txt', 'if true;then python scripts/train.py;fi',
    'python scripts/train.py &', 'python scripts/train.py --mode "unterminated',
    'uv run --unsupported python scripts/train.py', 'torchrun --unsupported scripts/train.py',
    'time python scripts/train.py', '! python scripts/train.py', 'wrapper scripts/train.py',
    'python scripts/train.py --file *.json', 'python ~/scripts/train.py'])
def test_ambiguous_or_unsupported_commands_stay_unknown(command):
    assert select_command(command, ['script:scripts/train.py'])['state'] == 'unknown'


def test_no_merging_flags_or_first_arbitrary_script_guess(control):
    entry = control['entry']
    entry['launch_target'] = ['script:scripts/train.py']
    entry['violation_when'] = [{'launch_flag': {'flag': '--mode', 'op': 'eq', 'value': 'bad'}}]
    control['manifest'].update(stacks=['synthetic'], launch_command='other --mode bad; python scripts/train.py --mode good')
    dump(control['entry_path'], entry)
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'pass'
    control['manifest']['launch_command'] = 'python other.py --mode bad'
    dump(control['path'], control['manifest'])
    assert result(control)['entries'][0]['status'] == 'not_applicable'
    assert select_command('first --mode bad;second --mode good')['state'] == 'unknown'


def source_default(control, *, launch=False):
    source = control['path'].parent / 'source'
    source.mkdir()
    file = source / 'defaults.py'
    file.write_text('use_master = False\n')
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    pin = {'repo': 'example/synthetic', 'sha': 'a' * 40, 'path': 'defaults.py', 'line': 1}
    control['entry']['claim_basis'] = 'source_bytes'
    control['entry']['applies_when'].append({'source_files': {'files': [{'path': 'defaults.py', 'sha256': digest}]}})
    body = {'op': 'eq', 'value': False, 'default_when_absent': {'value': False, 'source': pin, 'sha256': digest}}
    if launch:
        control['entry']['launch_target'] = ['script:scripts/train.py']
        scripts = source / 'scripts'
        scripts.mkdir()
        script = scripts / 'train.py'
        script.write_text('from defaults import use_master\n')
        control['entry']['applies_when'][-1]['source_files']['files'].append({
            'path': 'scripts/train.py', 'sha256': hashlib.sha256(script.read_bytes()).hexdigest()})
        control['manifest']['command_directory'] = 'source'
        body['flag'] = '--use_master'
    else:
        body['path'] = 'use_master'
    control['entry']['violation_when'] = [{'launch_flag' if launch else 'config_key': body}]
    control['manifest'].update(stacks=['synthetic'], runtime_source_root='source')
    dump(control['config'], {})
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    return file, body


def test_absent_config_default_requires_matching_source_and_actual_context(control):
    file, body = source_default(control)
    report = result(control)
    assert report['status'] == 'error'
    witness = report['entries'][0]['violation'][0]
    assert witness['observed'] is False and witness['value_origin'].startswith('source_backed')
    assert witness['default_source']['line'] == 1
    dump(control['config'], {'use_master': True})
    assert result(control)['status'] == 'pass'
    dump(control['config'], {'use_master': None})
    assert result(control)['status'] == 'pass'  # Present null is unequal to False, never defaulted.
    assert 'default_source' not in result(control)['entries'][0]['violation'][0]
    dump(control['config'], {})
    file.write_text('use_master = True\n')
    assert result(control)['status'] == 'unknown'
    file.write_text('use_master = False\n')
    control['manifest'].pop('config')
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'


def test_launch_absence_exists_false_is_not_null_or_no_command(control):
    _, body = source_default(control, launch=True)
    assert result(control)['status'] == 'unknown'  # Missing command cannot provide defaults.
    control['manifest']['launch_command'] = 'python scripts/train.py'
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'error'
    body.pop('default_when_absent')
    body['op'] = 'exists'
    body.pop('value')
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'pass'
    body.update(op='eq', value=None)
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'unknown'
    body['default_when_absent'] = {'value': None, 'source': {'repo': 'example/synthetic', 'sha': 'a' * 40, 'path': 'defaults.py', 'line': 1},
                                   'sha256': hashlib.sha256((control['path'].parent / 'source/defaults.py').read_bytes()).hexdigest()}
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'error'  # Pinned null is distinct from absent/unknown.


def test_default_cannot_come_from_matching_source_elsewhere_or_unresolved_module(control):
    source_default(control, launch=True)
    control['manifest']['launch_command'] = 'python scripts/train.py'
    directory = control['manifest'].pop('command_directory')
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'
    control['manifest']['command_directory'] = directory
    control['entry']['launch_target'].append('module:synthetic.train')
    control['manifest']['launch_command'] = 'python -m synthetic.train'
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'
    other = control['path'].parent / 'elsewhere'
    (other / 'scripts').mkdir(parents=True)
    (other / 'scripts/train.py').write_text('from defaults import use_master\n')
    control['manifest'].update(command_directory='elsewhere', launch_command='python scripts/train.py')
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'


@pytest.mark.parametrize('op,actual,expected', [('lt', 5e-6, 'error'), ('lt', 2 ** -15, 'pass'),
    ('lte', 2 ** -15, 'error'), ('gt', 1, 'error'), ('gte', True, 'unknown')])
def test_numeric_config_bounds_are_finite_and_bool_is_not_number(control, op, actual, expected):
    control['entry']['violation_when'] = [{'config_key': {'path': 'lr', 'op': op, 'value': 2 ** -15}}]
    dump(control['entry_path'], control['entry'])
    dump(control['config'], {'lr': actual})
    assert result(control)['status'] == expected


@pytest.mark.parametrize('actual,op,value,expected', [(None, 'eq', 'text', 'pass'), (None, 'ne', 'text', 'error'),
    (False, 'eq', 0, 'pass'), (False, 'ne', 0, 'error'), (1, 'eq', True, 'pass'), ({}, 'eq', 'text', 'unknown')])
def test_present_scalar_null_and_type_identity_differ_from_missing(control, actual, op, value, expected):
    control['entry']['violation_when'] = [{'config_key': {'path': 'mode', 'op': op, 'value': value}}]
    dump(control['entry_path'], control['entry'])
    dump(control['config'], {'mode': actual})
    assert result(control)['status'] == expected
    dump(control['config'], {})
    assert result(control)['status'] == 'unknown'


def test_real_pep440_prereleases_post_dev_epoch_local_and_invalid(control):
    assert version('3.0.0b1') < version('3.0.0rc1') < version('3.0.0') < version('3.0.0.post1')
    assert version('3.0.0.dev1') < version('3.0.0b1')
    assert version('1!1') > version('999.0')
    assert version('1.2+cpu') > version('1.2')
    control['entry']['applies_when'][0]['package_version']['version'] = {'eq': '3.0.0b1'}
    dump(control['entry_path'], control['entry'])
    control['lock'].write_text('synthetic==3.0.0b1\n')
    assert result(control)['status'] == 'error'
    control['lock'].write_text('synthetic==3.0.0\n')
    assert result(control)['entries'][0]['status'] == 'not_applicable'
    control['lock'].write_text('synthetic==not-a-version\n')
    assert result(control)['status'] == 'unknown'


def test_unique_registry_processor_selector_and_duplicate_ambiguity(control):
    path = control['path'].parent / 'processor.json'
    step = {'registry_name': 'normalizer_processor', 'config': {'normalization_mapping': {'VISUAL': 'MEAN_STD'}}}
    dump(path, {'steps': [step]})
    control['manifest']['processor'] = path.name
    control['entry']['violation_when'] = [{'processor_key': {'path': 'normalizer_processor.config.normalization_mapping.VISUAL', 'op': 'eq', 'value': 'MEAN_STD'}}]
    dump(control['path'], control['manifest'])
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'error'
    assert result(control)['entries'][0]['violation'][0]['processor_registry_index'] == 0
    for document in [{'steps': [step, step]}, {'steps': []}, {'steps': [False]}, {'steps': [step], 'normalizer_processor': {}}]:
        dump(path, document)
        assert result(control)['status'] == 'unknown'
    dump(path, {'steps': [dict(step, config={'normalization_mapping': {'VISUAL': 'IDENTITY'}})]})
    assert result(control)['status'] == 'pass'


def test_missing_parent_is_not_absent_final_key_and_profile_binding_is_required(control):
    _, body = source_default(control)
    body['path'] = 'nested.use_master'
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'unknown'
    dump(control['config'], {'nested': {}})
    assert result(control)['status'] == 'error'
    dump(control['config'], {'nested': False})
    assert result(control)['status'] == 'unknown'
    control['entry']['applies_when'].pop()
    with pytest.raises(ValueError, match='source_files'):
        validate_entry(control['entry'])
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'unknown'


@pytest.mark.parametrize('field', ['field', 'checkpoint', 'status', 'published_value', 'source_index'])
@pytest.mark.parametrize('bad', [None, {}, [], float('nan'), 'x' * 17000])
def test_recipe_metadata_malformed_fields_remain_structured(control, field, bad):
    entry = recipe(control)
    entry['recipe_field'][field] = bad
    dump(control['entry_path'], entry)
    report = result(control)
    # Null and containers can be legitimate JSON values; the schema still
    # rejects an unbound published_value that disagrees with its predicate.
    assert isinstance(report, dict) and report['status'] == 'unknown'
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('field', ['value', 'source', 'sha256'])
@pytest.mark.parametrize('bad', [None, {}, [], float('inf'), 'x' * 17000])
def test_default_metadata_malformed_fields_remain_structured(control, field, bad):
    _, body = source_default(control)
    body['default_when_absent'][field] = bad
    dump(control['entry_path'], control['entry'])
    report = result(control)
    # A null/container default is a real typed literal, not permission to
    # coerce it to the expected False. Invalid source metadata is UNKNOWN.
    assert isinstance(report, dict) and report['status'] in ('unknown', 'pass')
    json.dumps(report, allow_nan=False)


def test_source_default_is_version_specific_even_with_applicability_alternatives(control):
    file, body = source_default(control)
    before = hashlib.sha256(file.read_bytes()).hexdigest()
    file.write_text('use_master = True\n')
    after = hashlib.sha256(file.read_bytes()).hexdigest()
    profile = control['entry']['applies_when'][-1]['source_files']['files'][0]
    profile.pop('sha256')
    profile['sha256_in'] = [before, after]
    dump(control['entry_path'], control['entry'])
    assert result(control)['status'] == 'unknown'  # The default is pinned only to the first blob.
    file.write_text('use_master = False\n')
    control['manifest'].pop('stacks')
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'  # No selected stack, no inferred default.


def test_raw_flag_text_is_not_silently_converted_to_native_boolean(control):
    control['entry']['launch_target'] = ['entrypoint:synthetic-eval']
    control['entry']['violation_when'] = [{'launch_flag': {'flag': '--enabled', 'op': 'eq', 'value': True}}]
    control['manifest'].update(stacks=['synthetic'], launch_command='synthetic-eval --enabled True')
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'pass'
    control['manifest']['launch_command'] = 'synthetic-eval --enabled'
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'error'


def test_recipe_cli_default_mcp_gates_and_chooser_agree(control, capsys):
    import asyncio

    from policy_doctor.product_cli import main
    from policy_doctor.product_mcp import create_product_server
    from policy_doctor.workflow_readiness import choose_workflow

    recipe(control)
    assert main(['check', str(control['path']), '--json', '--fail-on', 'error']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'warning'
    assert main(['check', str(control['path']), '--json', '--fail-on', 'warning']) == 1
    capsys.readouterr()

    async def run():
        server = create_product_server([str(control['path'].parent)])
        reply = await server.call_tool('preflight_check', {'source': str(control['path'])})
        report = reply.structuredContent
        assert report['status'] == 'warning'
        assert report['findings'][0]['first_fix'] == 'set mode to "published" to reproduce the published setup'
    asyncio.run(run())
    plan = choose_workflow('upgrade', manifest=str(control['path']))
    assert plan['can_do_now'][0]['mcp_request']['name'] == 'preflight_check'


def test_recipe_requires_explicit_checkpoint_identity_and_names_actual_selected_id(control):
    entry = recipe(control)
    control['manifest'].pop('checkpoint_id')
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'unknown'
    control['manifest']['checkpoint_id'] = 'another/checkpoint'
    dump(control['path'], control['manifest'])
    assert result(control)['entries'][0]['status'] == 'unknown'
    entry['applies_when'][-1]['checkpoint_id_in'].append('another/checkpoint')
    dump(control['entry_path'], entry)
    assert 'for another/checkpoint ' in result(control)['findings'][0]['message']
    entry['recipe_field']['checkpoint'] = 'unbound/reference'
    dump(control['entry_path'], entry)
    assert result(control)['status'] == 'unknown' and result(control)['catalog_errors']


@pytest.mark.parametrize('command', [
    'uv run python scripts/train.py',
    'uv run --directory nested python scripts/train.py',
    'cd nested; python scripts/train.py',
    'source activate; python scripts/train.py',
    'env PYTHONPATH=elsewhere python scripts/train.py',
    'PYTHONPATH=elsewhere python scripts/train.py',
])
def test_launch_default_does_not_guess_preparation_cwd_or_wrapper_environment(control, command):
    source_default(control, launch=True)
    control['manifest']['launch_command'] = command
    dump(control['path'], control['manifest'])
    report = result(control)
    assert report['status'] == 'unknown'
    assert report['entries'][0]['scope_gate']['command']['state'] == 'selected'
    assert report['entries'][0]['scope_gate']['command']['source_binding_safe'] is False
    assert 'not source-bound' in report['entries'][0]['violation'][0]['reason']


def test_direct_source_bound_default_and_quoted_semicolon_remain_supported(control):
    source_default(control, launch=True)
    control['manifest']['launch_command'] = 'python scripts/train.py --label "literal;content"'
    dump(control['path'], control['manifest'])
    assert result(control)['status'] == 'error'
    assert result(control)['entries'][0]['scope_gate']['command']['source_binding_safe'] is True


@pytest.mark.parametrize('field', ['config', 'processor', 'metadata_paths', 'lockfiles', 'runtime_source_root', 'command_directory'])
def test_other_user_home_alias_in_nested_paths_is_structured_unknown(control, field):
    alias = '~roboclue_absent_user_20261029_abcdef0123456789/not.json'
    if field in ('runtime_source_root', 'command_directory'):
        source_default(control, launch=field == 'command_directory')
        if field == 'command_directory':
            control['manifest']['launch_command'] = 'python scripts/train.py'
    if field == 'processor':
        control['entry']['violation_when'] = [{'processor_key': {'path': 'mode', 'op': 'eq', 'value': 'bad'}}]
        dump(control['entry_path'], control['entry'])
    control['manifest'][field] = [alias] if field in ('metadata_paths', 'lockfiles') else alias
    dump(control['path'], control['manifest'])
    report = result(control)
    assert report['status'] == 'unknown'
    json.dumps(report, allow_nan=False)


def test_unresolved_home_in_api_source_or_roots_and_expander_runtime_error_are_bounded(control, monkeypatch):
    from pathlib import Path

    from policy_doctor.register.engine import _expanded

    alias = '~roboclue_absent_user_20261029_abcdef0123456789/input.json'
    assert assess_register(alias)['status'] == 'unknown'
    assert assess_register(control['path'], read_roots=[alias])['status'] == 'unknown'

    def unresolved_home(self):
        raise RuntimeError('controlled unresolved home directory')

    monkeypatch.setattr(Path, 'expanduser', unresolved_home)
    with pytest.raises(ValueError, match='home directory'):
        _expanded(Path('/a/direct/path'))
    assert assess_register(control['path'])['status'] == 'unknown'


def test_nested_unresolved_home_cli_mcp_error_gate_agrees(control, capsys):
    import asyncio

    from policy_doctor.product_cli import main
    from policy_doctor.product_mcp import create_product_server

    control['manifest']['config'] = '~roboclue_absent_user_20261029_abcdef0123456789/config.json'
    dump(control['path'], control['manifest'])
    assert main(['check', str(control['path']), '--json', '--fail-on', 'error']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'

    async def run():
        server = create_product_server([str(control['path'].parent)])
        reply = await server.call_tool('preflight_check', {'source': str(control['path'])})
        assert not reply.isError and reply.structuredContent['status'] == 'unknown'
    asyncio.run(run())


def test_zero_occurrence_is_a_weak_absence_hint_never_source_error(control):
    source = control['path'].parent / 'source'
    source.mkdir()
    file = source / 'module.py'
    file.write_text('# the target literal is absent\n')
    control['entry']['claim_basis'] = 'source_bytes'
    control['entry']['applies_when'].append({'source_contains': {'path': 'module.py', 'literal': 'use_quantile_norm=True', 'occurrences': {'eq': 0}}})
    control['manifest']['runtime_source_root'] = 'source'
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    report = result(control)
    assert report['status'] == 'warning'
    assert report['entries'][0]['applicability'][1]['observed_occurrences'] == 0
    file.write_text('use_quantile_norm=True\n')
    assert result(control)['entries'][0]['status'] == 'not_applicable'
    file.unlink()
    assert result(control)['status'] == 'unknown'
    control['entry']['applies_when'][-1]['source_contains']['occurrences'] = {'gte': 0}
    dump(control['entry_path'], control['entry'])
    assert result(control)['catalog_errors'] and result(control)['status'] == 'unknown'
