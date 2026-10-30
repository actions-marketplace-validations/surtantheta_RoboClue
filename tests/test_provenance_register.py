"""Synthetic engine controls; these do not qualify any public incident entry."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from policy_doctor.product_cli import main
from policy_doctor.register import assess_register
from policy_doctor.register.engine import load_entries
from policy_doctor.register.schema import validate_entry, version
from policy_doctor.server import create_server
from policy_doctor.workflow_readiness import choose_workflow


def dump(path, document):
    path.write_text(json.dumps(document))
    return path


def synthetic_entry():
    return {'schema_version': 1, 'id': 'SYNTHETIC-ENGINE-CONTROL', 'kind': 'contract',
        'title': 'Synthetic settings disagree', 'stacks': ['synthetic'], 'claim_basis': 'release_metadata',
        'applies_when': [{'package_version': {'package': 'synthetic', 'version': {'gte': '1.2', 'lt': '2.0'}}}],
        'violation_when': [{'config_key': {'path': 'mode', 'op': 'eq', 'value': 'bad'}}],
        'severity': 'ERROR', 'evidence': [{'url': 'https://example.org/synthetic-report',
            'quote': 'Synthetic test fixture only, never a public incident claim.',
            'source': {'repo': 'example/synthetic', 'sha': 'a' * 40, 'path': 'example.py', 'line': 7}}],
        'healthy_controls': ['Corrected setting', 'Unaffected release'], 'first_fix': 'Use the intended synthetic setting.',
        'unknown_when': 'Read the missing synthetic evidence.', 'last_verified': '2026-10-29',
        'validation': {'status': 'confirmed', 'record': 'synthetic-test-only'}}


@pytest.fixture
def control(tmp_path, monkeypatch):
    from policy_doctor.register import engine
    entries = tmp_path / 'entries'
    entries.mkdir()
    entry = synthetic_entry()
    entry_path = dump(entries / 'SYNTHETIC.json', entry)
    config = dump(tmp_path / 'config.json', {'mode': 'bad'})
    lock = tmp_path / 'requirements.txt'
    lock.write_text('synthetic==1.2.0\n')
    manifest = {'schema_version': 1, 'kind': 'provenance_register_input',
                'stacks': ['synthetic'], 'lockfiles': ['requirements.txt'], 'config': 'config.json'}
    path = dump(tmp_path / 'input.json', manifest)
    original = engine.load_entries
    monkeypatch.setattr(engine, 'load_entries', lambda directory=None: original(entries if directory is None else directory))
    return {'path': path, 'manifest': manifest, 'entry': entry, 'entry_path': entry_path,
            'config': config, 'lock': lock, 'entries': entries}


def check(control):
    return assess_register(control['path'])


def test_positive_two_healthy_unknown_and_numeric_boundary(control):
    report = check(control)
    assert report['status'] == 'error'
    witness = report['entries'][0]['applicability'][0]
    assert witness['observations'][0]['sha256'] == hashlib.sha256(control['lock'].read_bytes()).hexdigest()
    assert witness['source_identity'] == 'not_attested'
    assert report['findings'][0]['first_fix'] == control['entry']['first_fix']
    dump(control['config'], {'mode': 'corrected'})
    assert check(control)['entries'][0]['status'] == 'consistent'
    dump(control['config'], {'mode': 'bad'})
    control['lock'].write_text('synthetic==2.0\n')
    assert check(control)['entries'][0]['status'] == 'not_applicable'
    control['lock'].write_text('synthetic==1.2.0\n')
    dump(control['config'], {})
    assert check(control)['status'] == 'unknown'
    assert version('1.10') > version('1.9')
    assert version('1.2') == version('1.2.0')


@pytest.mark.parametrize('pin', [None, '1..2', 'NaN'])
def test_unsupported_release_is_unknown(control, pin):
    control['lock'].write_text('synthetic==' + str(pin) + '\n')
    assert check(control)['status'] == 'unknown'


@pytest.mark.parametrize('name', ['uv.lock', 'poetry.lock', 'pip-freeze.txt'])
def test_all_lockfile_forms(control, name):
    path = control['path'].parent / name
    path.write_text('[[package]]\nname = "synthetic"\nversion = "1.2"\n' if name.endswith('.lock') else 'synthetic==1.2\n')
    control['manifest']['lockfiles'] = [name]
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'error'


@pytest.mark.parametrize('content', ['synthetic>=1.2', 'synthetic==1.2; python_version>"3"', '-r outside.txt',
                                     '[[package]]\nname = ["synthetic"]\nversion = "1.2"'])
def test_ambiguous_requirements_never_establish_error(control, content):
    control['lock'].write_text(content)
    assert check(control)['status'] == 'unknown'


def test_static_dist_info_no_import_conflict_and_forged_commit(control):
    root = control['path'].parent
    dist = root / 'synthetic-1.2.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Name: synthetic\nVersion: 1.2\n')
    # A matching commit is recorded, but is never a substitute for source bytes.
    dump(dist / 'direct_url.json', {'url': 'https://example.org/repo',
        'vcs_info': {'vcs': 'git', 'commit_id': 'a' * 40}})
    control['manifest']['metadata_paths'] = [dist.name]
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'error'
    assert check(control)['entries'][0]['applicability'][0]['observations'][0]['declared_commit'] == 'a' * 40
    (dist / 'METADATA').write_text('Name: synthetic\nVersion: 1.3\n')
    assert check(control)['status'] == 'unknown'
    (dist / 'METADATA').write_text('Name: synthetic\nVersion: 1.2\n')
    dump(dist / 'direct_url.json', {'url': 'https://example.org/repo', 'vcs_info': {}})
    assert check(control)['status'] == 'unknown'


def test_source_hash_required_and_mismatch_unknown(control):
    root = control['path'].parent / 'source'
    root.mkdir()
    source = root / 'module.py'
    source.write_text('setting = "bad"\n')
    control['entry']['claim_basis'] = 'source_bytes'
    control['entry']['applies_when'].append({'source_files': {'files': [
        {'path': 'module.py', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}})
    dump(control['entry_path'], control['entry'])
    assert check(control)['status'] == 'unknown'
    control['manifest']['runtime_source_root'] = 'source'
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'error'
    source.write_text('setting = "good"\n')
    assert check(control)['status'] == 'unknown'
    control['entry']['applies_when'].pop()
    dump(control['entry_path'], control['entry'])
    assert check(control)['catalog_errors']
    assert check(control)['status'] == 'unknown'


@pytest.mark.parametrize('source_text', ['# use_quantile_norm=True\n',
    'example = "use_quantile_norm=True"\n', 'if False:\n    use_quantile_norm=True\n',
    'use_quantile_norm=True\nuse_quantile_norm=False\n'])
def test_source_literal_comment_dead_code_override_can_never_error(control, source_text):
    source = control['path'].parent / 'source'
    source.mkdir()
    (source / 'module.py').write_text(source_text)
    control['entry']['claim_basis'] = 'source_bytes'
    control['entry']['applies_when'].append({'source_contains': {'path': 'module.py', 'literal': 'use_quantile_norm=True'}})
    dump(control['entry_path'], control['entry'])
    assert check(control)['status'] == 'unknown'
    control['manifest']['runtime_source_root'] = 'source'
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'warning'
    assert check(control)['entries'][0]['source_identity'] == 'weak_literal_hint_only'


def test_config_processor_checkpoint_launch_and_types(control):
    processor = dump(control['path'].parent / 'processor.json', {'steps': {'normalization': True}})
    control['manifest'].update(processor=processor.name, checkpoint_id='known/checkpoint',
                               launch_command='python train.py --env=LIBERO --train')
    control['entry']['applies_when'].extend([{'checkpoint_id_in': ['known/checkpoint']},
        {'processor_key': {'path': 'steps.normalization', 'op': 'eq', 'value': True}},
        {'launch_flag': {'flag': '--env', 'op': 'in', 'value': ['LIBERO', 'OTHER']}},
        {'launch_flag': {'flag': '--train', 'op': 'exists'}}])
    dump(control['entry_path'], control['entry'])
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'error'
    dump(processor, {'steps': {'normalization': 1}})
    assert check(control)['entries'][0]['status'] == 'not_applicable'
    dump(processor, {'steps': {'normalization': True}})
    control['manifest']['launch_command'] += ' --env LIBERO'
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'unknown'
    control['manifest']['launch_command'] = 'python train.py --env $(touch SHOULD_NEVER_EXIST)'
    dump(control['path'], control['manifest'])
    check(control)
    assert not (control['path'].parent / 'SHOULD_NEVER_EXIST').exists()


@pytest.mark.parametrize('field', list(synthetic_entry()))
@pytest.mark.parametrize('bad', [None, False, [], {}, 2, float('nan'), 'x' * 17000])
def test_every_entry_field_malformed_is_bounded_unknown(control, field, bad):
    control['entry'][field] = bad
    dump(control['entry_path'], control['entry'])
    report = check(control)
    assert report['status'] == 'unknown'
    assert report['catalog_errors']


@pytest.mark.parametrize('field', ['schema_version', 'kind', 'metadata_paths', 'lockfiles', 'collect_installed',
                                  'config', 'processor', 'launch_command', 'checkpoint_id', 'runtime_source_root'])
@pytest.mark.parametrize('bad', [None, [], {}, 2, float('inf'), 'x' * 17000])
def test_every_input_field_malformed_is_bounded_unknown(control, field, bad):
    control['manifest'][field] = bad
    dump(control['path'], control['manifest'])
    if field == 'metadata_paths' and bad == []:
        assert check(control)['status'] == 'error'  # An empty optional selector is valid.
        return
    assert check(control)['status'] == 'unknown'


@pytest.mark.parametrize('bounds', [{'gte': '2', 'lt': '1'}, {'gte': '1', 'gt': '1'},
    {'eq': '1', 'lte': '1'}, {'lt': '1', 'gte': '1'}, {'commit_range': 'abcdef'}, {'gte': None}, []])
def test_invalid_bounds_rejected(control, bounds):
    control['entry']['applies_when'][0]['package_version']['version'] = bounds
    dump(control['entry_path'], control['entry'])
    report = check(control)
    assert report['status'] == 'unknown'
    assert report['catalog_errors']  # Rejected schema, not a silently excluded rule.


@pytest.mark.parametrize('key', ['config', 'processor'])
@pytest.mark.parametrize('bad', [None, [], False, 10 ** 100, {'mode': float('nan')}, {'mode': 'x' * 17000}])
def test_document_malformed_values(control, key, bad):
    if key == 'processor':
        control['entry']['violation_when'] = [{'processor_key': {'path': 'mode', 'op': 'eq', 'value': 'bad'}}]
        dump(control['entry_path'], control['entry'])
        control['manifest']['processor'] = 'config.json'
        dump(control['path'], control['manifest'])
    dump(control['config'], bad)
    assert check(control)['status'] == 'unknown'


def test_symlinks_fifos_oversize_depth_duplicate_keys_and_read_roots(control):
    root = control['path'].parent
    link = root / 'linked.json'
    link.symlink_to(control['config'])
    control['manifest']['config'] = link.name
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'unknown'
    fifo = root / 'pipe'
    os.mkfifo(fifo)
    control['manifest']['config'] = fifo.name
    dump(control['path'], control['manifest'])
    assert check(control)['status'] == 'unknown'
    control['manifest']['config'] = 'config.json'
    dump(control['path'], control['manifest'])
    control['config'].write_text(' ' * (256 * 1024 + 1))
    assert check(control)['status'] == 'unknown'
    control['config'].write_text('{"mode":"bad","mode":"good"}')
    assert check(control)['status'] == 'unknown'
    control['config'].write_text('[' * 2000 + '0' + ']' * 2000)
    assert check(control)['status'] == 'unknown'
    allowed = root / 'narrow'
    allowed.mkdir()
    report = assess_register(control['path'], read_roots=[allowed])
    assert report['status'] == 'unknown' and 'authorized' in report['findings'][0]['message']


def test_empty_catalog_and_duplicate_id_are_unknown(control):
    empty = control['path'].parent / 'empty'
    empty.mkdir()
    assert assess_register(control['path'], entries_directory=empty)['status'] == 'unknown'
    dump(control['entries'] / 'duplicate.json', control['entry'])
    entries, errors = load_entries(control['entries'])
    assert not entries and errors
    assert check(control)['status'] == 'unknown'


@pytest.mark.parametrize('operator,pin,entry_status', [
    ('gt', '1.2', 'not_applicable'), ('gt', '1.2.0', 'not_applicable'),
    ('gt', '1.2.1', 'error'), ('lte', '1.2', 'error'),
    ('lte', '1.2.0', 'error'), ('lte', '1.2.1', 'not_applicable'),
])
def test_release_interval_boundary_distinguishes_equal_and_next_patch(control, operator, pin, entry_status):
    control['entry']['applies_when'][0]['package_version']['version'] = {operator: '1.2'}
    dump(control['entry_path'], control['entry'])
    control['lock'].write_text('synthetic==' + pin + '\n')
    report = check(control)
    assert not report['catalog_errors']
    assert report['entries'][0]['status'] == entry_status
    assert report['status'] == ('error' if entry_status == 'error' else 'unknown')


def test_independent_error_cannot_be_downgraded_by_another_rules_warning(control, capsys):
    warning = copy.deepcopy(control['entry'])
    warning.update(id='SYNTHETIC-WARNING-CONTROL', severity='WARNING')
    dump(control['entries'] / 'warning.json', warning)
    report = check(control)
    assert {row['status'] for row in report['entries']} == {'error', 'warning'}
    assert report['status'] == 'error'
    assert main(['check', str(control['path']), '--json', '--fail-on', 'error']) == 1
    actual = json.loads(capsys.readouterr().out)
    assert actual['status'] == 'error'
    assert {row['status'] for row in actual['findings']} == {'error', 'warning'}


def test_exact_pin_does_not_erase_an_unresolved_constraint_for_same_package(control):
    control['lock'].write_text('synthetic==1.2\nsynthetic>=1.0\n')
    report = check(control)
    assert report['status'] == 'unknown'
    assert report['entries'][0]['status'] == 'unknown'
    assert not any(row['status'] == 'error' for row in report['findings'])


def test_unrecognized_manifest_option_is_input_error_instead_of_silent_ignored_evidence(control):
    control['manifest']['confgi'] = 'intended-other-config.json'
    dump(control['path'], control['manifest'])
    report = check(control)
    assert report['status'] == 'unknown'
    assert report['findings'][0]['code'] == 'register.input_unknown'
    assert not report['entries']


@pytest.mark.parametrize('document', [[], ['mode', 'bad'], 'bad', 1, False])
def test_nonobject_configuration_reports_the_document_problem(control, document):
    dump(control['config'], document)
    report = check(control)
    assert report['status'] == 'unknown'
    witness = report['entries'][0]['violation'][0]
    assert 'object' in witness['reason'].lower()
    assert 'observed' not in witness


def test_json_subset_yaml_catalog_entry_is_evaluated(control):
    control['entry_path'].rename(control['entry_path'].with_suffix('.yaml'))
    report = check(control)
    assert not report['catalog_errors']
    assert report['status'] == 'error'
    assert report['entries'][0]['id'] == control['entry']['id']


@pytest.mark.parametrize('alteration', ['too_many_hashes', 'pending', 'multiline_title', 'bare_occurrences'])
def test_unvalidated_or_unsupported_rule_is_rejected_not_silently_applied(control, alteration):
    if alteration == 'too_many_hashes':
        control['entry']['claim_basis'] = 'source_bytes'
        control['entry']['applies_when'].append({'source_files': {'files': [
            {'path': 'module.py', 'sha256_in': [format(i, '064x') for i in range(33)]}]}})
    elif alteration == 'pending':
        control['entry']['validation']['status'] = 'pending'
    elif alteration == 'multiline_title':
        control['entry']['title'] = 'Actual condition\nFORGED PASS'
    else:
        control['entry']['claim_basis'] = 'source_bytes'
        control['entry']['applies_when'].append({'source_contains': {
            'path': 'module.py', 'literal': 'mode', 'occurrences': 2}})
    dump(control['entry_path'], control['entry'])
    report = check(control)
    assert report['status'] == 'unknown' and report['catalog_errors']
    assert not report['entries']
    assert not any(row['status'] == 'error' for row in report['findings'])


def test_shallow_bounded_json_rejects_depth_without_relying_on_parser_recursion_failure(control):
    value = 'small leaf'
    for _ in range(18):
        value = {'child': value}
    dump(control['config'], {'mode': 'bad', 'notes': value})
    report = check(control)
    assert report['status'] == 'unknown'
    witness = report['entries'][0]['violation'][0]
    assert 'depth' in witness['reason'].lower()


@pytest.mark.parametrize('value,expected', [('bad', 'error'), ('good', 'pass'), (None, 'unknown')])
def test_cli_mcp_gate_and_default_chooser_agree(control, capsys, value, expected):
    dump(control['config'], {} if value is None else {'mode': value})
    code = main(['check', str(control['path']), '--json', '--fail-on', 'error'])
    cli = json.loads(capsys.readouterr().out)
    assert cli['kind'] == 'provenance_register' and cli['status'] == expected
    assert code == (1 if expected == 'error' else 0)

    async def run():
        server = create_server([str(control['path'].parent)], server_name='RoboClue')
        result = await server.call_tool('preflight_check', {'source': str(control['path']), 'response_format': 'detailed'})
        structured = result[1]
        assert structured['kind'] == cli['kind']
        assert structured['status'] == cli['status']
        for options in [{'checkpoint': str(control['config'])}, {'video_headers': False}, {'max_files': 1}]:
            with pytest.raises(Exception, match='selectors'):
                await server.call_tool('preflight_check', {'source': str(control['path']), **options})
    asyncio.run(run())
    plan = choose_workflow('upgrade', manifest=str(control['path']))
    assert plan['assessment_status'] == 'not_run'
    assert plan['can_do_now'][0]['mcp_request']['name'] == 'preflight_check'
    assert plan['can_do_now'][0]['local_cli_argv'][:2] == ['roboclue', 'check']


def test_weak_warning_does_not_fail_error_gate(control, capsys):
    control['entry']['severity'] = 'WARNING'
    dump(control['entry_path'], control['entry'])
    assert main(['check', str(control['path']), '--json', '--fail-on', 'error']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'warning'


def test_default_cli_and_mcp_preserve_evidence_and_first_fix(control, capsys):
    assert main(['check', str(control['path']), '--fail-on', 'error']) == 1
    output = capsys.readouterr().out
    assert 'config.json#mode="bad"' in output and 'example.py:7' in output
    assert control['entry']['first_fix'] in output

    async def run():
        server = create_server([str(control['path'].parent)], server_name='RoboClue')
        _, result = await server.call_tool('preflight_check', {'source': str(control['path'])})
        assert result['status'] == 'error'
        assert result['findings'][0]['first_fix'] == control['entry']['first_fix']
        assert result['findings'][0]['evidence']
        assert result['findings'][0]['source_evidence'][0]['source']['line'] == 7
        assert len(json.dumps(result).encode()) <= 8192
    asyncio.run(run())


def test_compact_preview_bound_with_unicode_and_long_witnesses(control):
    from policy_doctor.register.engine import concise_register_summary

    report = check(control)
    report['source'] = '界' * 4000
    row = report['findings'][0]
    row['message'] = '界' * 400
    row['first_fix'] = '界' * 400
    row['evidence'] = [{'observed': '界' * 100} for _ in range(24)]
    row['source_evidence'][0]['url'] = 'https://example.org/' + '界' * 2000
    summary = concise_register_summary(report)
    assert len(json.dumps(summary).encode()) <= 8192
    assert summary['status'] == 'error'
    assert summary['findings'][0]['first_fix'] == row['first_fix']
    assert summary['findings'][0]['evidence_omitted'] >= 20


def test_nested_outside_path_mcp_unknown(control):
    inside = control['path'].parent / 'inside'
    inside.mkdir()
    manifest = copy.deepcopy(control['manifest'])
    manifest['config'] = str(control['config'])
    manifest['lockfiles'] = [str(control['lock'])]
    path = dump(inside / 'input.json', manifest)
    report = assess_register(path, read_roots=[inside])
    assert report['status'] == 'unknown'
    assert report['version_errors'] and 'authorized' in report['version_errors'][0]['reason']


def test_catalogue_schema_is_valid_json_and_vcs_ordering_unsupported():
    root = Path(__file__).parents[1]
    schema = json.loads((root / 'src/policy_doctor/register/schema-v1.json').read_text())
    assert schema['properties']['schema_version']['const'] == 1
    entry = synthetic_entry()
    entry['applies_when'] = [{'commit_range': {'after': '2026-01-01'}}]
    with pytest.raises(ValueError):
        validate_entry(entry)


@pytest.mark.parametrize('predicate', [
    {'launch_flag': {'flag': '--env', 'op': '__import__', 'value': 'anything'}},
    {'config_key': {'path': '../escape', 'op': 'eq', 'value': 1}},
    {'source_contains': {'path': 'module.py', 'literal': '', 'occurrences': {'gte': 1}}},
    {'source_contains': {'path': 'module.py', 'literal': 'text', 'occurrences': {'gte': True}}},
    {'source_contains': {'path': '../module.py', 'literal': 'text'}},
    {'source_files': {'files': [{'path': 'module.py', 'sha256': 'a' * 63}]}},
    {'source_files': {'files': [{'path': '/module.py', 'sha256': 'a' * 64}]}},
])
def test_nested_predicate_malformed_fields_never_run_code(control, predicate):
    control['entry']['applies_when'].append(predicate)
    dump(control['entry_path'], control['entry'])
    assert check(control)['status'] == 'unknown'
    assert check(control)['catalog_errors']


def test_literal_hint_cannot_be_a_violation_and_source_budget_is_explicit(control):
    control['entry']['violation_when'].append({'source_contains': {'path': 'x.py', 'literal': 'x'}})
    dump(control['entry_path'], control['entry'])
    assert check(control)['status'] == 'unknown'
    assert 'applicability' in check(control)['catalog_errors'][0]['reason']
    from policy_doctor.register.engine import _source_read

    with pytest.raises(ValueError, match='budget'):
        _source_read(control['config'], {'source_reads': 256, 'source_bytes': 0})
    with pytest.raises(ValueError, match='budget'):
        _source_read(control['config'], {'source_reads': 0, 'source_bytes': 64 * 1024 * 1024})


def test_explicit_source_hash_alternatives_are_or_and_paths_are_and(control):
    source = control['path'].parent / 'source'
    source.mkdir()
    module = source / 'module.py'
    values = [b'first verified historical blob\n', b'second verified historical blob\n']
    hashes = [hashlib.sha256(value).hexdigest() for value in values]
    other = source / 'other.py'
    other.write_bytes(b'other required file\n')
    rows = [{'path': 'module.py', 'sha256_in': hashes},
            {'path': 'other.py', 'sha256': hashlib.sha256(other.read_bytes()).hexdigest()}]
    control['entry']['claim_basis'] = 'source_bytes'
    control['entry']['applies_when'].append({'source_files': {'files': rows}})
    control['manifest']['runtime_source_root'] = 'source'
    dump(control['path'], control['manifest'])
    dump(control['entry_path'], control['entry'])
    for value in values:
        module.write_bytes(value)
        report = check(control)
        assert report['status'] == 'error'
        assert report['entries'][0]['applicability'][1]['files'][0]['sha256'] == hashlib.sha256(value).hexdigest()
    module.write_bytes(b'unlisted historical blob\n')
    assert check(control)['status'] == 'unknown'
    module.write_bytes(values[0])
    other.write_bytes(b'changed required file\n')
    assert check(control)['status'] == 'unknown'


@pytest.mark.parametrize('hashes', [[], ['a' * 64], ['a' * 64] * 2, ['a' * 64] * 33,
                                    [None, 'a' * 64], ['a' * 63, 'b' * 64], None, {}])
def test_malformed_source_alternatives_never_match(control, hashes):
    control['entry']['applies_when'].append({'source_files': {'files': [{'path': 'x.py', 'sha256_in': hashes}]}})
    dump(control['entry_path'], control['entry'])
    assert check(control)['status'] == 'unknown'
    assert check(control)['catalog_errors']


def test_source_alternatives_mutually_exclusive_and_repeated_paths_invalid(control):
    for rows in [[{'path': 'x.py', 'sha256': 'a' * 64, 'sha256_in': ['a' * 64, 'b' * 64]}],
                 [{'path': 'x.py', 'sha256': 'a' * 64}, {'path': 'x.py', 'sha256': 'b' * 64}]]:
        entry = synthetic_entry()
        entry['applies_when'].append({'source_files': {'files': rows}})
        with pytest.raises(ValueError):
            validate_entry(entry)


def test_unknown_precedes_false_within_applicability_and_violation(control):
    control['entry']['applies_when'].append({'checkpoint_id_in': ['known/checkpoint']})
    dump(control['entry_path'], control['entry'])
    control['lock'].write_text('synthetic==2.0\n')  # evidenced false plus unresolved identity
    assert check(control)['status'] == 'unknown'
    control['entry'] = synthetic_entry()
    control['entry']['violation_when'].append({'processor_key': {'path': 'missing', 'op': 'eq', 'value': 'bad'}})
    dump(control['entry_path'], control['entry'])
    control['lock'].write_text('synthetic==1.2\n')
    dump(control['config'], {'mode': 'corrected'})  # false violation plus unresolved second observation
    assert check(control)['status'] == 'unknown'


def test_actual_default_product_mcp_returns_structured_register(control):
    from policy_doctor.product_mcp import create_product_server

    async def run():
        server = create_product_server([str(control['path'].parent)])
        result = await server.call_tool('preflight_check', {'source': str(control['path'])})
        assert result.structuredContent['kind'] == 'provenance_register'
        assert result.structuredContent['status'] == 'error'
        assert result.structuredContent['findings'][0]['first_fix'] == control['entry']['first_fix']
    asyncio.run(run())
