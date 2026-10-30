"""Community contribution/discovery controls, not new native qualification."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.product_mcp import BASIC_TOOLS
from policy_doctor.rule_catalog import (
    get_rule,
    render_catalog,
    render_rule,
    rule_catalog,
    rules_for_code,
    validate_catalog,
)

ROOT = Path(__file__).parents[1]


def walkthrough():
    spec = importlib.util.spec_from_file_location('contribution_walkthrough', ROOT / 'examples/rules/contribution_walkthrough.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_walkthrough()


def test_stable_ids_exact_diagnostic_mapping_and_independent_copies():
    catalog = rule_catalog()
    assert not validate_catalog(catalog)
    ids = [r['id'] for r in catalog['rules']]
    assert catalog['count'] == len(ids) == len(set(ids))
    assert {'RC001', 'RC002', 'RC003', 'RC004', 'RC005', 'RC006', 'RC007', 'RC008', 'RC009', 'RC010', 'RC011', 'RC013'} <= set(ids)
    assert json.loads(json.dumps(catalog)) == catalog
    for rule in catalog['rules']:
        assert get_rule(rule['id']) == rule
        for code in rule['diagnostic_codes']:
            assert [r['id'] for r in rules_for_code(code)] == [rule['id']]
    catalog['rules'][0]['diagnostic_codes'].clear()
    get_rule('RC001')['frameworks'][0]['revision'] = 'untested'
    assert rules_for_code('units.joint_mode_mismatch')[0]['id'] == 'RC001'
    assert get_rule('RC001')['frameworks'][0]['revision'] != 'untested'
    assert rules_for_code('units.some_future_rule') == []
    assert rules_for_code('units.') == []


@pytest.mark.parametrize('rule_id', ['RC999', 'RC01', 'rc001', '../RC001', '', None, True])
def test_invalid_or_unknown_ids_are_not_empty_success(rule_id):
    with pytest.raises(InputError, match='rule|Rule'):
        get_rule(rule_id)


def test_catalog_pages_guides_tests_and_default_mcp_routes_exist():
    for rule in rule_catalog()['rules']:
        page = ROOT / rule['docs']
        assert page.is_file() and page.stem == rule['id']
        assert rule['id'] in page.read_text()
        assert (ROOT / rule['guide']).is_file()
        assert all((ROOT / name).is_file() for name in rule['tests'])
        assert rule['mcp']['tool'] in BASIC_TOOLS
        assert 'roboclue check ' in rule['cli']
        for profile in rule['frameworks']:
            assert re.fullmatch('[a-f0-9]{40}', profile['revision'])
        assert rule['not_checked'] and rule['scope']
    for page in (ROOT / 'docs/rules').glob('*.md'):
        for destination in re.findall(r'\]\(([^)]+)\)', page.read_text()):
            if '://' in destination or destination.startswith('#'):
                continue
            assert (page.parent / destination.split('#', 1)[0]).exists(), (page, destination)


def test_codes_are_existing_machine_codes_not_new_presentation_names(tmp_path):
    # Include preexisting assertions for generated codes such as groot.<side>_config.
    # Exclude the catalog/new tests so a typo cannot validate itself.
    literals = set()
    for folder in (ROOT / 'src/policy_doctor', ROOT / 'tests'):
        for path in folder.rglob('*.py'):
            if path.name in {'rule_catalog.py', 'test_rule_catalog.py'}:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    literals.add(node.value)
    # Curated register codes are composed from validated entry IDs at runtime.
    # Require their real emission, rather than accepting a presentation typo or
    # teaching the assertion a hard-coded copy of the catalogue's names.
    from policy_doctor.register import assess_register

    manifest = tmp_path / 'missing-register-evidence.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'provenance_register_input'}))
    emitted = assess_register(manifest)
    assert emitted['status'] == 'unknown'
    literals.update(finding['code'] for finding in emitted['findings'])
    for rule in rule_catalog()['rules']:
        for code in rule['diagnostic_codes']:
            assert code in literals, code


@pytest.mark.parametrize('mutation, fragment', [
    ('id', 'stable'), ('duplicate_id', 'duplicates rule ID'), ('duplicate_code', 'duplicates diagnostic'),
    ('count', 'count'), ('upstream', 'upstream'), ('frameworks', 'profiles'), ('mcp', 'MCP'),
    ('scope', 'scope'), ('bad_url', 'invalid upstream'), ('insecure_url', 'https'),
])
def test_contribution_metadata_errors_are_actionable(mutation, fragment):
    catalog = rule_catalog()
    first = catalog['rules'][0]
    if mutation == 'duplicate_id':
        catalog['rules'][1]['id'] = first['id']
    elif mutation == 'duplicate_code':
        catalog['rules'][1]['diagnostic_codes'].append(first['diagnostic_codes'][0])
    elif mutation == 'count':
        catalog['count'] += 1
    elif mutation == 'bad_url':
        first['upstream'][0]['url'] = 'https://['
    elif mutation == 'insecure_url':
        first['upstream'][0]['url'] = 'file:///secret'
    else:
        first[mutation] = [] if mutation in {'upstream', 'frameworks'} else None
    assert any(fragment in error for error in validate_catalog(catalog))


@pytest.mark.parametrize('bad', [None, [], {}, {'schema_version': True, 'rules': [], 'count': 0}])
def test_structural_validation_does_not_claim_scientific_validation(bad):
    if bad is None:
        assert validate_catalog() == []
    else:
        assert validate_catalog(bad)


def test_rendered_discovery_retains_scope_versions_and_commands():
    assert all(rule['id'] in render_catalog() for rule in rule_catalog()['rules'])
    text = render_rule(get_rule('RC004'))
    assert 'DeltaActions' in text and 'roboclue check' in text
    assert 'dependency version' in text and 'not complete consumer compatibility' in text
    assert '215abfb217dbac7d5f1273282331b9b1866c0479' in text


def test_worked_example_uses_real_detector_and_keeps_missing_unsupported_unknown():
    result = walkthrough()
    assert result['status'] == 'pass'
    assert {name: item['status'] for name, item in result['controls'].items()} == {
        'broken': 'error', 'healthy': 'consistent', 'corrected': 'consistent',
        'missing': 'unknown', 'unsupported_shape': 'unknown'}
    assert all(c['input_unchanged'] for c in result['controls'].values())
    assert result['controls']['broken']['codes'] == ['dataset.statistics_inconsistent']
    assert result['controls']['corrected']['codes'] == []




def test_actual_cli_catalog_list_rule_json_and_invalid_id(capsys):
    from policy_doctor.product_cli import main

    assert main(['advanced', 'rules']) == 0
    assert 'RC001' in capsys.readouterr().out
    assert main(['advanced', 'rules', 'RC005', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['id'] == 'RC005' and 'salvage.resume_hazard' in result['diagnostic_codes']
    assert main(['advanced', 'rules', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['count'] == len(rule_catalog()['rules'])
    assert main(['advanced', 'rules', 'RC999']) == 2
    captured = capsys.readouterr()
    assert 'Unknown rule RC999' in captured.err and not captured.out
