"""Default previews stay bounded without losing the full diagnostic contract."""
import asyncio
import copy
import hashlib
import json

import pytest
from test_provenance_register import control as control
from test_provenance_register import dump
from test_register_recipes import recipe

from policy_doctor.product_mcp import create_product_server
from policy_doctor.register.engine import assess_register, concise_register_summary

# Explicitly export the shared fixture for pytest and supported Ruff versions.
__all__ = ['control']


def assert_views(control, report):
    before = copy.deepcopy(report)
    preview = concise_register_summary(report)
    assert len(json.dumps(preview).encode()) <= 8192
    assert report == before
    assert preview['status'] == report['status']
    assert preview['finding_counts'][report['status']] >= 1
    encoded = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    assert preview['full_report_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert preview['full_report_size_bytes'] == len(encoded)

    async def run():
        server = create_product_server([str(control['path'].parent)])
        response = await server.call_tool('preflight_check', {'source': str(control['path'])})
        structured = response.structuredContent if hasattr(response, 'structuredContent') else response[1]
        content = response.content if hasattr(response, 'content') else response[0]
        assert structured['status'] == report['status']
        assert len(json.dumps(structured).encode()) <= 8192
        assert all(len(item.text.encode()) <= 8192 for item in content if item.type == 'text')
    asyncio.run(run())
    return preview['findings'][0]


@pytest.mark.parametrize('character', ['x', '\u754c', '\U0001f600', '"'])
def test_long_messages_and_first_fixes_are_explicitly_omitted_not_clipped(control, character):
    entry = control['entry']
    entry['title'] = character * 2048
    entry['first_fix'] = 'python repair.py --setting ' + character * 2000
    dump(control['entry_path'], entry)
    report = assess_register(control['path'])
    assert report['status'] == 'error'
    finding = assert_views(control, report)
    assert finding['message_truncated']
    for key in ('message', 'first_fix'):
        if not finding.get(key + '_truncated'):
            assert finding[key] == report['findings'][0][key]
            continue
        original = report['findings'][0][key].encode()
        assert finding[key + '_sha256'] == hashlib.sha256(original).hexdigest()
        assert finding[key + '_size_bytes'] == len(original)
    if finding.get('first_fix_truncated'):
        assert finding['first_fix'] == 'Retrieve the detailed report for the complete first fix before changing settings.'
    assert report['findings'][0]['first_fix'] == entry['first_fix']


@pytest.mark.parametrize('character', ['x', '\u754c', '\U0001f600'])
def test_recipe_large_actual_value_keeps_warning_caveat_and_original_value(control, character):
    recipe(control)
    actual = character * 3000
    dump(control['config'], {'mode': actual})
    report = assess_register(control['path'])
    assert report['status'] == 'warning'
    finding = assert_views(control, report)
    assert finding['message_truncated']
    assert 'differs from the published evaluation recipe' in finding['message']
    assert 'Results are not comparable to the published numbers.' in finding['message']
    assert finding['first_fix'] == report['findings'][0]['first_fix']
    witness = finding['evidence'][0]
    assert witness['phase'] == 'violation' and witness['op'] == 'ne'
    assert witness['observed_type'] == 'string' and witness['observed_truncated']
    assert report['findings'][0]['evidence'][0]['observed'] == actual


def test_maximum_unicode_source_and_selector_keep_decisive_typed_witness(control):
    entry = control['entry']
    selector = 'x' * 250
    entry['title'] = '\u754c' * 2048
    entry['first_fix'] = '\u754c' * 2048
    entry['violation_when'] = [{'config_key': {'path': selector, 'op': 'eq', 'value': True}}]
    source = entry['evidence'][0]
    source['url'] = 'https://example.org/' + '\u754c' * 2000
    source['quote'] = '\u754c' * 2048
    source['source'].update(repo='\u754c' * 256, path='\u754c' * 512)
    dump(control['entry_path'], entry)
    dump(control['config'], {selector: True})
    report = assess_register(control['path'])
    assert report['status'] == 'error' and not report['catalog_errors']
    finding = assert_views(control, report)
    witness = finding['evidence'][0]
    assert witness['phase'] == 'violation' and witness['op'] == 'eq'
    assert witness['observed'] is True and witness['expected'] is True
    assert witness['selector'] == selector
    assert witness['witness_sha256']
    assert finding['source_evidence'][0]['source']['sha'] == 'a' * 40
    assert report['findings'][0]['source_evidence'][0]['source']['path'] == '\u754c' * 512
