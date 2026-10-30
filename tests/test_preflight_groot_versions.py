"""Version-specific GR00T evidence must survive each public preflight surface."""
import asyncio
import hashlib
import json

import pytest
import test_groot_contract as fixtures

from policy_doctor.artifacts import InputError
from policy_doctor.cli import human, main
from policy_doctor.preflight import _example_key, _omitted, _reduced, preflight
from policy_doctor.server import _summary, create_server


def test_default_checks_both_and_renderers_keep_each_profile(tmp_path):
    root = fixtures._dataset(tmp_path / 'data')
    report = preflight(str(root))
    part = report['components']['groot_contract']
    assert part['mode'] == 'both'
    assert part['versions_run'] == ['n1.5', 'n1.7']
    assert set(part['versions']) == {'n1.5', 'n1.7'}
    assert part['coverage']['not_checked']
    text = human(report)
    assert part['headline'] in text
    for version, verdict in part['versions'].items():
        assert f"{version}: {verdict['status']}" in text
    concise = _summary(report)['components']['groot_contract']
    for key in ('headline', 'remedy', 'mode', 'versions_run', 'fails_on'):
        assert concise[key] == part[key]
    for version, summary in concise['versions'].items():
        for key in ('status', 'counts', 'failure', 'applies_to', 'statement'):
            assert summary[key] == part['versions'][version][key]
        original_scope = part['versions'][version]['not_checked']
        scope = summary['not_checked']
        if isinstance(scope, dict):
            assert scope['total'] == len(original_scope)
            assert scope['preview'] == original_scope[:len(scope['preview'])]
            assert scope['omitted'] == len(original_scope) - len(scope['preview'])
            assert scope['sha256'] == hashlib.sha256(json.dumps(
                original_scope, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                allow_nan=False).encode()).hexdigest()
            assert scope['retrieval'] == 'detailed'
            assert 'Metadata only' in concise['not_checked_scope']
        else:
            assert scope == original_scope
    for reduce in (_reduced, _omitted):
        bounded = reduce('groot_contract', part)
        for key in ('headline', 'remedy', 'mode', 'versions_run', 'fails_on', 'versions'):
            assert bounded[key] == part[key]
        assert bounded['coverage']['not_checked'] == part['coverage']['not_checked']


def test_detected_n17_and_explicit_selector(tmp_path):
    root = fixtures._dataset(tmp_path / 'data')
    config = fixtures._config(tmp_path)
    detected = preflight(str(root), groot_modality_config=str(config))['components']['groot_contract']
    assert detected['mode'] == 'detected'
    assert detected['versions_run'] == ['n1.7']
    explicit = preflight(str(root), groot_version='n1.5')['components']['groot_contract']
    assert explicit['mode'] == 'explicit'
    assert explicit['versions_run'] == ['n1.5']
    assert all(f['groot_version'] == 'n1.5' for f in explicit['findings'])


def test_cli_version_reaches_engine(tmp_path, capsys):
    root = fixtures._dataset(tmp_path / 'data')
    code = main(['preflight', str(root), '--groot-version', 'n1.7', '--json'])
    assert code in (0, 1, 3)
    report = json.loads(capsys.readouterr().out)
    assert report['components']['groot_contract']['versions_run'] == ['n1.7']


def test_mcp_schema_and_detailed_result_are_version_specific(tmp_path):
    root = fixtures._dataset(tmp_path / 'data')
    server = create_server([str(tmp_path)])

    async def run():
        tools = await server.list_tools()
        schema = next(t.inputSchema for t in tools if t.name == 'preflight_check')
        variants = schema['properties']['groot_version']['anyOf']
        assert any(v.get('enum') == ['n1.5', 'n1.7'] for v in variants)
        blocks = await server.call_tool('preflight_check', {
            'source': str(root), 'groot_version': 'n1.5', 'response_format': 'detailed'})
        texts = blocks[0] if isinstance(blocks, tuple) else blocks
        result = json.loads(next(b.text for b in texts if getattr(b, 'type', None) == 'text'))
        assert result['components']['groot_contract']['versions_run'] == ['n1.5']
    asyncio.run(run())


@pytest.mark.parametrize('value', ['auto', '', 'N1.5', True, 15, []])
def test_invalid_versions_refuse_before_io(value):
    with pytest.raises(InputError, match='groot-version'):
        preflight('/not-read', groot_version=value)


def test_optout_and_explicit_version_are_conflicting(tmp_path):
    root = fixtures._dataset(tmp_path / 'data')
    with pytest.raises(InputError, match='groot-version'):
        preflight(str(root), groot_contract=False, groot_version='n1.5')


def test_capped_examples_distinguish_identical_codes_between_versions():
    finding = {'code': 'groot.stats', 'status': 'error'}
    assert _example_key({**finding, 'groot_version': 'n1.5'}) != _example_key({**finding, 'groot_version': 'n1.7'})
