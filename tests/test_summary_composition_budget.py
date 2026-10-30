"""Concise composition keeps checked-version conditions and independent source points."""

import asyncio
import copy
import hashlib
import importlib
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import test_action_conventions as action_fixtures
import test_consumer_frontdoors as consumer_fixtures
import test_preflight_review_fixes as review_fixtures
import test_units_manifest as units_fixtures

from policy_doctor.server import SUMMARY_LIMIT, _summary

engine = importlib.import_module('policy_doctor.preflight')
consumer_job, ordinary = consumer_fixtures.consumer_job, consumer_fixtures.ordinary
openpi_source = action_fixtures.openpi_source


@asynccontextmanager
async def client_for(root):
    mcp = pytest.importorskip('mcp')
    from mcp.client.stdio import stdio_client

    source = str(Path(__file__).resolve().parents[1] / 'src')
    bootstrap = ('import sys,runpy;sys.dont_write_bytecode=True;sys.path.insert(0,' + repr(source) + ');'
                 'sys.argv=["policy_doctor","serve","--root",' + repr(str(root)) + '];'
                 'runpy.run_module("policy_doctor",run_name="__main__")')
    params = mcp.StdioServerParameters(command=sys.executable, args=['-I', '-B', '-c', bootstrap],
        cwd=str(root), env={**os.environ, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
    async with stdio_client(params) as (read, write):
        async with mcp.ClientSession(read, write) as client:
            await client.initialize()
            yield client


def assert_versions(short, full):
    compact = short['components']['groot_contract']
    original = full['components']['groot_contract']
    for key in ('headline', 'remedy', 'mode', 'groot_version', 'versions_run', 'fails_on', 'counts', 'finding_codes'):
        assert compact[key] == original[key]
    for name, version in original['versions'].items():
        for key in ('status', 'counts', 'failure', 'applies_to', 'statement', 'command'):
            assert compact['versions'][name][key] == version[key]
        inventory = compact['versions'][name]['not_checked']
        if isinstance(inventory, dict):
            items = version['not_checked']
            assert inventory['total'] == len(items)
            assert inventory['omitted'] == len(items) - len(inventory['preview'])
            assert inventory['preview'] == items[:len(inventory['preview'])]
            assert inventory['sha256'] == hashlib.sha256(json.dumps(
                items, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            assert inventory['retrieval'] == 'detailed'
        else:
            assert inventory == version['not_checked']


def test_combined_consumer_units_keeps_full_conditions_without_mutating_report(consumer_job):
    full = engine.preflight(**consumer_fixtures.request(consumer_job, units=True, groot_version=None))
    original = json.dumps(full, sort_keys=True)
    compact = _summary(full)
    assert json.dumps(full, sort_keys=True) == original
    assert len(json.dumps(compact).encode()) <= SUMMARY_LIMIT == 8192
    assert len(compact['findings_shown']) >= 3
    assert_versions(compact, full)
    consumer_fixtures.assert_source_witness(compact['components']['consumer_comparison'],
                                            consumer_fixtures.parquet_identity(consumer_job))
    assert compact['components']['units']['limits'] == engine.UNITS_LIMITS


@pytest.mark.parametrize('reported_size', [200_000, 2_000_000, 200_000_000])
def test_detailed_fallback_keeps_combined_witnesses_and_report_identity(consumer_job, reported_size):
    full = engine.preflight(**consumer_fixtures.request(consumer_job, units=True, groot_version=None))
    original = json.dumps(full, sort_keys=True)
    compact = _summary(full, reported_size)
    assert json.dumps(full, sort_keys=True) == original
    assert len(json.dumps(compact).encode()) <= SUMMARY_LIMIT == 8192
    assert compact['status'] == full['status'] and compact['counts'] == full['counts']
    assert compact['truncated_from_bytes'] == reported_size
    assert compact['full_report_sha256'] == hashlib.sha256(json.dumps(
        full, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert len(compact['findings_shown']) >= 3
    assert_versions(compact, full)
    consumer_fixtures.assert_source_witness(compact['components']['consumer_comparison'],
                                            consumer_fixtures.parquet_identity(consumer_job))
    assert compact['components']['units']['limits'] == engine.UNITS_LIMITS
    assert '--json --output FILE' in compact['next']


def test_actual_stdio_combined_consumer_units_with_unicode_source(consumer_job, tmp_path):
    parent = tmp_path / ('机器人🙂' * 12)
    parent.mkdir()
    source = Path(consumer_job['source'])
    moved = parent / source.name
    source.rename(moved)
    job = {**consumer_job, 'source': str(moved)}
    arguments = {key: val for key, val in consumer_fixtures.request(job, units=True, groot_version=None).items()
                 if key != 'read_roots'}
    full = engine.preflight(**consumer_fixtures.request(job, units=True, groot_version=None))

    async def run():
        async with client_for(tmp_path) as client:
            response = await client.call_tool('preflight_check', arguments)
            assert not response.isError, response.content
            short = response.structuredContent
            assert len(json.dumps(short).encode()) <= SUMMARY_LIMIT == 8192
            assert len(short['findings_shown']) >= 3
            assert_versions(short, full)
            part = short['components']['consumer_comparison']
            consumer_fixtures.assert_source_witness(part, consumer_fixtures.parquet_identity(job))
            assert part['first_final_value_difference']['before'] != part['first_final_value_difference']['after']
            source_witness = part['first_value_divergence']['reference_source']
            assert source_witness['directory_truncated'] is True
            assert source_witness['full_directory_sha256'] == hashlib.sha256(str(moved).encode()).hexdigest()
    asyncio.run(asyncio.wait_for(run(), timeout=120))


def test_equal_coordinates_in_different_windows_do_not_share_point_reference(consumer_job):
    full = engine.preflight(**consumer_fixtures.request(consumer_job, groot_version='n1.7'))
    windows = full['components']['consumer_comparison']['comparison']['samples']
    other = copy.deepcopy(windows[0])
    other['requested_selection']['episode_index'] = 999
    other['first_value_divergence'] = None
    windows.insert(0, other)
    full['findings'] += [{'code': 'generated.pressure', 'status': 'warning', 'message': 'pressure ' * 100,
                         'evidence': []} for _ in range(40)]
    compact = _summary(full)['components']['consumer_comparison']
    assert compact['first_observed_divergence']['stage'] == other['first_observed_divergence']['stage']
    assert 'source_point_from' not in compact['first_observed_divergence']
    assert compact['first_value_divergence']['requested_selection']['episode_index'] == 72


@pytest.mark.parametrize('pressure', [False, True])
def test_long_remedy_is_whole_or_withheld_with_full_original_digest(pressure):
    remedy = 'Use --required-mode=false only if the policy matches. ' + 'Explanation. ' * 60
    remedy += 'Otherwise retrain the policy; this final alternative must not disappear.'
    finding = {'code': 'regression.conditional_remedy', 'status': 'error', 'message': 'A saved mode disagrees.',
               'remedy': remedy, 'evidence': [{'path': 'config.json', 'pointer': '/mode', 'sha256': 'a' * 64}]}
    findings = [finding]
    if pressure:
        findings += [{'code': 'generated.pressure', 'status': 'warning', 'message': 'pressure ' * 100,
                      'remedy': '', 'evidence': []} for _ in range(40)]
    result = _summary({'kind': 'dataset_preflight', 'status': 'error', 'findings': findings})
    assert len(json.dumps(result).encode()) <= SUMMARY_LIMIT == 8192
    item = next(f for f in result['findings_shown'] if f['code'] == finding['code'])
    assert item['remedy'] == '' and item['remedy_omitted'] is True
    assert item['full_remedy_sha256'] == hashlib.sha256(remedy.encode()).hexdigest()
    assert item['full_remedy_sha256'] != hashlib.sha256(remedy[:300].encode()).hexdigest()
    assert "response_format='detailed'" in result['next']


def test_actual_stdio_pressure_keeps_each_named_consumer_verdict(tmp_path, openpi_source):
    names = ['delta_' + str(i) for i in range(6)] + ['gripper']
    dataset = action_fixtures.write_dataset(tmp_path / 'mixed', 'delta', dims=6, gripper=True, names=names)[0]
    checkpoint = action_fixtures.write_checkpoint(tmp_path / 'relative_processor', names)
    (dataset / 'meta/modality.json').write_text(json.dumps(review_fixtures.wide_modality(30)))
    arguments = {'source': str(dataset), 'checkpoint': str(checkpoint), 'openpi_config': 'pi0_libero',
                 'openpi_source_root': str(openpi_source)}
    full = engine.preflight(**arguments)
    expected = full['components']['action_conventions']['verdicts']['per_consumer']
    assert set(expected.values()) == {'error', 'pass'} and len(expected) == 2

    async def run():
        async with client_for(tmp_path) as client:
            response = await client.call_tool('preflight_check', arguments)
            assert not response.isError, response.content
            body = response.structuredContent
            assert len(json.dumps(body).encode()) <= SUMMARY_LIMIT == 8192
            assert len(body['findings_shown']) >= 3
            assert body['components']['action_conventions']['per_consumer'] == expected
            assert_versions(body, full)
    asyncio.run(asyncio.wait_for(run(), timeout=120))


def test_actual_stdio_buried_units_witness_and_csv_alias_refusal(tmp_path):
    root = units_fixtures.so_dataset(tmp_path, 'm100')
    (root / 'meta/modality.json').write_text(json.dumps(review_fixtures.wide_modality(30)))
    csv = root / 'neighbor.csv'
    csv.write_text('action\n1\n')
    full = engine.preflight(str(root), robot_command=units_fixtures.command())
    assert full['components']['groot_contract']['counts']['error'] >= 30
    expected = next(f for f in full['findings'] if f['code'] == 'units.joint_mode_mismatch')

    async def run():
        async with client_for(tmp_path) as client:
            refused = await client.call_tool('preflight_check', {'source': str(csv)})
            assert refused.isError
            assert str(csv) in refused.content[0].text and 'meta/info.json' in refused.content[0].text
            detailed = await client.call_tool('preflight_check', {'source': str(root),
                'robot_command': units_fixtures.command(), 'response_format': 'detailed'})
            assert not detailed.isError, detailed.content
            complete = detailed.structuredContent
            exact = next(f for f in complete['findings'] if f['code'] == expected['code'])
            assert exact['status'] == expected['status'] == 'error'
            assert exact['evidence'] == expected['evidence']
            response = await client.call_tool('preflight_check', {'source': str(root),
                                                                 'robot_command': units_fixtures.command()})
            assert not response.isError, response.content
            body = response.structuredContent
            assert len(json.dumps(body).encode()) <= SUMMARY_LIMIT == 8192
            assert body['status'] == 'error'
            assert body['full_report_sha256'] == hashlib.sha256(json.dumps(
                complete, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                allow_nan=False).encode()).hexdigest()
            findings = body['findings_shown']
            assert len(findings) >= 3
            unit = next(f for f in findings[:3] if f['code'] == expected['code'])
            assert unit['remedy'] == expected['remedy'] or (
                unit.get('remedy_omitted') is True and unit.get('remedy') == ''
                and unit.get('full_remedy_sha256') == hashlib.sha256(expected['remedy'].encode()).hexdigest())
            for original in expected['evidence'][:2]:
                witness = next(e for e in unit['evidence'] if e.get('sha256') == original.get('sha256'))
                assert_evidence_path_identity(unit, witness, original)
                assert witness['pointer'] == original['pointer']
            assert_versions(body, full)
            alias = await client.call_tool('preflight_check', {'source': str(root / 'meta/info.json')})
            assert not alias.isError, alias.content
    asyncio.run(asyncio.wait_for(run(), timeout=120))


def assert_evidence_path_identity(finding, witness, original):
    """A displayed prefix needs an explicit clipping marker and full identity."""
    if witness['path'] == original['path']:
        return
    assert witness['path'] and original['path'].startswith(witness['path'])
    assert len(json.dumps(witness['path'], ensure_ascii=True)) - 2 <= 256
    assert len(json.dumps(original['path'][:len(witness['path']) + 1], ensure_ascii=True)) - 2 > 256
    assert witness['full_path_sha256'] == hashlib.sha256(original['path'].encode()).hexdigest()
    index = finding['evidence'].index(witness)
    assert f'evidence/{index}/path' in finding['text_truncated']


def test_bounded_evidence_path_identity_rejects_missing_or_wrong_proof():
    original = {'path': '/recording/' + 'x' * 500, 'sha256': 'a' * 64, 'pointer': '/action'}
    witness = dict(original, path=original['path'][:256],
                   full_path_sha256=hashlib.sha256(original['path'].encode()).hexdigest())
    finding = {'evidence': [witness], 'text_truncated': ['evidence/0/path']}
    assert_evidence_path_identity(finding, witness, original)
    assert_evidence_path_identity({'evidence': [original]}, original, original)
    for change in ('missing_digest', 'wrong_digest', 'missing_marker', 'wrong_prefix', 'short_prefix'):
        broken = copy.deepcopy(finding)
        shown = broken['evidence'][0]
        if change == 'missing_digest':
            shown.pop('full_path_sha256')
        elif change == 'wrong_digest':
            shown['full_path_sha256'] = '0' * 64
        elif change == 'missing_marker':
            broken['text_truncated'] = []
        elif change == 'wrong_prefix':
            shown['path'] = '/different'
        else:
            shown['path'] = original['path'][:1]
        with pytest.raises((AssertionError, KeyError)):
            assert_evidence_path_identity(broken, shown, original)
