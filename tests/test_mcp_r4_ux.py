"""Default MCP discoverability, scoped language and unchanged evidence contracts."""
from __future__ import annotations

import asyncio
import copy
import json

from test_mcp_r3_frontdoor import strip_descriptions

from policy_doctor.deployment_answer import deployment_tool_result
from policy_doctor.mcp_presentation import evidence_tool_result, workflow_tool_result
from policy_doctor.product_mcp import BASIC_TOOLS, create_product_server
from policy_doctor.server import _summary, create_server


def test_every_default_parameter_is_described_without_weakening_schema(tmp_path):
    async def collect():
        product = await create_product_server([str(tmp_path)]).list_tools()
        legacy = await create_server([str(tmp_path)]).list_tools()
        advanced = await create_product_server([str(tmp_path)], advanced=True).list_tools()
        return product, {tool.name: tool for tool in legacy}, advanced

    product, legacy, advanced = asyncio.run(collect())
    assert {tool.name for tool in product} == BASIC_TOOLS
    assert len(advanced) == 29
    assert sum(len(tool.inputSchema['properties']) for tool in product) >= 86
    for tool in product:
        assert tool.description
        for key, schema in tool.inputSchema['properties'].items():
            assert len(schema.get('description', '')) >= 15, (tool.name, key)
        assert strip_descriptions(tool.inputSchema) == strip_descriptions(legacy[tool.name].inputSchema)


def test_workflow_text_is_a_plan_not_duplicate_json():
    report = {'assessment_status': 'not_run', 'can_do_now': [{'mcp_request': {
        'name': 'preflight_check', 'arguments': {'source': '/declared', 'lerobot_version': '0.6.2'}}}],
        'missing_evidence': ['actual robot command']}
    before = copy.deepcopy(report)
    response = workflow_tool_result(report)
    assert response.structuredContent == before == report
    assert 'preflight_check' in response.content[0].text
    assert 'no assessment has run' in response.content[0].text
    assert not response.content[0].text.lstrip().startswith('{')
    assert '/declared' not in response.content[0].text


def test_units_without_cameras_always_have_readable_scoped_text():
    report = {'kind': 'deployment_units_answer', 'status': 'error',
        'components': {'units': {'status': 'error', 'detected_units': ['RANGE_M100_100'], 'declared_units': ['DEGREES']}},
        'next': 'Review --robot.use_degrees=false', 'full_report_sha256': 'a' * 64,
        'details': 'response_format=detailed', 'framework_context': {'runtime_revalidated': False}}
    before = copy.deepcopy(report)
    result = deployment_tool_result(report)
    text = result.content[0].text
    assert result.structuredContent == before == report
    assert '-100..100 across the calibrated range' in text and 'robot degrees' in text
    assert '--robot.use_degrees=false' in text
    assert all(raw not in text for raw in ('RANGE_M100_100', 'DEGREES', 'fleet', 'a' * 64))
    assert len(json.dumps(result.model_dump(mode='json', exclude_none=True)).encode()) <= 2560


def test_health_text_distinguishes_numeric_row_comparison_from_saved_self_consistency():
    report = {'kind': 'dataset_health_card', 'status': 'pass', 'components': {
        'dataset_statistics': {'status': 'pass'}, 'numeric_rows': {'status': 'pass',
            'statistics_comparison': {'status': 'pass', 'complete_cohort': True, 'compared_features': ['action']}}}}
    before = copy.deepcopy(report)
    result = evidence_tool_result(report, detailed=False)
    assert 'match the covered rows' in result.content[0].text
    assert 'image statistics and quantiles were not compared' in result.content[0].text
    assert result.structuredContent == before == report
    report['components'].pop('numeric_rows')
    assert 'self-consistency only' in evidence_tool_result(report, detailed=False).content[0].text
    report['components']['numeric_rows'] = {'status': 'unknown', 'statistics_comparison': {
        'status': 'unknown', 'complete_cohort': False, 'reason': 'row budget exhausted'}}
    assert 'row budget exhausted' in evidence_tool_result(report, detailed=False).content[0].text


def test_summary_preserves_new_numeric_comparison_and_old_statistics_scope():
    report = {'kind': 'dataset_preflight', 'status': 'pass', 'components': {
        'numeric_rows': {'status': 'pass', 'statistics_comparison': {
            'status': 'pass', 'complete_cohort': True, 'compared_features': ['action'],
            'compared_metrics': ['mean'], 'rows': 400, 'mismatch_count': 0}},
        'dataset_statistics': {'status': 'pass', 'observed_row_statistics_compared': False}}}
    summary = _summary(report)
    assert summary['components']['numeric_rows']['statistics_comparison']['compared_features'] == ['action']
    assert summary['components']['dataset_statistics']['observed_row_statistics_compared'] is False


def test_health_frontdoor_carries_numeric_comparison_to_readable_text():
    from policy_doctor.big_bets_frontdoor import summarize

    card = {'kind': 'dataset_health_card', 'status': 'pass', 'report': {
        'status': 'pass', 'components': {'numeric_rows': {'status': 'pass', 'statistics_comparison': {
            'status': 'pass', 'complete_cohort': True, 'compared_features': ['action'],
            'compared_metrics': ['mean', 'std'], 'rows': 400}}, 'dataset_statistics': {'status': 'pass'}}}}
    brief = summarize(card)
    assert brief['components']['numeric_rows']['statistics_comparison']['complete_cohort']
    assert 'match the covered rows' in evidence_tool_result(brief, detailed=False).content[0].text
