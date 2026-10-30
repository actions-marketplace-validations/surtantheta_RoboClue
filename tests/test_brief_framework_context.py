"""The deployment brief remains bounded with unusual installed metadata."""
from __future__ import annotations

import json

import test_units_manifest as fixtures

from policy_doctor import framework_versions
from policy_doctor.deployment_answer import (
    BRIEF_LIMIT,
    BRIEF_RESULT_LIMIT,
    deployment_answer,
    deployment_tool_result,
)
from policy_doctor.preflight import preflight


def test_long_local_version_is_omitted_without_losing_the_error(tmp_path, monkeypatch):
    root = fixtures.so_dataset(tmp_path, 'm100')
    report = preflight(str(root), robot_command=fixtures.command(), video_headers=False, groot_contract=False)
    original_version = '\U0001f9be' * 200
    metadata = {'frameworks': [{'id': 'lerobot', 'version': original_version, 'source_identity': 'not_revalidated'}]}
    monkeypatch.setattr(framework_versions, 'collect_framework_versions', lambda: metadata)
    answer = deployment_answer(report, source=str(root), robot_command=fixtures.command())
    context = answer['framework_context']
    assert 'version' not in context  # Unrelated local metadata is not a consumer version declaration.
    assert context['runtime_revalidated'] is False and 'consumer environment unverified' in context['environment']
    assert metadata['frameworks'][0]['version'] == original_version
    assert answer['status'] == 'error'
    assert len(json.dumps(answer).encode()) <= BRIEF_LIMIT
    result = deployment_tool_result(answer)
    assert len(json.dumps(result.model_dump(mode='json', exclude_none=True)).encode()) <= BRIEF_RESULT_LIMIT
    assert result.structuredContent['status'] == 'error'
    assert 'error' in result.content[0].text.lower()


def test_overlong_recap_does_not_force_an_unbounded_envelope():
    answer = {'kind': 'deployment_units_answer', 'status': 'pass', 'scope': 'Scoped units only.',
              'components': {'units': {'status': 'pass', 'compared_with': 'robot',
                  'detected_units': ['x' * 1300], 'declared_units': ['x' * 1300]}},
              'framework_context': {'source_identity': 'not_installed'},
              'full_report_sha256': 'a' * 64, 'details': "Repeat with response_format='detailed'."}
    result = deployment_tool_result(answer)
    assert result.structuredContent['status'] == 'unknown'
    assert result.structuredContent['report_status'] == 'pass'
    assert result.structuredContent['answer_status'] == 'details_required'
    assert 'No concise approval' in result.structuredContent['scope']
    assert len(json.dumps(result.model_dump(mode='json', exclude_none=True)).encode()) <= BRIEF_RESULT_LIMIT
