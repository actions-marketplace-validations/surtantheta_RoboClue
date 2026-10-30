"""Bounded presentation controls, using generated evidence rather than private data."""
import copy
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.first_use_views import MAX_VIEW_BYTES, flight_view, health_view
from policy_doctor.flight_recorder import FlightRecorder, triage_flight_recording


def card():
    finding = {'code': 'video.interval_past_end', 'status': 'error', 'message': 'Past end', 'episodes': [
        {'episode_index': i, 'video_key': 'front', 'file': f'video-{i}.mp4',
         'episode_from_s': 2., 'episode_last_frame_s': 3., 'file_last_frame_s': 1.}
        for i in range(6)]}
    return {'kind': 'dataset_health_card', 'status': 'error',
            'request': {'source': 'hf://example/generated', 'config': {'max_bytes': 1024}},
            'source_revision': {'kind': 'hub_commit', 'value': 'a' * 40},
            'report': {'kind': 'dataset_preflight', 'status': 'error', 'counts': {'error': 1},
                       'findings': [finding], 'components': {
                           'episode_windows': {'findings': [finding]},
                           'video_index': {'findings': [copy.deepcopy(finding)]}},
                       'provenance': [{'source': 'hf://example/generated', 'downloaded_bytes': 500,
                                       'video_headers': {'bytes': 100}}]}}


def test_health_deduplicates_associations_preserves_status_counts_and_input():
    report = card()
    original = copy.deepcopy(report)
    view = health_view(report)
    assert report == original
    assert view['overall_status'] == 'error'
    assert view['available_unique_issues'] == 6
    assert len(view['issues']) == 4 and view['issues_omitted'] == 2
    first = view['issues'][0]
    assert first['episode'] == 0 and first['camera'] == 'front' and first['source_file'] == 'video-0.mp4'
    assert first['conflicting_values'] == {'episode_from_s': 2., 'episode_last_frame_s': 3., 'file_last_frame_s': 1.}
    assert 'verify' in first['action'] and 'No repair was applied' in first['action']
    assert view['source_revision'] == 'a' * 40
    assert view['byte_accounting']['combined_recorded_bytes'] == 600


def test_health_preserves_conflicting_evidence_and_reports_unknown_bytes():
    report = card()
    report['report']['components']['video_index']['findings'][0]['episodes'][0]['file_last_frame_s'] = .5
    report['report']['components']['video_index']['findings_truncated'] = True
    report['report'].pop('provenance')
    view = health_view(report)
    assert view['available_unique_issues'] == 7
    assert view['source_findings_truncated'] is True
    assert view['byte_accounting']['combined_recorded_bytes'] is None
    assert view['byte_accounting']['video_header_range_bytes'] is None


def test_health_deduplicates_recorded_provenance_bytes():
    report = card()
    report['report']['provenance'] *= 2
    assert health_view(report)['byte_accounting']['combined_recorded_bytes'] == 600


def test_health_surfaces_actual_episode_coverage_reason_without_mutating_card(tmp_path):
    pytest.importorskip('pyarrow.parquet')
    from policy_doctor.windows import audit_episode_windows

    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v2.1'}))
    audit = audit_episode_windows(tmp_path, episode_index=0, anchor_index=0, delta_indices=[0])
    finding, = audit['findings']
    assert finding == {
        'code': 'episode.coverage_unknown', 'status': 'unknown',
        'reason': 'Only native v3 metadata with the standard data_path template is supported'}
    report = card()
    report.update(status='unknown', card_id='retained-health-card')
    report['report'].update(status='unknown', counts={'unknown': 1},
                            findings=[finding], components={'episode_windows': audit})
    original = copy.deepcopy(report)

    view = health_view(report)

    assert view['overall_status'] == 'unknown'
    assert view['available_unique_issues'] == 1
    assert view['issues'][0]['message'] == finding['reason']
    assert view['issues'][0]['status'] == 'unknown'
    assert report == original


@pytest.mark.parametrize('message', [None, 'Authoritative message'])
def test_health_reason_fallback_preserves_message_precedence_and_distinct_evidence(message):
    report = card()
    findings = [{'code': 'episode.coverage_unknown', 'status': 'unknown', 'message': message,
                 'reason': reason} for reason in ['First coverage reason', 'Second coverage reason']]
    report['report']['components'] = {'episode_windows': {'findings': findings}}
    view = health_view(report)
    assert [issue['message'] for issue in view['issues']] == (
        ['First coverage reason', 'Second coverage reason'] if message is None else [message])


@pytest.mark.parametrize('text', ['x', '\u0000', '界', '😀'])
def test_compact_outputs_have_strict_byte_bounds_with_long_text(text):
    report = card()
    report['request']['source'] = text * 10000
    for component in report['report']['components'].values():
        episodes = component['findings'][0]['episodes']
        for index in range(2):
            episodes.append({**episodes[0], 'episode_index': 6 + index})
        for index, episode in enumerate(episodes):
            episode['file'] = text * 10000 + str(index)
            episode['video_key'] = text * 10000
    health = health_view(report, max_rows=8)
    assert len(health['issues']) == 8
    assert len(health['source']) <= 256
    assert len(json.dumps(health, ensure_ascii=True).encode()) <= MAX_VIEW_BYTES
    findings = [{'code': 'tracking', 'status': 'pass', 'event_ids': [text * 10000] * 10,
                 'details': {'comparison_boundary': 'request_to_return', 'joint': text * 10000,
                             'unit': text * 10000, 'absolute_residual': 0., 'threshold': 0.}}] * 50
    view = flight_view({'kind': 'flight_recording_triage', 'status': 'unknown', 'findings': findings}, max_rows=8)
    assert len(json.dumps(view, ensure_ascii=True).encode()) <= MAX_VIEW_BYTES
    assert len(view['software_comparisons']) == 8 and view['software_comparisons_omitted'] == 42


@pytest.mark.parametrize('bound', [0, 9, True, '4', None])
def test_invalid_row_bounds_refused(bound):
    with pytest.raises(InputError, match='max_rows'):
        health_view(card(), max_rows=bound)
    with pytest.raises(InputError, match='max_rows'):
        flight_view({}, max_rows=bound)


@pytest.mark.parametrize('function', [flight_view, health_view])
def test_partial_concise_report_refused(function):
    with pytest.raises(InputError, match='complete'):
        function({'kind': 'flight_recording_triage', 'findings_shown': []})


def test_legacy_tracking_cannot_be_relabelled_from_stream_id_or_timestamp(tmp_path):
    path = tmp_path / 'flight.jsonl'
    vector = {'joint_names': ['axis.pos'], 'units': ['declared_unit'], 'values': [.5], 'semantics': 'position'}
    with FlightRecorder(path, clocks={'clock': {'unit': 's', 'domain': 'declared'}},
                        provenance={'kind': 'generated_control'}) as recorder:
        recorder.command('send-request:1', timestamp=0., clock='clock', stream='send_action.request', **vector)
        recorder.feedback('send-return:2', command_id='send-request:1', timestamp=0., clock='clock',
                          stream='send_action.return', **vector)
    report = triage_flight_recording(path, tracking_tolerance={'axis.pos': {'unit': 'declared_unit', 'absolute': .01}})
    tracking = [f for f in report['findings'] if f['code'] == 'tracking']
    assert len(tracking) == 1 and tracking[0]['status'] == 'pass'
    assert 'comparison_boundary' not in tracking[0]['details']
    view = flight_view(report)
    assert view['software_comparisons'] == []
    assert view['physical_feedback_association']['status'] == 'unknown'


def test_software_counts_cover_omitted_rows_and_prioritize_warning():
    findings = [{'code': 'tracking', 'status': status, 'event_ids': ['request', 'return'],
                 'details': {'comparison_boundary': 'request_to_return', 'absolute_residual': residual}}
                for status, residual in [('pass', 0.)] * 12 + [('warning', 1.)]]
    report = {'kind': 'flight_recording_triage', 'status': 'warning', 'findings': findings}
    original = copy.deepcopy(report)
    view = flight_view(report)
    assert report == original
    assert view['software_comparisons'][0]['status'] == 'warning'
    assert view['software_counts'] == {'pass': 12, 'warning': 1, 'unknown': 0}
    assert view['software_comparisons_omitted'] == 9


@pytest.mark.parametrize('point,association,verified,expected', [
    ('before_original_send_action', 'same_original_send_action_call_return', False, 'request_to_return'),
    ('before_original_send_action', 'same_original_send_action_call_return', None, None),
    ('before_original_send_action', 'same_original_send_action_call_return', True, None),
    ('caller_policy_output', 'explicit_caller_policy_to_request_not_physical_feedback', None, 'policy_to_request'),
    ('caller_policy_output', 'unknown', None, None),
    ('unspecified', 'same_original_send_action_call_return', False, None),
])
def test_boundary_requires_explicit_metadata_and_keeps_numeric_verdict(tmp_path, point, association, verified, expected):
    path = tmp_path / 'metadata-control.jsonl'
    vector = {'joint_names': ['axis'], 'units': ['u'], 'values': [1.], 'semantics': 'position'}
    metadata = {'association': association}
    if verified is not None:
        metadata['physical_delivery_verified'] = verified
    with FlightRecorder(path, clocks={'clock': {'unit': 's', 'domain': 'declared'}},
                        provenance={'kind': 'generated_control'}) as recorder:
        recorder.command('first', timestamp=0., clock='clock', stream='unrelated-a',
                         metadata={'observation_point': point}, **vector)
        recorder.feedback('second', command_id='first', timestamp=0., clock='clock', stream='unrelated-b',
                          metadata=metadata, **{**vector, 'values': [2.]})
    report = triage_flight_recording(path, tracking_tolerance={'axis': {'unit': 'u', 'absolute': .1}})
    finding, = [f for f in report['findings'] if f['code'] == 'tracking']
    assert finding['status'] == 'warning'
    assert finding['details']['absolute_residual'] == 1. and finding['details']['threshold'] == .1
    assert finding['details'].get('comparison_boundary') == expected


@pytest.mark.parametrize('raw_status', ['pass', 'unknown', 'warning', 'error'])
def test_no_declared_videos_display_na_separately_from_raw_status(raw_status):
    report = card()
    report['report']['components']['video_index'] = {
        'status': raw_status, 'coverage': {'status': 'not_applicable', 'declared_video_features': 0, 'files_checked': 0},
        'findings': []}
    original = copy.deepcopy(report)
    view = health_view(report)
    assert report == original
    assert view['video_index']['display_status'] == ('N/A' if raw_status == 'pass' else raw_status.upper())
    assert view['video_index']['reported_status'] == raw_status
    assert view['video_index']['files_checked'] == 0
    if raw_status == 'pass':
        assert 'no video content was validated' in view['video_index']['interpretation']
    assert view['overall_status'] == report['status']


@pytest.mark.parametrize('coverage', [{}, {'declared_video_features': 0}, {'status': 'not_applicable'},
    {'status': 'not_applicable', 'declared_video_features': False},
    {'status': 'unknown', 'declared_video_features': 0},
    {'status': 'complete', 'declared_video_features': 1, 'files_checked': 1}])
def test_video_na_requires_explicit_not_applicable_and_zero_declared_features(coverage):
    report = card()
    report['report']['components']['video_index'].update(status='pass', coverage=coverage)
    view = health_view(report)
    assert view['video_index']['display_status'] == 'PASS'
    assert view['video_index']['reported_status'] == 'pass'


@pytest.mark.parametrize('enabled', [False, True])
def test_contract_enabled_is_distinct_from_absent_recipe_selectors(enabled):
    report = card()
    report['request']['config'].update(groot_contract=enabled, groot_modality_config=None,
                                      groot_config_name=None, groot_version=None)
    report['report']['coverage'] = {'groot_contract_requested': False}
    report['report']['components']['groot_contract'] = {'status': 'error', 'findings': []}
    original = copy.deepcopy(report)
    view = health_view(report)
    assert report == original
    assert view['groot_contract']['enabled'] is enabled
    assert view['groot_contract']['recipe_selectors_supplied'] is False
    assert view['groot_contract']['modality_config_supplied'] is False
    assert view['groot_contract']['reported_status'] == 'error'
    assert view['overall_status'] == 'error'


@pytest.mark.parametrize('selector,value', [('groot_version', 'n1.7'), ('groot_config_name', 'selected'),
                                          ('groot_modality_config', '/selected/config.py')])
def test_contract_recipe_selection_comes_from_actual_card_config(selector, value):
    report = card()
    report['request']['config'].update(groot_contract=True, groot_modality_config=None,
                                      groot_config_name=None, groot_version=None)
    report['request']['config'][selector] = value
    report['request']['consumer'] = 'arbitrary label'
    selection = health_view(report)['groot_contract']
    assert selection['enabled'] is True and selection['recipe_selectors_supplied'] is True
    assert selection['modality_config_supplied'] is (selector == 'groot_modality_config')
    assert selection['selected_version'] == (value if selector == 'groot_version' else None)


def test_contract_selection_is_unknown_without_recorded_settings_even_if_component_exists():
    report = card()
    report['request']['consumer'] = 'groot'
    report['report']['components']['groot_contract'] = {'status': 'pass', 'findings': []}
    selection = health_view(report)['groot_contract']
    assert selection['enabled'] is None and selection['recipe_selectors_supplied'] is None
    assert selection['reported_status'] == 'pass'


def test_plain_preflight_legacy_recipe_flag_does_not_infer_contract_enablement():
    report = card()['report']
    report['coverage'] = {'groot_contract_requested': False}
    selection = health_view(report)['groot_contract']
    assert selection['enabled'] is None and selection['recipe_selectors_supplied'] is False


@pytest.mark.parametrize('value', ['true', 0, 1, False, None])
def test_contract_enablement_never_coerces_nonboolean_values(value):
    report = card()
    report['request']['config']['groot_contract'] = value
    assert health_view(report)['groot_contract']['enabled'] is (False if value is False else None)
