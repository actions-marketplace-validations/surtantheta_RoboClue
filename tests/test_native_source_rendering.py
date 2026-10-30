import pytest

from policy_doctor.cli import human
from policy_doctor.reports import markdown_report


@pytest.mark.parametrize('render', [human, markdown_report])
def test_source_report_names_sources_and_separates_execution_from_replay(render):
    criterion = {'status': 'error', 'code': 'source_contradiction',
                 'reason': 'Loaded processor assets differ from selected restoration assets.'}
    audit = {'kind': 'native_processor_source_audit', 'status': 'error',
        'source_criterion': criterion,
        'source_selection': {'selected_checkpoint': 'example/selected@pin',
            'consumer_argument': 'example/base', 'reference_argument': 'selected local checkpoint'},
        'source_binding_status': 'error', 'native_processing_status': 'complete',
        'captured_output_parity_status': 'error', 'effective_statistics_status': 'error',
        'native_replay_status': 'unknown',
        'normalizer_input_statistics': {
            'consumer': [{'observed_input_stat_keys': []}],
            'reference': [{'observed_input_stat_keys': ['observation.state']}]}}
    report = {'status': 'error', 'scope': 'Recorded native source evidence.',
        'summary': 'Explicit restoration criterion.', 'audits': [audit], 'findings': [],
        'inventory': {'captures': ['/capture']}, 'coverage': {}, 'intake_problems': [],
        'next_actions': [], 'case_recipe': None,
        'comparison': {'kind': 'native_processor_source_comparison',
            'first_supported_difference': criterion, 'reference_audit_status': 'pass',
            'same_correction_input': True, 'scope': 'Same criterion on both recorded views.'}}
    rendered = render(report)
    assert 'Source criterion: error' in rendered
    assert 'example/base' in rendered and 'example/selected@pin' in rendered
    assert 'Native preprocessing: complete' in rendered
    assert 'native numerical replay: unknown' in rendered
    assert 'explicit criterion not supplied' not in rendered
    assert 'named raw value' not in rendered
    assert 'None | None' not in rendered


def test_native_source_parent_human_does_not_dump_raw_worker_evidence():
    result = {'kind': 'native_processor_source_observation', 'status': 'error',
        'execution_status': 'complete', 'scope': 'Observed CPU preprocessing only.',
        'candidate': {'status': 'error',
            'source_criterion': {'status': 'error', 'reason': 'Selected assets differ.'},
            'native_processing_status': 'complete',
            'source_selection': {'selected_checkpoint': 'selected', 'consumer_argument': 'example/base'}},
        'corrected': {'status': 'pass', 'source_criterion': {'status': 'pass', 'reason': 'Selected assets match.'},
            'native_processing_status': 'complete'},
        'candidate_directory': '/evidence/candidate', 'corrected_directory': '/evidence/corrected'}
    rendered = human(result)
    assert 'Candidate source evidence:' in rendered and 'Corrected source evidence:' in rendered
    assert 'example/base' in rendered and 'Native execution: complete' in rendered
    assert 'policy-doctor investigate /evidence/candidate --reference /evidence/corrected' in rendered
    assert len(rendered) < 2500


def test_native_source_runtime_drift_has_no_processing_success_claim():
    rendered = human({'kind': 'native_processor_source_reproduction', 'status': 'unknown',
                      'verifier_matches': False, 'candidate_evaluated': False})
    assert 'Native execution: not evaluated' in rendered
    assert 'complete' not in rendered and 'verifier matches: False' in rendered
