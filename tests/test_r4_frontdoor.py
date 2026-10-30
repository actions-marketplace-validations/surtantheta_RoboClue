"""Public intent gates and readable answers for the round-four incidents."""
import json
import shlex

import pytest

from policy_doctor import product_cli
from policy_doctor.console_text import display_command


@pytest.mark.parametrize(('status', 'threshold', 'expected'), [
    ('warning', None, 1), ('warning', 'error', 0), ('warning', 'warning', 1),
    ('unknown', None, 3), ('pass', None, 0), ('error', None, 1),
])
def test_default_gate_never_clears_a_warning(monkeypatch, capsys, status, threshold, expected):
    report = {'status': status, 'findings': [{'status': status, 'message': 'Review the declared quaternion order.'}]}
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'sim'))
    options = ['--fail-on', threshold] if threshold else []
    assert product_cli.main(['check', 'recording.hdf5', *options, '--json']) == expected
    assert json.loads(capsys.readouterr().out)['status'] == status


@pytest.mark.parametrize('threshold', [None, 'error', 'warning'])
def test_unselected_disagreeing_groot_profiles_need_input_not_an_assumed_consumer(monkeypatch, capsys, threshold):
    report = {'status': 'error', 'mode': 'both', 'versions': {
        'n1.5': {'status': 'warning'}, 'n1.7': {'status': 'error'}},
        'findings': [{'status': 'error', 'code': 'groot.statistics', 'message': 'N1.7 cannot read these statistics.'}],
        'selected_assessment': {'name': 'groot_consumer_selection', 'status': 'unknown'}}
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'groot'))
    options = ['--fail-on', threshold] if threshold else []
    assert product_cli.main(['check', 'dataset', *options]) == 3
    text = capsys.readouterr().out
    assert 'N1.5' in text and 'N1.7' in text and 'Choose' in text
    assert '--groot-version' in text


def test_next_flags_are_idempotent_and_do_not_touch_literal_suffix():
    arguments = ['roboclue', 'check', 'dataset', '--video-headers', '--', '--explain']
    result = shlex.split(product_cli._with_options(arguments, '--video-headers', '--explain'))
    assert result[:result.index('--')].count('--video-headers') == 1
    assert result[:result.index('--')].count('--explain') == 1
    assert result[-1] == '--explain'


def test_nested_command_quoting_keeps_task_readable_and_preserves_argv():
    command = "lerobot-rollout --robot.type=so101_follower --task='pick up the cup'"
    arguments = ['roboclue', 'check', 'dataset', '--robot-command', command]
    text = display_command(arguments)
    assert shlex.split(text) == arguments
    assert "'\"'\"'" not in text


def test_pass_has_an_explanation_rather_than_a_problem():
    text = product_cli.render({'status': 'pass'}, source='dataset', kind='dataset')
    assert '\nDetails:' in text and '\nProblem:' not in text
    assert len(text.splitlines()[1]) <= 300


def test_partial_saved_stats_never_omit_the_uncompared_rows_scope():
    report = {'kind': 'dataset_health_card', 'status': 'unknown', 'report': {'components': {
        'dataset_statistics': {'status': 'pass'}, 'video_index': {'status': 'unknown'}}}}
    text = product_cli.render(report, source='dataset', kind='dataset')
    assert 'self-consistent' in text and 'not compared with dataset rows' in text


def test_actual_numeric_comparison_is_named_separately_from_saved_self_consistency():
    report = {'kind': 'dataset_health_card', 'status': 'unknown', 'report': {'components': {
        'dataset_statistics': {'status': 'pass'}, 'video_index': {'status': 'unknown'},
        'numeric_rows': {'status': 'pass', 'statistics_comparison': {
            'status': 'pass', 'compared_features': ['action'], 'rows': 165}}}}}
    text = product_cli.render(report, source='dataset', kind='dataset')
    assert 'match 165 rows' in text and 'not compared with dataset rows' not in text


def test_precision_ambiguity_does_not_erase_completed_comparisons_or_claim_a_pass():
    report = {'kind': 'dataset_health_card', 'status': 'unknown', 'report': {'components': {
        'dataset_statistics': {'status': 'pass'},
        'numeric_rows': {'status': 'unknown', 'statistics_comparison': {
            'status': 'unknown', 'compared_features': ['action'], 'rows': 103,
            'compared_metrics': ['min', 'max', 'mean'], 'unassessed_metrics': {'action': ['std']},
            'precision_unassessed': [{'feature': 'action', 'metric': 'std'}]}}}}}
    text = product_cli.render(report, source='dataset', kind='dataset')
    assert 'Some numeric statistics match the rows' in text
    assert 'rounding leaves standard deviation unverified' in text
    assert 'not compared with dataset rows' not in text and 'match 103 rows' not in text


def test_statistics_scope_names_only_metrics_actually_compared():
    from policy_doctor.console_text import statistics_scope

    text = statistics_scope({'status': 'pass', 'rows': 12, 'compared_features': ['action'],
                             'compared_metrics': ['min', 'max']})
    assert '(min, max)' in text and 'mean' not in text and 'std' not in text


def test_recovery_leads_with_resume_hazard_before_a_secondary_unreadable_file():
    report = {'status': 'error', 'findings': [
        {'status': 'error', 'code': 'salvage.file_unreadable', 'message': 'Episode 2 cannot be read.'},
        {'status': 'error', 'code': 'salvage.resume_hazard', 'message': 'Episode indices will be reused.'}],
        'episodes': {'keep': [[0, 1]]},
        'plan': {'validity': {'current_directory_resume_status': 'unsafe'}}}
    text = product_cli.render(report, source='dataset', kind='recovery')
    assert 'Episode indices will be reused' in text and 'Episode 2' not in text


def test_metadata_total_typo_is_not_described_as_an_established_crash():
    report = {'status': 'error', 'assessment_kind': 'metadata_totals_mismatch', 'findings': [
        {'status': 'error', 'code': 'salvage.resume_hazard', 'message': 'Declared total differs.'}],
        'episodes': {'keep': [[0, 9]]},
        'plan': {'validity': {'current_directory_resume_status': 'unsafe'}}}
    text = product_cli.render(report, source='dataset', kind='recovery')
    assert 'Saved totals disagree' in text and 'crashed recording was not established' in text


def test_recovery_pass_does_not_say_compared_statistics_were_uncompared():
    report = {'status': 'pass', 'findings': [], 'episodes': {'keep': [[0, 9]]},
              'statistics_comparison': {'status': 'pass', 'rows': 165,
                                       'compared_features': ['action'], 'compared_metrics': ['min', 'max', 'mean', 'std']}}
    text = product_cli.render(report, source='dataset', kind='recovery')
    assert 'match 165 rows' in text and 'not compared' not in text


def test_openpi_raw_key_failure_cannot_be_waived_by_a_consistent_delta_transform(monkeypatch, capsys):
    report = {'status': 'unknown', 'components': {
        'action_conventions': {'status': 'pass', 'verdicts': {'consumer': 'pass'}, 'findings': []},
        'openpi_context': {'status': 'error', 'missing_keys': ['image', 'wrist_image'],
                          'dataset_codebase_version': 'v3.0', 'findings': [
                              {'status': 'error', 'code': 'openpi.raw_key_missing', 'message': 'Missing raw input keys.'}]}}}
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'conventions'))
    assert product_cli.main(['check', 'dataset']) == 1
    text = capsys.readouterr().out
    assert 'input keys: MISMATCH' in text and 'Delta-transform choice: CONSISTENT' in text
    assert 'image, wrist_image' in text and 'matching' in text
    assert report['components']['action_conventions']['verdicts']['consumer'] == 'pass'


def test_openpi_unknown_raw_keys_do_not_get_a_default_pass(monkeypatch, capsys):
    report = {'status': 'unknown', 'components': {
        'action_conventions': {'status': 'pass', 'verdicts': {'consumer': 'pass'}, 'findings': []},
        'openpi_context': {'status': 'unknown', 'findings': []}}}
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'conventions'))
    assert product_cli.main(['check', 'dataset']) == 3
    text = capsys.readouterr().out
    assert 'input keys: INCOMPLETE' in text and 'not established' in text


@pytest.mark.parametrize(('version', 'expected'), [(None, 3), ('0.6.1', 0), ('999.0', 3)])
def test_branded_advanced_preflight_enforces_actual_consumer_version(tmp_path, capsys, version, expected):
    import test_units_manifest as units_fixtures

    from policy_doctor.big_bets_frontdoor import main as advanced_main

    root = units_fixtures.so_dataset(tmp_path, 'm100')
    arguments = ['preflight', str(root), '--robot-command', units_fixtures.command('--robot.use_degrees=false'),
                 '--require', 'deployment-units', '--no-video-headers', '--no-groot-contract', '--json']
    if version:
        arguments += ['--lerobot-version', version]
    assert advanced_main(arguments) == expected
    report = json.loads(capsys.readouterr().out)
    assert report['components']['units']['consumer_version']['declared_version'] == version
    assert report['coverage']['required_assessments']['deployment-units']['status'] == ('pass' if expected == 0 else 'unknown')


def test_explicit_video_headers_reach_auto_selected_recovery(tmp_path, monkeypatch):
    from policy_doctor import salvage

    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v3.0'}))
    observed = []
    def check(source, **options):
        observed.append(options['video_headers'])
        return {'status': 'error', 'findings': [{'status': 'error', 'code': 'salvage.resume_hazard'}]}
    monkeypatch.setattr(salvage, 'check_salvage', check)
    args = product_cli._parser().parse_args(['check', str(root), '--video-headers'])
    _, kind = product_cli._assess(args)
    assert kind == 'recovery' and observed == [True]
