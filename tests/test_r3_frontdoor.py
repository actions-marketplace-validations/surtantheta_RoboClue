"""Round-3 user-visible verdict regressions, independently of detector internals."""
import json

import pytest

from policy_doctor import product_cli


def conventions():
    return {'kind': 'dataset_preflight', 'status': 'unknown', 'components': {
        'video_index': {'status': 'unknown'},
        'action_conventions': {'status': 'warning', 'findings': [
            {'code': 'convention.fingerprint', 'status': 'unknown', 'message': 'Roll, pitch and yaw conventions are undecided.'},
            {'code': 'convention.rotation_declared_conflict', 'status': 'warning',
             'message': 'State labels rx, ry, rz, rw claim a quaternion, but observed values are not unit-norm.'},
        ], 'verdicts': {'consumer': 'pass'}},
    }, 'coverage': {'required_assessments': {'conventions': {
        'status': 'unknown', 'unresolved': ['Roll, pitch and yaw conventions are undecided.']}}}}


@pytest.mark.parametrize(('threshold', 'required', 'expected'), [
    ('warning', False, 1), ('warning', True, 1), ('unknown', False, 1),
    ('error', False, 0), ('error', True, 3), (None, False, 1), (None, True, 3),
])
def test_convention_warning_is_never_hidden_by_global_unknown(monkeypatch, capsys, threshold, required, expected):
    report = conventions()
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'conventions'))
    argv = ['check', 'DATASET']
    if threshold:
        argv += ['--fail-on', threshold]
    if required:
        argv += ['--require', 'conventions']
    assert product_cli.main(argv) == expected
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 3
    assert 'CONSISTENT' in out and 'WARNING' in out
    assert 'convention.rotation_declared_conflict' in out and 'not unit-norm' in out
    assert 'undecided' in out


def test_machine_convention_answer_scopes_original_consumer_verdict(tmp_path, monkeypatch):
    from policy_doctor import preflight

    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v2.1'}))
    original = conventions()
    monkeypatch.setattr(preflight, 'preflight', lambda *args, **kwargs: original)
    args = product_cli._parser().parse_args(['check', str(root), '--openpi-config', 'pi0_libero'])
    report, kind = product_cli._assess(args)
    assert kind == 'conventions'
    assert report['components']['action_conventions']['verdicts']['consumer'] == 'pass'
    scope = report['selected_assessment']['scope']
    assert 'delta-transform choice only' in scope and 'installed-loader compatibility' in scope
    assert report['selected_assessment']['status'] == 'warning'


def test_unversioned_groot_verdict_names_each_profile_and_keeps_native_failure(monkeypatch, capsys):
    report = {'kind': 'groot_dataset_contract', 'status': 'error', 'mode': 'both',
              'versions': {'n1.5': {'status': 'warning'}, 'n1.7': {'status': 'error'}},
              'headline': 'Fails on Isaac-GR00T N1.7 fine-tuning; no error on N1.5.',
              'findings': [{'status': 'error', 'code': 'groot.statistics',
                            'message': 'Fine-tuning cannot read the declared statistics.'}]}
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'groot'))
    assert product_cli.main(['check', 'DATASET']) == 1
    out = capsys.readouterr().out
    first = out.splitlines()[0]
    assert 'N1.5' in first and 'N1.7' in first and 'ERROR' in first and 'WARNING' in first
    assert '--groot-version' in out and 'No consumer version was selected' in out


@pytest.mark.parametrize('message', [
    {'previous': 19, 'current': 35, 'episode': 0},
    "{'previous': 19, 'current': 35, 'episode': 0}",
])
def test_dict_evidence_becomes_a_readable_gap(message):
    report = {'status': 'error', 'findings': [{'code': 'rows.index_gap', 'status': 'error', 'message': message}]}
    out = product_cli.render(report, source='DATASET', kind='dataset')
    line = out.splitlines()[1]
    assert '19' in line and '35' in line and 'episode 0' in line
    assert '{' not in line


def test_sim_metadata_pass_never_prints_python_dict_as_problem():
    report = {'status': 'pass', 'findings': [{'status': 'pass', 'code': 'robomimic.env_args',
                                           'message': "[source] env_args parsed: {'env_name': 'Pick', 'type': 2}."}]}
    out = product_cli.render(report, source='file.hdf5', kind='sim')
    assert 'metadata was parsed' in out and 'simulator' in out and '{' not in out


def test_saved_statistics_pass_explains_it_did_not_compare_rows():
    report = {'kind': 'dataset_health_card', 'status': 'pass', 'report': {'status': 'pass',
              'components': {'dataset_statistics': {'status': 'pass'}, 'numeric_rows': {'status': 'pass'}}}}
    out = product_cli.render(report, source='DATASET', kind='dataset')
    assert 'self-consistent' in out and 'not compared with dataset rows' in out
    assert 'robot identity' in out and 'selected metadata/row checks only' in out


def test_recovery_copy_separates_integrity_from_statistics_and_original():
    report = {'kind': 'recovery_copy', 'status': 'unknown', 'copy_created': True,
              'destination': 'NEW COPY', 'original_unchanged': True,
              'statistics': {'status': 'unknown'},
              'verification': {'recording_integrity': 'pass'}}
    out = product_cli.render(report, source='ORIGINAL', kind='recovery')
    assert 'Recovery copy: UNKNOWN; original unchanged' in out
    assert 'Recording integrity: PASS' in out and 'statistics: UNKNOWN' in out
    assert 'do not treat this copy as training/resume-ready' in out
    assert "roboclue check 'NEW COPY' --question recovery" in out
    report['statistics']['status'] = 'pass'
    report['status'] = 'pass'
    out = product_cli.render(report, source='ORIGINAL', kind='recovery')
    assert 'statistics: PASS' in out and 'Native reload, actual resume and training remain separate checks' in out


def test_recovery_statistics_limit_remains_visible_after_integrity_pass():
    report = {'kind': 'recording_salvage', 'status': 'pass', 'plan': {}}
    out = product_cli.render(report, source='COPY', kind='recovery')
    assert 'statistic values were not compared with dataset rows' in out
    report['plan']['stats_note'] = 'original aggregates include removed episodes'
    out = product_cli.render(report, source='COPY', kind='recovery')
    assert 'meta/stats.json can still include dropped episodes' in out


@pytest.mark.parametrize('options', [
    ['--lerobot-version', '0.6.2'],
    ['--write-copy', 'NEW', '--no-video-headers'],
    ['--write-copy', 'NEW', '--units'],
])
def test_consumer_version_and_copy_flags_cannot_silently_select_an_unrelated_check(tmp_path, capsys, options):
    source = tmp_path / 'dataset'
    (source / 'meta').mkdir(parents=True)
    (source / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v3.0'}))
    assert product_cli.main(['check', str(source), *options, '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'input_error'


def test_summary_keeps_complete_consequence_sentence_before_omission():
    report = {'status': 'error', 'findings': [{'status': 'error', 'code': 'example',
              'message': 'Do not resume this directory. ' + 'additional source detail ' * 200}]}
    out = product_cli.render(report, source='DATASET', kind='dataset')
    line = out.splitlines()[1]
    assert 'Do not resume this directory.' in line and '--explain' in line
    assert not line.endswith(('det...', 'deta...', 'source...'))


def test_arbitrary_json_is_not_assumed_to_be_training_config(tmp_path, capsys):
    source = tmp_path / 'vqa.json'
    source.write_text(json.dumps({'question': 'Which object moved?', 'answer': 'cup'}))
    assert product_cli.main(['check', str(source)]) == 3
    out = capsys.readouterr().out
    assert 'semantic role' in out and '--help' in out and 'REVIEWED_CONFIG' not in out


def test_machine_input_error_has_json_on_stdout_and_a_failing_exit(tmp_path, capsys):
    assert product_cli.main(['check', str(tmp_path / 'missing'), '--json']) == 2
    io = capsys.readouterr()
    report = json.loads(io.out)
    assert report['status'] == 'input_error' and report['error']['code'] == 'invalid_input'
    assert 'Not found' in report['error']['message'] and not io.err


@pytest.mark.parametrize('info_alias', [False, True])
def test_saving_a_report_cannot_modify_the_original_dataset(tmp_path, capsys, info_alias):
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    info = root / 'meta/info.json'
    original = json.dumps({'codebase_version': 'v3.0'}).encode()
    info.write_bytes(original)
    destination = root / 'report.json'
    source = info if info_alias else root
    assert product_cli.main(['check', str(source), '--output', str(destination), '--json']) == 2
    report = json.loads(capsys.readouterr().out)
    assert 'outside the input directory' in report['error']['message']
    assert not destination.exists() and info.read_bytes() == original


def test_explicit_unknown_version_is_named_without_inventing_a_missing_command():
    report = {'kind': 'declared_robot_interface', 'status': 'unknown', 'components': {
        'units': {'status': 'unknown', 'manifest': {'deployment': {'robot_type': 'so101_follower', 'arms': {}},
                  'features': {'action': {'coordinates': [{'unit': 'RANGE_M100_100'}]}}},
                  'consumer_version': {'declared_version': '999.0', 'tested_profile': None,
                      'comparison_allowed': False, 'reason': 'No qualified source profile for this caller version.'}},
        'camera_keys': {'requested': False}}}
    out = product_cli.render(report, source='DATASET', kind='units')
    assert 'INCOMPLETE' in out and '999.0' in out and '-100..100' in out
    assert 'No deployment command was supplied' not in out
    assert 'verified units comparison is not available' in out


def test_explicit_known_profile_still_names_collection_role_and_unverified_fleet():
    report = {'kind': 'declared_robot_interface', 'status': 'pass', 'components': {
        'units': {'status': 'pass', 'manifest': {'deployment': {'robot_type': 'so101_follower', 'arms': {}}},
                  'consumer_version': {'declared_version': '0.6.2', 'tested_profile': 'LeRobot146d847',
                      'comparison_allowed': True, 'command_role': 'collection_only'}},
        'camera_keys': {'requested': False}}}
    out = product_cli.render(report, source='DATASET', kind='units')
    assert 'collects data' in out and 'does not run a trained policy' in out and '0.6.2' in out
    assert '146d847' not in out and report['components']['units']['consumer_version']['tested_profile'] == 'LeRobot146d847'


@pytest.mark.parametrize('suffix', ['json', 'yaml'])
def test_frontdoor_accepts_the_documented_robot_config_formats(tmp_path, monkeypatch, capsys, suffix):
    from policy_doctor import units_manifest as engine

    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v2.1', 'features': {}}))
    config = tmp_path / ('robot.' + suffix)
    declaration = {'robot': {'type': 'so101_follower', 'port': '/dev/ttyACM0', 'use_degrees': False}}
    if suffix == 'json':
        config.write_text(json.dumps(declaration))
    else:
        import yaml

        config.write_text(yaml.safe_dump(declaration))
    report = {'kind': 'units_manifest', 'status': 'pass', 'manifest': {
        'deployment': {'robot_type': 'so101_follower', 'arms': {}}, 'features': {}}}
    monkeypatch.setattr(engine, 'units_manifest', lambda *args, **kwargs: report)
    assert product_cli.main(['check', str(root), '--robot-config', str(config), '--lerobot-version', '0.6.1', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['kind'] == 'declared_robot_interface'


def test_warning_gate_honors_component_status_when_examples_were_bounded(monkeypatch, capsys):
    report = {'kind': 'declared_robot_interface', 'status': 'unknown', 'components': {
        'units': {'kind': 'units_manifest', 'status': 'warning', 'findings': [], 'manifest': {}},
        'camera_keys': {'status': 'unknown', 'requested': True, 'findings': []},
    }}
    monkeypatch.setattr(product_cli, '_assess', lambda args: (report, 'units'))
    assert product_cli.main(['check', 'DATASET', '--fail-on', 'warning', '--json']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'


def test_dataset_warning_survives_an_unrelated_video_gap():
    warning = {'code': 'dataset.constant_dimension', 'status': 'warning', 'message': 'A saved action dimension has zero spread; review whether that is intentional.'}
    report = {'kind': 'dataset_health_card', 'status': 'unknown', 'report': {'status': 'unknown',
              'components': {'dataset_statistics': {'status': 'warning', 'findings': [warning]},
                             'video_index': {'status': 'unknown'}}}}
    out = product_cli.render(report, source='DATASET', kind='dataset')
    assert 'WARNING' in out and 'dataset.constant_dimension' in out and 'zero spread' in out
    assert 'video index' in out and 'INCOMPLETE' in out


def test_empty_examples_do_not_invent_camera_match_or_hide_convention_warning():
    camera = {'kind': 'declared_robot_interface', 'status': 'unknown', 'components': {
        'units': {'status': 'pass', 'manifest': {'deployment': {'robot_type': 'so101_follower', 'arms': {}}}},
        'camera_keys': {'requested': True, 'status': 'unknown', 'findings': []},
    }}
    out = product_cli.render(camera, source='DATASET', kind='units')
    assert 'camera names incomplete' in out
    assert 'camera keys match' not in out and 'Camera-key assessment: UNKNOWN' in out
    convention = conventions()
    convention['components']['action_conventions']['findings'] = []
    out = product_cli.render(convention, source='DATASET', kind='conventions')
    assert 'WARNING' in out and 'omitted' in out


def test_bounded_component_severity_is_still_visible_in_generic_and_sim_answers():
    generic = {'kind': 'dataset_health_card', 'status': 'unknown', 'report': {'status': 'unknown',
               'components': {'dataset_statistics': {'status': 'error', 'findings': []},
                              'video_index': {'status': 'unknown'}}}}
    out = product_cli.render(generic, source='DATASET', kind='dataset')
    assert 'ERROR' in out and 'INCOMPLETE' in out and '--explain' in out
    sim = {'status': 'warning', 'findings': [
        {'code': 'isaaclab.conversion_quaternion_order', 'status': 'pass'},
        {'code': 'isaaclab.quaternions_outside_root_pose', 'status': 'warning',
         'message': 'A separate end-effector quaternion convention remains unverified.'},
    ]}
    out = product_cli.render(sim, source='recording.hdf5', kind='sim')
    assert 'CONSISTENT' in out and 'WARNING' in out and 'end-effector' in out


def test_mcp_health_recap_names_statistics_scope_without_rewriting_evidence():
    from policy_doctor.mcp_presentation import evidence_tool_result

    report = {'kind': 'dataset_health_card', 'status': 'pass', 'card_id': 'original-identity',
              'components': {'dataset_statistics': {'status': 'pass'}}}
    response = evidence_tool_result(report, detailed=False)
    text = response.content[0].text
    assert 'self-consistency only' in text and 'not compared' in text
    assert response.structuredContent == report and 'scope' not in report['components']['dataset_statistics']
