"""Public headings and bounded previews must describe exactly what was assessed."""
import copy
import json
import shlex

import pytest

from policy_doctor.artifacts import Artifact, Snapshot
from policy_doctor.cli import human, main, parser
from policy_doctor.diagnostics import Report, _labels, _statistics_checks, dataset_statistics_findings

PUBLIC = {'prog': 'roboclue advanced', 'brand': 'RoboClue', 'command_prefix': ('roboclue', 'advanced')}


def test_canonical_help_and_usage_use_the_caller_entry_point(capsys):
    with pytest.raises(SystemExit) as error:
        main(['--help'], **PUBLIC)
    assert error.value.code == 0
    output = capsys.readouterr()
    assert 'usage: roboclue advanced' in output.out
    assert 'usage: policy-doctor' not in output.out
    with pytest.raises(SystemExit) as error:
        parser(prog='roboclue advanced').parse_args(['not-a-command'])
    assert error.value.code == 2
    assert 'usage: roboclue advanced' in capsys.readouterr().err


def test_legacy_help_stays_compatible(capsys):
    with pytest.raises(SystemExit):
        parser().parse_args(['--help'])
    assert 'usage: policy-doctor' in capsys.readouterr().out


def test_heading_and_command_change_without_altering_artifact_paths_or_reports():
    path = '/dataset/policy-doctor with spaces/case'
    report = {'kind': 'exported_incident_case', 'status': 'error', 'directory': path}
    before = copy.deepcopy(report)
    public = human(report, brand='RoboClue', command_prefix=('roboclue', 'advanced'))
    assert public.startswith('RoboClue incident case: error\n')
    rerun = next(line for line in public.splitlines() if line.startswith('Rerun: '))
    assert shlex.split(rerun.removeprefix('Rerun: ')) == ['roboclue', 'advanced', 'reproduce-case', path]
    assert report == before
    assert human(report).startswith('Policy Doctor incident case: error\n')


def test_canonical_human_brand_does_not_rewrite_embedded_legacy_provenance():
    raw = {'kind': 'unrecognized_report', 'status': 'unknown', 'findings': [],
           'artifacts': [{'path': '/policy-doctor with spaces/info.json'}],
           'source': {'engine': 'policy-deployment-doctor'}}
    before = copy.deepcopy(raw)
    output = human(raw, brand='RoboClue', command_prefix=('roboclue', 'advanced'))
    assert output.startswith('RoboClue: unknown')
    assert raw == before


def test_openpi_pass_is_labelled_delta_consistency_not_native_compatibility():
    report = {'kind': 'dataset_preflight', 'status': 'unknown', 'components': {
        'action_conventions': {'title': 'Action convention', 'status': 'unknown',
            'verdicts': {'consumer': 'pass', 'dataset_fingerprint': 'partly undecided',
                         'consumers': ["openpi config 'pi0_libero'"]}}}}
    output = human(report, brand='RoboClue')
    assert 'delta transform consistency: pass' in output
    assert 'consumer: pass' not in output
    assert 'keys, framework versions and native training compatibility are not established' in output
    assert report['components']['action_conventions']['verdicts']['consumer'] == 'pass'


def statistics_snapshot(tmp_path, width, scenario):
    key = 'action'
    zeros = [0.] * width
    values = {
        'constant': {'mean': zeros, 'std': zeros, 'min': zeros, 'max': zeros},
        'small': {'mean': zeros, 'std': [1e-8] * width},
        'collapsed': {'q01': zeros, 'q99': zeros},
        'invalid': {'mean': zeros, 'std': [-1.] * width},
        'healthy': {'mean': zeros, 'std': [1.] * width},
    }[scenario]
    stats = {key: values}
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(stats))
    return Snapshot('dataset', tmp_path, {'stats': stats},
        features={key: {'dtype': 'float32', 'shape': [width], 'names': [f'joint{i}' for i in range(width)]}},
        artifacts=[Artifact(str(path), 'dataset_stats', '0' * 64, path.stat().st_size)],
        json_documents={str(path): stats})


@pytest.mark.parametrize(('scenario', 'code'), [
    ('constant', 'dataset.constant_dimension'), ('small', 'dataset.small_spread'),
    ('collapsed', 'dataset.collapsed_quantiles'), ('invalid', 'dataset.statistics_invalid')])
@pytest.mark.parametrize('width', [1, 8, 9, 16, 22])
def test_statistics_preview_counts_all_dimensions_without_claiming_list_is_complete(tmp_path, scenario, code, width):
    snapshot = statistics_snapshot(tmp_path, width, scenario)
    before = copy.deepcopy(snapshot.config)
    findings = dataset_statistics_findings(snapshot)
    hit = next(item for item in findings if item['code'] == code)
    assert hit['status'] == 'warning'
    assert hit['indices'] == list(range(min(width, 16)))
    assert hit['indices_total'] == width
    assert hit['indices_omitted'] == max(0, width - 16)
    assert ('...' in hit['message']) is (width > 8)
    if width > 8:
        assert f'{width - 8} more; {width} total' in hit['message']
        assert f'{width - 1} (joint{width - 1})' not in hit['message']
    assert snapshot.config == before


def test_no_statistical_warning_is_invented_for_healthy_dimensions(tmp_path):
    assert dataset_statistics_findings(statistics_snapshot(tmp_path, 22, 'healthy')) == []


def test_short_label_list_is_not_marked_as_truncated():
    assert _labels([1, 3], None) == '1, 3'
    assert _labels([1], ['zero', 'one']) == '1 (one)'


def test_checkpoint_spread_preview_counts_are_not_lost(tmp_path):
    width = 22
    key = 'action'
    path = tmp_path / 'stats.safetensors'
    feature = {'type': 'ACTION', 'shape': [width]}
    step = {'name': 'unnormalizer_processor', 'state_file': str(path),
            'config': {'features': {key: feature}, 'norm_map': {'ACTION': 'MEAN_STD'}},
            'stats': {key: {'mean': [0.] * width, 'std': [0.] * width}}}
    checkpoint = Snapshot('checkpoint', tmp_path, {'type': 'act'}, features={key: feature}, post=[step],
        artifacts=[Artifact(str(path), 'processor_state', '0' * 64, 0)])
    report = Report()
    _statistics_checks(checkpoint, None, report)
    hit = next(f for f in report.findings if f.code == 'normalization.constant_dimension')
    assert hit.status == 'pass'  # Preserved conditional affine calculation, not a hardware pass.
    assert '14 more; 22 total' in hit.message
    evidence = hit.evidence[0]
    assert evidence['indices_total'] == 22 and evidence['indices_omitted'] == 6
    assert evidence['saved_values_omitted'] == 14
    assert evidence['inverse_scales'] == [0.] * 16


class FindingRecorder:
    def __init__(self):
        self.findings = []

    def add(self, code, status, message, *evidence, **kwargs):
        self.findings.append({'code': code, 'status': status, 'message': message, **kwargs})


@pytest.mark.parametrize('version', ['n15', 'n17'])
def test_missing_language_mapping_suggests_matching_config_before_dataset_edit(version):
    from policy_doctor import groot_contract, groot_contract_n15

    module = groot_contract_n15 if version == 'n15' else groot_contract
    report = FindingRecorder()
    module._language(report, {'annotation': {'existing': {'original_key': 'task_index'}}}, {}, set(),
                     {'language': {'keys': ['annotation.missing'], 'delta_indices': [0]}})
    finding = next(item for item in report.findings if item['code'] == 'groot.language_key')
    assert finding['status'] == 'error'
    assert finding['remedy'].startswith('First select or review a training config matching')
    assert 'Do not edit a public dataset merely to silence' in finding['remedy']
    assert 'source-backed mapping in a separate candidate' in finding['remedy']


@pytest.mark.parametrize('version', ['n15', 'n17'])
def test_missing_state_group_suggests_matching_config_before_dataset_edit(version):
    from policy_doctor import groot_contract, groot_contract_n15

    module = groot_contract_n15 if version == 'n15' else groot_contract
    report = FindingRecorder()
    prefix = 'state.' if version == 'n15' else ''
    action_prefix = 'action.' if version == 'n15' else ''
    module._config_keys(report, {'state': {'existing': {}}, 'action': {'existing': {}}, 'video': {}},
                        {'state': {'keys': [prefix + 'missing']},
                         'action': {'keys': [action_prefix + 'existing']}, 'video': {'keys': []}})
    finding = next(item for item in report.findings if item['code'] == 'groot.state_config')
    assert finding['status'] == 'error'
    assert finding['remedy'].startswith('First select or review a training config matching')
    assert 'Do not edit a public dataset merely to silence' in finding['remedy']
    assert 'source-backed state' in finding['remedy']


@pytest.mark.parametrize(('mean', 'std', 'low', 'high', 'constraint'), [
    (2., .1, 0., 1., 'mean_within_range'),
    (-2., .1, 0., 1., 'mean_within_range'),
    (.5, 0., 0., 1., 'zero_std_with_nonzero_range'),
    (.5, 0., .3, .3, 'mean_within_range'),
])
def test_impossible_summary_statistics_are_errors_without_claiming_row_mismatch(tmp_path, mean, std, low, high, constraint):
    snapshot = statistics_snapshot(tmp_path, 22, 'healthy')
    snapshot.config['stats']['action'] = {'mean': [mean] * 22, 'std': [std] * 22,
                                          'min': [low] * 22, 'max': [high] * 22}
    findings = dataset_statistics_findings(snapshot)
    hit = next(f for f in findings if f['code'] == 'dataset.statistics_inconsistent' and f['constraint'] == constraint)
    assert hit['status'] == 'error'
    assert hit['indices_total'] == 22 and hit['indices_omitted'] == 6
    assert '14 more; 22 total' in hit['message']
    assert 'Dataset rows were not compared' in hit['message']


@pytest.mark.parametrize(('mean', 'std', 'low', 'high'), [
    (.5, .1, 0., 1.), (0., 0., 0., 0.), (1. + 5e-7, .1, 0., 1.),
    (1., 0., 1. - 5e-7, 1.), (1e9 + 500., 10., 0., 1e9),
])
def test_consistent_or_rounding_scale_statistics_do_not_get_new_errors(tmp_path, mean, std, low, high):
    snapshot = statistics_snapshot(tmp_path, 1, 'healthy')
    snapshot.config['stats']['action'] = {'mean': [mean], 'std': [std], 'min': [low], 'max': [high]}
    assert not any(f['code'] == 'dataset.statistics_inconsistent' for f in dataset_statistics_findings(snapshot))
