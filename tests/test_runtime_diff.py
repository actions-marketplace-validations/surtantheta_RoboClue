import json

from test_runtime_capture import LoadedPipeline

from policy_doctor.capture import capture_runtime
from policy_doctor.runtime_diff import compare_runtime_captures


def test_loaded_override_diff_has_exact_statistics_evidence(case, tmp_path):
    config = json.loads((case['checkpoint'] / 'config.json').read_text())
    deployment = json.loads(case['deployment'].read_text())
    for name, shift in [('reference', 0.), ('candidate', 2.)]:
        pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
        for pipeline in (pre, post):
            for state in pipeline.states.values():
                state['action.mean'][0] += shift
        capture_runtime(tmp_path / name, policy_config=config, preprocessor=pre, postprocessor=post, deployment=deployment)
    report = compare_runtime_captures(str(tmp_path / 'reference'), str(tmp_path / 'candidate'))
    assert report['status'] == 'different'
    assert len(report['changes']) == 2
    assert all(c['pointer'].endswith('/statistics/action/mean') for c in report['changes'])
    assert all(c['evidence'][0]['tensor'] == 'action.mean' for c in report['changes'])
    assert report['reference_audit_status'] == report['candidate_audit_status'] == 'pass'
    assert compare_runtime_captures(str(tmp_path / 'reference'), str(tmp_path / 'reference'))['equal']


def test_declared_runtime_semantics_are_compared_but_not_certified(case, tmp_path):
    config = json.loads((case['checkpoint'] / 'config.json').read_text())
    deployment = json.loads(case['deployment'].read_text())
    for name, unit in [('reference', 'rad'), ('candidate', 'deg')]:
        pre, post = LoadedPipeline(case, 'pre'), LoadedPipeline(case, 'post')
        deployment['units'][0] = deployment['training_contract']['units'][0] = unit
        capture_runtime(tmp_path / name, policy_config=config, preprocessor=pre, postprocessor=post, deployment=deployment)
    report = compare_runtime_captures(str(tmp_path / 'reference'), str(tmp_path / 'candidate'))
    assert report['status'] == 'different'
    assert len(report['changes']) == 2
    assert all(c['pointer_kind'] == 'unverified_runtime_declaration' for c in report['changes'])
    assert all(c['evidence'][0]['path'].endswith('deployment.json') for c in report['changes'])
