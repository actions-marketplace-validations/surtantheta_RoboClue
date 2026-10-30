import json

import pytest

from policy_doctor.cases import export_sample_case, export_trajectory_case
from policy_doctor.demo import create_controller_demo
from policy_doctor.investigation import investigate_run


@pytest.mark.parametrize('sample', [True, False])
def test_recipient_discovery_keeps_existing_trajectory_cases_opaque(tmp_path, sample):
    native, case = tmp_path / 'native', tmp_path / 'case'
    create_controller_demo(str(native))
    if sample:
        export_sample_case(str(native / 'reference.json'), str(case), [.25])
    else:
        export_trajectory_case(str(native / 'reference.json'), str(native / 'candidate.json'), str(case), [.25])
    report = investigate_run(str(case))
    assert report['status'] == 'unknown' and not report['audits']
    assert report['inventory']['trajectory_cases'] == [str(case)]
    assert not report['inventory']['trajectories']
    assert any(a['tool'] == 'reproduce_trajectory_case' and a['arguments'] == {'directory': str(case)} for a in report['next_actions'])


def test_normal_run_manifest_still_receives_ordinary_classification(tmp_path):
    manifest = {'schema_version': 1, 'features': {'action': {'shape': [1], 'names': ['joint']}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    report = investigate_run(str(tmp_path))
    assert report['inventory']['deployments'] == [str(tmp_path / 'manifest.json')]
    assert not report['inventory']['retained_checks']
    assert report['coverage']['files_seen'] == 1
    assert report['coverage']['bytes_inspected'] == (tmp_path / 'manifest.json').stat().st_size
