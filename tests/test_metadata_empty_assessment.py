"""An empty declared interface is not evidence of metadata compatibility."""

import asyncio
import json

import pytest

from policy_doctor.cli import main
from policy_doctor.demo import create_policy_demo
from policy_doctor.hub import check_checkpoint
from policy_doctor.retained import check_retained_candidate, retain_check
from policy_doctor.server import create_server


@pytest.fixture
def empty_pair(tmp_path):
    demo = create_policy_demo(tmp_path / 'demo')
    checkpoint = tmp_path / 'empty_checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({
        'type': 'act', 'input_features': {}, 'output_features': {}}))
    for name, registry in [('pre', 'normalizer_processor'), ('post', 'unnormalizer_processor')]:
        (checkpoint / f'policy_{name}processor.json').write_text(json.dumps({
            'steps': [{'registry_name': registry, 'config': {'features': {}, 'norm_map': {}}}]}))
    return checkpoint, demo['corrected_dataset']


def test_empty_assessment_is_unknown_without_changing_default_scope(empty_pair):
    checkpoint, dataset = empty_pair
    default = check_checkpoint(str(checkpoint), str(dataset))
    assert default['status'] == 'unknown'
    assert [f['code'] for f in default['findings']] == ['deployment.missing']
    report = check_checkpoint(str(checkpoint), str(dataset), scope='metadata')
    assert report['status'] == 'unknown'
    assert report['counts'] == {'error': 0, 'warning': 0, 'unknown': 1, 'pass': 0}
    assert [f['code'] for f in report['findings']] == ['metadata.not_assessed']
    assert report['deployment_status'] == 'not_assessed'


def test_empty_assessment_cli_does_not_exit_success(empty_pair, capsys):
    checkpoint, dataset = empty_pair
    assert main(['check', str(checkpoint), '--dataset', str(dataset), '--scope', 'metadata', '--json']) == 3
    assert json.loads(capsys.readouterr().out)['status'] == 'unknown'


@pytest.mark.parametrize('response_format', ['concise', 'detailed'])
def test_empty_assessment_mcp_never_claims_pass(empty_pair, tmp_path, response_format):
    checkpoint, dataset = empty_pair
    server = create_server([str(tmp_path)], allow_hub=True)
    async def run():
        return await server.call_tool('check_policy_metadata', {
            'source': str(checkpoint), 'dataset': str(dataset), 'scope': 'metadata',
            'response_format': response_format})
    response = asyncio.run(run())
    body = response[1] if isinstance(response, tuple) else response
    assert body['status'] == 'unknown'
    assert body['counts']['pass'] == 0


def test_empty_assessment_cannot_certify_retained_later_candidate(empty_pair, tmp_path):
    checkpoint, dataset = empty_pair
    case = tmp_path / 'case'
    saved = retain_check('policy_metadata', checkpoint, case, dataset=dataset, parameters={'scope': 'metadata'})
    assert saved['original_status'] == 'unknown'
    later = check_retained_candidate(case, dataset)
    assert later['candidate_evaluated'] and later['candidate_status'] == 'unknown'
    assert later['status'] == 'unknown'
