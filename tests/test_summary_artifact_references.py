"""A concise finding can share a file identity without sharing distinct observations."""
import copy
import hashlib
import json

import pytest

from policy_doctor.server import _deduplicate_finding_artifacts, _summary


def evidence(**extra):
    return {'path': '/recording/processor_config.json', 'sha256': 'a' * 64,
            'pointer': '/first', 'value': 'false', **extra}


def expand(items):
    result = []
    for index, item in enumerate(items):
        value = copy.deepcopy(item)
        anchor = value.pop('artifact_from', None)
        if anchor is not None:
            assert type(anchor) is int and 0 <= anchor < index
            assert 'artifact_from' not in items[anchor]
            for key in ('path', 'sha256', 'full_path_sha256'):
                if key in items[anchor]:
                    value[key] = items[anchor][key]
        result.append(value)
    return result


def test_references_expand_exactly_and_never_chain_or_share_observations():
    items = [evidence(full_path_sha256='b' * 64),
             evidence(pointer='/second', value='17', full_path_sha256='b' * 64),
             evidence(pointer='/third', value={'nested': True}, full_path_sha256='b' * 64)]
    original = copy.deepcopy(items)
    findings = [{'code': 'example', 'status': 'error', 'evidence': items}]
    _deduplicate_finding_artifacts(findings)
    assert [item.get('artifact_from') for item in items] == [None, 0, 0]
    assert items[0] == original[0]
    assert expand(items) == original
    before = copy.deepcopy(findings)
    _deduplicate_finding_artifacts(findings)
    assert findings == before


@pytest.mark.parametrize('different', [
    {'path': '/other/processor_config.json'},
    {'sha256': 'b' * 64},
    {'full_path_sha256': 'c' * 64},
])
def test_distinct_full_file_identity_never_becomes_a_reference(different):
    items = [evidence(full_path_sha256='b' * 64), evidence(**{'full_path_sha256': 'b' * 64, **different})]
    original = copy.deepcopy(items)
    _deduplicate_finding_artifacts([{'evidence': items}])
    assert items == original


def test_same_preview_requires_equal_full_path_identity_and_hash_presence():
    items = [evidence(), evidence(full_path_sha256='b' * 64), evidence(sha256=None)]
    original = copy.deepcopy(items)
    _deduplicate_finding_artifacts([{'evidence': items}])
    assert items == original


def test_reference_indices_are_local_to_each_finding():
    findings = [{'evidence': [evidence(), evidence(pointer='/second')]},
                {'evidence': [evidence(), evidence(pointer='/other')]}]
    originals = [copy.deepcopy(item['evidence']) for item in findings]
    _deduplicate_finding_artifacts(findings)
    for finding, original in zip(findings, originals, strict=True):
        assert finding['evidence'][1]['artifact_from'] == 0
        assert expand(finding['evidence']) == original


def test_small_summary_keeps_direct_file_identities():
    report = {'kind': 'example', 'status': 'error', 'findings': [
        {'code': 'example', 'status': 'error', 'message': 'Changed', 'remedy': 'Inspect',
         'evidence': [evidence(), evidence(pointer='/second')]}]}
    summary = _summary(report)
    assert all('artifact_from' not in item for item in summary['findings_shown'][0]['evidence'])


def test_summary_keeps_original_identity_when_an_existing_preview_is_shortened_again():
    findings = [{'code': f'example.{index}', 'status': 'error', 'message': 'Changed', 'remedy': 'Inspect',
                 'evidence': [evidence(path='/' + 'x' * 600, full_path_sha256='b' * 64),
                              evidence(path='/' + 'x' * 600, full_path_sha256='c' * 64,
                                       pointer='/second', full_pointer_sha256='d' * 64,
                                       full_value_sha256='e' * 64)]} for index in range(12)]
    report = {'kind': 'example', 'status': 'error', 'findings': findings}
    original = copy.deepcopy(report)
    summary = _summary(report)
    assert report == original
    assert len(summary['findings_shown']) >= 3
    for finding in summary['findings_shown']:
        assert finding['evidence'][0]['full_path_sha256'] == 'b' * 64
        second = finding['evidence'][1]
        assert second['full_path_sha256'] == 'c' * 64
        assert second['full_pointer_sha256'] == 'd' * 64 and second['full_value_sha256'] == 'e' * 64
        assert 'artifact_from' not in second


def test_unbounded_diagnostic_code_keeps_a_prefix_and_full_original_digest():
    code = '原因' * 128
    summary = _summary({'kind': 'example', 'status': 'error', 'findings': [
        {'code': code, 'status': 'error', 'message': 'Changed', 'evidence': []}]})
    finding = summary['findings_shown'][0]
    assert code.startswith(finding['code']) and finding['code'] != code
    assert len(json.dumps(finding['code']).encode()) - 2 <= 128
    assert finding['full_code_sha256'] == hashlib.sha256(code.encode()).hexdigest()


def test_statistics_message_keeps_its_complete_mode_uncertainty_prefix():
    prefix = ('The active normalization mode and epsilon are not established by stats.json; '
              'IDENTITY leaves values unchanged.')
    message = prefix + ' Saved conditional detail.' * 80
    summary = _summary({'kind': 'example', 'status': 'warning', 'findings': [
        {'code': 'dataset.constant_dimension', 'status': 'warning', 'message': message,
         'remedy': 'Verify the intended mode.', 'evidence': []}]})
    finding = summary['findings_shown'][0]
    assert finding['message'].startswith(prefix) and len(finding['message']) <= 400
    assert finding['full_message_sha256'] == hashlib.sha256(message.encode()).hexdigest()
