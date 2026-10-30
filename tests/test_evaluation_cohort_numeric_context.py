"""Native JSON reserialization must not create a fictitious numeric context drift."""

import json

import pytest

from policy_doctor.evaluation_cohort import audit_evaluation_cohort

TABLE = {'groups': '/groups', 'group_id': '/name', 'records': '/rows', 'record_id': '/id', 'outcome': '/success'}


@pytest.mark.parametrize(('left', 'right', 'status'), [
    (10, 10.0, 'pass'),
    (True, 1, 'error'),
    ('1', 1, 'error'),
    (2**53 + 1, 2**53, 'error'),
    (2**53 + 1, float(2**53), 'error'),
    ({'hz': [10, -0.0]}, {'hz': [10.0, 0]}, 'pass'),
    ([True], [1.0], 'error'),
    ([1, 2], [2, 1], 'error'),
])
def test_selected_context_preserves_exact_numeric_meaning(tmp_path, left, right, status):
    roots = []
    for role, value in [('reference', left), ('candidate', right)]:
        root = tmp_path / role
        root.mkdir()
        (root / 'native.json').write_text(json.dumps({'setting': value,
            'groups': [{'name': 'task', 'rows': [{'id': 0, 'success': True}]}]}))
        roots.append(root)
    report = audit_evaluation_cohort(roots[1], reference=roots[0], table=TABLE, equal_fields=['/setting'])
    assert report['status'] == status
    field = report['fields'][0]
    assert field['status'] == status and 'numeric-format' in field['reason']
    if left == 10 and right == 10.0:
        assert field['reference']['canonical_sha256'] != field['candidate']['canonical_sha256']


def test_numeric_context_equivalence_does_not_coerce_native_ids(tmp_path):
    for role, identity in [('reference', 1), ('candidate', '1')]:
        root = tmp_path / role
        root.mkdir()
        (root / 'native.json').write_text(json.dumps({'groups': [{'name': 'task',
            'rows': [{'id': identity, 'success': True}]}]}))
    report = audit_evaluation_cohort(tmp_path / 'candidate', reference=tmp_path / 'reference',
                                    table=TABLE, equal_fields=[])
    assert report['status'] == 'error' and report['cohort_comparison']['missing_count'] == 1
