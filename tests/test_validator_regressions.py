"""Independent regressions from observed workflow failures, using generated inputs."""
import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.investigation import investigate_run
from policy_doctor.rollouts import compare_rollouts


@pytest.mark.parametrize('left,right', [(2**53, 2**53 + 1), (2**63 - 1, 2**63)])
def test_cross_signed_integer_difference_is_preserved(tmp_path, left, right):
    reference = tmp_path / 'reference.npz'
    candidate = tmp_path / 'candidate.npz'
    np.savez(reference, step=np.array([0]), actions=np.array([left], dtype=np.int64))
    np.savez(candidate, step=np.array([0]), actions=np.array([right], dtype=np.uint64))
    report = compare_rollouts(str(reference), str(candidate), alignment_key='step')
    assert report['observed_difference'] is True
    assert report['first_observed_difference']['reference_value'] == left
    assert report['first_observed_difference']['candidate_value'] == right


def test_cross_signed_sample_ids_do_not_falsely_align(tmp_path):
    base = 2**63 - 100
    reference = tmp_path / 'reference.npz'
    candidate = tmp_path / 'candidate.npz'
    np.savez(reference, step=np.array([base, base + 1], dtype=np.int64), actions=np.zeros(2))
    np.savez(candidate, step=np.array([base + 1, base + 2], dtype=np.uint64), actions=np.zeros(2))
    report = compare_rollouts(str(reference), str(candidate), alignment_key='step')
    assert report['alignment']['status'] == 'unknown'
    assert report['comparison_complete'] is False


def test_integer_csv_does_not_silently_round_mixed_range(tmp_path):
    reference = tmp_path / 'reference.csv'
    candidate = tmp_path / 'candidate.csv'
    reference.write_text('step,actions\n0,-1\n1,9223372036854775807\n')
    candidate.write_text('step,actions\n0,-1\n1,9223372036854775808\n')
    try:
        report = compare_rollouts(str(reference), str(candidate), alignment_key='step', atol=0, rtol=0)
    except InputError:
        return  # Explicit unsupported-range refusal preserves the evidence boundary.
    assert report['observed_difference'] is True


def test_incomplete_discovery_does_not_autoassociate_visible_candidate(case):
    root = case['checkpoint']
    hidden = root / 'unsearched' / 'another-checkpoint'
    hidden.mkdir(parents=True)
    (hidden / 'config.json').write_bytes((root / 'config.json').read_bytes())
    report = investigate_run(str(root), max_depth=0)
    assert report['coverage']['truncated'] is True
    assert report['audits'] == []
    assert report['status'] == 'unknown'
