import json

import numpy as np
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.trace import trace_actions, transform


def test_chunk_anchor_and_absolute_gripper(case):
    report = trace_actions(str(case["checkpoint"]), str(case["sample"]), str(case["deployment"]))
    # Hand calculation: x = normalized * 2 + 1 + 10; gripper = normalized * .25 + .5.
    assert report["commands"] == [[11., .5], [13., .75], [15., .25]]
    assert report["status"] == "pass"
    assert np.allclose(report["state_normalization"]["normalized"], [1, 1])


@pytest.mark.parametrize("mode,stats,expected", [
    ("MEAN_STD", {"mean": [3, 4], "std": [2, 3]}, [[5, 1]]),
    ("MIN_MAX", {"min": [0, 10], "max": [10, 20]}, [[10, 10]]),
    ("QUANTILES", {"q01": [0, 10], "q99": [10, 20]}, [[10, 10]]),
    ("QUANTILE10", {"q10": [0, 10], "q90": [10, 20]}, [[10, 10]]),
    ("IDENTITY", {}, [[1, -1]]),
])
def test_independent_inverse_examples(mode, stats, expected):
    assert np.allclose(transform(np.array([[1., -1.]]), stats, mode, 1e-8, True), expected)


def test_zero_range_uses_upstream_epsilon():
    result = transform(np.array([[0.]]), {"min": [2.], "max": [2.]}, "MIN_MAX", 1e-4, True)
    assert result[0, 0] == pytest.approx(2.00005)


def test_missing_anchor_refuses_guess(case):
    data = json.loads(case["sample"].read_text())
    data.pop("reference_state")
    case["put"](case["sample"], data)
    with pytest.raises(InputError, match="reference_state"):
        trace_actions(str(case["checkpoint"]), str(case["sample"]))


def test_unsupported_postprocessor_refuses_partial_trace(case):
    path = case["checkpoint"] / "policy_postprocessor.json"
    data = json.loads(path.read_text())
    data["steps"].append({"registry_name": "custom_ik", "config": {}})
    case["put"](path, data)
    with pytest.raises(InputError, match="Unsupported"):
        trace_actions(str(case["checkpoint"]), str(case["sample"]))


def test_limits_and_expected_command_disagreement(case):
    data = json.loads(case["sample"].read_text())
    data["model_output"][0][1] = 10
    case["put"](case["sample"], data)
    report = trace_actions(str(case["checkpoint"]), str(case["sample"]), str(case["deployment"]))
    assert report["status"] == "error"
    assert len([f for f in report["findings"] if f["status"] == "error"]) == 2


@pytest.mark.parametrize("value", [[[1, 2, 3]], [1, 2], [], [[float("nan"), 1]]])
def test_bad_output_shapes_and_values(case, value):
    data = json.loads(case["sample"].read_text())
    data["model_output"] = value
    case["put"](case["sample"], data)
    with pytest.raises(InputError):
        trace_actions(str(case["checkpoint"]), str(case["sample"]))
