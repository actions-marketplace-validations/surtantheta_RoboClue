import json

import pytest

from policy_doctor.artifacts import InputError, load_checkpoint, load_deployment, step_kind
from policy_doctor.diagnostics import audit
from policy_doctor.trace import trace_actions


def edit(case, filename, transform):
    path = case["checkpoint"] / filename
    obj = json.loads(path.read_text())
    transform(obj)
    case["put"](path, obj)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("training_contract", []),
    ("units", []), ("units", ["", "rad"]), ("frame", []), ("command_representation", "guess"),
    ("normalization_mapping", []), ("rename_map", None)])
def test_bad_runtime_declarations_rejected(case, field, value):
    obj = json.loads(case["deployment"].read_text())
    obj[field] = value
    case["put"](case["deployment"], obj)
    with pytest.raises(InputError):
        load_deployment(case["deployment"])


@pytest.mark.parametrize("field,value", [("chunk_size", "four"), ("n_obs_steps", True),
                                      ("n_action_steps", 0), ("normalization_mapping", [])])
def test_bad_checkpoint_configuration_rejected(case, field, value):
    edit(case, "config.json", lambda d: d.update({field: value}))
    with pytest.raises(InputError):
        load_checkpoint(case["checkpoint"])


def test_duplicate_decoder_never_gets_clean_report_or_trace(case):
    edit(case, "policy_postprocessor.json", lambda d: d["steps"].append(d["steps"][0]))
    report = audit(str(case["checkpoint"]))
    assert any(f["code"] == "processor.ambiguous" for f in report["findings"])
    with pytest.raises(InputError, match="ambiguous"):
        trace_actions(str(case["checkpoint"]), str(case["sample"]))


def test_invalid_order_blocks_numeric_trace(case):
    edit(case, "policy_postprocessor.json", lambda d: d["steps"].reverse())
    with pytest.raises(InputError, match="processor_order"):
        trace_actions(str(case["checkpoint"]), str(case["sample"]))


def test_invalid_normalization_epsilon(case):
    edit(case, "policy_postprocessor.json", lambda d: d["steps"][0]["config"].update(eps=0))
    report = audit(str(case["checkpoint"]))
    assert any(f["code"] == "normalization.epsilon" for f in report["findings"])
    with pytest.raises(InputError):
        trace_actions(str(case["checkpoint"]), str(case["sample"]))


def test_corrupt_binary_stats_returns_useful_error(case):
    (case["checkpoint"] / "pre.safetensors").write_bytes(b"corrupt")
    with pytest.raises(InputError, match="normalization state"):
        load_checkpoint(case["checkpoint"])


def test_custom_named_class_does_not_impersonate_supported_processor():
    assert step_kind("my_custom_module.AbsoluteActionsProcessorStep") == "unsupported"
    assert step_kind("lerobot.processor.relative_action_processor.AbsoluteActionsProcessorStep") == "absolute"


def test_unsupported_decoder_is_unknown_even_in_metadata(case):
    edit(case, "policy_postprocessor.json", lambda d: d["steps"].append({"class": "custom.InverseKinematics", "config": {}}))
    report = audit(str(case["checkpoint"]), str(case["dataset"]), str(case["deployment"]))
    assert report["status"] == "unknown"


def test_string_false_is_not_a_boolean(case):
    edit(case, "policy_preprocessor.json", lambda d: d["steps"][0]["config"].update(enabled="false"))
    with pytest.raises(InputError, match="boolean"):
        load_checkpoint(case["checkpoint"])


def test_processor_flag_config_disagreement(case):
    edit(case, "config.json", lambda d: d.update(use_relative_actions=False))
    assert any(f["code"] == "action.config_relative_mismatch" for f in audit(str(case["checkpoint"]))["findings"])


def test_configuration_symlink_cannot_escape_checkpoint(case, tmp_path):
    config = case["checkpoint"] / "config.json"
    outside = tmp_path / "outside.json"
    config.rename(outside)
    config.symlink_to(outside)
    with pytest.raises(InputError, match="symlink"):
        load_checkpoint(case["checkpoint"])


@pytest.mark.parametrize("limits", ["bad", [[1, 0], [0, 1]], [[0, 1]], [[0, "bad"], [0, 1]]])
def test_invalid_limits_are_input_errors(case, limits):
    data = json.loads(case["deployment"].read_text())
    data["limits"] = limits
    case["put"](case["deployment"], data)
    with pytest.raises(InputError, match="limits"):
        load_deployment(case["deployment"])


@pytest.mark.parametrize("actual,status", [(1, "pass"), (2, "error"), (None, "unknown")])
def test_observation_history_is_not_confused_with_action_horizon(case, actual, status):
    edit(case, "config.json", lambda d: d.update(n_obs_steps=1))
    data = json.loads(case["deployment"].read_text())
    data["n_obs_steps"] = actual
    case["put"](case["deployment"], data)
    findings = audit(str(case["checkpoint"]), deployment=str(case["deployment"]))["findings"]
    assert next(f for f in findings if f["code"] == "observation.history")["status"] == status
