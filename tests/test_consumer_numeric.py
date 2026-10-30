"""Generated native-layout artifacts and scalar equations; no native execution."""

import copy
import hashlib
import json

import numpy as np
import pytest

from policy_doctor.consumer_context import resolve_native_processor_context
from policy_doctor.consumer_numeric import compare_consumer_numeric, trace_consumer_numeric

TAG = "unitree_g1_full_body_with_waist_height_nav_cmd"
STATE = {"left_leg": 6, "right_leg": 6, "waist": 3, "left_arm": 7,
         "right_arm": 7, "left_hand": 7, "right_hand": 7}
ACTION = {"left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7,
          "waist": 3, "base_height_command": 1, "navigate_command": 3}


def artifacts():
    # Pinned native enum names/order and actual processor_config/statistics roles.
    modalities = {
        "video": {"delta_indices": [0], "modality_keys": ["ego_view"]},
        "state": {"delta_indices": [0], "modality_keys": list(STATE)},
        "action": {"delta_indices": list(range(50)), "modality_keys": list(ACTION),
                   "action_configs": [{"rep": "RELATIVE" if i < 2 else "ABSOLUTE",
                                       "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
                                      for i in range(7)]},
        "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
    }
    config = {"processor_class": "Gr00tN1d7Processor", "processor_kwargs": {
        "modality_configs": {TAG: modalities}, "max_state_dim": 64, "max_action_dim": 64,
        "max_action_horizon": 50, "use_percentiles": False, "use_relative_action": False,
        "clip_outliers": False, "apply_sincos_state_encoding": False, "use_mean_std": False}}
    stats = {TAG: {}}
    for role, widths in (("state", STATE), ("action", ACTION)):
        stats[TAG][role] = {key: {"min": [-2.0] * width, "max": [2.0] * width,
            "mean": [0.5] * width, "std": [0.25] * width,
            "q01": [-1.0] * width, "q99": [1.0] * width} for key, width in widths.items()}
    stats[TAG]["relative_action"] = {key: copy.deepcopy(stats[TAG]["action"][key])
                                     for key in ("left_arm", "right_arm")}
    for group in stats[TAG]["relative_action"].values():
        group["min"], group["max"] = [-1.0] * 7, [1.0] * 7
    return config, stats


def save(tmp_path, name, config, stats):
    root = tmp_path / name
    root.mkdir()
    (root / "processor_config.json").write_text(json.dumps(config))
    (root / "statistics.json").write_text(json.dumps(stats))
    return resolve_native_processor_context(root, read_roots=[tmp_path])


def sample():
    return {"states": {key: np.full((1, width), 0.25, dtype=np.float32) for key, width in STATE.items()},
            "actions": {key: np.full((50, width), 0.75, dtype=np.float32) for key, width in ACTION.items()},
            "selection": {"state_indices": [123], "action_indices": list(range(123, 173)),
                          "source_key": "GENERATED_demo", "native_episode_id": 7}}


def digest(values):
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def field(report, label):
    return next(f for f in report["fields"] if f["field"] == label)


def test_complete_trace_hashes_native_typed_inner_outputs_without_arrays(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "a", config, stats)
    raw = sample()
    report = trace_consumer_numeric(context, raw, read_roots=[tmp_path])
    assert report["status"] == "PASS" and report["computed_fields"] == 14
    expected = np.full((50, 7), 0.375, dtype=np.float32)
    assert field(report, "action.left_arm")["stages"]["clipped"]["sha256"] == digest(expected)
    assert field(report, "state.left_arm")["stages"]["clipped"]["sha256"] == digest(
        np.full((1, 7), 0.125, dtype=np.float32))
    assert all(value == "UNKNOWN" for value in report["boundaries"].values())
    assert np.array_equal(raw["actions"]["left_arm"], np.full((50, 7), .75, dtype=np.float32))
    json.dumps(report, allow_nan=False)


def test_declared_relative_flag_changes_both_parameters_and_arm_value_effect(tmp_path):
    config, stats = artifacts()
    ref = save(tmp_path, "absolute", config, stats)
    config["processor_kwargs"]["use_relative_action"] = True
    candidate = save(tmp_path, "relative", config, stats)
    report = compare_consumer_numeric(ref, sample(), candidate, sample(), read_roots=[tmp_path])
    assert report["status"] == "PASS"  # Recipe differences do not establish which is intended.
    assert report["numeric_projection_value_effect"] == "CHANGED"
    assert report["final_inner_numeric_counts"]["value_coordinates"] == 700
    assert report["first_value_divergence"]["stage"] == "representation"
    assert report["first_value_divergence"]["field"] == "action.left_arm"
    assert report["active_interpretation_changed"]
    assert field(report["reference"], "action.left_arm")["active_parameter_sha256"] != field(
        report["candidate"], "action.left_arm")["active_parameter_sha256"]
    for label in ["state.left_arm", "action.waist", "action.left_hand", "action.navigate_command"]:
        assert field(report["reference"], label)["stages"] == field(report["candidate"], label)["stages"]
    assert report["final_wrapper_model_target_effect"] == report["native_parity"] == "UNKNOWN"


@pytest.mark.parametrize("percentiles,statistic,replacement", [(False, "q01", -0.5), (True, "min", -2.25)])
def test_inactive_stat_follows_actual_saved_percentile_branch(tmp_path, percentiles, statistic, replacement):
    config, stats = artifacts()
    config["processor_kwargs"]["use_percentiles"] = percentiles
    ref = save(tmp_path, "before", config, stats)
    stats[TAG]["state"]["left_arm"][statistic][0] = replacement
    after = save(tmp_path, "after", config, stats)
    report = compare_consumer_numeric(ref, sample(), after, sample(), read_roots=[tmp_path])
    assert report["status"] == "PASS" and report["artifact_identity_changed"]
    assert not report["active_interpretation_changed"]
    assert report["numeric_projection_value_effect"] == "EQUAL"
    assert report["first_observed_divergence"] is None
    assert report["final_inner_numeric_counts"] == {
        "value_coordinates": 0, "dtype_fields": 0, "shape_fields": 0, "byte_fields": 0}


def test_serialized_global_meanstd_flag_does_not_select_per_key_meanstd(tmp_path):
    config, stats = artifacts()
    ref = save(tmp_path, "before", config, stats)
    config["processor_kwargs"]["use_mean_std"] = True
    after = save(tmp_path, "after", config, stats)
    report = compare_consumer_numeric(ref, sample(), after, sample(), read_roots=[tmp_path])
    assert report["artifact_identity_changed"] and not report["active_interpretation_changed"]
    assert report["numeric_projection_value_effect"] == "EQUAL"
    assert field(report["candidate"], "state.left_arm")["operation"] == "minmax"


def test_clipping_masks_do_not_conflate_representation_or_raw_values(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"]["clip_outliers"] = True
    context = save(tmp_path, "checkpoint", config, stats)
    raw = sample()
    raw["actions"]["left_arm"][:] = 10
    report = trace_consumer_numeric(context, raw, read_roots=[tmp_path])
    f = field(report, "action.left_arm")
    assert f["stages"]["normalized"]["sha256"] == digest(np.full((50, 7), 5, dtype=np.float32))
    assert f["stages"]["clipped"]["sha256"] == digest(np.ones((50, 7), dtype=np.float32))
    assert f["clipping_changed_coordinates"]["true_coordinates"] == 350
    assert f["out_of_range_before_clipping"]["true_coordinates"] == 350


@pytest.mark.parametrize("modality", ["state", "action"])
def test_zero_std_raw_passthrough_and_per_role_clipping(tmp_path, modality):
    config, stats = artifacts()
    config["processor_kwargs"]["clip_outliers"] = True
    config["processor_kwargs"]["modality_configs"][TAG][modality]["mean_std_embedding_keys"] = ["left_arm"]
    stats[TAG][modality]["left_arm"]["std"][0] = 0.0
    context = save(tmp_path, "checkpoint", config, stats)
    raw = sample()
    raw["states" if modality == "state" else "actions"]["left_arm"][:] = 2.0
    report = trace_consumer_numeric(context, raw, read_roots=[tmp_path])
    f = field(report, modality + ".left_arm")
    rows = 1 if modality == "state" else 50
    expected = np.full((rows, 7), 6, dtype=np.float32)
    expected[:, 0] = 2
    assert f["stages"]["normalized"]["sha256"] == digest(expected)
    assert f["clip"] is (modality == "action")
    assert f["stages"]["clipped"]["sha256"] == digest(
        np.ones_like(expected) if modality == "action" else expected)


def test_minmax_native_isclose_mask_keeps_nearconstant_coordinates_zero(tmp_path):
    config, stats = artifacts()
    group = stats[TAG]["state"]["left_arm"]
    group["min"], group["max"] = [1.0] * 7, [1.000001] * 7
    context = save(tmp_path, "checkpoint", config, stats)
    report = trace_consumer_numeric(context, sample(), read_roots=[tmp_path])
    f = field(report, "state.left_arm")
    assert f["inactive_normalization_coordinates"]["true_coordinates"] == 7
    assert f["stages"]["normalized"]["sha256"] == digest(np.zeros((1, 7), dtype=np.float32))


def test_minmax_two_assignment_casts_match_independent_scalar_oracle(tmp_path):
    config, stats = artifacts()
    group = stats[TAG]["state"]["left_arm"]
    group["min"], group["max"] = [-0.3] * 7, [0.7] * 7
    context = save(tmp_path, "checkpoint", config, stats)
    raw = sample()
    raw["states"]["left_arm"][:] = np.float32(.1)
    fraction = np.float32((float(np.float32(.1)) + .3) / (0.7 + .3))
    two_casts = np.float32(np.float32(2) * fraction - np.float32(1))
    one_cast = np.float32(2 * ((float(np.float32(.1)) + .3) / (0.7 + .3)) - 1)
    assert two_casts != one_cast
    report = trace_consumer_numeric(context, raw, read_roots=[tmp_path])
    assert field(report, "state.left_arm")["stages"]["normalized"]["sha256"] == digest(
        np.full((1, 7), two_casts, dtype=np.float32))


@pytest.mark.parametrize("percentiles", [False, True])
def test_relative_full_horizon_parameters_override_percentiles_with_native_minmax(tmp_path, percentiles):
    config, stats = artifacts()
    config["processor_kwargs"].update(use_relative_action=True, use_percentiles=percentiles)
    for key in ("left_arm", "right_arm"):
        group = stats[TAG]["relative_action"][key]
        for name, value in list(group.items()):
            group[name] = [list(value) for _ in range(50)]
        group["max"] = [[1.0 + row / 50] * 7 for row in range(50)]
    context = save(tmp_path, "checkpoint", config, stats)
    report = trace_consumer_numeric(context, sample(), read_roots=[tmp_path])
    expected = np.empty((50, 7), dtype=np.float64)
    selected_stats = stats[TAG]["relative_action"]["left_arm"]
    # Preserve the saved bound's rounding before subtraction. Regrouping this
    # as (2 + row / 50) is algebraically equal but not bitwise native behavior.
    assert any((1.0 + row / 50) - (-1.0) != 2 + row / 50 for row in range(50))
    for row in range(50):
        high = selected_stats["max"][row][0]
        low = selected_stats["min"][row][0]
        expected[row, :] = 2 * ((.75 - .25 - low) / (high - low)) - 1
    assert report["status"] == "PASS"
    assert field(report, "action.left_arm")["stages"]["normalized"]["sha256"] == digest(expected)
    assert field(report, "action.left_arm")["stages"]["representation"]["dtype"] == "<f8"


def test_meanstd_tiny_nonzero_std_is_not_isclose_masked(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"]["modality_configs"][TAG]["state"]["mean_std_embedding_keys"] = ["left_arm"]
    stats[TAG]["state"]["left_arm"]["std"] = [1e-10] * 7
    context = save(tmp_path, "checkpoint", config, stats)
    report = trace_consumer_numeric(context, sample(), read_roots=[tmp_path])
    f = field(report, "state.left_arm")
    assert f["active_normalization_coordinates"]["true_coordinates"] == 7
    assert f["stages"]["normalized"]["sha256"] == digest(np.full((1, 7), -2.5e9, dtype=np.float32))


def test_dtype_change_is_separate_from_equal_value_effect(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    before, after = sample(), sample()
    for modality in ("states", "actions"):
        after[modality] = {key: value.astype(np.float64) for key, value in after[modality].items()}
    report = compare_consumer_numeric(context, before, context, after, read_roots=[tmp_path])
    assert report["status"] == "PASS" and report["numeric_projection_value_effect"] == "EQUAL"
    assert report["final_inner_numeric_counts"]["dtype_fields"] == 14
    assert report["first_observed_divergence"]["change"] == "dtype"
    assert report["first_value_divergence"] is None
    assert report["final_wrapper_model_target_effect"] == "UNKNOWN"


def test_signed_zero_is_byte_change_without_numeric_value_effect(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    before, after = sample(), sample()
    before["states"]["left_arm"][:] = 0.0
    after["states"]["left_arm"][:] = -0.0
    report = compare_consumer_numeric(context, before, context, after, read_roots=[tmp_path])
    assert report["first_observed_divergence"]["stage"] == "raw"
    assert report["first_observed_divergence"]["change"] == "bytes"
    assert report["first_value_divergence"] is None and report["numeric_projection_value_effect"] == "EQUAL"


def test_precise_changed_field_coordinate_and_caller_index_without_source_authentication(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    before, after = sample(), sample()
    after["actions"]["left_arm"][2, 3] += 1
    after["selection"]["action_indices"][2] = 999
    report = compare_consumer_numeric(context, before, context, after, read_roots=[tmp_path])
    assert report["final_inner_numeric_counts"]["value_coordinates"] == 1
    first = report["first_value_divergence"]
    assert first["field"] == "action.left_arm" and first["coordinate"] == [2, 3]
    assert first["stage"] == "raw" and first["row"]["caller_selected_index"] == 125
    assert first["row"]["source_membership"] == "UNKNOWN"
    assert first["reference_row"]["caller_selected_index"] == 125
    assert first["candidate_row"]["caller_selected_index"] == 999 and report["selection_changed"]


def test_known_wrapper_capacity_error_survives_complete_inner_equivalence(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"]["max_action_dim"] = 1
    context = save(tmp_path, "checkpoint", config, stats)
    assert context["status"] == "ERROR"
    report = compare_consumer_numeric(context, sample(), context, sample(), read_roots=[tmp_path])
    assert report["status"] == "ERROR" and report["numeric_projection_value_effect"] == "EQUAL"
    assert any(f["code"] == "consumer.action_capacity" for f in report["reference"]["findings"])
    assert report["final_wrapper_model_target_effect"] == "UNKNOWN"


def test_unsupported_sincos_keeps_known_capacity_error_and_does_not_clear_numeric_coverage(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"].update(apply_sincos_state_encoding=True, max_action_dim=1)
    config["processor_kwargs"]["modality_configs"][TAG]["state"]["sin_cos_embedding_keys"] = ["left_arm"]
    context = save(tmp_path, "checkpoint", config, stats)
    report = compare_consumer_numeric(context, sample(), context, sample(), read_roots=[tmp_path])
    assert report["status"] == "ERROR" and report["numeric_projection_value_effect"] == "UNKNOWN"
    assert not report["complete_field_coverage"]


def test_source_artifact_drift_is_unknown_without_silent_reresolution(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    (tmp_path / "checkpoint/statistics.json").write_text("{}")
    report = compare_consumer_numeric(context, sample(), context, sample(), read_roots=[tmp_path])
    assert report["status"] == report["numeric_projection_value_effect"] == "UNKNOWN"
    assert not report["complete_field_coverage"]
    assert report["artifact_identity_changed"] is report["active_interpretation_changed"] is None


def test_revalidated_severity_survives_caller_erased_capacity_finding(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"]["max_action_dim"] = 1
    context = save(tmp_path, "checkpoint", config, stats)
    context["status"], context["findings"] = "PASS", []
    report = trace_consumer_numeric(context, sample(), read_roots=[tmp_path])
    assert report["status"] == "ERROR"
    assert any(f["code"] == "consumer.action_capacity" for f in report["findings"])


def test_revalidated_clean_context_does_not_trust_injected_error(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    context["status"] = "ERROR"
    context["findings"] = [{"status": "ERROR", "code": "invented", "message": "caller edited"}]
    report = trace_consumer_numeric(context, sample(), read_roots=[tmp_path])
    assert report["status"] == "PASS" and not report["findings"]


def test_partial_context_keeps_fresh_required_relative_statistics_error(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"]["use_relative_action"] = True
    del stats[TAG]["relative_action"]
    context = save(tmp_path, "checkpoint", config, stats)
    context["status"], context["findings"] = "PASS", []
    report = compare_consumer_numeric(context, sample(), context, sample(), read_roots=[tmp_path])
    assert report["status"] == "ERROR" and report["numeric_projection_value_effect"] == "UNKNOWN"
    assert any(f["code"] == "consumer.relative_statistics_missing" for f in report["reference"]["findings"])


def test_no_descriptor_keeps_fresh_independent_native_horizon_error(tmp_path):
    config, stats = artifacts()
    config["processor_kwargs"]["max_action_horizon"] = 1
    config["processor_kwargs"]["modality_configs"][TAG]["state"]["modality_keys"].reverse()
    context = save(tmp_path, "checkpoint", config, stats)
    assert context["descriptor"] is None and context["status"] == "ERROR"
    context["status"], context["findings"] = "PASS", []
    report = trace_consumer_numeric(context, sample(), read_roots=[tmp_path])
    assert report["status"] == "ERROR"
    assert any(f["code"] == "consumer.action_horizon_capacity" for f in report["findings"])


def test_unavailable_authority_does_not_probe_or_claim_numeric_equivalence(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    other = tmp_path / "other"
    other.mkdir()
    report = compare_consumer_numeric(context, sample(), context, sample(), read_roots=[other])
    assert report["status"] == report["numeric_projection_value_effect"] == "UNKNOWN"


@pytest.mark.parametrize("bad", ["object", "width", "nonfinite", "missing"])
def test_unavailable_raw_field_prevents_false_equivalence(tmp_path, bad):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    raw = sample()
    if bad == "object":
        class Hook:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Untrusted array hook must not execute")
        raw["actions"]["left_arm"] = np.empty((50, 7), dtype=object)
        raw["actions"]["left_arm"].fill(Hook())
    elif bad == "width":
        raw["actions"]["left_arm"] = np.ones((50, 6), dtype=np.float32)
    elif bad == "missing":
        del raw["actions"]["left_arm"]
    else:
        raw["actions"]["left_arm"][0, 0] = np.nan
    report = compare_consumer_numeric(context, raw, context, raw, read_roots=[tmp_path])
    assert report["numeric_projection_value_effect"] == "UNKNOWN"
    assert not report["complete_field_coverage"]
    assert report["status"] == ("ERROR" if bad in {"nonfinite", "missing"} else "UNKNOWN")


def test_fixed_16_witness_cap_keeps_full_changed_coordinate_count(tmp_path):
    config, stats = artifacts()
    context = save(tmp_path, "checkpoint", config, stats)
    before, after = sample(), sample()
    after["actions"]["left_arm"] += 1
    report = compare_consumer_numeric(context, before, context, after, read_roots=[tmp_path])
    assert report["final_inner_numeric_counts"]["value_coordinates"] == 350
    assert len(report["first_value_differences"]) == 16 and report["first_value_differences_omitted"] == 334
    assert "occurrences" in report["counting_scope"]
    assert len(json.dumps(report, allow_nan=False).encode()) < 65536
