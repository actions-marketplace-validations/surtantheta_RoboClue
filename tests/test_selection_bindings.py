"""Identity-witness reader controls; generated arrays do not constitute native execution."""

import copy

import numpy as np
import pytest

from policy_doctor.selection_bindings import validate_local_binding


def packet():
    arrays = {
        "q": np.zeros((1, 3, 2), dtype=np.float32),
        "s": np.zeros((1, 2), dtype=np.float32),
        "h": np.zeros((1, 2), dtype=np.float32),
    }
    view = {
        "base_token": "o000003",
        "shape": [1, 3, 2],
        "stride": [6, 2, 1],
        "storage_offset": 0,
        "element_size": 4,
    }
    witness = {
        "schema_version": 1,
        "status": "pass",
        "mode": "queue",
        "basis": "query_view_mapping",
        "selected_array": "s",
        "selected_token": "o000002",
        "queue_head": None,
        "query": {"query_id": 0, "array": "q", "token": "o000001"},
        "selected_view": dict(view, shape=[1, 2], stride=[6, 1]),
        "query_view": view,
        "query_offset": 0,
        "update": None,
        "relations": {
            "selected_is_pre_head": None,
            "same_live_base": True,
            "actions_is_prediction": None,
            "selected_is_update_return": None,
        },
        "reason": None,
        "history": "unknown",
    }
    event = {
        "settings": {"temporal_ensemble_coeff": None, "n_action_steps": 2},
        "selected": "s",
        "predictions": [{"query_id": 0, "array": "q"}],
        "pre_state": {"queue": [], "queue_maxlen": 2},
        "local_binding": witness,
    }
    manifest = {
        "coverage": {"target": "one_policy_on_owner_thread", "complete_target_call_history": True},
        "binding_scope": {
            "schema_version": 1,
            "token_scope": "one_capture_owner_thread",
            "off_path_update_calls": 0,
            "external_state_mutations": "untracked",
        },
        "environment_selection": {"source_num_envs": 1},
    }
    return event, manifest, arrays


def audit(event, manifest, arrays):
    return validate_local_binding(event, manifest, arrays, source_supported=True, numeric_status="pass")


def test_old_capture_stays_unknown():
    event, manifest, arrays = packet()
    event.pop("local_binding")
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_exact_view_binding_is_separate_from_history():
    result = audit(*packet())
    assert result["status"] == "pass" and result["query_offset"] == 0
    assert result["history"] == "unknown"


@pytest.mark.parametrize(
    "edit",
    [
        lambda w: w["selected_view"].update(storage_offset=2),
        lambda w: w["selected_view"].update(base_token="o000004"),
        lambda w: w["relations"].update(same_live_base=False),
        lambda w: w.update(query_offset=True),
        lambda w: w["query_view"].update(stride=[6, 0, 1]),
        lambda w: w["query"].update(query_id=True),
        lambda w: w.update(selected_token="0xdeadbeef"),
    ],
)
def test_equal_values_do_not_overrule_identity_or_view_mismatch(edit):
    event, manifest, arrays = packet()
    edit(event["local_binding"])
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_missing_or_boolean_coverage_and_off_path_updates():
    event, manifest, arrays = packet()
    for key, value in [
        ("off_path_update_calls", True),
        ("off_path_update_calls", 1),
        ("schema_version", True),
    ]:
        modified = copy.deepcopy(manifest)
        modified["binding_scope"][key] = value
        assert audit(event, modified, arrays)["status"] == "unknown"


def test_actual_prehead_only_proves_local_head_not_old_query():
    event, manifest, arrays = packet()
    w = event["local_binding"]
    event["pre_state"]["queue"] = ["h"]
    event["predictions"] = []
    w.update(
        basis="pre_queue_head_identity",
        queue_head={"array": "h", "token": "o000002"},
        query=None,
        query_offset=None,
        query_view=None,
    )
    w["relations"]["selected_is_pre_head"] = True
    result = audit(event, manifest, arrays)
    assert result["status"] == "pass" and "query_id" not in result
    w["queue_head"]["token"] = "o000005"
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_ensemble_argument_return_relationship_requires_both_actual_joins():
    event, manifest, arrays = packet()
    w = event["local_binding"]
    event["settings"]["temporal_ensemble_coeff"] = 0.5
    w.update(
        mode="ensemble",
        basis="ensemble_argument_return_identity",
        query_offset=None,
        update={
            "helper_token": "o000006",
            "actions_token": "o000001",
            "prediction_token": "o000001",
            "return_token": "o000002",
            "selected_token": "o000002",
            "call_count": 1,
        },
    )
    w["relations"].update(actions_is_prediction=True, selected_is_update_return=True)
    assert audit(event, manifest, arrays)["status"] == "pass"
    w["update"]["actions_token"] = "o000007"
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_selected_native_rows_need_matching_view_mapping():
    event, manifest, arrays = packet()
    manifest["environment_selection"]["source_num_envs"] = 4096
    # Saved arrays still contain one explicitly selected row; descriptors cover the full native batch.
    event["local_binding"]["query_view"]["shape"][0] = 4096
    event["local_binding"]["selected_view"]["shape"][0] = 4096
    assert audit(event, manifest, arrays)["status"] == "pass"
    event["local_binding"]["selected_view"]["shape"][0] = 1
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_claimed_pass_cannot_override_unsupported_source_or_failed_arithmetic():
    event, manifest, arrays = packet()
    assert (
        validate_local_binding(event, manifest, arrays, source_supported=False, numeric_status="pass")[
            "status"
        ]
        == "unknown"
    )
    assert (
        validate_local_binding(event, manifest, arrays, source_supported=True, numeric_status="error")[
            "status"
        ]
        == "unknown"
    )


def test_zero_based_producer_token_is_valid_and_limit_is_strict():
    event, manifest, arrays = packet()
    event["local_binding"]["query"]["token"] = "o000000"
    assert audit(event, manifest, arrays)["status"] == "pass"
    event["local_binding"]["query"]["token"] = "o004096"
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_overlapping_native_batch_layout_does_not_bind_even_for_equal_values():
    event, manifest, arrays = packet()
    manifest["environment_selection"]["source_num_envs"] = 2
    for name in ("query_view", "selected_view"):
        event["local_binding"][name]["shape"][0] = 2
        event["local_binding"][name]["stride"][0] = 0
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_same_live_object_cannot_be_both_query_rank3_and_selected_rank2():
    event, manifest, arrays = packet()
    event["local_binding"]["selected_token"] = event["local_binding"]["query"]["token"]
    assert audit(event, manifest, arrays)["status"] == "unknown"


def test_ensemble_helper_and_tensor_roles_cannot_share_a_live_token():
    event, manifest, arrays = packet()
    witness = event["local_binding"]
    event["settings"]["temporal_ensemble_coeff"] = 0.5
    witness.update(
        mode="ensemble",
        basis="ensemble_argument_return_identity",
        query_offset=None,
        update={
            "helper_token": "o000001",
            "actions_token": "o000001",
            "prediction_token": "o000001",
            "return_token": "o000002",
            "selected_token": "o000002",
            "call_count": 1,
        },
    )
    witness["relations"].update(actions_is_prediction=True, selected_is_update_return=True)
    assert audit(event, manifest, arrays)["status"] == "unknown"
