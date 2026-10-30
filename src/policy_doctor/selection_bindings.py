"""Validate saved local identity witnesses; never infer identity from values alone."""

from __future__ import annotations

import re

import numpy as np

SCOPE = (
    "Capture-observed local object/view relationships only, conditional on the recorded runtime evidence. "
    "Opaque tokens are not authenticated telemetry; full query history, external writes and downstream commands remain unknown."
)
FIELDS = {
    "schema_version",
    "status",
    "mode",
    "basis",
    "selected_array",
    "selected_token",
    "queue_head",
    "query",
    "selected_view",
    "query_view",
    "query_offset",
    "update",
    "relations",
    "reason",
    "history",
}
RELATIONS = {"selected_is_pre_head", "same_live_base", "actions_is_prediction", "selected_is_update_return"}


def _token(value):
    return type(value) is str and re.fullmatch(r"o[0-9]{6}", value) is not None and 0 <= int(value[1:]) < 4096


def _integer(value, lower=0, upper=2**63 - 1):
    return type(value) is int and lower <= value <= upper


def _equal(arrays, left, right):
    return (
        type(left) is str
        and type(right) is str
        and left in arrays
        and right in arrays
        and arrays[left].dtype == arrays[right].dtype
        and np.array_equal(arrays[left], arrays[right])
    )


def _view(value, shape, dtype):
    if (
        type(value) is not dict
        or set(value) != {"base_token", "shape", "stride", "storage_offset", "element_size"}
        or not _token(value["base_token"])
        or value["shape"] != shape
        or type(value["shape"]) is not list
        or any(type(n) is not int for n in value["shape"])
        or type(value["stride"]) is not list
        or len(value["stride"]) != len(shape)
        or any(not _integer(n) for n in value["stride"])
        or not _integer(value["storage_offset"])
        or type(value["element_size"]) is not int
        or value["element_size"] != dtype.itemsize
    ):
        return False
    span = 1
    for step, length in sorted((s, n) for n, s in zip(shape, value["stride"], strict=True) if n > 1):
        if step < span:
            return False
        span += (length - 1) * step
        if span > 2**63 - 1:
            return False
    return value["storage_offset"] + span <= 2**63 - 1


def validate_local_binding(event, manifest, arrays, *, source_supported, numeric_status):
    """Return a narrow local verdict independently of historical lineage."""

    def unknown(reason):
        return {"status": "unknown", "reason": reason, "scope": SCOPE, "history": "unknown"}

    witness = event.get("local_binding")
    if witness is None:
        return unknown("No local identity witness was captured; old numeric snapshots cannot be upgraded")
    if type(witness) is not dict or set(witness) != FIELDS:
        return unknown("Local binding fields are missing or unsupported")
    coverage, scope = manifest.get("coverage"), manifest.get("binding_scope")
    if (
        not source_supported
        or numeric_status != "pass"
        or type(coverage) is not dict
        or coverage.get("target") != "one_policy_on_owner_thread"
        or coverage.get("complete_target_call_history") is not True
    ):
        return unknown(
            "Supported source, complete observed coverage and consistent numeric replay are required"
        )
    if (
        type(scope) is not dict
        or set(scope)
        != {"schema_version", "token_scope", "off_path_update_calls", "external_state_mutations"}
        or type(scope["schema_version"]) is not int
        or scope["schema_version"] != 1
        or scope["token_scope"] != "one_capture_owner_thread"
        or type(scope["off_path_update_calls"]) is not int
        or scope["off_path_update_calls"] != 0
        or scope["external_state_mutations"] != "untracked"
    ):
        return unknown("Local token scope or off-path update coverage is absent or unsupported")
    if (
        type(witness["schema_version"]) is not int
        or witness["schema_version"] != 1
        or witness["status"] != "pass"
        or witness["history"] != "unknown"
        or witness["selected_array"] != event.get("selected")
        or not _token(witness["selected_token"])
        or witness["reason"] is not None
    ):
        return unknown("Local witness does not establish a supported selected-return binding")
    mode = "queue" if event["settings"]["temporal_ensemble_coeff"] is None else "ensemble"
    relation = witness["relations"]
    if (
        witness["mode"] != mode
        or type(relation) is not dict
        or set(relation) != RELATIONS
        or any(value is not None and type(value) is not bool for value in relation.values())
    ):
        return unknown("Binding mode or identity relations are malformed")
    selected_ref = event.get("selected")
    if type(selected_ref) is not str or selected_ref not in arrays:
        return unknown("The selected-return array is absent")
    selected = arrays[selected_ref]
    query = witness["query"]

    def actual_query():
        predictions = event.get("predictions")
        return (
            type(query) is dict
            and set(query) == {"query_id", "array", "token"}
            and _integer(query["query_id"], 0, 255)
            and _token(query["token"])
            and type(predictions) is list
            and len(predictions) == 1
            and query["query_id"] == predictions[0]["query_id"]
            and query["array"] == predictions[0]["array"]
            and type(query["array"]) is str
            and query["array"] in arrays
        )

    if mode == "queue" and witness["basis"] == "pre_queue_head_identity":
        head, queue = witness["queue_head"], event["pre_state"].get("queue")
        if (
            type(queue) is not list
            or not queue
            or event.get("predictions")
            or type(head) is not dict
            or set(head) != {"array", "token"}
            or head["array"] != queue[0]
            or not _token(head["token"])
            or head["token"] != witness["selected_token"]
            or relation["selected_is_pre_head"] is not True
            or query is not None
            or witness["query_offset"] is not None
            or witness["query_view"] is not None
            or witness["update"] is not None
            or not _equal(arrays, head["array"], selected_ref)
        ):
            return unknown("The observed queue head does not bind to this selected return")
        return {
            "status": "pass",
            "basis": witness["basis"],
            "queue_head_array": head["array"],
            "selected_array": selected_ref,
            "history": "unknown",
            "scope": SCOPE,
        }
    if mode == "queue" and witness["basis"] == "query_view_mapping":
        if (
            event["pre_state"].get("queue")
            or not actual_query()
            or query["token"] == witness["selected_token"]
            or witness["queue_head"] is not None
            or witness["update"] is not None
            or relation["same_live_base"] is not True
            or not _integer(witness["query_offset"], 0, 255)
        ):
            return unknown("The current query and selected view are not joined")
        chunk = arrays[query["array"]]
        if chunk.ndim != 3 or selected.ndim != 2:
            return unknown("Query/selection view dimensions are unsupported")
        batch = manifest["environment_selection"]["source_num_envs"]
        qv, sv, offset = witness["query_view"], witness["selected_view"], witness["query_offset"]
        if (
            not _view(qv, [batch, chunk.shape[1], chunk.shape[2]], chunk.dtype)
            or not _view(sv, [batch, selected.shape[1]], selected.dtype)
            or selected.dtype != chunk.dtype
            or qv["base_token"] != sv["base_token"]
            or qv["stride"][1] == 0
            or sv["stride"] != [qv["stride"][0], qv["stride"][2]]
            or sv["storage_offset"] != qv["storage_offset"] + offset * qv["stride"][1]
        ):
            return unknown("The saved views do not map one exact native query offset")
        steps = min(event["settings"]["n_action_steps"], chunk.shape[1])
        expected_offset = max(0, steps - event["pre_state"]["queue_maxlen"])
        if (
            offset != expected_offset
            or offset >= chunk.shape[1]
            or selected.shape != chunk[:, offset].shape
            or not np.array_equal(selected, chunk[:, offset])
        ):
            return unknown("Query offset or saved row values contradict native queue retention")
        return {
            "status": "pass",
            "basis": witness["basis"],
            "query_id": query["query_id"],
            "query_offset": offset,
            "selected_array": selected_ref,
            "history": "unknown",
            "scope": SCOPE,
        }
    if mode == "ensemble" and witness["basis"] == "ensemble_argument_return_identity":
        update = witness["update"]
        required = {
            "helper_token",
            "actions_token",
            "prediction_token",
            "return_token",
            "selected_token",
            "call_count",
        }
        if (
            not actual_query()
            or query["token"] == witness["selected_token"]
            or witness["queue_head"] is not None
            or witness["query_offset"] is not None
            or type(update) is not dict
            or set(update) != required
            or type(update["call_count"]) is not int
            or update["call_count"] != 1
            or any(not _token(update[k]) for k in required - {"call_count"})
            or update["actions_token"] != update["prediction_token"]
            or update["prediction_token"] != query["token"]
            or update["return_token"] != update["selected_token"]
            or update["selected_token"] != witness["selected_token"]
            or update["helper_token"] in (query["token"], witness["selected_token"])
            or relation["actions_is_prediction"] is not True
            or relation["selected_is_update_return"] is not True
        ):
            return unknown("Actual update argument/return identity is not joined to this query/selection")
        return {
            "status": "pass",
            "basis": witness["basis"],
            "query_id": query["query_id"],
            "selected_array": selected_ref,
            "history": "unknown",
            "scope": SCOPE,
        }
    return unknown("No supported local relationship is established")
