"""Adapter contracts with narrow generated component doubles, without payloads."""

import copy
import hashlib
import json

import pytest

from policy_doctor import consumer_preflight, consumer_sample_plan, consumer_workflow
from policy_doctor.artifacts import InputError


@pytest.fixture
def job(tmp_path, monkeypatch):
    paths = {name: str(tmp_path / name) for name in (
        "reference_dataset", "reference_checkpoint", "candidate_dataset", "candidate_checkpoint")}
    roots = [tmp_path]
    plan = consumer_preflight.prepare_consumer_comparison(
        paths["candidate_dataset"], paths["candidate_checkpoint"],
        reference_dataset=paths["reference_dataset"],
        reference_checkpoint=paths["reference_checkpoint"], read_roots=roots)
    artifacts = {"processor_config": {"filename": "processor_config.json", "sha256": "a" * 64,
                                      "size_bytes": 123}}
    contexts = {side: {"status": "PASS", "artifacts": copy.deepcopy(artifacts), "findings": []}
                for side in ("reference", "candidate")}
    selected = [{"episode_index": 72, "anchor_index": 52, "allow_padding": False}]
    readset = [{"path": "meta/info.json", "sha256": "b" * 64, "size_bytes": 100},
               {"path": "meta/episodes.jsonl", "sha256": "c" * 64, "size_bytes": 200}]
    planning = {"planning_status": "PASS", "status": "PASS", "samples": selected,
                "context_artifacts": copy.deepcopy(artifacts), "readset": readset, "findings": []}
    evidence = [{"kind": "selected_file", "value": {"dataset": side, **item}}
                for side in ("reference", "candidate") for item in readset]
    comparison = {"status": "PASS", "numeric_projection_value_effect": "EQUAL", "findings": [],
                  "evidence": evidence, "contexts": copy.deepcopy(contexts), "samples": [],
                  "coverage": {"requested_windows": 1, "fully_projected_windows": 1},
                  "boundaries": {"final_wrapper_model_target_effect": "UNKNOWN"}}
    calls = {"contexts": [], "planning": [], "workflow": []}

    def resolve(path, **kwargs):
        calls["contexts"].append((path, kwargs))
        side = next(side for side in contexts if path == paths[side + "_checkpoint"])
        return copy.deepcopy(contexts[side])

    def choose(dataset, context, **kwargs):
        calls["planning"].append((dataset, context, kwargs))
        return copy.deepcopy(planning)

    def compare(*args, **kwargs):
        calls["workflow"].append((args, kwargs))
        return copy.deepcopy(comparison)

    monkeypatch.setattr(consumer_preflight, "resolve_native_processor_context", resolve)
    monkeypatch.setattr(consumer_sample_plan, "plan_consumer_samples", choose)
    monkeypatch.setattr(consumer_workflow, "compare_native_consumer_samples", compare)
    return {"paths": paths, "roots": roots, "plan": plan, "contexts": contexts,
            "planning": planning, "comparison": comparison, "calls": calls}


def run(job):
    return consumer_preflight.check_consumer_comparison(job["plan"], read_roots=job["roots"])


def codes(report):
    return {finding["code"] for finding in report["findings"]}


def test_automatic_plan_selectors_and_exact_readset_are_bound_to_workflow(job):
    report = run(job)
    assert report["status"] == "pass" and report["numeric_projection_value_effect"] == "EQUAL"
    assert job["calls"]["planning"][0][0] == job["paths"]["reference_dataset"]
    args, kwargs = job["calls"]["workflow"][0]
    assert args == tuple(job["paths"][name] for name in (
        "reference_dataset", "reference_checkpoint", "candidate_dataset", "candidate_checkpoint"))
    assert kwargs == {"samples": job["planning"]["samples"], "require_equal": False,
                      "read_roots": job["roots"]}
    assert report["planning"]["readset"] == job["planning"]["readset"]
    assert all(value == "UNKNOWN" for value in report["boundaries"].values())
    assert any(item["path"].endswith("meta/info.json") for item in report["artifacts"])


@pytest.mark.parametrize("change", ["different_hash", "different_size", "missing", "multiple_versions"])
def test_reference_metadata_drift_or_missing_workflow_read_cannot_pass(job, change):
    evidence = job["comparison"]["evidence"]
    target = next(item for item in evidence if item["value"]["dataset"] == "reference"
                  and item["value"]["path"] == "meta/info.json")
    if change == "missing":
        evidence.remove(target)
    elif change == "multiple_versions":
        extra = copy.deepcopy(target)
        extra["value"]["sha256"] = "d" * 64
        evidence.append(extra)
    elif change == "different_size":
        target["value"]["size_bytes"] += 1
    else:
        target["value"]["sha256"] = "d" * 64
    report = run(job)
    assert report["status"] == "unknown"
    assert "consumer.plan_input_drift" in codes(report)
    assert report["coverage"]["fully_projected_windows"] == 1


def test_candidate_matching_metadata_cannot_substitute_for_missing_reference_read(job):
    job["comparison"]["evidence"] = [item for item in job["comparison"]["evidence"]
                                          if item["value"]["dataset"] == "candidate"]
    assert run(job)["status"] == "unknown"


def test_legitimate_candidate_metadata_changes_are_compared_not_reference_plan_drift(job):
    for item in job["comparison"]["evidence"]:
        if item["value"]["dataset"] == "candidate":
            item["value"]["sha256"] = "d" * 64
            item["value"]["size_bytes"] += 1
    report = run(job)
    assert report["status"] == "pass"
    assert "consumer.plan_input_drift" not in codes(report)


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_saved_context_identity_drift_after_auto_selection_is_unknown(job, side):
    job["comparison"]["contexts"][side]["artifacts"]["processor_config"]["sha256"] = "d" * 64
    report = run(job)
    cause = next(f for f in report["findings"] if f["code"] == "consumer.context_input_drift")
    assert report["status"] == "unknown" and cause["side"] == side


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_explicit_selection_also_binds_initial_and_actual_saved_context(job, side):
    job["plan"]["samples"] = [{"episode_index": 72, "anchor_index": 52}]
    job["comparison"]["contexts"][side]["artifacts"]["processor_config"]["sha256"] = "d" * 64
    report = run(job)
    assert report["status"] == "unknown"
    assert report["planning"] is None
    assert any(f["code"] == "consumer.context_input_drift" and f["side"] == side
               for f in report["findings"])
    captured = {item["sha256"] for item in report["artifacts"]
                if item["path"] == job["paths"][side + "_checkpoint"] + "/processor_config.json"}
    assert captured == {"a" * 64, "d" * 64}


def test_planner_context_mismatch_stops_workflow_without_silently_refreshing(job):
    job["planning"]["context_artifacts"]["processor_config"]["sha256"] = "d" * 64
    report = run(job)
    assert report["status"] == "unknown"
    assert "consumer.comparison_unavailable" in codes(report)
    assert job["calls"]["workflow"] == []


def test_unavailable_auto_plan_does_not_fabricate_or_execute_windows(job):
    job["planning"].update(planning_status="UNKNOWN", status="UNKNOWN", samples=[])
    report = run(job)
    assert report["status"] == "unknown" and job["calls"]["workflow"] == []
    assert report["coverage"]["requested_windows"] == 0


def test_declared_count_warning_does_not_hide_usable_selected_comparison(job):
    job["planning"]["findings"] = [{"code": "consumer.plan_episode_count_mismatch", "status": "WARNING",
        "message": "Declared 251; observed 208. Selection uses observed records."}]
    report = run(job)
    assert report["status"] == "warning"
    assert report["coverage"]["fully_projected_windows"] == 1
    assert job["calls"]["workflow"][0][1]["samples"] == job["planning"]["samples"]


@pytest.mark.parametrize("failure_stage", ["planning", "workflow"])
def test_known_saved_error_survives_later_unavailable_work(job, monkeypatch, failure_stage):
    original = {"code": "consumer.capacity", "status": "ERROR", "message": "Action capacity is too small.",
                "pointer": "/processor_kwargs/max_action_dim", "next": "Inspect the declared capacity."}
    job["contexts"]["candidate"].update(status="ERROR", findings=[original])
    def unavailable(*args, **kwargs):
        raise InputError("Selected data unavailable.")
    module, function = (consumer_sample_plan, "plan_consumer_samples") if failure_stage == "planning" else (
        consumer_workflow, "compare_native_consumer_samples")
    monkeypatch.setattr(module, function, unavailable)
    report = run(job)
    assert report["status"] == "error" and report["numeric_projection_value_effect"] == "UNKNOWN"
    assert "consumer.comparison_unavailable" in codes(report)
    cause = report["findings"][0]
    assert cause["code"] == original["code"] and cause["message"] == original["message"]
    assert cause["pointer"] == original["pointer"] and cause["side"] == "candidate"
    concise = consumer_preflight.concise_consumer_comparison(report)
    assert concise["first_error"] == {"code": original["code"], "message": original["message"],
                                       "side": "candidate", "pointer": original["pointer"]}
    assert concise["full_native_processor_and_model_target"] == "UNKNOWN"


def test_precise_unsupported_context_cause_survives_unavailable_auto_selection(job):
    original = {"code": "consumer.processor_class_unsupported", "status": "UNKNOWN",
                "message": "Saved processor class is outside the supported numeric profile.",
                "pointer": "/processor_class"}
    job["contexts"]["reference"].update(status="UNKNOWN", findings=[original])
    job["planning"].update(planning_status="UNKNOWN", status="UNKNOWN", samples=[], findings=[
        {**original, "cause_scope": "saved_native_context"}])
    report = run(job)
    assert report["status"] == "unknown" and job["calls"]["workflow"] == []
    cause = next(f for f in report["findings"] if f["code"] == original["code"])
    assert cause["message"] == original["message"]
    assert cause["pointer"] == original["pointer"] and cause["side"] == "reference"
    concise = consumer_preflight.concise_consumer_comparison(report)
    assert concise["first_unresolved"] == {"code": original["code"], "message": original["message"],
                                            "pointer": original["pointer"], "side": "reference"}
    assert concise["first_error"] is None


def test_readset_drift_cannot_erase_independently_proven_error(job):
    job["comparison"]["status"] = "ERROR"
    job["comparison"]["findings"] = [{"code": "consumer.selected_value_invariant_changed",
        "status": "ERROR", "message": "Selected value changed under requested equality."}]
    job["comparison"]["evidence"][0]["value"]["sha256"] = "d" * 64
    report = run(job)
    assert report["status"] == "error"
    assert {"consumer.plan_input_drift", "consumer.selected_value_invariant_changed"} <= codes(report)


@pytest.mark.parametrize("require_equal", [False, True])
def test_equality_intent_forwarded_and_observational_change_is_not_automatic_error(job, monkeypatch, require_equal):
    job["plan"]["require_equal"] = require_equal
    def changed(*args, **kwargs):
        job["calls"]["workflow"].append((args, kwargs))
        value = copy.deepcopy(job["comparison"])
        value["numeric_projection_value_effect"] = "CHANGED"
        if kwargs["require_equal"]:
            value["status"] = "ERROR"
            value["findings"] = [{"code": "consumer.selected_value_invariant_changed", "status": "ERROR",
                                  "message": "Changed selected values violate requested equality."}]
        return value
    monkeypatch.setattr(consumer_workflow, "compare_native_consumer_samples", changed)
    report = run(job)
    assert report["status"] == ("error" if require_equal else "pass")
    assert report["numeric_projection_value_effect"] == "CHANGED"
    assert job["calls"]["workflow"][0][1]["require_equal"] is require_equal


def test_explicit_selectors_bypass_auto_planning_without_changing_them(job, monkeypatch):
    selected = [{"episode_index": 250, "anchor_index": 188, "allow_padding": True}]
    job["plan"]["samples"] = selected
    monkeypatch.setattr(consumer_sample_plan, "plan_consumer_samples", lambda *a, **k: pytest.fail("auto plan"))
    report = run(job)
    assert report["planning"] is None
    assert job["calls"]["workflow"][0][1]["samples"] == selected


@pytest.mark.parametrize("path_name", ["reference_dataset", "reference_checkpoint", "candidate_dataset", "candidate_checkpoint"])
def test_all_four_paths_preauthorized_before_any_context_or_planner_read(tmp_path, monkeypatch, path_name):
    paths = {name: tmp_path / name for name in (
        "reference_dataset", "reference_checkpoint", "candidate_dataset", "candidate_checkpoint")}
    paths[path_name] = tmp_path.parent / "outside"
    monkeypatch.setattr(consumer_preflight, "resolve_native_processor_context", lambda *a, **k: pytest.fail("read"))
    monkeypatch.setattr(consumer_sample_plan, "plan_consumer_samples", lambda *a, **k: pytest.fail("read"))
    with pytest.raises(InputError):
        consumer_preflight.prepare_consumer_comparison(paths["candidate_dataset"], paths["candidate_checkpoint"],
            reference_dataset=paths["reference_dataset"], reference_checkpoint=paths["reference_checkpoint"],
            read_roots=[tmp_path])


@pytest.mark.parametrize("samples", [[], [{}], [{"episode_index": True, "anchor_index": 0}],
    [{"episode_index": 72, "anchor_index": -1}], [{"episode_index": 72, "anchor_index": 0, "allow_padding": 1}],
    [{"episode_index": 72, "anchor_index": 0, "raw_sample": {}}],
    [{"episode_index": 72, "anchor_index": 0}] * 10])
def test_invalid_selectors_refused_during_preparation(job, samples):
    with pytest.raises(InputError):
        consumer_preflight.prepare_consumer_comparison(job["paths"]["candidate_dataset"],
            job["paths"]["candidate_checkpoint"], reference_checkpoint=job["paths"]["reference_checkpoint"],
            samples=samples, read_roots=job["roots"])
    assert job["calls"] == {"contexts": [], "planning": [], "workflow": []}


def test_no_comparison_requested_keeps_existing_path_and_strict_boolean(job):
    assert consumer_preflight.prepare_consumer_comparison("uninspected-dataset", "uninspected-checkpoint") is None
    with pytest.raises(InputError):
        consumer_preflight.prepare_consumer_comparison("dataset", "checkpoint", require_equal=1)
    with pytest.raises(InputError):
        consumer_preflight.prepare_consumer_comparison("dataset", None, reference_checkpoint="reference")


def test_concise_result_keeps_first_divergence_source_location_and_error(job):
    divergence = {"stage": "normalized", "field": "action.left_arm", "coordinate": [0, 2],
        "change": "value", "reference_row": {"episode_index": 72, "physical_row": 52},
        "candidate_row": {"episode_index": 72, "physical_row": 52}, "evidence": [2],
        "unbounded_internal_detail": "omit"}
    job["comparison"]["samples"] = [{"first_observed_divergence": None},
                                        {"first_observed_divergence": divergence}]
    report = run(job)
    concise = consumer_preflight.concise_consumer_comparison(report)
    assert concise["first_observed_divergence"] == {key: value for key, value in divergence.items()
                                                     if key != "unbounded_internal_detail"}
    assert concise["coverage"]["fully_projected_windows"] == 1
    assert concise["first_error"] is None
    assert concise["full_native_processor_and_model_target"] == "UNKNOWN"


def test_concise_inactive_identity_change_keeps_equal_value_effect_and_false_active_change(job):
    job["comparison"]["samples"] = [{"artifact_identity_changed": True,
        "active_interpretation_changed": False, "numeric_projection_value_effect": "EQUAL",
        "first_observed_divergence": None, "first_value_divergence": None}]
    concise = consumer_preflight.concise_consumer_comparison(run(job))
    assert concise["status"] == "pass" and concise["numeric_projection_value_effect"] == "EQUAL"
    assert concise["artifact_identity_changed"] is True
    assert concise["active_interpretation_changed"] is False
    assert concise["first_value_divergence"] is concise["first_final_value_difference"] is None


@pytest.mark.parametrize("observations,expected", [([False, False], False), ([False, None], None),
    ([False, True, None], True), ([], None)])
def test_concise_identity_aggregate_does_not_turn_unobserved_coverage_into_false(job, observations, expected):
    job["comparison"]["samples"] = [{"artifact_identity_changed": value,
                                        "active_interpretation_changed": value} for value in observations]
    concise = consumer_preflight.concise_consumer_comparison(run(job))
    assert concise["artifact_identity_changed"] is expected
    assert concise["active_interpretation_changed"] is expected


@pytest.mark.parametrize("long_directory", [False, True])
def test_concise_dtype_first_keeps_later_value_cause_and_bound_source_locator(job, long_directory):
    comparison = job["comparison"]
    selection = {"episode_index": 72, "anchor_index": 52, "allow_padding": False}
    row = {"sample_row": 0, "caller_selected_index": 52, "source_membership": "UNKNOWN"}
    dtype_cause = {"stage": "raw", "field": "state.left_arm", "coordinate": [0, 0], "change": "dtype"}
    value_cause = {"stage": "normalized", "field": "action.left_arm", "coordinate": [0, 2],
                   "reference_row": row, "candidate_row": row}
    final_witness = {"field": "action.left_arm", "coordinate": [0, 0],
                     "reference": 0.125, "candidate": 0.25, "reference_row": row, "candidate_row": row}
    window = {"artifact_identity_changed": True, "active_interpretation_changed": True,
        "requested_selection": selection, "first_observed_divergence": dtype_cause,
        "first_value_divergence": value_cause, "first_value_differences": [final_witness], "source_rows": {}}
    comparison["samples"] = [window]
    comparison["numeric_projection_value_effect"] = "CHANGED"
    comparison["sources"] = {}
    for side in ("reference", "candidate"):
        refs = []
        for path, digest in (("data/chunk-000/episode_000072.parquet", "d" * 64),
                             ("meta/modality.json", "e" * 64), ("meta/tasks.jsonl", "f" * 64)):
            refs.append(len(comparison["evidence"]))
            comparison["evidence"].append({"kind": "selected_file", "value": {
                "dataset": side, "path": path, "sha256": digest, "size_bytes": 123}})
        mapping_index = len(comparison["evidence"])
        comparison["evidence"].append({"kind": "declared_slices", "value": [{
            "modality": "action", "key": "left_arm", "column": "action", "start": 15, "end": 22}]})
        window["source_rows"][side] = {"field_mapping_ref": mapping_index, "artifact_refs": refs}
        comparison["sources"][side + "_dataset"] = ("/bounded/" + "x" * 600) if long_directory else (
            job["paths"][side + "_dataset"])
    concise = consumer_preflight.concise_consumer_comparison(run(job))
    assert concise["first_observed_divergence"] == dtype_cause
    assert concise["first_final_value_difference"] == final_witness
    witness = concise["first_value_divergence"]
    assert witness["stage"] == "normalized" and witness["field"] == "action.left_arm"
    assert witness["requested_selection"] == selection
    assert witness["reference_row"] == witness["candidate_row"] == row
    final_source = concise["first_final_value_source"]
    assert final_source["stage"] == "clipped" and final_source["coordinate"] == [0, 0]
    assert final_source["field"] == final_witness["field"]
    assert final_source["requested_selection"] == selection
    assert final_source["reference_row"] == final_source["candidate_row"] == row
    assert final_source["source_identity_from"] == "first_value_divergence"
    assert "reference_source" not in final_source and "candidate_source" not in final_source
    for side in ("reference", "candidate"):
        assert witness["source_mapping"][side] == {"column": "action", "declared_slice": [15, 22],
                                                   "source_coordinate": 17}
        assert final_source["source_mapping"][side] == {"column": "action", "declared_slice": [15, 22],
                                                        "source_coordinate": 15}
        source = witness[side + "_source"]
        directory = comparison["sources"][side + "_dataset"]
        assert source["dataset_directory"] == directory[:512]
        assert source["directory_truncated"] is long_directory
        assert len(source["files"]) == 2
        assert [item["path"] for item in source["files"]] == [
            "data/chunk-000/episode_000072.parquet", "meta/modality.json"]
        assert [item["sha256"] for item in source["files"]] == ["d" * 64, "e" * 64]
        if long_directory:
            assert source["full_directory_sha256"] == hashlib.sha256(directory.encode()).hexdigest()
        else:
            assert "full_directory_sha256" not in source
    assert concise["full_native_processor_and_model_target"] == "UNKNOWN"


def test_non_ascii_source_and_causes_fit_mcp_summary_without_losing_value_locator(job):
    from policy_doctor import server

    comparison = job["comparison"]
    row = {"sample_row": 0, "caller_selected_index": 52, "source_membership": "UNKNOWN"}
    selection = {"episode_index": 72, "anchor_index": 52, "allow_padding": False}
    column = "观测坐标" * 64
    window = {"artifact_identity_changed": True, "active_interpretation_changed": False,
        "requested_selection": selection, "source_rows": {},
        "first_observed_divergence": {"stage": "raw", "field": "state.left_arm",
                                       "coordinate": [0, 0], "change": "dtype"},
        "first_value_divergence": {"stage": "normalized", "field": "action.left_arm",
            "coordinate": [0, 2], "reference_row": row, "candidate_row": row},
        "first_value_differences": [{"field": "action.left_arm", "coordinate": [0, 0],
            "reference": 0.125, "candidate": 0.25, "reference_row": row, "candidate_row": row}]}
    comparison.update(samples=[window], numeric_projection_value_effect="CHANGED", sources={})
    for side in ("reference", "candidate"):
        refs = []
        for path, digest in (("data/chunk-000/episode_000072.parquet", "d" * 64),
                             ("meta/modality.json", "e" * 64)):
            refs.append(len(comparison["evidence"]))
            comparison["evidence"].append({"kind": "selected_file", "value": {
                "dataset": side, "path": path, "sha256": digest, "size_bytes": 123}})
        mapping = len(comparison["evidence"])
        comparison["evidence"].append({"kind": "declared_slices", "value": [{"modality": "action",
            "key": "left_arm", "column": column, "start": 15, "end": 22}]})
        window["source_rows"][side] = {"field_mapping_ref": mapping, "artifact_refs": refs}
        comparison["sources"][side + "_dataset"] = "/" + "机器人路径" * 128
    cause = {"code": "原因" * 128, "message": "无法读取处理器配置" * 128,
             "pointer": "/" + "字段" * 256}
    job["contexts"]["candidate"].update(status="ERROR", findings=[{**cause, "status": "ERROR"}])
    comparison["findings"] = [{**cause, "status": "UNKNOWN", "side": "reference"}]
    component = run(job)
    full = {"kind": "dataset_preflight", "status": "error", "scope": "selected numeric comparison",
            "components": {"consumer_comparison": component}, "findings": component["findings"]}
    summary = server._summary(full)
    assert len(json.dumps(summary, allow_nan=False).encode()) <= 8192
    assert summary["full_report_sha256"] == hashlib.sha256(json.dumps(
        full, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    concise = summary["components"]["consumer_comparison"]
    assert concise["status"] == "error" and concise["numeric_projection_value_effect"] == "CHANGED"
    assert concise["artifact_identity_changed"] is True
    assert concise["active_interpretation_changed"] is False
    assert concise["first_observed_divergence"]["change"] == "dtype"
    assert concise["first_value_divergence"]["coordinate"] == [0, 2]
    assert concise["first_final_value_difference"]["coordinate"] == [0, 0]
    assert concise["first_final_value_source"]["source_identity_from"] == "first_value_divergence"

    def escaped_bytes(text):
        return len(json.dumps(text, ensure_ascii=True).encode()) - 2

    for side in ("reference", "candidate"):
        source = concise["first_value_divergence"][side + "_source"]
        directory = comparison["sources"][side + "_dataset"]
        assert escaped_bytes(source["dataset_directory"]) <= 512
        assert directory.startswith(source["dataset_directory"])
        assert source["directory_truncated"] is True
        assert source["full_directory_sha256"] == hashlib.sha256(directory.encode()).hexdigest()
        assert [item["sha256"] for item in source["files"]] == ["d" * 64, "e" * 64]
        for key, expected in (("first_value_divergence", 17), ("first_final_value_source", 15)):
            mapping = concise[key]["source_mapping"][side]
            assert mapping["source_coordinate"] == expected and mapping["declared_slice"] == [15, 22]
            assert escaped_bytes(mapping["column"]) <= 128
            assert column.startswith(mapping["column"]) and mapping["column_truncated"] is True
            assert mapping["full_column_sha256"] == hashlib.sha256(column.encode()).hexdigest()
    for key in ("first_error", "first_unresolved"):
        item = concise[key]
        assert {"code", "message", "pointer"} <= set(item["text_truncated"])
        for field, bound in (("code", 128), ("message", 512), ("pointer", 256)):
            assert escaped_bytes(item[field]) <= bound
            assert cause[field].startswith(item[field])
    assert concise["full_native_processor_and_model_target"] == "UNKNOWN"
