"""Generated ordinary folders through the selected numeric workflow only."""

import copy
import json
import shutil

import pytest
from test_consumer_numeric import artifacts
from test_consumer_samples import ordinary  # noqa: F401

from policy_doctor.artifacts import InputError
from policy_doctor.consumer_workflow import compare_native_consumer_samples


@pytest.fixture
def job(ordinary, tmp_path):  # noqa: F811
    dataset = ordinary[0]
    ref = tmp_path / "reference_checkpoint"
    cand = tmp_path / "candidate_checkpoint"
    config, stats = artifacts()
    for root in (ref, cand):
        root.mkdir()
        (root / "processor_config.json").write_text(json.dumps(config))
        (root / "statistics.json").write_text(json.dumps(stats))
    candidate = tmp_path / "candidate_dataset"
    shutil.copytree(dataset, candidate)
    return {"reference_dataset": dataset, "reference_checkpoint": ref,
            "candidate_dataset": candidate, "candidate_checkpoint": cand,
            "samples": [{"episode_index": 72, "anchor_index": 5}],
            "read_roots": [tmp_path]}


def run(job, **changes):
    return compare_native_consumer_samples(**{**job, **changes})


def edit_json(root, name, function):
    path = root / name
    value = json.loads(path.read_text())
    function(value)
    path.write_text(json.dumps(value))


def test_same_ordinary_folders_complete_requested_work_and_pool_evidence(job):
    report = run(job, samples=job["samples"] * 3)
    assert report["status"] == "PASS"
    assert report["requested_scope"] == "selected supported inner numeric comparison"
    assert report["numeric_projection_value_effect"] == "EQUAL"
    assert report["coverage"]["fully_projected_windows"] == 3
    assert all(value == "UNKNOWN" for value in report["boundaries"].values())
    assert len(report["evidence"]) == 9  # four files each side plus one slice mapping
    assert "descriptor" not in report["contexts"]["reference"]
    assert "reference" not in report["samples"][0]  # no full trace repetition
    assert len(json.dumps(report)) < 256 * 1024


def test_changed_raw_is_observational_until_selected_equality_is_requested(job):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = job["candidate_dataset"] / "data/chunk-000/episode_000072.parquet"
    table = pq.read_table(path).to_pydict()
    table["action"][5][15] += 1.0
    pq.write_table(pa.table(table), path)
    observation = run(job)
    assert observation["status"] == "PASS"
    assert observation["numeric_projection_value_effect"] == "CHANGED"
    assert observation["samples"][0]["first_value_divergence"]["stage"] == "raw"
    assert observation["samples"][0]["first_value_divergence"]["reference_row"]["caller_selected_index"] == 5
    invariant = run(job, require_equal=True)
    assert invariant["status"] == "ERROR"
    assert invariant["findings"][0]["code"] == "consumer.selected_value_invariant_changed"


def test_active_statistics_change_starts_at_normalization_not_raw(job):
    def change(stats):
        stats[next(iter(stats))]["state"]["left_arm"]["max"][0] = 4.0
    edit_json(job["candidate_checkpoint"], "statistics.json", change)
    report = run(job)
    assert report["status"] == "PASS"
    window = report["samples"][0]
    assert window["active_interpretation_changed"] is True
    assert window["first_value_divergence"]["stage"] == "normalized"
    assert window["numeric_projection_value_effect"] == "CHANGED"


@pytest.mark.parametrize("percentiles,inactive", [(False, "q01"), (True, "min")])
def test_inactive_stats_identity_change_has_no_numeric_effect(job, percentiles, inactive):
    for root in (job["reference_checkpoint"], job["candidate_checkpoint"]):
        edit_json(root, "processor_config.json", lambda config: config["processor_kwargs"].update(
            use_percentiles=percentiles))
    def change(stats):
        stats[next(iter(stats))]["state"]["left_arm"][inactive][0] = -3.0
    edit_json(job["candidate_checkpoint"], "statistics.json", change)
    report = run(job, require_equal=True)
    assert report["status"] == "PASS"
    window = report["samples"][0]
    assert window["artifact_identity_changed"] is True
    assert window["active_interpretation_changed"] is False
    assert window["numeric_projection_value_effect"] == "EQUAL"


def test_actual_selected_mode_change_is_at_representation_with_unknown_full_target(job):
    edit_json(job["candidate_checkpoint"], "processor_config.json", lambda config:
              config["processor_kwargs"].update(use_relative_action=True))
    report = run(job)
    assert report["status"] == "PASS"
    assert report["samples"][0]["first_value_divergence"]["stage"] == "representation"
    assert report["boundaries"]["final_wrapper_model_target_effect"] == "UNKNOWN"


def test_one_bad_window_preserves_other_work_and_context_error(job):
    edit_json(job["candidate_checkpoint"], "processor_config.json", lambda config:
              config["processor_kwargs"].update(max_action_dim=1))
    report = run(job, samples=[job["samples"][0], {"episode_index": 999, "anchor_index": 5}])
    assert report["status"] == "ERROR"
    assert report["contexts"]["candidate"]["status"] == "ERROR"
    assert report["samples"][0]["complete_field_coverage"] is True
    assert report["samples"][1]["numeric_projection_value_effect"] == "UNKNOWN"
    assert report["coverage"]["fully_projected_windows"] == 1


def test_missing_requested_source_is_unknown_without_losing_other_side_evidence(job):
    (job["candidate_dataset"] / "data/chunk-000/episode_000072.parquet").unlink()
    report = run(job, require_equal=True)
    assert report["status"] == "UNKNOWN"
    assert report["numeric_projection_value_effect"] == "UNKNOWN"
    assert report["samples"][0]["source_rows"]["reference"]["status"] == "PASS"
    assert report["samples"][0]["findings"][0]["side"] == "candidate"


def test_missing_relative_stats_preserves_context_error_even_without_sample(job):
    edit_json(job["candidate_checkpoint"], "processor_config.json", lambda config:
              config["processor_kwargs"].update(use_relative_action=True))
    def change(stats):
        del stats[next(iter(stats))]["relative_action"]
    edit_json(job["candidate_checkpoint"], "statistics.json", change)
    report = run(job)
    assert report["status"] == "ERROR"
    assert report["contexts"]["candidate"]["status"] == "ERROR"
    assert report["samples"][0]["numeric_projection_value_effect"] == "UNKNOWN"
    known = [f for f in report["contexts"]["candidate"]["findings"] if f["status"] == "ERROR"]
    assert known
    for cause in known:
        top = next(f for f in report["findings"] if f["code"] == cause["code"]
                   and f.get("pointer") == cause.get("pointer"))
        assert top["side"] == "candidate"
        assert top["cause_scope"] == "saved_native_context"
        assert top["message"] == cause["message"]
        assert top["pointer"] == cause["pointer"]
    assert report["findings"][0]["status"] == "ERROR"


@pytest.mark.parametrize("samples", [[], [{}], [{"episode_index": True, "anchor_index": 0}],
    [{"episode_index": 72, "anchor_index": -1}], [{"episode_index": 72, "anchor_index": 0, "allow_padding": 1}],
    [{"episode_index": 72, "anchor_index": 0, "raw_sample": {}}],
    [{"episode_index": 72, "anchor_index": 0}] * 10])
def test_selector_arguments_refused_before_intake(job, monkeypatch, samples):
    from policy_doctor import consumer_workflow
    monkeypatch.setattr(consumer_workflow, "resolve_native_processor_context", lambda *a, **k: pytest.fail("read"))
    with pytest.raises(InputError):
        run(job, samples=samples)


@pytest.mark.parametrize("key", ["reference_dataset", "reference_checkpoint", "candidate_dataset", "candidate_checkpoint"])
def test_all_paths_authorized_before_any_context_or_source_read(job, tmp_path, monkeypatch, key):
    from policy_doctor import consumer_workflow
    monkeypatch.setattr(consumer_workflow, "resolve_native_processor_context", lambda *a, **k: pytest.fail("read"))
    with pytest.raises(InputError):
        run(job, **{key: tmp_path.parent / "outside"})


def test_literal_parent_tilde_and_unknown_bool_refused_before_intake(job, monkeypatch):
    from policy_doctor import consumer_workflow
    monkeypatch.setattr(consumer_workflow, "resolve_native_processor_context", lambda *a, **k: pytest.fail("read"))
    for value in ("~/checkpoint", "/a/link/../checkpoint"):
        with pytest.raises(InputError):
            run(job, candidate_checkpoint=value)
    with pytest.raises(InputError):
        run(job, require_equal=1)


def test_nine_windows_output_and_application_read_allowances_are_bounded(job):
    report = run(job, samples=[copy.deepcopy(job["samples"][0]) for _ in range(9)])
    assert report["status"] == "PASS"
    assert len(json.dumps(report, allow_nan=False).encode()) <= report["bounds"]["max_report_bytes"]
    assert report["bounds"]["conservative_body_read_allowance_bytes"] == 700 * 1024 * 1024
    assert len(report["evidence"]) == 9


def test_explicit_tail_clamping_reports_current_rows_not_historical_eligibility(job):
    report = run(job, samples=[{"episode_index": 72, "anchor_index": 59, "allow_padding": True}])
    selection = report["samples"][0]["source_rows"]["candidate"]
    assert selection["action_indices"] == [59] * 50
    assert selection["clamped_occurrences"] == 49
    assert selection["native_training_start_eligible"] is False
    assert report["status"] == "PASS"


def test_between_windows_file_mutation_prevents_stable_batch_pass(job, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from policy_doctor import consumer_workflow

    original = consumer_workflow.load_consumer_sample
    candidate = job["candidate_dataset"]
    calls = 0

    def select(*args, **kwargs):
        nonlocal calls
        # Both sides of the first window have completed stable capture. Change
        # an unselected coordinate between windows, so numeric equality alone
        # cannot detect the mixed source versions.
        calls += 1
        if calls == 3:
            path = candidate / "data/chunk-000/episode_000072.parquet"
            table = pq.read_table(path).to_pydict()
            table["action"][0][0] += 1.0
            pq.write_table(pa.table(table), path)
        return original(*args, **kwargs)

    monkeypatch.setattr(consumer_workflow, "load_consumer_sample", select)
    report = run(job, samples=job["samples"] * 2)
    assert report["numeric_projection_value_effect"] == "EQUAL"
    assert report["status"] == "UNKNOWN"
    assert report["input_stability"]["status"] == "UNKNOWN"
    drift = report["input_stability"]["differing_observations"][0]
    assert drift["relative_path"] == "data/chunk-000/episode_000072.parquet"
    assert drift["first_observation"]["sample_index"] == 0
    assert drift["different_observation"]["sample_index"] == 1
    assert all(window["status"] == "PASS" for window in report["samples"])
    assert all(window["source_rows"]["candidate"]["status"] == "PASS" for window in report["samples"])


def test_shared_root_cross_side_versions_are_one_file_identity_and_keep_error(job, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from policy_doctor import consumer_workflow

    original = consumer_workflow.load_consumer_sample
    calls = 0

    def select(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            path = job["reference_dataset"] / "data/chunk-000/episode_000072.parquet"
            table = pq.read_table(path).to_pydict()
            table["action"][5][15] += 1.0
            pq.write_table(pa.table(table), path)
        return original(*args, **kwargs)

    monkeypatch.setattr(consumer_workflow, "load_consumer_sample", select)
    report = run(job, candidate_dataset=job["reference_dataset"], require_equal=True)
    assert report["status"] == "ERROR"
    assert report["input_stability"]["status"] == "UNKNOWN"
    assert report["numeric_projection_value_effect"] == "CHANGED"
    drift = report["input_stability"]["differing_observations"][0]
    assert drift["first_observation"]["side"] == "reference"
    assert drift["different_observation"]["side"] == "candidate"
    assert report["samples"][0]["source_rows"]["reference"]["artifact_refs"]
    assert report["samples"][0]["source_rows"]["candidate"]["artifact_refs"]


def test_distinct_stable_copies_with_different_bytes_are_not_input_instability(job):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = job["candidate_dataset"] / "data/chunk-000/episode_000072.parquet"
    table = pq.read_table(path).to_pydict()
    table["action"][5][15] += 1.0
    pq.write_table(pa.table(table), path)
    report = run(job, samples=job["samples"] * 2)
    assert report["status"] == "PASS"
    assert report["numeric_projection_value_effect"] == "CHANGED"
    assert report["input_stability"]["status"] == "PASS"
    assert report["input_stability"]["differing_observations"] == []


def test_fresh_context_causes_stay_top_level_once_across_nine_windows(job):
    edit_json(job["candidate_checkpoint"], "processor_config.json", lambda config:
              config["processor_kwargs"].update(max_action_dim=1))
    report = run(job, samples=job["samples"] * 9)
    causes = [f for f in report["contexts"]["candidate"]["findings"] if f["status"] == "ERROR"]
    for cause in causes:
        retained = [f for f in report["findings"] if f["code"] == cause["code"]
                    and f.get("pointer") == cause.get("pointer") and f.get("side") == "candidate"]
        assert len(retained) == 1
        assert retained[0]["message"] == cause["message"]
    assert report["status"] == "ERROR"
    assert report["coverage"]["fully_projected_windows"] == 9
    assert len(json.dumps({"status": report["status"], "findings": report["findings"]}).encode()) < 8192
