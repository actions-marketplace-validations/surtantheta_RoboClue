"""Saved normalizer statistics judged by what LeRobot 146d847 does with them at deployment and fine-tuning.

Patterns from the simulated beta: a checkpoint whose saved statistics were shifted (mean +10, std x3) against its
dataset went unflagged, and constant dimensions were never reported. Warnings stay conditional: a constant
dimension can be legitimate, and different statistics are not proof of a faulty checkpoint.
"""
import json

import numpy as np
from safetensors.numpy import save_file

from policy_doctor.artifacts import load_dataset
from policy_doctor.diagnostics import audit, dataset_statistics_findings

SAVED = {"action.mean": [1., 0.5], "action.std": [2., 0.25],
         "observation.state.mean": [10., 0.5], "observation.state.std": [2., 0.25]}


def save(case, side, **changes):
    stats = {k: np.array(changes.get(k.replace(".", "_"), v), dtype=float) for k, v in SAVED.items()}
    save_file(stats, case["checkpoint"] / f"{side}.safetensors")


def dataset_stats(case, stats):
    (case["dataset"] / "meta" / "stats.json").write_text(json.dumps(stats))


def matching():
    return {"observation.state": {"mean": [10., 0.5], "std": [2., 0.25], "min": [0., 0.], "max": [20., 1.]},
            "action": {"mean": [1., 0.5], "std": [2., 0.25], "min": [-5., 0.], "max": [5., 1.]}}


def found(report, code):
    return [f for f in report["findings"] if f["code"] == code]


def run(case):
    return audit(str(case["checkpoint"]), str(case["dataset"]))


def test_constant_input_dimension_is_a_conditional_warning(case):
    save(case, "pre", observation_state_std=[2., 0.])
    hit = found(run(case), "normalization.constant_input")
    assert len(hit) == 1 and hit[0]["status"] == "warning"
    assert "dimension(s) 1 (gripper)" in hit[0]["message"] and "1e+08" in hit[0]["message"]
    assert "normalize_processor.py:361-365" in hit[0]["message"] and "physical effect are unknown" in hit[0]["message"]
    assert hit[0]["evidence"][0]["indices"] == [1] and hit[0]["evidence"][0]["saved_values"] == [0.5]


def test_constant_action_dimension_is_reported_without_alarm(case):
    # Public LeKiwi pick-and-place: base velocities never move, so the policy never drives the base. Legitimate.
    save(case, "post", action_std=[2., 0.])
    report = run(case)
    hit = found(report, "normalization.constant_dimension")
    assert len(hit) == 1 and hit[0]["status"] == "pass" and "1 (gripper)" in hit[0]["message"]
    assert hit[0]["evidence"][0]["inverse_scales"] == [0.] and "normalize_processor.py:363-364" in hit[0]["message"]
    assert not found(report, "normalization.constant_input")  # The preprocessor's action stats are training-only.


def test_matching_dataset_statistics_pass_with_stats_json_evidence(case):
    dataset_stats(case, matching())
    hits = found(run(case), "normalization.dataset_statistics")
    assert {(f["status"], f["message"].split(":")[0]) for f in hits} == {("pass", "pre"), ("pass", "post")}
    assert all(f["evidence"][1]["path"].endswith("meta/stats.json") and f["evidence"][1]["value_kind"] == "source"
               for f in hits)


def test_shifted_saved_statistics_warn_with_the_deployment_consequence(case):
    # The beta deployment tester's plant: saved state mean +10 and std x3 against the dataset.
    dataset_stats(case, matching())
    save(case, "pre", observation_state_mean=[20., 0.5], observation_state_std=[6., 0.25])
    hit = [f for f in found(run(case), "normalization.dataset_statistics") if f["status"] != "pass"]
    assert len(hit) == 1 and hit[0]["status"] == "warning" and hit[0]["message"].startswith("pre:observation.state")
    assert "offset -1.667 and scale x0.3333 on dimension 0 (shoulder)" in hit[0]["message"]
    assert "lerobot_eval.py:808-818" in hit[0]["message"] and "not evidence of a faulty checkpoint" in hit[0]["message"]
    assert hit[0]["evidence"][0]["offsets"] == [-1.6667, 0.0]


def test_named_training_dataset_adds_causes_without_claiming_a_fault(case):
    dataset_stats(case, matching())
    save(case, "pre", observation_state_mean=[20., 0.5], observation_state_std=[6., 0.25])
    (case["checkpoint"] / "train_config.json").write_text(json.dumps({"dataset": {"repo_id": "org/data"}, "resume": False}))
    report = run(case)
    hit = next(f for f in found(report, "normalization.dataset_statistics") if f["status"] == "warning")
    assert "names org/data as its training dataset" in hit["message"] and "hotswap_stats" in hit["message"]
    assert "none of these by itself makes the checkpoint faulty" in hit["message"]
    source = hit["evidence"][-1]
    assert source["pointer"] == "/dataset/repo_id" and source["value"] == "org/data" and source["value_kind"] == "source"
    assert any(a["role"] == "training_config" for a in report["artifacts"])


def test_small_differences_pass_within_the_stated_threshold(case):
    stats = matching()
    stats["observation.state"]["mean"] = [10.2, 0.5]
    dataset_stats(case, stats)
    hit = next(f for f in found(run(case), "normalization.dataset_statistics") if f["message"].startswith("pre:"))
    assert hit["status"] == "pass" and "stays close to them" in hit["message"] and "offset 0.1 " in hit["message"]


def test_rebuilding_policy_types_are_unknown_not_warned(case):
    config = json.loads((case["checkpoint"] / "config.json").read_text())
    (case["checkpoint"] / "config.json").write_text(json.dumps(dict(config, type="evo1")))
    dataset_stats(case, matching())
    save(case, "pre", observation_state_mean=[20., 0.5], observation_state_std=[6., 0.25])
    hit = next(f for f in found(run(case), "normalization.dataset_statistics") if f["status"] != "pass")
    assert hit["status"] == "unknown" and "does not model" in hit["message"]


def test_dataset_lacking_quantiles_warns_only_about_fine_tuning(case):
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        text = (case["checkpoint"] / name).read_text().replace('"STATE": "MEAN_STD"', '"STATE": "QUANTILES"')
        (case["checkpoint"] / name).write_text(text)
    save_file({**{k: np.array(v) for k, v in SAVED.items()}, "observation.state.q01": np.array([1., 0.]),
               "observation.state.q99": np.array([19., 1.])}, case["checkpoint"] / "pre.safetensors")
    dataset_stats(case, matching())
    hit = next(f for f in found(run(case), "normalization.dataset_statistics") if f["message"].startswith("pre:"))
    assert hit["status"] == "warning" and "lacks q01/q99" in hit["message"]
    assert "Deploying the checkpoint is unaffected" in hit["message"] and "normalize_processor.py:389-395" in hit["message"]


def test_dataset_without_a_feature_entry_warns_of_unnormalized_fine_tuning(case):
    stats = matching()
    del stats["observation.state"]
    dataset_stats(case, stats)
    hit = next(f for f in found(run(case), "normalization.dataset_statistics") if f["message"].startswith("pre:"))
    assert hit["status"] == "warning" and "has no statistics for" in hit["message"] and "332-334" in hit["message"]


def test_unreadable_training_config_only_drops_the_explanation(case):
    dataset_stats(case, matching())
    save(case, "pre", observation_state_mean=[20., 0.5], observation_state_std=[6., 0.25])
    (case["checkpoint"] / "train_config.json").write_text("{not json")
    hit = next(f for f in found(run(case), "normalization.dataset_statistics") if f["status"] == "warning")
    assert "not evidence of a faulty checkpoint" in hit["message"]


def test_no_dataset_statistics_and_healthy_saved_statistics_add_nothing(case):
    report = run(case)
    assert not [f for f in report["findings"] if f["code"].startswith("normalization.") and f["code"] not in
                ("normalization.statistics", "normalization.pair_mode", "normalization.config_mismatch")]


def test_statistics_are_compared_at_their_own_width_under_a_stale_declaration(case):
    # Public SmolVLA LeKiwi: config declares a [6] state while saved and dataset statistics have 9 values.
    info = json.loads((case["dataset"] / "meta" / "info.json").read_text())
    info["features"]["observation.state"].update(shape=[3], names=["shoulder", "gripper", "wrist"])
    (case["dataset"] / "meta" / "info.json").write_text(json.dumps(info))
    stats = matching()
    stats["observation.state"] = {"mean": [10., 0.5, 1.], "std": [2., 0.25, 1.]}
    dataset_stats(case, stats)
    save_file({**{k: np.array(v) for k, v in SAVED.items()}, "observation.state.mean": np.array([20., 0.5, 1.]),
               "observation.state.std": np.array([6., 0.25, 1.])}, case["checkpoint"] / "pre.safetensors")
    hit = next(f for f in found(run(case), "normalization.dataset_statistics") if f["message"].startswith("pre:"))
    assert hit["status"] == "warning" and "dimension 0 (shoulder)" in hit["message"]


def test_dataset_constant_dimensions_before_training(case):
    stats = matching()
    stats["action"].update(mean=[1., 0.3], std=[2., 0.], min=[-5., 0.3], max=[5., 0.3])
    stats["observation.state"].update(std=[0., 0.25], min=[10., 0.], max=[10., 1.])
    dataset_stats(case, stats)
    findings = {f["evidence"][0]["pointer"].removeprefix("/"): f
                for f in dataset_statistics_findings(load_dataset(case["dataset"]))}
    action, state = findings["action"], findings["observation.state"]
    assert action["status"] == state["status"] == "warning" and action["code"] == "dataset.constant_dimension"
    assert "1 (gripper)" in action["message"] and "exactly zero std returns the saved mean" in action["message"]
    assert "0 (shoulder)" in state["message"] and "configured epsilon" in state["message"] and state["indices"] == [0]
    assert action["evidence"][0]["path"].endswith("meta/stats.json") and action["evidence"][0]["value_kind"] == "source"


def test_dataset_dimension_that_varies_by_range_is_not_constant(case):
    stats = matching()
    stats["action"].update(std=[2., 0.])  # min/max still show dimension 1 moving from 0 to 1.
    dataset_stats(case, stats)
    findings = dataset_statistics_findings(load_dataset(case["dataset"]))
    assert not any(f['code'] == 'dataset.constant_dimension' for f in findings)
    contradiction = next(f for f in findings if f['code'] == 'dataset.statistics_inconsistent')
    assert contradiction['status'] == 'error' and contradiction['constraint'] == 'zero_std_with_nonzero_range'


def test_bookkeeping_columns_are_never_reported_as_constant_inputs(case):
    info = json.loads((case["dataset"] / "meta" / "info.json").read_text())
    info["features"]["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    (case["dataset"] / "meta" / "info.json").write_text(json.dumps(info))
    stats = matching()
    stats["task_index"] = {"mean": [0.0], "std": [0.0], "min": [0], "max": [0]}
    dataset_stats(case, stats)
    assert dataset_statistics_findings(load_dataset(case["dataset"])) == []


# Regressions from the independent validator of 1bd599b (validator-032/REPORT.md).

def statuses(report, code, prefix):
    return [f for f in found(report, code) if f["message"].startswith(prefix)]


def test_constant_saved_action_that_the_dataset_moves_is_a_warning(case):
    # Blocker 1: public LeKiwi checkpoint (base velocities constant) against data where the base moves.
    save(case, "post", action_mean=[1., 0.5], action_std=[2., 0.])
    stats = matching()
    dataset_stats(case, stats)
    report = run(case)
    hit = statuses(report, "normalization.dataset_statistics", "post:action")
    assert len(hit) == 1 and hit[0]["status"] == "warning"
    assert "1 (gripper) have exactly zero saved spread" in hit[0]["message"]
    assert "inverse returns the saved mean regardless of model output" in hit[0]["message"]
    assert hit[0]["evidence"][0]["constant_in_checkpoint_but_not_dataset"] == [1]


def test_constant_saved_input_that_the_dataset_varies_is_a_warning(case):
    save(case, "pre", observation_state_std=[2., 0.])
    dataset_stats(case, matching())
    hit = statuses(run(case), "normalization.dataset_statistics", "pre:observation.state")
    assert hit[0]["status"] == "warning" and "std + eps" in hit[0]["message"]


def test_narrower_data_inside_the_saved_range_passes(case):
    stats = matching()
    stats["observation.state"]["std"] = [0.6, 0.25]
    dataset_stats(case, stats)
    hit = statuses(run(case), "normalization.dataset_statistics", "pre:observation.state")
    assert hit[0]["status"] == "pass" and "x0.3 on dimension 0" in hit[0]["message"]


def test_wider_data_than_the_saved_statistics_warns(case):
    stats = matching()
    stats["observation.state"]["std"] = [6., 0.25]
    dataset_stats(case, stats)
    hit = statuses(run(case), "normalization.dataset_statistics", "pre:observation.state")
    assert hit[0]["status"] == "warning" and "normalized scale > 2" in hit[0]["message"]


def test_malformed_dataset_statistics_are_reported_not_crashed_on(case):
    for bad in (["a", "b"], [10 ** 400, 1.], [[1., 2.]], [True, 1.]):
        stats = matching()
        stats["observation.state"]["std"] = bad
        dataset_stats(case, stats)
        report = run(case)
        hit = statuses(report, "normalization.dataset_statistics", "pre:observation.state")
        assert hit[0]["status"] == "warning" and "has std values that are not lists of finite numbers" in hit[0]["message"], bad
        json.dumps(report, allow_nan=False)


def test_odd_dataset_entries_are_described_without_invented_behavior(case):
    stats = matching()
    stats["observation.state"] = {}
    dataset_stats(case, stats)
    assert "has no statistics for" in statuses(run(case), "normalization.dataset_statistics", "pre:")[0]["message"]
    stats["observation.state"] = "oops"
    dataset_stats(case, stats)
    message = statuses(run(case), "normalization.dataset_statistics", "pre:")[0]["message"]
    assert "is a str, not a statistics object" in message and "raises" not in message


def test_keys_the_normalizer_skips_are_not_judged(case):
    for name in ("policy_preprocessor.json",):
        document = json.loads((case["checkpoint"] / name).read_text())
        document["steps"][1]["config"]["normalize_observation_keys"] = ["observation.environment_state"]
        (case["checkpoint"] / name).write_text(json.dumps(document))
    save(case, "pre", observation_state_std=[2., 0.])
    assert not found(run(case), "normalization.constant_input")


def test_plugin_policy_types_are_unknown(case):
    config = json.loads((case["checkpoint"] / "config.json").read_text())
    (case["checkpoint"] / "config.json").write_text(json.dumps(dict(config, type="my_plugin_policy")))
    dataset_stats(case, matching())
    save(case, "pre", observation_state_mean=[20., 0.5], observation_state_std=[6., 0.25])
    hit = statuses(run(case), "normalization.dataset_statistics", "pre:")
    assert hit[0]["status"] == "unknown" and "does not model" in hit[0]["message"]


def test_unusable_epsilon_and_symlinked_training_config_do_not_crash(case):
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        text = (case["checkpoint"] / name).read_text().replace('"eps": 1e-08', '"eps": ' + str(10 ** 400))
        (case["checkpoint"] / name).write_text(text)
    dataset_stats(case, matching())
    run(case)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        text = (case["checkpoint"] / name).read_text().replace('"eps": ' + str(10 ** 400), '"eps": 1e-08')
        (case["checkpoint"] / name).write_text(text)
    save(case, "pre", observation_state_mean=[20., 0.5], observation_state_std=[6., 0.25])
    (case["checkpoint"] / "train_config.json").symlink_to(case["checkpoint"] / "config.json")
    report = run(case)
    hit = statuses(report, "normalization.dataset_statistics", "pre:")[0]
    assert "do not establish what data" in hit["message"]
    assert not any(a["role"] == "training_config" for a in report["artifacts"])


def test_offsets_beyond_float_range_stay_reportable(case):
    stats = matching()
    stats["observation.state"]["mean"] = [1e308, 0.5]
    dataset_stats(case, stats)
    save(case, "pre", observation_state_mean=[-1e308, 0.5])
    report = run(case)
    hit = statuses(report, "normalization.dataset_statistics", "pre:")[0]
    assert hit["status"] == "warning" and "beyond float range" in hit["message"]
    json.dumps(report, allow_nan=False)


def test_base_checkpoint_without_statistics_says_why(case):
    save_file({k: np.array(v) for k, v in SAVED.items() if k.startswith("action")}, case["checkpoint"] / "pre.safetensors")
    hit = next(f for f in found(run(case), "normalization.statistics") if f["message"].startswith("pre:observation.state"))
    assert hit["status"] == "error" and hit["message"] == "pre:observation.state: Missing mean statistics."
    assert "as base checkpoints ship" in hit["remedy"]


def test_dataset_helper_survives_malformed_values_and_finds_collapsed_quantiles(case):
    stats = matching()
    stats["action"].update(std=[2., 0.], min=[-5., 0.3], max=[5., 0.3], mean="bad")
    stats["observation.state"].update(q01=[1., 0.5], q99=[19., 0.5])
    dataset_stats(case, stats)
    findings = {f["code"]: f for f in dataset_statistics_findings(load_dataset(case["dataset"]))}
    assert findings["dataset.statistics_unassessed"]["status"] == "unknown"
    collapsed = findings["dataset.collapsed_quantiles"]
    assert collapsed["indices"] == [1] and "configured epsilon" in collapsed["message"] and "397-421" in collapsed["message"]


def test_negative_dataset_std_is_invalid_not_constant(case):
    stats = matching()
    stats["action"]["std"] = [2., -0.1]
    dataset_stats(case, stats)
    codes_found = [f["code"] for f in dataset_statistics_findings(load_dataset(case["dataset"]))]
    assert codes_found == ["dataset.statistics_invalid"]


def test_dataset_statistics_of_another_width_warn_for_fine_tuning(case):
    stats = matching()
    stats["observation.state"] = {"mean": [10., 0.5, 1.], "std": [2., 0.25, 1.]}
    dataset_stats(case, stats)
    hit = statuses(run(case), "normalization.dataset_statistics", "pre:")[0]
    assert hit["status"] == "warning" and "has 3 values per statistic" in hit["message"] and "have 2" in hit["message"]
