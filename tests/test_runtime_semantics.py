import json

from policy_doctor.diagnostics import audit


def mutate(case, path, fn):
    data = json.loads(path.read_text())
    fn(data)
    case["put"](path, data)


def run(case):
    return audit(str(case["checkpoint"]), str(case["dataset"]), str(case["deployment"]))


def test_known_relative_model_requires_runtime_interpretation(case):
    mutate(case, case["deployment"], lambda d: d.pop("model_output_representation"))
    findings = run(case)["findings"]
    assert any(f["code"] == "action.model_representation" and f["status"] == "unknown" for f in findings)


def test_decoder_absolute_output_checked_even_when_self_declarations_agree(case):
    def update(d):
        d["command_representation"] = "relative"
        d["training_contract"]["command_representation"] = "relative"

    mutate(case, case["deployment"], update)
    findings = run(case)["findings"]
    assert any(f["code"] == "action.command_representation" and f["status"] == "error" for f in findings)


def test_saved_rename_processor_is_applied_to_raw_interfaces(case):
    mutate(case, case["checkpoint"] / "policy_preprocessor.json",
           lambda d: d["steps"].insert(0, {"registry_name": "rename_observations_processor",
                                         "config": {"rename_map": {"observation.images.raw": "observation.images.front"}}}))
    for path in (case["dataset"] / "meta/info.json", case["deployment"]):
        mutate(case, path, lambda d: d["features"].update({"observation.images.raw": d["features"].pop("observation.images.front")}))
    assert run(case)["status"] == "pass"


def test_runtime_adapter_does_not_rewrite_training_dataset_keys(case):
    mutate(case, case["deployment"], lambda d: d.update(rename_map={"observation.images.front": "observation.images.other"}))
    findings = run(case)["findings"]
    assert any(f["code"] == "deployment.feature_missing" for f in findings)
    assert not any(f["code"] == "dataset.feature_missing" for f in findings)
