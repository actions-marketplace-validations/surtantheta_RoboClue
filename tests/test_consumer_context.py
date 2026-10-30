"""Generated native-layout artifacts; no real run or native execution is claimed."""

import copy
import json
import os

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.consumer_context import (
    ACTION_WIDTHS,
    DEFAULTS,
    PROFILE,
    STATE_WIDTHS,
    load_consumer_numeric_parameters,
    resolve_native_processor_context,
)


def native_layout():
    # Generated independently of the resolver: native JSON enum NAMES and the
    # source-pinned G1 field order, not a caller-authored product recipe.
    state = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    action = [
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    modalities = {
        "video": {"delta_indices": [0], "modality_keys": ["ego_view"]},
        "state": {"delta_indices": [0], "modality_keys": state},
        "action": {
            "delta_indices": list(range(50)),
            "modality_keys": action,
            "action_configs": [
                {
                    "rep": "RELATIVE" if i < 2 else "ABSOLUTE",
                    "type": "NON_EEF",
                    "format": "DEFAULT",
                    "state_key": None,
                }
                for i in range(7)
            ],
        },
        "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
    }
    config = {
        "processor_class": "Gr00tN1d7Processor",
        "processor_kwargs": {
            "modality_configs": {PROFILE: modalities},
            "max_state_dim": 64,
            "max_action_dim": 64,
            "max_action_horizon": 50,
            "use_percentiles": False,
            "use_relative_action": False,
            "clip_outliers": True,
            "apply_sincos_state_encoding": False,
            "use_mean_std": False,
            "exclude_state": False,
            "state_dropout_prob": 0.0,
        },
    }
    stats = {PROFILE: {}}
    for modality, widths in (("state", STATE_WIDTHS), ("action", ACTION_WIDTHS)):
        stats[PROFILE][modality] = {
            key: {
                "min": [-2.0] * width,
                "max": [2.0] * width,
                "mean": [0.5] * width,
                "std": [0.25] * width,
                "q01": [-1.0] * width,
                "q99": [1.0] * width,
            }
            for key, width in widths.items()
        }
    return config, stats


@pytest.fixture
def saved(tmp_path):
    config, stats = native_layout()
    root = tmp_path / "checkpoint"
    root.mkdir()

    def write():
        (root / "processor_config.json").write_text(json.dumps(config))
        (root / "statistics.json").write_text(json.dumps(stats))

    write()
    return root, config, stats, write


def resolve(saved, **kwargs):
    root, _, _, write = saved
    write()
    return resolve_native_processor_context(root, read_roots=[root.parent], **kwargs)


def test_explicit_artifact_context_is_scoped_and_has_no_numeric_arrays(saved):
    report = resolve(saved)
    assert report["status"] == "PASS"
    assert report["axes"]["processor_numeric_context"]["status"] == "PASS"
    assert len(report["descriptor"]["fields"]) == 14
    assert report["descriptor"]["widths"] == {"state": 43, "action": 35}
    assert report["descriptor"]["action_offset_count"] == 50
    assert all(
        report["axes"][key]["status"] == "UNKNOWN"
        for key in (
            "installed_native_source",
            "historical_execution",
            "full_training_sample",
            "model_compatibility",
            "selected_sample_comparison",
            "source_association",
        )
    )
    assert "parameters" not in json.dumps(report)
    assert report["coverage"]["native_code_executed"] is False
    assert len(json.dumps(report).encode()) < 32768


def test_native_defaults_are_artifact_defaults_not_finetune_defaults(saved):
    root, config, _, _ = saved
    for name in DEFAULTS:
        config["processor_kwargs"].pop(name, None)
    report = resolve(saved)
    assert report["status"] == "ERROR"  # Native29 capacities are too small for G1.
    descriptor = report["descriptor"]
    assert descriptor["flags"] == DEFAULTS
    assert len(descriptor["defaulted_pointers"]) == len(DEFAULTS)
    assert descriptor["flags"]["use_percentiles"] is False
    assert descriptor["flags"]["max_action_horizon"] == 50
    assert root.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("use_percentiles", 1),
        ("use_relative_action", "false"),
        ("max_action_horizon", True),
        ("max_state_dim", 64.0),
        ("state_dropout_prob", True),
        ("state_dropout_prob", -0.1),
        ("state_dropout_prob", 1.1),
    ],
)
def test_strict_numeric_flag_types(saved, flag, value):
    saved[1]["processor_kwargs"][flag] = value
    assert resolve(saved)["status"] == "UNKNOWN"


def test_percentile_parameters_are_selected_and_mapped_to_native_min_max(saved):
    saved[1]["processor_kwargs"]["use_percentiles"] = True
    report = resolve(saved)
    internal = load_consumer_numeric_parameters(report, read_roots=[saved[0]])
    field = internal["fields"][0]
    assert field["parameter_names"] == ["q01", "q99"]
    assert field["parameters"] == {"min": [-1.0] * 6, "max": [1.0] * 6}


def test_inactive_statistics_and_inert_global_meanstd_are_identity_only(saved):
    before = resolve(saved)
    saved[2][PROFILE]["state"]["left_arm"]["q01"][0] = -42.0
    saved[1]["processor_kwargs"]["use_mean_std"] = True
    after = resolve(saved)
    assert after["status"] == "PASS"
    assert before["artifacts"] != after["artifacts"]
    assert before["descriptor"]["interpretation_sha256"] == after["descriptor"]["interpretation_sha256"]
    saved[2][PROFILE]["state"]["left_arm"]["min"][0] = -3.0
    changed = resolve(saved)
    assert changed["descriptor"]["interpretation_sha256"] != after["descriptor"]["interpretation_sha256"]


def test_true_percentile_inactive_minimum_does_not_change_active_digest(saved):
    saved[1]["processor_kwargs"]["use_percentiles"] = True
    before = resolve(saved)
    saved[2][PROFILE]["state"]["left_arm"]["min"][0] = -5.0
    after = resolve(saved)
    assert before["descriptor"]["interpretation_sha256"] == after["descriptor"]["interpretation_sha256"]


def test_per_key_sincos_precedes_meanstd_and_state_meanstd_does_not_clip(saved):
    kwargs = saved[1]["processor_kwargs"]
    kwargs["apply_sincos_state_encoding"] = True
    declaration = kwargs["modality_configs"][PROFILE]
    declaration["state"]["sin_cos_embedding_keys"] = ["left_arm"]
    declaration["state"]["mean_std_embedding_keys"] = ["left_arm", "right_arm"]
    declaration["action"]["mean_std_embedding_keys"] = ["left_arm"]
    report = resolve(saved)
    fields = {(f["modality"], f["key"]): f for f in report["descriptor"]["fields"]}
    assert fields["state", "left_arm"]["operation"] == "sincos"
    assert fields["state", "left_arm"]["output_dim"] == 14
    assert fields["state", "left_arm"]["clip"] is False
    assert fields["state", "right_arm"]["operation"] == "meanstd"
    assert fields["state", "right_arm"]["clip"] is False
    assert fields["action", "left_arm"]["clip"] is True
    assert report["descriptor"]["widths"]["state"] == 50


def test_relative_override_is_per_step_and_uses_relative_min_not_q01(saved):
    kwargs = saved[1]["processor_kwargs"]
    kwargs.update(use_relative_action=True, use_percentiles=True)
    relative = saved[2][PROFILE]["relative_action"] = {}
    for key in ("left_arm", "right_arm"):
        relative[key] = {
            name: [copy.deepcopy(values) for _ in range(50)]
            for name, values in saved[2][PROFILE]["action"][key].items()
        }
    report = resolve(saved)
    assert report["status"] == "PASS"
    field = load_consumer_numeric_parameters(report)["fields"][7]
    assert field["relative"] is True and field["state_reference"] == "left_arm"
    assert field["parameter_shapes"]["min"] == [50, 7]
    assert field["parameters"]["min"][0] == [-2.0] * 7
    assert report["descriptor"]["fields"][-1]["relative"] is False


def test_missing_relative_stats_error_is_visible_despite_numeric_unknown(saved):
    saved[1]["processor_kwargs"].update(use_relative_action=True, max_action_horizon=40)
    report = resolve(saved)
    assert report["status"] == "ERROR"
    assert report["axes"]["processor_numeric_context"]["status"] == "UNKNOWN"
    assert report["axes"]["declared_processor_capacity"]["status"] == "ERROR"
    assert any(f["code"] == "consumer.relative_statistics_missing" for f in report["findings"])
    assert any(f["code"] == "consumer.action_horizon_capacity" for f in report["findings"])


def test_horizon_is_count_not_span_even_when_profile_is_unsupported(saved):
    kwargs = saved[1]["processor_kwargs"]
    kwargs["max_action_horizon"] = 49
    kwargs["modality_configs"][PROFILE]["action"]["delta_indices"] = list(range(0, 100, 2))
    report = resolve(saved)
    assert report["status"] == "ERROR"
    assert any(f["code"] == "consumer.action_horizon_capacity" for f in report["findings"])


@pytest.mark.parametrize(
    "mutation", ["enum", "missing", "width", "negative", "reversed", "duplicate", "bool"]
)
def test_malformed_or_unsupported_context_never_passes(saved, mutation):
    if mutation == "enum":
        saved[1]["processor_kwargs"]["modality_configs"][PROFILE]["action"]["action_configs"][0]["rep"] = (
            "relative"
        )
    elif mutation == "missing":
        del saved[2][PROFILE]["state"]["left_leg"]["mean"]
    elif mutation == "width":
        saved[2][PROFILE]["state"]["left_leg"]["max"] = [1.0]
    elif mutation == "negative":
        saved[2][PROFILE]["state"]["left_leg"]["std"][0] = -1
    elif mutation == "reversed":
        saved[2][PROFILE]["state"]["left_leg"]["min"][0] = 8
    elif mutation == "duplicate":
        saved[1]["processor_kwargs"]["modality_configs"][PROFILE]["state"]["modality_keys"][0] = "right_leg"
    else:
        saved[2][PROFILE]["state"]["left_leg"]["std"][0] = True
    report = resolve(saved)
    assert report["status"] == "UNKNOWN"
    assert report["axes"]["processor_numeric_context"]["status"] == "UNKNOWN"


def test_unselected_entry_changes_file_identity_not_selected_interpretation(saved):
    config, stats = saved[1:3]
    config["processor_kwargs"]["modality_configs"]["other_native_tag"] = copy.deepcopy(
        config["processor_kwargs"]["modality_configs"][PROFILE]
    )
    stats["other_native_tag"] = copy.deepcopy(stats[PROFILE])
    assert resolve(saved)["status"] == "UNKNOWN"  # No silent selection.
    before = resolve(saved, embodiment=PROFILE)
    stats["other_native_tag"]["state"]["left_leg"]["min"][0] = -4
    after = resolve(saved, embodiment=PROFILE)
    assert before["status"] == after["status"] == "PASS"
    assert before["descriptor"]["interpretation_sha256"] == after["descriptor"]["interpretation_sha256"]
    assert before["artifacts"] != after["artifacts"]
    assert after["axes"]["full_native_constructor"]["status"] == "UNKNOWN"


def test_context_reread_rejects_identity_change_even_inactive_change(saved):
    report = resolve(saved)
    assert len(load_consumer_numeric_parameters(report)["fields"]) == 14
    saved[2][PROFILE]["state"]["left_arm"]["q01"][0] = -42
    saved[3]()
    with pytest.raises(InputError, match="changed"):
        load_consumer_numeric_parameters(report)


@pytest.mark.parametrize(
    "state_excluded,dropout,expected", [(True, 0, "PASS"), (False, 0, "ERROR"), (False, 0.5, "UNKNOWN")]
)
def test_state_capacity_respects_exclusion_and_unrecorded_dropout_mode(
    saved, state_excluded, dropout, expected
):
    kwargs = saved[1]["processor_kwargs"]
    kwargs.update(
        apply_sincos_state_encoding=True,
        max_state_dim=43,
        exclude_state=state_excluded,
        state_dropout_prob=dropout,
    )
    kwargs["modality_configs"][PROFILE]["state"]["sin_cos_embedding_keys"] = ["left_arm"]
    report = resolve(saved)
    assert report["status"] == expected


@pytest.mark.parametrize(
    "filename,body",
    [
        ("processor_config.json", b'{"x":1,"x":2}'),
        ("statistics.json", b'{"bad":NaN}'),
        ("statistics.json", b'{"bad":1e999}'),
        ("statistics.json", b"\xff"),
        ("embodiment_id.json", b'{"tag":true}'),
    ],
)
def test_bad_fixed_json_is_controlled_unknown(saved, filename, body):
    root = saved[0]
    (root / filename).write_bytes(body)
    report = resolve_native_processor_context(root)
    assert report["status"] == "UNKNOWN" and report["descriptor"] is None


@pytest.mark.parametrize("field", ["model_name", "transformers_loading_kwargs"])
def test_unread_remote_or_secret_kwargs_do_not_escape_report(saved, field):
    saved[1]["processor_kwargs"][field] = "SECRET_DO_NOT_ECHO:https://invalid/weights"
    report = resolve(saved)
    assert "SECRET_DO_NOT_ECHO" not in json.dumps(report)
    assert report["status"] == "PASS"


def test_unknown_keyword_refusal_does_not_copy_its_value(saved):
    saved[1]["processor_kwargs"]["custom_secret"] = "SECRET_DO_NOT_ECHO"
    report = resolve(saved)
    assert report["status"] == "UNKNOWN"
    assert "SECRET_DO_NOT_ECHO" not in json.dumps(report)


@pytest.mark.parametrize("case", ["outside", "parent", "tilde"])
def test_bad_authority_refuses_before_any_open(saved, monkeypatch, case):
    root = saved[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("No filesystem open is authorized")

    monkeypatch.setattr(os, "open", forbidden)
    path, roots = root, [root / "unrelated"]
    if case == "parent":
        path, roots = root / ".." / "checkpoint", [root.parent]
    elif case == "tilde":
        path, roots = "~/checkpoint", None
    with pytest.raises(InputError):
        resolve_native_processor_context(path, read_roots=roots)


@pytest.mark.parametrize("case", ["leaf_link", "ancestor_link", "fifo", "oversize"])
def test_nonregular_or_aliased_fixed_files_are_not_read(saved, monkeypatch, case):
    root = saved[0]
    if case == "ancestor_link":
        alias = root.parent / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    else:
        target = root / "statistics.json"
        target.unlink()
        if case == "leaf_link":
            outside = root.parent / "outside.json"
            outside.write_text("{}")
            target.symlink_to(outside)
        elif case == "fifo":
            os.mkfifo(target)
        else:
            with target.open("wb") as stream:
                stream.truncate(4 * 1024 * 1024 + 1)
    original = os.read

    def read(fd, size):
        if case != "ancestor_link" and os.fstat(fd).st_ino == os.stat(root / "statistics.json").st_ino:
            raise AssertionError("Rejected file must not be body-read")
        return original(fd, size)

    monkeypatch.setattr(os, "read", read)
    assert resolve_native_processor_context(root)["status"] == "UNKNOWN"


def test_missing_optional_mapping_and_embedded_stats_follow_native_file_precedence(saved):
    saved[1]["processor_kwargs"]["statistics"] = {"fake": "SECRET_DO_NOT_ECHO"}
    report = resolve(saved)
    assert report["status"] == "PASS"
    assert report["coverage"]["file_count"] == 2
    assert "SECRET_DO_NOT_ECHO" not in json.dumps(report)


def test_parameter_reread_retains_fresh_error_when_caller_strips_status(saved):
    saved[1]["processor_kwargs"]["max_action_horizon"] = 40
    context = resolve(saved)
    assert context["status"] == "ERROR"
    context.update(status="PASS", findings=[], axes={})
    reread = load_consumer_numeric_parameters(context)
    assert reread["context_status"] == "ERROR"
    assert any(f["code"] == "consumer.action_horizon_capacity" for f in reread["context_findings"])
    assert reread["context_axes"]["declared_processor_capacity"]["status"] == "ERROR"


def test_parameter_reread_does_not_inherit_fabricated_caller_error(saved):
    context = resolve(saved)
    context.update(status="ERROR", findings=[{"status": "ERROR", "code": "fabricated"}])
    reread = load_consumer_numeric_parameters(context)
    assert reread["context_status"] == "PASS"
    assert reread["context_findings"] == []


def test_partial_identity_matched_context_preserves_error_and_available_parameters(saved):
    saved[1]["processor_kwargs"].update(use_relative_action=True, max_action_horizon=40)
    context = resolve(saved)
    assert not context["descriptor"]["numeric_context_complete"]
    context.update(status="PASS", findings=[], axes={})
    reread = load_consumer_numeric_parameters(context)
    assert reread["context_status"] == "ERROR"
    assert reread["context_axes"]["processor_numeric_context"]["status"] == "UNKNOWN"
    assert len(reread["fields"]) == 12
    assert reread["interpretation_sha256"] is None
    assert any(f["code"] == "consumer.relative_statistics_missing" for f in reread["context_findings"])


def test_unavailable_descriptor_still_returns_fresh_known_error(saved):
    kwargs = saved[1]["processor_kwargs"]
    kwargs["max_action_horizon"] = 40
    kwargs["modality_configs"][PROFILE]["action"]["delta_indices"] = list(range(51))
    context = resolve(saved)
    assert context["descriptor"] is None and context["status"] == "ERROR"
    context.update(status="PASS", findings=[])
    reread = load_consumer_numeric_parameters(context)
    assert reread["context_status"] == "ERROR"
    assert reread["descriptor"] is None and reread["fields"] == []


def test_reread_rejects_bool_integer_descriptor_impostor(saved):
    context = resolve(saved)
    context["descriptor"]["flags"]["use_relative_action"] = 0
    with pytest.raises(InputError, match="interpretation changed"):
        load_consumer_numeric_parameters(context)
