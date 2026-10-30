"""Regressions from the 22 October real-world census of popular public LeRobot policies.

Each fixture reproduces the *declaration pattern* observed in a pinned public checkpoint
(see docs/real_world/); the expected verdicts follow the pinned LeRobot 146d847 source.
Values are generated; no public artifact is redistributed here.
"""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from policy_doctor import hub
from policy_doctor.artifacts import InputError, load_dataset, step_kind
from policy_doctor.diagnostics import audit

STATE = {"type": "STATE", "shape": [2]}
ACTION = {"type": "ACTION", "shape": [2]}
CAMERA = {"type": "VISUAL", "shape": [3, 8, 8]}


def _checkpoint(root, *, config=None, pre=(), post=(), states=None):
    root.mkdir(parents=True)
    base = {"type": "act", "input_features": {"observation.state": STATE, "observation.images.top": CAMERA},
            "output_features": {"action": ACTION},
            "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}}
    base.update(config or {})
    (root / "config.json").write_text(json.dumps(base))
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": list(pre)}))
    (root / "policy_postprocessor.json").write_text(json.dumps({"steps": list(post)}))
    for name, keys in (states or {}).items():
        save_file({f"{k}.{s}": np.array(v, dtype=np.float32) for k, stats in keys.items() for s, v in stats.items()},
                  str(root / name))
    return str(root)


def _norm(name, norm_map, features, state_file=None):
    step = {"registry_name": name, "config": {"eps": 1e-8, "norm_map": norm_map, "features": features}}
    if state_file:
        step["state_file"] = state_file
    return step


def _codes(report):
    return {(f["code"], f["status"]) for f in report["findings"]}


MEAN_STD = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
QUANTILES = {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]}


def test_verified_native_steps_are_neutral_not_custom():
    for name in ("tokenizer_processor", "smolvla_new_line_processor", "pi0_new_line_processor",
                 "pi05_prepare_state_tokenizer_processor_step", "xvla_imagenet_normalize",
                 "lerobot.processor.tokenizer_processor.TokenizerProcessorStep"):
        assert step_kind(name) == "neutral", name
    # A look-alike class in another module, or an unverified step, stays unsupported.
    assert step_kind("custom.module.TokenizerProcessorStep") == "unsupported"
    assert step_kind("image_crop_resize_processor") == "unsupported"


def test_neutral_steps_no_longer_force_unknown(tmp_path):
    root = _checkpoint(tmp_path / "cp", pre=[{"registry_name": "tokenizer_processor", "config": {}},
                                              {"registry_name": "pi0_new_line_processor", "config": {}}])
    codes = _codes(audit(root))
    assert ("processor.native_neutral", "pass") in codes
    assert not any(code.startswith("processor.unsupported_") for code, _ in codes)


def _one_sided(root, policy, mode="QUANTILES"):
    identity = {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"}
    stats = {"QUANTILES": QUANTILES, "MEAN_STD": MEAN_STD, "MIN_MAX": {"min": [-1.0, -1.0], "max": [1.0, 1.0]}}[mode]
    return _checkpoint(root, config={"type": policy, "normalization_mapping": identity},
                       pre=[_norm("normalizer_processor", identity, {"action": ACTION})],
                       post=[_norm("unnormalizer_processor", {"ACTION": mode}, {"action": ACTION}, "post.safetensors")],
                       states={"post.safetensors": {"action": stats}})


def test_one_sided_action_normalization_depends_on_the_native_design(tmp_path):
    # lerobot/lingbot_va_* documents this design; the same pattern on ACT is a fault (validator probe S2);
    # an unregistered policy type cannot be decided from metadata.
    for policy, expected in (("lingbot_va", "pass"), ("act", "error"), ("vendor_policy", "unknown")):
        report = audit(_one_sided(tmp_path / policy, policy))
        statuses = {f["status"] for f in report["findings"] if f["code"] in ("normalization.pair_mode", "normalization.config_mismatch")}
        assert statuses == {expected}, (policy, statuses)


def test_lingbot_style_one_sided_quantiles_is_unknown_not_error(tmp_path):
    # lerobot/lingbot_va_*: config IDENTITY, preprocessor IDENTITY, postprocessor QUANTILES with its own state.
    identity = {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"}
    root = _checkpoint(tmp_path / "cp", config={"type": "lingbot_va", "normalization_mapping": identity},
                       pre=[_norm("normalizer_processor", identity, {"action": ACTION})],
                       post=[_norm("unnormalizer_processor", {"ACTION": "QUANTILES"}, {"action": ACTION}, "post.safetensors")],
                       states={"post.safetensors": {"action": QUANTILES}})
    report = audit(root)
    codes = _codes(report)
    assert ("normalization.pair_mode", "pass") in codes and ("normalization.config_mismatch", "pass") in codes
    assert not any(status == "error" for _, status in codes)
    assert any("lingbot_va" in f["message"] for f in report["findings"] if f["code"] == "normalization.pair_mode")


def test_two_sided_mode_contradictions_remain_errors(tmp_path):
    # Preprocessor normalizes targets but outputs are never unnormalized: still a definite fault.
    root = _checkpoint(tmp_path / "a", pre=[_norm("normalizer_processor", {"ACTION": "MEAN_STD"}, {"action": ACTION}, "pre.safetensors")],
                       post=[_norm("unnormalizer_processor", {"ACTION": "IDENTITY"}, {"action": ACTION})],
                       states={"pre.safetensors": {"action": MEAN_STD}})
    assert ("normalization.pair_mode", "error") in _codes(audit(root))
    # Two different stats-backed modes.
    root = _checkpoint(tmp_path / "b", pre=[_norm("normalizer_processor", {"ACTION": "MEAN_STD"}, {"action": ACTION}, "pre.safetensors")],
                       post=[_norm("unnormalizer_processor", {"ACTION": "QUANTILES"}, {"action": ACTION}, "post.safetensors")],
                       states={"pre.safetensors": {"action": MEAN_STD}, "post.safetensors": {"action": QUANTILES}})
    assert ("normalization.pair_mode", "error") in _codes(audit(root))
    # Config declares normalization the saved processor skips.
    root = _checkpoint(tmp_path / "c", pre=[_norm("normalizer_processor", {"STATE": "IDENTITY"}, {"observation.state": STATE})])
    assert ("normalization.config_mismatch", "error") in _codes(audit(root))


def test_checkpoint_without_statistics_stays_an_error_with_the_native_consequence(tmp_path):
    # lerobot/xvla-base, lerobot/pi0_old: MEAN_STD declared, no statistics file referenced at all. Deploying it as
    # saved passes values through unnormalized, so it stays an error (validator probe S4), with the consequence.
    root = _checkpoint(tmp_path / "cp", pre=[_norm("normalizer_processor", {"ACTION": "MEAN_STD"}, {"action": ACTION})],
                       post=[_norm("unnormalizer_processor", {"ACTION": "MEAN_STD"}, {"action": ACTION})])
    stats = [f for f in audit(root)["findings"] if f["code"] == "normalization.statistics"]
    assert stats and all(f["status"] == "error" for f in stats)
    assert all("normalize_processor.py:333" in f["message"] and "cannot be deployed as saved" in f["message"] for f in stats)


def test_referenced_state_without_statistics_is_still_an_error(tmp_path):
    root = _checkpoint(tmp_path / "cp", pre=[_norm("normalizer_processor", {"ACTION": "MEAN_STD"}, {"action": ACTION}, "pre.safetensors")],
                       states={"pre.safetensors": {"observation.state": MEAN_STD}})
    assert ("normalization.statistics", "error") in _codes(audit(root))


def test_unbound_features_that_the_step_leaves_unchanged_are_not_warnings(tmp_path):
    # lerobot/smolvla_base (IDENTITY cameras) and jialei02/lawam (state inside an action-only unnormalizer).
    root = _checkpoint(tmp_path / "cp",
                       pre=[_norm("normalizer_processor", {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                                  {"observation.image2": CAMERA, "observation.state": STATE, "action": ACTION}, "pre.safetensors")],
                       post=[_norm("unnormalizer_processor", {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                                   {"observation.velocity": STATE, "action": ACTION}, "post.safetensors")],
                       states={"pre.safetensors": {"observation.state": MEAN_STD, "action": MEAN_STD},
                               "post.safetensors": {"observation.velocity": MEAN_STD, "action": MEAN_STD}})
    unbound = {f["message"].split()[2]: f["status"] for f in audit(root)["findings"] if f["code"] == "processor.feature_unbound"}
    assert unbound == {"observation.image2": "pass", "observation.velocity": "pass"}


def test_unbound_normalized_observation_still_warns(tmp_path):
    root = _checkpoint(tmp_path / "cp",
                       pre=[_norm("normalizer_processor", {"STATE": "MEAN_STD"}, {"observation.velocity": STATE}, "pre.safetensors")],
                       states={"pre.safetensors": {"observation.velocity": MEAN_STD}})
    assert ("processor.feature_unbound", "warning") in _codes(audit(root))


def test_groot_native_relative_actions_pass_with_source(tmp_path):
    # sreetz-nv GR00T N1.7 checkpoints: use_relative_actions in config, realised inside the decode step.
    root = _checkpoint(tmp_path / "cp", config={"type": "groot", "use_relative_actions": True},
                       post=[{"registry_name": "groot_n1_7_action_decode_v1", "config": {"use_relative_action": True}}])
    report = audit(root)
    codes = _codes(report)
    assert ("action.config_relative_native", "pass") in codes
    assert ("action.config_relative_mismatch", "error") not in codes


def test_unverified_native_relative_step_is_unknown_and_absent_handling_is_error(tmp_path):
    root = _checkpoint(tmp_path / "a", config={"use_relative_actions": True},
                       post=[{"registry_name": "vendor_decode", "config": {"use_relative_actions": True}}])
    assert ("action.config_relative_native", "unknown") in _codes(audit(root))
    root = _checkpoint(tmp_path / "b", config={"use_relative_actions": True})
    assert ("action.config_relative_mismatch", "error") in _codes(audit(root))


def _dataset(root, names):
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": {
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(names)], "names": names}}}))
    return str(root)


def test_duplicate_dataset_coordinate_names_are_reported_not_fatal(tmp_path):
    # IPEC-COMMUNITY LIBERO datasets used by nvidia/gr00t17 checkpoints: one label per coordinate, repeated.
    labels = ["x", "y", "gripper", "gripper"]
    ds = _dataset(tmp_path / "ds", labels)
    assert load_dataset(ds).features["action"]["coordinate_names_status"] == "duplicate"
    root = _checkpoint(tmp_path / "cp", config={"input_features": {"observation.state": {"type": "STATE", "shape": [4]}},
                                               "output_features": {"action": {"type": "ACTION", "shape": [4]}}})
    report = audit(root, ds)
    dup = [f for f in report["findings"] if f["code"] == "dataset.coordinate_names_duplicate"]
    assert {f["status"] for f in dup} == {"warning"} and "gripper" in dup[0]["message"]
    assert any(f["code"] == "dataset.feature_shape" and f["status"] == "pass" for f in report["findings"])


def test_duplicate_labels_inside_a_single_axis_mapping(tmp_path):
    # IPEC-COMMUNITY/libero_*: {"motors": [..., "gripper", "gripper"]} for an eight-coordinate state.
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    motors = ["x", "y", "gripper", "gripper"]
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": {
        "observation.state": {"dtype": "float32", "shape": [4], "names": {"motors": motors}}}}))
    state = load_dataset(str(root)).features["observation.state"]
    assert state["coordinate_names_status"] == "duplicate" and state["raw_names"] == {"motors": motors}
    cp = _checkpoint(tmp_path / "cp", config={"input_features": {"observation.state": {"type": "STATE", "shape": [4]}},
                                             "output_features": {}})
    dup = [f for f in audit(cp, str(root))["findings"] if f["code"] == "dataset.coordinate_names_duplicate"]
    assert dup and dup[0]["status"] == "warning" and "['gripper']" in dup[0]["message"]


def test_policy_padded_empty_cameras_are_not_missing_dataset_features(tmp_path):
    # lerobot/pi0*_libero_finetuned_v044: config pads observation.images.empty_camera_0 itself.
    cam = {"type": "VISUAL", "shape": [3, 8, 8]}
    root = _checkpoint(tmp_path / "cp", config={"type": "pi05", "empty_cameras": 1, "input_features": {
        "observation.state": STATE, "observation.images.top": cam, "observation.images.empty_camera_0": cam}})
    ds = tmp_path / "ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": {
        "observation.state": {"dtype": "float32", "shape": [2]}, "action": {"dtype": "float32", "shape": [2]},
        "observation.images.top": {"dtype": "video", "shape": [8, 8, 3], "names": ["height", "width", "channels"]}}}))
    missing = {f["message"].split()[0]: f["status"] for f in audit(root, str(ds))["findings"]
               if f["code"] == "dataset.feature_missing"}
    assert missing.get("observation.images.empty_camera_0") == "pass"


def test_identity_feature_shape_difference_is_a_noop(tmp_path):
    # lerobot/xvla-libero: normalizer lists image shapes that differ from config, in IDENTITY mode.
    big = {"type": "VISUAL", "shape": [3, 16, 16]}
    root = _checkpoint(tmp_path / "cp", pre=[_norm("normalizer_processor", {"VISUAL": "IDENTITY"},
                                                   {"observation.images.top": big})])
    shape = [f for f in audit(root)["findings"] if f["code"] == "processor.feature_shape"]
    assert shape and all(f["status"] == "pass" for f in shape)


def test_duplicate_checkpoint_names_stay_refused(tmp_path):
    root = _checkpoint(tmp_path / "cp", config={"output_features": {"action": {"type": "ACTION", "shape": [2],
                                                                              "names": ["a", "a"]}}})
    with pytest.raises(InputError, match="duplicates"):
        audit(root)


def test_unavailable_dataset_keeps_the_policy_check(monkeypatch, tmp_path):
    root = _checkpoint(tmp_path / "cp", pre=[{"registry_name": "tokenizer_processor", "config": {}}])

    def snapshot(source, *, repo_type="model", revision="main", fetch=False):
        raise hub.HubNotPublic("Only public ungated Hub repositories are accepted; no credentials are used")

    monkeypatch.setattr(hub, "resolve_hub_snapshot", snapshot)
    report = hub.check_checkpoint(root, "hf://org/gated-data", fetch=True)
    codes = _codes(report)
    assert ("hub.metadata_unavailable", "unknown") in codes and ("processor.native_neutral", "pass") in codes


def _dataset_with_cameras(root, cameras, state=2):
    (root / "meta").mkdir(parents=True)
    features = {"observation.state": {"dtype": "float32", "shape": [state]}, "action": {"dtype": "float32", "shape": [2]}}
    features.update({f"observation.images.{c}": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]}
                     for c in cameras})
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": features}))
    return str(root)


def test_camera_tolerant_policy_missing_camera_is_a_warning(tmp_path):
    # jayantrathi/smolvla_lang_grasp_v1: three declared cameras, one-camera training dataset.
    cam = {"type": "VISUAL", "shape": [3, 480, 640]}
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "input_features": {
        "observation.state": STATE, "observation.images.top": cam, "observation.images.side": cam}})
    ds = _dataset_with_cameras(tmp_path / "ds", ["top"])
    missing = [f for f in audit(cp, ds)["findings"] if f["code"] == "dataset.feature_missing"]
    assert [f["status"] for f in missing] == ["warning"] and "modeling_smolvla.py" in missing[0]["message"]


def test_missing_camera_is_still_an_error_for_other_policies_or_with_no_camera(tmp_path):
    cam = {"type": "VISUAL", "shape": [3, 480, 640]}
    cp = _checkpoint(tmp_path / "act", config={"type": "act", "input_features": {
        "observation.state": STATE, "observation.images.top": cam, "observation.images.side": cam}})
    ds = _dataset_with_cameras(tmp_path / "ds", ["top"])
    assert ("dataset.feature_missing", "error") in _codes(audit(cp, ds))
    cp = _checkpoint(tmp_path / "smol", config={"type": "smolvla", "input_features": {
        "observation.state": STATE, "observation.images.side": cam}})
    assert ("dataset.feature_missing", "error") in _codes(audit(cp, ds))


def test_native_resize_makes_camera_shape_differences_a_pass(tmp_path):
    cam = {"type": "VISUAL", "shape": [3, 256, 256]}
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "resize_imgs_with_padding": [512, 512],
                                             "input_features": {"observation.state": STATE, "observation.images.top": cam}})
    ds = _dataset_with_cameras(tmp_path / "ds", ["top"])
    shapes = {f["status"] for f in audit(cp, ds)["findings"]
              if f["code"] == "dataset.feature_shape" and "observation.images.top" in f["message"]}
    assert shapes == {"pass"}


def test_unmodeled_step_makes_relative_verdicts_unknown(tmp_path):
    # lerobot-data-collection/folding_final saves delta_actions_processor with exactly this config: the generic
    # relative step's own fields (relative_action_processor.py:119-120). lerobot/g05_base adds action_names.
    root = _checkpoint(tmp_path / "cp", config={"use_relative_actions": True},
                       pre=[{"registry_name": "delta_actions_processor", "config": {"enabled": True, "exclude_joints": ["gripper"]}}],
                       post=[{"registry_name": "absolute_actions_processor", "config": {"enabled": True}}])
    codes = _codes(audit(root))
    assert ("action.relative_pair", "unknown") in codes and ("action.config_relative_mismatch", "unknown") in codes
    assert not any(status == "error" and code.startswith("action.") for code, status in codes)


def _state_checkpoint(root, saved_dim, policy="smolvla", max_state_dim=32):
    states = None if saved_dim is None else {"pre.safetensors": {"observation.state": {"mean": [0.0] * saved_dim,
                                                                                       "std": [1.0] * saved_dim}}}
    pre = [_norm("normalizer_processor", {"STATE": "MEAN_STD"}, {"observation.state": {"type": "STATE", "shape": [6]}},
                 "pre.safetensors" if saved_dim else None)]
    config = {"type": policy, "input_features": {"observation.state": {"type": "STATE", "shape": [6]}}}
    if max_state_dim is not None:
        config["max_state_dim"] = max_state_dim
    return _checkpoint(root, config=config, pre=pre, states=states)


def _state_status(report):
    return {f["status"] for f in report["findings"] if f["code"] == "dataset.feature_shape" and "observation.state" in f["message"]}


def test_stale_input_declaration_is_decided_by_the_saved_statistics(tmp_path):
    # Sounness/deksprime/edgarcancinoe SmolVLA fine-tunes: config state [6] kept from smolvla_base,
    # saved statistics and the training dataset both [9]/[16]/[8] (factory.py:303-305).
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    stale = audit(_state_checkpoint(tmp_path / "stale", 9), ds)
    assert _state_status(stale) == {"warning"}
    assert any("factory.py:303-305" in f["message"] for f in stale["findings"] if f["code"] == "dataset.feature_shape")
    assert _state_status(audit(_state_checkpoint(tmp_path / "real", 6), ds)) == {"error"}   # statistics match the config
    assert _state_status(audit(_state_checkpoint(tmp_path / "none", None), ds)) == {"warning"}  # visible, undecided


def test_output_shape_mismatch_stays_an_error(tmp_path):
    cp = _checkpoint(tmp_path / "cp", config={"output_features": {"action": {"type": "ACTION", "shape": [7]}}})
    ds = _dataset_with_cameras(tmp_path / "ds", ["top"])
    assert any(f["code"] == "dataset.feature_shape" and f["status"] == "error" and "action" in f["message"]
               for f in audit(cp, ds)["findings"])


def _deployment(root, cameras):
    root.write_text(json.dumps({"schema_version": 1, "fps": 10, "features": {
        "observation.state": {"type": "STATE", "shape": [2]}, "action": {"type": "ACTION", "shape": [2]},
        **{f"observation.images.{c}": {"type": "VISUAL", "shape": [3, 480, 640]} for c in cameras}}}))
    return str(root)


def test_deployment_missing_a_trained_camera_is_still_an_error(tmp_path):
    # Validator probe S1: the camera-tolerant rule applies to training datasets only.
    cam = {"type": "VISUAL", "shape": [3, 480, 640]}
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "input_features": {
        "observation.state": STATE, "observation.images.front": cam, "observation.images.wrist": cam}})
    report = audit(cp, None, _deployment(tmp_path / "dep.json", ["front"]))
    assert ("deployment.feature_missing", "error") in _codes(report)


def test_unrelated_unmodeled_step_does_not_excuse_a_broken_relative_pair(tmp_path):
    # Validator probe S3: only a switchable (Boolean `enabled`) unmodeled step can stand in for the pair.
    root = _checkpoint(tmp_path / "cp", config={"use_relative_actions": True},
                       pre=[{"registry_name": "relative_actions_processor", "config": {"enabled": True}},
                            {"registry_name": "vendor_image_resize", "config": {"size": 224}}])
    assert ("action.relative_pair", "error") in _codes(audit(root))


def test_state_tokenizer_before_the_normalizer_is_not_neutral(tmp_path):
    # Validator probe S7: pi05 bins state assuming it is already normalized (processor_pi05.py:75).
    tok = {"registry_name": "pi05_prepare_state_tokenizer_processor_step", "config": {}}
    norm = _norm("normalizer_processor", {"STATE": "MEAN_STD"}, {"observation.state": STATE}, "pre.safetensors")
    states = {"pre.safetensors": {"observation.state": MEAN_STD}}
    before = _codes(audit(_checkpoint(tmp_path / "before", pre=[tok, norm], states=states)))
    after = _codes(audit(_checkpoint(tmp_path / "after", pre=[norm, tok], states=states)))
    assert ("processor.native_order", "unknown") in before
    assert ("processor.native_neutral", "pass") in after and not any(c == "processor.native_order" for c, _ in after)


def test_unnormalizer_in_the_preprocessor_does_not_get_the_postprocessor_noop(tmp_path):
    # Validator probe S6: only postprocessor transitions carry the action alone (factory.py:226-232).
    root = _checkpoint(tmp_path / "cp",
                       pre=[_norm("unnormalizer_processor", {"STATE": "MEAN_STD"}, {"observation.velocity": STATE}, "pre.safetensors")],
                       states={"pre.safetensors": {"observation.velocity": MEAN_STD}})
    assert ("processor.feature_unbound", "warning") in _codes(audit(root))


def test_native_resize_never_passes_a_channel_mismatch(tmp_path):
    # Validator probe S5: grayscale dataset images against a three-channel policy.
    cam = {"type": "VISUAL", "shape": [3, 32, 32]}
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "resize_imgs_with_padding": [512, 512],
                                             "input_features": {"observation.state": STATE, "observation.images.top": cam}})
    ds = tmp_path / "ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": {
        "observation.state": {"dtype": "float32", "shape": [2]}, "action": {"dtype": "float32", "shape": [2]},
        "observation.images.top": {"dtype": "video", "shape": [32, 32, 1], "names": ["height", "width", "channels"]}}}))
    statuses = {f["status"] for f in audit(cp, str(ds))["findings"]
                if f["code"] == "dataset.feature_shape" and "observation.images.top" in f["message"]}
    assert statuses and "pass" not in statuses


# ---- Validator round 2 (d5fdd9c): fixtures that got genuine faults past the new rules. ----

def test_stale_declaration_rule_applies_only_to_state_padding_policies(tmp_path):
    # Validator C1/C4: ACT, diffusion, VQ-BeT and TD-MPC size layers from the declared width
    # (modeling_act.py:314, modeling_diffusion.py:213), so a width mismatch is real whatever the statistics say.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=8)
    for policy in ("act", "diffusion", "vqbet", "tdmpc"):
        for saved in (8, None):
            report = audit(_state_checkpoint(tmp_path / f"{policy}_{saved}", saved, policy=policy, max_state_dim=None), ds)
            assert _state_status(report) == {"error"}, (policy, saved)


def test_stale_declaration_needs_room_to_pad(tmp_path):
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    assert _state_status(audit(_state_checkpoint(tmp_path / "narrow", 9, max_state_dim=8), ds)) == {"error"}
    assert _state_status(audit(_state_checkpoint(tmp_path / "unset", 9, max_state_dim=None), ds)) == {"error"}


def test_stale_declaration_does_not_contradict_the_statistics_check(tmp_path):
    # Sounness/smolvla_lekiwi_pick_and_place_v2@70bf3ca: saved statistics [9], config [6]. The normalizer broadcasts
    # the saved statistics and never reads the declared shape, so "mean must contain 6 values" is the same stale
    # declaration, not separate damage (validator real_stale.log).
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    report = audit(_state_checkpoint(tmp_path / "stale", 9), ds)
    stats = [f for f in report["findings"] if f["code"] == "normalization.statistics" and "observation.state" in f["message"]]
    assert stats and {f["status"] for f in stats} == {"warning"}
    assert "stale declaration" in stats[0]["message"]
    assert not any(f["status"] == "error" for f in report["findings"])
    stale = next(f for f in report["findings"] if f["code"] == "dataset.feature_shape" and "observation.state" in f["message"])
    assert "modeling_smolvla.py:370-371" in stale["message"] and "modeling_smolvla.py:448-451" in stale["message"]
    # Without a dataset the checkpoint alone is still inconsistent: the error stays.
    alone = audit(_state_checkpoint(tmp_path / "alone", 9))
    assert ("normalization.statistics", "error") in _codes(alone)


def test_undecided_stale_declaration_says_the_hub_never_decides(tmp_path):
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    report = audit(_state_checkpoint(tmp_path / "none", None), ds)
    stale = next(f for f in report["findings"] if f["code"] == "dataset.feature_shape" and "observation.state" in f["message"])
    assert stale["status"] == "warning" and "never decides" in stale["message"]


def test_lingbot_exemption_is_quantiles_only(tmp_path):
    # Validator A1/A1m: processor_lingbot_va.py:17-19, 59-63 specifies QUANTILES with q01/q99.
    for mode in ("MEAN_STD", "MIN_MAX"):
        report = audit(_one_sided(tmp_path / mode, "lingbot_va", mode))
        statuses = {f["status"] for f in report["findings"] if f["code"] in ("normalization.pair_mode", "normalization.config_mismatch")}
        assert statuses == {"error"}, (mode, statuses)
    report = audit(_one_sided(tmp_path / "q", "lingbot_va", "QUANTILES"))
    assert {f["status"] for f in report["findings"] if f["code"] == "normalization.pair_mode"} == {"pass"}


def test_state_tokenizer_order_applies_to_class_paths_and_identity_state(tmp_path):
    # Validator E1d (class path) and E2 (a normalizer that leaves STATE at IDENTITY).
    tok = {"class": "lerobot.policies.pi05.processor_pi05.Pi05PrepareStateTokenizerProcessorStep", "config": {}}
    norm = _norm("normalizer_processor", {"STATE": "MEAN_STD"}, {"observation.state": STATE}, "pre.safetensors")
    identity = _norm("normalizer_processor", {"STATE": "IDENTITY"}, {"observation.state": STATE}, "pre.safetensors")
    states = {"pre.safetensors": {"observation.state": MEAN_STD}}
    assert ("processor.native_order", "unknown") in _codes(audit(_checkpoint(tmp_path / "cls", pre=[tok, norm], states=states)))
    assert ("processor.native_order", "unknown") in _codes(audit(_checkpoint(tmp_path / "id", pre=[identity, tok], states=states)))
    ok = _codes(audit(_checkpoint(tmp_path / "ok", pre=[norm, tok], states=states)))
    assert ("processor.native_neutral", "pass") in ok and not any(c == "processor.native_order" for c, _ in ok)
    # A pi05 tokenizer that keeps state out of the prompt does not bin it (processor_pi05.py:54, 74).
    off = dict(tok, config={"include_state_in_prompt": False})
    off_codes = _codes(audit(_checkpoint(tmp_path / "off", pre=[off, norm], states=states)))
    assert not any(c == "processor.native_order" for c, _ in off_codes)


def test_switchable_rule_ignores_unrelated_or_disabled_steps(tmp_path):
    # Validator B1, B2, B3: a bare Boolean `enabled` is not evidence of native relative actions.
    rel = {"registry_name": "relative_actions_processor", "config": {"enabled": True}}
    for name, config in (("my_image_augment", {"enabled": False}), ("color_jitter", {"enabled": True}),
                         ("vendor_delta", {"enabled": False, "exclude_joints": []})):
        root = _checkpoint(tmp_path / name, config={"use_relative_actions": True},
                           pre=[rel, {"registry_name": name, "config": config}])
        assert ("action.relative_pair", "error") in _codes(audit(root)), name
    root = _checkpoint(tmp_path / "b3", config={"use_relative_actions": True},
                       pre=[{"registry_name": "logging_step", "config": {"enabled": False}}])
    assert ("action.config_relative_mismatch", "error") in _codes(audit(root))


def test_base_checkpoint_normalizer_without_features_is_described_as_saved(tmp_path):
    # lerobot/pi05_base@22c30e5 and pi05_libero_base@5fcf06f: normalizer features {} before the state tokenizer.
    tok = {"registry_name": "pi05_prepare_state_tokenizer_processor_step", "config": {}}
    empty = _norm("normalizer_processor", {"STATE": "QUANTILES"}, {})
    report = audit(_checkpoint(tmp_path / "base", pre=[empty, tok]))
    order = [f for f in report["findings"] if f["code"] == "processor.native_order"]
    assert order and order[0]["status"] == "unknown" and "declares no features" in order[0]["message"]


# ---- Validator round 3 (734003b) ----

def _state_deployment(root, width):
    root.write_text(json.dumps({"schema_version": 1, "fps": 10, "features": {
        "observation.state": {"type": "STATE", "shape": [width]}, "action": {"type": "ACTION", "shape": [2]}}}))
    return str(root)


def test_deployment_that_cannot_be_normalized_keeps_the_statistics_error(tmp_path):
    # Validator defect 1, reproduced on Sounness@70bf3ca with `policy-doctor template` output: config [6], saved
    # statistics [9], dataset [9], deployment [6]. Normalization broadcasts [6] against [9] and fails.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    cp = _state_checkpoint(tmp_path / "cp", 9)
    report = audit(cp, ds, _state_deployment(tmp_path / "dep6.json", 6))
    codes = _codes(report)
    assert ("deployment.statistics_width", "error") in codes and ("normalization.statistics", "error") in codes
    assert report["status"] == "error"
    # A deployment that supplies the trained width keeps the reconciliation (the runner refusal is reported separately).
    ok = audit(cp, ds, _state_deployment(tmp_path / "dep9.json", 9))
    assert ("normalization.statistics", "warning") in _codes(ok) and "deployment.statistics_width" not in {c for c, _ in _codes(ok)}


def test_deployment_supplying_the_trained_width_is_not_blamed_for_a_stale_config(tmp_path):
    # Round-1 beta (Arjun, Sounness@70bf3ca): a correct 9-wide LeKiwi contract got ERROR deployment.feature_shape with
    # an image remedy, and check-repair counted it as an introduced error. The default sync rollout engine pads the
    # state; only remote inference validates it against the stale config.json.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    cp = _state_checkpoint(tmp_path / "cp", 9)
    for dataset in (ds, None):
        report = audit(cp, dataset, _state_deployment(tmp_path / "dep9.json", 9))
        shape = [f for f in report["findings"] if f["code"] == "deployment.feature_shape" and "observation.state" in f["message"]]
        assert [f["status"] for f in shape] == ["warning"]
        assert "stale" in shape[0]["message"] and "policy_runner.py:186" in shape[0]["message"]
        assert "input_features" in shape[0]["remedy"] and "image" not in shape[0]["remedy"]
    from policy_doctor.repair import check_repair
    repair = check_repair(cp, _state_deployment(tmp_path / "dep6.json", 6), str(tmp_path / "dep9.json"), ds)
    assert repair["introduced_error_counts"] == {} and repair["resolved_error_counts"].get("deployment.statistics_width") == 1


def test_template_from_the_training_dataset_passes_its_own_interface_audit(tmp_path):
    # Round-1 beta (Arjun): `template` copied the stale [6], and its own audit then rejected the result.
    from policy_doctor.template import deployment_template
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    cp = _state_checkpoint(tmp_path / "cp", 9)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(deployment_template(cp)))
    assert ("deployment.statistics_width", "error") in _codes(audit(cp, ds, str(stale)))
    value = deployment_template(cp, ds)
    assert value["features"]["observation.state"]["shape"] == [9] and value["fps"] == 10
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps(value))
    report = audit(cp, ds, str(fixed))
    assert not {c for c, s in _codes(report) if s == "error"} & {"deployment.statistics_width", "deployment.feature_shape"}


def test_vector_shape_error_keeps_a_vector_remedy(tmp_path):
    # ACT sizes layers from the declared width, so the generic error stays, with advice about vectors, not images.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    report = audit(_state_checkpoint(tmp_path / "cp", 9, policy="act"), ds, _state_deployment(tmp_path / "dep9.json", 9))
    shape = next(f for f in report["findings"] if f["code"] == "deployment.feature_shape" and "observation.state" in f["message"])
    assert shape["status"] == "error" and "declared shape [6]" in shape["remedy"] and "camera" not in shape["remedy"]


def test_trained_width_comes_from_the_statistics_the_mode_uses(tmp_path):
    # Validator defect 6 (N2): QUANTILES uses q01/q99; mean/std at the dataset width do not make [6] quantiles stale.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=8)
    pre = [_norm("normalizer_processor", {"STATE": "QUANTILES"}, {"observation.state": {"type": "STATE", "shape": [6]}},
                 "pre.safetensors")]
    stats = {"mean": [0.0] * 8, "std": [1.0] * 8, "q01": [-1.0] * 6, "q99": [1.0] * 6}
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "max_state_dim": 32, "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]}}}, pre=pre, states={"pre.safetensors": {"observation.state": stats}})
    assert _state_status(audit(cp, ds)) == {"error"}


def test_include_state_in_prompt_is_honoured_for_pi05_only(tmp_path):
    # Validator defect 4: the pi0_fast step has no such field and always bins state (processor_pi0_fast.py:43-73).
    tok = {"registry_name": "pi0_fast_prepare_state_tokenizer_processor_step", "config": {"include_state_in_prompt": False}}
    norm = _norm("normalizer_processor", {"STATE": "MEAN_STD"}, {"observation.state": STATE}, "pre.safetensors")
    report = audit(_checkpoint(tmp_path / "cp", pre=[tok, norm], states={"pre.safetensors": {"observation.state": MEAN_STD}}))
    assert ("processor.native_order", "unknown") in _codes(report)


def test_flagged_postprocessor_step_does_not_excuse_a_missing_absolute_step(tmp_path):
    # Validator defect 7 (B6): no pinned source shows an arbitrary flagged step reversing relative actions.
    root = _checkpoint(tmp_path / "cp", config={"use_relative_actions": True},
                       pre=[{"registry_name": "relative_actions_processor", "config": {"enabled": True}}],
                       post=[{"registry_name": "vendor_decode", "config": {"use_relative_actions": True}}])
    assert ("action.relative_pair", "error") in _codes(audit(root))


def test_stale_warning_does_not_claim_normalization_the_saved_normalizer_skips(tmp_path):
    # Validator round 4 residual (D6f): with state absent from the normalizer's features, LeRobot leaves it raw
    # (normalize_processor.py:281-284); the warning must not say runtime normalization uses the statistics.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=9)
    pre = [_norm("normalizer_processor", {"STATE": "MEAN_STD"}, {}, "pre.safetensors")]
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "max_state_dim": 32, "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]}}}, pre=pre,
        states={"pre.safetensors": {"observation.state": {"mean": [0.0] * 9, "std": [1.0] * 9}}})
    stale = next(f for f in audit(cp, ds)["findings"] if f["code"] == "dataset.feature_shape" and "observation.state" in f["message"])
    assert stale["status"] == "warning" and "leaves this state unnormalized" in stale["message"]
    assert "runtime normalization uses the saved statistics" not in stale["message"]


def test_undecided_statistics_never_excuse_a_deployment_width(tmp_path):
    # Validator 1a: P1 (IDENTITY state, no width decided) and P2 (statistics not available) must keep the 0.38 error;
    # a smolvla runs a wrong 7-wide state silently when its normalizer does not decide the width.
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=6)
    for label, cp in (("identity", _checkpoint(tmp_path / "identity", config={"type": "smolvla", "max_state_dim": 32,
                          "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
                          "normalization_mapping": {"STATE": "IDENTITY", "ACTION": "MEAN_STD"}})),
                      ("no-stats", _state_checkpoint(tmp_path / "nostats", None))):
        for dataset in (None, ds):
            report = audit(cp, dataset, _state_deployment(tmp_path / f"dep7-{label}.json", 7))
            shape = [f["status"] for f in report["findings"]
                     if f["code"] == "deployment.feature_shape" and "observation.state" in f["message"]]
            assert shape == ["error"], (label, dataset, shape)


def test_output_shape_remedy_names_output_features(tmp_path):
    # Validator 1b: an action (output) mismatch must not tell users to edit input_features.
    cp = _checkpoint(tmp_path / "cp", config={"output_features": {"action": {"type": "ACTION", "shape": [7]}}})
    dep = tmp_path / "dep.json"
    dep.write_text(json.dumps({"schema_version": 1, "fps": 10, "features": {
        "observation.state": STATE, "action": {"type": "ACTION", "shape": [6]}}}))
    shape = next(f for f in audit(cp, None, str(dep))["findings"]
                 if f["code"] == "deployment.feature_shape" and f["message"].startswith("action"))
    assert "output_features" in shape["remedy"] and "input_features" not in shape["remedy"]


def test_unused_statistics_never_excuse_a_deployment_width(tmp_path):
    # Codex review: config [6], deployment [7] and a saved but unused mean [7] (IDENTITY mode, or a normalizer that
    # excludes the state) must keep the deployment error, with or without a dataset [6] that agrees with the config.
    stats = {"pre.safetensors": {"observation.state": {"mean": [0.0] * 7, "std": [1.0] * 7}}}
    state6 = {"observation.state": {"type": "STATE", "shape": [6]}}
    config = {"type": "smolvla", "max_state_dim": 32, "input_features": state6}
    identity = _checkpoint(tmp_path / "identity", config=config, states=stats,
                           pre=[_norm("normalizer_processor", {"STATE": "IDENTITY"}, state6, "pre.safetensors")])
    excluded = _checkpoint(tmp_path / "excluded", config=config, states=stats,
                           pre=[_norm("normalizer_processor", {"STATE": "MEAN_STD"}, {}, "pre.safetensors")])
    ds = _dataset_with_cameras(tmp_path / "ds", [], state=6)
    for cp in (identity, excluded):
        for dataset in (None, ds):
            report = audit(cp, dataset, _state_deployment(tmp_path / "dep7.json", 7))
            shape = [f["status"] for f in report["findings"]
                     if f["code"] == "deployment.feature_shape" and "observation.state" in f["message"]]
            assert shape == ["error"], (cp, dataset, shape)
    # The active-statistics case (B6, Sounness) keeps its warning.
    active = audit(_state_checkpoint(tmp_path / "active", 9), _dataset_with_cameras(tmp_path / "ds9", [], state=9),
                   _state_deployment(tmp_path / "dep9.json", 9))
    assert [f["status"] for f in active["findings"] if f["code"] == "deployment.feature_shape"
            and "observation.state" in f["message"]] == ["warning"]


def test_dataset_disagreeing_with_the_statistics_keeps_the_deployment_error(tmp_path):
    # Validator R5: config [6], training dataset [6], active statistics [7], deployment [7]. The dataset contradicts
    # the stale-declaration explanation, so the deployment width stays an error.
    cp = _checkpoint(tmp_path / "cp", config={"type": "smolvla", "max_state_dim": 32,
                                             "input_features": {"observation.state": {"type": "STATE", "shape": [6]}}},
                     pre=[_norm("normalizer_processor", {"STATE": "MEAN_STD"}, {"observation.state": {"type": "STATE", "shape": [6]}},
                                "pre.safetensors")],
                     states={"pre.safetensors": {"observation.state": {"mean": [0.0] * 7, "std": [1.0] * 7}}})
    report = audit(cp, _dataset_with_cameras(tmp_path / "ds", [], state=6), _state_deployment(tmp_path / "dep7.json", 7))
    assert ("deployment.feature_shape", "error") in _codes(report)
    assert ("normalization.statistics", "error") in _codes(report)


def _renamed_dataset(root, widths):
    (root / "meta").mkdir(parents=True)
    features = {k: {"dtype": "float32", "shape": [w]} for k, w in widths.items()}
    features["action"] = {"dtype": "float32", "shape": [2]}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": features}))
    return str(root)


def _renamed_checkpoint(root, saved, rename):
    state6 = {"observation.state": {"type": "STATE", "shape": [6]}}
    return _checkpoint(root, config={"type": "smolvla", "max_state_dim": 32, "input_features": state6},
                       pre=[{"registry_name": "rename_observations_processor", "config": {"rename_map": rename}},
                            _norm("normalizer_processor", {"STATE": "MEAN_STD"}, state6, "pre.safetensors")],
                       states={"pre.safetensors": {"observation.state": {"mean": [0.0] * saved, "std": [1.0] * saved}}})


def _deployment_state_status(report):
    return [f["status"] for f in report["findings"]
            if f["code"] == "deployment.feature_shape" and "observation.state" in f["message"]]


def test_r5_guard_uses_effective_keys_after_saved_renames(tmp_path):
    # Codex review (R5_RENAMED_CONTRADICTION, R5_RENAMED_MATCH, RAW_STATE_RENAMED_AWAY).
    proprio = {"observation.proprio": "observation.state"}
    contradiction = audit(_renamed_checkpoint(tmp_path / "c1", 7, proprio),
                          _renamed_dataset(tmp_path / "d1", {"observation.proprio": 6}),
                          _state_deployment(tmp_path / "dep7.json", 7))
    assert _deployment_state_status(contradiction) == ["error"]
    assert ("normalization.statistics", "error") in _codes(contradiction)  # not reclassified as a stale declaration
    match = audit(_renamed_checkpoint(tmp_path / "c2", 9, proprio),
                  _renamed_dataset(tmp_path / "d2", {"observation.proprio": 9}),
                  _state_deployment(tmp_path / "dep9.json", 9))
    assert _deployment_state_status(match) == ["warning"]
    # Saved renames apply to the deployment too, so it supplies raw observation.proprio, which becomes the state.
    away_map = {**proprio, "observation.state": "observation.other"}
    away = audit(_renamed_checkpoint(tmp_path / "c3", 7, away_map),
                 _renamed_dataset(tmp_path / "d3", {"observation.proprio": 6, "observation.state": 7}),
                 _proprio_deployment(tmp_path / "dep7b.json", 7))
    assert _deployment_state_status(away) == ["error"]  # the raw [7] renamed away must not excuse the deployment
    assert ("normalization.statistics", "error") in _codes(away)
    kept = audit(_renamed_checkpoint(tmp_path / "c4", 7, away_map),
                 _renamed_dataset(tmp_path / "d4", {"observation.proprio": 7, "observation.state": 6}),
                 _proprio_deployment(tmp_path / "dep7c.json", 7))
    assert _deployment_state_status(kept) == ["warning"]  # effective state [7] agrees; the raw [6] renamed away is not it


def _proprio_deployment(root, width):
    root.write_text(json.dumps({"schema_version": 1, "fps": 10, "features": {
        "observation.proprio": {"type": "STATE", "shape": [width]}, "action": {"type": "ACTION", "shape": [2]}}}))
    return str(root)
