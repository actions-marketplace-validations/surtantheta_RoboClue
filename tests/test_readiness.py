"""Static robot readiness: checkpoint and training data against the robot interface read from its command.

Behaviours are those of LeRobot 146d847 cited in readiness.py. Fixtures come from the shared ``case`` checkpoint
(ACT, state/action [2], one 32x32 camera ``front``, training data at 30 fps).
"""
import json

import pytest

from policy_doctor.readiness import readiness


@pytest.fixture(autouse=True)
def rollout_ready(case):
    """lerobot-rollout overrides the device and rename steps (context.py:576-579), so a deployable export has them."""
    path = case["checkpoint"] / "policy_preprocessor.json"
    document = json.loads(path.read_text())
    document["steps"] += [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
                          {"registry_name": "device_processor", "config": {"device": "cpu"}}]
    path.write_text(json.dumps(document))


def robot(**changes):
    interface = {
        "program": "rollout", "robot_type": "test_arm", "robot_id": None, "source": {"kind": "command", "sha256": "0" * 64},
        "fps": {"value": 30, "declared": True},
        "cameras": {"front": {"type": "opencv", "width": 32, "height": 32, "fps": 30}},
        "features": {"observation.state": {"names": ["shoulder", "gripper"], "shape": [2]},
                     "observation.images.front": {"shape": [32, 32, 3]}},
        "action_names": ["shoulder", "gripper"], "rename_map": {}, "policy_path": None, "policy_overrides": {},
        "flags": {}, "unknown": []}
    interface.update(changes)
    return interface


def cameras(**shapes):
    return {f"observation.images.{name}": {"shape": shape} for name, shape in shapes.items()}


def state(names):
    return {"observation.state": {"names": names, "shape": [len(names)]}}


def edit(case, name, update):
    path = case["checkpoint"] / name
    path.write_text(json.dumps(update(json.loads(path.read_text()))))


def two_cameras(policy_type):
    def update(config):
        config["type"] = policy_type
        config["input_features"]["observation.images.wrist"] = {"type": "VISUAL", "shape": [3, 32, 32]}
        return config
    return update


def run(case, interface, dataset=True):
    return readiness(str(case["checkpoint"]), interface, str(case["dataset"]) if dataset else None)


def codes(report):
    return {f["code"]: f for f in report["findings"]}


def test_matching_robot_clears_every_decidable_axis(case):
    report = run(case, robot())
    assert report["status"] == "pass"
    assert {axis: value["status"] for axis, value in report["axes"].items()} == {
        "loading": "cleared", "cameras": "cleared", "image_size": "cleared", "state": "cleared", "action": "cleared", "joint_order": "cleared",
        "declared_rate": "cleared", "normalization": "cleared", "relative_actions": "cleared", "units": "unknown"}


def test_disjoint_cameras_make_lerobot_rollout_refuse(case):
    report = run(case, robot(features={**state(["shoulder", "gripper"]), **cameras(wrist=[32, 32, 3])}))
    hit = codes(report)["readiness.rollout_refuses"]
    assert hit["status"] == "error" and "rollout/context.py:539-559" in hit["message"]
    assert report["axes"]["cameras"]["status"] == "not_cleared"


def test_missing_camera_under_a_rename_map_crashes_act(case):
    interface = robot(features={**state(["shoulder", "gripper"]), **cameras(wrist=[32, 32, 3])},
                      rename_map={"observation.images.wrist": "observation.images.top"})
    hit = codes(run(case, interface))["readiness.cameras_missing"]
    assert "raises KeyError" in hit["message"] and "modeling_act.py" in hit["message"]


def test_smolvla_missing_first_camera_shifts_the_rest(case):
    edit(case, "config.json", two_cameras("smolvla"))
    interface = robot(features={**state(["shoulder", "gripper"]), **cameras(wrist=[32, 32, 3])})
    hit = codes(run(case, interface))["readiness.cameras_missing"]
    assert "pads missing ones only up to empty_cameras" in hit["message"]
    assert "move into earlier input slots" in hit["message"]


def test_pi0_missing_last_camera_is_padded_without_a_shift(case):
    edit(case, "config.json", two_cameras("pi0"))
    hit = codes(run(case, robot()))["readiness.cameras_missing"]
    assert "appends a blank (-1) image" in hit["message"] and "earlier input slots" not in hit["message"]


def test_image_size_is_resolved_per_policy_type(case):
    other = robot(features={**state(["shoulder", "gripper"]), **cameras(front=[48, 64, 3])})
    assert run(case, other)["axes"]["image_size"]["status"] == "unknown"  # ACT: not modeled
    edit(case, "config.json", lambda c: dict(c, type="smolvla"))
    hit = codes(run(case, other))["readiness.image_resized"]
    assert hit["status"] == "warning" and "aspect ratio changes" in hit["message"]
    same = robot(features={**state(["shoulder", "gripper"]), **cameras(front=[64, 64, 3])})
    assert codes(run(case, same))["readiness.image_resized"]["status"] == "pass"


def test_state_width_mismatch_and_silent_zero_padding(case):
    wide = robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])}, action_names=["a", "b", "c"])
    hit = codes(run(case, wide))["readiness.state_width_mismatch"]
    assert hit["status"] == "error" and "raises a shape error at the first inference" in hit["message"]
    # With state statistics the normalizer meets the width first and raises (normalize_processor.py:352-424).
    edit(case, "config.json", lambda c: dict(c, type="smolvla", max_state_dim=32))
    hit = codes(run(case, wide))["readiness.state_width_mismatch"]
    assert "raises a shape error at the first inference" in hit["message"]

    def identity_state(document):
        for step in document["steps"]:
            if step["registry_name"] == "normalizer_processor":
                step["config"]["norm_map"] = dict(step["config"]["norm_map"], STATE="IDENTITY")
        return document
    edit(case, "policy_preprocessor.json", identity_state)
    hit = codes(run(case, wide))["readiness.state_width_mismatch"]
    assert "zero-pads the state to max_state_dim without an error" in hit["message"]


def test_permuted_joints_are_an_error_unless_the_checkpoint_saves_its_order(case):
    swapped = robot(features={**state(["gripper", "shoulder"]), **cameras(front=[32, 32, 3])},
                    action_names=["gripper", "shoulder"])
    hit = codes(run(case, swapped))["readiness.order_permuted"]
    assert hit["status"] == "error" and "each joint receives another joint's value" in hit["message"]
    edit(case, "config.json", lambda c: dict(c, action_feature_names=["shoulder", "gripper"]))
    assert codes(run(case, swapped))["readiness.order_permuted"]["status"] == "pass"


def test_loop_rate_against_training_rate(case):
    slow = robot(fps={"value": 10.0, "declared": False, "source": "LeRobot 146d847 src/lerobot/rollout/configs.py:300 (default)"},
                 cameras={"front": {"type": "opencv", "width": 32, "height": 32, "fps": 30}})
    hit = codes(run(case, slow))["readiness.rate_mismatch"]
    assert "training data was recorded at 30 fps" in hit["message"] and "the default" in hit["message"]
    assert "A chunk of 4 actions then spans 0.40 s instead of 0.13 s" in hit["message"]
    lagging = robot(cameras={"front": {"type": "opencv", "width": 32, "height": 32, "fps": 15}})
    assert codes(run(case, lagging))["readiness.camera_slower_than_loop"]["status"] == "warning"


def test_without_training_data_order_and_timing_are_unknown(case):
    report = run(case, robot(), dataset=False)
    assert report["axes"]["joint_order"]["status"] == report["axes"]["declared_rate"]["status"] == "unknown"
    # Undecided axes keep the verdict at unknown: a pass must not hide what was not checked.
    assert report["status"] == "unknown" and "unknown (joint_order, declared_rate, units)" in report["summary"]


def test_rename_collisions_and_unused_renames(case):
    interface = robot(features={**state(["shoulder", "gripper"]), **cameras(top=[32, 32, 3], side=[32, 32, 3])},
                      rename_map={"observation.images.top": "observation.images.front",
                                  "observation.images.side": "observation.images.front",
                                  "observation.images.ghost": "observation.images.x"})
    found = codes(run(case, interface))
    assert found["readiness.rename_collision"]["status"] == "error"
    assert found["readiness.rename_unused"]["status"] == "warning"


def test_relative_actions_without_saved_names_make_the_gripper_relative(case):
    def drop_names(document):
        document["steps"][0]["config"].pop("action_names")
        return document
    edit(case, "policy_preprocessor.json", drop_names)
    hit = codes(run(case, robot()))["readiness.all_relative"]
    assert hit["status"] == "warning" and "the gripper included" in hit["message"]


def test_robot_side_unknowns_are_carried_into_the_axes(case):
    interface = robot(unknown=["robot type unitree_g1 is not modeled: its state and action names are unknown"],
                      features=cameras(front=[32, 32, 3]), action_names=None)
    report = run(case, interface)
    assert report["axes"]["state"]["status"] == "unknown"
    assert any("unitree_g1" in reason for reason in report["axes"]["state"]["reasons"])


def test_rollout_replaces_the_saved_training_rename_map(case):
    # Public Sounness/smolvla_lekiwi_pick_and_place_v2 saves front/top/wrist -> camera1/2/3 in its preprocessor.
    saved = {"observation.images.top": "observation.images.front"}

    def add_rename(document):
        document["steps"].insert(0, {"registry_name": "rename_observations_processor", "config": {"rename_map": saved}})
        return document
    edit(case, "policy_preprocessor.json", add_rename)
    interface = robot(features={**state(["shoulder", "gripper"]), **cameras(top=[32, 32, 3])})
    hit = codes(run(case, interface))["readiness.rename_replaced"]
    assert hit["status"] == "error" and "context.py:578" in hit["message"]
    assert hit["remedy"] == """Add --rename_map='{"observation.images.top": "observation.images.front"}' to the command."""
    assert codes(run(case, dict(interface, rename_map=saved)))["readiness.rename_as_trained"]["status"] == "pass"


def test_image_size_follows_the_training_data_not_a_stale_declaration(case):
    # Fine-tuning keeps the base checkpoint's input_features (factory.py:303-305), so config.json may say 256x256
    # while training saw the dataset's 32x32 frames.
    edit(case, "config.json", lambda c: {**c, "type": "smolvla",
                                         "input_features": {**c["input_features"],
                                                            "observation.images.front": {"type": "VISUAL", "shape": [3, 256, 256]}}})
    report = run(case, robot())
    assert codes(report)["readiness.image_size_match"]["message"].endswith("as in training data observation.images.front.")
    stale = codes(run(case, robot(), dataset=False))["readiness.image_resized"]
    assert "config.json (may be stale after fine-tuning)" in stale["message"]


def test_rename_matters_only_when_the_effective_keys_differ(case):
    saved = {"observation.images.top": "observation.images.front"}

    def add_rename(document):
        document["steps"].insert(0, {"registry_name": "rename_observations_processor", "config": {"rename_map": saved}})
        return document
    edit(case, "policy_preprocessor.json", add_rename)
    already = codes(run(case, robot()))  # the robot's camera is already named front
    assert "readiness.rename_replaced" not in already and already["readiness.rename_as_trained"]["status"] == "pass"
    superset = robot(features={**state(["shoulder", "gripper"]), **cameras(top=[32, 32, 3])},
                     rename_map={**saved, "observation.images.side": "observation.images.x"})
    assert "readiness.rename_replaced" not in codes(run(case, superset))


def test_renaming_the_state_away_is_an_error(case):
    hit = codes(run(case, robot(rename_map={"observation.state": "observation.environment_state"})))["readiness.state_renamed_away"]
    assert hit["status"] == "error" and "never receives observation.state" in hit["message"]


def test_cameras_the_policy_pads_itself_are_not_missing(case):
    edit(case, "config.json", lambda c: {**c, "type": "smolvla", "empty_cameras": 1, "input_features": {
        **c["input_features"], "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 32, 32]}}})
    found = codes(run(case, robot()))
    assert "readiness.cameras_missing" not in found and "padded by the policy" in found["readiness.cameras_match"]["message"]


def test_relative_mask_follows_the_saved_names_by_position(case):
    swapped = robot(features={**state(["gripper", "shoulder"]), **cameras(front=[32, 32, 3])},
                    action_names=["gripper", "shoulder"])
    hit = codes(run(case, swapped, dataset=False))["readiness.relative_mask_misplaced"]
    assert "position 1 is gripper in training and shoulder here" in hit["message"]


def test_rollout_cannot_load_without_processors_or_overridable_steps(case):
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (case["checkpoint"] / name).unlink()
    assert codes(run(case, robot()))["readiness.processors_missing"]["status"] == "error"


def test_rollout_overrides_need_device_and_rename_steps(case):
    def strip(document):
        document["steps"] = [s for s in document["steps"] if s["registry_name"] != "device_processor"]
        return document
    edit(case, "policy_preprocessor.json", strip)
    hit = codes(run(case, robot()))["readiness.override_without_step"]
    assert "device_processor" in hit["message"] and "pipeline.py:1393-1397" in hit["message"]


def test_record_commands_get_no_rollout_claims(case):
    report = run(case, robot(program="record"))
    assert report["axes"]["loading"]["status"] == "unknown" and "runs no policy" in report["axes"]["loading"]["reasons"][0]


def test_unresolved_rate_and_cameras_stay_unknown(case):
    report = run(case, robot(fps={"value": None, "declared": None, "source": "--fps not resolved"},
                             unresolved=["fps", "robot.cameras"]))
    assert report["axes"]["declared_rate"]["status"] == "unknown" and report["axes"]["cameras"]["status"] == "unknown"
    assert report["status"] == "unknown"


def test_checkpoint_without_cameras_can_pass(case):
    edit(case, "config.json", lambda c: {**c, "input_features": {"observation.state": c["input_features"]["observation.state"]}})
    report = run(case, robot())
    assert report["axes"]["cameras"]["status"] == "cleared" and report["status"] == "pass"


def test_odd_visual_shapes_do_not_crash(case):
    edit(case, "config.json", lambda c: {**c, "input_features": {**c["input_features"],
                                                                 "observation.images.front": {"type": "VISUAL", "shape": [3, 640]}}})
    report = run(case, robot(), dataset=False)
    assert report["axes"]["image_size"]["status"] == "unknown"


def test_withheld_state_is_uninspected_not_missing(case):
    # Codex C-2 keeps HubSnapshot.omitted_states for processor state listed at a pin but not fetched.
    (case["checkpoint"] / "pre.safetensors").unlink()
    wide = robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])}, action_names=["shoulder", "gripper"])
    missing = run(case, wide)
    assert missing["axes"]["normalization"]["status"] == "not_cleared"  # genuinely absent stays an error
    report = readiness(str(case["checkpoint"]), wide, str(case["dataset"]),
                       omitted_states={"policy_preprocessor.json": ["/steps/1"]})
    found = codes(report)
    assert found["normalization.statistics_uninspected"]["status"] == "unknown"
    assert not any(f["code"] == "normalization.statistics" and f["status"] == "error" and f["message"].startswith("pre:")
                   for f in report["findings"])
    assert report["axes"]["state"]["status"] == "unknown" and "were not fetched" in report["axes"]["state"]["reasons"][0]


@pytest.mark.parametrize("key,role", [("observation.state", "STATE"), ("action", "ACTION")])
@pytest.mark.parametrize("mode,expected", [("IDENTITY", None), ("MEAN_STD", 9), ("QUANTILES", 2)])
def test_width_uses_only_effective_mode_vectors(key, role, mode, expected):
    from policy_doctor.readiness import _trained_width
    step = {"config": {"features": {key: {"type": role, "shape": [2]}}, "norm_map": {role: mode}},
            "stats": {key: {"mean": [0.] * 9, "std": [1.] * 9, "q01": [-1.] * 2, "q99": [1.] * 2}}}
    assert _trained_width(step, key) == expected


@pytest.mark.parametrize("axis,filename,state_file,key", [
    ("state", "policy_preprocessor.json", "pre.safetensors", "observation.state"),
    ("action", "policy_postprocessor.json", "post.safetensors", "action"),
])
def test_withheld_active_width_unknown_even_when_robot_matches_declaration(case, axis, filename, state_file, key):
    (case["checkpoint"] / state_file).unlink()
    pointer = "/steps/1" if axis == "state" else "/steps/0"
    report = readiness(str(case["checkpoint"]), robot(), str(case["dataset"]), omitted_states={filename: [pointer]})
    assert report["axes"][axis]["status"] == "unknown"
    assert not any(f["code"] == f"readiness.{axis}_width_match" for f in report["findings"])
    # The other side is not withheld and retains its own statistics verdict.
    assert any(f["code"] == "normalization.statistics" and f["status"] == "pass" for f in report["findings"])
    assert key in " ".join(report["axes"][axis].get("reasons", []))


def _replace_saved_stats(case, state_width=2, action_width=2, **extra):
    import numpy as np
    from safetensors.numpy import save_file
    stats = {"observation.state.mean": np.zeros(state_width), "observation.state.std": np.ones(state_width),
             "action.mean": np.zeros(action_width), "action.std": np.ones(action_width), **extra}
    for name in ["pre.safetensors", "post.safetensors"]:
        save_file(stats, case["checkpoint"] / name)


@pytest.mark.parametrize("policy", ["act", "smolvla"])
def test_identity_ignores_carried_dataset_statistics(case, policy):
    _replace_saved_stats(case, state_width=9)
    edit(case, "config.json", lambda c: dict(c, type=policy, max_state_dim=32,
                                              normalization_mapping={**c["normalization_mapping"], "STATE": "IDENTITY"}))
    def identity(document):
        for step in document["steps"]:
            if step["registry_name"] == "normalizer_processor":
                step["config"]["norm_map"]["STATE"] = "IDENTITY"
        return document
    edit(case, "policy_preprocessor.json", identity)
    report = run(case, robot(), dataset=False)
    assert report["axes"]["state"]["status"] == "cleared"
    assert "readiness.state_width_mismatch" not in codes(report)
    assert "as trained" not in codes(report)["readiness.state_width_match"]["message"]


def test_active_quantiles_ignore_unconsumed_mean_width(case):
    import numpy as np
    _replace_saved_stats(case, state_width=9, **{"observation.state.q01": np.zeros(2), "observation.state.q99": np.ones(2)})
    edit(case, "config.json", lambda c: dict(c, normalization_mapping={**c["normalization_mapping"], "STATE": "QUANTILES"}))
    def quantile(document):
        for step in document["steps"]:
            if step["registry_name"] == "normalizer_processor":
                step["config"]["norm_map"]["STATE"] = "QUANTILES"
        return document
    edit(case, "policy_preprocessor.json", quantile)
    report = run(case, robot(), dataset=False)
    assert report["axes"]["state"]["status"] == "cleared"
    assert "readiness.state_width_mismatch" not in codes(report)


def test_provided_width_validates_active_statistics_not_stale_declaration(case):
    _replace_saved_stats(case, state_width=3)
    edit(case, "config.json", lambda c: dict(c, type="smolvla", max_state_dim=32))
    interface = robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])})
    report = run(case, interface, dataset=False)
    assert report["axes"]["state"]["status"] == "cleared"
    assert not any(f["code"] == "normalization.statistics" and f["status"] == "error" and f["message"].startswith("pre:observation.state:")
                   for f in report["findings"])
    assert codes(report)["normalization.saved_shape_declaration"]["status"] in ("warning", "unknown")
    assert "config.json declares 2" in codes(report)["readiness.state_width_match"]["message"]


@pytest.mark.parametrize("policy", ["act", "smolvla"])
def test_known_active_mismatch_survives_unrelated_omission(case, policy):
    _replace_saved_stats(case, state_width=3)
    edit(case, "config.json", lambda c: dict(c, type=policy, max_state_dim=32))
    report = readiness(str(case["checkpoint"]), robot(), omitted_states={"policy_postprocessor.json": ["/steps/0"]})
    assert report["axes"]["state"]["status"] == "not_cleared"
    assert "shape error" in codes(report)["readiness.state_width_mismatch"]["message"]


def test_absent_state_and_unusable_active_statistics_do_not_clear_width(case):
    (case["checkpoint"] / "pre.safetensors").unlink()
    report = run(case, robot(), dataset=False)
    assert report["axes"]["normalization"]["status"] == "not_cleared"
    assert report["axes"]["state"]["status"] == "unknown"
    assert any(f["code"] == "normalization.statistics" and f["status"] == "error" for f in report["findings"])


def test_scalar_statistics_broadcast_without_becoming_a_one_value_contract(case):
    _replace_saved_stats(case, state_width=1)
    report = run(case, robot(), dataset=False)
    assert report["axes"]["state"]["status"] in ("cleared", "unknown")
    assert "readiness.state_width_mismatch" not in codes(report)
    assert "one" in json.dumps(report["axes"]["state"]).lower() or any("broadcast" in f["message"] for f in report["findings"])


@pytest.mark.parametrize("policy", ["plugin_robot", "groot"])
def test_rebuilt_or_plugin_effective_width_stays_unknown(case, policy):
    edit(case, "config.json", lambda c: dict(c, type=policy))
    report = run(case, robot(), dataset=False)
    assert report["axes"]["state"]["status"] == report["axes"]["action"]["status"] == "unknown"
    assert report["axes"]["normalization"]["status"] == "unknown"


def test_unsupported_policy_overrides_invalidate_axes_without_hiding_errors(case):
    interface = robot(unsupported_policy_overrides=["normalization_mapping"])
    report = run(case, interface)
    assert report["axes"]["normalization"]["status"] == report["axes"]["state"]["status"] == "unknown"
    interface["rename_map"] = {"observation.state": "elsewhere"}
    report = run(case, interface)
    assert report["axes"]["state"]["status"] == "not_cleared"
    assert codes(report)["readiness.state_renamed_away"]["status"] == "error"


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_declared_rate_cannot_crash_or_clear_axis(case, rate):
    report = run(case, robot(fps={"value": rate, "declared": True}))
    assert report["axes"]["declared_rate"]["status"] == "unknown"
    report = run(case, robot(fps={"value": rate, "declared": True}, rename_map={"observation.state": "elsewhere"}))
    assert report["axes"]["state"]["status"] == "not_cleared"


def test_fake_policy_secrets_absent_from_readiness_and_axes_are_unknown(case):
    from policy_doctor.robot_interface import from_command
    interface = from_command("lerobot-rollout --robot.type=so100_follower --policy.api_key=FAKE_NEVER_OUTPUT "
                             "--policy.token=$FAKE_NEVER_OUTPUT")
    report = run(case, interface, dataset=False)
    assert "FAKE_NEVER_OUTPUT" not in json.dumps(report)
    # Independent camera/width errors may remain, but no affected default-based axis is cleared.
    assert all(value["status"] != "cleared" for value in report["axes"].values())


def test_active_input_normalization_compares_provided_state_width(case):
    interface = robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])})
    report = run(case, interface, dataset=False)
    assert report["axes"]["normalization"]["status"] == "not_cleared"
    assert codes(report)["readiness.normalization_provided_width"]["status"] == "error"


def test_robot_action_count_does_not_establish_unnormalizer_input(case):
    report = run(case, robot(action_names=["a", "b", "c"]), dataset=False)
    assert report["axes"]["action"]["status"] == "not_cleared"
    # The declared model and saved postprocessor both have2 coordinates; the robot asks for3.
    assert report["axes"]["normalization"]["status"] == "cleared"
    assert "readiness.normalization_provided_width" not in codes(report)


@pytest.mark.parametrize("policy", ["act", "smolvla"])
def test_matching_robot_and_post_stats_do_not_hide_model_output_mismatch(case, policy):
    _replace_saved_stats(case, action_width=3)
    edit(case, "config.json", lambda c: dict(c, type=policy, max_state_dim=32))
    report = run(case, robot(action_names=["a", "b", "c"]), dataset=False)
    assert report["axes"]["action"]["status"] == "not_cleared"
    assert codes(report)["readiness.action_model_output_width"]["status"] == "error"
    assert any(f["code"] == "normalization.statistics" and f["status"] == "error" and f["message"].startswith("post:action:")
               for f in report["findings"])
    assert not any(f["code"] == "normalization.saved_shape_declaration" and f["message"].startswith("post:action:")
                   for f in report["findings"])


def test_unmodeled_output_path_does_not_clear_from_robot_post_stat_equality(case):
    _replace_saved_stats(case, action_width=3)
    edit(case, "config.json", lambda c: dict(c, type="diffusion"))
    report = run(case, robot(action_names=["a", "b", "c"]), dataset=False)
    assert report["axes"]["action"]["status"] == "unknown"
    assert any("model-output width" in reason for reason in report["axes"]["action"].get("reasons", []))


@pytest.mark.parametrize("axis", ["state", "action"])
def test_malformed_active_vectors_are_not_reclassified_as_stale_metadata(case, axis):
    import numpy as np
    key = "observation.state" if axis == "state" else "action"
    _replace_saved_stats(case, **{key+".std": np.array([-1., 1.])})
    report = run(case, robot(), dataset=False)
    assert any(f["code"] == "normalization.statistics" and f["status"] == "error" and key in f["message"] for f in report["findings"])
    assert not any(f["code"] == "normalization.saved_shape_declaration" and key in f["message"] for f in report["findings"])


def test_bad_epsilon_survives_otherwise_matching_vectors(case):
    _replace_saved_stats(case, state_width=3)
    edit(case, "config.json", lambda c: dict(c, type="smolvla", max_state_dim=32))
    def bad(document):
        for step in document["steps"]:
            if step["registry_name"] == "normalizer_processor":
                step["config"]["eps"] = 0
        return document
    edit(case, "policy_preprocessor.json", bad)
    report = run(case, robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])}), dataset=False)
    assert codes(report)["normalization.epsilon"]["status"] == "error"
    assert any(f["code"] == "normalization.statistics" and f["status"] == "error" and f["message"].startswith("pre:observation.state") for f in report["findings"])


def test_act_projection_contradiction_survives_matching_numeric_vectors(case):
    _replace_saved_stats(case, state_width=3)
    report = run(case, robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])}), dataset=False)
    assert report["axes"]["state"]["status"] == "not_cleared"
    assert codes(report)["readiness.state_projection_width"]["status"] == "error"


def test_normalization_observation_selection_ignores_excluded_carried_stats(case):
    _replace_saved_stats(case, state_width=9)
    def exclude(document):
        for step in document["steps"]:
            if step["registry_name"] == "normalizer_processor":
                step["config"]["normalize_observation_keys"] = []
        return document
    edit(case, "policy_preprocessor.json", exclude)
    report = run(case, robot(), dataset=False)
    assert report["axes"]["state"]["status"] == "cleared"
    assert "readiness.state_width_mismatch" not in codes(report)
    assert not any(f["code"] == "normalization.statistics" and f["status"] == "error" and "observation.state" in f["message"]
                   for f in report["findings"])


def test_unused_post_observation_metadata_is_not_a_runtime_normalization_error(case):
    _replace_saved_stats(case, state_width=9)
    # Only change the *pre* active step to valid width9; the postprocessor carries unused state declarations too.
    interface = robot(features={**state([str(i) for i in range(9)]), **cameras(front=[32, 32, 3])})
    edit(case, "config.json", lambda c: dict(c, type="smolvla", max_state_dim=32))
    report = run(case, interface, dataset=False)
    inactive = codes(report)["normalization.inactive_statistics"]
    assert inactive["status"] == "unknown" and inactive["message"].startswith("post:observation.state:")
    assert not any(f["code"] == "normalization.statistics" and f["status"] == "error" and f["message"].startswith("post:observation.state:")
                   for f in report["findings"])


@pytest.mark.parametrize("withheld", [False, True])
def test_identity_state_file_loading_is_separate_from_unused_numeric_state(case, withheld):
    def identity(document):
        for step in document["steps"]:
            if step["registry_name"] == "normalizer_processor":
                step["config"]["norm_map"]["STATE"] = "IDENTITY"
        return document
    edit(case, "policy_preprocessor.json", identity)
    (case["checkpoint"] / "pre.safetensors").unlink()
    report = readiness(str(case["checkpoint"]), robot(), omitted_states={"policy_preprocessor.json": ["/steps/1"]} if withheld else None)
    assert report["axes"]["loading"]["status"] == ("unknown" if withheld else "not_cleared")
    found = codes(report)
    assert found["readiness.processor_state_uninspected" if withheld else "readiness.processor_state_missing"]["status"] == ("unknown" if withheld else "error")


def test_withheld_pre_state_does_not_hide_absent_post_loading_error(case):
    (case["checkpoint"] / "pre.safetensors").unlink()
    (case["checkpoint"] / "post.safetensors").unlink()
    report = readiness(str(case["checkpoint"]), robot(), omitted_states={"policy_preprocessor.json": ["/steps/1"]})
    assert report["axes"]["loading"]["status"] == "not_cleared"
    found = codes(report)
    assert found["readiness.processor_state_uninspected"]["status"] == "unknown"
    assert found["readiness.processor_state_missing"]["status"] == "error"


def test_unknown_step_missing_reference_is_metadata_unknown_not_native_failure(case):
    def custom(document):
        document["steps"].append({"class": "custom.untrusted.Step", "config": {}, "state_file": "absent.safetensors"})
        return document
    edit(case, "policy_preprocessor.json", custom)
    report = run(case, robot(), dataset=False)
    assert report["axes"]["loading"]["status"] == "unknown"
    assert "readiness.processor_state_missing" not in codes(report)


@pytest.mark.parametrize("key,role,side,axis", [("observation.state", "STATE", "pre", "state"), ("action", "ACTION", "post", "action")])
@pytest.mark.parametrize("mode,fields", [("MEAN_STD", ("mean", "std")), ("QUANTILES", ("q01", "q99"))])
def test_native_mixed_scalar_vector_stats_are_not_a_numerical_failure(case, key, role, side, axis, mode, fields):
    import numpy as np

    from policy_doctor.readiness import _trained_width
    low = np.array([0.])
    high = np.array([1., 2.])
    _replace_saved_stats(case, **{key+"."+fields[0]: low, key+"."+fields[1]: high})
    edit(case, "config.json", lambda c: dict(c, normalization_mapping={**c["normalization_mapping"], role: mode}))
    def select(document):
        for step in document["steps"]:
            if "norm_map" in step["config"]:
                step["config"]["norm_map"][role] = mode
        return document
    for name in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        edit(case, name, select)
    direct = {"config": {"features": {key: {"type": role, "shape": [2]}}, "norm_map": {role: mode}},
              "stats": {key: {fields[0]: low.tolist(), fields[1]: high.tolist()}}}
    assert _trained_width(direct, key) == 2
    report = run(case, robot(), dataset=False)
    assert report["axes"][axis]["status"] == "cleared"
    assert any(f["code"] == "normalization.broadcast_statistics" and f["status"] == "unknown" and f["message"].startswith(f"{side}:{key}:")
               for f in report["findings"])
    assert not any(f["code"] == "normalization.statistics" and f["status"] == "error" and f["message"].startswith(f"{side}:{key}:")
                   for f in report["findings"])


@pytest.mark.parametrize("policy", ["diffusion", "vqbet", "tdmpc"])
def test_matching_normalizer_width_does_not_clear_unmodeled_stale_model_input(case, policy):
    _replace_saved_stats(case, state_width=3)
    edit(case, "config.json", lambda c: dict(c, type=policy))
    report = run(case, robot(features={**state(["a", "b", "c"]), **cameras(front=[32, 32, 3])}), dataset=False)
    assert report["axes"]["state"]["status"] == "unknown"
    assert any("stale state declaration" in reason for reason in report["axes"]["state"].get("reasons", []))



@pytest.mark.parametrize("policy", ["evo1", "unregistered_plugin"])
def test_unmodeled_factory_valid_shape_disagreement_is_not_native_failure(case, policy):
    _replace_saved_stats(case, state_width=3)
    edit(case, "config.json", lambda c: dict(c, type=policy))
    report = run(case, robot(), dataset=False)
    assert report["axes"]["normalization"]["status"] == "unknown"
    assert report["axes"]["loading"]["status"] == "unknown"
    assert not any(f["status"] in ("error", "warning", "pass") and f["code"].startswith("normalization.")
                   for f in report["findings"])


@pytest.mark.parametrize("matching_input", [False, True])
def test_unmodeled_normalization_override_never_uses_default_width_or_coefficient(case, matching_input):
    _replace_saved_stats(case, state_width=3)
    edit(case, "config.json", lambda c: dict(c, type="smolvla", max_state_dim=32))
    interface = robot(unsupported_policy_overrides=["normalization_mapping"])
    if matching_input:
        interface["features"]["observation.state"] = state(["a", "b", "c"])["observation.state"]
    report = run(case, interface, dataset=False)
    assert report["axes"]["state"]["status"] == report["axes"]["normalization"]["status"] == "unknown"
    assert not any(f["status"] in ("error", "warning", "pass") and f["code"].startswith("normalization.")
                   for f in report["findings"])


@pytest.mark.parametrize("context", ["factory", "override"])
@pytest.mark.parametrize("invalid", ["negative_std", "epsilon"])
def test_unmodeled_effective_context_keeps_observed_invalid_metadata(case, context, invalid):
    import numpy as np
    if context == "factory":
        edit(case, "config.json", lambda c: dict(c, type="evo1"))
    interface = robot(unsupported_policy_overrides=["normalization_mapping"] if context == "override" else [])
    if invalid == "negative_std":
        _replace_saved_stats(case, **{"observation.state.std": np.array([-1., 1.])})
        code = "normalization.statistics"
    else:
        def bad(document):
            for step in document["steps"]:
                if step["registry_name"] == "normalizer_processor":
                    step["config"]["eps"] = -1
            return document
        edit(case, "policy_preprocessor.json", bad)
        code = "normalization.epsilon"
    report = run(case, interface, dataset=False)
    assert any(f["code"] == code and f["status"] == "error" for f in report["findings"])
    assert report["axes"]["normalization"]["status"] == "not_cleared"


@pytest.mark.parametrize("policy", ["evo1", "unregistered_plugin"])
def test_unmodeled_factory_missing_copy_does_not_assert_native_loader_route(case, policy):
    edit(case, "config.json", lambda c: dict(c, type=policy))
    (case["checkpoint"] / "pre.safetensors").unlink()
    report = run(case, robot(), dataset=False)
    assert report["axes"]["loading"]["status"] == "unknown"
    assert codes(report)["readiness.processor_copy_incomplete"]["status"] == "unknown"
    assert "readiness.processor_state_missing" not in codes(report)


def test_config_only_image_aspect_reference_is_not_historical_training_geometry(case):
    edit(case, "config.json", lambda c: dict(c, type="smolvla"))
    report = run(case, robot(features={**state(["a", "b"]), **cameras(front=[16, 32, 3])}), dataset=False)
    message = codes(report)["readiness.image_resized"]["message"]
    assert "Actual training geometry is unverified" in message
    assert "than in training" not in message
