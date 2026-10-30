"""Robot side of a deployment, read from the command a user runs.

The command texts are the documented examples in LeRobot 146d847 (src/lerobot/scripts/lerobot_record.py:24-66 and
src/lerobot/scripts/lerobot_rollout.py:43-49), with the docstring's escaped backslashes as a shell would see them.
"""
import json

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.robot_interface import from_command, from_config

SO_ARM = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

RECORD = r"""lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --dataset.repo_id=<my_username>/<my_dataset_name> \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=true
"""

BIMANUAL = r"""lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \
  --dataset.num_episodes=25 \
  --dataset.single_task="Grab and handover the red cube to the other arm"
"""

ROLLOUT = r"""lerobot-rollout \
    --strategy.type=base \
    --policy.path=lerobot/act_koch_real \
    --robot.type=koch_follower \
    --robot.port=/dev/ttyACM0 \
    --task="pick up cube" --duration=30
"""


def test_documented_record_command():
    robot = from_command(RECORD)
    assert robot["program"] == "record" and robot["robot_type"] == "so100_follower" and robot["robot_id"] == "black"
    assert robot["features"]["observation.state"]["names"] == SO_ARM and robot["action_names"] == SO_ARM
    assert robot["features"]["observation.images.laptop"] == {"shape": [480, 640, 3]}
    assert robot["cameras"]["laptop"]["fps"] == 30 and robot["flags"] == {"use_degrees": True}
    assert robot["fps"] == {"value": 30, "declared": False, "source": "LeRobot 146d847 src/lerobot/configs/dataset.py:33 (default)"}
    assert robot["unknown"] == []


def test_documented_bimanual_command_with_multiline_flow_mappings():
    robot = from_command(BIMANUAL)
    assert sorted(robot["cameras"]) == ["left_top", "left_wrist", "right_front", "right_wrist"]
    assert robot["features"]["observation.state"]["names"] == [f"left_{n}" for n in SO_ARM] + [f"right_{n}" for n in SO_ARM]
    assert robot["features"]["observation.images.right_front"]["shape"] == [480, 640, 3]
    assert robot["unknown"] == []


def test_documented_rollout_command():
    robot = from_command(ROLLOUT)
    assert robot["program"] == "rollout" and robot["robot_type"] == "koch_follower"
    assert robot["policy_path"] == "lerobot/act_koch_real" and robot["cameras"] == {}
    assert robot["fps"]["value"] == 30.0 and robot["fps"]["declared"] is False
    assert "rollout/configs.py:300" in robot["fps"]["source"]


def test_rename_map_fps_depth_rotation_and_missing_resolution():
    robot = from_command(
        "uv run lerobot-rollout --robot.type=so101_follower --fps=15 --robot.use_degrees=false "
        "--rename_map='{\"observation.images.top\": \"observation.images.front\"}' "
        "--robot.cameras='{front: {type: intelrealsense, serial_number_or_name: \"0123\", width: 640, height: 480, "
        "fps: 30, use_depth: true}, wrist: {type: opencv, index_or_path: 1, rotation: 90}}' "
        "--policy.path lerobot/smolvla_base --policy.n_action_steps=10")
    assert robot["fps"] == {"value": 15, "declared": True} and robot["flags"] == {"use_degrees": False}
    assert robot["rename_map"] == {"observation.images.top": "observation.images.front"}
    assert robot["features"]["observation.images.front_depth"] == {"shape": [480, 640, 1]}
    assert robot["features"]["observation.images.wrist"] == {"shape": None}
    assert any("wrist: width/height not set" in reason for reason in robot["unknown"])
    assert robot["policy_path"] == "lerobot/smolvla_base" and robot["policy_overrides"] == {"n_action_steps": 10}


def test_mobile_manipulator_reports_base_velocities_and_default_cameras():
    robot = from_command("lerobot-rollout --robot.type=lekiwi --robot.port=/dev/x")
    assert robot["action_names"][-3:] == ["x.vel", "y.vel", "theta.vel"] and len(robot["action_names"]) == 9
    # Without --robot.cameras, LeKiwi opens its configured defaults; the wrist camera is rotated to portrait.
    assert robot["features"]["observation.images.front"]["shape"] == [480, 640, 3]
    assert robot["features"]["observation.images.wrist"]["shape"] == [640, 480, 3]
    assert "config_lekiwi.py:24-42" in robot["cameras"]["wrist"]["default"]
    custom = from_command("lerobot-rollout --robot.type=lekiwi --robot.cameras='{top: {type: opencv, width: 64, height: 48}}'")
    assert list(custom["cameras"]) == ["top"]


def test_unmodeled_robot_type_is_unknown_not_guessed():
    robot = from_command("lerobot-rollout --robot.type=unitree_g1 --policy.path=x/y")
    assert "observation.state" not in robot["features"] and robot["action_names"] is None
    assert any("unitree_g1 is not modeled" in reason for reason in robot["unknown"])


def test_finds_the_lerobot_command_inside_a_script():
    text = ("# my run\nexport HF_USER=me\nsource .venv/bin/activate && "
            "python -m lerobot.scripts.lerobot_rollout --robot.type=so101_follower # trailing comment\n"
            "lerobot-record --robot.type=koch_follower\n")
    robot = from_command(text)
    assert robot["program"] == "rollout" and robot["robot_type"] == "so101_follower"
    assert any("1 further lerobot command" in reason for reason in robot["unknown"])


def test_malformed_camera_value_is_reported_not_fatal():
    robot = from_command("lerobot-rollout --robot.type=so101_follower --robot.cameras='{front: {type: opencv'")
    assert robot["cameras"] == {} and any("--robot.cameras could not be parsed" in r for r in robot["unknown"])


def test_config_file(tmp_path):
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({"robot": {"type": "so101_follower", "cameras": {
        "top": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}}},
        "policy": {"path": "org/ckpt"}, "fps": 10, "strategy": {"type": "base"}}))
    robot = from_config(path)
    assert robot["program"] == "rollout" and robot["fps"] == {"value": 10, "declared": True}
    assert robot["features"]["observation.images.top"]["shape"] == [720, 1280, 3]
    assert robot["source"]["kind"] == "config" and len(robot["source"]["sha256"]) == 64


@pytest.mark.parametrize("text", ["python train.py --robot.type=so101_follower", "x" * (256 * 1024 + 1)])
def test_text_without_a_lerobot_command_or_too_large_is_refused(text):
    with pytest.raises(InputError):
        from_command(text)


def test_shell_expansion_and_config_files_stay_unresolved():
    for text in ("lerobot-rollout --robot.type=so101_follower --fps=$FPS", 'lerobot-rollout --robot.type=so101_follower --fps "${FPS:-10}"'):
        robot = from_command(text)
        assert robot["fps"]["value"] is None and "fps" in robot["unresolved"]
        assert any("shell expansion" in reason for reason in robot["unknown"])
    robot = from_command("lerobot-rollout --config_path=deploy.yaml --robot.port=/dev/x")
    assert robot["fps"]["value"] is None and "robot.cameras" in robot["unresolved"] and robot["cameras"] == {}
    assert any("--config_path deploy.yaml are not read" in reason for reason in robot["unknown"])


def test_yaml_one_one_values_and_json_escapes():
    robot = from_command("lerobot-rollout --robot.type=so101_follower --robot.cameras='{d: {type: intelrealsense, "
                         "width: 640, height: 480, use_depth: no, use_rgb: on}, w: {type: opencv, width: 0640, height: 480}}' "
                         "--rename_map='{\"observation.images.w\": \"observation.images.fr\\u006fnt\"}'")
    assert "observation.images.d_depth" not in robot["features"] and "observation.images.d" in robot["features"]
    assert robot["features"]["observation.images.w"]["shape"] is None
    assert robot["rename_map"] == {"observation.images.w": "observation.images.front"}


def test_program_must_be_in_command_position():
    robot = from_command("which lerobot-rollout && lerobot-rollout --robot.type=koch_follower")
    assert robot["robot_type"] == "koch_follower"


def test_bimanual_use_degrees_is_per_arm():
    robot = from_command("lerobot-rollout --robot.type=bi_so_follower --robot.left_arm_config.use_degrees=false "
                         "--robot.right_arm_config.use_degrees=false")
    assert robot["flags"] == {"use_degrees": {"left": False, "right": False}}


def test_lekiwi_client_image_sizes_come_from_the_host():
    robot = from_command("lerobot-rollout --robot.type=lekiwi_client --robot.remote_ip=10.0.0.2")
    assert robot["features"]["observation.images.front"]["shape"] is None
    assert any("set by the host" in reason for reason in robot["unknown"])


def test_yaml_comments_inside_a_multiline_camera_block():
    robot = from_command("lerobot-rollout --robot.type=so101_follower --robot.cameras='{front: {type: opencv, index_or_path: 0,\n"
                         " width: 640,  # px\n height: 480, fps: 30}}'")
    assert robot["features"]["observation.images.front"]["shape"] == [480, 640, 3] and robot["unknown"] == []


@pytest.mark.parametrize("value", ["FAKE_SECRET_DO_NOT_RETAIN", "$FAKE_SECRET_DO_NOT_RETAIN", "{nested: FAKE_SECRET_DO_NOT_RETAIN}"])
def test_unsupported_policy_values_never_leave_command_parser(value):
    robot = from_command(f"lerobot-rollout --robot.type=so101_follower --policy.api_key='{value}' "
                         f"--policy.token='{value}' --policy.max_state_dim=64 --policy.n_action_steps=7")
    assert "FAKE_SECRET_DO_NOT_RETAIN" not in json.dumps(robot)
    assert robot["policy_overrides"] == {"n_action_steps": 7}
    assert set(robot["unsupported_policy_overrides"]) == {"api_key", "token", "max_state_dim"}
    assert any("not modeled" in reason for reason in robot["unknown"])


def test_config_policy_override_allowlist_never_serializes_unknown_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"robot": {"type": "so101_follower"}, "policy": {
        "api_key": "FAKE_SECRET_VALUE", "token": {"nested": "FAKE_SECRET_VALUE"}, "n_action_steps": 4}}))
    robot = from_config(path)
    assert "FAKE_SECRET_VALUE" not in json.dumps(robot)
    assert robot["policy_overrides"] == {"n_action_steps": 4}
    assert set(robot["unsupported_policy_overrides"]) == {"api_key", "token"}


@pytest.mark.parametrize("value", ["bad", "0", "-1", "true", "$FAKE_STEP_SECRET"])
def test_unresolved_modeled_policy_override_is_not_defaulted(value):
    robot = from_command(f"lerobot-rollout --robot.type=so101_follower --policy.n_action_steps='{value}'")
    assert "n_action_steps" not in robot["policy_overrides"]
    assert "policy.n_action_steps" in robot["unresolved"]
    assert "FAKE_STEP_SECRET" not in json.dumps(robot)


@pytest.mark.parametrize("value", ["0", "-1", ".nan", ".inf"])
def test_nonpositive_or_nonfinite_declared_rate_is_unknown(value):
    interface = from_command(f"lerobot-rollout --robot.type=so101_follower --fps={value}")
    assert interface["fps"]["value"] is None
    assert interface["unknown"]



def test_known_native_camera_depth_defaults_do_not_apply_to_plugins():
    defaults = from_command("lerobot-rollout --robot.type=lekiwi")
    assert all(camera["use_depth"] is False for camera in defaults["cameras"].values())
    assert not any("whether it adds a depth image" in note for note in defaults["unknown"])
    for kind in ("opencv", "intelrealsense"):
        robot = from_command("lerobot-rollout --robot.type=so101_follower --robot.cameras='"
                             + json.dumps({"front": {"type": kind, "width": 64, "height": 48}}) + "'")
        assert robot["cameras"]["front"]["use_depth"] is False
    plugin = from_command("lerobot-rollout --robot.type=so101_follower "
                          "--robot.cameras='{front: {type: custom, width: 64, height: 48}}'")
    assert plugin["cameras"]["front"]["use_depth"] is None
    assert any("whether it adds a depth image" in note for note in plugin["unknown"])
