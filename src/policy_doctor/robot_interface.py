"""The robot side of a deployment, read from the command or config file a user will actually run.

LeRobot builds the policy-facing robot interface statically from the robot type and its configuration: motor
readings become ``observation.state``/``action`` names in robot order, filtered to ``.pos``/``.vel`` channels, and
each configured camera becomes ``observation.images.<name>`` with shape ``(height, width, 3)``
(LeRobot 146d847 src/lerobot/utils/feature_utils.py:48-95, src/lerobot/rollout/context.py:513-547). This module
reproduces that from text alone: it never imports LeRobot, opens a port or touches hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

from .artifacts import LEROBOT_REVISION, InputError, read_json

LEROBOT = f"LeRobot {LEROBOT_REVISION[:7]}"
MAX_TEXT_BYTES = 256 * 1024
MAX_DEPTH = 16
PROGRAMS = {"lerobot-rollout": "rollout", "lerobot-record": "record",
            "lerobot.scripts.lerobot_rollout": "rollout", "lerobot.scripts.lerobot_record": "record"}
# Loop rate when the command does not set one.
DEFAULT_FPS = {"rollout": (30.0, "src/lerobot/rollout/configs.py:300"),
               "record": (30, "src/lerobot/configs/dataset.py:33")}

_SO_ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_LEKIWI = tuple(f"arm_{m}.pos" for m in _SO_ARM) + ("x.vel", "y.vel", "theta.vel")
# State and action channels per robot type, in the order the robot class declares them. Each robot here reports the
# same channels as observation state and as action.
ROBOTS = {
    "so100_follower": ([f"{m}.pos" for m in _SO_ARM], "src/lerobot/robots/so_follower/so_follower.py:72-104"),
    "so101_follower": ([f"{m}.pos" for m in _SO_ARM], "src/lerobot/robots/so_follower/so_follower.py:72-104"),
    "koch_follower": ([f"{m}.pos" for m in _SO_ARM], "src/lerobot/robots/koch_follower/koch_follower.py:51-84"),
    "omx_follower": ([f"{m}.pos" for m in _SO_ARM], "src/lerobot/robots/omx_follower/omx_follower.py:51-84"),
    "lekiwi": (list(_LEKIWI), "src/lerobot/robots/lekiwi/lekiwi.py:78-108"),
    "lekiwi_client": (list(_LEKIWI), "src/lerobot/robots/lekiwi/lekiwi_client.py:88-106"),
    "bi_so_follower": ([f"left_{m}.pos" for m in _SO_ARM] + [f"right_{m}.pos" for m in _SO_ARM],
                       "src/lerobot/robots/bi_so_follower/bi_so_follower.py:98-108"),
}
# Cameras a robot type opens when the command sets no --robot.cameras (width/height are the delivered image size).
DEFAULT_CAMERAS = {
    robot: ({"front": {"type": "opencv", "width": 640, "height": 480, "fps": 30, "rotation": 180, "use_depth": False},
             "wrist": {"type": "opencv", "width": 480, "height": 640, "fps": 30, "rotation": 90, "use_depth": False}},
            "src/lerobot/robots/lekiwi/config_lekiwi.py:24-42, 57, 110")
    for robot in ("lekiwi", "lekiwi_client")
}
# Robot types whose configuration declares use_degrees (default True; src/lerobot/robots/so_follower/config_so_follower.py:42).
USE_DEGREES = {"so100_follower", "so101_follower"}


class _Flow:
    """Bounded parser for the YAML flow values draccus accepts on the command line ({a: 1, "b": [x, 'y']})."""

    def __init__(self, text: str, *, reject_duplicates=False):
        self.text, self.i = text, 0
        self.reject_duplicates = reject_duplicates

    def parse(self):
        value = self.value(0)
        self.space()
        if self.i != len(self.text):
            raise ValueError(f"unexpected text at position {self.i}")
        return value

    def space(self):
        while self.i < len(self.text):
            char = self.text[self.i]
            if char in " \t\r\n":
                self.i += 1
            elif char == "#" and (self.i == 0 or self.text[self.i - 1] in " \t\r\n"):
                while self.i < len(self.text) and self.text[self.i] != "\n":  # a YAML comment runs to the line end
                    self.i += 1
            else:
                break

    def value(self, depth: int):
        if depth > MAX_DEPTH:
            raise ValueError("nesting too deep")
        self.space()
        if self.i >= len(self.text):
            raise ValueError("missing value")
        char = self.text[self.i]
        if char == "{":
            return self.mapping(depth)
        if char == "[":
            return self.sequence(depth)
        if char in "'\"":
            return self.quoted()
        return _scalar(self.plain(",}]"))

    def mapping(self, depth: int) -> dict:
        self.i += 1
        result: dict = {}
        while True:
            self.space()
            if self.i < len(self.text) and self.text[self.i] == "}":
                self.i += 1
                return result
            key = self.quoted() if self.i < len(self.text) and self.text[self.i] in "'\"" else self.key()
            self.space()
            if self.i >= len(self.text) or self.text[self.i] != ":":
                raise ValueError(f"expected ':' after key {key!r}")
            self.i += 1
            if self.reject_duplicates and str(key) in result:
                raise ValueError('duplicate declaration key ' + str(key))
            result[str(key)] = self.value(depth + 1)
            self.space()
            if self.i < len(self.text) and self.text[self.i] == ",":
                self.i += 1
            elif self.i >= len(self.text) or self.text[self.i] != "}":
                raise ValueError("expected ',' or '}' in mapping")

    def sequence(self, depth: int) -> list:
        self.i += 1
        result: list = []
        while True:
            self.space()
            if self.i < len(self.text) and self.text[self.i] == "]":
                self.i += 1
                return result
            result.append(self.value(depth + 1))
            self.space()
            if self.i < len(self.text) and self.text[self.i] == ",":
                self.i += 1
            elif self.i >= len(self.text) or self.text[self.i] != "]":
                raise ValueError("expected ',' or ']' in sequence")

    def key(self) -> str:
        start = self.i
        while self.i < len(self.text):
            char = self.text[self.i]
            following = self.text[self.i + 1] if self.i + 1 < len(self.text) else " "
            if char == ":" and (following in " \t\r\n{['\"" or following in ",}"):
                break
            if char in ",{}[]":
                raise ValueError("malformed key")
            self.i += 1
        key = self.text[start:self.i].strip()
        if not key:
            raise ValueError("empty key")
        return key

    def plain(self, stops: str) -> str:
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in stops:
            if self.text[self.i] == "#" and self.text[self.i - 1] in " \t":
                break  # an inline comment ends a plain value
            self.i += 1
        return self.text[start:self.i].strip()

    def quoted(self) -> str:
        quote, self.i = self.text[self.i], self.i + 1
        out = []
        while self.i < len(self.text):
            char = self.text[self.i]
            if char == quote:
                if quote == "'" and self.text[self.i + 1:self.i + 2] == "'":
                    out.append("'")
                    self.i += 2
                    continue
                self.i += 1
                return "".join(out)
            if char == "\\" and quote == '"' and self.i + 1 < len(self.text):
                escape = self.text[self.i + 1]
                if escape == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", self.text[self.i + 2:self.i + 6]):
                    out.append(chr(int(self.text[self.i + 2:self.i + 6], 16)))
                    self.i += 6
                    continue
                out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(escape, escape))
                self.i += 2
                continue
            out.append(char)
            self.i += 1
        raise ValueError("unterminated quoted string")


def _scalar(text: str):
    lowered = text.lower()
    if lowered in ("true", "false", "yes", "no", "on", "off"):  # YAML 1.1 booleans, as draccus reads them
        return lowered in ("true", "yes", "on")
    if lowered in ("null", "~", "none", ""):
        return None
    if re.fullmatch(r"[+-]?0\d+", text):
        return text  # YAML 1.1 reads a leading zero as octal; kept as text so it is reported, not misread
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
        return float(text)
    return text


def _value(text: str, *, strict=False):
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        return _Flow(stripped, reject_duplicates=strict).parse()
    return _scalar(stripped)


def _commands(text: str, *, strict=False) -> list[list[str]]:
    """Split shell text into commands at unquoted newlines, ';', '&&', '||' and '|', dropping unquoted comments.

    Quoted values may span lines (the documented bimanual example does), so lines are not split naively.
    """
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    segments, current, quote, index = [], [], None, 0
    while index < len(text):
        char = text[index]
        if quote:
            if strict and quote == '"' and (char == '`' or char == '$' and text[index:index + 2] == '$('):
                raise ValueError('Supply one resolved robot command without command substitution')
            current.append(char)
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and index + 1 < len(text):
                current.append(text[index + 1])
                index += 1
        elif char in "'\"":
            quote = char
            current.append(char)
        elif char == "\\" and index + 1 < len(text):
            current.extend((char, text[index + 1]))
            index += 1
        elif char == "#" and (not current or current[-1] in " \t"):
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        elif strict and (char in ';&|<>' or char == '`' or char == '$' and text[index:index + 2] == '$('):
            raise ValueError('Supply one robot command without shell control operators, redirection or command substitution')
        elif char in "\n;&|":
            segments.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if quote:
        raise ValueError("unterminated quotation")
    segments.append("".join(current))
    return [shlex.split(segment) for segment in segments if segment.strip()]


WRAPPERS = {"uv", "run", "sudo", "env", "time", "nohup", "exec", "poetry", "pixi", "python", "python3", "-u", "--"}


def _program(tokens: list[str]) -> tuple[str | None, int]:
    """The LeRobot program this command runs, if it sits in command position (after env and launcher words)."""
    for index, token in enumerate(tokens):
        name = Path(token).name
        if name in PROGRAMS:
            return PROGRAMS[name], index
        if token == "-m" and index + 1 < len(tokens) and tokens[index + 1] in PROGRAMS:
            return PROGRAMS[tokens[index + 1]], index + 1
        if not (name in WRAPPERS or re.fullmatch(r"python3?(\.\d+)?", name) or re.fullmatch(r"[A-Za-z_]\w*=.*", token)):
            return None, -1
    return None, -1


def _nested(pairs: list[tuple[str, Any]]) -> dict:
    config: dict = {}
    for key, value in pairs:
        parts = key.split(".")
        node = config
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value  # Like draccus, a later occurrence of the same argument wins.
    return config


def from_command(text: str, *, strict=False) -> dict:
    """Interface of the ``lerobot-rollout`` or ``lerobot-record`` command in ``text`` (a pasted command or script)."""
    if type(strict) is not bool:
        raise InputError('strict must be a boolean.')
    if not isinstance(text, str) or len(text.encode()) > MAX_TEXT_BYTES:
        raise InputError(f"Expected command text no larger than {MAX_TEXT_BYTES} bytes")
    try:
        commands = _commands(text, strict=strict)
    except ValueError as exc:
        raise InputError(f"Cannot split the command like a shell would: {exc}") from exc
    if strict and len(commands) != 1:
        raise InputError('Supply exactly one robot command; separate shell steps cannot share a consumer comparison.')
    found = [(cmd,) + _program(cmd) for cmd in commands]
    found = [item for item in found if item[1] is not None]
    if not found:
        raise InputError("No lerobot-rollout or lerobot-record command found in the text")
    tokens, program, start = found[0]
    pairs, unknown, unresolved, index = [], [], [], start + 1
    seen_robot_keys = set()
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            key, raw = token[2:].split("=", 1)
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            key, raw = token[2:], tokens[index + 1]
            index += 1
        else:
            key, raw = token[2:], "true"
            if strict and (key == 'robot' or key.startswith('robot.')):
                raise InputError('Robot setting --' + key + ' needs an explicit value; for booleans use =true or =false.')
        if strict and (key == 'robot' or key.startswith('robot.')):
            if any(key == prior or key.startswith(prior + '.') or prior.startswith(key + '.') for prior in seen_robot_keys):
                raise InputError('Repeated robot setting --' + key + ' is ambiguous.')
            seen_robot_keys.add(key)
            if '$' in key or '`' in key:
                unknown.append('Robot option name depends on shell expansion; its declaration is unresolved.')
                unresolved.append(key)
                index += 1
                continue
        if key.startswith("policy.") and key not in ("policy.path", "policy.n_action_steps"):
            # Keep the unsupported name for uncertainty, never its arbitrary or secret-bearing value.
            pairs.append((key, None))
            index += 1
            continue
        if "$" in raw or "`" in raw:
            if key.startswith(("robot.", "policy.")) or key in ("fps", "dataset.fps", "rename_map", "config_path"):
                detail = "" if key.startswith("policy.") else f" ({raw})"
                unknown.append(f"--{key} depends on shell expansion{detail}; its value is not resolved")
                unresolved.append(key)
        else:
            try:
                pairs.append((key, _value(raw, strict=strict)))
            except ValueError as exc:
                if strict and (key == 'robot' or key.startswith('robot.')):
                    raise InputError('Cannot parse robot declaration --' + key + '; use an explicit correctly quoted native value.') from exc
                detail = "" if key.startswith("policy.") else f" ({exc})"
                unknown.append(f"--{key} could not be parsed as a value{detail}")
                unresolved.append(key)
        index += 1
    source = {"kind": "command", "sha256": hashlib.sha256(text.encode()).hexdigest(), "program_index": len(found) - 1}
    config = _nested(pairs)
    if strict:
        from .robot_contract import validate_robot_config, validate_robot_flag_names

        validate_robot_flag_names(config, seen_robot_keys)
        validate_robot_config(config)
    result = _interface(config, program, source, unknown, extra_commands=len(found) - 1, unresolved=unresolved)
    if strict:
        from .robot_contract import declaration_details

        result['declaration'] = declaration_details(config, result)
    return result


def from_config(path: str | Path, *, strict=False) -> dict:
    """Interface of a draccus config file (JSON, or YAML when PyYAML is installed) for lerobot-rollout/record."""
    if type(strict) is not bool:
        raise InputError('strict must be a boolean.')
    if strict:
        from .robot_contract import declaration_details, read_robot_config, validate_robot_config

        config, source = read_robot_config(path)
        validate_robot_config(config)
        program = 'record' if isinstance(config.get('dataset'), dict) and 'strategy' not in config else 'rollout'
        result = _interface(config, program, source, [], extra_commands=0)
        result['declaration'] = declaration_details(config, result)
        return result
    path = Path(path).expanduser()
    if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
        raise InputError(f"Expected a config file no larger than {MAX_TEXT_BYTES} bytes: {path}")
    if path.suffix.lower() == ".json":
        config = read_json(path)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise InputError("Reading a YAML config needs PyYAML: install policy-deployment-doctor[ros], or pass the "
                             "command line instead") from exc
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise InputError(f"Invalid YAML config: {exc}") from exc
        if not isinstance(config, dict):
            raise InputError("A rollout/record config must be a mapping")
    program = "record" if isinstance(config.get("dataset"), dict) and "strategy" not in config else "rollout"
    source = {"kind": "config", "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return _interface(config, program, source, [], extra_commands=0)


def _camera(name: str, raw: Any, unknown: list[str]) -> dict:
    if not isinstance(raw, dict):
        unknown.append(f"camera {name}: configuration is not a mapping")
        return {"type": None, "width": None, "height": None, "fps": None}
    camera = {"type": raw.get("type"), "width": raw.get("width"), "height": raw.get("height"), "fps": raw.get("fps"),
              "rotation": raw.get("rotation"), "color_mode": raw.get("color_mode"),
              "use_rgb": raw.get("use_rgb", True),
              "use_depth": raw.get("use_depth", False if raw.get("type") in {"opencv", "intelrealsense"} else None)}
    for flag, default in (("use_rgb", True), ("use_depth", False)):
        if flag == "use_depth" and flag not in raw and camera[flag] is None:
            continue  # An unmodeled camera type does not inherit a native color-only default.
        if not isinstance(camera[flag], bool):
            unknown.append(f"camera {name}: {flag} {camera[flag]!r} is not a boolean")
            camera[flag] = default if flag == "use_rgb" else None
    for field in ("width", "height"):
        if camera[field] is not None and (type(camera[field]) is not int or camera[field] <= 0):
            unknown.append(f"camera {name}: {field} {camera[field]!r} is not a positive integer")
            camera[field] = None
    if camera["width"] is None or camera["height"] is None:
        unknown.append(f"camera {name}: width/height not set, so the camera's own default resolution decides the "
                       f"image shape at connect time ({LEROBOT} src/lerobot/cameras/opencv/camera_opencv.py:228-232)")
    return camera


def _cameras(robot: dict, robot_type: str | None, unknown: list[str]) -> dict:
    """Camera configurations keyed as they appear in observations."""
    cameras: dict = {}
    if "cameras" not in robot and robot_type in DEFAULT_CAMERAS:
        defaults, source = DEFAULT_CAMERAS[robot_type]
        return {name: dict(camera, default=f"{LEROBOT} {source}") for name, camera in defaults.items()}
    top = robot.get("cameras") or {}
    if not isinstance(top, dict):
        unknown.append("robot.cameras is not a mapping")
        top = {}
    for name, raw in top.items():
        cameras[str(name)] = _camera(str(name), raw, unknown)
    if robot_type == "bi_so_follower":
        # Per-arm cameras are prefixed by side; top-level cameras keep their names (bi_so_follower.py:100-108).
        for side in ("left", "right"):
            arm = robot.get(f"{side}_arm_config") or {}
            arm_cameras = arm.get("cameras") if isinstance(arm, dict) else None
            for name, raw in (arm_cameras or {}).items() if isinstance(arm_cameras, dict) else ():
                cameras[f"{side}_{name}"] = _camera(f"{side}_{name}", raw, unknown)
    return cameras


def _interface(config: dict, program: str, source: dict, unknown: list[str], *, extra_commands: int,
               unresolved: list[str] | None = None) -> dict:
    unresolved = list(unresolved or [])
    if isinstance(config.get("config_path"), str):
        # draccus loads this file first and lets command-line values override it (src/lerobot/configs/parser.py:411-430).
        unknown.append(f"values from --config_path {config['config_path']} are not read; pass that file with "
                       "--robot-config, or give every value on the command line")
        given = {"fps": "fps" in config, "robot.cameras": isinstance(config.get("robot"), dict) and "cameras" in config["robot"],
                 "robot.type": isinstance(config.get("robot"), dict) and "type" in config["robot"],
                 "rename_map": "rename_map" in config, "dataset.fps": isinstance(config.get("dataset"), dict) and "fps" in config["dataset"]}
        unresolved += [field for field, present in given.items() if not present]
    robot = config.get("robot") if isinstance(config.get("robot"), dict) else {}
    robot_type = robot.get("type") if isinstance(robot.get("type"), str) else None
    if robot_type is None:
        unknown.append("no --robot.type given")
    names, names_source = ROBOTS.get(robot_type, (None, None)) if robot_type else (None, None)
    if robot_type and names is None:
        unknown.append(f"robot type {robot_type} is not modeled: its state and action names are unknown")
    cameras = {} if "robot.cameras" in unresolved else _cameras(robot, robot_type, unknown)
    features: dict = {}
    if names is not None:
        features["observation.state"] = {"names": list(names), "shape": [len(names)], "source": names_source}
    if robot_type == "lekiwi_client" and cameras:
        # The client decodes whatever frames the LeKiwi host sends; their size is set by the host's camera config
        # (src/lerobot/robots/lekiwi/lekiwi_client.py:109-110, 203-232).
        unknown.append("camera image sizes on lekiwi_client are set by the host's camera configuration, not this command")
    for name, camera in cameras.items():
        shape = ([camera["height"], camera["width"]] if camera["width"] and camera["height"]
                 and robot_type != "lekiwi_client" else None)
        if camera.get("use_rgb", True):
            features[f"observation.images.{name}"] = {"shape": shape + [3] if shape else None}
        if camera.get("use_depth") is None:
            unknown.append(f"camera {name}: whether it adds a depth image is unknown")
        elif camera.get("use_depth"):  # Depth streams are separate one-channel images (so_follower.py:86-97).
            features[f"observation.images.{name}_depth"] = {"shape": shape + [1] if shape else None}
    field = "fps" if program == "rollout" else "dataset.fps"
    if program == "rollout":
        declared = config.get("fps")
    else:
        declared = config["dataset"].get("fps") if isinstance(config.get("dataset"), dict) else None
    default, default_source = DEFAULT_FPS[program]
    if field in unresolved:
        fps = {"value": None, "declared": None, "source": f"--{field} not resolved"}
    elif isinstance(declared, (int, float)) and not isinstance(declared, bool):
        try:
            valid = math.isfinite(declared) and declared > 0
        except OverflowError:
            valid = False
        if valid:
            fps = {"value": declared, "declared": True}
        else:
            unknown.append(f"--{field} must be a positive finite number")
            fps = {"value": None, "declared": None, "source": f"--{field} invalid"}
    elif declared is not None:
        unknown.append(f"--{field} {declared!r} is not a number")
        fps = {"value": None, "declared": None, "source": f"--{field} not a number"}
    else:
        fps = {"value": default, "declared": False, "source": f"{LEROBOT} {default_source} (default)"}
    rename_map = config.get("rename_map") or {}
    if not isinstance(rename_map, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in rename_map.items()):
        unknown.append("rename_map is not a mapping of strings")
        rename_map = {}
    policy = config.get("policy") if isinstance(config.get("policy"), dict) else {}
    unsupported_policy = sorted(key for key in policy if key not in ("path", "n_action_steps"))
    if unsupported_policy:
        unknown.append(f"policy overrides {unsupported_policy} are not modeled; their values were not retained")
    policy_overrides = {}
    if "n_action_steps" in policy:
        steps = policy["n_action_steps"]
        if type(steps) is int and steps > 0:
            policy_overrides["n_action_steps"] = steps
        else:
            unknown.append("policy.n_action_steps is not a positive integer; its effective value is unresolved")
            unresolved.append("policy.n_action_steps")
    flags = {}
    if robot_type == "bi_so_follower":
        # Each arm has its own use_degrees (src/lerobot/robots/bi_so_follower/bi_so_follower.py:64, 78).
        arms = {side: robot.get(f"{side}_arm_config") if isinstance(robot.get(f"{side}_arm_config"), dict) else {}
                for side in ("left", "right")}
        flags["use_degrees"] = {side: arm.get("use_degrees", True) if isinstance(arm.get("use_degrees", True), bool) else None
                                for side, arm in arms.items()}
    elif robot_type in USE_DEGREES:
        flags["use_degrees"] = robot.get("use_degrees", True) if isinstance(robot.get("use_degrees", True), bool) else None
    if extra_commands:
        unknown.append(f"{extra_commands} further lerobot command(s) in the text were not read; only the first is used")
    return {"schema_version": 1, "kind": "robot_interface", "program": program, "source": source,
            "robot_type": robot_type, "robot_id": robot.get("id"), "fps": fps, "cameras": cameras,
            "features": features, "action_names": list(names) if names is not None else None,
            "rename_map": dict(rename_map), "policy_path": policy.get("path"),
            "policy_overrides": policy_overrides, "unsupported_policy_overrides": unsupported_policy,
            "flags": flags, "unknown": unknown, "unresolved": unresolved,
            "scope": ("Read from the command or config text only. The hardware was not contacted; calibration, wiring "
                      "and actual camera output are not checked.")}


def to_json(interface: dict) -> str:
    return json.dumps(interface, indent=2, sort_keys=True)
