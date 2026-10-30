"""Generate a small declared synthetic example; it contains no robot recordings."""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def main():
    root = Path(__file__).parent / "demo"
    cp, ds = root / "checkpoint", root / "dataset/meta"
    cp.mkdir(parents=True, exist_ok=True)
    ds.mkdir(parents=True, exist_ok=True)

    def write(path, data):
        path.write_text(json.dumps(data, indent=2) + "\n")

    names = ["shoulder", "gripper"]
    state = {"type": "STATE", "shape": [2]}
    action = {"type": "ACTION", "shape": [2]}
    mapping = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    features = {"observation.state": state, "action": action}
    write(cp / "config.json", {"type": "act", "input_features": {"observation.state": state},
                              "output_features": {"action": action}, "normalization_mapping": mapping,
                              "chunk_size": 4, "n_action_steps": 4})
    normalization = {"features": features, "norm_map": mapping, "eps": 1e-8}
    write(cp / "policy_preprocessor.json", {"steps": [
        {"registry_name": "relative_actions_processor", "config": {"enabled": True, "action_names": names, "exclude_joints": ["gripper"]}},
        {"registry_name": "normalizer_processor", "config": normalization, "state_file": "pre.safetensors"}]})
    write(cp / "policy_postprocessor.json", {"steps": [
        {"registry_name": "unnormalizer_processor", "config": normalization, "state_file": "post.safetensors"},
        {"registry_name": "absolute_actions_processor", "config": {"enabled": True}}]})
    stats = {"action.mean": np.array([1., .5]), "action.std": np.array([2., .25]),
             "observation.state.mean": np.array([10., .5]), "observation.state.std": np.array([2., .25])}
    save_file(stats, cp / "pre.safetensors")
    save_file(stats, cp / "post.safetensors")
    write(ds / "info.json", {"codebase_version": "v3.0", "robot_type": "synthetic_arm", "fps": 30,
                            "features": {k: dict(v, names=names) for k, v in features.items()}})
    semantics = {"units": ["rad", "normalized"], "frame": "joint", "command_representation": "absolute", "gripper_convention": "0_closed_1_open"}
    healthy = {"schema_version": 1, "robot_type": "synthetic_arm", "fps": 30, "execute_steps": 4,
               "features": {k: dict(v, names=names) for k, v in features.items()}, **semantics,
               "training_contract": dict(semantics), "model_output_representation": "relative", "limits": [[-20, 20], [0, 1]]}
    write(root / "healthy.json", healthy)
    broken = json.loads(json.dumps(healthy))
    broken["features"]["action"]["names"] = ["gripper", "shoulder"]
    broken["units"] = ["deg", "normalized"]
    broken["execute_steps"] = 20
    write(root / "broken.json", broken)
    write(root / "sample.json", {"observation.state": [12., .75], "reference_state": [10., .5],
                                "model_output": [[0., 0.], [1., 1.], [2., -1.]],
                                "expected_commands": [[11., .5], [13., .75], [15., .25]]})
    print(root.resolve())


if __name__ == "__main__":
    main()
