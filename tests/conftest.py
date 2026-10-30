import json

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def case(tmp_path):
    cp, ds = tmp_path / "checkpoint", tmp_path / "dataset"
    cp.mkdir()
    (ds / "meta").mkdir(parents=True)
    names = ["shoulder", "gripper"]
    features = {"observation.state": {"type": "STATE", "shape": [2]},
                "observation.images.front": {"type": "VISUAL", "shape": [3, 32, 32]},
                "action": {"type": "ACTION", "shape": [2]}}
    mapping = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD", "VISUAL": "IDENTITY"}

    def put(path, data):
        path.write_text(json.dumps(data))
        return path

    put(cp / "config.json", {"type": "act", "input_features": {k: v for k, v in features.items() if k != "action"},
                            "output_features": {"action": features["action"]}, "normalization_mapping": mapping,
                            "chunk_size": 4, "n_action_steps": 4})
    config = {"features": features, "norm_map": mapping, "eps": 1e-8}
    put(cp / "policy_preprocessor.json", {"steps": [
        {"registry_name": "relative_actions_processor", "config": {"enabled": True, "action_names": names, "exclude_joints": ["gripper"]}},
        {"registry_name": "normalizer_processor", "config": config, "state_file": "pre.safetensors"}]})
    put(cp / "policy_postprocessor.json", {"steps": [
        {"registry_name": "unnormalizer_processor", "config": config, "state_file": "post.safetensors"},
        {"registry_name": "absolute_actions_processor", "config": {"enabled": True}}]})
    stats = {"action.mean": np.array([1., 0.5]), "action.std": np.array([2., 0.25]),
             "observation.state.mean": np.array([10., 0.5]), "observation.state.std": np.array([2., 0.25])}
    save_file(stats, cp / "pre.safetensors")
    save_file(stats, cp / "post.safetensors")
    put(ds / "meta/info.json", {"fps": 30, "robot_type": "test_arm", "codebase_version": "v3.0", "features": {
        "action": {"shape": [2], "names": names, "dtype": "float32"},
        "observation.state": {"shape": [2], "names": names, "dtype": "float32"},
        "observation.images.front": {"shape": [32, 32, 3], "names": ["height", "width", "channel"], "dtype": "video"}}})
    semantics = {"units": ["rad", "normalized"], "frame": "joint", "command_representation": "absolute", "gripper_convention": "0_closed_1_open"}
    deployment = put(tmp_path / "deployment.json", {"schema_version": 1, "robot_type": "test_arm", "fps": 30,
        "execute_steps": 4, "features": {k: dict(v, names=names) if k in ("action", "observation.state") else v for k, v in features.items()},
        **semantics, "training_contract": semantics, "model_output_representation": "relative", "limits": [[-20, 20], [0, 1]]})
    sample = put(tmp_path / "sample.json", {"observation.state": [12., 0.75], "reference_state": [10., 0.5],
        "model_output": [[0., 0.], [1., 1.], [2., -1.]], "expected_commands": [[11., 0.5], [13., 0.75], [15., 0.25]]})
    return {"checkpoint": cp, "dataset": ds, "deployment": deployment, "sample": sample, "put": put}
