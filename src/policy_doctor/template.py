"""Deployment declaration template. Nulls intentionally preserve unresolved semantics."""

from .artifacts import load_checkpoint, load_dataset


def deployment_template(checkpoint: str, dataset: str | None = None) -> dict:
    cp = load_checkpoint(checkpoint)
    ds = load_dataset(dataset) if dataset is not None else None
    features = {}
    for key, feature in cp.features.items():
        if key not in ("action", "observation.state"):
            features[key] = feature
        elif ds is not None and key in ds.features:
            # The training data is the interface the policy learned; config.json can carry a stale base width.
            features[key] = dict(feature, shape=ds.features[key]["shape"], names=ds.features[key].get("names"))
        else:
            features[key] = dict(feature, names=None)
    semantics = {"units": None, "frame": None, "command_representation": None, "gripper_convention": None}
    notes = ("Shapes/default horizon are copied from the checkpoint, not measured on hardware. Replace features with the "
             "actual runtime interface; fill semantics from collection/driver evidence.")
    if ds is not None:
        notes = ("State/action shapes and names, fps and robot_type come from the training dataset's meta/info.json; "
                 "cameras and the default horizon come from the checkpoint. None is measured on hardware: replace "
                 "features with the actual runtime interface and fill semantics from collection/driver evidence.")
    return {"schema_version": 1,
            "robot_type": ds.config.get("robot_type") if ds is not None else None,
            "fps": ds.config.get("fps") if ds is not None else None,
            "execute_steps": cp.config.get("n_action_steps"),
            "n_obs_steps": cp.config.get("n_obs_steps"),
            "features": features,
            "rename_map": {}, "model_output_representation": None,
            **semantics, "training_contract": dict(semantics),
            "notes": notes}
