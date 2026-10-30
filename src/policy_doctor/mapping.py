"""Compose declared observation-key renames without changing action dimensions."""

from .artifacts import InputError, Snapshot, step_kind


def validate_rename_map(mapping) -> dict[str, str]:
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not k or
                                          not isinstance(v, str) or not v for k, v in mapping.items()):
        raise InputError("rename_map must map nonempty string source keys to nonempty string target keys")
    if any(k == "action" or v == "action" for k, v in mapping.items()):
        raise InputError("Observation rename_map collides with the action interface")
    if len(set(mapping.values())) != len(mapping):
        raise InputError("rename_map has colliding targets")
    return mapping


def feature_mapping(checkpoint: Snapshot, interface: Snapshot) -> dict[str, str]:
    """Runtime adapter first, then saved observation rename processors in order.

    Dataset keys pass only through saved processors. Original keys remain the
    provenance addresses even when the effective policy key changes.
    """
    chain = []
    if interface.kind == "deployment":
        chain.append(validate_rename_map(interface.config.get("rename_map", {})))
    chain.extend(validate_rename_map(s["config"].get("rename_map", {}))
                 for s in checkpoint.pre if step_kind(s["name"]) == "rename")
    effective = {key: key for key in interface.features}
    for mapping in chain:
        effective = {source: mapping.get(target, target) for source, target in effective.items()}
        if len(set(effective.values())) != len(effective):
            raise InputError("rename_map collides with an existing feature")
    return effective
