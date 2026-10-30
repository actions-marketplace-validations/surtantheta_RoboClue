"""CPU numerical trace of saved vector normalization and decoded action chunks.

This is not a model forward pass. It operates on caller-supplied model outputs.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .artifacts import InputError, load_checkpoint, load_deployment, read_json, sha256, step_kind
from .diagnostics import (
    MODES,
    Report,
    _processor_checks,
    _relative_order_checks,
    check_stat,
    normal_mode,
    normalizer,
)


def vector_array(value, name: str, dim: int, chunk: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{name} must contain numeric values") from exc
    expected = array.ndim == 2 and array.shape[0] > 0 if chunk else array.ndim == 1
    if not expected or array.shape[-1] != dim or not np.isfinite(array).all():
        raise InputError(f"{name} must be {'[time, ' + str(dim) + ']' if chunk else '[' + str(dim) + ']'} finite numbers")
    if array.size > 100_000:
        raise InputError("Saved trace exceeds 100,000 values")
    return array


def transform(array: np.ndarray, stats: dict, mode: str, eps: float, inverse: bool) -> np.ndarray:
    problem = check_stat(stats, mode, array.shape[-1])
    if problem:
        raise InputError(problem)
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not np.isfinite(eps) or eps <= 0:
        raise InputError("Normalization eps must be finite and positive")
    if mode == "IDENTITY":
        return array.copy()
    if mode == "MEAN_STD":
        mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        return array * std + mean if inverse else (array - mean) / (std + eps)
    low_key, high_key = MODES[mode]
    lo, hi = np.asarray(stats[low_key]), np.asarray(stats[high_key])
    span = np.where(hi == lo, eps, hi - lo)
    return (array + 1) / 2 * span + lo if inverse else 2 * (array - lo) / span - 1


def trace_actions(checkpoint: str, sample: str, deployment: str | None = None) -> dict:
    from pathlib import Path

    cp = load_checkpoint(checkpoint)
    dep = load_deployment(deployment) if deployment else None
    validation = Report()
    _processor_checks(cp, validation)
    _relative_order_checks(cp, [dep] if dep else [], validation)
    errors = [f.code for f in validation.findings if f.status == "error"]
    if errors:
        raise InputError("Checkpoint processor checks failed: " + ", ".join(sorted(set(errors))))
    if normalizer(cp, "post") is None:
        raise InputError("Action trace requires exactly one saved unnormalizer")
    data = read_json(sample)
    feature = cp.features.get("action")
    if feature is None or len(feature["shape"]) != 1:
        raise InputError("Action trace requires a declared one-dimensional action feature")
    dim = feature["shape"][0]
    output = vector_array(data.get("model_output"), "model_output", dim, chunk=True)
    if cp.config.get("chunk_size") is not None and len(output) > cp.config["chunk_size"]:
        raise InputError("Saved model output exceeds the checkpoint chunk_size")
    if not cp.post:
        raise InputError("A saved postprocessor is required; legacy embedded decoding is unsupported")
    stages = [{"stage": "model_output", "values": output.tolist()}]
    supported = {"unnormalize", "absolute", "device"}
    for step in cp.post:
        stage_kind = step_kind(step["name"])
        if stage_kind not in supported:
            raise InputError(f"Unsupported postprocessor {step['name']}; refusing an incomplete action trace")
        if step.get("missing_state"):
            raise InputError("Cannot trace with a missing processor state")
        if stage_kind == "unnormalize":
            mode = normal_mode(step, "action", feature)
            output = transform(output, step["stats"].get("action", {}), mode, step["config"].get("eps", 1e-8), True)
        if stage_kind == "absolute" and step["config"].get("enabled"):
            relative = [s for s in cp.pre if step_kind(s["name"]) == "relative" and s["config"].get("enabled")]
            if len(relative) != 1:
                raise InputError("Absolute decoding requires exactly one paired enabled relative processor")
            config = relative[0]["config"]
            anchor = vector_array(data.get("reference_state"), "reference_state (raw state at chunk generation)", dim)
            action_names, excluded = config.get("action_names"), config.get("exclude_joints", [])
            if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
                raise InputError("exclude_joints must be a list of names")
            if excluded and (not isinstance(action_names, list) or len(action_names) != dim):
                raise InputError("Excluded joints require complete action_names to build an unambiguous mask")
            mask = [not any(token.lower() in name.lower() for token in excluded if token)
                    for name in action_names] if excluded else [True] * dim
            output = output + anchor * np.asarray(mask, dtype=float)
        if not np.isfinite(output).all():
            raise InputError("Action decoding produced non-finite values")
        stages.append({"stage": step["name"], "values": output.tolist(),
                       "evidence": {"path": step["source"], "pointer": step["pointer"]}})
    pre = normalizer(cp, "pre")
    state_trace = None
    if "observation.state" in data:
        state_feature = cp.features.get("observation.state")
        if state_feature is None or len(state_feature["shape"]) != 1:
            raise InputError("Saved state has no supported checkpoint state feature")
        state = vector_array(data["observation.state"], "observation.state", state_feature["shape"][0])
        if pre is None:
            raise InputError("State normalization trace requires exactly one saved normalizer")
        mode = normal_mode(pre, "observation.state", state_feature)
        normalized = transform(state, pre["stats"].get("observation.state", {}), mode, pre["config"].get("eps", 1e-8), False)
        state_trace = {"raw": state.tolist(), "normalized": normalized.tolist(), "mode": mode,
                       "scope": "Isolated vector normalization, not image/language/model preprocessing."}
    findings = []
    if "expected_commands" in data:
        expected = vector_array(data["expected_commands"], "expected_commands", dim, chunk=True)
        if expected.shape != output.shape:
            raise InputError("expected_commands shape differs from the traced chunk")
        findings.append({"code": "trace.expected_commands", "status": "pass" if np.allclose(expected, output, rtol=1e-6, atol=1e-8) else "error",
                         "max_absolute_error": float(np.max(np.abs(expected - output)))})
    if dep and dep.config.get("limits") is not None:
        bounds = np.asarray(dep.config["limits"], dtype=float)
        if bounds.shape != (dim, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
            raise InputError("Deployment limits must be [action_dim, 2] ordered finite bounds")
        failures = np.argwhere((output < bounds[:, 0]) | (output > bounds[:, 1]))
        findings.append({"code": "trace.command_limits", "status": "error" if len(failures) else "pass",
                         "violations": [{"time": int(t), "dimension": int(d), "value": float(output[t, d]),
                                         "bounds": bounds[d].tolist()} for t, d in failures[:100]],
                         "total_violations": len(failures)})
    sample_path = Path(sample).resolve()
    return {"schema_version": 1, "scope": "Saved model-output decoding only; no model inference or robot commands.",
            "status": "error" if any(f["status"] == "error" for f in findings) else "pass",
            "stages": stages, "commands": output.tolist(), "state_normalization": state_trace,
            "findings": findings, "unsupported_preprocessors": [s["name"] for s in cp.pre if step_kind(s["name"]) == "unsupported"],
            "artifacts": [asdict(a) for a in cp.artifacts] + ([asdict(a) for a in dep.artifacts] if dep else []) +
                         [{"path": str(sample_path), "sha256": sha256(sample_path), "role": "saved_sample", "size_bytes": sample_path.stat().st_size}]}
