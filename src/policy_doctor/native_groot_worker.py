"""Experimental standalone GR00T state worker (Python 3.10+).

Only an operator-configured trusted runtime may execute this worker. JSON data
cannot select modules or callables. Resource/offline process controls belong to
the parent; this module does not load a model or execute artifact code.
"""

import contextlib
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

MAX_REQUEST = 1024 * 1024
MAX_RESPONSE = 128 * 1024
MAX_METADATA = 4 * 1024 * 1024
MAX_DIM = 4096
FLAGS = ("use_percentiles", "clip_outliers", "apply_sincos_state_encoding", "use_relative_action")
PROVIDER = "groot_state"
UNVERIFIED = [
    "Declared state metadata does not establish physical units, frames or training meaning.",
    "frame_index is a selected sample coordinate; only an explicit selection_basis establishes whether a native frame column or single-episode physical-row fallback supplied it.",
    "Camera/video, language, actions, model predictions, inference and physical behavior are untested.",
    "Manual original-metadata key grouping and float32 conversion precede the native state component; no native dataset loader ran.",
    "Sin/cos state encoding does not establish learned or statistics-based normalization.",
    "Source/checkpoint historical association and redistribution rights require independent evidence.",
]


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a nonempty relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(p in {".", ".."} for p in value.split("/")):
        raise ValueError("Absolute paths and path traversal are forbidden in retained evidence")
    return value


def _directory(value):
    path = Path(value)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_dir():
        raise ValueError("Input root must be an existing directory without symlinks")
    return path.resolve()


def _read(root, relative):
    path = root / _relative(relative)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Metadata/source file missing or symlinked: " + relative)
    if path.stat().st_size > MAX_METADATA:
        raise ValueError("Metadata/source file exceeds size limit: " + relative)
    data = path.read_bytes()
    if len(data) > MAX_METADATA:
        raise ValueError("Metadata/source file exceeds size limit: " + relative)
    return data


def _metadata(root, relative):
    data = _read(root, relative)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=pairs,
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + item)))
    if not isinstance(value, dict):
        raise ValueError("Metadata must be a JSON object: " + relative)
    return value, data


def _integer(value, label):
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(label + " must be a bounded nonnegative integer")
    return value


def _prepare(request):
    if request.get("operation") not in {"probe", "replay"}:
        raise ValueError("operation must be probe or replay")
    source = _directory(request["source_root"])
    model, dataset = _directory(request["model"]), _directory(request["dataset"])
    config, config_bytes = _metadata(model, "processor_config.json")
    statistics, stats_bytes = _metadata(model, "statistics.json")
    modality, modality_bytes = _metadata(dataset, "meta/modality.json")
    info, info_bytes = _metadata(dataset, "meta/info.json")
    kwargs = config.get("processor_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("Missing original processor_kwargs")
    for key in FLAGS:
        if type(kwargs.get(key)) is not bool:
            raise ValueError("Original checkpoint boolean flag missing: " + key)
    configs = kwargs.get("modality_configs", {})
    state_config = configs.get("gr1", {}).get("state", {})
    keys = state_config.get("modality_keys")
    if (not isinstance(keys, list) or not 1 <= len(keys) <= 32
            or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", k) for k in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("Bounded original GR1 state keys are required")
    bindings = []
    for key in keys:
        item = modality.get("state", {}).get(key)
        if not isinstance(item, dict):
            raise ValueError("Original state modality mapping missing: " + key)
        original = item.get("original_key", "observation.state")
        if not isinstance(original, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", original):
            raise ValueError("Invalid original column name")
        start, end = item.get("start"), item.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= MAX_DIM:
            raise ValueError("Invalid original metadata slice: " + key)
        shape = info.get("features", {}).get(original, {}).get("shape")
        if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or not end <= shape[0] <= MAX_DIM:
            raise ValueError("Original declared state width missing or inconsistent: " + original)
        bindings.append({"key": key, "original_key": original, "start": start, "end": end,
                         "declared_column_shape": shape})
    selected, rows = request.get("samples"), request.get("raw_samples")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 2 or not isinstance(rows, list) or len(rows) != len(selected):
        raise ValueError("One or two selected real samples and corresponding raw_samples are required")
    identities = set()
    coordinate_modes = set()
    for selector, row in zip(selected, rows, strict=False):
        if not isinstance(selector, dict) or not isinstance(row, dict):
            raise ValueError("Sample entries must be objects")
        identity = row.get("identity", {})
        if not isinstance(identity, dict) or set(identity) != {"episode_index", "frame_index", "index", "timestamp", "physical_row"}:
            raise ValueError("Exact native identity/selection coordinate fields are required")
        if set(selector) == {"episode_index", "physical_row"}:
            coordinate = "physical_row"
            if row.get("selection_basis") != "single_episode_physical_row":
                raise ValueError("Physical-row selection requires explicit single_episode_physical_row basis")
        elif set(selector) == {"episode_index", "frame_index"}:
            coordinate = "frame_index"
        else:
            raise ValueError("Select explicit episode/frame or episode/physical_row coordinates")
        coordinate_modes.add(coordinate)
        if len(coordinate_modes) > 1:
            raise ValueError("Cannot mix frame and physical-row selectors")
        for key in ("episode_index", "index", "physical_row"):
            _integer(identity.get(key), key)
        if identity.get("frame_index") is not None or coordinate == "frame_index":
            _integer(identity.get("frame_index"), "frame_index")
        if "selection_basis" in row and row["selection_basis"] not in {"original_frame_index", "single_episode_physical_row"}:
            raise ValueError("Unknown native sample selection_basis")
        timestamp = identity.get("timestamp")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError("Original timestamp must be finite")
        pair = identity["episode_index"], identity[coordinate]
        if pair in identities or pair != (selector.get("episode_index"), selector.get(coordinate)):
            raise ValueError("Raw identity does not match unique selected native coordinate")
        for key in ("episode_index", coordinate):
            _integer(selector.get(key), key)
        identities.add(pair)
        origin = row.get("source", {})
        if not isinstance(origin, dict) or set(origin) != {"path", "sha256"}:
            raise ValueError("Exact original source path/hash fields are required")
        _relative(origin.get("path"))
        if not isinstance(origin.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", origin["sha256"]):
            raise ValueError("Original Parquet SHA256 required")
        columns = row.get("columns")
        if not isinstance(columns, dict) or len(columns) > 32:
            raise ValueError("Bounded original columns object required")
        for binding in bindings:
            values = columns.get(binding["original_key"])
            if not isinstance(values, list) or len(values) != binding["declared_column_shape"][0]:
                raise ValueError("Original state column width disagrees with metadata")
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4e38 for v in values):
                raise ValueError("Original state column must contain finite float32-representable numbers")
    return source, kwargs, statistics, bindings, rows, {
        "metadata/model/processor_config.json": config_bytes,
        "metadata/model/statistics.json": stats_bytes,
        "metadata/dataset/meta/modality.json": modality_bytes,
        "metadata/dataset/meta/info.json": info_bytes,
    }


def _replay_request(request):
    capsule = _directory(request["capsule_root"])
    manifest, _ = _metadata(capsule, "worker_manifest.json")
    if manifest.get("provider") != PROVIDER or manifest.get("schema_version") != 1:
        raise ValueError("Invalid retained worker manifest")
    retained = manifest.get("retained_files")
    if not isinstance(retained, list) or not 1 <= len(retained) <= 128:
        raise ValueError("Invalid retained file inventory")
    for item in retained:
        if _digest(_read(capsule, item["path"])) != item["sha256"]:
            raise ValueError("Retained file hash mismatch: " + item["path"])
    raw, _ = _metadata(capsule, manifest["raw_samples_path"])
    return {"operation": "replay", "source_root": request["source_root"],
            "model": str(capsule / _relative(manifest["model_root"])),
            "dataset": str(capsule / _relative(manifest["dataset_root"])),
            "output": request["output"], "samples": manifest["samples"],
            "raw_samples": raw["raw_samples"], "artifact_receipts": manifest.get("artifact_receipts", []),
            "expected_source_files": manifest["source_files"]}


def run(request):
    """Execute a fixed native state route; return bounded JSON-compatible evidence."""
    result = {"provider": PROVIDER, "execution_status": "error", "scope": "native GR1 state component only",
              "samples": [], "native_versions": {}, "source_files": [], "exports": [], "unverified": list(UNVERIFIED)}
    output = None
    try:
        if not isinstance(request, dict) or len(_json_bytes(request)) > MAX_REQUEST:
            raise ValueError("Worker request must be a bounded JSON object")
        if request.get("operation") == "replay":
            request = _replay_request(request)
        source, kwargs, statistics, bindings, rows, retained = _prepare(request)
        receipts = request.get("artifact_receipts", [])
        if not isinstance(receipts, list) or len(receipts) > 128:
            raise ValueError("Invalid artifact receipt inventory")
        for receipt in receipts:
            if receipt.get("root") not in {"model", "dataset"}:
                raise ValueError("Invalid artifact receipt root")
            _relative(receipt.get("path"))
            _integer(receipt.get("size_bytes"), "artifact size")
            if not re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", "")):
                raise ValueError("Invalid artifact receipt hash")
            if not isinstance(receipt.get("role"), str) or len(receipt["role"]) > 128:
                raise ValueError("Invalid artifact receipt role")
        # Original membership receipts come from bytes actually present in the
        # authorized original roots. A raw-column capture assertion alone is
        # insufficient, including an upstream Git LFS pointer in place of data.
        if request["operation"] == "probe":
            receipts = []
            for retained_path, data in retained.items():
                root_name, relative = retained_path.removeprefix("metadata/").split("/", 1)
                receipts.append({"root": root_name, "path": relative, "sha256": _digest(data),
                                 "size_bytes": len(data), "role": "original_native_metadata"})
            seen_sources = {}
            dataset_root = _directory(request["dataset"])
            for row in rows:
                relative = _relative(row["source"]["path"])
                if relative not in seen_sources:
                    path = dataset_root / relative
                    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
                        raise ValueError("Original Parquet source missing or symlinked")
                    size = path.stat().st_size
                    if size > 128 * 1024 * 1024:
                        raise ValueError("Original Parquet exceeds pilot byte bound")
                    digest = hashlib.sha256()
                    count = 0
                    with path.open("rb") as stream:
                        while block := stream.read(1024 * 1024):
                            count += len(block)
                            if count > 128 * 1024 * 1024:
                                raise ValueError("Original Parquet grew beyond pilot byte bound")
                            digest.update(block)
                    if count != size:
                        raise ValueError("Original Parquet changed during hashing")
                    seen_sources[relative] = {"root": "dataset", "path": relative,
                        "sha256": digest.hexdigest(), "size_bytes": size, "role": "original_parquet"}
                    receipts.append(seen_sources[relative])
                if seen_sources[relative]["sha256"] != row["source"]["sha256"]:
                    raise ValueError("Original Parquet bytes disagree with raw-sample source SHA256")
        result["artifact_receipts"] = receipts
        expected_sources = request.get("expected_source_files", [])
        for item in expected_sources:
            if _digest(_read(source, item["path"])) != item["sha256"]:
                raise ValueError("Trusted runtime source differs from retained source pin")
        candidate_output = Path(request["output"])
        if any(p.is_symlink() for p in (candidate_output, *candidate_output.parents)):
            raise ValueError("Output may not contain symlinks")
        if candidate_output.exists() and (not candidate_output.is_dir() or any(candidate_output.iterdir())):
            raise ValueError("Output must be a new or empty staging directory")
        candidate_output.mkdir(mode=0o700, parents=False, exist_ok=True)
        output = candidate_output
        sys.path.insert(0, str(source))
        # Static imports only. The caller's configured source tree is trusted;
        # metadata cannot choose a module, class, callback or executable.
        import numpy as np
        import scipy
        from gr00t.data.state_action.state_action_processor import StateActionProcessor

        imported = {}
        for name, module in list(sys.modules.items()):
            if name == "gr00t" or name.startswith("gr00t."):
                filename = getattr(module, "__file__", None)
                if filename:
                    path = Path(filename).resolve()
                    relative = path.relative_to(source).as_posix()
                    imported[relative] = _digest(_read(source, relative))
        result["source_files"] = [{"path": path, "sha256": digest} for path, digest in sorted(imported.items())]
        result["native_versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            processor = StateActionProcessor(modality_configs=kwargs["modality_configs"], statistics=statistics,
                                             **{key: kwargs[key] for key in FLAGS})
            processor.eval()
        if captured.tell() > 32768:
            raise ValueError("Native constructor log exceeds bound")
        if captured.getvalue():
            result["native_log"] = captured.getvalue()
        def export(path, data):
            target = output / path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            result["exports"].append({"path": path, "sha256": _digest(data), "bytes": len(data)})
        def summary(array):
            value = np.asarray(array)
            finite = bool(np.isfinite(value).all())
            if value.size > MAX_DIM * 2 or not finite:
                raise ValueError("Native output exceeds size bound or contains nonfinite values")
            return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _digest(value.tobytes(order="C")),
                    "finite": finite, "min": float(value.min()) if value.size else None,
                    "max": float(value.max()) if value.size else None, "preview": value.reshape(-1)[:8].tolist()}
        for path, data in retained.items():
            export(path, data)
        export("inputs/raw_samples.json", _json_bytes({"raw_samples": rows}))
        for number, row in enumerate(rows):
            sample = {"identity": row["identity"], "selection_basis": row.get("selection_basis", "unspecified"),
                      "source": row["source"], "bindings": bindings, "routes": []}
            route = {"name": "native_state", "status": "error", "steps": [{
                "class": "gr00t.data.state_action.state_action_processor.StateActionProcessor",
                "method": "apply_state", "source_file": "gr00t/data/state_action/state_action_processor.py",
                "source_sha256": imported["gr00t/data/state_action/state_action_processor.py"],
                "parameters": {key: kwargs[key] for key in FLAGS}, "embodiment_tag": "gr1",
                "preparation": "Original metadata slices; original numeric column -> float32; add single-step axis",
            }], "input": {}, "output": {}}
            sample["routes"].append(route)
            result["samples"].append(sample)
            try:
                state = {b["key"]: np.asarray(row["columns"][b["original_key"]], dtype=np.float32)[None, b["start"]:b["end"]] for b in bindings}
                route["input"] = {key: summary(value) for key, value in state.items()}
                with contextlib.redirect_stdout(captured):
                    processed = processor.apply_state(state, "gr1")
                route["output"] = {key: summary(processed[key]) for key in state}
                for index, (key, value) in enumerate(processed.items()):
                    relative = f"samples/{number:02d}/native_state/{index:02d}.json"
                    export(relative, _json_bytes({"key": key, "shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}))
                    route["output"][key]["export"] = relative
                base = processor.get_state_dim("gr1")
                expanded = processor.get_state_dim("gr1", include_sincos_expansion=True)
                observed = sum(v.shape[-1] for v in processed.values())
                route["dimensions"] = {"native_declared_base": base, "native_declared_expanded": expanded,
                                        "observed_output": observed, "checkpoint_max_state_dim": kwargs.get("max_state_dim")}
                route["dimension_observation"] = {
                    "base_differs_from_observed": base != observed,
                    "expanded_matches_observed": expanded == observed,
                    "interpretation": "Configured sin/cos can expand state width; a declared/base width difference alone is not a policy fault.",
                }
                route["status"] = "complete"
            except Exception as exc:
                route["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
        export("worker_manifest.json", _json_bytes({
            "schema_version": 1, "provider": PROVIDER,
            "raw_samples_path": "inputs/raw_samples.json",
            "model_root": "metadata/model", "dataset_root": "metadata/dataset",
            "samples": request["samples"], "source_files": result["source_files"],
            "artifact_receipts": receipts, "retained_files": list(result["exports"]),
            "route_recipe": "Static native GR1 StateActionProcessor.apply_state; original metadata slicing and float32 conversion",
        }))
        completed = sum(s["routes"][0]["status"] == "complete" for s in result["samples"])
        result["execution_status"] = "complete" if completed == len(rows) else "partial" if completed else "error"
        if any(_digest(_read(source, item["path"])) != item["sha256"] for item in result["source_files"]):
            raise ValueError("Native source changed during execution")
    except Exception as exc:
        result["execution_status"] = "error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    if len(_json_bytes(result)) > MAX_RESPONSE:
        result = {"provider": PROVIDER, "execution_status": "error", "scope": "native GR1 state component only",
                  "samples": [], "native_versions": {}, "source_files": [], "exports": [],
                  "unverified": list(UNVERIFIED), "error": {"type": "ValueError", "message": "Response exceeds 128 KiB"}}
    if output is not None and output.is_dir() and not (output / "result.json").exists():
        (output / "result.json").write_bytes(_json_bytes(result))
    return result


def main():
    try:
        data = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(data) > MAX_REQUEST:
            raise ValueError("Request exceeds 1 MiB")
        request = json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        result = run(request)
    except Exception as exc:
        result = {"provider": PROVIDER, "execution_status": "error", "scope": "native GR1 state component only",
                  "samples": [], "native_versions": {}, "source_files": [], "exports": [],
                  "unverified": list(UNVERIFIED), "error": {"type": type(exc).__name__, "message": str(exc)[:2048]}}
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    return 0 if result["execution_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
