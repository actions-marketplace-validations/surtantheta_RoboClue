"""Experimental standalone native LeRobot worker; no policy_doctor or top-level ML imports.

Only the explicitly configured trusted runtime supplies executable code. Artifacts are
bounded data. This pilot supports XVLA saved/current preprocessing, not policy inference.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import string
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import get_origin

PROVIDER = "lerobot"
SCHEMA = "policy_doctor.native_lerobot.v1"
MAX_RESPONSE = 128 * 1024
MAX_JSON = 2 * 1024 * 1024
MAX_ARRAY = 32 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
SCOPE = (
    "Observed native XVLA preprocessing and retained-input/export self-consistency only; "
    "no model, policy output, inference, simulator, GPU, training-correctness or competence claim."
)
STEP_FIELDS = {
    "rename_observations_processor": {"rename_map"},
    "to_batch_processor": set(),
    "tokenizer_processor": {
        "max_length", "task_key", "padding_side", "padding", "truncation", "tokenizer_name"
    },
    "xvla_image_to_float": {"image_keys", "validate_range"},
    "xvla_imagenet_normalize": {"image_keys"},
    "xvla_image_scale": {"image_keys"},
    "xvla_add_domain_id": {"domain_id"},
    "xvla_rotation_6d_to_axis_angle": {"expected_action_dim"},
    "device_processor": {"device", "float_dtype"},
    "normalizer_processor": {"eps", "features", "norm_map", "normalize_observation_keys"},
    "unnormalizer_processor": {"eps", "features", "norm_map", "normalize_observation_keys"},
}
STEP_CLASSES = {
    "rename_observations_processor": "lerobot.processor.rename_processor.RenameObservationsProcessorStep",
    "to_batch_processor": "lerobot.processor.batch_processor.AddBatchDimensionProcessorStep",
    "tokenizer_processor": "lerobot.processor.tokenizer_processor.TokenizerProcessorStep",
    "device_processor": "lerobot.processor.device_processor.DeviceProcessorStep",
    "normalizer_processor": "lerobot.processor.normalize_processor.NormalizerProcessorStep",
    "unnormalizer_processor": "lerobot.processor.normalize_processor.UnnormalizerProcessorStep",
    **{name: "lerobot.policies.xvla.processor_xvla." + cls for name, cls in {
        "xvla_image_to_float": "XVLAImageToFloatProcessorStep",
        "xvla_imagenet_normalize": "XVLAImageNetNormalizeProcessorStep",
        "xvla_image_scale": "XVLAImageScaleProcessorStep",
        "xvla_add_domain_id": "XVLAAddDomainIdProcessorStep",
        "xvla_rotation_6d_to_axis_angle": "XVLARotation6DToAxisAngleProcessorStep",
    }.items()},
}
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "added_tokens.json", "config.json",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _integer(value, label, maximum=10_000_000):
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{label} must be a bounded nonnegative integer")
    return value


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value or ":" in value:
        raise ValueError("Expected a safe relative artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("Artifact path must not escape its root")
    return path.as_posix()


def _within(root, relative):
    relative = _relative(relative)
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlinked artifact paths are unsupported")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its root")
    return path


def _hash(path, maximum=MAX_FILE):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError(f"Missing, symlinked or oversized file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, maximum=MAX_JSON):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError(f"Missing or oversized JSON: {path.name}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _write_json(path, value):
    data = _canonical(value)
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _no_artifact_code(value):
    if isinstance(value, dict):
        if set(value) & {"auto_map", "trust_remote_code", "custom_pipelines", "_target_"}:
            raise ValueError("Executable artifact configuration is unsupported")
        for child in value.values():
            _no_artifact_code(child)
    elif isinstance(value, list):
        for child in value:
            _no_artifact_code(child)


def _features(value):
    if not isinstance(value, dict) or len(value) > 16:
        raise ValueError("Unsupported feature map")
    for key, feature in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", key):
            raise ValueError("Unsupported feature name")
        if not isinstance(feature, dict) or set(feature) != {"type", "shape"}:
            raise ValueError("Unsupported feature constructor")
        if feature["type"] not in {"VISUAL", "STATE", "ACTION", "ENV"}:
            raise ValueError("Unsupported feature type")
        shape = feature["shape"]
        if not isinstance(shape, list) or not 1 <= len(shape) <= 4:
            raise ValueError("Unsupported feature shape")
        for dimension in shape:
            if not 1 <= _integer(dimension, "feature dimension", 4096):
                raise ValueError("Empty feature dimension")
        if math.prod(shape) > 4 * 1024 * 1024:
            raise ValueError("Feature is too large")


def _tokenizer_tree(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Tokenizer artifact must be a local directory")
    files = list(root.iterdir())
    if not files or len(files) > 16:
        raise ValueError("Unsupported tokenizer artifact inventory")
    for path in files:
        if path.name not in TOKENIZER_FILES or not path.is_file() or path.is_symlink():
            raise ValueError("Unsupported tokenizer artifact file")
        _hash(path, MAX_ARRAY)
        if path.suffix == ".json":
            cfg = _read_json(path, MAX_ARRAY)
            _no_artifact_code(cfg)
            if path.name == "tokenizer_config.json" and cfg.get("tokenizer_class") not in {
                None, "BartTokenizer", "BartTokenizerFast", "RobertaTokenizer", "RobertaTokenizerFast"
            }:
                raise ValueError("Unsupported tokenizer class")


def _validate_pipeline(root, filename, *, replay=False):
    """Validate every constructor and artifact before native deserialization."""
    config = _read_json(_within(root, filename))
    if not isinstance(config, dict) or set(config) - {"name", "steps"}:
        raise ValueError("Unsupported pipeline declaration")
    steps = config.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
        raise ValueError("Unsupported processor step count")
    _no_artifact_code(config)
    for entry in steps:
        if not isinstance(entry, dict) or set(entry) - {"registry_name", "config", "state_file", "artifacts"}:
            raise ValueError("Only allowlisted native registry steps are supported; class imports are refused")
        name = entry.get("registry_name")
        cfg = entry.get("config", {})
        if name not in STEP_FIELDS or not isinstance(cfg, dict) or set(cfg) - STEP_FIELDS[name]:
            raise ValueError("Unsupported native step or constructor field")
        if "features" in cfg:
            _features(cfg["features"])
        if "norm_map" in cfg and (not isinstance(cfg["norm_map"], dict) or any(
            k not in {"VISUAL", "STATE", "ACTION", "ENV"} or v not in {
                "IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10"
            } for k, v in cfg["norm_map"].items()
        )):
            raise ValueError("Unsupported normalization map")
        if "eps" in cfg and (type(cfg["eps"]) not in {int, float} or not 0 < cfg["eps"] <= 1):
            raise ValueError("Unsupported normalization epsilon")
        if "normalize_observation_keys" in cfg and cfg["normalize_observation_keys"] is not None and (
            not isinstance(cfg["normalize_observation_keys"], list) or len(cfg["normalize_observation_keys"]) > 16 or
            any(not isinstance(key, str) or not re.fullmatch(r"observation\.[A-Za-z0-9_.]+", key)
                for key in cfg["normalize_observation_keys"])
        ):
            raise ValueError("Unsupported normalization observation key list")
        if "image_keys" in cfg and cfg["image_keys"] is not None and (
            not isinstance(cfg["image_keys"], list) or len(cfg["image_keys"]) > 8 or
            any(not isinstance(k, str) or not re.fullmatch(r"observation\.images\.[A-Za-z0-9_.]+", k)
                for k in cfg["image_keys"])
        ):
            raise ValueError("Unsupported image key list")
        if "validate_range" in cfg and type(cfg["validate_range"]) is not bool:
            raise ValueError("Unsupported range flag")
        for field in {"domain_id", "expected_action_dim"} & cfg.keys():
            _integer(cfg[field], field, 4096)
        if name == "device_processor":
            if cfg.get("device", "cpu") not in {"cpu", "cuda", "cuda:0"} or cfg.get("float_dtype") not in {
                None, "float32", "float64", "float16", "bfloat16"
            }:
                raise ValueError("Unsupported device constructor")
            if replay and cfg.get("device", "cpu") != "cpu":
                raise ValueError("Replay export must be CPU only")
        if name == "rename_observations_processor":
            mapping = cfg.get("rename_map", {})
            if not isinstance(mapping, dict) or len(mapping) > 16 or any(
                not isinstance(v, str) or not re.fullmatch(r"observation\.[A-Za-z0-9_.]+", v)
                for pair in mapping.items() for v in pair
            ):
                raise ValueError("Unsupported observation rename map")
        artifacts = entry.get("artifacts", {})
        if not isinstance(artifacts, dict) or set(artifacts) - {"tokenizer_name"}:
            raise ValueError("Unsupported native artifact reference")
        if artifacts and name != "tokenizer_processor":
            raise ValueError("Artifacts only supported for tokenizer steps")
        if "state_file" in entry:
            state = _within(root, entry["state_file"])
            if state.suffix != ".safetensors":
                raise ValueError("Only native safetensors step state is supported")
            _hash(state, MAX_ARRAY)
        if name == "tokenizer_processor":
            if not 1 <= _integer(cfg.get("max_length", 512), "token length", 4096):
                raise ValueError("Empty token sequence")
            if cfg.get("task_key", "task") != "task" or cfg.get("padding_side", "right") not in {
                "left", "right"
            } or cfg.get("padding", "max_length") not in {"max_length", "longest"}:
                raise ValueError("Unsupported tokenizer parameters")
            if type(cfg.get("truncation", True)) is not bool:
                raise ValueError("Unsupported tokenizer truncation")
            token_name = cfg.get("tokenizer_name")
            if "tokenizer_name" in artifacts:
                if token_name != artifacts["tokenizer_name"]:
                    raise ValueError("Tokenizer artifact/config disagreement")
                _tokenizer_tree(_within(root, token_name))
            elif replay or token_name != "facebook/bart-large":
                raise ValueError("Only pinned runtime BART cache or exported local tokenizer is supported")
    return config


def _validate_dataset_info(root):
    info = _read_json(_within(root, "meta/info.json"))
    _no_artifact_code(info)
    if info.get("storage_format", "parquet") not in {"parquet", None}:
        raise ValueError("Only the native parquet dataset reader is supported")
    features = info.get("features", {})
    if not isinstance(features, dict) or len(features) > 32:
        raise ValueError("Unsupported native dataset feature inventory")
    for key, feature in features.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", key) or not isinstance(feature, dict):
            raise ValueError("Unsupported dataset feature key")
        if key.startswith("observation.images.") and feature.get("dtype") != "video":
            raise ValueError("This pilot requires original video-backed image observations")
    for key, allowed in [("data_path", {"chunk_index", "file_index"}),
                         ("video_path", {"chunk_index", "file_index", "video_key"})]:
        template = info.get(key)
        if template is None and key == "video_path":
            continue
        if not isinstance(template, str):
            raise ValueError("Unsupported native dataset path template")
        for _, field, spec, conversion in string.Formatter().parse(template):
            if field is not None and (field not in allowed or spec not in {"", "03d", "06d"} or conversion):
                raise ValueError("Unsupported dataset template field")
        _relative(template.format(chunk_index=0, file_index=0, video_key="camera"))
    return info


def _validate_request(request):
    if not isinstance(request, dict) or request.get("operation") not in {"probe", "replay"}:
        raise ValueError("Operation must be probe or replay")
    allowed = {"operation", "source_root", "output", "cache_root"}
    allowed |= {"model", "dataset", "samples"} if request["operation"] == "probe" else {"capsule_root"}
    if set(request) - allowed:
        raise ValueError("Unsupported worker request fields")
    for key in {"source_root", "output"} | ({"model", "dataset"} if request["operation"] == "probe"
                                         else {"capsule_root"}):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError(f"Missing path: {key}")
    if request["operation"] == "probe":
        samples = request.get("samples")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 2:
            raise ValueError("Select one or two original samples")
        identities = []
        for sample in samples:
            if not isinstance(sample, dict) or set(sample) != {"episode_index", "frame_index"}:
                raise ValueError("Samples require episode_index and frame_index only")
            identities.append(tuple(_integer(sample[k], k) for k in ["episode_index", "frame_index"]))
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate sample selection")


def _output_path(request):
    output = Path(request["output"]).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Worker output must be a new empty private directory")
    for key in ["model", "dataset", "capsule_root", "source_root"]:
        if key in request:
            input_path = Path(request[key]).resolve(strict=True)
            if not input_path.is_dir() or output.is_relative_to(input_path) or input_path.is_relative_to(output):
                raise ValueError("Output and input roots must be separate directories")
    return output


def _error_result(error):
    return {"provider": PROVIDER, "execution_status": "error", "scope": SCOPE, "samples": [],
            "native_versions": {}, "source_files": [], "exports": [],
            "unverified": ["Native execution did not complete; no compatibility certificate."],
            "native_execution_error": {"type": type(error).__name__, "message": str(error)[:2000]}}


def run(request):
    """Run only fixed maintained APIs; all native/array imports occur here."""
    result = _error_result(ValueError("Native execution has not started"))
    owned_output = False
    try:
        _validate_request(request)
        output = _output_path(request)
        source = Path(request["source_root"]).resolve(strict=True)
        if not source.is_dir():
            raise ValueError("Trusted source root must be a directory")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        owned_output = True
        runtime = output / "runtime"
        runtime.mkdir()
        (runtime / "tmp").mkdir()
        tempfile.tempdir = str(runtime / "tmp")
        cache = Path(request["cache_root"]).resolve() if request.get("cache_root") and request["operation"] == "probe" else runtime / "hf-cache"
        os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "HF_HUB_CACHE": str(cache), "HF_DATASETS_CACHE": str(runtime / "datasets-cache"),
                           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "1"})
        sys.dont_write_bytecode = True
        import_root = source if (source / "lerobot/__init__.py").is_file() else source.parent
        if not (import_root / "lerobot/__init__.py").is_file():
            raise ValueError("Trusted source must contain the native lerobot package")
        sys.path.insert(0, str(import_root))
        # Preflight artifact constructors before any native deserializer is invoked.
        if request["operation"] == "probe":
            model, dataset = Path(request["model"]).resolve(), Path(request["dataset"]).resolve()
            model_config = _read_json(_within(model, "config.json"))
            _no_artifact_code(model_config)
            if model_config.get("type") != "xvla":
                raise ValueError("This pilot supports the native XVLA config only")
            if model_config.get("tokenizer_name", "facebook/bart-large") != "facebook/bart-large":
                raise ValueError("Unsupported config tokenizer source")
            for key in ["input_features", "output_features"]:
                _features(model_config.get(key, {}))
            _validate_pipeline(model, "policy_preprocessor.json")
            _validate_pipeline(model, "policy_postprocessor.json")
            _validate_dataset_info(dataset)
        else:
            capsule = Path(request["capsule_root"]).resolve()
            manifest = _read_json(_within(capsule, "worker_manifest.json"))
            if manifest.get("schema") != SCHEMA or manifest.get("provider") != PROVIDER:
                raise ValueError("Unsupported replay manifest")
            if not isinstance(manifest.get("samples"), list) or not 1 <= len(manifest["samples"]) <= 2:
                raise ValueError("Unsupported retained sample count")
            if not isinstance(manifest.get("exports"), list) or not 1 <= len(manifest["exports"]) <= 2:
                raise ValueError("Unsupported retained export count")
            for export in manifest["exports"]:
                if export.get("route") not in {"saved", "current_factory"}:
                    raise ValueError("Unsupported replay route")
                export_root = _within(capsule, export["path"])
                for item in export["files"]:
                    path = _within(export_root, item["path"])
                    if _hash(path, MAX_ARRAY) != item["sha256"]:
                        raise ValueError("Native export file digest mismatch")
                _validate_pipeline(export_root, export["config_filename"], replay=True)
        import lerobot
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.xvla import processor_xvla
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.processor import PolicyProcessorPipeline, ProcessorStepRegistry
        from lerobot.processor.factory import load_pretrained_policy_processors
        from transformers import AutoTokenizer

        if request["operation"] == "probe":
            AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True, trust_remote_code=False)
        torch.set_num_threads(1)
        if not Path(lerobot.__file__).resolve().is_relative_to(source):
            raise ValueError("Imported native package does not match the trusted source root")
        for name, qualified in STEP_CLASSES.items():
            cls = ProcessorStepRegistry.get(name)
            if cls.__module__ + "." + cls.__qualname__ != qualified:
                raise ValueError("Native registry differs from the supported class allowlist")
        result = {"provider": PROVIDER, "execution_status": "complete", "scope": SCOPE,
                  "samples": [], "native_versions": {}, "source_files": [], "exports": [],
                  "artifact_receipts": [], "unverified": [
                      "Expected training recipe and training-time source revision are unverified.",
                      "No original raw policy action prediction, postprocessor-output or task-success test.",
                      "Current factory differences may be legal; no preferred route is inferred.",
                      "Replay retains derived reader arrays, not an independent original-video re-decode.",
                      "Checkpoint weights, unselected dataset files and unrelated media are omitted."]}
        for package in ["lerobot", "torch", "torchvision", "datasets", "pyarrow", "av", "transformers", "tokenizers"]:
            with contextlib.suppress(importlib.metadata.PackageNotFoundError):
                result["native_versions"][package] = importlib.metadata.version(package)
        sources = {}
        receipts = {}
        total_bytes = 0

        def provenance(symbol):
            symbol = get_origin(symbol) or symbol
            path = Path(inspect.getsourcefile(symbol) or "").resolve()
            if not path.is_relative_to(source):
                raise ValueError("Native symbol source is outside the trusted package")
            rel = path.relative_to(source).as_posix()
            record = {"path": rel, "sha256": _hash(path, MAX_JSON), "size_bytes": path.stat().st_size}
            sources[rel] = record
            return {"qualified_name": symbol.__module__ + "." + symbol.__qualname__, "source": record}

        def receipt(root_name, root, relative, role):
            nonlocal total_bytes
            relative = _relative(str(relative))
            if (root_name, relative) in receipts:
                return
            path = _within(root, relative)
            size = path.stat().st_size
            if size > MAX_FILE or total_bytes + size > MAX_TOTAL:
                result["unverified"].append(f"Original file omitted by receipt byte limit: {root_name}/{relative}")
                return
            total_bytes += size
            receipts[root_name, relative] = {"root": root_name, "path": relative, "sha256": _hash(path),
                                             "size_bytes": size, "role": role}

        def number(value):
            scalar = value.item() if hasattr(value, "item") else value
            if isinstance(scalar, (bool, int)):
                return scalar
            scalar = float(scalar)
            return scalar if math.isfinite(scalar) else str(scalar)

        def summary(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous()
                if value.numel() * value.element_size() > MAX_ARRAY:
                    raise ValueError("Native tensor exceeds observation bound")
                raw = value.view(torch.uint8).numpy().tobytes() if value.dtype == torch.bfloat16 else value.numpy().tobytes()
                flat = value.reshape(-1)
                return {"dtype": str(value.numpy().dtype) if value.dtype != torch.bfloat16 else "bfloat16",
                        "shape": list(value.shape), "sha256": hashlib.sha256(raw).hexdigest(),
                        "min": number(flat.min()) if flat.numel() else None,
                        "max": number(flat.max()) if flat.numel() else None,
                        "preview": [number(x) for x in flat[:8]]}
            if isinstance(value, dict):
                if len(value) > 64:
                    raise ValueError("Native transition has too many fields")
                return {str(k): summary(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
            if isinstance(value, (list, tuple)):
                if len(value) > 64:
                    raise ValueError("Native transition sequence is too long")
                return [summary(x) for x in value]
            if value is None or isinstance(value, (bool, int, float, str)):
                if isinstance(value, str) and len(value) > 4096:
                    raise ValueError("Native text exceeds observation bound")
                return number(value) if isinstance(value, float) else value
            raise ValueError("Unsupported native output value type")

        def parameters(step):
            params = copy.deepcopy(step.get_config())
            if getattr(step.__class__, "_registry_name", None) == "tokenizer_processor":
                tokenizer = step.input_tokenizer
                params["tokenizer_name"] = {
                    "role": "native_tokenizer_content", "class": type(tokenizer).__name__,
                    "vocabulary_sha256": hashlib.sha256(_canonical(tokenizer.get_vocab())).hexdigest(),
                    "special_tokens_sha256": hashlib.sha256(_canonical(tokenizer.special_tokens_map)).hexdigest()}
            _no_artifact_code(params)
            return params

        def inventory(pipe):
            entries = []
            for step in pipe.steps:
                name = getattr(step.__class__, "_registry_name", None)
                if name not in STEP_FIELDS or step.__class__.__module__ + "." + step.__class__.__qualname__ != STEP_CLASSES[name]:
                    raise ValueError("Factory produced an unsupported native step")
                entries.append({"registry_name": name, **provenance(step.__class__), "parameters": parameters(step)})
            return entries

        def execution_error(error):
            message = str(error)
            for key in ["source_root", "model", "dataset", "capsule_root", "output", "cache_root"]:
                if isinstance(request.get(key), str):
                    message = message.replace(str(Path(request[key]).resolve()), "<" + key + ">")
            return {"type": type(error).__name__, "message": message[:2000]}

        def observe(pipe, data, label):
            trace = []
            route = {"name": label, "status": "complete", "steps": inventory(pipe), "input": summary(data)}

            def before(index, transition):
                trace.append({"position": "before", "step_index": index, "transition": summary(transition)})

            def after(index, transition):
                trace.append({"position": "after", "step_index": index, "transition": summary(transition)})

            pipe.before_step_hooks.append(before)
            pipe.after_step_hooks.append(after)
            try:
                route["output"] = summary(pipe(copy.deepcopy(data)))
            except Exception as error:
                route.update(status="error", native_execution_error=execution_error(error))
            finally:
                pipe.before_step_hooks.remove(before)
                pipe.after_step_hooks.remove(after)
            return route, trace

        for symbol in [PolicyProcessorPipeline, load_pretrained_policy_processors, PreTrainedConfig, XVLAConfig,
                       processor_xvla.make_xvla_pre_post_processors]:
            provenance(symbol)
        (output / "samples").mkdir()
        (output / "traces").mkdir()
        (output / "exports").mkdir()
        (output / "declarations").mkdir()
        sample_records = []
        routes = {}
        if request["operation"] == "probe":
            import pyarrow as pa
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            pa.set_cpu_count(1)
            pa.set_io_thread_count(1)
            provenance(LeRobotDataset)
            if set(model_config) - ({field.name for field in dataclasses.fields(XVLAConfig)} | {"type"}):
                raise ValueError("Unsupported policy config constructor field")
            episodes = sorted({x["episode_index"] for x in request["samples"]})
            dataset_object = LeRobotDataset("native-local-probe", root=dataset, episodes=episodes,
                                           video_backend="pyav", download_videos=False,
                                           return_uint8=False, image_transforms=None)
            for symbol in [type(dataset_object.meta), type(dataset_object.reader)]:
                provenance(symbol)
            for rel in ["meta/info.json", "meta/tasks.parquet", "meta/stats.json"]:
                if (dataset / rel).exists():
                    receipt("dataset", dataset, rel, "native_metadata")
            # Native metadata loads these original episode-parquet files as one table.
            episode_files = sorted((dataset / "meta/episodes").glob("*/*.parquet"))
            if len(episode_files) > 128:
                raise ValueError("Episode metadata inventory exceeds pilot bound")
            for path in episode_files:
                receipt("dataset", dataset, path.relative_to(dataset).as_posix(), "native_episode_metadata")
            for rel in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
                receipt("model", model, rel, "original_processor_configuration")
            for filename in ["policy_preprocessor.json", "policy_postprocessor.json"]:
                cfg = _read_json(model / filename)
                for entry in cfg["steps"]:
                    if "state_file" in entry:
                        receipt("model", model, entry["state_file"], "original_native_step_state")
                    for token_dir in entry.get("artifacts", {}).values():
                        for path in _within(model, token_dir).iterdir():
                            receipt("model", model, path.relative_to(model).as_posix(), "original_tokenizer_artifact")
            offsets, offset = {}, 0
            for episode in episodes:
                length = int(dataset_object.meta.episodes[episode]["length"])
                offsets[episode] = (offset, length)
                offset += length
            for index, selected in enumerate(request["samples"]):
                episode, frame = selected["episode_index"], selected["frame_index"]
                if frame >= offsets[episode][1]:
                    raise ValueError("Requested frame is outside its original native episode")
                sample = dataset_object[offsets[episode][0] + frame]
                identity = {key: int(sample[key]) for key in ["index", "episode_index", "frame_index", "task_index"]}
                identity["timestamp"] = float(sample["timestamp"])
                if identity["episode_index"] != episode or identity["frame_index"] != frame:
                    raise ValueError("Native reader returned a different original sample identity")
                task = sample.get("task")
                if not isinstance(task, str) or len(task) > 4096:
                    raise ValueError("Native reader did not supply bounded task text")
                data = {key: value for key, value in sample.items()
                        if key.startswith("observation.") and isinstance(value, torch.Tensor)}
                data["task"] = task
                associated_action = sample.get("action")
                bindings = [{"root": "dataset", "path": str(dataset_object.meta.get_data_file_path(episode))}]
                for key in dataset_object.meta.video_keys:
                    bindings.append({"root": "dataset", "path": str(dataset_object.meta.get_video_file_path(episode, key))})
                for binding in bindings:
                    binding["path"] = _relative(binding["path"])
                    receipt("dataset", dataset, binding["path"], "original_sample_data_or_media")
                array_fields, arrays = [], {}
                for item, key in enumerate(sorted(k for k in data if isinstance(data[k], torch.Tensor))):
                    array = data[key].detach().cpu().contiguous().numpy()
                    if array.dtype.hasobject or array.nbytes > MAX_ARRAY:
                        raise ValueError("Unsupported retained native input array")
                    arrays[f"a{item}"] = array
                    array_fields.append({"key": key, "array": f"a{item}"})
                if isinstance(associated_action, torch.Tensor):
                    arrays["associated_action"] = associated_action.detach().cpu().numpy()
                array_path = output / "samples" / f"sample_{index}.npz"
                np.savez(array_path, **arrays)
                metadata = {"identity": identity, "task": task, "array_fields": array_fields,
                            "source_bindings": bindings, "associated_action": summary(associated_action)}
                metadata_path = output / "samples" / f"sample_{index}.json"
                _write_json(metadata_path, metadata)
                sample_records.append({"identity": identity, "arrays": {"path": array_path.relative_to(output).as_posix(),
                    "sha256": _hash(array_path, MAX_ARRAY)}, "metadata": {"path": metadata_path.relative_to(output).as_posix(),
                    "sha256": _hash(metadata_path)}, "data": data, "metadata_value": metadata})
            config = PreTrainedConfig.from_pretrained(model, local_files_only=True, cli_overrides=["--device=cpu"])
            for name in ["saved", "current_factory"]:
                try:
                    if name == "saved":
                        pre, _ = load_pretrained_policy_processors(model, preprocessor_overrides={"device_processor": {"device": "cpu"}})
                    else:
                        from lerobot.policies.factory import make_pre_post_processors
                        provenance(make_pre_post_processors)
                        pre, _ = make_pre_post_processors(config)
                    inventory(pre)
                    routes[name] = pre
                except Exception as error:
                    routes[name] = {"name": name, "status": "error", "native_execution_error": execution_error(error)}
            exports = []
            for name, pipe in routes.items():
                if isinstance(pipe, dict):
                    continue
                declaration = _write_json(output / "declarations" / f"{name}.json", pipe.get_config())
                declaration["path"] = f"declarations/{name}.json"
                export_root = output / "exports" / name
                try:
                    pipe.save_pretrained(export_root)
                    candidates = []
                    for path in export_root.glob("*.json"):
                        value = _read_json(path)
                        if isinstance(value, dict) and isinstance(value.get("steps"), list):
                            candidates.append(path)
                    if len(candidates) != 1:
                        raise ValueError("Could not unambiguously discover native exported config filename")
                    _validate_pipeline(export_root, candidates[0].name, replay=True)
                    files = []
                    for path in sorted(export_root.rglob("*")):
                        if path.is_symlink():
                            raise ValueError("Native export contains a symlink")
                        if path.is_file():
                            if len(files) >= 64 or path.suffix not in {".json", ".txt", ".safetensors", ".model"}:
                                raise ValueError("Unsupported native export inventory")
                            files.append({"path": path.relative_to(export_root).as_posix(), "sha256": _hash(path, MAX_ARRAY),
                                          "size_bytes": path.stat().st_size})
                    exports.append({"route": name, "path": f"exports/{name}", "config_filename": candidates[0].name,
                                    "files": files, "original_declaration": declaration})
                except Exception as error:
                    result["unverified"].append(f"Native export unavailable for {name}: {type(error).__name__}: {str(error)[:500]}")
                    result["execution_status"] = "partial"
        else:
            exports = manifest["exports"]
            for expected_source in manifest.get("source_files", []):
                actual_path = _within(source, expected_source["path"])
                if _hash(actual_path, MAX_JSON) != expected_source["sha256"]:
                    raise ValueError("Replay native source hash differs from the retained source")
                sources[expected_source["path"]] = expected_source
            for export in exports:
                name = export["route"]
                root_path = _within(capsule, export["path"])
                routes[name] = PolicyProcessorPipeline.from_pretrained(root_path, config_filename=export["config_filename"],
                                                                       local_files_only=True)
            for record in manifest["samples"]:
                metadata_path = _within(capsule, record["metadata"]["path"])
                array_path = _within(capsule, record["arrays"]["path"])
                if _hash(metadata_path) != record["metadata"]["sha256"] or _hash(array_path, MAX_ARRAY) != record["arrays"]["sha256"]:
                    raise ValueError("Retained sample digest mismatch")
                metadata = _read_json(metadata_path)
                if metadata.get("identity") != record["identity"] or not isinstance(metadata.get("task"), str):
                    raise ValueError("Retained sample metadata identity mismatch")
                fields = metadata.get("array_fields")
                if not isinstance(fields, list) or not 1 <= len(fields) <= 16:
                    raise ValueError("Unsupported retained native input field count")
                with zipfile.ZipFile(array_path) as archive:
                    if sum(info.file_size for info in archive.infolist()) > MAX_ARRAY or len(archive.infolist()) > 17:
                        raise ValueError("Retained array archive exceeds decode bounds")
                data = {"task": metadata["task"]}
                with np.load(array_path, allow_pickle=False) as archive:
                    for item in fields:
                        key, array_key = item["key"], item["array"]
                        if not isinstance(key, str) or not re.fullmatch(r"observation\.[A-Za-z0-9_.]+", key):
                            raise ValueError("Unsupported retained native input key")
                        if not isinstance(array_key, str) or not re.fullmatch(r"a[0-9]+", array_key):
                            raise ValueError("Unsupported retained array member")
                        array = archive[array_key]
                        if array.dtype.kind not in "buif" or array.nbytes > MAX_ARRAY or array.ndim > 4:
                            raise ValueError("Unsupported retained numeric array")
                        data[key] = torch.from_numpy(array.copy())
                sample_records.append({**record, "data": data, "metadata_value": metadata})
        result["exports"] = exports
        for index, record in enumerate(sample_records):
            sample_result = {"identity": record["identity"], "task": record["metadata_value"]["task"],
                             "source_bindings": record["metadata_value"]["source_bindings"],
                             "associated_action": record["metadata_value"]["associated_action"], "routes": []}
            for name, pipe in routes.items():
                if isinstance(pipe, dict):
                    sample_result["routes"].append(pipe)
                    result["execution_status"] = "partial"
                    continue
                route, trace = observe(pipe, record["data"], name)
                trace_path = output / "traces" / f"sample_{index}_{name}.json"
                trace_receipt = _write_json(trace_path, trace)
                trace_receipt["path"] = trace_path.relative_to(output).as_posix()
                route["trace"] = trace_receipt
                sample_result["routes"].append(route)
                if route["status"] != "complete":
                    result["execution_status"] = "partial"
            result["samples"].append(sample_result)
        result["source_files"] = sorted(sources.values(), key=lambda item: item["path"])
        result["artifact_receipts"] = sorted(receipts.values(), key=lambda item: (item["root"], item["path"]))
        if request["operation"] == "probe":
            retained = [{k: v for k, v in record.items() if k not in {"data", "metadata_value"}} for record in sample_records]
            _write_json(output / "worker_manifest.json", {"schema": SCHEMA, "provider": PROVIDER,
                "samples": retained, "exports": exports, "source_files": result["source_files"],
                "expected_samples": result["samples"]})
        else:
            result["replay_matches"] = _canonical(result["samples"]) == _canonical(manifest.get("expected_samples"))
            if not result["replay_matches"]:
                result["execution_status"] = "partial"
                result["unverified"].append("Retained input/export numerical projection did not match the original probe.")
        if len(_canonical(result)) > MAX_RESPONSE:
            raise ValueError("Native worker response exceeds 128KiB; full bounded artifacts remain in staging")
        return result
    except Exception as error:
        result["execution_status"] = "error"
        result["native_execution_error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        if owned_output:
            with contextlib.suppress(OSError, ValueError):
                output = Path(request["output"])
                if output.is_dir() and not (output / "native_error.txt").exists():
                    with (output / "native_error.txt").open("x") as stream:
                        stream.write(traceback.format_exc()[-16000:])
        return result if len(_canonical(result)) <= MAX_RESPONSE else _error_result(error)


def main():
    """Bounded JSON stdin; native chatter goes to stderr, stdout is a tiny receipt."""
    request = None
    try:
        raw = sys.stdin.buffer.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("Worker request exceeds 64KiB")
        request = json.loads(raw)
        _validate_request(request)
        output = _output_path(request)
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt = _write_json(output / "result.json", result)
        print(json.dumps({"provider": PROVIDER, "execution_status": result["execution_status"],
                          "result": receipt}, separators=(",", ":")))
        return 0 if result["execution_status"] != "error" else 1
    except Exception as error:
        print(json.dumps({"provider": PROVIDER, "execution_status": "error", "message": str(error)[:500]},
                         separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
