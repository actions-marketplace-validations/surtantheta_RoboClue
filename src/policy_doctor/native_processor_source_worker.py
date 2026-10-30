"""Fixed opt-in SmolVLA source-observation worker. No native imports at module load.

The trusted runtime supplies executable code; checkpoint artifacts are bounded data.
This worker observes input preprocessing, not policies, predictions or task competence.
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
import shutil
import string
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import get_origin

PROVIDER = "lerobot_processor_source"
TOKENIZER_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
MAX_JSON = 2 * 1024 * 1024
MAX_ARRAY = 32 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024
MAX_RESPONSE = 128 * 1024
MAX_TOTAL = 256 * 1024 * 1024
STEP_FIELDS = {
    "rename_observations_processor": {"rename_map"}, "to_batch_processor": set(),
    "smolvla_new_line_processor": set(),
    "tokenizer_processor": {"max_length", "task_key", "padding_side", "padding", "truncation", "tokenizer_name"},
    "device_processor": {"device", "float_dtype"},
    "normalizer_processor": {"eps", "features", "norm_map", "normalize_observation_keys"},
    "unnormalizer_processor": {"eps", "features", "norm_map", "normalize_observation_keys"},
}
STEP_CLASSES = {
    "rename_observations_processor": "lerobot.processor.rename_processor.RenameObservationsProcessorStep",
    "to_batch_processor": "lerobot.processor.batch_processor.AddBatchDimensionProcessorStep",
    "smolvla_new_line_processor": "lerobot.processor.newline_task_processor.NewLineTaskProcessorStep",
    "tokenizer_processor": "lerobot.processor.tokenizer_processor.TokenizerProcessorStep",
    "device_processor": "lerobot.processor.device_processor.DeviceProcessorStep",
    "normalizer_processor": "lerobot.processor.normalize_processor.NormalizerProcessorStep",
    "unnormalizer_processor": "lerobot.processor.normalize_processor.UnnormalizerProcessorStep",
}
TOKENIZER_FILES = {"README.md", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "merges.txt", "vocab.json", "chat_template.json", "chat_template.jinja",
                   "processor_config.json", "preprocessor_config.json"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value or ":" in value:
        raise ValueError("Unsafe relative artifact path")
    if PurePosixPath(value).is_absolute() or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("Escaping or ambiguous artifact path")
    return value


def _within(root, relative):
    root = Path(root)
    current = root
    for part in Path(_relative(relative)).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlinked artifact path refused")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Escaping artifact path")
    return current


def _hash(path, maximum=MAX_FILE):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("Missing, symlinked or oversized file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path, maximum=MAX_JSON):
    _hash(path, maximum)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _no_code(value, *, vocabulary=False):
    if vocabulary:
        if not isinstance(value, dict) or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in value.items()):
            raise ValueError("Unsupported literal tokenizer vocabulary")
        return
    if isinstance(value, dict):
        if set(value) & {"auto_map", "_target_", "class_name", "module", "custom_pipelines", "trust_remote_code"}:
            raise ValueError("Executable artifact declaration refused")
        for child in value.values():
            _no_code(child)
    elif isinstance(value, list):
        for child in value:
            _no_code(child)


def _features(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 16:
        raise ValueError("Unsupported feature map")
    for key, feature in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", key):
            raise ValueError("Unsupported feature name")
        if not isinstance(feature, dict) or set(feature) != {"type", "shape"} or feature["type"] not in {"VISUAL", "STATE", "ACTION", "ENV"}:
            raise ValueError("Unsupported feature constructor")
        shape = feature["shape"]
        if not isinstance(shape, list) or not 1 <= len(shape) <= 4 or any(type(x) is not int or not 1 <= x <= 4096 for x in shape) or math.prod(shape) > 4 * 1024 * 1024:
            raise ValueError("Unsupported feature shape")


def _tokenizer_tree(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Local tokenizer directory required")
    files = list(root.iterdir())
    if not files or len(files) > len(TOKENIZER_FILES):
        raise ValueError("Unsupported tokenizer inventory")
    for path in files:
        if path.name not in TOKENIZER_FILES:
            raise ValueError("Unsupported tokenizer artifact file")
        _hash(path, 4 * 1024 * 1024)
        if path.suffix == ".jinja" and path.stat().st_size > 65536:
            raise ValueError("Oversized native tokenizer template")
        if path.suffix == ".json":
            value = _read_json(path, 4 * 1024 * 1024)
            if path.name == "vocab.json":
                _no_code(value, vocabulary=True)
            elif path.name == "tokenizer.json" and isinstance(value, dict) and isinstance(value.get("model"), dict) and "vocab" in value["model"]:
                _no_code(value["model"]["vocab"], vocabulary=True)
                safe = copy.deepcopy(value)
                safe["model"]["vocab"] = {}
                _no_code(safe)
            else:
                _no_code(value)
            if path.name == "tokenizer_config.json" and value.get("tokenizer_class") not in {None, "GPT2Tokenizer", "GPT2TokenizerFast"}:
                raise ValueError("Unsupported built-in tokenizer class")


def _validate_pipeline(root, filename, *, replay=False):
    """Pure stdlib prefilter; accepts approved remote ID or safe declared local assets."""
    root = Path(root)
    cfg = _read_json(_within(root, filename))
    _no_code(cfg)
    if not isinstance(cfg, dict) or set(cfg) - {"name", "steps"} or not isinstance(cfg.get("steps"), list) or not 1 <= len(cfg["steps"]) <= 16:
        raise ValueError("Unsupported native pipeline declaration")
    for entry in cfg["steps"]:
        if not isinstance(entry, dict) or set(entry) - {"registry_name", "config", "state_file", "artifacts"}:
            raise ValueError("Unsupported native constructor declaration")
        name, c = entry.get("registry_name"), entry.get("config", {})
        if name not in STEP_FIELDS or not isinstance(c, dict) or set(c) - STEP_FIELDS[name]:
            raise ValueError("Unsupported built-in step or constructor field")
        if "features" in c:
            _features(c["features"])
        if "norm_map" in c and (not isinstance(c["norm_map"], dict) or any(k not in {"VISUAL", "STATE", "ACTION", "ENV"} or v not in {"IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10"} for k, v in c["norm_map"].items())):
            raise ValueError("Unsupported normalization map")
        if "eps" in c and (type(c["eps"]) not in {int, float} or not 0 < c["eps"] <= 1):
            raise ValueError("Unsupported normalization epsilon")
        if "normalize_observation_keys" in c and c["normalize_observation_keys"] is not None and (not isinstance(c["normalize_observation_keys"], list) or len(c["normalize_observation_keys"]) > 16 or any(not isinstance(k, str) or not re.fullmatch(r"observation\.[A-Za-z0-9_.]+", k) for k in c["normalize_observation_keys"])):
            raise ValueError("Unsupported normalization keys")
        if name == "rename_observations_processor":
            mapping = c.get("rename_map", {})
            if not isinstance(mapping, dict) or len(mapping) > 16 or any(not isinstance(x, str) or not re.fullmatch(r"observation\.[A-Za-z0-9_.]+", x) for pair in mapping.items() for x in pair):
                raise ValueError("Unsupported rename mapping")
        if name == "device_processor" and (c.get("device", "cpu") not in {"cpu", "cuda", "cuda:0"} or c.get("float_dtype") not in {None, "float32", "float64", "float16", "bfloat16"}):
            raise ValueError("Unsupported device constructor")
        if "state_file" in entry:
            sf = entry["state_file"]
            if not isinstance(sf, str) or Path(sf).name != sf or not sf.endswith(".safetensors"):
                raise ValueError("Unsupported state filename")
            _hash(_within(root, sf), MAX_ARRAY)
        artifacts = entry.get("artifacts", {})
        if not isinstance(artifacts, dict) or set(artifacts) - {"tokenizer_name"}:
            raise ValueError("Unsupported step artifacts")
        if artifacts and name != "tokenizer_processor":
            raise ValueError("Artifacts only supported for native tokenizer")
        if name == "tokenizer_processor":
            for field in {"max_length"} & c.keys():
                if type(c[field]) is not int or not 1 <= c[field] <= 2048:
                    raise ValueError("Unsupported tokenizer length")
            if c.get("padding_side", "right") not in {"left", "right"} or c.get("padding", "max_length") not in {"max_length", "longest", True, False} or type(c.get("truncation", True)) is not bool:
                raise ValueError("Unsupported tokenizer options")
            if not isinstance(c.get("task_key", "task"), str) or len(c.get("task_key", "task")) > 128:
                raise ValueError("Unsupported task key")
            if artifacts:
                relative = artifacts["tokenizer_name"]
                if c.get("tokenizer_name") != relative:
                    raise ValueError("Local tokenizer declaration must match its artifact")
                _tokenizer_tree(_within(root, relative))
            elif c.get("tokenizer_name") != TOKENIZER_ID:
                raise ValueError("Unapproved tokenizer source")
    return cfg


def validate_model(root):
    """Pure stdlib validation; training-origin strings are data and are never followed."""
    root = Path(root)
    cfg = _read_json(_within(root, "config.json"))
    _no_code(cfg)
    if not isinstance(cfg, dict) or cfg.get("type") != "smolvla" or cfg.get("vlm_model_name", TOKENIZER_ID) != TOKENIZER_ID:
        raise ValueError("Only the supported built-in SmolVLA/tokenizer family is allowed")
    for key in ["input_features", "output_features"]:
        _features(cfg.get(key))
    origin = cfg.get("pretrained_path")
    if origin is not None and (not isinstance(origin, str) or not origin or len(origin) > 512 or "\x00" in origin):
        raise ValueError("Unsupported literal training-origin field")
    if "tokenizer_max_length" in cfg and (type(cfg["tokenizer_max_length"]) is not int or not 1 <= cfg["tokenizer_max_length"] <= 2048):
        raise ValueError("Unsupported configured tokenizer length")
    for name in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        _validate_pipeline(root, name)
    return cfg


def _validate_request(request):
    if not isinstance(request, dict) or request.get("operation") not in {"probe", "replay"}:
        raise ValueError("Operation must be probe or replay")
    allowed = {"operation", "source_root", "model", "samples", "processor_source", "intent", "approved_sources", "output", "cache_root", "selected_identity"}
    allowed |= {"dataset"} if request["operation"] == "probe" else {"raw_samples"}
    if set(request) - allowed:
        raise ValueError("Unsupported worker request fields")
    for key in ["source_root", "model", "output"] + (["dataset"] if request["operation"] == "probe" else []):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError("Missing worker path")
    samples = request.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("Source auditor currently requires exactly one sample")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"episode_index", "frame_index"} or any(type(x) is not int or not 0 <= x <= 10000000 for x in sample.values()):
            raise ValueError("Unsupported native sample selector")
    if request.get("processor_source") not in {"selected_checkpoint", "serialized_pretrained_path", "fresh_training"}:
        raise ValueError("Unsupported processor source mode")
    if request.get("intent") is not None and (not isinstance(request["intent"], str) or len(request["intent"]) > 256):
        raise ValueError("Intent must be a bounded declared string or null")
    identity_claim = request.get("selected_identity")
    if identity_claim is not None and (not isinstance(identity_claim, str) or not identity_claim.isprintable() or len(identity_claim) > 512):
        raise ValueError("Selected identity is a printable label of at most 512 characters or null")
    approved = request.get("approved_sources", [])
    if not isinstance(approved, list) or len(approved) > 8:
        raise ValueError("Unsupported approved source list")
    seen = set()
    for entry in approved:
        if not isinstance(entry, dict) or set(entry) != {"repo_id", "revision", "model"} or not isinstance(entry["repo_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", entry["repo_id"]) or any(x in {".", ".."} for x in entry["repo_id"].split("/")) or not isinstance(entry["revision"], str) or not re.fullmatch(r"[a-fA-F0-9]{40}", entry["revision"]) or not isinstance(entry["model"], str) or not entry["model"] or entry["repo_id"] in seen:
            raise ValueError("Approved source must be an exact unique Hub ID/pinned revision/staged root")
        seen.add(entry["repo_id"])
    if request["operation"] == "replay" and (not isinstance(request.get("raw_samples"), list) or len(request["raw_samples"]) != 1 or any(not isinstance(x, dict) or set(x) != {"arrays", "metadata"} or any(not isinstance(v, str) or not Path(v).is_absolute() for v in x.values()) for x in request["raw_samples"])):
        raise ValueError("Replay requires exactly one retained private array/metadata pair")


def _resolve_consumer_source(request, config):
    mode = request["processor_source"]
    if mode != "serialized_pretrained_path":
        return {"mode": mode}
    origin = config.get("pretrained_path")
    match = [x for x in request.get("approved_sources", []) if x["repo_id"] == origin]
    if not isinstance(origin, str) or len(match) != 1:
        raise ValueError("Serialized source is not an explicitly approved pinned Hub ID")
    return {"mode": mode, **match[0]}


def _route_plan(mode):
    if mode == "selected_checkpoint":
        return {"consumer": "native-call-0001", "selected_reference": "native-call-0001"}
    return {"consumer": "native-call-0001", "selected_reference": "native-call-0002"}


def _route_alias(row, role):
    """Role label is metadata; shared call_id/trace remains the actual call witness."""
    return {**copy.deepcopy(row), "name": role}


def _model_files(root):
    """Positive original metadata/state/tokenizer closure; never discover weights."""
    root = Path(root)
    validate_model(root)
    paths = {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    for filename in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        for step in _read_json(root / filename)["steps"]:
            if "state_file" in step:
                paths.add(step["state_file"])
            for relative in step.get("artifacts", {}).values():
                for path in _within(root, relative).iterdir():
                    paths.add(path.relative_to(root).as_posix())
    for name in ["README.md", "LICENSE", "LICENSE.txt", "NOTICE"]:
        if (root / name).is_file():
            paths.add(name)
    return sorted(paths)


def _output_path(request):
    output = Path(request["output"])
    if output.is_symlink():
        raise ValueError("Symlinked output refused")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("New empty private output required")
    inputs = [request[k] for k in ["source_root", "model", "dataset", "cache_root"] if request.get(k)]
    inputs += [x["model"] for x in request.get("approved_sources", [])]
    inputs += [v for x in request.get("raw_samples", []) for v in x.values()]
    for value in inputs:
        path = Path(value).resolve(strict=True)
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Output must not overlap source/input roots")
    return output


def _error(error):
    return {"type": type(error).__name__, "message": str(error)[:2000]}


def _finish_receipt(output, value):
    raw = _canonical(value)
    if len(raw) > MAX_RESPONSE:
        value = {"provider": PROVIDER, "execution_status": "error", "native_execution_error": {"type": "ValueError", "message": "Completion exceeds response bound"},
                 "result": value.get("result"), "manifest": value.get("manifest"), "payload_files": []}
        raw = _canonical(value)
    if output is not None:
        (output / "worker_receipt.json").write_bytes(raw)
    return value


def run(request):
    """Return a tiny completion descriptor; full bounded raw evidence is on disk."""
    output = None
    result = {"provider": PROVIDER, "execution_status": "error", "scope": "Actual native SmolVLA input preprocessing only; no policy/model/inference/task-success claim", "samples": []}
    try:
        _validate_request(request)
        output = _output_path(request)
        model = Path(request["model"]).resolve(strict=True)
        source = Path(request["source_root"]).resolve(strict=True)
        config_data = validate_model(model)
        consumer = _resolve_consumer_source(request, config_data)
        approved_models = {}
        for entry in request.get("approved_sources", []):
            if consumer["mode"] != "serialized_pretrained_path" or entry["repo_id"] != consumer["repo_id"]:
                continue
            root = Path(entry["model"]).resolve(strict=True)
            validate_model(root)
            approved_models[entry["repo_id"]] = (entry, root)
        if request["operation"] == "probe":
            dataset = Path(request["dataset"]).resolve(strict=True)
            info = _read_json(_within(dataset, "meta/info.json"))
            allowed = {"chunk_index", "file_index", "video_key"}
            for template in [info.get("data_path"), info.get("video_path")]:
                if template is None:
                    continue
                for _, field, spec, conversion in string.Formatter().parse(template):
                    if field is not None and (field not in allowed or spec not in {"", "03d", "06d"} or conversion):
                        raise ValueError("Unsupported native dataset template")
                _relative(template.format(chunk_index=0, file_index=0, video_key="camera"))
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        runtime = output / "runtime"
        runtime.mkdir()
        (runtime / "tmp").mkdir()
        tempfile.tempdir = str(runtime / "tmp")
        cache = runtime / "hf-cache"
        cache.mkdir()
        # Materialize only explicit approved source closures in a new offline cache.
        for repo_id, (entry, root) in approved_models.items():
            snapshot = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / entry["revision"]
            for relative in _model_files(root):
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_within(root, relative), target)
        remote_tokenizer_needed = request["processor_source"] == "fresh_training"
        for root in [model] + [x[1] for x in approved_models.values()]:
            for step in _read_json(root / "policy_preprocessor.json")["steps"]:
                if step["registry_name"] == "tokenizer_processor" and not step.get("artifacts"):
                    remote_tokenizer_needed = True
        if remote_tokenizer_needed:
            if not request.get("cache_root"):
                raise ValueError("An operator-approved pinned tokenizer cache is required; no hidden fallback")
            original_cache = Path(request["cache_root"]).resolve(strict=True)
            token_relative = "models--" + TOKENIZER_ID.replace("/", "--") + "/snapshots/" + TOKENIZER_REVISION
            token_source = _within(original_cache, token_relative)
            token_target = cache / token_relative
            # Runtime cache is operator-trusted; normal Hub blob symlinks may resolve
            # inside that explicit cache, while artifact-local symlinks remain refused.
            for name in sorted(TOKENIZER_FILES):
                path = token_source / name
                if not path.exists():
                    continue
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(original_cache) or not resolved.is_file() or resolved.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("Approved tokenizer cache file escapes its root or bound")
                token_target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resolved, token_target / name)
            _tokenizer_tree(token_target)
            refs = token_target.parent.parent / "refs"
            refs.mkdir(exist_ok=True)
            (refs / "main").write_text(TOKENIZER_REVISION)
        os.environ.update({"HF_HUB_CACHE": str(cache), "HF_HOME": str(runtime / "hf-home"), "HF_DATASETS_CACHE": str(runtime / "datasets-cache"),
                           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "1"})
        sys.dont_write_bytecode = True
        root_for_import = source if (source / "lerobot/__init__.py").is_file() else source.parent
        if not (root_for_import / "lerobot/__init__.py").is_file():
            raise ValueError("Approved native package source required")
        sys.path.insert(0, str(root_for_import))
        # Observer uses audit/profile hooks, preserving native function implementations.
        observer = {"active": True, "busy": False, "loaded": {}}
        roots = {"selected": model, "hf_cache": cache}
        if request["operation"] == "probe":
            roots["dataset"] = dataset

        def record_loaded(path, witness):
            if not observer["active"] or observer["busy"] or not isinstance(path, (str, bytes, os.PathLike)):
                return
            p = Path(os.fsdecode(path)).absolute()
            for label, root in roots.items():
                if p.is_relative_to(root) and p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".parquet", ".mp4", ".jinja", ".md"}:
                    observer["busy"] = True
                    try:
                        relative = p.relative_to(root).as_posix()
                        row = observer["loaded"].setdefault((label, relative), {"root": label, "path": relative, "sha256": _hash(p), "size_bytes": p.stat().st_size, "witnesses": []})
                        if witness not in row["witnesses"]:
                            row["witnesses"].append(witness)
                    finally:
                        observer["busy"] = False
                    break

        def audit(event, args):
            if not observer["active"]:
                return
            if event == "socket.connect":
                raise PermissionError("Offline native worker refuses network access")
            if event == "open" and args:
                mode = args[1] if len(args) > 1 else None
                if mode is None or isinstance(mode, str) and not any(x in mode for x in "wa+"):
                    record_loaded(args[0], "Python audit open read")

        def profile(frame, event, arg):
            if not observer["active"] or observer["busy"]:
                return
            filename = frame.f_code.co_filename.replace("\\", "/")
            if event == "call" and frame.f_code.co_name == "load_file" and filename.endswith("/safetensors/torch.py"):
                record_loaded(frame.f_locals.get("filename"), "actual safetensors.torch.load_file argument")
            elif event == "call" and frame.f_code.co_name == "_from_pretrained" and "transformers/" in filename:
                resolved = frame.f_locals.get("resolved_vocab_files", {})
                if isinstance(resolved, dict):
                    for path in resolved.values():
                        if isinstance(path, str):
                            record_loaded(path, "actual tokenizer _from_pretrained resolved file")

        sys.addaudithook(audit)
        previous_profile = sys.getprofile()
        sys.setprofile(profile)
        try:
            import lerobot
            import numpy as np
            import pyarrow as pa
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla import processor_smolvla
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.processor import PolicyProcessorPipeline, ProcessorStepRegistry
            torch.set_num_threads(1)
            pa.set_cpu_count(1)
            pa.set_io_thread_count(1)
            if not Path(lerobot.__file__).resolve().is_relative_to(source):
                raise ValueError("Imported native package differs from approved source root")
            for name, expected in STEP_CLASSES.items():
                cls = ProcessorStepRegistry.get(name)
                if cls.__module__ + "." + cls.__qualname__ != expected:
                    raise ValueError("Native registry class mismatch")
            if set(config_data) - ({f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}):
                raise ValueError("Unknown policy config constructor fields")
            cfg = PreTrainedConfig.from_pretrained(model, local_files_only=True, cli_overrides=["--device=cpu"])
            if consumer["mode"] == "serialized_pretrained_path":
                if cfg.pretrained_path is None or str(cfg.pretrained_path) != consumer["repo_id"]:
                    raise ValueError("Parsed native origin differs from the explicitly approved serialized Hub ID")
                if Path(consumer["repo_id"]).exists():
                    raise ValueError("Approved Hub ID is shadowed by a local working-directory path")
            config_loaded = copy.deepcopy(list(observer["loaded"].values()))
            sources = {}

            def provenance(symbol):
                symbol = get_origin(symbol) or symbol
                path = Path(inspect.getsourcefile(symbol)).resolve()
                if not path.is_relative_to(source):
                    raise ValueError("Native symbol outside approved package")
                relative = path.relative_to(source).as_posix()
                sources[relative] = {"path": relative, "sha256": _hash(path, MAX_JSON), "size_bytes": path.stat().st_size}
                return {"qualified_name": symbol.__module__ + "." + symbol.__qualname__, "source": sources[relative]}

            def summary(value):
                if isinstance(value, torch.Tensor):
                    t = value.detach().cpu().contiguous()
                    if t.numel() * t.element_size() > MAX_ARRAY:
                        raise ValueError("Native tensor exceeds input bound")
                    flat = t.reshape(-1)
                    raw = t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
                    return {"dtype": "bfloat16" if t.dtype == torch.bfloat16 else str(t.numpy().dtype), "shape": list(t.shape), "sha256": hashlib.sha256(raw).hexdigest(), "min": flat.min().item() if flat.numel() else None,
                            "max": flat.max().item() if flat.numel() else None, "preview": flat[:8].tolist()}
                if isinstance(value, np.ndarray):
                    return summary(torch.from_numpy(value.copy()))
                if isinstance(value, dict):
                    if len(value) > 128:
                        raise ValueError("Native mapping bound")
                    return {str(k): summary(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
                if isinstance(value, (list, tuple)):
                    if len(value) > 128:
                        raise ValueError("Native sequence bound")
                    return [summary(v) for v in value]
                if value is None or isinstance(value, (str, bool, int, float)):
                    if isinstance(value, str) and len(value) > 8192:
                        raise ValueError("Native text bound")
                    return value
                raise ValueError("Unsupported native trace value")

            def inventory(pipe):
                rows = []
                for step in pipe.steps:
                    name = getattr(type(step), "_registry_name", None)
                    if name not in STEP_CLASSES:
                        raise ValueError("Unexpected actual native step")
                    parameters = copy.deepcopy(step.get_config())
                    if name == "tokenizer_processor":
                        tokenizer = step.input_tokenizer
                        parameters["tokenizer_name"] = {"role": "native_tokenizer_content", "class": type(tokenizer).__name__, "vocabulary_sha256": hashlib.sha256(_canonical(tokenizer.get_vocab())).hexdigest(),
                                                        "special_tokens_sha256": hashlib.sha256(_canonical(tokenizer.special_tokens_map)).hexdigest()}
                    row = {"registry_name": name, **provenance(type(step)), "parameters": summary(parameters)}
                    if name in {"normalizer_processor", "unnormalizer_processor"}:
                        row.update(effective_tensor_stats=summary(step._tensor_stats), native_state_dict=summary(step.state_dict()), stats_explicitly_provided=bool(step._stats_explicitly_provided))
                    rows.append(row)
                return rows

            for symbol in [PreTrainedConfig, SmolVLAConfig, PolicyProcessorPipeline, make_pre_post_processors, processor_smolvla.make_smolvla_pre_post_processors]:
                provenance(symbol)
            sample_records = []
            dataset_receipts = {}
            for index, selected in enumerate(request["samples"]):
                if request["operation"] == "probe":
                    from lerobot.datasets.lerobot_dataset import LeRobotDataset
                    provenance(LeRobotDataset)
                    episode, frame = selected["episode_index"], selected["frame_index"]
                    ds = LeRobotDataset("native-source-observation", root=dataset, episodes=[episode], video_backend="pyav", download_videos=False, return_uint8=False, image_transforms=None)
                    if frame >= int(ds.meta.episodes[episode]["length"]):
                        raise ValueError("Frame outside original native episode")
                    raw = ds[frame]
                    identity = {k: int(raw[k]) for k in ["episode_index", "frame_index", "index", "task_index"]}
                    identity["timestamp"] = float(raw["timestamp"])
                    data = {k: v for k, v in raw.items() if k.startswith("observation.") and isinstance(v, torch.Tensor)}
                    data["task"] = raw["task"]
                    meta = {"identity": identity, "task": raw["task"], "fields": [], "source_bindings": [str(ds.meta.get_data_file_path(episode))] + [str(ds.meta.get_video_file_path(episode, key)) for key in ds.meta.video_keys], "associated_original_action": summary(raw.get("action"))}
                    for relative in meta["source_bindings"] + ["meta/info.json", "meta/stats.json", "meta/tasks.parquet"]:
                        path = _within(dataset, relative)
                        if path.is_file():
                            dataset_receipts[relative] = {"root": "dataset", "path": relative, "sha256": _hash(path), "size_bytes": path.stat().st_size,
                                                          "role": "actual native metadata-selected original source; source receipt only, not retained video re-decode"}
                    arrays = {}
                    for n, key in enumerate(sorted(k for k in data if isinstance(data[k], torch.Tensor))):
                        arrays[f"a{n}"] = data[key].numpy()
                        meta["fields"].append({"key": key, "array": f"a{n}"})
                    if isinstance(raw.get("action"), torch.Tensor):
                        arrays["associated_action"] = raw["action"].numpy()
                else:
                    retained = request["raw_samples"][index]
                    meta = _read_json(Path(retained["metadata"]))
                    array_path = Path(retained["arrays"])
                    _hash(array_path, MAX_ARRAY)
                    with zipfile.ZipFile(array_path) as archive:
                        if len(archive.infolist()) > 17 or sum(x.file_size for x in archive.infolist()) > MAX_ARRAY:
                            raise ValueError("Retained archive exceeds decode bound")
                    identity = meta["identity"]
                    data = {"task": meta["task"]}
                    arrays = {}
                    fields = meta.get("fields", meta.get("array_fields"))
                    if not isinstance(fields, list) or not 1 <= len(fields) <= 16 or not isinstance(meta["task"], str) or len(meta["task"]) > 4096:
                        raise ValueError("Unsupported retained input metadata")
                    with np.load(array_path, allow_pickle=False) as archive:
                        for item in fields:
                            key, member = item["key"], item["array"]
                            if not re.fullmatch(r"observation\.[A-Za-z0-9_.]+", key) or not re.fullmatch(r"a[0-9]+", member):
                                raise ValueError("Unsupported retained array binding")
                            a = archive[member]
                            if a.dtype.kind not in "buif" or a.nbytes > MAX_ARRAY or a.ndim > 4:
                                raise ValueError("Unsupported retained numerical array")
                            arrays[member] = a.copy()
                            data[key] = torch.from_numpy(a.copy())
                        if "associated_action" in archive:
                            a = archive["associated_action"]
                            if a.dtype.kind not in "buif" or a.nbytes > MAX_ARRAY:
                                raise ValueError("Unsupported retained action association")
                            arrays["associated_action"] = a.copy()
                    meta["fields"] = fields
                if any(identity[k] != selected[k] for k in ["episode_index", "frame_index"]):
                    raise ValueError("Actual original identity differs from requested selector")
                array_target = output / "samples" / f"sample_{index}.npz"
                array_target.parent.mkdir(exist_ok=True)
                np.savez(array_target, **arrays)
                metadata_target = output / "samples" / f"sample_{index}.json"
                metadata_target.write_bytes(_canonical(meta))
                sample_records.append({"identity": identity, "data": data, "input": summary(data), "arrays": {"path": array_target.relative_to(output).as_posix(), "sha256": _hash(array_target), "size_bytes": array_target.stat().st_size},
                                       "metadata": {"path": metadata_target.relative_to(output).as_posix(), "sha256": _hash(metadata_target), "size_bytes": metadata_target.stat().st_size}})
            plan = _route_plan(request["processor_source"])
            pipelines, constructions, exports = {}, [], []
            loaded_all = list(config_loaded)
            for role, call_id in plan.items():
                if call_id in pipelines:
                    continue
                observer["loaded"] = {}
                selected_call = role == "selected_reference" or request["processor_source"] == "selected_checkpoint"
                factory_kwargs = {"preprocessor_overrides": {"device_processor": {"device": "cpu"}}}
                # Native configs without a device step cannot accept an unused override.
                chosen_root = model if selected_call else approved_models[consumer["repo_id"]][1] if consumer["mode"] == "serialized_pretrained_path" else None
                if chosen_root is not None and not any(x["registry_name"] == "device_processor" for x in _read_json(chosen_root / "policy_preprocessor.json")["steps"]):
                    factory_kwargs = {}
                if chosen_root is not None and any(x["registry_name"] == "device_processor" for x in _read_json(chosen_root / "policy_postprocessor.json")["steps"]):
                    factory_kwargs["postprocessor_overrides"] = {"device_processor": {"device": "cpu"}}
                if selected_call:
                    factory_source, revision = model, None
                    source_description = {"kind": "selected_checkpoint", "root": "selected", "pretrained_path": "selected staged checkpoint", "pretrained_revision": None}
                elif consumer["mode"] == "serialized_pretrained_path":
                    factory_source, revision = consumer["repo_id"], consumer["revision"]
                    source_description = {"kind": "serialized_pretrained_path", "pretrained_path": factory_source, "pretrained_revision": revision}
                else:
                    factory_source, revision, factory_kwargs = None, None, {}
                    source_description = {"kind": "fresh_training", "pretrained_path": None, "pretrained_revision": None}
                source_description["overrides"] = copy.deepcopy(factory_kwargs)
                try:
                    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=factory_source, pretrained_revision=revision, **factory_kwargs)
                    loaded = copy.deepcopy(list(observer["loaded"].values()))
                    loaded_all += loaded
                    pipelines[call_id] = {"pre": pre, "loaded_files": loaded, "post_inventory": inventory(post), "factory_arguments": source_description}
                    constructions.append({"call_id": call_id, "status": "complete", "factory_arguments": source_description, "loaded_files": loaded})
                    exportroot = output / "exports" / call_id
                    pre.save_pretrained(exportroot)
                    configs = [p for p in exportroot.glob("*.json") if isinstance(_read_json(p).get("steps"), list)]
                    if len(configs) != 1:
                        raise ValueError("Ambiguous native export config")
                    _validate_pipeline(exportroot, configs[0].name, replay=True)
                    paths = {configs[0].name}
                    for step in _read_json(configs[0])["steps"]:
                        if "state_file" in step:
                            paths.add(step["state_file"])
                        for relative in step.get("artifacts", {}).values():
                            paths |= {p.relative_to(exportroot).as_posix() for p in _within(exportroot, relative).iterdir()}
                    files = [{"path": rel, "sha256": _hash(_within(exportroot, rel)), "size_bytes": _within(exportroot, rel).stat().st_size} for rel in sorted(paths)]
                    exports.append({"route": role, "call_id": call_id, "path": exportroot.relative_to(output).as_posix(), "config_filename": configs[0].name, "files": files})
                except Exception as error:
                    if call_id not in pipelines:
                        constructions.append({"call_id": call_id, "status": "error", "factory_arguments": source_description, "native_execution_error": _error(error)})
                        pipelines[call_id] = {"error": _error(error), "loaded_files": copy.deepcopy(list(observer["loaded"].values())), "factory_arguments": source_description}
                    else:
                        pipelines[call_id]["export_error"] = _error(error)
                        constructions[-1]["export_error"] = _error(error)
            samples = []
            for index, sample in enumerate(sample_records):
                actual_calls = {}
                for _role, call_id in plan.items():
                    if call_id in actual_calls:
                        continue
                    item = pipelines[call_id]
                    row = {"call_id": call_id, "status": "complete", "input": sample["input"], "factory_arguments": item["factory_arguments"], "loaded_files": item["loaded_files"]}
                    if "error" in item:
                        row.update(status="error", execution_stage="construction", native_execution_error=item["error"])
                        actual_calls[call_id] = row
                        continue
                    pre = item["pre"]
                    trace = []
                    row.update(steps=inventory(pre), postprocessor_inventory_no_execution=item["post_inventory"])
                    def before(i, transition, trace=trace):
                        trace.append({"position": "before", "step_index": i, "transition": summary(transition)})
                    def after(i, transition, trace=trace):
                        trace.append({"position": "after", "step_index": i, "transition": summary(transition)})
                    pre.before_step_hooks.append(before)
                    pre.after_step_hooks.append(after)
                    try:
                        row["output"] = summary(pre(copy.deepcopy(sample["data"])))
                    except Exception as error:
                        row.update(status="error", native_execution_error=_error(error))
                    finally:
                        pre.before_step_hooks.remove(before)
                        pre.after_step_hooks.remove(after)
                    row["steps_after"] = inventory(pre)
                    trace_path = output / "traces" / f"sample_{index}_{call_id}.json"
                    trace_path.parent.mkdir(exist_ok=True)
                    trace_path.write_bytes(_canonical(trace))
                    row["trace"] = {"path": trace_path.relative_to(output).as_posix(), "sha256": _hash(trace_path), "size_bytes": trace_path.stat().st_size}
                    if "export_error" in item:
                        row["export_error"] = item["export_error"]
                    actual_calls[call_id] = row
                samples.append({"identity": sample["identity"], "input": sample["input"], "routes": {role: _route_alias(actual_calls[call_id], role) for role, call_id in plan.items()}, "arrays": sample["arrays"], "metadata": sample["metadata"]})
            # Copy exact approved original payloads, not runtime/cache trees.
            binding_candidates = {(x["root"], x["path"]): x for x in loaded_all if x["root"] in {"selected", "hf_cache"}}
            for relative in _model_files(model):
                path = _within(model, relative)
                binding_candidates.setdefault(("selected", relative), {"root": "selected", "path": relative, "sha256": _hash(path), "size_bytes": path.stat().st_size, "witnesses": ["approved selected original metadata/state/tokenizer dependency closure"]})
            for repo_id, (entry, root) in approved_models.items():
                prefix = "models--" + repo_id.replace("/", "--") + "/snapshots/" + entry["revision"] + "/"
                for relative in _model_files(root):
                    cached = _within(cache, prefix + relative)
                    binding_candidates.setdefault(("hf_cache", prefix + relative), {"root": "hf_cache", "path": prefix + relative, "sha256": _hash(cached), "size_bytes": cached.stat().st_size,
                                                                                     "witnesses": ["approved alternate original dependency closure; distinct from actual load witness"]})
            if remote_tokenizer_needed:
                for path in token_target.iterdir():
                    relative = path.relative_to(cache).as_posix()
                    binding_candidates.setdefault(("hf_cache", relative), {"root": "hf_cache", "path": relative, "sha256": _hash(path), "size_bytes": path.stat().st_size,
                                                                          "witnesses": ["approved fixed tokenizer dependency closure; distinct from actual load witness"]})
            bindings = []
            for (label, relative), entry in sorted(binding_candidates.items()):
                original = _within(roots[label], relative)
                if label == "hf_cache":
                    permitted = ["models--" + TOKENIZER_ID.replace("/", "--") + "/snapshots/" + TOKENIZER_REVISION + "/"]
                    permitted += ["models--" + repo.replace("/", "--") + "/snapshots/" + spec[0]["revision"] + "/" for repo, spec in approved_models.items()]
                    if not any(relative.startswith(prefix) for prefix in permitted):
                        raise ValueError("Actual loaded cache source is not explicitly approved")
                retained = "originals/" + label + "/" + relative
                target = output / retained
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                if _hash(target) != entry["sha256"]:
                    raise ValueError("Original binding changed during retention")
                bindings.append({**entry, "retained_path": retained})
            all_complete = all(row["status"] == "complete" and "export_error" not in row for sample in samples for row in sample["routes"].values())
            result = {"provider": PROVIDER, "operation": request["operation"], "execution_status": "complete" if all_complete else "partial",
                      "scope": "Actual native SmolVLA input preprocessing and caller-declared source lineage; no policy/model/inference/competence", "samples": samples,
                      "identity": samples[0]["identity"], "input": samples[0]["input"], "routes": samples[0]["routes"], "construction": constructions,
                      "config_lineage": {"intent": request.get("intent"), "processor_source": request["processor_source"], "selected_identity_claim": request.get("selected_identity"),
                                         "serialized_training_source": config_data.get("pretrained_path"), "parsed_pretrained_path": str(cfg.pretrained_path) if cfg.pretrained_path is not None else None,
                                         "selected_bundle_sha256": hashlib.sha256(_canonical([{k: v for k, v in x.items() if k in {"path", "sha256", "size_bytes"}} for x in bindings if x["root"] == "selected"])).hexdigest(),
                                         "native_semantics": "cfg.pretrained_path may name fine-tuned-FROM origin; factory honors the explicit source argument; no automatic repair or loader-defect claim"},
                      "bindings": bindings, "exports": exports, "source_files": list(sources.values()), "original_config_loaded_files": config_loaded,
                      "original_dataset_receipts": list(dataset_receipts.values()),
                      "native_versions": {k: importlib.metadata.version(k) for k in ["torch", "torchvision", "transformers", "tokenizers", "datasets", "pyarrow", "av", "safetensors", "draccus"]},
                      "unverified": ["Caller intent/selected_identity are claims, not authenticated history", "Training dataset revision and exact training-row membership unknown", "Replay repeats factory processing on retained arrays and original model/stat/tokenizer bytes; no video re-decode", "Native source/runtime is a separately qualified external dependency"]}
            manifest = {"schema": "native_processor_source.v1", "provider": PROVIDER, "operation": request["operation"], "result_path": "result.json",
                        "samples": [{k: x[k] for k in ["identity", "arrays", "metadata"]} for x in samples], "exports": exports, "bindings": bindings}
        finally:
            observer["active"] = False
            sys.setprofile(previous_profile)
        raw = _canonical(result)
        if len(raw) > MAX_JSON:
            raise ValueError("Native raw report exceeds two MiB bound")
        (output / "result.json").write_bytes(raw)
        payload_paths = {"result.json"}
        for sample in manifest["samples"]:
            payload_paths |= {sample["arrays"]["path"], sample["metadata"]["path"]}
        for sample in result["samples"]:
            payload_paths |= {x["trace"]["path"] for x in sample["routes"].values() if "trace" in x}
        for export in manifest["exports"]:
            payload_paths |= {export["path"] + "/" + x["path"] for x in export["files"]}
        payload_paths |= {x["retained_path"] for x in bindings}
        payload_files = [{"path": rel, "sha256": _hash(_within(output, rel)), "size_bytes": _within(output, rel).stat().st_size} for rel in sorted(payload_paths)]
        if sum(x["size_bytes"] for x in payload_files) > MAX_TOTAL:
            raise ValueError("Native retained payload exceeds total bound")
        manifest["payload_files"] = payload_files
        (output / "worker_manifest.json").write_bytes(_canonical(manifest))
        result_descriptor = {"path": "result.json", "sha256": _hash(output / "result.json"), "size_bytes": len(raw)}
        return _finish_receipt(output, {"provider": PROVIDER, "operation": request["operation"], "execution_status": result["execution_status"], "result": result_descriptor,
                "manifest": {"path": "worker_manifest.json", "sha256": _hash(output / "worker_manifest.json"), "size_bytes": (output / "worker_manifest.json").stat().st_size}, "payload_files": payload_files})
    except Exception as error:
        result.update(execution_status="error", native_execution_error=_error(error))
        descriptor = None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            raw = _canonical(result)
            if len(raw) > MAX_JSON:
                raw = _canonical({"provider": PROVIDER, "execution_status": "error", "native_execution_error": _error(error)})
            (output / "result.json").write_bytes(raw)
            descriptor = {"path": "result.json", "sha256": _hash(output / "result.json"), "size_bytes": len(raw)}
        return _finish_receipt(output, {"provider": PROVIDER, "operation": request.get("operation") if isinstance(request, dict) else None, "execution_status": "error", "native_execution_error": _error(error), "result": descriptor, "payload_files": [descriptor] if descriptor else []})


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            raise ValueError("Worker request exceeds bound")
        request = json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            result = run(request)
    except Exception as error:
        result = {"provider": PROVIDER, "execution_status": "error", "native_execution_error": _error(error), "payload_files": []}
    encoded = _canonical(result)
    if len(encoded) > MAX_RESPONSE:
        encoded = _canonical({"provider": PROVIDER, "execution_status": "error", "native_execution_error": {"type": "ValueError", "message": "Completion exceeds response bound"}, "result": result.get("result"), "manifest": result.get("manifest"), "payload_files": []})
    sys.stdout.buffer.write(encoded + b"\n")


if __name__ == "__main__":
    main()
