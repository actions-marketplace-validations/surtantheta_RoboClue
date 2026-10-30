"""Will Isaac-GR00T read this dataset the way its modality.json says? A static GR00T dataset contract check.

Isaac-GR00T 944fe70 reads a LeRobot v2 dataset through meta/modality.json: state and action groups are [start:end]
slices of a vector column, camera entries map training-config keys to video features, and annotations name a column.
Every rule cites the pinned source line that makes it true, and each finding names the path it applies to:

- fine-tuning: launch_finetune -> DatasetFactory.build (gr00t/data/dataset/factory.py:57-69) first runs generate_stats
  and generate_rel_stats (gr00t/data/stats.py:251-291, 415-435), which write or refresh meta/stats.json and
  meta/relative_stats.json, and only then opens the dataset with LeRobotEpisodeLoader;
- direct loaders: LeRobotEpisodeLoader opened without that step, as gr00t/eval/open_loop_eval.py:327 and
  gr00t/policy/replay_policy.py:94 do.

Two mistakes pass through GR00T on both paths without an error: a slice past the vector's width is cut short in the
data and in the statistics alike, and a camera entry that reads another camera's video (paired by position when the
names differ from the config, or by name when the original_keys are swapped). Both were reproduced with GR00T's own
code (.validation/k4-groot/data/native_probe.json, .validation/k4-groot/validator/out/native_results*.json).

The files are read, never executed: a modality config .py file is parsed with ast, and GR00T is not imported.
Parquet contents, video frames, the model's embodiment metadata and training success are not checked.

Isaac-GR00T N1.5 reads datasets differently; check_groot_dataset(..., groot_version="n1.5") applies its own rules
(groot_contract_n15). Every report names the profile it ran in "groot_version" ("n1.7", "n1.5", or None when the
version could not be determined).

check_groot_dataset_versions(dataset, modality_config=None, *, config_name=None, groot_version=None) decides which
profile(s) apply and combines them per dataset:
- groot_version="n1.5" / "n1.7": that profile only;
- a config and no version: the profile its form names (groot_contract_n15.detect_version for a .py file; a bare
  config_name must be a built-in N1.5 data config); when the form is ambiguous nothing is run (status unknown),
  never both profiles against one config;
- no config and no version: BOTH profiles, combined (STATUS_RULE): status "error" when any version has an error,
  else the worst of warning > unknown > pass. With no config, N1.5 crash errors hold whatever the data config reads
  (N1.5 opens every modality.json entry); N1.7 entry errors hold only if the training config reads the cited entry.
It returns one dict that preflight can carry as its groot_contract component. The findings are already bounded (do
not re-bound them); a renderer must show headline, remedy and each version's verdict, and any further reduction must
keep each finding's groot_version:
  {"schema_version": 3, "kind": "groot_dataset_contract", "mode": "explicit" | "detected" | "both" | "undetermined",
   "status", "headline" (one sentence naming each failing version, how it fails and on which path, e.g. "Fails on
   Isaac-GR00T N1.5 (crash at open); no error on Isaac-GR00T N1.7."), "status_rule", "remedy" (the exact
   --groot-version command), "groot_version" (the single version run, else None), "versions_run", "fails_on",
   "dataset", "consumer", "modality_config_name", "counts" (summed over versions),
   "versions": {version: {"consumer", "status", "counts", "fails", "failure" ("crash at open" | "crash" |
   "silently wrong" | "crash at open; also silently wrong" ...), "applies_to" (paths of its errors), "error_codes",
   "errors_certain", "errors_if_read", "statement", "command", "modality_config_name", "paths", "not_checked",
   "refused" (InputError text, only when that profile refused the input)}},
   "files", "findings" (each finding of every version, with "groot_version" and "certainty"; at most
   COMBINED_MAX_FINDINGS / COMBINED_MAX_BYTES, but always at least one per (version, code) for every error and
   unknown code), "total_findings", "findings_truncated",
   "coverage": {"complete", "modality_config_supplied", "modality_config_name", "versions_run", "not_checked"},
   "scope"}
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from .artifacts import MAX_STATS_BYTES, InputError, read_json, sha256

GROOT = "Isaac-GR00T 944fe70"
LOADER = "gr00t/data/dataset/lerobot_episode_loader.py"
STATS = "gr00t/data/stats.py"
PROCESSOR = "gr00t/data/state_action/state_action_processor.py"
FACTORY = "gr00t/data/dataset/factory.py"
DIRECT_USERS = "gr00t/eval/open_loop_eval.py:327, gr00t/policy/replay_policy.py:94"
FT, DIRECT, BOTH = "fine-tuning", "direct loaders", "fine-tuning and direct loaders"
DEFAULT_COLUMNS = {"state": "observation.state", "action": "action"}  # loader:60-63
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")  # stats.py:224
FINGERPRINTS = "__fingerprints__"  # stats.py:62
LANG_KEYS = ("task", "sub_task")  # loader:65
MAX_LINES = 1_000_000
MAX_CONFIG_BYTES = 1024 * 1024
MAX_RANGE = 100_000
_UNREAD = "<not statically readable>"


def stats_fingerprint(feature_name: str, feature_meta: dict) -> str:
    """GR00T's cache key for one info.json feature's statistics (gr00t/data/stats.py:183-198, reimplemented)."""
    payload = {"feature": feature_name, "dtype": feature_meta.get("dtype"), "shape": feature_meta.get("shape")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stale_reason(stats: dict, features: dict, column: str) -> str | None:
    """Why fine-tuning would recompute stats.json[column], or None when it keeps it (stats.py:201-229)."""
    feature = features.get(column)
    if feature is None:
        return "the column is not in meta/info.json"
    entry = stats.get(column)
    if not isinstance(entry, dict):
        return "it has no entry"
    missing = [name for name in STAT_NAMES if name not in entry]
    if missing:
        return f"its entry lacks {missing}"
    fingerprints = stats.get(FINGERPRINTS)
    fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
    if column not in fingerprints:
        return f"it has no {FINGERPRINTS} entry"
    if fingerprints[column] != stats_fingerprint(column, feature):
        return f"its {FINGERPRINTS} entry does not match the column's dtype and shape in meta/info.json"
    return None


def _contained(root: Path, relative: str) -> Path:
    """Resolve a dataset file; refuse a symlink that leads outside the dataset.

    One layout is accepted: a Hugging Face cache snapshot (<repo cache>/snapshots/<revision>/...) whose files are
    symlinks into the same repo cache's blobs/ directory, which is how a dataset downloaded without --local-dir looks.
    """
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise InputError(f"Dataset references must be paths within the dataset: {relative}")
    path = (root / relative).resolve()
    if path.is_relative_to(root):
        return path
    # The dataset may sit at the snapshot root or in a subfolder of it; blobs/ must be a real directory of the same
    # repo cache (a symlinked blobs/ could point anywhere).
    for parent in (root, *root.parents):
        if parent.parent.name == "snapshots":
            cache = parent.parent.parent
            blobs = cache / "blobs"
            if not blobs.is_symlink() and blobs.is_dir() and path.parent == blobs.resolve() \
                    and blobs.resolve().is_relative_to(cache.resolve()):
                return path
            break
    raise InputError(f"{relative} escapes the dataset directory {root} through a symlink")


class _Report:
    def __init__(self, root: Path):
        self.root, self.findings, self.files = root, [], {}

    def file(self, name: str) -> Path:
        path = _contained(self.root, name)
        if path.is_file() and name not in self.files:
            self.files[name] = sha256(path)
        return path

    def add(self, code: str, status: str, message: str, *evidence: tuple, remedy: str = "",
            applies_to: str = BOTH) -> None:
        self.findings.append({"code": code, "status": status, "message": message, "remedy": remedy,
                              "applies_to": applies_to,
                              "evidence": [{"path": name, "sha256": self.files.get(name), "pointer": pointer,
                                            "value": value} for name, pointer, value in evidence]})


def _read_cache(path: Path) -> tuple[str, object]:
    """Read stats.json or relative_stats.json the way GR00T's json.load does: NaN allowed, last duplicate wins."""
    if not path.is_file():
        return "missing", None
    if path.stat().st_size == 0:  # fine-tuning treats it as no cache (stats.py:80); json.load in the loader fails
        return "unreadable", "the file is empty"
    if path.stat().st_size > MAX_STATS_BYTES:
        return "too large", f"{path.stat().st_size} bytes, over the {MAX_STATS_BYTES}-byte bound"
    try:
        return "ok", json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:  # json.JSONDecodeError is a ValueError
        return "unreadable", str(exc)[:200]


def _scan_jsonl(path: Path, check) -> dict:
    """Parse every line as GR00T does (json.loads per line, loader:165, 170); collect problems by kind."""
    problems, lines, truncated = {}, 0, False
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if number > MAX_LINES:
                    truncated = True
                    break
                lines = number
                if not line.strip():
                    kinds = ["blank line"]
                else:
                    try:
                        row = json.loads(line)
                    except (ValueError, RecursionError):
                        kinds = ["not JSON"]
                    else:
                        kinds = check(row) if isinstance(row, dict) else ["not a JSON object"]
                for kind in kinds:
                    problems.setdefault(kind, []).append(number)
    except UnicodeDecodeError:
        problems.setdefault("not UTF-8 text", []).append(lines + 1)
    return {"lines": lines, "problems": problems, "truncated": truncated}


def _names(feature: dict, width: int | None) -> list | None:
    names = feature.get("names")
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    ok = isinstance(names, list) and all(isinstance(n, str) for n in names) and len(names) == width
    return names if ok else None  # GR00T never reads names; they only label findings


def _width(feature: dict) -> int | None:
    shape = feature.get("shape")
    return shape[0] if isinstance(shape, list) and len(shape) == 1 and type(shape[0]) is int else None


def _size(feature: dict) -> int | None:
    shape = feature.get("shape")
    if not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape):
        return None
    size = 1
    for n in shape:
        size *= n
    return size


# ---------------------------------------------------------------- modality config (.py, parsed with ast only)

def _call_name(node: ast.AST) -> str | None:
    func = node.func if isinstance(node, ast.Call) else None
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _constants(tree: ast.Module) -> dict:
    """Module-level NAME = "str" / ["str", ...], so keys like CAMERAS can be read statically. A name bound anywhere
    else in the file, or used through an attribute (CAMERAS.append(...)), is not treated as a constant."""
    stores, touched = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stores[node.id] = stores.get(node.id, 0) + 1
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            touched.add(node.value.id)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            values[node.target.id] = node.value
    return {name: value for name, value in values.items() if stores.get(name) == 1 and name not in touched
            and (_string(value, {}) is not None or _strings(value, {}) is not None)}


def _resolve(node, constants: dict):
    return constants.get(node.id, node) if isinstance(node, ast.Name) else node


def _string(node, constants: dict) -> str | None:
    node = _resolve(node, constants)
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _strings(node, constants: dict) -> list[str] | None:
    node = _resolve(node, constants)
    if isinstance(node, (ast.List, ast.Tuple)) and all(_string(e, constants) is not None for e in node.elts):
        return [_string(e, constants) for e in node.elts]
    return None


def _int(node) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _int(node.operand)
        return None if value is None else -value
    return None


def _ints(node) -> list[int] | None:
    """delta_indices as GR00T builds them: a literal list of ints, or list(range(...)) with literal bounds."""
    if isinstance(node, ast.List):
        values = [_int(e) for e in node.elts]
        return None if any(v is None for v in values) else values
    if isinstance(node, ast.Call) and _call_name(node) == "list" and len(node.args) == 1 and not node.keywords:
        inner = node.args[0]
        if isinstance(inner, ast.Call) and _call_name(inner) == "range" and 1 <= len(inner.args) <= 3 \
                and not inner.keywords:
            bounds = [_int(a) for a in inner.args]
            if any(b is None for b in bounds) or len(bounds) == 3 and bounds[2] == 0:
                return None
            values = range(*bounds)
            return list(values) if len(values) <= MAX_RANGE else None
    return None


def _action_config(item, constants: dict) -> dict:
    """One ActionConfig(...) call or dict literal (gr00t/data/types.py:77-81, 116-124) -> rep/type/format/state_key."""
    if isinstance(item, ast.Call) and _call_name(item) == "ActionConfig" \
            and all(k.arg is not None for k in item.keywords):
        fields = dict(zip(("rep", "type", "format", "state_key"), item.args, strict=False))
        fields.update({k.arg: k.value for k in item.keywords})
        enum = {name: fields[name].attr if isinstance(fields.get(name), ast.Attribute) else None
                for name in ("rep", "type", "format")}
    elif isinstance(item, ast.Dict) and all(_string(k, constants) is not None for k in item.keys):
        fields = {_string(k, constants): v for k, v in zip(item.keys, item.values, strict=True)}
        enum = {name: _string(fields.get(name), constants) for name in ("rep", "type", "format")}
    else:
        return {"rep": None, "type": None, "format": None, "state_key": _UNREAD}
    state = fields.get("state_key")
    if state is None or isinstance(state, ast.Constant) and state.value is None:
        state_key = None
    else:
        state_key = _string(state, constants)
        state_key = _UNREAD if state_key is None else state_key
    return {**enum, "state_key": state_key}


def _modality(name: str, value: ast.Call, constants: dict) -> dict:
    if any(k.arg is None for k in value.keywords):
        raise InputError(f"{name}: ModalityConfig(**...) cannot be read statically")
    fields = dict(zip(("delta_indices", "modality_keys", "sin_cos_embedding_keys", "mean_std_embedding_keys",
                       "action_configs"), value.args, strict=False))
    fields.update({k.arg: k.value for k in value.keywords})
    keys = _strings(fields.get("modality_keys"), constants)
    if keys is None:
        raise InputError(f"{name}: modality_keys must be a literal list of strings to be read statically")
    entry = {"keys": keys, "delta_indices": _ints(_resolve(fields.get("delta_indices"), constants))}
    actions = _resolve(fields.get("action_configs"), constants)
    if actions is None or isinstance(actions, ast.Constant) and actions.value is None:
        return entry
    if not isinstance(actions, ast.List):
        entry["action_configs"] = _UNREAD
        return entry
    items = [_action_config(_resolve(item, constants), constants) for item in actions.elts]
    entry["action_configs"] = items
    entry["relative"] = {key: (item["state_key"] or key) if item["state_key"] != _UNREAD else _UNREAD
                         for key, item in zip(keys, items, strict=False) if item["rep"] == "RELATIVE"}
    return entry


def _config_name(node: ast.Dict, parents: dict, constants: dict) -> str | None:
    child, parent = node, parents.get(node)
    while isinstance(parent, (ast.Call, ast.keyword, ast.Starred)):  # dict(**{...})
        child, parent = parent, parents.get(parent)
    if isinstance(parent, ast.Dict):
        for key, value in zip(parent.keys, parent.values, strict=True):
            if value is child:
                return _string(key, constants)
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
        return parent.targets[0].id
    if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
        return parent.target.id
    return None


def _parse_config(path: str | Path, config_name: str | None) -> tuple[str | None, dict]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
        raise InputError(f"Expected a modality config .py file no larger than {MAX_CONFIG_BYTES} bytes: {path}")
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))  # honours a coding cookie, else UTF-8
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise InputError(f"Cannot parse modality config {path}: {exc}") from exc
    constants = _constants(tree)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.Dict) and node.keys
                  and all(k is not None and _string(k, constants) is not None for k in node.keys)
                  and any(_call_name(v) == "ModalityConfig" for v in node.values)]
    named = [(_config_name(node, parents, constants), node) for node in candidates]
    if config_name is not None:
        chosen = [node for name, node in named if name == config_name]
    elif len(named) == 1:
        chosen = [named[0][1]]
    else:  # several dicts: the one passed to register_modality_config(...) is what GR00T trains with
        registered = set()
        for node in ast.walk(tree):
            if _call_name(node) == "register_modality_config":
                given = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "config"), None)
                if isinstance(given, ast.Name):
                    registered.add(given.id)
        chosen = [node for name, node in named if name in registered]
    if len(chosen) != 1:
        names = sorted(str(n) for n, _ in named)
        if not named:
            raise InputError(f"Found no dict of ModalityConfig(...) values in {path}.")
        if config_name is not None:
            raise InputError(f"{path} has {len(chosen)} modality config dicts named {config_name!r}; its config "
                             f"dicts are {names}.")
        raise InputError(f"{path} has {len(named)} modality config dicts {names}, and not exactly one is passed to "
                         "register_modality_config(...). Pass config_name to choose one (a variable name, or a "
                         "key of MODALITY_CONFIGS such as libero_sim).")
    node = chosen[0]
    config = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if _call_name(value) == "ModalityConfig":
            config[_string(key, constants)] = _modality(_string(key, constants), value, constants)
    return _config_name(node, parents, constants), config


def read_modality_config(path: str | Path, config_name: str | None = None) -> dict:
    """Parse a GR00T modality config .py file without executing it.

    Returns {modality: {"keys": [...], "delta_indices": [...] or None, ...}}; the action modality also carries
    "action_configs" (a list, or a marker that it is not a literal list) and "relative" {action_key: state_key}.
    With several config dicts in one file (gr00t/configs/data/embodiment_configs.py), config_name selects one by
    its variable name or MODALITY_CONFIGS key; otherwise the one passed to register_modality_config(...) is used.
    """
    return _parse_config(path, config_name)[1]


# ---------------------------------------------------------------- dataset rules

def _info(report: _Report, info: dict) -> dict:
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2"):
        report.add("groot.format", "error",
                   f"meta/info.json declares codebase_version {version or 'none'}; {GROOT} reads LeRobot v2 metadata "
                   f"(meta/episodes.jsonl, meta/tasks.jsonl; {LOADER}:162-171).",
                   ("meta/info.json", "/codebase_version", version),
                   remedy="Convert with scripts/lerobot_conversion/convert_v3_to_v2.py from Isaac-GR00T.")
    missing = [key for key in ("data_path", "chunks_size") if key not in info]
    if missing:
        report.add("groot.metadata", "error", f"meta/info.json has no {missing}; GR00T reads both when it opens the "
                   f"dataset and raises KeyError ({LOADER}:198, 201).", ("meta/info.json", "", sorted(info)),
                   remedy="Restore the LeRobot v2 info.json fields.")
    features = info.get("features")
    if not isinstance(features, dict):
        report.add("groot.metadata", "error", f"meta/info.json has no features object; fine-tuning reads "
                   f"info.json['features'] before anything else ({STATS}:254-256).",
                   ("meta/info.json", "/features", type(features).__name__), applies_to=FT,
                   remedy="Restore the LeRobot v2 features map.")
        return {}
    no_dtype = [k for k, v in features.items() if not isinstance(v, dict) or v.get("dtype") is None]
    if no_dtype:
        report.add("groot.metadata", "error", f"meta/info.json features {no_dtype[:5]} have no dtype; fine-tuning "
                   f"reads every feature's dtype to pick the columns it computes statistics for and fails "
                   f"({STATS}:256).", ("meta/info.json", "/features", no_dtype[:5]), applies_to=FT,
                   remedy="Give every feature a dtype.")
    return {k: v for k, v in features.items() if isinstance(v, dict)}


def _config_keys(report: _Report, modality: dict, config: dict) -> None:
    for side in ("state", "action"):
        if side not in config:
            report.add("groot.config", "error",
                       f"The config has no {side} modality; fine-tuning reads statistics for every state and action "
                       f"key and raises KeyError ({LOADER}:514-515; relative statistics read config['action'] at "
                       f"{STATS}:417).", ("meta/modality.json", "", None), applies_to=FT,
                       remedy=f"Add a {side} ModalityConfig to the config.")
            continue
        entries = modality.get(side) if isinstance(modality.get(side), dict) else {}
        missing = [k for k in config[side]["keys"] if k not in entries]
        if missing:
            report.add(f"groot.{side}_config", "error",
                       f"Config {side} keys {missing} are not modality.json groups; fine-tuning raises KeyError when "
                       f"it reads their statistics ({LOADER}:517), and the sample loader only prints a warning and "
                       f"drops them ({LOADER}:339-341).",
                       ("meta/modality.json", f"/{side}", sorted(entries)),
                       remedy=f"First select or review a training config matching the dataset's existing {side} groups. "
                              "Do not edit a public dataset merely to silence this check. Only when a reviewed owned "
                              f"recipe requires it, add source-backed {side} groups to a separate candidate or correct the config keys.")


def _groups(report: _Report, modality: dict, features: dict, side: str, config: dict | None) -> dict:
    """Check the state/action slices GR00T reads; return {group: (column, start, end)} for valid used ones."""
    entries = modality.get(side)
    used = None if config is None else (config.get(side) or {}).get("keys", [])
    if not isinstance(entries, dict) or not entries:
        if used is None or used:
            report.add(f"groot.{side}_groups", "error", f"modality.json has no {side} groups; GR00T reads {side} "
                       f"only through them ({LOADER}:325-341, 514-530).", ("meta/modality.json", f"/{side}", entries),
                       remedy=f"Declare each {side} group with start/end indices into its column.")
        return {}
    valid, covered, by_column = {}, {}, {}
    for group, entry in entries.items():
        pointer = f"/{side}/{group}"
        evidence = [("meta/modality.json", pointer, entry)]
        fault = None  # (code, status, problem, consequence, remedy, applies_to)
        column = entry.get("original_key", DEFAULT_COLUMNS[side]) if isinstance(entry, dict) else None
        feature = features.get(column) if isinstance(column, str) else None
        width = _width(feature) if feature is not None else None
        start, end = (entry.get("start"), entry.get("end")) if isinstance(entry, dict) else (None, None)
        shared = None  # a fault of the column itself, reported once per column
        if not isinstance(entry, dict):
            fault = ("slice", "error", f"{side}.{group} is not an object",
                     f"GR00T reads start, end and original_key from it ({LOADER}:330-333).", "", BOTH)
        elif not isinstance(column, str):
            fault = ("column", "error", f"{side}.{group} has original_key {column!r}, which is not a column name",
                     f"GR00T indexes the parquet table and the statistics with it ({LOADER}:333-336, 517-527).",
                     "Set original_key to a column name.", BOTH)
        elif feature is None:
            shared = "which meta/info.json does not declare"
            fault = ("column", "error", f"{side}.{group} reads column {column}, {shared}", "", "", FT)
        elif "float" not in str(feature.get("dtype")):
            shared = f"which meta/info.json declares with dtype {feature.get('dtype')!r}"
            fault = ("column", "error", f"{side}.{group} reads column {column}, {shared}", "", "", FT)
        elif type(start) is not int or type(end) is not int or start < 0 or end <= start:
            fault = ("slice", "error", f"{side}.{group} has start={start!r}, end={end!r}",
                     f"GR00T slices x[start:end] ({LOADER}:336), so this group would be empty or wrong.",
                     "Use integers with 0 <= start < end.", BOTH)
        elif width is None:
            fault = ("slice", "unknown", f"{column} has no one-dimensional shape in meta/info.json",
                     f"so the {side}.{group} slice cannot be checked.", "", BOTH)
            evidence.append(("meta/info.json", f"/features/{column}", feature))
        elif end > width:
            names = _names(feature, width)
            kept = max(0, min(end, width) - start)
            fault = ("slice", "error", f"{side}.{group} slices [{start}:{end}] of {column}, which has {width} values",
                     f"GR00T does not check the range: the data ({LOADER}:336) and the normalization statistics "
                     f"({LOADER}:527-530) are both cut to {kept} value(s), with no error or warning. A modality.json "
                     "copied from another robot does this.",
                     f"Rewrite the {side} slices for this robot's {width}-value {column}"
                     + (f" ({', '.join(names)})." if names else "."), BOTH)
            evidence.append(("meta/info.json", f"/features/{column}/shape", feature.get("shape")))
        if used is not None and group not in used:
            if fault is not None and fault[1] in ("error", "warning", "unknown"):
                report.add(f"groot.{side}_{fault[0]}", "pass",
                           f"{side}.{group} is not one of the config's {side} keys, so GR00T never reads it "
                           f"({LOADER}:328-342, 514-530); noted only: {fault[2]}.", *evidence)
            continue
        if shared is not None:
            by_column.setdefault((column, shared), []).append(group)
            continue
        if fault is not None:
            report.add(f"groot.{side}_{fault[0]}", fault[1], f"{fault[2]}. {fault[3]}", *evidence, remedy=fault[4],
                       applies_to=fault[5])
            continue
        valid[group] = (column, start, end)
        for index in range(start, end):
            covered.setdefault((column, index), []).append(group)
        names = _names(feature, width)
        if names:
            report.add(f"groot.{side}_slice", "pass", f"{side}.{group} reads {column}[{start}:{end}] = {names[start:end]}.",
                       *evidence)
    for (column, problem), groups in by_column.items():
        report.add(f"groot.{side}_column", "error",
                   f"{side} groups {groups} read column {column}, {problem}. Fine-tuning computes statistics only for "
                   f"info.json features whose dtype contains 'float' ({STATS}:256) and drops any other entry from "
                   f"meta/stats.json ({STATS}:274-276), so it raises KeyError when it reads this column's statistics "
                   f"({LOADER}:527).", ("meta/modality.json", f"/{side}", {g: entries[g] for g in groups}),
                   ("meta/info.json", f"/features/{column}", features.get(column)), applies_to=FT,
                   remedy=f"Declare {column} in meta/info.json as a float vector (for example dtype float32) with "
                          "its real shape.")
    for (column, index), owners in sorted(covered.items()):
        if len(owners) > 1:
            report.add(f"groot.{side}_overlap", "warning",
                       f"{column}[{index}] belongs to {owners}; GR00T feeds it to the model in each group.",
                       ("meta/modality.json", f"/{side}", entries), remedy="Make groups disjoint unless the duplication is intended.")
    for column in sorted({c for c, _, _ in valid.values()}):
        width = _width(features[column])
        names = _names(features[column], width)
        unused = [i for i in range(width) if (column, i) not in covered]
        if unused:
            label = [names[i] for i in unused] if names else unused
            report.add(f"groot.{side}_unused", "warning",
                       f"{column} values {label} are in no {side} group"
                       + (" the config reads" if used is not None else "") + ", so GR00T never reads them.",
                       ("meta/modality.json", f"/{side}", entries), ("meta/info.json", f"/features/{column}", features[column]),
                       remedy="Add them to a group, or confirm they are meant to be ignored.")
    # A gripper group that reads no gripper value, or an arm group that reads one, points at the wrong joints.
    for group, (column, start, end) in valid.items():
        names = _names(features[column], _width(features[column]))
        if not names:
            continue
        span = [n.lower() for n in names[start:end]]
        if "gripper" in group.lower() and not any("gripper" in n for n in span) and any("gripper" in n.lower() for n in names):
            report.add(f"groot.{side}_names", "warning",
                       f"{side}.{group} reads {names[start:end]}, none of which is a gripper value.",
                       ("meta/modality.json", f"/{side}/{group}", entries[group]), remedy="Check the slice indices against meta/info.json names.")
        elif "gripper" not in group.lower() and any("gripper" in n for n in span) and \
                any("gripper" in other.lower() for other in entries if other != group and (used is None or other in used)):
            report.add(f"groot.{side}_names", "warning",
                       f"{side}.{group} reads a gripper value ({names[start:end]}) although a separate gripper group exists.",
                       ("meta/modality.json", f"/{side}/{group}", entries[group]), remedy="Check the slice indices against meta/info.json names.")
    return valid


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[._\-\s]+", text.lower()) if t]


def _distinct(columns: list[str]) -> dict[str, set]:
    """Whole-token sets of each column after the token prefix all columns share (observation.images...)."""
    split = {c: _tokens(c) for c in columns}
    lists = [t for t in split.values() if t]
    prefix = 0
    if len(lists) > 1:
        while all(len(t) > prefix + 1 for t in lists) and len({t[prefix] for t in lists}) == 1:
            prefix += 1
    elif lists:
        prefix = len(lists[0]) - 1
    return {c: set(t[prefix:]) for c, t in split.items()}


def _designate(cameras: list[str], labels: dict[str, set]) -> dict[str, str]:
    """Pair a config camera with the label it names: each must be the other's unique best whole-token match."""
    def score(camera, label):
        a, b = set(_tokens(camera)), labels[label]
        return len(a & b) / len(a | b) if a | b else 0.0

    table = {(c, label): score(c, label) for c in cameras for label in labels}
    named = {}
    for camera in cameras:
        best = max((table[camera, label] for label in labels), default=0.0)
        tops = [label for label in labels if table[camera, label] == best]
        if best > 0 and len(tops) == 1 and [c for c in cameras if table[c, tops[0]] >= best] == [camera]:
            named[camera] = tops[0]
    return named


def _cameras(report: _Report, modality: dict, features: dict, info: dict, config: dict | None) -> None:
    entries = modality.get("video")
    wanted = list(dict.fromkeys(config["video"]["keys"])) if config and "video" in config else None
    if wanted is None:
        if not isinstance(entries, dict):
            if config is None:
                report.add("groot.video", "unknown", "modality.json declares no video entries.",
                           ("meta/modality.json", "/video", entries))
            return
        for key, entry in entries.items():
            _camera_column(report, key, entry, features, used=config is None)
        return
    if not isinstance(entries, dict):
        if entries is None and not info.get("video_path"):
            report.add("groot.video", "warning", f"The config reads cameras {wanted}, but modality.json has no video "
                       f"section and meta/info.json no video_path, so GR00T loads no camera ({LOADER}:416-417).",
                       ("meta/modality.json", "/video", entries), remedy="Declare the cameras in modality.json.")
        else:
            report.add("groot.video", "error", f"The config reads cameras {wanted}, but modality.json has "
                       + ("no video section" if entries is None else "a video section that is not an object")
                       + f"; GR00T fails when it loads the first sample ({LOADER}:282-284, 426).",
                       ("meta/modality.json", "/video", entries),
                       remedy="Declare each config camera under modality.json video with its original_key.")
        return
    if not info.get("video_path"):
        report.add("groot.video", "warning", f"meta/info.json has no video_path, so GR00T loads no camera for the "
                   f"config's {wanted} ({LOADER}:416-417).", ("meta/info.json", "/video_path", None),
                   remedy="Restore video_path in meta/info.json.")
    declared = list(entries)
    positional = any(k not in entries for k in wanted)  # loader:285
    if positional and len(wanted) != len(declared):
        report.add("groot.video_mapping", "error",
                   f"Config cameras {wanted} do not match modality.json {declared} by name or count; GR00T asserts "
                   f"({LOADER}:285-291).", ("meta/modality.json", "/video", declared),
                   remedy="First choose or review a training config matching this dataset's actual cameras. Do not rename a public dataset to silence the check. Only change an owned separate candidate's mapping when the independently reviewed recipe and camera sources require it.")
        return
    mapping = dict(zip(wanted, declared, strict=True)) if positional else {k: k for k in wanted}
    reads = {}
    for camera, key in mapping.items():
        column = _camera_column(report, key, entries[key], features, used=True)
        if column is not None:
            reads[camera] = column
    for key in declared:
        if key not in mapping.values():
            _camera_column(report, key, entries[key], features, used=False)
    columns = [c for c in (e.get("original_key", f"observation.images.{k}") if isinstance(e, dict) else None
                           for k, e in entries.items()) if isinstance(c, str)]
    by_column = _designate(wanted, _distinct(list(dict.fromkeys(columns))))
    owner = {column: camera for camera, column in by_column.items()}
    by_entry = _designate(wanted, {k: set(_tokens(k)) for k in declared}) if positional else {}
    swapped = {}
    for camera in wanted:
        column = reads.get(camera)
        if column is None:
            continue
        if owner.get(column, camera) != camera:
            swapped[camera] = f"{column} is named for config camera {owner[column]}"
        elif camera in by_column and by_column[camera] != column:
            swapped[camera] = f"the video named for it is {by_column[camera]}"
        elif camera not in by_column and column not in owner and by_entry.get(camera, mapping[camera]) != mapping[camera]:
            swapped[camera] = f"it is paired with entry {mapping[camera]}, while entry {by_entry[camera]} is named for it"
    pairs = ", ".join(f"{k} reads {reads[k]}" for k in wanted if k in reads)
    if positional:
        message = (f"Config cameras {wanted} are not all in modality.json {declared}, so GR00T pairs them by position "
                   f"({LOADER}:279-297); GR00T's own log says so (\"Video key mismatch between model config and "
                   f"dataset. Auto-mapping by position\"), but training continues: {pairs}")
    else:
        message = (f"Config cameras {wanted} are paired with modality.json video entries by name and each reads its "
                   f"entry's original_key ({LOADER}:282-285, 425-428), with no warning: {pairs}")
    if swapped:
        message += ". Swapped: " + "; ".join(f"{k} ({why})" for k, why in swapped.items()) + "."
    report.add("groot.video_mapping", "error" if swapped else "warning" if positional else "pass",
               message if swapped or positional else message + ".", ("meta/modality.json", "/video", entries),
               remedy="Name the modality.json video entries after the config's cameras, each pointing at the matching "
                      "original_key." if swapped or positional else "")


def _camera_column(report: _Report, key: str, entry, features: dict, used: bool) -> str | None:
    pointer = ("meta/modality.json", f"/video/{key}", entry)
    if not isinstance(entry, dict):
        problem, consequence = f"video.{key} is not an object", f"GR00T calls .get on it ({LOADER}:426-428)."
        column = None
    else:
        column = entry.get("original_key", f"observation.images.{key}")  # loader:426-428
        if not isinstance(column, str):
            problem, consequence = f"video.{key} has original_key {column!r}", f"GR00T looks it up in meta/info.json features ({LOADER}:429)."
        elif column not in features:
            problem = f"video.{key} reads {column}, which meta/info.json does not declare"
            consequence = f"GR00T asserts the key is a declared feature when it loads the first sample ({LOADER}:429-431)."
        else:
            dtype = features[column].get("dtype")
            if dtype not in ("video", "image") and used:
                report.add("groot.video_column", "warning",
                           f"video.{key} reads {column}, declared with dtype {dtype!r}. GR00T checks only that the key "
                           f"is declared ({LOADER}:429), then decodes the file video_path names for it as a video "
                           f"({LOADER}:433-446).", pointer, ("meta/info.json", f"/features/{column}/dtype", dtype),
                           remedy="Point original_key at the camera's video feature.")
            return column
    if used:
        report.add("groot.video_column", "error", f"{problem}; {consequence}", pointer,
                   ("meta/info.json", "/features", sorted(features)), remedy="Point original_key at a declared video feature.")
    else:
        report.add("groot.video_column", "pass", f"video.{key} is not read by the config's cameras ({LOADER}:420-428); "
                   f"noted only: {problem}.", pointer)
    return None


def _language(report: _Report, modality: dict, features: dict, stats_keys: set, config: dict | None) -> str | None:
    """Check the language GR00T loads; return the first language key (the only one GR00T reads, loader:260-273)."""
    entries = modality.get("annotation")
    if config is None:
        for key, entry in (entries.items() if isinstance(entries, dict) else ()):
            column = entry.get("original_key", f"annotation.{key}") if isinstance(entry, dict) else None
            if column not in features:
                report.add("groot.annotation_column", "unknown",
                           f"annotation.{key} reads column {column}, which meta/info.json does not declare. It matters "
                           f"only if the training config's language key is annotation.{key} ({LOADER}:381-384).",
                           ("meta/modality.json", f"/annotation/{key}", entry), remedy="Pass the training modality config to decide.")
        return None
    language = config.get("language")
    if language is None:
        return None
    keys, delta = language["keys"], language["delta_indices"]
    if not keys:
        report.add("groot.language_key", "error", f"The config's language modality has no keys; GR00T asserts "
                   f"({LOADER}:257-259).", ("meta/modality.json", "/annotation", None))
        return None
    if delta is None:
        report.add("groot.language_delta", "unknown", "The config's language delta_indices are not a literal list, "
                   f"so GR00T's single-timestep requirement ({LOADER}:274) was not checked.", ("meta/modality.json", "", None))
    elif delta != [0]:
        report.add("groot.language_delta", "error", f"The config's language delta_indices are {delta}; GR00T asserts "
                   f"they are [0] (\"Only single timestep is supported for language modality\", {LOADER}:274).",
                   ("meta/modality.json", "", delta), remedy="Use delta_indices=[0] for language.")
    first = keys[0]
    if len(keys) > 1:
        report.add("groot.language_key", "pass", f"The config lists language keys {keys}; GR00T reads only the "
                   f"first, {first} ({LOADER}:260-273).", ("meta/modality.json", "/annotation", None))
    if first in LANG_KEYS:  # read from meta/episodes.jsonl (loader:374-375, 593-597), checked with that file
        return first
    if not first.startswith("annotation."):
        report.add("groot.language_key", "error", f"Config language key {first} is neither task, sub_task nor an "
                   f"annotation.* key; GR00T asserts ({LOADER}:374-376).", ("meta/modality.json", "/annotation", None),
                   remedy="Use annotation.<modality.json annotation entry>, task or sub_task.")
        return first
    subkey = first[len("annotation."):]
    if not isinstance(entries, dict) or subkey not in entries:
        report.add("groot.language_key", "error",
                   f"Config language key {first} has no modality.json annotation entry {subkey}; GR00T "
                   + ("asserts" if isinstance(entries, dict) else "fails reading the annotation section")
                   + f" ({LOADER}:377-381).", ("meta/modality.json", "/annotation",
                                               sorted(entries) if isinstance(entries, dict) else entries),
                   remedy="First select or review a training config matching the dataset's existing language annotation. "
                          "Do not edit a public dataset merely to silence this check. Only if the reviewed owned recipe "
                          f"requires annotation.{subkey}, add a source-backed mapping in a separate candidate and recheck it.")
        return first
    entry = entries[subkey]
    pointer = ("meta/modality.json", f"/annotation/{subkey}", entry)
    column = entry.get("original_key", first) if isinstance(entry, dict) else None  # loader:381
    if not isinstance(column, str):
        report.add("groot.annotation_column", "error", f"annotation.{subkey} "
                   + ("is not an object" if not isinstance(entry, dict) else f"has original_key {column!r}")
                   + f"; GR00T reads original_key from it and indexes the parquet table with it ({LOADER}:381-382).",
                   pointer, remedy="Point original_key at task_index or another declared column.")
        return first
    feature = features.get(column)
    if feature is None:
        if column in stats_keys:
            report.add("groot.annotation_column", "pass",
                       f"The config's language key {first} reads column {column}. meta/info.json does not declare it, "
                       "but meta/stats.json has statistics for it, so the parquet files most likely hold it "
                       f"({LOADER}:381-384).", pointer)
        else:
            # Validator recheck: GR00T reads the parquet directly, so a column missing only from the metadata still
            # works; the metadata cannot decide, so this is a warning rather than an error.
            report.add("groot.annotation_column", "warning",
                       f"The config's language key {first} reads column {column}, which neither meta/info.json nor "
                       f"meta/stats.json lists. GR00T reads it from the parquet files and fails with KeyError only if "
                       f"they lack it ({LOADER}:381-382); a LeRobot v2 info.json normally lists every parquet column.",
                       pointer, remedy="Confirm the parquet files hold this column, or point original_key at task_index.")
        return first
    size, dtype = _size(feature), feature.get("dtype")
    if size is not None and size > 1:
        report.add("groot.annotation_values", "error",
                   f"The config's language key {first} reads {column}, a {feature.get('shape')} {dtype} vector; GR00T "
                   f"looks each value up in the task map from meta/tasks.jsonl and fails on a vector ({LOADER}:382-384).",
                   pointer, ("meta/info.json", f"/features/{column}", feature),
                   remedy="Point original_key at task_index or another integer task-index column.")
    elif "int" not in str(dtype):
        report.add("groot.annotation_values", "warning",
                   f"The config's language key {first} reads {column} of dtype {dtype!r}; GR00T looks each value up "
                   f"in the task map keyed by meta/tasks.jsonl task_index integers ({LOADER}:171, 382-384).",
                   pointer, ("meta/info.json", f"/features/{column}", feature),
                   remedy="Point original_key at task_index or another integer task-index column.")
    else:
        report.add("groot.annotation_column", "pass", f"The config's language key {first} reads {column} ({dtype}), "
                   f"looked up in meta/tasks.jsonl ({LOADER}:381-384).", pointer)
    return first


def _metadata_lines(report: _Report, language: str | None) -> None:
    def episode(row):
        kinds = []
        if "length" not in row:
            kinds.append("no length")
        else:
            try:
                int(row["length"])
            except (TypeError, ValueError):
                kinds.append("length is not an integer")
        if "episode_index" not in row:
            kinds.append("no episode_index")
        if language == "task" and not (isinstance(row.get("tasks"), list) and row["tasks"]):
            kinds.append("no tasks list")
        if language == "sub_task" and not (isinstance(row.get("sub_tasks"), list) and all(
                isinstance(s, dict) and all(k in s for k in ("start", "end", "text")) for s in row["sub_tasks"])):
            kinds.append("no sub_tasks list")
        return kinds

    def task(row):
        return [] if "task_index" in row and "task" in row else ["no task_index or task"]

    consequences = {
        ("episodes", "blank line"): f"GR00T parses every line as JSON and fails ({LOADER}:163-165).",
        ("episodes", "not JSON"): f"GR00T parses every line as JSON and fails ({LOADER}:163-165).",
        ("episodes", "not a JSON object"): f"GR00T indexes each row by key and fails ({LOADER}:213).",
        ("episodes", "not UTF-8 text"): f"GR00T reads it as text and fails ({LOADER}:164-165).",
        ("episodes", "no length"): f"GR00T reads every row's length when it opens the dataset and raises KeyError ({LOADER}:211-213).",
        ("episodes", "length is not an integer"): f"GR00T converts every row's length with int() when it opens the dataset ({LOADER}:213).",
        ("episodes", "no episode_index"): f"GR00T raises KeyError when it loads that episode ({LOADER}:586-587).",
        ("episodes", "no tasks list"): f"The config's language key is task, so GR00T picks a row's instruction from its tasks list ({LOADER}:540-541).",
        ("episodes", "no sub_tasks list"): f"The config's language key is sub_task, so GR00T reads each row's sub_tasks start/end/text and raises KeyError ({LOADER}:543-553).",
        ("tasks", "blank line"): f"GR00T parses every line as JSON and fails ({LOADER}:168-170).",
        ("tasks", "not JSON"): f"GR00T parses every line as JSON and fails ({LOADER}:168-170).",
        ("tasks", "not a JSON object"): f"GR00T builds its task map from task_index and task and fails ({LOADER}:171).",
        ("tasks", "not UTF-8 text"): f"GR00T reads it as text and fails ({LOADER}:169-170).",
        ("tasks", "no task_index or task"): f"GR00T builds its task map from both and raises KeyError ({LOADER}:171).",
    }
    for kind, check in (("episodes", episode), ("tasks", task)):
        name = f"meta/{kind}.jsonl"
        path = report.file(name)
        if not path.is_file():
            report.add("groot.metadata", "error", f"{name} is missing; GR00T opens it unconditionally ({LOADER}:162-171).",
                       (name, "", None), remedy="Convert the dataset to LeRobot v2 metadata.")
            continue
        scan = _scan_jsonl(path, check)
        for problem, numbers in scan["problems"].items():
            report.add(f"groot.{kind}", "error", f"{name}: {len(numbers)} line(s) with {problem} (first {numbers[:5]}). "
                       + consequences[kind, problem], (name, "", numbers[:5]),
                       remedy="Write one JSON object per line, with no blank lines"
                              + (", each with episode_index and length." if kind == "episodes" else
                                 ", each with task_index and task."))
        if scan["truncated"]:
            report.add(f"groot.{kind}", "unknown", f"{name} has more than {MAX_LINES} lines; only the first {MAX_LINES} "
                       "were checked.", (name, "", MAX_LINES))


def _statistics(report: _Report, state: str, stats, groups: dict, features: dict) -> None:
    columns = sorted({c for side in groups.values() for c, _, _ in side.values()})
    if state == "missing":
        report.add("groot.statistics", "pass",
                   f"meta/stats.json is missing. Fine-tuning generates it before loading ({FACTORY}:57-59, "
                   f"{STATS}:251-291). Direct loaders ({DIRECT_USERS}) need it first: the loader asserts it exists "
                   f"({LOADER}:180-182).", ("meta/stats.json", "", None),
                   remedy="Before a direct-loader evaluation, run python gr00t/data/stats.py --dataset-path <dataset> "
                          "--embodiment-tag <tag>.")
        return
    if state == "too large":
        report.add("groot.statistics", "unknown", f"meta/stats.json was not read ({stats}).", ("meta/stats.json", "", None))
        return
    if state == "unreadable":
        report.add("groot.statistics", "warning",
                   f"meta/stats.json cannot be parsed ({stats}). Fine-tuning discards and regenerates it "
                   f"({STATS}:79-86, 251-291), but direct loaders ({DIRECT_USERS}) fail reading it ({LOADER}:183-184).",
                   ("meta/stats.json", "", None), applies_to=DIRECT, remedy="Delete it and regenerate it with gr00t/data/stats.py.")
        return
    if not isinstance(stats, dict):
        report.add("groot.statistics", "error", f"meta/stats.json is a JSON {type(stats).__name__}, not an object; "
                   f"fine-tuning fails reading its cache entries ({STATS}:210-212).", ("meta/stats.json", "", None),
                   applies_to=FT, remedy="Delete it and regenerate it with gr00t/data/stats.py.")
        return
    for column in columns:
        width = _width(features[column])
        reason = _stale_reason(stats, features, column)
        entry = stats.get(column) if isinstance(stats.get(column), dict) else {}
        widths = {n: len(v) if isinstance(v, list) else type(v).__name__ for n, v in entry.items() if n in STAT_NAMES}
        wrong = {n: w for n, w in widths.items() if w != width}
        if reason is not None:
            report.add("groot.statistics", "pass",
                       f"meta/stats.json {column}: {reason}, so fine-tuning recomputes it from the parquet files "
                       f"({STATS}:201-229, 284-291)" + (f" (its widths {wrong} do not match the column's {width})"
                                                        if wrong else "")
                       + f". Direct loaders only need the file to exist ({LOADER}:180-184).",
                       ("meta/stats.json", f"/{column}", widths or None))
        elif wrong:
            report.add("groot.statistics", "error",
                       f"meta/stats.json {column} statistics have widths {wrong}, but the column has {width} values. "
                       f"Its {FINGERPRINTS} entry matches meta/info.json, so fine-tuning keeps it ({STATS}:201-229) "
                       f"and slices it with the same [start:end] as the data ({LOADER}:527-530): groups get statistics "
                       "for other joints, or none, with no error.", ("meta/stats.json", f"/{column}", wrong),
                       applies_to=FT, remedy=f"Delete meta/stats.json (or its {column} entry) so fine-tuning regenerates it.")


def _relative_file(report: _Report, state: str, relative, config: dict | None) -> None:
    items = ((config or {}).get("action") or {}).get("action_configs")
    # generate_rel_stats rewrites the file at its end (stats.py:435), but with a relative action it first opens the
    # loader (stats.py:321), which reads the old file; so only action_configs without a relative action rewrite it.
    rewrites = isinstance(items, list) and all(i["rep"] not in (None, "RELATIVE") for i in items)
    if state == "too large":
        report.add("groot.relative_statistics", "unknown", f"meta/relative_stats.json was not read ({relative}).",
                   ("meta/relative_stats.json", "", None))
    elif state == "unreadable":
        report.add("groot.relative_statistics", "warning" if rewrites else "error",
                   f"meta/relative_stats.json cannot be parsed ({relative}); the loader reads it whenever it exists "
                   f"and fails ({LOADER}:186-189). "
                   + ("Fine-tuning rewrites it first, because the config has action_configs and no relative action "
                      f"({STATS}:418-435); direct loaders still fail." if rewrites else
                      "Fine-tuning rewrites it first only when the config has action_configs and no relative action "
                      f"({STATS}:418-435); with a relative action, generate_rel_stats opens the loader ({STATS}:321) "
                      f"before it rewrites the file ({STATS}:435)."),
                   ("meta/relative_stats.json", "", None), applies_to=DIRECT if rewrites else BOTH,
                   remedy="Delete it; fine-tuning regenerates it when the config uses relative actions.")
    elif state == "ok" and not isinstance(relative, dict):
        report.add("groot.relative_statistics", "error",
                   f"meta/relative_stats.json is a JSON {type(relative).__name__}, not an object; the loader fails "
                   f"removing its {FINGERPRINTS} entry ({LOADER}:186-194).", ("meta/relative_stats.json", "", None),
                   remedy="Delete it; fine-tuning regenerates it when the config uses relative actions.")


def _relative(report: _Report, config: dict | None, groups: dict, state: str, relative) -> None:
    action = (config or {}).get("action")
    if not action or "action_configs" not in action:
        return
    items, keys = action["action_configs"], action["keys"]
    if items == _UNREAD:
        report.add("groot.relative_config", "unknown", "The config's action_configs is not a literal list, so relative "
                   "actions (their state keys, delta alignment, widths and statistics) were not checked.",
                   ("meta/modality.json", "/action", keys), remedy="Write action_configs as a literal list of ActionConfig(...).")
        return
    if len(items) != len(keys):
        report.add("groot.relative_config", "error", f"The config has {len(items)} action_configs for {len(keys)} "
                   "action keys; GR00T asserts the counts match when it builds the config (gr00t/data/types.py:112-115).",
                   ("meta/modality.json", "/action", keys), remedy="Give one ActionConfig per action key, in order.")
        return
    rel = []
    for key, item in zip(keys, items, strict=True):
        if item["rep"] is None:
            report.add("groot.relative_config", "unknown", f"action_configs entry for {key} could not be read "
                       "statically, so whether it is relative was not checked.", ("meta/modality.json", f"/action/{key}", None))
        elif item["rep"] == "RELATIVE":
            rel.append((key, item))
    if not rel:
        return
    state_config = config.get("state") or {}
    state_keys = state_config.get("keys", [])
    state_delta, action_delta = state_config.get("delta_indices"), action.get("delta_indices")
    if not state_delta or not action_delta:
        report.add("groot.relative_delta", "unknown", "The config's state or action delta_indices are not a literal "
                   f"list (or list(range(...))), so the alignment fine-tuning asserts ({STATS}:317-320) was not checked.",
                   ("meta/modality.json", "", None))
    elif state_delta[-1] != action_delta[0]:
        report.add("groot.relative_delta", "error",
                   f"The config uses relative actions {[k for k, _ in rel]}, and its state delta_indices end at "
                   f"{state_delta[-1]} while its action delta_indices start at {action_delta[0]}; fine-tuning asserts "
                   f"they are equal before training ({STATS}:316-320).", ("meta/modality.json", "", None), applies_to=FT,
                   remedy="Make the last state delta index equal the first action delta index (usually both 0).")
    for key, item in rel:
        state_key = key if not item["state_key"] else item["state_key"]
        if state_key == _UNREAD:
            report.add("groot.relative_state", "unknown", f"Relative action {key} has a state_key that is not a "
                       "literal string, so its reference state was not checked.", ("meta/modality.json", "", None))
            continue
        if state_key not in state_keys:
            report.add("groot.relative_state", "error",
                       f"Relative action {key} is taken against state {state_key} (its state_key, or the action key "
                       f"when state_key is unset: {STATS}:310, getting_started/data_config.md:218-220), which is not one "
                       f"of the config's state keys {state_keys}; fine-tuning asserts before training ({STATS}:311), "
                       f"and the processor raises KeyError ({PROCESSOR}:377-383).",
                       ("meta/modality.json", "/state", state_keys), applies_to=FT,
                       remedy=f"Add {state_key} to the config's state keys (and modality.json), or set state_key.")
            continue
        if item["type"] == "NON_EEF" and key in groups.get("action", {}) and state_key in groups.get("state", {}):
            widths = [groups[side][k][2] - groups[side][k][1] for side, k in (("action", key), ("state", state_key))]
            if widths[0] != widths[1]:
                report.add("groot.relative_width", "error",
                           f"Relative joint-space action {key} has {widths[0]} values but its reference state "
                           f"{state_key} has {widths[1]}; GR00T raises when it subtracts them "
                           f"(gr00t/data/state_action/pose.py:270-274, reached from {STATS}:352-356 at fine-tuning and "
                           f"{PROCESSOR}:654-656 when actions are converted).",
                           ("meta/modality.json", f"/action/{key}", widths[0]), remedy="Make the two slices cover the same joints.")
        if state != "ok" or not isinstance(relative, dict) or key not in relative:
            report.add("groot.relative_statistics", "pass",
                       f"meta/relative_stats.json has no entry for relative action {key}. Fine-tuning generates it "
                       f"before loading ({FACTORY}:60, {STATS}:415-435); the loader treats the file as optional "
                       f"({LOADER}:186-194).", ("meta/relative_stats.json", "",
                                                sorted(k for k in relative if k != FINGERPRINTS)
                                                if isinstance(relative, dict) else None))


def check_groot_dataset(dataset: str | Path, modality_config: str | Path | None = None,
                        config_name: str | None = None, *, groot_version: str = "n1.7") -> dict:
    """Check a GR00T-flavoured LeRobot v2 dataset (and optionally its training modality config) statically.

    groot_version picks the GR00T release whose loader the rules follow:
    - "n1.7" (default): Isaac-GR00T 944fe70, the rules in this module;
    - "n1.5": Isaac-GR00T N1.5 4af2b62 (groot_contract_n15); modality_config is then an N1.5 data config .py file
      and config_name a class name or DATA_CONFIG_MAP key (a built-in key alone, such as so100, needs no file);
    - "auto": decided from the config file's form (N1.5 BaseDataConfig classes, or N1.7 ModalityConfig dicts with
      bare group names) when exactly one is present; otherwise the report is "unknown" and no rule is applied.
    check_groot_dataset_versions combines the profiles when the version is not known.
    """
    if groot_version != "n1.7":
        from . import groot_contract_n15 as n15

        if groot_version == "auto":
            detected = n15.detect_version(modality_config) if modality_config is not None else None
            if detected is None:
                return n15.unknown_version(dataset, "no config file was given" if modality_config is None else
                                           f"the form of {modality_config} does not say which: it needs N1.5 "
                                           "BaseDataConfig classes, or N1.7 ModalityConfig dicts with bare group "
                                           "names (N1.5 writes video./state./action. keys), and not both")
            groot_version = detected
        if groot_version == "n1.5":
            return n15.check(dataset, modality_config, config_name)
        if groot_version != "n1.7":
            raise InputError(f"groot_version must be 'n1.7', 'n1.5' or 'auto', not {groot_version!r}")
    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Dataset directory not found: {root}")
    report = _Report(root)
    info = read_json(report.file("meta/info.json"))
    features = _info(report, info)
    modality_path = report.file("meta/modality.json")
    if not modality_path.is_file():
        report.add("groot.modality", "error", f"meta/modality.json is missing; GR00T opens it unconditionally ({LOADER}:173-176).",
                   ("meta/modality.json", "", None), remedy="Copy and adapt an Isaac-GR00T examples/*/modality.json.")
        return _finish(report, root, None, None)
    modality = read_json(modality_path)
    name, config = _parse_config(modality_config, config_name) if modality_config is not None else (None, None)
    stats_state, stats = _read_cache(report.file("meta/stats.json"))
    relative_state, relative = _read_cache(report.file("meta/relative_stats.json"))
    if config is not None:
        _config_keys(report, modality, config)
    groups = {side: _groups(report, modality, features, side, config) for side in ("state", "action")}
    _cameras(report, modality, features, info, config)
    language = _language(report, modality, features, set(stats) if isinstance(stats, dict) else set(), config)
    _metadata_lines(report, language)
    _statistics(report, stats_state, stats, groups, features)
    _relative_file(report, relative_state, relative, config)
    _relative(report, config, groups, relative_state, relative)
    if config is None:
        report.add("groot.config", "unknown", "No modality config given, so config keys, camera pairing, language and "
                   "relative actions were not checked, and every modality.json entry was treated as read.",
                   ("meta/modality.json", "", None),
                   remedy="Pass the training modality config .py file (for example examples/SO100/so100_config.py).")
    return _finish(report, root, config, name)


def _finish(report: _Report, root: Path, config: dict | None, name: str | None) -> dict:
    counts = {s: sum(f["status"] == s for f in report.findings) for s in ("error", "warning", "unknown", "pass")}
    status = next((s for s in ("error", "warning", "unknown") if counts[s]), "pass")
    return {"schema_version": 2, "kind": "groot_dataset_contract", "status": status, "counts": counts,
            "dataset": str(root), "consumer": GROOT, "groot_version": "n1.7", "modality_config": config,
            "modality_config_name": name,
            "files": report.files, "findings": report.findings,
            "paths": {FT: f"launch_finetune -> {FACTORY}:57-69: generate_stats and generate_rel_stats, then the loader",
                      DIRECT: f"LeRobotEpisodeLoader without that step ({DIRECT_USERS})"},
            "scope": "Static metadata check of meta/modality.json against meta/info.json, the jsonl metadata, "
                     "statistics and an optional training modality config. Parquet contents, video frames, embodiment "
                     "metadata, training and task success are not checked."}


# ---------------------------------------------------------------- both GR00T versions, per dataset

VERSION_LABELS = {"n1.5": "Isaac-GR00T N1.5", "n1.7": "Isaac-GR00T N1.7"}
COMBINED_MAX_FINDINGS, COMBINED_MAX_BYTES = 50, 128 * 1024
N17_NOT_CHECKED = ["Parquet contents", "video frames", "embodiment metadata", "training"]
N17_NOT_CHECKED_WITHOUT_CONFIG = ["config keys", "camera pairing", "language", "relative actions"]
# N1.7 faults of the dataset as a whole (the loader opens these files for any config); every other N1.7 error found
# without a config is about one modality.json entry and holds only if the training config reads that entry.
_N17_DATASET_CODES = {"groot.format", "groot.metadata", "groot.modality", "groot.episodes", "groot.tasks",
                      "groot.relative_statistics"}
STATUS_RULE = ("error when any Isaac-GR00T version that was checked has an error: a fault that crashes or silently "
               "mis-trains one supported release is not an advisory for that release's users, and without a version "
               "the check cannot tell that the user is not on it. The headline names each failing version and how it "
               "fails, and remedy gives the --groot-version command for one version's verdict alone. Otherwise the "
               "worst of warning, unknown and pass over the versions checked.")
# The profiles word every fault that GR00T does not report with one of these phrases.
_SILENT = re.compile(r"with no (error|warning)|training continues")
_AT_OPEN = re.compile(r"\bopen")


def check_groot_dataset_versions(dataset: str | Path, modality_config: str | Path | None = None, *,
                                 config_name: str | None = None, groot_version: str | None = None) -> dict:
    """Check a dataset against the Isaac-GR00T version(s) it may be trained with; see the module docstring.

    groot_version "n1.5" / "n1.7" runs that profile only. Otherwise a config (a .py file, or for N1.5 a built-in
    config_name) selects the profile its form names, and nothing is run when the form is ambiguous; with neither,
    both profiles run and are combined per dataset (STATUS_RULE)."""
    from . import groot_contract_n15 as n15

    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Dataset directory not found: {root}")
    supplied = modality_config is not None or config_name is not None
    if groot_version is not None:
        if groot_version not in VERSION_LABELS:
            raise InputError(f"groot_version must be 'n1.5', 'n1.7' or None, not {groot_version!r}")
        if groot_version == "n1.7" and modality_config is None and config_name is not None:
            raise InputError("config_name selects a dict inside an N1.7 modality config .py file; pass the file too")
        mode, versions = "explicit", [groot_version]
    elif supplied:
        if modality_config is not None:
            path = Path(modality_config)
            if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
                raise InputError(f"Expected a config .py file no larger than {MAX_CONFIG_BYTES} bytes: {path}")
            detected = n15.detect_version(path)
            reason = (f"the form of {path.name} does not say which: it needs N1.5 BaseDataConfig classes, or N1.7 "
                      "ModalityConfig dicts with bare group names (N1.5 writes video./state./action. keys), and not "
                      "both")
        else:
            module, _, name = config_name.rpartition(":")
            builtin = n15._BUILTIN
            detected = "n1.5" if module in ("", n15._BUILTIN_MODULE) and (name in builtin.maps or name in
                                                                          builtin.classes) else None
            reason = (f"config_name {config_name!r} without a file names no built-in N1.5 data config, and an N1.7 "
                      "modality config is read from its .py file")
        if detected is None:
            return _combined(root, {}, "undetermined", supplied, unknown=n15.unknown_version(root, reason),
                             reason=reason)
        mode, versions = "detected", [detected]
    else:
        mode, versions = "both", ["n1.5", "n1.7"]
    reports, refused = {}, {}
    for version in versions:
        try:
            reports[version] = check_groot_dataset(root, modality_config, config_name, groot_version=version)
        except InputError as exc:
            if mode != "both":
                raise
            refused[version] = exc
    if not reports:
        raise next(iter(refused.values()))
    return _combined(root, reports, mode, supplied, refused={v: str(e) for v, e in refused.items()})


def _certainty(version: str, finding: dict, supplied: bool) -> str:
    if supplied:
        return "for the supplied config"
    if version == "n1.5":
        return "if the data config reads the cited entry" if _SILENT.search(finding["message"]) else "any data config"
    return "any training config" if finding["code"] in _N17_DATASET_CODES else \
        "if the training config reads the cited entry"


def _failure(errors: list) -> str | None:
    silent = [f for f in errors if _SILENT.search(f["message"])]
    loud = [f for f in errors if not _SILENT.search(f["message"])]
    parts = []
    if loud:
        parts.append("crash at open" if any(_AT_OPEN.search(f["message"]) for f in loud) else "crash")
    if silent:
        parts.append("also silently wrong" if loud else "silently wrong")
    return "; ".join(parts) or None


def _combined(root: Path, reports: dict, mode: str, supplied: bool, unknown: dict | None = None, reason: str = "",
              refused: dict | None = None) -> dict:
    refused = refused or {}
    per, tagged, files = {}, [], {}
    for version, report in reports.items():
        errors = [f for f in report["findings"] if f["status"] == "error"]
        # an N1.5 config that could not be read leaves the rules config-less (its coverage says so)
        read = supplied and not (version == "n1.5" and "config keys" in report["coverage"]["not_checked"])
        certainty = [_certainty(version, f, read) for f in errors]
        if_read = sum(c.startswith("if the") for c in certainty)
        paths = sorted({f["applies_to"] for f in errors})
        if read:
            statement = f"Checked against the supplied {VERSION_LABELS[version]} config."
        elif version == "n1.5":
            statement = ("No data config was given or read. N1.5 opens every modality.json entry when it loads the dataset, "
                         "so its crash errors occur whatever the data config reads; its silent errors apply if the "
                         "config reads the cited entry.")
        else:
            statement = ("No training modality config was given. N1.7 reads only the groups and cameras the config "
                         "names, so every entry was treated as read: its errors about an entry apply if the config "
                         "reads that entry; dataset-level ones (metadata files) apply to any config.")
        not_checked = report.get("coverage", {}).get("not_checked") if version == "n1.5" else \
            ([] if read else N17_NOT_CHECKED_WITHOUT_CONFIG) + N17_NOT_CHECKED
        consumer = report["consumer"] if version == "n1.5" else f"Isaac-GR00T N1.7 {GROOT.split()[-1]}"
        per[version] = {"consumer": consumer, "status": report["status"], "counts": report["counts"],
                        "fails": bool(errors), "failure": _failure(errors), "applies_to": paths,
                        "error_codes": sorted({f["code"] for f in errors}), "errors_certain": len(errors) - if_read,
                        "errors_if_read": if_read, "statement": statement,
                        "command": f"--groot-version {version}", "modality_config_name": report["modality_config_name"],
                        "paths": report["paths"], "not_checked": not_checked}
        files.update(report["files"])
        tagged += [{**f, "groot_version": version, "certainty": _certainty(version, f, read)}
                   for f in report["findings"]]
    for version, reason in refused.items():
        per[version] = {"consumer": None, "status": "unknown", "counts": {s: 0 for s in ("error", "warning", "unknown",
                                                                                          "pass")},
                        "fails": False, "failure": None, "applies_to": [], "error_codes": [], "errors_certain": 0,
                        "errors_if_read": 0, "statement": f"{VERSION_LABELS[version]} refused the input: {reason}",
                        "command": f"--groot-version {version}", "modality_config_name": None, "paths": {},
                        "not_checked": ["every rule: the input was refused"], "refused": reason}
        tagged.append({"code": "groot.version", "status": "unknown", "message": f"The {VERSION_LABELS[version]} profile "
                       f"refused the input: {reason}", "remedy": "", "applies_to": BOTH, "evidence": [],
                       "groot_version": version, "certainty": "any training config"})
    if unknown is not None:
        files.update(unknown["files"])
        tagged += [{**f, "groot_version": None, "certainty": "any training config"} for f in unknown["findings"]]
    per = {v: per[v] for v in sorted(per)}
    failing = [v for v in per if per[v]["fails"]]
    passing = [v for v in per if not per[v]["fails"] and v not in refused]
    statuses = [p["status"] for p in per.values()] + ([unknown["status"]] if unknown else [])
    status = "error" if failing else next((s for s in ("warning", "unknown") if s in statuses), "pass")
    if unknown is None:
        headline, remedy = _headline(per, failing, passing, mode, supplied)
        if refused:
            headline = headline[:-1] + "; " + " and ".join(VERSION_LABELS[v] for v in refused) + " refused the input."
    else:
        headline = f"The Isaac-GR00T version could not be determined from the config, so no rule was applied: {reason}."
        remedy = "Rerun with --groot-version n1.5 or --groot-version n1.7 (Python: groot_version=\"n1.5\" / \"n1.7\")."
    counts = {s: sum(f["status"] == s for f in tagged) for s in ("error", "warning", "unknown", "pass")}
    finding_codes = {}
    for finding in tagged:
        by_status = finding_codes.setdefault(finding["code"], {})
        by_status[finding["status"]] = by_status.get(finding["status"], 0) + 1
    findings = _bound(tagged)
    names = sorted({p["modality_config_name"] for p in per.values() if p["modality_config_name"]})
    not_checked = [f"{VERSION_LABELS[v]}: {item}" for v, p in per.items() for item in p["not_checked"]]
    return {"schema_version": 3, "kind": "groot_dataset_contract", "mode": mode, "status": status,
            "headline": headline, "status_rule": STATUS_RULE, "remedy": remedy,
            "groot_version": next(iter(per)) if len(per) == 1 and unknown is None else None,
            "versions_run": [v for v in per if v not in refused], "fails_on": failing, "dataset": str(root),
            "consumer": "; ".join(p["consumer"] for p in per.values() if p["consumer"]) or None,
            "modality_config_name": names[0] if len(names) == 1 else None, "counts": counts,
            "finding_codes": finding_codes, "versions": per,
            "files": dict(sorted(files.items())), "findings": findings, "total_findings": len(tagged),
            "findings_truncated": len(findings) < len(tagged),
            "coverage": {"complete": counts["unknown"] == 0, "modality_config_supplied": supplied,
                         "modality_config_name": names[0] if len(names) == 1 else None,
                         "versions_run": [v for v in per if v not in refused],
                         "not_checked": not_checked or ["every dataset rule: the Isaac-GR00T version is unknown"]},
            "scope": "Static metadata check of the dataset as each named Isaac-GR00T version reads it (see each "
                     "version's not_checked). Parquet contents, video frames, training and task success are not "
                     "checked."}


def _phrase(version: str, summary: dict) -> str:
    paths = summary["applies_to"]
    where = " fine-tuning" if paths == [FT] else " direct loaders" if paths == [DIRECT] else ""
    how = summary["failure"] + (", if the config reads the cited entries"
                                if summary["errors_if_read"] and not summary["errors_certain"] else "")
    return f"{VERSION_LABELS[version]}{where} ({how})"


def _headline(per: dict, failing: list, passing: list, mode: str, supplied: bool) -> tuple[str, str]:
    config = "" if supplied else " and the training config"
    if not failing:
        head = "No error on " + " or ".join(VERSION_LABELS[v] for v in passing) + " within the checked metadata" + (
            "" if supplied else "; no training config was given, so config keys, cameras and transforms were not "
            "checked") + "."
        remedy = ("Rerun with --groot-version n1.5 or --groot-version n1.7 and the training config to also check "
                  "config keys, cameras and transforms." if mode == "both" else
                  "" if supplied else f"Pass the training config with --groot-version {next(iter(per))} to also "
                  "check config keys, cameras and transforms.")
        return head, remedy
    head = "Fails on " + " and on ".join(_phrase(v, per[v]) for v in failing)
    if passing:
        head += "; no error on " + " or ".join(VERSION_LABELS[v] for v in passing)
    head += "."
    if mode == "both" and len(failing) == 1 and passing and not per[passing[0]].get("refused"):
        bad, good = failing[0], passing[0]
        remedy = (f"If you train with {VERSION_LABELS[good]}, rerun with --groot-version {good} (Python: "
                  f"groot_version=\"{good}\"){config} for its verdict alone. If you train with {VERSION_LABELS[bad]}, "
                  f"fix the cited {bad} errors, then rerun with --groot-version {bad}{config}.")
    elif mode == "both":
        remedy = ("Fix the cited errors of the release you train with, then rerun with --groot-version n1.5 or "
                  f"--groot-version n1.7{config}.")
    else:
        remedy = f"Fix the cited {VERSION_LABELS[failing[0]]} errors, then rerun with --groot-version {failing[0]}."
    return head, remedy


def _bound(findings: list) -> list:
    """At most COMBINED_MAX_FINDINGS findings within COMBINED_MAX_BYTES, the most severe first, but always the first
    finding of every (version, code, status) that is an error or unknown, so no version's errors are dropped."""
    def size(items):
        return len(json.dumps(items, default=str).encode())

    rank = {"error": 0, "unknown": 1, "warning": 2, "pass": 3}
    kept, seen = [], set()
    for index, finding in enumerate(findings):
        key = (finding["groot_version"], finding["code"], finding["status"])
        if finding["status"] in ("error", "unknown") and key not in seen:
            seen.add(key)
            kept.append(index)
    chosen = set(kept)
    rest = sorted((i for i in range(len(findings)) if i not in chosen),
                  key=lambda i: (rank.get(findings[i]["status"], 1), i))[:max(0, COMBINED_MAX_FINDINGS - len(kept))]
    while rest and size([findings[i] for i in kept + rest]) > COMBINED_MAX_BYTES:
        rest.pop()
    result = [findings[i] for i in sorted(kept + rest)]
    if size(result) > COMBINED_MAX_BYTES:  # only the guaranteed findings are left: shorten their evidence
        result = [{**f, "evidence": [{**e, "value": None} for e in f["evidence"]], "evidence_trimmed": True}
                  for f in result]
    return result
