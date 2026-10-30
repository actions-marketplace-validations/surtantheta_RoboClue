"""Explicit, read-only diagnostic gates for downstream CI and pre-commit.

Manifests describe checks, never shell commands to execute. The only command
parsed as data is a declared robot deployment command; RoboClue never runs it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from .fs_guard import open_regular
from .product_cli import InputError, _safe_path

MANIFEST_LIMIT = 256 * 1024
PRECOMMIT_FILE_LIMIT = 1024 * 1024
CHECK_LIMIT = 32
CHECKS = frozenset({"config", "training", "dataset", "deployment-units", "groot", "conventions", "recovery", "sim"})
_COMMON = {"id", "check", "artifact", "fail_on", "max_bytes", "max_rows", "max_files"}
_FIELDS = {
    "config": {"reference"},
    "training": {"reference", "pin_file", "criterion_sha256"},
    "dataset": {"fetch", "revision", "video_headers"},
    "deployment-units": {"robot_command", "robot_config", "lerobot_version"},
    "groot": {"groot_version", "groot_config", "groot_config_name"},
    "conventions": {"openpi_config", "openpi_source_root"},
    "recovery": {"video_headers"},
    "sim": {"converted"},
}
_PATHS = {"reference", "pin_file", "robot_config", "groot_config", "openpi_source_root", "converted"}
_BOUNDS = {"max_bytes": 64 * 1024**2, "max_rows": 200000, "max_files": 1024}


def _text(value, name, limit=4096):
    if type(value) is not str or not value or len(value) > limit or not all(c.isprintable() for c in value):
        raise InputError(f"{name} must be a nonempty bounded string without control characters.")
    return value


def _local(value, parent, workspace, *, exists=True):
    value = _text(value, "path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "~" in path.parts:
        raise InputError("Manifest paths must be relative, without parent traversal, and stay in the workspace.")
    path = parent / path
    if not path.is_relative_to(workspace):
        raise InputError("Manifest path leaves the declared workspace.")
    if exists:
        return _safe_path(path)
    return path


def _json_file(path, limit):
    with open_regular(_safe_path(path), "automation JSON") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise InputError(f"Automation JSON exceeds its {limit}-byte limit: {path}")
    from .conversion_fidelity import _reference_json

    value = _reference_json(raw)
    if type(value) is not dict:
        raise InputError("Automation JSON must contain an object.")
    return value, hashlib.sha256(raw).hexdigest()


def load_manifest(path, *, workspace, allow_hub=False, precommit=False):
    """Validate declaration structure before any diagnostic or network intake.

    Individual artifact availability is checked when that check runs, so one
    mistyped file cannot prevent independent declared checks from returning.
    """
    workspace = _safe_path(workspace)
    if not workspace.is_dir():
        raise InputError("The workspace must be a directory.")
    manifest = _local(str(path), workspace, workspace)
    document, pin = _json_file(manifest, MANIFEST_LIMIT)
    if set(document) != {"schema_version", "checks"} or document["schema_version"] != 1 or type(document["schema_version"]) is not int:
        raise InputError("Manifest schema_version must be integer 1; only schema_version and checks are accepted.")
    checks = document["checks"]
    if type(checks) is not list or not 1 <= len(checks) <= CHECK_LIMIT:
        raise InputError(f"Manifest needs between 1 and {CHECK_LIMIT} explicit checks.")
    normalized = []
    ids = set()
    for check in checks:
        if type(check) is not dict or not {"id", "check", "artifact"} <= set(check):
            raise InputError("Each check needs id, check and artifact.")
        kind = check["check"]
        if type(kind) is not str or kind not in CHECKS:
            raise InputError("Unknown check; choose config, training, dataset, deployment-units, groot, conventions, recovery or sim.")
        if set(check) - (_COMMON | _FIELDS[kind]):
            raise InputError(f"Unsupported fields for {kind}: {sorted(set(check) - (_COMMON | _FIELDS[kind]))}")
        name = _text(check["id"], "check id", 64)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in ids or name == "result":
            raise InputError("Check IDs must be unique simple names; result is reserved for the aggregate receipt.")
        ids.add(name)
        if precommit and kind != "config":
            raise InputError("Pre-commit supports small JSON config/reference comparisons only; run dataset/training checks in CI or at startup.")
        item = dict(check)
        artifact = _text(check["artifact"], "artifact")
        if artifact.startswith("hf://"):
            if kind != "dataset" or not allow_hub or check.get("fetch") is not True:
                raise InputError("Hub intake requires a dataset check, manifest fetch=true and the operator's --allow-hub.")
            if not re.fullmatch(r"[0-9a-f]{40}", str(check.get("revision", ""))):
                raise InputError("Hub automation requires an explicit immutable 40-character lowercase revision SHA.")
        else:
            item["artifact"] = str(_local(artifact, manifest.parent, workspace, exists=False))
            if check.get("fetch"):
                raise InputError("fetch=true is only for explicit public hf:// dataset inputs.")
        for key in _PATHS & set(item):
            item[key] = str(_local(item[key], manifest.parent, workspace, exists=False))
        for key in ("fetch", "video_headers"):
            if key in item and type(item[key]) is not bool:
                raise InputError(f"{key} must be a Boolean.")
        for key, maximum in _BOUNDS.items():
            if key in item and (type(item[key]) is not int or not 1 <= item[key] <= maximum):
                raise InputError(f"{key} must be an integer between 1 and {maximum}; split larger scans explicitly.")
        threshold = item.get("fail_on", "unknown" if kind == "dataset" else "error")
        if threshold not in ("error", "warning", "unknown") or (kind == "dataset" and threshold != "unknown"):
            raise InputError("Dataset CI requires fail_on=unknown; other checks accept error, warning or unknown.")
        item["fail_on"] = threshold
        if kind in ("config", "training") and not item.get("reference"):
            raise InputError(f"{kind} needs an independently reviewed reference; parsing alone cannot clear a gate.")
        if kind == "training" and (bool(item.get("pin_file")) + bool(item.get("criterion_sha256")) != 1):
            raise InputError("Training checks need exactly one independently retained pin_file or criterion_sha256.")
        if "criterion_sha256" in item and not re.fullmatch(r"[0-9a-f]{64}", _text(item["criterion_sha256"], "criterion_sha256")):
            raise InputError("criterion_sha256 must be 64 lowercase hexadecimal characters.")
        if kind == "deployment-units" and bool(item.get("robot_command")) + bool(item.get("robot_config")) != 1:
            raise InputError("deployment-units needs exactly one actual robot_command or robot_config; neither is executed.")
        if kind == "conventions" and not (item.get("openpi_config") and item.get("openpi_source_root")):
            raise InputError("conventions needs openpi_config and its reviewed openpi_source_root.")
        if kind == "groot" and item.get("groot_version") not in ("n1.5", "n1.7"):
            raise InputError("groot needs explicit groot_version n1.5 or n1.7.")
        if kind == "sim" and not item.get("converted"):
            raise InputError("sim automation requires a declared source/export pair (converted).")
        for key in ("robot_command", "groot_config_name", "openpi_config"):
            if key in item:
                _text(item[key], key)
        if "lerobot_version" in item:
            _text(item["lerobot_version"], "lerobot_version", 64)
        if precommit:
            for key in ("artifact", "reference"):
                p = Path(item[key])
                if p.suffix.lower() != ".json":
                    raise InputError("Pre-commit compares explicitly configured .json files only.")
        normalized.append(item)
    return {"path": str(manifest), "sha256": pin, "workspace": str(workspace), "checks": normalized}


def command_for(check):
    """Return public CLI argv, never executable shell text."""
    kind = check["check"]
    args = ["check", check["artifact"], "--json", "--fail-on", check["fail_on"]]
    if kind in ("config", "training"):
        args += ["--kind", kind]
    elif kind == "dataset":
        args += ["--kind", "dataset", "--question", "dataset"]
    else:
        args += ["--require", kind]
    mapping = {key: "--" + key.replace("_", "-") for key in _PATHS | {
        "robot_command", "groot_version", "groot_config_name", "openpi_config", "criterion_sha256",
        "revision", "max_bytes", "max_rows", "max_files", "lerobot_version"}}
    for key, option in sorted(mapping.items()):
        if key in check:
            args += [option, str(check[key])]
    for key in ("fetch", "video_headers"):
        if check.get(key):
            args.append("--" + key.replace("_", "-"))
    return args


def _output_directory(value, protected):
    if value is None:
        return None
    raw = _text(str(value) if isinstance(value, Path) else value, "output directory")
    path = Path(raw).absolute()
    _safe_path(path.parent)
    if any(path.is_relative_to(directory) for directory in protected):
        raise InputError("Receipt output cannot be inside an input artifact directory.")
    if path.exists() or path.is_symlink():
        raise InputError("Output directory must be new; existing paths are never overwritten.")
    path.mkdir(mode=0o700)
    return path


def _protected_directories(checks):
    directories = []
    for check in checks:
        for key in _PATHS | {"artifact"}:
            value = check.get(key)
            if not value or value.startswith("hf://"):
                continue
            try:
                path = _safe_path(value)
                if path.is_dir():
                    directories.append(path)
            except (InputError, OSError):
                pass  # It will become a named, per-check input error, not a read.
    return directories


def _validate_check_inputs(check, *, precommit):
    for key in _PATHS | {"artifact"}:
        value = check.get(key)
        if value and not value.startswith("hf://"):
            _safe_path(value)
    if precommit:
        for key in ("artifact", "reference"):
            _json_file(Path(check[key]), PRECOMMIT_FILE_LIMIT)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _cli_error(stdout, stderr):
    for text in (stdout, stderr):
        try:
            value = json.loads(text)
        except ValueError:
            continue
        if type(value) is dict and type(value.get("error")) is dict:
            return value["error"]
    return {"code": "invalid_input", "message": " ".join(stderr.split())[:600] or "The check did not return valid JSON evidence."}


def _invalid_manifest_receipt(path, *, workspace, output, error):
    """Retain a global input error when a new safe output path was requested."""
    protected = []
    try:
        root = _safe_path(workspace)
        manifest_path = _local(str(path), root, root)
        document, _ = _json_file(manifest_path, MANIFEST_LIMIT)
        # Even an invalid schema can name existing input directories. Do not
        # write a receipt into them while explaining that schema error.
        for check in document.get("checks", []) if type(document.get("checks")) is list else []:
            if type(check) is not dict:
                continue
            paths = {}
            for key in _PATHS | {"artifact"}:
                value = check.get(key)
                if type(value) is str and not value.startswith("hf://"):
                    try:
                        paths[key] = str(_local(value, manifest_path.parent, root, exists=False))
                    except InputError:
                        pass
            protected.extend(_protected_directories([paths]))
    except (InputError, ValueError, OSError):
        pass
    destination = _output_directory(output, protected)
    result = {"schema_version": 1, "kind": "automation_gate", "manifest": str(path),
              "manifest_sha256": None, "checks": [], "exit_code": 2, "gate": "blocked",
              "error": {"code": "invalid_manifest", "message": str(error)},
              "scope": "The manifest was invalid; no diagnostic check ran. This receipt does not approve any artifact."}
    if destination:
        _write_json(destination / "result.json", result)
    return result


def run_manifest(path, *, workspace=".", allow_hub=False, output=None, filenames=None):
    """Run every selected declared check through the real public CLI."""
    from .product_cli import main as check_main
    from .product_cli import render as render_check

    manifest = load_manifest(path, workspace=workspace, allow_hub=allow_hub, precommit=filenames is not None)
    checks = manifest["checks"]
    if filenames:
        changed = {str(_local(p, Path(manifest["workspace"]), Path(manifest["workspace"]), exists=False)) for p in filenames}
        checks = [c for c in checks if {c["artifact"], c["reference"], manifest["path"]} & changed]
    protected = _protected_directories(manifest["checks"])
    destination = _output_directory(output, protected)
    results = []
    for check in checks:
        stdout, stderr = io.StringIO(), io.StringIO()
        args = command_for(check)
        error = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                _validate_check_inputs(check, precommit=filenames is not None)
                code = check_main(args)
            except (InputError, ValueError, OSError, ImportError) as problem:
                code = 2
                error = {"code": "invalid_input", "message": str(problem)}
            except SystemExit:
                code = 2  # Invalid declared option values must not abort other checks.
        raw = stdout.getvalue()
        try:
            report = json.loads(raw) if raw else None
        except ValueError:
            report = None
        if code == 0 and type(report) is not dict:
            code = 2  # A process exit without the requested evidence is not a check.
        if code == 2:
            error = error or _cli_error(raw, stderr.getvalue())
            if report is None:
                report = {"error": error}
        if type(report) is dict and check["check"] == "training":
            if report.get("status") == "unknown":
                code = 3  # A severity-only filter cannot waive the required target evidence.
            elif code == 0 and report.get("status") != "pass":
                code = 2
        if type(report) is dict and check["check"] == "config" and code == 0 and report.get("equal") is not True:
            code = 2
        result = {"id": check["id"], "check": check["check"], "exit_code": code,
                  "gate": "clear" if code == 0 else "blocked", "argv": ["roboclue", *args],
                  "scope": "Only the explicitly declared check and failure threshold; not whole-pipeline compatibility."}
        if error:
            result["error"] = error
            result["summary"] = "Input: " + " ".join(str(error.get("message", "Invalid check input")).split())[:600]
        elif type(report) is dict:
            result["summary"] = render_check(report, source=check["artifact"],
                kind="units" if check["check"] == "deployment-units" else check["check"], command=["roboclue", *args]).strip()
        else:
            result["summary"] = "Input: " + " ".join(stderr.getvalue().split())[:600]
        if destination:
            _write_json(destination / (check["id"] + ".json"), {"result": result, "report": report,
                        "stderr": stderr.getvalue()})
        results.append(result)
    codes = {c["exit_code"] for c in results}
    code = next((c for c in (2, 1, 3) if c in codes), 0)
    if codes - {0, 1, 2, 3}:
        code = 2
    result = {"schema_version": 1, "kind": "automation_gate", "manifest": manifest["path"],
              "manifest_sha256": manifest["sha256"], "checks": results, "exit_code": code,
              "gate": "not_selected" if not results else "clear" if code == 0 else "blocked",
              "scope": "Only selected declared checks; unselected or undeclared checks and physical success are unassessed."}
    if destination:
        _write_json(destination / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m roboclue.automation", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "pre-commit"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", default=".roboclue-precommit.json" if name == "pre-commit" else ".roboclue-ci.json")
        sub.add_argument("--workspace", default=".")
        sub.add_argument("--output", help="New directory for original per-check JSON and aggregate receipt")
        if name == "run":
            sub.add_argument("--allow-hub", action="store_true", help="Also requires per-check fetch=true and immutable revision")
        else:
            sub.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)
    try:
        result = run_manifest(args.manifest, workspace=args.workspace, allow_hub=getattr(args, "allow_hub", False),
                              output=args.output, filenames=args.filenames if args.command == "pre-commit" else None)
    except (InputError, ValueError, OSError, ImportError) as error:
        print("RoboClue automation input: " + " ".join(str(error).split()), file=sys.stderr)
        if args.output:
            try:
                _invalid_manifest_receipt(args.manifest, workspace=args.workspace, output=args.output, error=error)
            except (InputError, ValueError, OSError):
                pass  # An invalid/existing output path must never be overwritten.
        return 2
    for check in result["checks"]:
        print(f"{check['id']}: {check['gate']} (exit {check['exit_code']}; {check['check']})")
        print(check["summary"])
    print(f"RoboClue: {result['gate']}; {len(result['checks'])} declared check(s). Undeclared checks remain unassessed.")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
