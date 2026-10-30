"""One bounded robot-command/config -> checkpoint -> dataset readiness workflow.

Command text is parsed, never executed. Only operator-authorized metadata and
explicitly enabled processor statistics are read; no policy parameters load.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from . import hub, robot_interface
from .artifacts import MAX_JSON_BYTES, InputError, contained, dataset_root, read_json

_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_UNRESOLVED = object()
_PIN = re.compile(r"[0-9a-f]{40}\Z")
AXES = ("loading", "cameras", "image_size", "state", "action", "joint_order", "declared_rate",
        "normalization", "relative_actions", "units")


class AmbiguousReadinessReference(InputError):
    """A relative native path has no caller-authorized working directory."""


def _local(value, read_roots=None, base_directory=None, *, explicit=None, resolutions=None):
    """Authorize one local readiness input.

    explicit names a top-level input the caller supplied directly ("dataset", "checkpoint", "robot config",
    "deployment", "command directory", "OpenPI source root"). A symlink in that path is resolved once to the real
    path, recorded in resolutions, and every later rule applies from the real path; under MCP the real path must lie
    inside the read roots, so resolution never widens access. A path read from command text or a config
    (explicit=None) keeps refusing any symlink in its path. Links inside artifact trees are governed by their readers.
    """
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise InputError("Readiness local inputs must be explicit nonempty paths")
    p = Path(value)
    if str(p).startswith("~"):
        raise InputError("Readiness literal tilde paths are refused; use the intended absolute path without '~' expansion")
    # Simplifying link/../checkpoint can select a different artifact than native I/O.
    # Refuse parent components before any path probing or lexical simplification.
    if ".." in p.parts or (base_directory is not None and ".." in Path(base_directory).parts):
        raise InputError("Readiness parent path components are refused; use the intended absolute path without '..'")
    if not p.is_absolute():
        if base_directory is None:
            if read_roots is not None:
                raise AmbiguousReadinessReference("Relative input has no unambiguous authorized command directory; use an absolute path")
            base_directory = str(Path.cwd())
        p = Path(base_directory) / p
    p = Path(os.path.abspath(p))
    if explicit is not None:
        # With '..' refused, this resolution is the kernel's own lookup: the real path is the artifact native I/O opens.
        real = p.resolve()
        if read_roots is not None and not any(real.is_relative_to(Path(r).resolve()) for r in read_roots):
            raise InputError(f"Readiness {explicit} {value!r} is outside configured MCP read roots"
                             + (" once its symlinks are resolved" if real != p else "")
                             + "; pass a path inside the read roots or ask the operator to add its real location as a root")
        if real != p and resolutions is not None:
            resolutions.append({"role": explicit, "given": value, "real": str(real)})
        p = real
    # Authorize lexically before probing symlinks/existence, including extracted references.
    elif read_roots is not None and not any(p.is_relative_to(Path(r)) for r in read_roots):
        raise InputError("Readiness input is outside configured MCP read roots")
    links = [q for q in (p, *p.parents) if q.is_symlink()]
    if links:
        if explicit is not None:
            raise InputError(f"Readiness {explicit} path changed while it was being resolved; retry with the real path")
        link = links[-1]  # the outermost link, the first one a reader meets
        base = Path(os.path.abspath(base_directory)) if base_directory is not None else None
        shown = link.relative_to(base).as_posix() if base is not None and link.is_relative_to(base) else str(link)
        raise InputError(f"Readiness input symlink ancestry is refused: {shown!r} is a symbolic link in the path "
                         f"{value!r} read from the command or robot config. Write the real path there, or pass the "
                         "policy explicitly (--policy, MCP checkpoint): an explicitly named top-level path is "
                         "resolved once and reported.")
    p = p.resolve()
    if read_roots is not None and not any(p.is_relative_to(Path(r).resolve()) for r in read_roots):
        raise InputError("Readiness input is outside configured MCP read roots")
    return str(p)



def authorize_command_directory(value, *, read_roots=None, resolutions=None):
    """Authorize an explicitly declared absolute command base; a symlink in it is resolved once and reported."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or not Path(value).expanduser().is_absolute():
        raise InputError("command_directory must be an explicit absolute directory")
    path = _local(value, read_roots=read_roots, explicit="command directory", resolutions=resolutions)
    if not Path(path).is_dir():
        raise InputError("command_directory must be an existing authorized directory")
    return path

def _reference(value, *, native=False, hint=None, read_roots=None, base_directory=None, role=None, resolutions=None):
    if not isinstance(value, str) or not value.strip():
        raise InputError("Policy and dataset references must be nonempty strings")
    if value.startswith("hf://"):
        hub._repo(value)
        return value, "explicit_public_identifier"
    if "://" in value:
        raise InputError("Only explicit local paths or hf:// public identifiers are supported")
    if native and _REPO.fullmatch(value):
        if base_directory is None and read_roots is not None:
            if hint == "hf://" + value:
                return hint, "bare_native_identifier_disambiguated_by_explicit_public_policy"
            raise AmbiguousReadinessReference("A bare native policy reference may be a local path or Hub ID; supply an explicit hf:// policy or authorized command directory")
        candidate = _local(value, read_roots, base_directory)
        if Path(candidate).is_dir() or Path(candidate).is_file():
            return candidate, "existing_native_local_path_precedes_hub"
        return "hf://" + value, "bare_native_identifier_no_local_directory_at_declared_base"
    return _local(value, read_roots, base_directory, explicit=role, resolutions=resolutions), "explicit_local_path"


def _same_policy(a, b):
    def canonical(value):
        p = Path(value)
        return str(p.parent) if not value.startswith("hf://") and p.name == "config.json" else value
    return canonical(a) == canonical(b)


def _finding(code, status, message, remedy, evidence=None):
    return {"code": code, "status": status, "message": message, "remedy": remedy,
            "evidence": evidence or []}


def _unassessed(finding):
    return {"schema_version": 1, "kind": "robot_readiness", "status": finding["status"],
            "summary": finding["message"],
            "axes": {name: {"status": "unknown", "reasons": [finding["message"]]} for name in AXES},
            "findings": [finding], "scope": "Readiness could not be fully assessed; no command, model or robot executed."}


def prepare_readiness(*, robot_command=None, robot_config=None, checkpoint=None, dataset=None,
                      read_roots=None, base_directory=None):
    """Parse and authorize all caller-supplied paths before any public request."""
    if base_directory is None and read_roots is None:
        base_directory = str(Path.cwd())
    if (robot_command is None) == (robot_config is None):
        raise InputError("Supply exactly one of robot_command or robot_config")
    cp = ds = None
    resolved = []  # explicitly named top-level paths reached through a symlink: resolved once, reported
    if checkpoint is not None:
        cp, _ = _reference(checkpoint, read_roots=read_roots, base_directory=base_directory, role="checkpoint",
                           resolutions=resolved)
    if dataset is not None:
        ds, _ = _reference(dataset, read_roots=read_roots, base_directory=base_directory, role="dataset",
                           resolutions=resolved)
        if not ds.startswith("hf://"):
            ds = _local(str(dataset_root(ds)), read_roots, base_directory, explicit="dataset", resolutions=resolved)
    if robot_config is not None:
        path = _local(robot_config, read_roots, base_directory, explicit="robot config", resolutions=resolved)
        robot = robot_interface.from_config(path)
    else:
        robot = robot_interface.from_command(robot_command)
    embedded = robot.get("policy_path")
    interpretation = None
    if embedded is not None:
        embedded, interpretation = _reference(embedded, native=True, hint=cp,
            read_roots=read_roots, base_directory=base_directory)
    conflict = cp is not None and embedded is not None and not _same_policy(cp, embedded)
    return {"robot": robot, "checkpoint": cp or embedded, "dataset": ds,
            "policy_interpretation": interpretation, "policy_conflict": conflict,
            "explicit_policy": cp, "embedded_policy": embedded,
            "read_roots": read_roots, "base_directory": base_directory, "resolved_paths": resolved}


def _training_reference(root):
    root = Path(root)
    if root.name == "config.json":
        root = root.parent
    path = contained(root, "train_config.json")
    if not path.is_file():
        return None
    data = read_json(path)
    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("repo_id"), str):
        return None
    repo = dataset["repo_id"]
    if not _REPO.fullmatch(repo):
        raise InputError("Saved training dataset reference is not one supported public repository ID")
    revision = dataset.get("revision") or "main"
    hub._revision(revision)
    with path.open('rb') as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES or json.loads(raw) != data:
        raise InputError("Saved training configuration changed during bounded identity capture")
    return {"source": "hf://" + repo, "revision": revision,
            "immutable_revision_declared": bool(_PIN.fullmatch(revision)),
            "evidence": {"path": str(path), "pointer": "/dataset",
                         "sha256": hashlib.sha256(raw).hexdigest()}}


def _finish(report, *, plan, provenance, budget, training_reference=None, dataset_context=None):
    report["policy_interpretation"] = plan.get("policy_interpretation")
    report["relative_path_base"] = plan.get("base_directory")
    report["resolved_paths"] = list(plan.get("resolved_paths") or [])
    report["training_reference"] = training_reference
    report["dataset_context"] = dataset_context
    report["scope"] = (str(report.get("scope", "")) +
        " Dataset comparisons use declared references; historical training consumption is not authenticated.")
    report["provenance"] = provenance
    report["intake_budget"] = budget.summary()
    findings = report.setdefault("findings", [])
    overrides = set(plan["robot"].get("policy_overrides", {})) - {"n_action_steps"}
    if overrides:
        findings.append(_finding("readiness.policy_overrides_unassessed", "unknown",
            "Additional policy overrides were parsed but are not modeled: " + ", ".join(sorted(overrides)),
            "Supply an exported effective policy configuration or remove unsupported overrides before interpreting these axes."))
    undecided = [name for name, axis in report.get('axes', {}).items()
                 if name != 'units' and axis.get('status') == 'unknown']
    if undecided and not any(f.get('status') == 'unknown' for f in findings):
        findings.append(_finding('readiness.axes_unassessed', 'unknown',
            'Static axes remain unassessed: ' + ', '.join(undecided) + '.',
            'Inspect the per-axis reasons and supply the missing command declarations or comparison metadata.'))
    report["counts"] = {s: sum(f.get("status") == s for f in findings) for s in ("error", "warning", "unknown", "pass")}
    report["total_findings"] = len(findings)
    report["next_actions"] = list(dict.fromkeys(str(f["remedy"]) for f in findings
        if f.get("status") != "pass" and f.get("remedy")))[:12]
    if report["counts"]["error"]:
        report["status"] = "error"
    elif report["counts"]["warning"]:
        report["status"] = "warning"
    elif report["counts"]["unknown"]:
        report["status"] = "unknown"
    report["summary"] = {"error": "Supported command/policy compatibility or loading errors require review before running.",
        "warning": "Supported robot-policy checks found conditions to review.",
        "unknown": "Some robot-policy axes lack the evidence needed for a decision.",
        "pass": "The decidable static robot-policy axes agree within the supplied evidence."}.get(report["status"], report.get("summary", ""))
    if len(json.dumps(report, allow_nan=False).encode()) > 768 * 1024:
        raise InputError("Readiness report exceeds its bounded transport budget; narrow the supplied inputs")
    return report


def check_robot_readiness(*, robot_command=None, robot_config=None, checkpoint=None, dataset=None,
                         policy_revision="main", dataset_revision=None, fetch=False, processor_state=False,
                         max_bytes=8 * 1024 ** 2, max_files=16, read_roots=None, base_directory=None,
                         _prepared=None, _resolved_dataset=_UNRESOLVED, _checkpoint_snapshot=None, _budget=None):
    """Resolve one declared readiness job; internal resolved inputs are never MCP arguments."""
    from .readiness import readiness
    from .readiness_intake import ReadinessIntakeBudget, resolve_readiness_snapshot

    if type(fetch) is not bool or type(processor_state) is not bool:
        raise InputError("fetch and processor_state must be strict booleans")
    budget = _budget or ReadinessIntakeBudget(max_bytes=max_bytes, max_files=max_files)
    hub._revision(policy_revision)
    if dataset_revision is not None:
        hub._revision(dataset_revision)
    try:
        plan = _prepared or prepare_readiness(robot_command=robot_command, robot_config=robot_config,
            checkpoint=checkpoint, dataset=dataset, read_roots=read_roots, base_directory=base_directory)
    except AmbiguousReadinessReference as exc:
        plan = {"robot": {}}
        return _finish(_unassessed(_finding("readiness.reference_ambiguous", "unknown", str(exc),
            "Supply an authorized command_directory for relative native paths, use absolute local references, or explicitly supply checkpoint=hf://ORG/REPO for the intended public policy.")),
            plan=plan, provenance=[], budget=budget)
    provenance, substitutions = [], []
    if plan["policy_conflict"]:
        report = _unassessed(_finding("readiness.policy_reference_conflict", "error",
            "The supplied policy and the command/config policy identify different checkpoints; neither silently overrides the other.",
            "Use the policy actually selected by the robot command, or explicitly correct that command/config."))
        report["policy_references"] = {"supplied": plan["explicit_policy"], "embedded": plan["embedded_policy"]}
        return _finish(report, plan=plan, provenance=[], budget=budget)
    if plan["checkpoint"] is None:
        return _finish(_unassessed(_finding("readiness.policy_missing", "unknown",
            "The command/config declares no checkpoint and no policy was supplied.",
            "Pass --policy with the checkpoint used by this command.")), plan=plan, provenance=[], budget=budget)
    # Permission rejection is an invocation error, before any remote open.
    if not fetch and any(isinstance(v, str) and v.startswith("hf://") for v in (plan["checkpoint"], plan["dataset"])):
        raise InputError("Public readiness intake requires explicit fetch/operator Hub permission")
    with ExitStack() as stack:
        def resolve(source, repo_type, revision):
            if not source.startswith("hf://"):
                return source, None
            view = stack.enter_context(resolve_readiness_snapshot(source, repo_type=repo_type,
                revision=revision, fetch=fetch, budget=budget, processor_state=processor_state if repo_type == "model" else False))
            provenance.append(copy.deepcopy(view.provenance))
            substitutions.append((str(view.root.resolve()), source + "@" + view.provenance["resolved_revision"]))
            return str(view.root), view
        training_reference = context = None
        extra = []
        try:
            cp, cp_view = _checkpoint_snapshot or resolve(plan["checkpoint"], "model", policy_revision)
        except (InputError, OSError, ValueError, TypeError) as exc:
            return hub._portable(_finish(_unassessed(_finding("readiness.policy_unassessed", "unknown", str(exc),
                "Supply the supported checkpoint metadata within the authorized intake limits.")),
                plan=plan, provenance=provenance, budget=budget), substitutions)
        try:
            training_reference = _training_reference(cp)
        except (InputError, OSError, ValueError, TypeError) as exc:
            extra.append(_finding("readiness.training_reference_unassessed", "unknown", str(exc),
                "Supply an explicit dataset or correct the saved training dataset reference."))
        dataset_already_attempted = _resolved_dataset is not _UNRESOLVED
        ds, contextual = (None if not dataset_already_attempted else _resolved_dataset), False
        source = plan["dataset"]
        auto_followed = source is None and training_reference is not None
        selected_revision = dataset_revision or "main"
        if source is None and training_reference is not None:
            source = training_reference["source"]
            selected_revision = dataset_revision or training_reference["revision"]
            contextual = not (training_reference["immutable_revision_declared"] and
                              selected_revision == training_reference["revision"])
            if not fetch:
                extra.append(_finding("readiness.training_metadata_not_fetched", "unknown",
                    "Saved training dataset metadata was not fetched; the saved reference alone cannot establish dataset-dependent axes.",
                    "Explicitly enable bounded public intake or supply the intended local comparison dataset."))
                source = None
        if source is not None:
            context = {"source": source, "selected_revision": selected_revision,
                       "role": ("current_context_only" if contextual else "saved_pinned_training_reference"
                                if auto_followed else "user_declared_comparison_artifact"),
                       "historical_training_data_authenticated": False,
                       "not_historical_training_evidence": contextual}
            if ds is None and dataset_already_attempted:
                extra.append(_finding("readiness.dataset_snapshot_unavailable", "unknown",
                    "The shared preflight dataset intake failed; readiness did not retry or resolve a different snapshot.",
                    "Inspect the original dataset intake finding; independent checkpoint and robot axes remain available."))
            elif ds is None:
                try:
                    ds, _ = resolve(source, "dataset", selected_revision)
                except (InputError, OSError, ValueError, TypeError) as exc:
                    extra.append(_finding("readiness.dataset_unassessed", "unknown", str(exc),
                        "Supply supported comparison metadata within the authorized intake limits; independent robot-policy checks remain available."))
        if ds is not None:
            from .artifacts import load_dataset
            try:
                load_dataset(ds)
            except (InputError, OSError, ValueError, TypeError) as exc:
                extra.append(_finding("readiness.dataset_unassessed", "unknown", str(exc),
                    "Supply readable supported dataset metadata; independent checkpoint and robot checks remain available."))
                ds = None
        try:
            report = readiness(cp, plan["robot"], None if contextual else ds,
                omitted_states=cp_view.omitted_states if cp_view is not None else None)
            if contextual and ds is not None:
                from .artifacts import load_dataset
                try:
                    report["current_dataset_metadata"] = load_dataset(ds).summary()
                except (InputError, OSError, ValueError, TypeError) as exc:
                    extra.append(_finding("readiness.context_metadata_unassessed", "unknown", str(exc),
                        "Supply supported metadata for this contextual dataset snapshot."))
                extra.append(_finding("readiness.training_revision_unrecorded", "unknown",
                    "The saved dataset reference does not pin these bytes as the training snapshot; current metadata is contextual only.",
                    "Provide the immutable dataset revision actually used for training to assess training-dependent axes."))
        except (InputError, OSError, ValueError, TypeError) as exc:
            report = _unassessed(_finding("readiness.input_unassessed", "unknown", str(exc),
                "Inspect the cited input and supply the missing supported evidence."))
        report.setdefault("findings", []).extend(extra)
        report = _finish(report, plan=plan, provenance=provenance, budget=budget,
            training_reference=training_reference, dataset_context=context)
        return hub._portable(report, substitutions)
