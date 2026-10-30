"""Portable, hash-checked diagnostic regression cases, without weights or executable input."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import __version__
from .artifacts import InputError, contained, read_json, sha256
from .diagnostics import audit


def signature(report: dict) -> dict:
    return {"status": report["status"], "findings": sorted((f["code"], f["status"]) for f in report["findings"])}


def export_bundle(checkpoint: str, output: str, dataset: str | None = None, deployment: str | None = None) -> dict:
    report = audit(checkpoint, dataset, deployment)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise InputError("Bundle destination already exists; choose a new directory")
    cp_root = Path(checkpoint).resolve()
    if cp_root.is_file():
        cp_root = cp_root.parent
    ds_root = Path(dataset).resolve() if dataset else None
    if ds_root and ds_root.is_file():
        ds_root = ds_root.parent.parent if ds_root.parent.name == "meta" else ds_root.parent
    sources = []
    for artifact in report["artifacts"]:
        source = Path(artifact["path"])
        role = artifact["role"]
        if role.startswith("dataset_"):
            relative = Path("dataset") / source.relative_to(ds_root)
        elif role == "deployment_contract":
            relative = Path("deployment.json")
        else:
            relative = Path("checkpoint") / source.relative_to(cp_root)
        sources.append((source, relative, artifact["sha256"]))
    # Export into a temporary sibling, publishing only after all copies are complete.
    import tempfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".policy-doctor-", dir=destination.parent))
    try:
        entries = []
        for source, relative, expected_hash in sources:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != expected_hash:
                raise InputError("An input changed during export; rerun the audit")
            entries.append({"path": relative.as_posix(), "sha256": expected_hash})
        manifest = {"schema_version": 1, "engine_version": __version__, "files": entries, "checkpoint": "checkpoint",
                    "dataset": "dataset" if dataset else None, "deployment": "deployment.json" if deployment else None,
                    "expected": signature(report)}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "README.md").write_text("# Policy Doctor regression bundle\n\nInstall policy-deployment-doctor, then run:\n\n```sh\npolicy-doctor reproduce . --json\n```\n\nThis reproduces diagnostic findings, not robot behavior. Only metadata and processor statistics are included.\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"bundle": str(destination), "files": len(sources), "expected": signature(report)}


def reproduce_bundle(bundle: str) -> dict:
    root = Path(bundle).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise InputError("Invalid bundle manifest")
    listed = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise InputError("Invalid bundle file entry")
        path = contained(root, entry.get("path"))
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise InputError(f"Bundle artifact hash mismatch: {entry.get('path')}")
        listed.add(path)
    cp = contained(root, manifest.get("checkpoint"))
    ds = contained(root, manifest["dataset"]) if manifest.get("dataset") else None
    dep = contained(root, manifest["deployment"]) if manifest.get("deployment") else None
    report = audit(str(cp), str(ds) if ds else None, str(dep) if dep else None)
    if not {Path(a["path"]) for a in report["artifacts"]}.issubset(listed):
        raise InputError("Bundle audit consumed an artifact absent from its hash manifest")
    # JSON normalizes tuple pairs into lists.
    observed = json.loads(json.dumps(signature(report)))
    expected = manifest.get("expected")
    version_matches = manifest.get("engine_version") == __version__
    status = "error" if expected != observed else "pass" if version_matches else "unknown"
    return {"schema_version": 1, "status": status, "hashes_verified": len(listed),
            "recorded_engine_version": manifest.get("engine_version"), "current_engine_version": __version__,
            "engine_version_matches": version_matches,
            "version_notice": None if version_matches else "The diagnostic engine differs or was not recorded; changed findings may reflect a check-version change.",
            "expected": expected, "observed": observed, "scope": "Diagnostic reproduction, not policy or physical replay."}
