#!/usr/bin/env python3
"""Prepare pinned original artifacts and an explicit CPU runtime; run no diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "groot_public_example"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


class PreparationError(Exception):
    pass


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise PreparationError("Manifest paths must be bounded relative paths")
    return Path(*path.parts)


def download(record: dict, target: Path, *, opener=urllib.request.urlopen) -> dict:
    """Stream at most the declared byte count plus one; publish only verified bytes."""
    size = record["size_bytes"]
    if type(size) is not int or not 0 <= size <= MAX_ARTIFACT_BYTES:
        raise PreparationError("Artifact exceeds the fixed download limit")
    if not record["url"].startswith("https://"):
        raise PreparationError("Only original HTTPS artifact URLs are supported")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    started = time.monotonic()
    digest = hashlib.sha256()
    received = 0
    try:
        request = urllib.request.Request(record["url"], headers={"User-Agent": "policy-doctor-public-example/1"})
        with opener(request, timeout=45) as response, partial.open("xb") as output:
            if not response.geturl().startswith("https://"):
                raise PreparationError("Refusing non-HTTPS artifact redirect")
            while True:
                block = response.read(min(65536, size - received + 1))
                if not block:
                    break
                received += len(block)
                if received > size:
                    raise PreparationError("Artifact exceeds its exact declared size")
                output.write(block)
                digest.update(block)
        if received != size or digest.hexdigest() != record["sha256"]:
            raise PreparationError("Original artifact size or SHA-256 mismatch")
        os.replace(partial, target)
        return {"url": record["url"], "bytes": received, "sha256": digest.hexdigest(),
                "seconds": time.monotonic() - started, "attempts": 1}
    finally:
        partial.unlink(missing_ok=True)


def extract_member(root: Path, record: dict) -> None:
    """Copy one bounded, original wheel member without installing or executing it."""
    with zipfile.ZipFile(root / relative_path(record["archive"])) as archive:
        matches = [item for item in archive.infolist() if item.filename == record["member"]]
        if len(matches) != 1 or matches[0].file_size != record["size_bytes"]:
            raise PreparationError("Missing, duplicated or changed original wheel member")
        with archive.open(matches[0]) as source:
            data = source.read(record["size_bytes"] + 1)
    if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise PreparationError("Original wheel member hash mismatch")
    for value in record["destinations"]:
        target = root / relative_path(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def run(command: list[str], root: Path, label: str, receipt: dict) -> None:
    started = time.monotonic()
    log = root / "setup-logs" / (label + ".log")
    log.parent.mkdir(exist_ok=True)
    with log.open("wb") as output:
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=600, check=False)
    receipt["commands"].append({"argv": command, "returncode": completed.returncode,
                                "seconds": time.monotonic() - started,
                                "log": log.relative_to(root).as_posix()})
    if completed.returncode:
        raise PreparationError(f"{label} failed; inspect {log}")


def prepare(destination: Path, *, uv: Path | None, python: Path | None, wheel: Path | None) -> dict:
    # Deliberately refuse reuse: partial failures are reviewable, never a success cache.
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    started = time.monotonic()
    manifest_bytes = (EXAMPLE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = {"kind": "public_example_preparation", "status": "incomplete", "downloads": [],
               "commands": [], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "dataset_license": manifest["dataset_license"], "diagnosis_executed": False,
               "setup_cost": "All downloads, source selection, runtime installation and this helper's development are preparation work."}
    try:
        (root / "manifest.json").write_bytes(manifest_bytes)
        for number, record in enumerate(manifest["downloads"], 1):
            targets = [root / relative_path(value) for value in record["destinations"]]
            print(f"Downloading {number}/{len(manifest['downloads'])}: {record['destinations'][0]}", flush=True)
            item = download(record, targets[0])
            item["destinations"] = record["destinations"]
            receipt["downloads"].append(item)
            for target in targets[1:]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(targets[0], target)
        for member in manifest["wheel_members"]:
            extract_member(root, member)
        shutil.copyfile(EXAMPLE / "input-contract.json", root / "input-contract.json")
        if uv:
            run([str(uv), "venv", "--python", "3.12", str(root / "native-env")], root, "native-create", receipt)
            python = root / "native-env" / "bin" / "python"
            run([str(uv), "pip", "install", "--python", str(python), *manifest["native_requirements"]], root, "native-install", receipt)
        assert python is not None
        version_code = "import importlib.metadata,json; print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()},sort_keys=True))"
        run([str(python), "-I", "-c", version_code], root, "native-versions", receipt)
        actual = json.loads((root / "setup-logs/native-versions.log").read_text())
        normalized = {key.lower().replace("_", "-"): value for key, value in actual.items()}
        for requirement in manifest["native_requirements"]:
            name, version = requirement.split("==")
            if normalized.get(name) != version:
                raise PreparationError(f"Native runtime requires {requirement}")
        receipt["native_distributions"] = actual
        runtime = {"providers": {"groot_adapter": {
            "python": str(python), "source_root": str(root / "source/candidate"),
            "reference_source_root": str(root / "source/reference"),
            "groot_source_root": str(root / "source/groot")}}}
        (root / "native-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        if wheel:
            assert uv is not None
            receipt["product_wheel"] = {"name": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
            run([str(uv), "venv", "--python", "3.12", str(root / "product-env")], root, "product-create", receipt)
            run([str(uv), "pip", "install", "--python", str(root / "product-env/bin/python"), str(wheel) + "[mcp,dataset]"], root, "product-install", receipt)
            run([str(root / "product-env/bin/python"), "-I", "-c", version_code], root, "product-versions", receipt)
        receipt["status"] = "prepared"
        return receipt
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        receipt["artifact_download_bytes"] = sum(item["bytes"] for item in receipt["downloads"])
        (root / "preparation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New local directory; existing paths are refused")
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--uv", type=Path, help="Explicit uv executable: create fresh isolated CPU runtime")
    runtime.add_argument("--python", type=Path, help="Explicit existing native interpreter with the exact documented dependencies")
    parser.add_argument("--wheel", type=Path, help="Optional local product wheel; requires --uv and installs a separate product environment")
    args = parser.parse_args()
    if args.wheel and not args.uv:
        parser.error("--wheel requires --uv")
    for name in ("uv", "python", "wheel"):
        value = getattr(args, name)
        if value is not None:
            # Preserve a virtualenv interpreter's symlink location: resolving it
            # to the base executable would silently lose that environment.
            value = value.absolute()
            if not value.is_file():
                parser.error(f"--{name} must name a file")
            setattr(args, name, value)
    try:
        receipt = prepare(args.destination, uv=args.uv, python=args.python, wheel=args.wheel)
    except (OSError, PreparationError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt["status"], "destination": str(args.destination.resolve()),
                      "artifact_download_bytes": receipt["artifact_download_bytes"], "seconds": receipt["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
