"""Exercise installed downstream gates from an unrelated consumer directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(work):
    from policy_doctor import automation
    from policy_doctor.product_demo import run_demo

    origin = Path(automation.__file__).resolve()
    if "site-packages" not in origin.parts:
        raise RuntimeError("This smoke must use a freshly installed package outside the source tree.")
    work.mkdir()
    consumer = work / "consumer"
    consumer.mkdir()
    run_demo(consumer / "demo")
    retained = {name: hashlib.sha256((consumer / "demo" / name).read_bytes()).hexdigest()
                for name in ("source.json", "broken.json", "reference-contract.json", "criterion.sha256")}
    reference = {"camera_order": ["front", "wrist"], "action_type": "relative"}
    (consumer / "reviewed.json").write_text(json.dumps(reference))
    (consumer / "candidate.json").write_text(json.dumps(dict(reference, camera_order=["wrist", "front"])))
    calls = []

    def call(name, check, expected, *, precommit=False, filenames=None):
        manifest = consumer / "checks.json"
        manifest.write_text(json.dumps({"schema_version": 1, "checks": [check]}))
        argv = [sys.executable, "-I", "-B", "-m", "roboclue.automation", "pre-commit" if precommit else "run",
                "--manifest", "checks.json", "--workspace", str(consumer), "--output", str(work / name)]
        if precommit:
            argv += ["--", *(filenames or [])]
        process = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=90)
        (work / (name + ".stdout")).write_text(process.stdout)
        (work / (name + ".stderr")).write_text(process.stderr)
        assert process.returncode == expected, (name, process.returncode, expected, process.stdout, process.stderr)
        if name in ("config-fault", "precommit-fault"):
            assert "Problem:" in process.stdout and "camera_order" in process.stdout and "Next:" in process.stdout
        calls.append({"name": name, "exit_code": process.returncode, "argv": argv})

    config = {"id": "cameras", "check": "config", "artifact": "candidate.json", "reference": "reviewed.json"}
    call("config-fault", config, 1)
    call("precommit-fault", config, 1, precommit=True, filenames=["candidate.json"])
    call("precommit-unselected", config, 0, precommit=True, filenames=["unrelated.json"])
    call("precommit-option-filename", config, 0, precommit=True, filenames=["--manifest", "unrelated.json"])
    (consumer / "candidate.json").write_bytes((consumer / "reviewed.json").read_bytes())
    call("config-corrected", config, 0)
    call("precommit-corrected", config, 0, precommit=True, filenames=["candidate.json"])
    for phase, expected in (("broken", 1), ("corrected", 0), ("unknown", 3)):
        call("targets-" + phase, {"id": "targets", "check": "training", "artifact": "demo/" + phase + ".json",
             "reference": "demo/reference-contract.json", "pin_file": "demo/criterion.sha256"}, expected)
    call("missing-consumer", {"id": "deploy", "check": "deployment-units", "artifact": "demo"}, 2)
    call("forbidden-shell", dict(config, shell="touch PWNED"), 2)
    call("dataset-threshold-bypass", {"id": "data", "check": "dataset", "artifact": "demo", "fail_on": "error"}, 2)
    assert not (consumer / "PWNED").exists() and not (work / "PWNED").exists()
    assert retained == {name: hashlib.sha256((consumer / "demo" / name).read_bytes()).hexdigest() for name in retained}
    result = {"status": "PASS_INSTALLED_DOWNSTREAM_AUTOMATION", "origin": str(origin), "calls": calls,
              "original_training_evidence_unchanged": True,
              "scope": "Isolated consumer and actual installed module execution with generated controls; not hosted GitHub Actions, native training, human completion or physical success."}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir")
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    if output.exists() or not output.parent.is_dir():
        raise RuntimeError("Choose a new output file with an existing parent.")
    if args.work_dir:
        result = run(Path(args.work_dir).absolute())
    else:
        with tempfile.TemporaryDirectory(prefix="roboclue-automation-") as temp:
            result = run(Path(temp) / "work")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "actual_installed_calls": len(result["calls"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
