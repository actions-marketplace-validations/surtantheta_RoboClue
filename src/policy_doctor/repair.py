"""Compare an original and proposed deployment contract against unchanged inputs."""

from collections import Counter
from pathlib import Path

from .artifacts import InputError, sha256
from .compare import compare_configs
from .diagnostics import audit


def check_repair(checkpoint: str, before: str, candidate: str, dataset: str | None = None) -> dict:
    original = audit(checkpoint, dataset, before)
    proposed = audit(checkpoint, dataset, candidate)
    def invariant_artifacts(report):
        return sorted((a["role"], a["path"], a["sha256"]) for a in report["artifacts"] if a["role"] != "deployment_contract")

    if invariant_artifacts(original) != invariant_artifacts(proposed):
        raise InputError("Checkpoint/dataset artifacts changed between audits; cannot attribute differences to the candidate contract")
    declarations = [next(a for a in report["artifacts"] if a["role"] == "deployment_contract")
                    for report in (original, proposed)]
    for artifact in declarations:
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise InputError("Deployment contract changed after its audit; retry with stable files")
    changes = compare_configs(before, candidate)["changes"]
    for artifact in declarations:
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise InputError("Deployment contract changed during comparison; retry with stable files")
    old = Counter(f["code"] for f in original["findings"] if f["status"] == "error")
    new = Counter(f["code"] for f in proposed["findings"] if f["status"] == "error")
    unknown = Counter(f["code"] for f in proposed["findings"] if f["status"] == "unknown")
    return {"schema_version": 1, "status": proposed["status"],
            "before_status": original["status"], "candidate_status": proposed["status"],
            "resolved_error_counts": dict(old - new), "introduced_error_counts": dict(new - old),
            "remaining_error_counts": dict(new), "candidate_unknown_counts": dict(unknown),
            "configuration_changes": changes,
            "before_counts": original["counts"], "candidate_counts": proposed["counts"],
            "candidate_findings": proposed["findings"], "artifacts": proposed["artifacts"],
            "scope": "Candidate contract re-audit with fixed checkpoint/dataset artifacts. Clearing declared interface errors does not establish robot behavior or prove a causal repair."}
