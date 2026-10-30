"""Save scoped compatibility and zero-output decoding evidence for pinned artifacts."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from policy_doctor.artifacts import load_checkpoint
from policy_doctor.diagnostics import audit
from policy_doctor.trace import trace_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    provenance = json.loads((args.root / "provenance.json").read_text())
    cases = []
    for entry in provenance:
        name = entry["repository"].split("/")[-1]
        checkpoint = args.root / name
        snapshot = load_checkpoint(checkpoint)
        report = audit(str(checkpoint))
        result = {"repository": entry["repository"], "revision": entry["revision"],
                  "policy_type": snapshot.config["type"], "action_dimension": snapshot.features["action"]["shape"],
                  "audit_counts": report["counts"], "unknown_codes": sorted({f["code"] for f in report["findings"] if f["status"] == "unknown"}),
                  "weights_read": False}
        if name == "smolvla_libero":
            with tempfile.TemporaryDirectory() as directory:
                sample = Path(directory) / "sample.json"
                sample.write_text(json.dumps({"model_output": [[0.] * 7]}))
                trace = trace_actions(str(checkpoint), str(sample))
                expected = snapshot.post[0]["stats"]["action"]["mean"]
                np.testing.assert_allclose(trace["commands"][0], expected, rtol=1e-12)
                result["supplied_zero_output_trace"] = {"status": trace["status"], "max_difference_from_saved_mean": 0.0}
        assert report["counts"]["error"] == 0, report
        cases.append(result)
    print(json.dumps({"status": "pass", "cases": cases,
                      "scope": "Pinned metadata compatibility and supplied numerical action output; no model inference, runtime contract or physical evaluation."}, indent=2))


if __name__ == "__main__":
    main()
