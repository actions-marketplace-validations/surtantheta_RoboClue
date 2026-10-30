"""Optional offline tests against explicitly downloaded pinned public artifacts."""

import json
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError, load_checkpoint, load_dataset
from policy_doctor.dataset import scan_dataset
from policy_doctor.diagnostics import audit
from policy_doctor.trace import trace_actions

ROOT = Path(__file__).parents[1] / ".validation/pinned_public"
pytestmark = pytest.mark.skipif(not (ROOT / "provenance.json").exists(), reason="Run scripts/fetch_validation_artifacts.py .validation/pinned_public first")


def test_native_public_checkpoint_metadata():
    cp = load_checkpoint(ROOT / "smolvla_base")
    assert cp.config["type"] == "smolvla"
    assert cp.features["action"]["shape"] == [6]
    assert len(cp.artifacts) == 5
    assert cp.post[0]["stats"]  # Actual robot-namespaced stats, not synthetic flat action stats.


def test_public_base_requires_effective_runtime_export():
    report = audit(str(ROOT / "smolvla_base"))
    codes = {f["code"] for f in report["findings"]}
    assert {"processor.feature_unbound", "normalization.statistics", "deployment.missing"} <= codes
    # These are static export discrepancies, not a real-world policy failure result.
    stat_findings = [f for f in report["findings"] if f["code"] == "normalization.statistics"]
    assert all(f["evidence"][0]["state_artifact"]["sha256"] for f in stat_findings)


def test_complete_public_parquet_scan():
    ds = load_dataset(ROOT / "pusht")
    result = scan_dataset(str(ROOT / "pusht"), max_rows=100000)
    assert result["status"] == "pass"
    assert result["coverage"]["rows_scanned"] == ds.config["total_frames"] == 25650
    assert result["coverage"]["complete_parquet_scan"]
    assert result["coverage"]["videos_checked"] is False


def test_public_base_does_not_get_a_fabricated_trace(tmp_path):
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"model_output": [[0] * 6]}))
    with pytest.raises(InputError, match="normalization.statistics"):
        trace_actions(str(ROOT / "smolvla_base"), str(sample))
