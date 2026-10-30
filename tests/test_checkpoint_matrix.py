"""Optional pinned metadata compatibility checks across real policy families."""

import json
from pathlib import Path

import numpy as np
import pytest

from policy_doctor.artifacts import load_checkpoint
from policy_doctor.diagnostics import audit
from policy_doctor.trace import trace_actions

ROOT = Path(__file__).parents[1] / ".validation/checkpoint_matrix"
pytestmark = pytest.mark.skipif(not (ROOT / "provenance.json").exists(), reason="Run scripts/fetch_checkpoint_matrix.py .validation/checkpoint_matrix")


@pytest.mark.parametrize("name,policy,dimension", [
    ("act_aloha_sim_transfer_cube_human", "act", 14),
    ("diffusion_pusht", "diffusion", 2),
    ("smolvla_libero", "smolvla", 7),
])
def test_policy_family_metadata(name, policy, dimension):
    snapshot = load_checkpoint(ROOT / name)
    assert snapshot.config["type"] == policy
    assert snapshot.features["action"]["shape"] == [dimension]


@pytest.mark.parametrize("name", ["act_aloha_sim_transfer_cube_human", "diffusion_pusht"])
def test_legacy_normalization_is_unknown_without_reading_weights(name):
    report = audit(str(ROOT / name))
    assert report["counts"]["error"] == 0
    assert any(f["code"] == "processor.missing" and f["status"] == "unknown" for f in report["findings"])
    assert len(report["artifacts"]) == 1


def test_actual_finetuned_processors_have_valid_statistics():
    report = audit(str(ROOT / "smolvla_libero"))
    assert report["counts"]["error"] == 0
    assert sum(f["code"] == "normalization.statistics" and f["status"] == "pass" for f in report["findings"]) == 3


def test_actual_action_statistics_decode_a_saved_zero_output(tmp_path):
    root = ROOT / "smolvla_libero"
    snapshot = load_checkpoint(root)
    stats = snapshot.post[0]["stats"]["action"]
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"model_output": [[0.] * 7]}))
    result = trace_actions(str(root), str(sample))
    np.testing.assert_allclose(result["commands"][0], stats["mean"], rtol=1e-12)
    # This is a supplied numerical example with real checkpoint stats, not a model prediction or rollout.
    assert result["status"] == "pass"
