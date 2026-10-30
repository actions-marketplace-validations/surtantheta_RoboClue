"""Whole declared command/asset/remedy proposal; no serving/model execution."""
import base64
import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.openpi_intake import OpenPIStatisticsSnapshot, resolve_openpi_statistics
from policy_doctor.openpi_readiness import compare_openpi_readiness

FIXTURES = Path(__file__).parent / "fixtures"
ASSET = "gs://openpi-assets/checkpoints/pi0_base/assets/droid/norm_stats.json"
BASE = "uv run scripts/serve_policy.py policy:checkpoint --policy.dir=gs://openpi-assets/checkpoints/pi0_base "
REFERENCE = BASE + "--policy.config=pi0_droid"
CURRENT = BASE + "--policy.config=pi05_droid"


@pytest.fixture(scope="module")
def native_source(tmp_path_factory):
    source = tmp_path_factory.mktemp("original-openpi-source") / "source"
    shutil.copytree(FIXTURES / "openpi_source_profile_215abfb", source)
    for retained in source.rglob("*.txt"):
        retained.rename(retained.with_suffix(""))
    return source


def snapshot(document=None, *, source=ASSET, generation="1748456810418896", intake_status="pass"):
    """Original captured body by default; changed documents/metadata are GENERATED test controls."""
    folder = FIXTURES / "openpi_original_20261024"
    raw = (folder / "norm_stats.json").read_bytes() if document is None else json.dumps(document).encode()
    metadata = json.loads((folder / "object_metadata.json").read_bytes())
    metadata.update(name=source.removeprefix("gs://openpi-assets/"), generation=generation,
                    size=str(len(raw)), md5Hash=base64.b64encode(hashlib.md5(raw).digest()).decode())
    metadata_raw = json.dumps(metadata).encode()
    provenance = {"source": source, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
                  "content_identity_verified": True, "bucket": "openpi-assets", "object": metadata["name"],
                  "generation": generation, "declared_md5_base64": metadata["md5Hash"],
                  "declared_size_bytes": len(raw), "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
                  "historical_training_authority": "not_authenticated"}
    return OpenPIStatisticsSnapshot(intake_status, json.loads(raw), raw, metadata, metadata_raw,
                                    provenance, {}, {}, ("Generated intake interpretation note",) if intake_status == "unknown" else ())


def compare(native_source, current=None, reference=None, *, intent=True, current_command=CURRENT, reference_command=REFERENCE):
    return compare_openpi_readiness(current_command, reference_command, source_root=native_source,
                                    current_statistics=snapshot() if current is None else current,
                                    reference_statistics=snapshot() if reference is None else reference,
                                    restore_reference_intent=intent)


def test_original_command_asset_contradiction_and_exact_authorized_replay(native_source):
    stats = snapshot()
    assert hashlib.sha256(stats.raw).hexdigest() == "c6f9a4794b1df4901e618db3d3b5e02126dcc7c8d95b8f9735b5bc9ed75b95c9"
    before = copy.deepcopy(stats.document)
    report = compare(native_source, stats, stats)
    assert report["status"] == report["current"]["normalization"]["status"] == "error"
    assert report["reference"]["normalization"]["status"] == "pass"
    assert report["current"]["normalization"]["missing_bound_roles"] == ["actions", "state"]
    remedy = report["correction"]
    assert remedy["command"] == REFERENCE
    assert remedy["targeted_requirement_status"] == "pass" and remedy["overall_after_correction"] == "unknown"
    assert remedy["replay"]["resolution"]["model"]["model_type"] == "pi0"
    assert remedy["preserved_statistics"]["generation"] == "1748456810418896"
    assert remedy["preserved_statistics"]["sha256"] == hashlib.sha256(stats.raw).hexdigest()
    assert {item["field"] for item in remedy["changes"]} == {
        "model.model_type", "model.action_horizon", "normalization.use_quantile_norm"}
    assert stats.document == before and stats.document["norm_stats"]["actions"]["q01"] is None
    assert report["unknowns"]["checkpoint_weights_and_model_architecture"]
    assert "unbound" in report["unknowns"]["actual_command_directory"]
    assert "unproved" in report["unknowns"]["final_native_DroidOutputs_composition"]
    assert "not performed" in report["current"]["normalization"]["native_execution"]


def test_no_reference_intent_never_recommends_switching_mode(native_source):
    report = compare(native_source, intent=False)
    assert report["status"] == "error" and report["correction"] is None
    assert "unconditionally" in report["correction_reason"]


def test_same_healthy_commands_have_no_remedy_or_overall_certification(native_source):
    report = compare(native_source, current_command=REFERENCE)
    assert report["status"] == "unknown" and report["correction"] is None
    assert report["current"]["normalization"]["status"] == "pass"


def test_binding_disagreement_never_proposes_same_name_alias(native_source):
    for other in [snapshot(generation="1748456810418897"),
                  snapshot(source="gs://openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json")]:
        report = compare(native_source, reference=other)
        assert report["correction"] is None
    assert compare(native_source, reference=snapshot(generation="1748456810418897"))["current"]["normalization"]["status"] == "error"


@pytest.mark.parametrize("fault", ["sha", "raw", "document", "metadata", "missing"])
def test_incomplete_or_contradictory_original_evidence_stays_unknown(native_source, fault):
    stats = snapshot()
    if fault == "sha":
        stats = replace(stats, provenance={**stats.provenance, "sha256": "0" * 64})
    elif fault == "raw":
        stats = replace(stats, raw=stats.raw + b" ")
    elif fault == "document":
        stats = replace(stats, document={"norm_stats": {}})
    elif fault == "metadata":
        stats = replace(stats, metadata_raw=b"{}")
    else:
        stats = replace(stats, raw=None, document=None)
    report = compare(native_source, current=stats)
    assert report["current"]["normalization"]["status"] == "unknown"
    assert report["correction"] is None


@pytest.mark.parametrize("shape", ["scalar", "matrix", "missing_mean"])
def test_unmodeled_shapes_are_unknown_without_inventing_runtime_errors(native_source, shape):
    doc = copy.deepcopy(snapshot().document)
    values = doc["norm_stats"]["state"]
    if shape == "scalar":
        values["mean"], values["std"] = 0., 1.
    elif shape == "matrix":
        values["mean"], values["std"] = [[0.] * 32], [[1.] * 32]
    else:
        del values["mean"]
    report = compare(native_source, current=snapshot(doc), reference=snapshot(doc))
    assert report["current"]["normalization"]["status"] == "unknown"
    assert report["correction"] is None


@pytest.mark.parametrize("shape", ["aux_one", "state_one", "unequal"])
def test_global_constructor_requirement_precedes_numeric_shape_coverage(native_source, shape):
    doc = copy.deepcopy(snapshot().document)
    if shape == "aux_one":
        doc["norm_stats"]["unused_role"] = {"mean": [0.], "std": [1.], "q01": None, "q99": None}
    elif shape == "state_one":
        doc["norm_stats"]["state"]["mean"], doc["norm_stats"]["state"]["std"] = [0.], [1.]
    else:
        doc["norm_stats"]["state"]["std"] = [1.] * 31
    report = compare(native_source, snapshot(doc), snapshot(doc))
    assert report["current"]["normalization"]["status"] == "error"
    assert report["current"]["normalization"]["code"] == "openpi.normalization.missing_quantile_bounds"
    assert report["reference"]["normalization"]["status"] == "unknown"
    assert report["correction"] is None


def test_exact_local_bytes_support_conditional_diagnosis_without_cloud_or_history(native_source, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    asset = checkpoint / "assets/droid/norm_stats.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(snapshot().raw)
    stats = resolve_openpi_statistics(str(asset), read_roots=[tmp_path])
    current = f"uv run scripts/serve_policy.py policy:checkpoint --policy.dir={checkpoint} --policy.config=pi05_droid"
    reference = current.replace("pi05_droid", "pi0_droid")
    report = compare(native_source, stats, stats, current_command=current, reference_command=reference)
    binding = report["current"]["statistics_binding"]
    assert binding["status"] == "pass" and binding["identity_kind"] == "local_content_snapshot"
    assert binding["sha256"] == hashlib.sha256(stats.raw).hexdigest() and "generation" not in binding
    assert report["current"]["normalization"]["status"] == "error"
    assert report["reference"]["normalization"]["status"] == "pass"
    assert report["correction"] is None and "versioned public object" in report["correction_reason"]
    assert report["unknowns"]["historical_training_mode"]
    # A same-byte file at another path does not bind this command's selection.
    wrong_source = replace(stats, provenance={**stats.provenance, "source": str(tmp_path / "other.json")})
    report = compare(native_source, wrong_source, stats, current_command=current, reference_command=reference)
    assert report["current"]["normalization"]["status"] == "unknown"


def test_native_extra_fields_do_not_become_a_new_strict_schema(native_source):
    doc = copy.deepcopy(snapshot().document)
    doc["description"] = "GENERATED root metadata ignored by native Pydantic model"
    doc["norm_stats"]["state"]["annotation"] = {"GENERATED": "native-extra ignored field"}
    stats = snapshot(doc, intake_status="unknown")
    report = compare(native_source, stats, stats)
    assert report["current"]["normalization"]["status"] == "error"
    assert report["reference"]["normalization"]["status"] == "pass"
    assert report["correction"] is not None
    assert report["reference"]["intake_notes"]


def test_extra_role_uses_constructor_requirement_but_not_a_fake_role_error(native_source):
    doc = copy.deepcopy(snapshot().document)
    doc["norm_stats"]["auxiliary"] = copy.deepcopy(doc["norm_stats"]["state"])
    report = compare(native_source, snapshot(doc), snapshot(doc))
    assert "auxiliary" in report["current"]["normalization"]["missing_bound_roles"]
    assert report["reference"]["normalization"]["status"] == "unknown"
    assert "inverse selection is strict" in report["reference"]["normalization"]["message"]
    assert report["correction"] is None


@pytest.mark.parametrize("roles", [{}, {"actions"}])
def test_empty_or_missing_selector_not_declared_a_native_failure(native_source, roles):
    doc = copy.deepcopy(snapshot().document)
    doc["norm_stats"] = {} if not roles else {key: doc["norm_stats"][key] for key in roles}
    report = compare(native_source, snapshot(doc), snapshot(doc), current_command=REFERENCE)
    assert report["current"]["normalization"]["status"] == "unknown"
    assert report["correction"] is None


def test_invalid_unused_quantile_representation_cannot_get_typed_loader_clearance(native_source):
    doc = copy.deepcopy(snapshot().document)
    doc["norm_stats"]["state"]["q01"] = "not a native NDArray"
    report = compare(native_source, snapshot(doc), snapshot(doc), current_command=REFERENCE)
    assert report["current"]["normalization"]["status"] == "unknown"


def test_different_public_default_asset_and_unknown_command_never_alias_original(native_source):
    report = compare(native_source, current_command="uv run scripts/serve_policy.py --env=DROID")
    assert report["current"]["statistics_binding"]["status"] == "unknown"
    assert report["correction"] is None
    report = compare(native_source, current_command=CURRENT + " --policy.api_key=FAKE_SECRET_VALUE")
    assert report["correction"] is None and "FAKE_SECRET_VALUE" not in json.dumps(report)


def test_reference_intent_is_strict_boolean(native_source):
    with pytest.raises(InputError, match="strict boolean"):
        compare(native_source, intent="true")
