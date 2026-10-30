"""Saved native binding, explicit policy order and physical provenance are separate."""

import copy
import hashlib
import json

import numpy as np
import pytest
from layout_test_support import layout_input

from policy_doctor.artifacts import InputError


def fixture_data(*, ids=(0, 1), identity=True):
    metadata, _, sources = layout_input(identity=identity)
    metadata["runtime"].update(
        sensor_body_names=["elbow", "wrist"],
        requested_body_names=["elbow", "wrist"],
        resolved_body_ids=list(ids),
        preserve_order=True,
    )
    metadata["processing"].update(scale=2.0, modifiers=None, history=None)
    metadata["processing"].update(noise=None, delay=None)
    force = np.arange(12, dtype=np.float64).reshape(2, 2, 3) + 1
    torque = force + 100
    raw = np.concatenate((force[:, ids, :], torque[:, ids, :]), axis=-1).reshape(2, -1)
    arrays = {
        "source_force": force,
        "source_torque": torque,
        "raw_term_output": raw,
        "processed_term_output": raw * 2,
    }
    return metadata, arrays, sources


def capture(path, **kwargs):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data(**kwargs)
    write_layout_capture(path, metadata, arrays, sources=sources)
    return path


def _rehash(path):
    manifest = json.loads((path / "capture.json").read_text())
    for entry in manifest["files"]:
        entry["sha256"] = hashlib.sha256((path / entry["path"]).read_bytes()).hexdigest()
    (path / "capture.json").write_text(json.dumps(manifest))
    (path / "READY.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_sha256": hashlib.sha256((path / "capture.json").read_bytes()).hexdigest(),
            }
        )
    )


def test_real_pinned_sources_derive_native_components_and_replay(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture")
    result = audit_layout_capture(
        path,
        expected_names=["elbow", "wrist"],
        expected_frame="incoming_joint_frame",
        expected_quantity="incoming_joint_wrench",
    )
    assert result["status"] == "pass"
    assert result["raw_binding"]["status"] == result["processed_binding"]["status"] == "pass"
    assert result["layout"]["body_names"] == ["elbow", "wrist"]
    assert result["layout"]["component_names"][:6] == [
        f"elbow.{suffix}" for suffix in ("force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z")
    ]
    assert result["layout"]["units"] == {"force": "N", "torque": "N*m"}
    assert result["unverified"]["physical_sample_identity"] == "unknown"
    assert result["unverified"]["policy_concatenation"] == "unknown"


def test_order_contract_error_is_not_native_backend_bug(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture", ids=(1, 0))
    result = audit_layout_capture(path, expected_names=["elbow", "wrist"])
    assert result["status"] == "error" and result["raw_binding"]["status"] == "pass"
    assert result["layout"]["body_names"] == ["wrist", "elbow"]
    assert any(row["code"] == "layout.expected_order" for row in result["findings"])
    assert "backend bug" in result["scope"]


def test_raw_binding_error_has_exact_saved_coordinate(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    arrays["raw_term_output"][1, 7] += 9
    report = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert report["status"] == "error"
    finding = next(row for row in report["findings"] if row["code"] == "layout.raw_replay")
    assert finding["evidence"][0]["array"] == "raw_term_output"
    assert finding["evidence"][0]["index"] == [1, 7]
    assert finding["evidence"][0]["sha256"]


def test_missing_processed_stage_does_not_erase_order_error(tmp_path):
    from policy_doctor.layouts import audit_layout_capture, write_layout_capture

    metadata, arrays, sources = fixture_data(ids=(1, 0))
    arrays.pop("processed_term_output")
    path = tmp_path / "capture"
    write_layout_capture(path, metadata, arrays, sources=sources)
    result = audit_layout_capture(path, expected_names=["elbow", "wrist"])
    assert result["status"] == "error" and result["processed_binding"]["status"] == "unknown"


@pytest.mark.parametrize(
    "stage,value", [("modifiers", ["custom"]), ("clip", [-1, 1]), ("history", 3), ("scale", None)]
)
def test_unsupported_processing_keeps_raw_binding_available(tmp_path, stage, value):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["processing"][stage] = value
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["raw_binding"]["status"] == "pass"
    assert result["processed_binding"]["status"] == "unknown"


def test_processed_scale_contradiction_separate_from_raw(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    arrays["processed_term_output"][0, 0] += 1
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["raw_binding"]["status"] == "pass"
    assert result["processed_binding"]["status"] == "error"


@pytest.mark.parametrize("missing", ["term", "sensor_class", "sensor_mro", "revision"])
def test_missing_native_source_identity_is_unknown(tmp_path, missing):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["source"].pop(missing)
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["status"] == "unknown"


def test_changed_valid_source_is_unknown_not_authenticated(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    sources["observations.py"] += b"\n# changed source\n"
    digest = hashlib.sha256(sources["observations.py"]).hexdigest()
    metadata["source"]["files"]["observations.py"]["sha256"] = digest
    metadata["source"]["term"]["sha256"] = digest
    assert (
        write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)["status"] == "unknown"
    )


@pytest.mark.parametrize(
    "where",
    [
        "body_id_bool",
        "body_id_oob",
        "duplicate_ids",
        "duplicate_names",
        "hand_names",
        "nonfinite",
        "bad_source_hash",
        "array_shape",
        "object",
        "scale_bool",
    ],
)
def test_malformed_capture_refuses_without_publication(tmp_path, where):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    if where == "body_id_bool":
        metadata["runtime"]["resolved_body_ids"] = [False, 1]
    elif where == "body_id_oob":
        metadata["runtime"]["resolved_body_ids"] = [0, 2]
    elif where == "duplicate_ids":
        metadata["runtime"]["resolved_body_ids"] = [0, 0]
    elif where == "duplicate_names":
        metadata["runtime"]["sensor_body_names"] = ["elbow", "elbow"]
    elif where == "hand_names":
        metadata["runtime"]["component_names"] = ["trust me"]
    elif where == "nonfinite":
        arrays["source_force"][0, 0, 0] = np.nan
    elif where == "bad_source_hash":
        metadata["source"]["term"]["sha256"] = "0" * 64
    elif where == "array_shape":
        arrays["raw_term_output"] = np.ones((2, 13))
    elif where == "object":
        arrays["source_force"] = np.array([object()], dtype=object)
    else:
        metadata["processing"]["scale"] = True
    path = tmp_path / "capture"
    with pytest.raises(InputError):
        write_layout_capture(path, metadata, arrays, sources=sources)
    assert not path.exists()


def test_manifest_payload_and_commit_tampering_refuse(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture")
    (path / "arrays.npz").write_bytes((path / "arrays.npz").read_bytes() + b"tamper")
    with pytest.raises(InputError):
        audit_layout_capture(path)


def test_existing_destination_preserved_and_input_snapshots_unchanged(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    previous_metadata, previous_arrays = (
        copy.deepcopy(metadata),
        {key: value.copy() for key, value in arrays.items()},
    )
    path = tmp_path / "existing"
    path.mkdir()
    (path / "keep").write_text("keep")
    with pytest.raises(InputError):
        write_layout_capture(path, metadata, arrays, sources=sources)
    assert (path / "keep").read_text() == "keep" and metadata == previous_metadata
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], previous_arrays[key])


def test_moved_capture_originals_absent_still_audits(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture")
    moved = tmp_path / "moved"
    path.rename(moved)
    assert audit_layout_capture(moved)["status"] == "pass"


def test_symlink_payload_and_unlisted_file_refuse(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture")
    outside = tmp_path / "outside.npz"
    (path / "arrays.npz").rename(outside)
    (path / "arrays.npz").symlink_to(outside)
    with pytest.raises(InputError):
        audit_layout_capture(path)


def test_missing_context_unknown_without_erasing_numeric_observation(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata.pop("provenance")
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["status"] == "unknown"


def test_permutation_is_layout_change_with_equal_aligned_named_values(tmp_path):
    from policy_doctor.layouts import compare_layout_captures

    reference = capture(tmp_path / "reference")
    candidate = capture(tmp_path / "candidate", ids=(1, 0))
    result = compare_layout_captures(reference, candidate)
    assert result["status"] == "different"
    assert not result["ordered_layout_equal"] and result["named_values_equal"]
    assert not result["observed_named_value_difference"]
    assert result["classification"] == "permutation_with_equal_named_values"


def test_identical_raw_values_with_swapped_names_are_not_same_layout(tmp_path):
    from policy_doctor.layouts import compare_layout_captures, write_layout_capture

    metadata, arrays, sources = fixture_data()
    arrays = {key: np.zeros_like(value) for key, value in arrays.items()}
    a, b = tmp_path / "a", tmp_path / "b"
    write_layout_capture(a, metadata, arrays, sources=sources)
    metadata["runtime"]["sensor_body_names"] = ["wrist", "elbow"]
    write_layout_capture(b, metadata, arrays, sources=sources)
    result = compare_layout_captures(a, b)
    assert result["raw_values_equal"] and not result["ordered_layout_equal"]
    assert result["status"] == "different"


def test_cross_capture_identity_missing_or_different_is_unknown(tmp_path):
    from policy_doctor.layouts import compare_layout_captures, write_layout_capture

    a = capture(tmp_path / "a")
    b = capture(tmp_path / "b", identity=False)
    result = compare_layout_captures(a, b)
    assert result["status"] == "unknown" and result["named_values_equal"] is None
    metadata, arrays, sources = fixture_data()
    metadata["runtime"]["sensor_identity"]["prim_path"] = "/Other"
    write_layout_capture(tmp_path / "c", metadata, arrays, sources=sources)
    assert compare_layout_captures(a, tmp_path / "c")["status"] == "unknown"


def test_expected_identity_is_namespace_contract_and_values_may_change(tmp_path):
    from policy_doctor.layouts import audit_layout_capture, write_layout_capture

    metadata, arrays, sources = fixture_data()
    arrays = {key: value + 10 for key, value in arrays.items()}
    arrays["processed_term_output"] = arrays["raw_term_output"] * 2
    path = tmp_path / "new"
    write_layout_capture(path, metadata, arrays, sources=sources)
    assert (
        audit_layout_capture(path, expected_identity=metadata["runtime"]["sensor_identity"])["status"]
        == "pass"
    )
    assert audit_layout_capture(path, expected_identity={"prim_path": "/Other"})["status"] == "error"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_names": [True]},
        {"expected_names": ["elbow", "elbow"]},
        {"expected_frame": True},
        {"expected_quantity": ""},
        {"expected_identity": {}},
        {"atol": True},
        {"rtol": -1},
    ],
)
def test_invalid_acceptance_and_tolerances_refuse(tmp_path, kwargs):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture")
    with pytest.raises(InputError):
        audit_layout_capture(path, **kwargs)


def test_bounded_values_and_source_bytes_refuse_before_publication(tmp_path, monkeypatch):
    from policy_doctor import layouts

    metadata, arrays, sources = fixture_data()
    monkeypatch.setattr(layouts, "MAX_VALUES", 1)
    with pytest.raises(InputError):
        layouts.write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize("where", ["module", "repository", "origin"])
def test_malformed_source_labels_refuse_rather_than_crash(tmp_path, where):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    if where == "module":
        metadata["source"]["sensor_class"]["module"] = ["unhashable"]
    elif where == "repository":
        metadata["source"]["repository"] = {"unhashable": True}
    else:
        metadata["provenance"]["origin"] = ["unhashable"]
    with pytest.raises(InputError):
        write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)


def test_unknown_sensor_cannot_borrow_base_frame_semantics(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["source"]["sensor_class"]["module"] = "custom.sensor"
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["raw_binding"]["status"] == "pass"
    assert result["status"] == "unknown" and result["layout"]["frame"] is None
    assert result["layout"]["quantity"] is None


def test_expected_identity_missing_is_unknown_but_local_order_available(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture", identity=False)
    assert audit_layout_capture(path, expected_names=["elbow", "wrist"])["status"] == "pass"
    assert audit_layout_capture(path, expected_identity={"prim_path": "/World/Robot"})["status"] == "unknown"


def test_same_read_bytes_guard_detects_capture_mutation(tmp_path, monkeypatch):
    from policy_doctor import layouts

    path = capture(tmp_path / "capture")
    original = layouts._report

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with (path / "arrays.npz").open("ab") as output:
            output.write(b"changed")
        return result

    monkeypatch.setattr(layouts, "_report", changed)
    with pytest.raises(InputError, match="changed"):
        layouts.audit_layout_capture(path)


def test_unlisted_payload_and_missing_commit_refuse(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture")
    extra = path / "extra.py"
    extra.write_text("never executed")
    with pytest.raises(InputError):
        audit_layout_capture(path)
    extra.unlink()
    (path / "READY.json").unlink()
    with pytest.raises(InputError):
        audit_layout_capture(path)


def test_writer_refuses_mutation_of_input_during_snapshot(tmp_path, monkeypatch):
    from policy_doctor import layouts

    metadata, arrays, sources = fixture_data()
    original = layouts.np.savez

    def mutation(*args, **kwargs):
        arrays["source_force"][0, 0, 0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(layouts.np, "savez", mutation)
    with pytest.raises(InputError, match="changed"):
        layouts.write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize("where", ["term", "sensor", "context"])
def test_unsupported_binding_cannot_turn_expected_order_or_identity_into_fault(tmp_path, where):
    from policy_doctor.layouts import audit_layout_capture, write_layout_capture

    metadata, arrays, sources = fixture_data()
    if where == "term":
        metadata["source"]["term"]["module"] = "custom.observations"
    elif where == "sensor":
        metadata["source"]["sensor_class"]["module"] = "custom.sensor"
    else:
        metadata["provenance"]["origin"] = "unknown"
    path = tmp_path / "capture"
    write_layout_capture(path, metadata, arrays, sources=sources)
    result = audit_layout_capture(
        path, expected_names=["wrist", "elbow"], expected_identity={"prim_path": "/Other"}
    )
    assert result["status"] == result["acceptance_contract"]["status"] == "unknown"
    assert not any(row["status"] == "error" for row in result["findings"])


def test_first_order_mismatch_has_actual_resolved_native_id(tmp_path):
    from policy_doctor.layouts import audit_layout_capture

    path = capture(tmp_path / "capture", ids=(1, 0))
    result = audit_layout_capture(path, expected_names=["elbow", "wrist"])
    assert result["acceptance_contract"]["first_mismatch"] == {
        "slot": 0,
        "expected_body": "elbow",
        "observed_body": "wrist",
        "native_resolved_body_id": 1,
    }


def test_layout_and_named_difference_keep_both_original_array_coordinates(tmp_path):
    from policy_doctor.layouts import compare_layout_captures, write_layout_capture

    a = capture(tmp_path / "a")
    metadata, arrays, sources = fixture_data(ids=(1, 0))
    arrays["source_force"][1, 0, 2] += 5
    arrays["raw_term_output"] = np.concatenate(
        (arrays["source_force"][:, [1, 0]], arrays["source_torque"][:, [1, 0]]), axis=-1
    ).reshape(2, -1)
    arrays["processed_term_output"] = arrays["raw_term_output"] * 2
    b = tmp_path / "b"
    write_layout_capture(b, metadata, arrays, sources=sources)
    result = compare_layout_captures(a, b)
    assert result["first_layout_difference"]["slot"] == 0
    first = result["first_named_difference"]
    assert first["component"] == "elbow.force.z"
    assert first["reference_index"] == [1, 2] and first["candidate_index"] == [1, 8]
    assert [row["index"] for row in first["evidence"]] == [[1, 2], [1, 8]]
    for row in first["evidence"]:
        assert row["array"] == "raw_term_output" and row["sha256"]


def test_original_license_text_can_travel_without_execution(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    license_path = __import__("pathlib").Path(__file__).parent / "fixtures/isaac_layout_sources/LICENSE.txt"
    sources["LICENSE.txt"] = license_path.read_bytes()
    metadata["source"]["files"]["LICENSE.txt"] = {
        "sha256": hashlib.sha256(sources["LICENSE.txt"]).hexdigest()
    }
    path = tmp_path / "capture"
    assert write_layout_capture(path, metadata, arrays, sources=sources)["status"] == "pass"
    assert (path / "sources/LICENSE.txt").read_bytes() == sources["LICENSE.txt"]


@pytest.mark.parametrize("stage", ["noise", "delay"])
def test_declared_extra_manager_effects_prevent_scale_only_certification(tmp_path, stage):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["processing"][stage] = {"native": "unsupported effect"}
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["raw_binding"]["status"] == "pass"
    assert result["processed_binding"]["status"] == "unknown"
    assert result["processed_binding"]["native_manager_semantics"] == "unknown"


@pytest.mark.parametrize("stage", ["modifiers", "clip", "noise", "history", "delay"])
def test_missing_manager_stage_evidence_is_not_disabled_stage_evidence(tmp_path, stage):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["processing"].pop(stage)
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert result["raw_binding"]["status"] == "pass"
    assert result["processed_binding"]["status"] == "unknown"


def test_shared_metadata_expansion_refused_before_json_serialization(tmp_path, monkeypatch):
    from policy_doctor import layouts

    metadata, arrays, sources = fixture_data()
    metadata["shared"] = [[["x"] * 512] * 512] * 2

    def forbidden(*args, **kwargs):
        pytest.fail("Unbounded shared JSON expansion reached serialization")

    monkeypatch.setattr(layouts.json, "dumps", forbidden)
    with pytest.raises(InputError):
        layouts.write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)


def test_explicit_environment_selection_discloses_omissions_not_temporal_ids(tmp_path):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["runtime"].update(source_num_envs=4096, env_indices=[7, 9])
    result = write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    selection = result["layout"]["environment_selection"]
    assert selection["source_num_envs"] == 4096 and selection["env_indices"] == [7, 9]
    assert selection["saved_rows"] == 2 and selection["omitted_envs"] == 4094
    assert "not temporal" in selection["scope"]
    assert "4094" in result["scope"]


@pytest.mark.parametrize(
    "fields",
    [
        {"source_num_envs": 4096},
        {"env_indices": [0, 1]},
        {"source_num_envs": True, "env_indices": [0, 1]},
        {"source_num_envs": 0, "env_indices": [0, 1]},
        {"source_num_envs": 1000001, "env_indices": [0, 1]},
        {"source_num_envs": 4096.0, "env_indices": [0, 1]},
        {"source_num_envs": 4096, "env_indices": [False, 1]},
        {"source_num_envs": 4096, "env_indices": [0, 0]},
        {"source_num_envs": 4096, "env_indices": [-1, 1]},
        {"source_num_envs": 4096, "env_indices": [0, 4096]},
        {"source_num_envs": 4096, "env_indices": [0]},
        {"source_num_envs": 4096, "env_indices": None},
        {"source_num_envs": 4096, "env_indices": [0, 1], "omitted_env_count": 4095},
        {"source_num_envs": 4096, "env_indices": [0, 1], "omitted_env_count": False},
    ],
)
def test_invalid_environment_selection_refuses_without_publication(tmp_path, fields):
    from policy_doctor.layouts import write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["runtime"].update(fields)
    with pytest.raises(InputError):
        write_layout_capture(tmp_path / "capture", metadata, arrays, sources=sources)
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize("candidate_indices,candidate_count", [([1, 0], 4096), ([0, 1], 2048), (None, None)])
def test_different_or_missing_environment_selection_cannot_align_rows(
    tmp_path, candidate_indices, candidate_count
):
    from policy_doctor.layouts import compare_layout_captures, write_layout_capture

    metadata, arrays, sources = fixture_data()
    metadata["runtime"].update(source_num_envs=4096, env_indices=[0, 1])
    a, b = tmp_path / "a", tmp_path / "b"
    write_layout_capture(a, metadata, arrays, sources=sources)
    if candidate_indices is None:
        metadata["runtime"].pop("source_num_envs")
        metadata["runtime"].pop("env_indices")
    else:
        metadata["runtime"].update(source_num_envs=candidate_count, env_indices=candidate_indices)
    write_layout_capture(b, metadata, arrays, sources=sources)
    result = compare_layout_captures(a, b)
    assert result["raw_values_equal"]
    assert result["status"] == "unknown" and result["named_values_equal"] is None
    assert result["first_named_difference"] is None


def test_matching_environment_selection_and_legacy_row_scope_are_distinct(tmp_path):
    from policy_doctor.layouts import compare_layout_captures, write_layout_capture

    metadata, arrays, sources = fixture_data()
    legacy = tmp_path / "legacy"
    report = write_layout_capture(legacy, metadata, arrays, sources=sources)
    assert report["layout"]["environment_selection"]["env_indices"] is None
    assert "Saved row positions only" in report["layout"]["environment_selection"]["scope"]
    metadata["runtime"].update(source_num_envs=4096, env_indices=[7, 9])
    explicit = tmp_path / "explicit"
    write_layout_capture(explicit, metadata, arrays, sources=sources)
    result = compare_layout_captures(explicit, explicit)
    assert result["status"] == "pass" and result["environment_selection_match"]
