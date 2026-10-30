"""Generated ordinary artifacts testing numeric selection and honest boundaries."""

import json
import os
import shutil

import numpy as np
import pytest
from test_consumer_context import native_layout

from policy_doctor.artifacts import InputError
from policy_doctor.consumer_context import resolve_native_processor_context
from policy_doctor.consumer_samples import load_consumer_sample

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture
def ordinary(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config, stats = native_layout()
    (checkpoint / "processor_config.json").write_text(json.dumps(config))
    (checkpoint / "statistics.json").write_text(json.dumps(stats))
    context = resolve_native_processor_context(checkpoint, read_roots=[tmp_path])
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {key: {"dtype": "float32", "shape": [width]} for key, width in (
            ("observation.state", 43), ("action", 43),
            ("teleop.base_height_command", 1), ("teleop.navigate_command", 3))},
    }
    slices = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15),
              "left_arm": (15, 22), "left_hand": (22, 29),
              "right_arm": (29, 36), "right_hand": (36, 43)}
    modality = {kind: {key: {"start": start, "end": end} for key, (start, end) in slices.items()}
                for kind in ("state", "action")}
    modality["action"].update({
        "base_height_command": {"start": 0, "end": 1, "original_key": "teleop.base_height_command"},
        "navigate_command": {"start": 0, "end": 3, "original_key": "teleop.navigate_command"},
    })
    for name, value in (("info", info), ("modality", modality)):
        (dataset / ("meta/" + name + ".json")).write_text(json.dumps(value))
    # Gapped stored IDs: native episode position and the stored ID must differ.
    (dataset / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 60}) + "\n"
        + json.dumps({"episode_index": 72, "length": 60}) + "\n")
    values = {
        "episode_index": [72] * 60, "frame_index": list(range(60)), "index": list(range(100, 160)),
        "observation.state": [[float(1000 * row + col) for col in range(43)] for row in range(60)],
        "action": [[float(10000 + 1000 * row + col) for col in range(43)] for row in range(60)],
        "teleop.base_height_command": [[float(row) + 0.125] for row in range(60)],
        "teleop.navigate_command": [[float(row) + offset for offset in (0.25, 0.5, 0.75)] for row in range(60)],
    }

    def write():
        pq.write_table(pa.table(values), dataset / "data/chunk-000/episode_000072.parquet")

    write()
    return dataset, context, values, write


def read(ordinary, **kwargs):
    dataset, context, _, _ = ordinary
    return load_consumer_sample(dataset, context, episode_index=72,
                                anchor_index=kwargs.pop("anchor_index", 5),
                                read_roots=[dataset.parent], **kwargs)


def test_ordinary_artifacts_select_native_order_slices_and_auxiliaries(ordinary):
    result = read(ordinary)
    raw = result["raw_sample"]
    np.testing.assert_array_equal(raw["states"]["right_arm"], [[5029 + i for i in range(7)]])
    np.testing.assert_array_equal(raw["actions"]["left_hand"][0], [15022 + i for i in range(7)])
    assert raw["actions"]["navigate_command"].dtype == np.float32
    assert raw["actions"]["navigate_command"].shape == (50, 3)
    assert raw["actions"]["base_height_command"][49, 0] == 54.125
    selection = raw["selection"]
    assert selection["native_episode_position"] == 1
    assert selection["stored_episode_index"] == 72
    assert selection["action_indices"] == list(range(5, 55))
    assert selection["action_global_indices"] == list(range(105, 155))
    assert selection["clamped_occurrences"] == 0
    assert selection["native_training_start_eligible"] is True
    assert len(json.dumps(selection).encode()) < 16384
    assert len(selection["readset"]) == 4
    assert result["source_evidence"]["status"] == "PASS"
    assert result["source_evidence"]["media_read"] is False
    assert result["source_evidence"]["full_sample_parity"] == "UNKNOWN"
    assert result["source_evidence"]["historical_execution"] == "UNKNOWN"


def test_padding_is_explicit_and_not_a_normal_training_start(ordinary):
    with pytest.raises(InputError, match="clamping"):
        read(ordinary, anchor_index=59)
    result = read(ordinary, anchor_index=59, allow_padding=True)["raw_sample"]
    assert result["selection"]["action_indices"] == [59] * 50
    assert result["selection"]["clamped_occurrences"] == 49
    assert result["selection"]["native_training_start_eligible"] is False
    np.testing.assert_array_equal(result["actions"]["left_arm"][0], result["actions"]["left_arm"][-1])


def test_last_complete_window_and_first_incomplete_window(ordinary):
    assert read(ordinary, anchor_index=10)["raw_sample"]["selection"]["action_indices"][-1] == 59
    with pytest.raises(InputError, match="clamping"):
        read(ordinary, anchor_index=11)


def test_actual_modality_file_controls_the_selected_columns(ordinary):
    dataset = ordinary[0]
    path = dataset / "meta/modality.json"
    modality = json.loads(path.read_text())
    modality["action"]["left_arm"]["original_key"] = "observation.state"
    path.write_text(json.dumps(modality))
    raw = read(ordinary)["raw_sample"]
    assert raw["actions"]["left_arm"][0, 0] == 5015
    assert next(f for f in raw["selection"]["fields"] if f["modality"] == "action"
                and f["key"] == "left_arm")["column"] == "observation.state"


def test_same_width_slice_permutation_follows_declarations_without_joint_truth_claim(ordinary):
    path = ordinary[0] / "meta/modality.json"
    modality = json.loads(path.read_text())
    modality["action"]["left_arm"].update(start=29, end=36)
    path.write_text(json.dumps(modality))
    raw = read(ordinary)["raw_sample"]
    np.testing.assert_array_equal(raw["actions"]["left_arm"], raw["actions"]["right_arm"])


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
def test_native_float32_boundary_is_independent_of_parquet_float_encoding(ordinary, dtype):
    values = ordinary[2]
    arrays = {key: pa.array(value, type=pa.list_(dtype)) if type(value[0]) is list
              else pa.array(value) for key, value in values.items()}
    pq.write_table(pa.table(arrays), ordinary[0] / "data/chunk-000/episode_000072.parquet")
    result = read(ordinary)["raw_sample"]
    assert all(a.dtype == np.float32 for modality in ("states", "actions") for a in result[modality].values())
    assert result["actions"]["left_arm"][0, 0] == 15015


def test_unselected_action_coordinates_do_not_create_selected_sample_failure(ordinary):
    # The pinned consumer never requests left_leg action coordinates.
    ordinary[2]["action"][5][0] = float("nan")
    ordinary[3]()
    assert read(ordinary)["source_evidence"]["status"] == "PASS"


def test_dataset_root_replacement_is_refused_even_with_hardlinked_files(ordinary, monkeypatch):
    from policy_doctor import consumer_samples

    original = consumer_samples._directory_fd
    dataset = ordinary[0]
    replaced = False

    def replacing_directory(root):
        nonlocal replaced
        result = original(root)
        if not replaced:
            replaced = True
            retained = dataset.with_name("retained")
            dataset.rename(retained)
            shutil.copytree(retained, dataset, copy_function=os.link)
        return result

    monkeypatch.setattr(consumer_samples, "_directory_fd", replacing_directory)
    with pytest.raises(InputError, match="directory identity"):
        read(ordinary)


def test_selected_source_mutation_during_intake_is_refused(ordinary, monkeypatch):
    from policy_doctor import consumer_samples

    original = consumer_samples._read
    mutated = False

    def mutating_read(root, relative, limit):
        nonlocal mutated
        result = original(root, relative, limit)
        if relative.endswith(".parquet") and not mutated:
            mutated = True
            ordinary[2]["action"][5][15] += 1
            ordinary[3]()
        return result

    monkeypatch.setattr(consumer_samples, "_read", mutating_read)
    with pytest.raises(InputError, match="artifact changed"):
        read(ordinary)


@pytest.mark.parametrize("column,value", [("frame_index", 4), ("episode_index", 0), ("index", None)])
def test_invalid_source_row_identity_is_not_certified(ordinary, column, value):
    ordinary[2][column][5] = value
    ordinary[3]()
    with pytest.raises(InputError):
        read(ordinary)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e100, None])
def test_selected_nonfinite_null_or_float32_overflow_is_refused(ordinary, value):
    ordinary[2]["action"][5][15] = value
    ordinary[3]()
    with pytest.raises(InputError):
        read(ordinary)


def test_missing_selected_auxiliary_column_is_not_silently_zero_filled(ordinary):
    del ordinary[2]["teleop.navigate_command"]
    ordinary[3]()
    with pytest.raises(InputError, match="missing"):
        read(ordinary)


def test_outside_roots_and_source_symlinks_are_refused(ordinary):
    dataset, context, _, _ = ordinary
    with pytest.raises(InputError, match="outside"):
        load_consumer_sample(dataset, context, episode_index=72, anchor_index=5, read_roots=[dataset])
    path = dataset / "data/chunk-000/episode_000072.parquet"
    moved = path.with_suffix(".original")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(InputError):
        read(ordinary)


def test_changed_checkpoint_requires_a_fresh_context(ordinary):
    path = ordinary[0].parent / "checkpoint/statistics.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(InputError, match="identity"):
        read(ordinary)


def test_partial_context_does_not_select_rows_from_incomplete_parameters(ordinary):
    dataset = ordinary[0]
    checkpoint = dataset.parent / "checkpoint"
    config = json.loads((checkpoint / "processor_config.json").read_text())
    config["processor_kwargs"]["use_relative_action"] = True
    (checkpoint / "processor_config.json").write_text(json.dumps(config))
    ordinary = (dataset, resolve_native_processor_context(checkpoint), ordinary[2], ordinary[3])
    assert ordinary[1]["status"] == "ERROR"
    with pytest.raises(InputError):
        read(ordinary)


@pytest.mark.parametrize("kwargs", [{"anchor_index": True}, {"anchor_index": -1}, {"allow_padding": 1}])
def test_strict_selection_types(ordinary, kwargs):
    with pytest.raises(InputError):
        read(ordinary, **kwargs)


def test_episode_length_mismatch_and_duplicate_ids_are_refused(ordinary):
    path = ordinary[0] / "meta/episodes.jsonl"
    path.write_text(json.dumps({"episode_index": 72, "length": 59}))
    with pytest.raises(InputError, match="row count"):
        read(ordinary)
    path.write_text((json.dumps({"episode_index": 72, "length": 60}) + "\n") * 2)
    with pytest.raises(InputError, match="duplicate"):
        read(ordinary)


def test_malformed_or_unsupported_modality_and_layout_are_refused(ordinary):
    path = ordinary[0] / "meta/modality.json"
    path.write_text('{"state": 12}')
    with pytest.raises(InputError, match="missing"):
        read(ordinary)
    path.write_text('{"state": {}, "state": {}}')
    with pytest.raises(InputError):
        read(ordinary)
