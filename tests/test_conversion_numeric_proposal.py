"""Generated arrays only; no original HDF5/Parquet/native converter execution."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from policy_doctor.artifacts import InputError
from policy_doctor.conversion_numeric import (
    MAX_ROWS,
    PERMUTATION,
    ROOT,
    SOURCE_HASHES,
    EpisodeProjection,
    Selection,
    compile_criterion,
    evaluate_numeric,
)

SOURCE = Path(__file__).parent / "fixtures" / "arena43"


@pytest.fixture
def closure():
    return {
        name: (SOURCE / (name + ".txt" if name.endswith(".py") else name)).read_bytes()
        for name in SOURCE_HASHES
    }


def native_data_permutation(closure):
    """Independent reference derived from native YAML/JSON data, not module indices."""
    source = yaml.safe_load(closure[ROOT + "embodiments/g1/43dof_joint_space.yaml"])["joints"]
    policy = yaml.safe_load(closure[ROOT + "embodiments/g1/gr00t_43dof_joint_space.yaml"])["joints"]
    groups = json.loads(closure[ROOT + "embodiments/g1/modality.json"])["action"]
    omitted = {
        "left_wrist_pose",
        "right_wrist_pose",
        "base_height_command",
        "navigate_command",
        "torso_orientation_rpy_command",
    }
    return tuple(source[joint] for group in groups if group not in omitted for joint in policy[group])


def generated(closure, dtype="float32", frames=(0, 2, 6)):
    selection = Selection("generated_trajectory_only", 17, frames)
    criterion = compile_criterion(closure, (selection,))
    action = (np.arange(8 * 43).reshape(8, 43) / 64).astype(dtype)
    state = action + np.array(100, dtype=dtype)
    independent_indices = native_data_permutation(closure)
    projected = EpisodeProjection(
        selection.source_key,
        17,
        state,
        action,
        np.take(state[:-1], independent_indices, axis=1),
        np.take(action[:-1], independent_indices, axis=1),
    )
    return criterion, projected


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_declared_remap_terminal_drop_and_projection_assurance(closure, dtype):
    criterion, episode = generated(closure, dtype)
    assert PERMUTATION == native_data_permutation(closure)
    result = evaluate_numeric(criterion, [episode])
    assert result["numeric_association_status"] == "pass"
    assert result["coverage"]["compared_selected_rows"] == 3
    assert result["status"] == "unknown" and result["assurance"] == "projection_only"
    assert result["source_membership"]["status"] == result["native_consumption"]["status"] == "unknown"
    # Noncontiguous/Fortran memory layout and signed zero do not change numeric values.
    episode.exported_action = np.asfortranarray(episode.exported_action)
    assert evaluate_numeric(criterion, [episode])["numeric_association_status"] == "pass"


@pytest.mark.parametrize("change", ["joint_permutation", "state_for_action", "one_step_shift"])
def test_shape_valid_generated_fault_and_same_criterion_corrected_projection(closure, change):
    criterion, episode = generated(closure)
    original = episode.exported_action.copy()
    if change == "joint_permutation":
        episode.exported_action[:, [0, 1]] = episode.exported_action[:, [1, 0]]
    elif change == "state_for_action":
        episode.exported_action = episode.exported_state.copy()
    else:
        episode.exported_action = np.roll(episode.exported_action, 1, axis=0)
    result = evaluate_numeric(criterion, [episode])
    assert result["status"] == result["numeric_association_status"] == "error"
    assert any(
        f["code"] == "conversion.numeric_joint_association" and f["role"] == "action"
        for f in result["findings"]
    )
    episode.exported_action = original
    repaired = evaluate_numeric(criterion, [episode])
    assert repaired["criterion_sha256"] == result["criterion_sha256"] == criterion.digest
    assert repaired["numeric_association_status"] == "pass" and repaired["status"] == "unknown"


def test_keep_terminal_row_and_truncated_selected_coverage(closure):
    criterion, episode = generated(closure)
    episode.exported_action = np.take(episode.source_action, native_data_permutation(closure), axis=1)
    result = evaluate_numeric(criterion, [episode])
    assert any(f["code"] == "conversion.terminal_trim_or_coverage" for f in result["findings"])
    episode.exported_action = episode.exported_action[:2]
    result = evaluate_numeric(criterion, [episode])
    assert result["status"] == "error" and result["counts"]["unknown"] > 0
    assert result["coverage"]["compared_selected_rows"] == 1
    assert not result["coverage"]["complete_selected_numeric_projection"]


def test_missing_duplicate_correspondence_and_missing_values_are_unknown(closure):
    criterion, episode = generated(closure)
    for supplied in [[], [episode, episode], [dataclasses.replace(episode, source_action=None)]]:
        result = evaluate_numeric(criterion, supplied)
        assert result["numeric_association_status"] == result["status"] == "unknown"
        assert result["coverage"]["compared_selected_rows"] == 0


def test_missing_other_episode_preserves_known_numeric_error(closure):
    criterion, episode = generated(closure)
    other = Selection("generated_missing_only", 41, (0,))
    criterion = compile_criterion(closure, criterion.selections + (other,))
    episode.exported_action[0, 0] += 1
    result = evaluate_numeric(criterion, [episode])
    assert result["status"] == "error" and result["counts"]["error"] and result["counts"]["unknown"]


def test_dtype_precision_and_nonfinite_not_silently_normalized(closure):
    criterion, episode = generated(closure)
    episode.exported_action = episode.exported_action.astype("float64")
    assert any(
        f["code"] == "conversion.dtype_changed" for f in evaluate_numeric(criterion, [episode])["findings"]
    )
    episode.exported_action[0, 0] += 1e-12
    assert any(
        f["code"] == "conversion.numeric_joint_association"
        for f in evaluate_numeric(criterion, [episode])["findings"]
    )
    episode.exported_action[0, 0] = np.nan
    assert evaluate_numeric(criterion, [episode])["status"] == "error"
    episode.source_action[0, 0] = np.inf
    result = evaluate_numeric(criterion, [episode])
    assert result["status"] == "error" and result["counts"]["unknown"] > 0


@pytest.mark.parametrize("frames", [(True,), (0, 0), (2, 0), (-1,), (MAX_ROWS,)])
def test_invalid_or_ambiguous_frame_selection_refused(closure, frames):
    with pytest.raises(InputError):
        generated(closure, frames=frames)


def test_terminal_source_frame_is_not_emitted_and_not_blessed(closure):
    criterion, episode = generated(closure, frames=(7,))
    result = evaluate_numeric(criterion, [episode])
    assert result["numeric_association_status"] == "unknown"
    assert result["coverage"]["compared_selected_rows"] == 0


def test_changed_or_missing_semantic_source_and_actual_path_edited_config_refused(closure):
    selection = (Selection("generated_only", 0, (0,)),)
    for changed in [
        {},
        {**closure, ROOT + "utils/joints_conversion.py": b"unreviewed source"},
        {**closure, ROOT + "lerobot/config/g1_static_apple_config.yaml": b"data_root: /different"},
    ]:
        with pytest.raises(InputError):
            compile_criterion(changed, selection)


def test_criterion_and_episode_identity_drift_refused(closure):
    criterion, episode = generated(closure)
    with pytest.raises(InputError):
        evaluate_numeric(dataclasses.replace(criterion, digest="0" * 64), [episode])
    with pytest.raises(InputError):
        evaluate_numeric(criterion, [dataclasses.replace(episode, native_episode_index=True)])
    with pytest.raises(InputError):
        evaluate_numeric(criterion, [dataclasses.replace(episode, source_key="unselected")])
    s = criterion.selections[0]
    with pytest.raises(InputError):
        compile_criterion(closure, (s, s))


def test_input_shape_objects_and_value_budget_refused_before_numeric_work(closure, monkeypatch):
    import policy_doctor.conversion_numeric as module

    criterion, episode = generated(closure)
    for value in [np.zeros((2, 42), dtype="float32"), np.zeros((2, 43), dtype=object), [[1] * 43]]:
        with pytest.raises(InputError):
            evaluate_numeric(criterion, [dataclasses.replace(episode, exported_action=value)])
    monkeypatch.setattr(module, "MAX_VALUES_BYTES", 1)
    with pytest.raises(InputError):
        evaluate_numeric(criterion, [episode])


def test_bounded_findings_keep_complete_error_counts(closure):
    criterion, episode = generated(closure, frames=tuple(range(7)))
    selections, episodes = [], []
    for i in range(10):
        selections.append(Selection("generated_only_" + str(i), i, tuple(range(7))))
        episodes.append(
            dataclasses.replace(
                episode,
                source_key=selections[-1].source_key,
                native_episode_index=i,
                exported_action=episode.exported_action + 1,
            )
        )
    criterion = compile_criterion(closure, tuple(selections))
    result = evaluate_numeric(criterion, episodes)
    assert result["findings_total"] == 70 and result["findings_truncated"]
    assert len(result["findings"]) == 64


@pytest.mark.parametrize("key", [".", ".."])
def test_reserved_original_group_locator_refused(closure, key):
    with pytest.raises(InputError):
        compile_criterion(closure, (Selection(key, 0, (0,)),))


def test_nonreserved_dotted_group_name_is_not_guessed_or_rejected(closure):
    criterion = compile_criterion(closure, (Selection("generated..dotted", 0, (0,)),))
    assert criterion.selections[0].source_key == "generated..dotted"
