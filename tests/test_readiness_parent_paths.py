"""Regression for native path selection, separate from source-profile inspection."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.readiness_workflow import (
    _local,
    authorize_command_directory,
    prepare_readiness,
)


@pytest.mark.parametrize('value', ['../checkpoint', 'link/../checkpoint', '/inside/link/../checkpoint'])
def test_literal_parent_components_refuse_before_any_probe(value):
    with patch('os.lstat', side_effect=AssertionError('path was probed')), \
            patch.object(Path, 'resolve', side_effect=AssertionError('path was resolved')):
        with pytest.raises(InputError, match='parent path components'):
            _local(value, read_roots=[Path('/inside')], base_directory='/inside')


def test_declared_base_parent_components_are_not_simplified():
    with patch('os.lstat', side_effect=AssertionError('base was probed')):
        with pytest.raises(InputError, match='parent path components'):
            _local('checkpoint', read_roots=[Path('/inside')], base_directory='/inside/link/..')


@pytest.mark.parametrize('value', ['~/checkpoint', '~another/checkpoint', '~'])
def test_literal_tilde_components_refuse_before_expansion_or_probes(value):
    with patch.object(Path, 'expanduser', side_effect=AssertionError('literal home was expanded')), \
            patch('os.lstat', side_effect=AssertionError('path was probed')):
        with pytest.raises(InputError, match='literal tilde'):
            _local(value, read_roots=[Path('/inside')], base_directory='/inside')


def test_pasted_lerobot_tilde_reference_does_not_select_operator_home():
    with patch('os.lstat', side_effect=AssertionError('native literal reference was probed')):
        with pytest.raises(InputError, match='literal tilde'):
            prepare_readiness(robot_command='lerobot-rollout --robot.type=lekiwi --policy.path="~/checkpoint"',
                              read_roots=[Path('/inside')], base_directory='/inside')


def test_command_directory_parent_components_refuse_before_probes():
    with patch('os.lstat', side_effect=AssertionError('directory was probed')):
        with pytest.raises(InputError, match='parent path components'):
            authorize_command_directory('/inside/link/..', read_roots=[Path('/inside')])


def test_native_literal_and_lexical_simplification_can_select_different_artifacts(tmp_path):
    inside = tmp_path / 'inside'
    outside = tmp_path / 'outside'
    for parent, body in [(inside, 'inside'), (outside, 'native')]:
        (parent / 'checkpoint').mkdir(parents=True)
        (parent / 'checkpoint/config.json').write_text(body)
    (outside / 'child').mkdir()
    (inside / 'link').symlink_to(outside / 'child', target_is_directory=True)
    literal = inside / 'link/../checkpoint/config.json'
    # Ordinary OS lookup is the native path semantics; abspath loses the link.
    assert literal.read_text() == 'native'
    assert Path(os.path.abspath(literal)).read_text() == 'inside'
    with patch('os.lstat', side_effect=AssertionError('ambiguous command reference was probed')):
        with pytest.raises(InputError, match='parent path components'):
            prepare_readiness(
                robot_command=(f'lerobot-rollout --robot.type=lekiwi '
                               f'--policy.path={inside / "link/../checkpoint"}'),
                read_roots=[inside], base_directory=str(inside),
            )


def test_existing_canonical_checkpoint_selection_is_preserved(tmp_path):
    selected = tmp_path / 'checkpoint'
    selected.mkdir()
    assert _local(str(selected), read_roots=[tmp_path]) == str(selected)


def test_dots_inside_a_name_are_not_parent_components(tmp_path):
    selected = tmp_path / 'checkpoint..backup'
    selected.mkdir()
    assert _local(str(selected), read_roots=[tmp_path]) == str(selected)
