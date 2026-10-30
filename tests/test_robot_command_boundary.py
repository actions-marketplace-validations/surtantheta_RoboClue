"""A consumer gate must not quietly select one command from a shell program."""
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.robot_interface import from_command

BASE = 'lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.use_degrees=false'


@pytest.mark.parametrize('text', [
    BASE + ' && lerobot-rollout --robot.type=so101_follower --robot.use_degrees=true',
    BASE + '; lerobot-rollout --robot.type=so101_follower --robot.use_degrees=true',
    BASE + '\nlerobot-rollout --robot.type=so101_follower --robot.use_degrees=true',
    BASE + ' | cat',
    BASE + ' || echo failed',
    BASE + ' &',
    BASE + ' > output.txt',
    BASE + ' 2>/dev/null',
    BASE + ' < input.txt',
    BASE + ' --robot.id=$(echo fleet)',
    BASE + ' --robot.id="$(echo fleet)"',
    BASE + ' --robot.id=`echo fleet`',
    BASE + ' --robot.id="`echo fleet`"',
    'echo preparing\n' + BASE,
    BASE + '\nexit 0',
    BASE + ' --robot.port=<(echo device)',
])
def test_canonical_robot_command_rejects_ambiguous_shell_programs(text):
    with pytest.raises(InputError, match='one robot command|one resolved robot command'):
        from_command(text, strict=True)


@pytest.mark.parametrize('extra', [
    " --robot.id=';'",
    ' --robot.id ";"',
    ' --robot.id="&&"',
    " --robot.id='a|b'",
    " --robot.id='input > output'",
    r' --robot.id=a\;b',
    r' --robot.id=a\&b',
    r' --robot.id=a\|b',
    " --robot.id='$(literal)'",
    " --robot.id='`literal`'",
    ' --robot.cameras="{up: {type: opencv, index_or_path: 0}, side: {type: opencv, index_or_path: 1}}"',
    ' # documentation with ; && | > $(ignored)',
])
def test_canonical_robot_command_preserves_quoted_or_escaped_data(extra):
    result = from_command(BASE + extra, strict=True)
    assert result['robot_type'] == 'so101_follower'
    assert result['program'] == 'record'


def test_line_continuation_and_comment_do_not_invent_another_command():
    result = from_command('# explanation\n' + BASE.replace(' --robot.port', ' \\\n --robot.port') + '\n', strict=True)
    assert result['robot_type'] == 'so101_follower'


def test_legacy_script_intake_retains_its_explicit_first_command_behavior():
    text = BASE + ' && lerobot-rollout --robot.type=so101_follower --robot.use_degrees=true'
    result = from_command(text)
    assert result['program'] == 'record'
    assert result['source']['program_index'] == 1
    assert result['unknown']
