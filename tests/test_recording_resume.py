"""The public wrapper must stop before a recorder can touch an unsafe root."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest
from test_salvage import hashes, healthy, meta_for, rows_for, write

from policy_doctor.artifacts import InputError
from policy_doctor.product_cli import main as cli
from policy_doctor.resume_guard import assess_resume, bind_recording_command, recording_lock


def recorder(tmp_path):
    executable = tmp_path / 'bin' / 'lerobot-record'
    executable.parent.mkdir()
    executable.write_text('#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n'
                          "args=sys.argv[1:]\nmarker=next(x.split('=',1)[1] for x in args if x.startswith('--marker='))\n"
                          "Path(marker).write_text(json.dumps(args))\nraise SystemExit(int(next((x.split('=',1)[1] for x in args if x.startswith('--exit=')), '0')))\n")
    executable.chmod(0o755)
    return str(executable)


def command(root, executable='lerobot-record', **kw):
    return [executable, '--dataset.root=' + str(root), '--resume=true', *['--' + k + '=' + str(v) for k, v in kw.items()]]


def test_actual_recorder_stays_unstarted_until_separate_corrected_copy_clears(tmp_path):
    data = rows_for([(0, 5), (1, 6)])
    broken = write(tmp_path / 'broken', [data], [meta_for([(0, 5)])], 1, 5)
    before = hashes(broken)
    executable = recorder(tmp_path)
    marker = tmp_path / 'started.json'
    assert cli(['guard', 'resume', str(broken), '--', *command(broken, executable, marker=marker)]) == 1
    assert not marker.exists() and hashes(broken) == before
    # Independent fixture defines the intended committed cohort; the gate
    # neither invents that intent nor repairs the original automatically.
    corrected = healthy(tmp_path / 'copy', episodes=((0, 5),))
    corrected_before = hashes(corrected)
    expected = command(corrected, executable, marker=marker)
    assert cli(['guard', 'resume', str(corrected), '--', *expected]) == 0
    assert json.loads(marker.read_text()) == expected[1:]
    assert hashes(broken) == before and hashes(corrected) == corrected_before


@pytest.mark.parametrize('exit_code', [0, 1, 7, 42])
def test_downstream_arguments_and_exit_status_preserved(tmp_path, exit_code):
    root = healthy(tmp_path)
    marker = tmp_path / 'a path with spaces.json'
    expected = command(root, recorder(tmp_path), marker=marker, exit=exit_code, task='a task; $(literal) `literal`')
    assert cli(['guard', 'resume', str(root), '--', *expected]) == exit_code
    assert json.loads(marker.read_text()) == expected[1:]
    assert not (tmp_path / 'literal').exists()


def test_dry_run_json_never_launches_even_healthy_recorder(tmp_path, capsys):
    root = healthy(tmp_path)
    marker = tmp_path / 'never'
    before = hashes(root)
    code = cli(['guard', 'resume', str(root), '--dry-run', '--json', '--', *command(root, recorder(tmp_path), marker=marker)])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result['status'] == 'pass' and result['launched'] is False
    assert result['current_directory_resume_status'] == 'no_hazard_detected'
    assert not marker.exists() and hashes(root) == before


@pytest.mark.parametrize('tail', [[], ['--resume=false'], ['--resume'], ['--resume=yes'], ['--resume=true'],
                                 ['--resume=true', '--dataset.root=missing'], ['--resume=true', '--dataset.rooot=missing'],
                                 ['--resume=true', '--dataset.root'], ['--resume=true', '--config_path=anything.json']])
def test_absent_mistyped_or_ambiguous_binding_refused(tmp_path, tail):
    root = healthy(tmp_path)
    with pytest.raises(InputError):
        bind_recording_command(root, ['lerobot-record', *tail])


@pytest.mark.parametrize('suffix', [['--resume=true'], ['--dataset.root=other'], ['--config_path=x'], ['--config=x'],
                                   ['--dataset={}'], ['--'], ['&&'], ['--robot.port', '/dev/ttyACM0']])
def test_overrides_duplicates_and_shell_syntax_refused(tmp_path, suffix):
    root = healthy(tmp_path)
    with pytest.raises(InputError):
        bind_recording_command(root, [*command(root), *suffix])


def test_other_dataset_and_arbitrary_shell_cannot_receive_clearance(tmp_path):
    root = healthy(tmp_path / 'first')
    other = healthy(tmp_path / 'second')
    with pytest.raises(InputError, match='differs'):
        bind_recording_command(root, command(other))
    with pytest.raises(InputError, match='console command'):
        bind_recording_command(root, ['bash', '-c', 'lerobot-record --resume=true'])


def test_unexpanded_tilde_in_native_root_is_not_mistaken_for_a_bound_path(tmp_path):
    root = healthy(tmp_path)
    with pytest.raises(InputError, match='tilde'):
        bind_recording_command(root, ['lerobot-record', '--dataset.root=~/recording', '--resume=true'])


def test_separated_root_and_resume_values_bind_without_modifying_them(tmp_path):
    root = healthy(tmp_path)
    args = ['lerobot-record', '--dataset.root', str(root), '--resume', 'true', '--robot.port=/dev/ttyACM0']
    assert bind_recording_command(root, args) == (root, args)


def test_required_partial_evidence_blocks_without_launch(tmp_path):
    root = healthy(tmp_path)
    marker = tmp_path / 'never'
    before = hashes(root)
    assert cli(['guard', 'resume', str(root), '--max-rows', '1', '--', *command(root, recorder(tmp_path), marker=marker)]) == 3
    assert not marker.exists() and hashes(root) == before


def test_missing_video_not_ignored_by_resume_gate(tmp_path):
    root = healthy(tmp_path, video=True)
    marker = tmp_path / 'never'
    assert cli(['guard', 'resume', str(root), '--', *command(root, recorder(tmp_path), marker=marker)]) in (1, 3)
    assert not marker.exists()


@pytest.mark.parametrize('version', ['v2.1', 'v3.1', 'v3.99'])
def test_unsupported_dataset_version_and_json_execution_refused(tmp_path, version):
    root = healthy(tmp_path, version=version)
    marker = tmp_path / 'never'
    wrapped = command(root, recorder(tmp_path), marker=marker)
    assert cli(['guard', 'resume', str(root), '--', *wrapped]) == 2
    assert cli(['guard', 'resume', str(root), '--json', '--', *wrapped]) == 2
    assert not marker.exists()


def test_second_wrapper_is_refused_while_first_holds_directory_lock(tmp_path):
    root = healthy(tmp_path)
    before = hashes(root)
    with recording_lock(root):
        assert cli(['guard', 'resume', str(root), '--dry-run', '--', *command(root)]) == 2
    assert assess_resume(root, command(root))['exit_code'] == 0 and hashes(root) == before


def test_actual_public_module_stops_before_command_on_bad_input(tmp_path):
    root = healthy(tmp_path)
    marker = tmp_path / 'never'
    argv = [sys.executable, '-B', '-m', 'roboclue', 'guard', 'resume', str(root), '--max-rows', '1', '--',
            *command(root, recorder(tmp_path), marker=marker)]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    run = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
    assert run.returncode == 3 and 'Resume blocked' in run.stdout
    assert not marker.exists()


def test_links_and_special_file_cannot_bind_recording_root(tmp_path):
    root = healthy(tmp_path)
    link = tmp_path / 'link'
    link.symlink_to(root)
    with pytest.raises(InputError, match='Symlink'):
        bind_recording_command(link, command(link))
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    with pytest.raises(InputError, match='regular file or directory'):
        bind_recording_command(fifo, command(fifo))


def test_parent_cancellation_reaches_recorder_before_lock_is_released(tmp_path):
    root = healthy(tmp_path)
    executable = tmp_path / 'lerobot-record'
    ready, received = tmp_path / 'ready', tmp_path / 'received'
    executable.write_text('#!' + sys.executable + '\nimport signal,sys,time\nfrom pathlib import Path\n'
                          'def stop(number,frame):\n'
                          ' Path(' + repr(str(received)) + ').write_text(str(number))\n'
                          ' time.sleep(0.2)\n raise SystemExit(7)\n'
                          'signal.signal(signal.SIGTERM,stop)\nsignal.alarm(10)\n'
                          'Path(' + repr(str(ready)) + ').write_text("ready")\n'
                          'while True: time.sleep(0.01)\n')
    executable.chmod(0o755)
    argv = [sys.executable, '-B', '-m', 'roboclue', 'guard', 'resume', str(root), '--', *command(root, str(executable))]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=2)
        process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not received.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received.read_text() == str(signal.SIGTERM)
        with pytest.raises(InputError, match='Another RoboClue'):
            with recording_lock(root):
                pass
        process.communicate(timeout=10)
        assert process.returncode == 7
        with recording_lock(root):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_resume_leads_with_observed_stale_row_hazard():
    from policy_doctor.resume_guard import _resume_problem

    findings = [
        {'code': 'salvage.metadata_plan', 'status': 'error', 'message': 'Metadata totals need updating.'},
        {'code': 'salvage.resume_hazard', 'status': 'error',
         'message': 'Resuming serves stale rows instead of newly recorded episode rows.'},
    ]
    assert _resume_problem(findings) == findings[1]['message']
    assert _resume_problem([]) == 'Required evidence is incomplete.'
