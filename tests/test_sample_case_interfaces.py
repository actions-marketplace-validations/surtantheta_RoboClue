import json
import os
import subprocess
import sys
from pathlib import Path

from policy_doctor.demo import create_controller_demo


def cli(*args):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    return subprocess.run([sys.executable, '-m', 'policy_doctor', *map(str, args)],
                          env=environment, text=True, capture_output=True, timeout=30)


def test_single_command_saves_unknown_without_claiming_success(tmp_path):
    source = tmp_path / 'late.json'
    source.write_text(json.dumps({'joint_names': ['j'], 'points': [
        {'positions': [1.], 'time_from_start': {'sec': 1, 'nanosec': 0}}]}))
    case = tmp_path / 'shared'
    result = cli('sample-trajectory', source, '--times', '0', '1', '--case', case, '--json')
    assert result.returncode == 3, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'unknown' and report['case_directory'] == str(case)
    assert (case / 'READY.json').is_file()
    replay = cli('reproduce-case', case, '--json')
    assert replay.returncode == 0, replay.stderr
    repeated = json.loads(replay.stdout)
    assert repeated['checks']['sample']['status'] == 'unknown'
    assert 'candidate' not in repeated['checks']


def test_single_command_keeps_bound_failure_and_human_handoff(tmp_path):
    demo = tmp_path / 'demo'
    create_controller_demo(demo)
    limits = tmp_path / 'limits.json'
    limits.write_text(json.dumps({'shoulder': [0., 1.]}))
    case = tmp_path / 'check with spaces'
    result = cli('sample-trajectory', demo / 'candidate.json', '--times', '.75',
                 '--position-limits', limits, '--case', case)
    assert result.returncode == 1, result.stderr
    assert 'reference samples: error' in result.stdout
    assert 'reproduce-case' in result.stdout and str(case) in result.stdout
    replay = cli('reproduce-case', case)
    assert replay.returncode == 0, replay.stderr
    assert 'Sample' in replay.stdout and 'error' in replay.stdout
    assert 'Candidate comparison' not in replay.stdout


def test_case_and_output_conflict_refuses_before_any_write(tmp_path):
    source = tmp_path / 'message.json'
    source.write_text('{}')
    case, report = tmp_path / 'case', tmp_path / 'report.json'
    result = cli('sample-trajectory', source, '--times', '0', '--case', case, '--output', report)
    assert result.returncode == 2
    assert not case.exists() and not report.exists()


def test_sampling_without_bounds_does_not_present_zero_as_checked(tmp_path):
    source = tmp_path / 'message.json'
    source.write_text(json.dumps({'joint_names': ['j'], 'points': [
        {'positions': [1.], 'time_from_start': {'sec': 0, 'nanosec': 0}}]}))
    result = cli('sample-trajectory', source, '--times', '0')
    assert result.returncode == 0, result.stderr
    assert 'bounds check not performed' in result.stdout
