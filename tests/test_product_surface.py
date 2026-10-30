"""Public branding and scope remain a thin interface to the existing engines."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from policy_doctor import product_cli


@pytest.mark.parametrize(('name', 'module'), [
    ('Gr00tTrainingGuard', 'groot_training_guard'),
    ('capture_gr00t_reference', 'groot_training_guard'),
    ('LeRobotTrainingGuard', 'lerobot_training_guard'),
    ('capture_lerobot_reference', 'lerobot_training_guard'),
    ('OpenpiTrainingGuard', 'openpi_training_guard'),
    ('capture_openpi_reference', 'openpi_training_guard'),
    ('RuntimeRecorder', 'runtime_recorder'),
    ('SignalSpec', 'runtime_recorder'),
    ('SendReceipt', 'runtime_recorder'),
    ('health_card', 'health_cards'),
    ('triage_flight_recording', 'flight_recorder'),
])
def test_public_import_is_the_original_implementation(name, module):
    import roboclue

    assert getattr(roboclue, name) is getattr(importlib.import_module('policy_doctor.' + module), name)
    with pytest.raises(AttributeError):
        _ = roboclue.a_typo_is_not_a_new_adapter


def test_module_version_does_not_import_mcp_or_dataset_readers():
    code = '''import runpy,sys
sys.argv=['roboclue','--version']
try:
    runpy.run_module('roboclue',run_name='__main__')
except SystemExit as e:
    assert e.code==0
assert not any(m in sys.modules for m in ('mcp','pyarrow','torch','h5py'))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('RoboClue ')


def test_canonical_metadata_and_entry_point():
    import tomllib

    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())['project']
    assert project['name'] == 'roboclue'
    assert project['scripts']['roboclue'] == 'policy_doctor.product_cli:main'
    assert 'policy-doctor' in project['scripts']  # installed scripts remain compatible


def test_json_errors_are_machine_readable(tmp_path, capsys):
    assert product_cli.main(['check', str(tmp_path / 'missing'), '--json']) == 2
    output = capsys.readouterr()
    assert not output.err
    report = json.loads(output.out)
    assert report['error']['code'] == 'invalid_input'
    assert report['status'] == 'input_error'


def test_saving_report_does_not_change_machine_status(tmp_path, capsys):
    config = tmp_path / 'config.json'
    config.write_text('{"intent":"not independently retained"}')
    output = tmp_path / 'report.json'
    assert product_cli.main(['check', str(config), '--output', str(output)]) == 3
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
    saved = json.loads(output.read_text())
    assert saved['status'] == 'unknown' and saved['kind'] == 'config_inspection'
