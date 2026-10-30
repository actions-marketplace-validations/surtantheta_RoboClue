"""A severity filter must not approve a declared saved-target check without intent."""
import json

import numpy as np
import pytest

from policy_doctor.product_cli import main
from policy_doctor.training_guard_capture import save_capture


@pytest.mark.parametrize('declare_reference', [False, True])
def test_unknown_training_check_cannot_clear_error_only_gate(tmp_path, capsys, declare_reference):
    capture = tmp_path / 'capture'
    save_capture({'inputs': {'action': np.zeros((1, 1, 1), dtype=np.float32),
                             'action_mask': np.ones((1, 1, 1), dtype=bool)}}, ['row0'], capture)
    argv = ['check', str(capture), '--kind', 'training', '--fail-on', 'error', '--json']
    if declare_reference:
        argv += ['--reference', str(capture)]
    assert main(argv) == 3
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'unknown' and report['code'] == 'missing_retained_reference'
