"""Real resource-limited child processes, containing generated stdlib fixtures only."""
import sys

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_processing import MAX_RESULT, _invoke


def generated_worker(tmp_path, name):
    script = tmp_path / name
    script.write_text('''import hashlib, json, pathlib, sys
r=json.load(sys.stdin)
out=pathlib.Path(r['output'])
raw=json.dumps({'execution_status':'complete','generated_fixture':'x'*r['size']}).encode()
(out/'result.json').write_bytes(raw)
receipt={'execution_status':'complete','result':{'path':'result.json','sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}}
(out/'worker_receipt.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
''')
    workspace = tmp_path / (name + '-workspace')
    workspace.mkdir()
    return script, workspace


def test_fixed_source_worker_returns_bounded_receipt_for_larger_raw_evidence(tmp_path):
    script, work = generated_worker(tmp_path, 'native_processor_source_worker.py')
    reply = _invoke(script, {'python': sys.executable},
                    {'output': str(work / 'output'), 'size': MAX_RESULT * 2}, work)
    assert reply['execution_status'] == 'complete'
    assert set(reply) == {'execution_status', 'result'}
    assert reply['result']['path'] == 'result.json'
    assert reply['result']['size_bytes'] > MAX_RESULT


def test_other_worker_cannot_use_source_receipt_to_bypass_existing_result_bound(tmp_path):
    script, work = generated_worker(tmp_path, 'other_worker.py')
    with pytest.raises(InputError):
        _invoke(script, {'python': sys.executable},
                {'output': str(work / 'output'), 'size': MAX_RESULT * 2}, work)


@pytest.mark.parametrize('kind', ['native_processor_source_attempt', 'native_processor_source_view'])
def test_source_native_replay_dispatches_fixed_parent_for_attempt_or_view(tmp_path, monkeypatch, kind):
    import json

    from policy_doctor import native_source_processing
    from policy_doctor.native_processing import reproduce_native_processing

    root = tmp_path / 'retained'
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'kind': kind}))
    calls = []

    def reproduce(directory, **kwargs):
        calls.append((directory, kwargs))
        return {'kind': 'native_processor_source_reproduction', 'status': 'pass'}

    monkeypatch.setattr(native_source_processing, 'reproduce_native_processor_source', reproduce)
    assert reproduce_native_processing(root, runtimes={'configured': True})['status'] == 'pass'
    assert calls == [(root, {'runtimes': {'configured': True}})]
