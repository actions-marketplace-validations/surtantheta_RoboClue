"""Run RC006's real detector on small generated controls, without a robotics stack.

This is a contribution-path example, not independent human or native execution
qualification. No new detector/plugin is loaded from an artifact or the network.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from policy_doctor.artifacts import Artifact, Snapshot
from policy_doctor.diagnostics import dataset_statistics_findings
from policy_doctor.rule_catalog import get_rule, rules_for_code, validate_catalog


def run_walkthrough() -> dict:
    """Exercise source-backed code discovery and five controls against the real check."""
    assert not validate_catalog()
    assert get_rule('RC006')['id'] == 'RC006'
    scenarios = {
        'broken': ({'mean': [2.], 'std': [.25], 'min': [0.], 'max': [1.]}, 'error'),
        'healthy': ({'mean': [.5], 'std': [.25], 'min': [0.], 'max': [1.]}, 'consistent'),
        'corrected': ({'mean': [.5], 'std': [.25], 'min': [0.], 'max': [1.]}, 'consistent'),
        'missing': (None, 'unknown'),
        'unsupported_shape': ({'mean': [[.5]], 'std': [[.25]], 'min': [[0.]], 'max': [[1.]]}, 'unknown'),
    }
    outcomes = {}
    with tempfile.TemporaryDirectory(prefix='roboclue-rule-contribution-') as directory:
        root = Path(directory)
        for name, (entry, expected) in scenarios.items():
            folder = root / name
            folder.mkdir()
            document = {'action': entry} if entry is not None else None
            raw = json.dumps(document).encode()
            path = folder / 'stats.json'
            path.write_bytes(raw)
            before = hashlib.sha256(raw).hexdigest()
            snapshot = Snapshot('dataset', folder, {'stats': document},
                features={'action': {'dtype': 'float32', 'shape': [1], 'names': ['joint.pos']}},
                artifacts=[Artifact(str(path), 'dataset_stats', before, len(raw))],
                json_documents={str(path): document})
            findings = dataset_statistics_findings(snapshot)
            relevant = [f for f in findings if any(r['id'] == 'RC006' for r in rules_for_code(f['code']))]
            status = 'error' if any(f['status'] == 'error' for f in relevant) else 'unknown' if any(f['status'] == 'unknown' for f in relevant) else 'consistent'
            assert status == expected, (name, findings)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before, name
            outcomes[name] = {'rule': 'RC006', 'status': status, 'codes': [f['code'] for f in relevant],
                              'input_unchanged': True}
    return {'kind': 'rule_contribution_walkthrough', 'status': 'pass', 'controls': outcomes,
            'scope': 'Generated saved-summary controls only. No rows, framework version, native training, human or physical behavior was qualified.'}


if __name__ == '__main__':
    print(json.dumps(run_walkthrough(), indent=2))
