"""Generate controlled native Parquet/sidecar role checks, without a robot or policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from policy_doctor.dispatch import audit_dispatch, make_dispatch_event


def create(destination: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(destination).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    recording = root / 'native.parquet'
    pq.write_table(pa.table({'episode_index': [0, 0], 'frame_index': [10, 11], 'action': [[2.], [2.]]}), recording)
    digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    reports = {}
    for case, role, bound in [('canonical', 'canonical_endpoint', True), ('contradiction', 'driver_return', True),
                              ('unknown_role', 'unknown', True), ('unbound', 'driver_return', False)]:
        events = [make_dispatch_event(f'tick-{i}', frame_index=i, canonical_dispatch={'joint.pos': 1.},
            canonical_endpoint={'joint.pos': 2.}, driver_input={'joint.pos': 4.}, driver_return={'joint.pos': .5},
            recorded_role=role) for i in [10, 11]]
        trace = root / f'{case}.json'
        trace.write_text(json.dumps({'schema_version': 1, 'kind': 'action_dispatch_evidence',
            'recording_sha256': digest if bound else None, 'episode_index': 0, 'events': events}, indent=2)+'\n')
        report = audit_dispatch(str(trace), str(recording), episode=0, action_names=['joint.pos'])
        (root / f'{case}-report.json').write_text(json.dumps(report, indent=2)+'\n')
        reports[case] = report['status']
    (root / 'README.md').write_text(
        '# Controlled action-boundary example\n\n'
        'Native recorded endpoint2 differs from interpolated dispatch1, driver input4 and return0.5. '
        'Canonical endpoint declaration passes; driver-return declaration contradicts the recording. '
        'Unknown role and missing hash binding remain unknown.\n\n'
        'Generated numbers; no robot, driver, policy, simulator or physical repair is demonstrated.\n\n'
        '```sh\npolicy-doctor audit-dispatch contradiction.json --recording native.parquet --episode 0 --action-names joint.pos\n```\n')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    print(json.dumps(create(parser.parse_args().destination), indent=2))
