"""Check supplied existing dataset files read-only; save compact, value-free evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy_doctor.rollouts import compare_rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lerobot', type=Path, required=True)
    parser.add_argument('--robomimic', type=Path, required=True)
    parser.add_argument('--lerobot-episode', type=int, default=0)
    parser.add_argument('--robomimic-episode', default='demo_0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for name, path, episode, index in [('lerobot', args.lerobot, args.lerobot_episode, 'frame_index'),
                                      ('robomimic', args.robomimic, args.robomimic_episode, None)]:
        report = compare_rollouts(str(path), str(path), reference_episode=episode, candidate_episode=episode,
                                  alignment_key=index)
        assert not report['observed_difference']
        assert report['streams'] and all(s['status'] == 'equal' for s in report['streams'])
        records[name] = {'read_compatible': True, 'comparison_status': report['status'],
            'alignment': report['alignment'], 'sha256': report['artifacts'][0]['sha256'],
            'size_bytes': report['artifacts'][0]['size_bytes'], 'episode': episode,
            'streams': [{k: stream[k] for k in ('key', 'reference_shape', 'rows_compared', 'role')} for stream in report['streams']],
            'excluded_streams': report['artifacts'][0].get('excluded_streams', [])}
    result = {'scope': 'Existing-file native numeric reader self-comparisons; no diagnosis accuracy, dataset-wide validation, inference or robot execution.',
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {'read_compatible': record['read_compatible'], 'status': record['comparison_status'],
                           'streams': len(record['streams']), 'rows': record['streams'][0]['rows_compared']}
                      for name, record in records.items()}))


if __name__ == '__main__':
    main()
