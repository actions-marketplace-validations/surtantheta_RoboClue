"""CPU integration against an explicitly trusted, pinned LeRobot source checkout.

Development validation only; the installed doctor never imports artifact Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    expected = '146d8474932c8b566717eb2738ba459337ab71c4'
    if revision != expected:
        raise SystemExit(f'Expected trusted upstream {expected}, got {revision}')
    sys.path.insert(0, str(args.upstream.resolve() / 'src'))
    import numpy as np
    import torch
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    from policy_doctor.capture import audit_capture, capture_runtime
    from policy_doctor.trace import trace_actions

    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[1]
    demo = project / 'examples/demo'
    args.work.mkdir(parents=True, exist_ok=True)
    # Exercise upstream's actual public loading/overrides, not fixture serializers.
    stats = {
        'action': {'mean': torch.tensor([2., .5]), 'std': torch.tensor([.5, .25])},
        'observation.state': {'mean': torch.tensor([10., .5]), 'std': torch.tensor([2., .25])},
    }
    pre = PolicyProcessorPipeline.from_pretrained(str(demo / 'checkpoint'),
        config_filename='policy_preprocessor.json', overrides={'normalizer_processor': {'stats': stats}})
    post = PolicyProcessorPipeline.from_pretrained(str(demo / 'checkpoint'),
        config_filename='policy_postprocessor.json', overrides={'unnormalizer_processor': {'stats': stats}})
    config = json.loads((demo / 'checkpoint/config.json').read_text())
    deployment = json.loads((demo / 'healthy.json').read_text())
    missing = capture_runtime(args.work / 'unwired', policy_config=config,
                              preprocessor=pre, postprocessor=post, deployment=deployment)
    assert any(f['code'] == 'capture.relative_binding' and f['status'] == 'error' for f in missing['findings'])
    post.steps[-1].relative_step = pre.steps[0]
    captured = capture_runtime(args.work / 'wired', policy_config=config,
                              preprocessor=pre, postprocessor=post, deployment=deployment)
    assert captured['status'] == 'pass', captured
    assert captured['capture']['relative_bindings'] == [{'post_step': 1, 'pre_step': 0, 'status': 'paired'}]
    sample = json.loads((demo / 'sample.json').read_text())
    sample['expected_commands'] = [[12., .5], [12.5, .75], [13., .25]]
    sample_path = args.work / 'sample.json'
    sample_path.write_text(json.dumps(sample))
    traced = trace_actions(str(args.work / 'wired/checkpoint'), str(sample_path), str(args.work / 'wired/deployment.json'))
    assert traced['status'] == 'pass'
    # Execute real CPU processors on the same supplied output/anchor. No model runs.
    pre({'observation.state': torch.tensor([[10., .5]])})
    decoded = post({'action': torch.tensor([sample['model_output']])})['action'].squeeze(0).numpy()
    error = float(np.max(np.abs(decoded - np.asarray(traced['commands']))))
    assert error < 1e-7, (decoded, traced['commands'])
    assert audit_capture(str(args.work / 'wired'), str(demo / 'dataset'))['status'] == 'pass'
    files = ['processor/pipeline.py', 'processor/normalize_processor.py', 'processor/relative_action_processor.py']
    hashes = {name: hashlib.sha256((args.upstream / 'src/lerobot' / name).read_bytes()).hexdigest() for name in files}
    record = {'status': 'pass', 'source_revision': revision, 'source_hashes': hashes,
              'python': sys.version.split()[0], 'torch': torch.__version__, 'device': 'cpu',
              'checks': ['actual upstream public pipeline loading', 'actual paired runtime stats override',
                         'unwired absolute decoder detected from live object topology',
                         'wired capture passes and records topology', 'captured supplied-output trace matches real CPU processor execution'],
              'max_absolute_command_error': error, 'commands': decoded.tolist(),
              'scope': 'Pinned loaded processor integration and supplied-output arithmetic; no model inference, driver, hardware or independent user study.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
