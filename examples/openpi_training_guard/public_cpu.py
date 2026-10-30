"""Portable real OpenPI CPU loader witness on selected public simulation rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--openpi-source', type=Path, required=True)
    parser.add_argument('--product-source', type=Path, help='Optional checkout src directory; otherwise use the installed RoboClue package')
    parser.add_argument('--phase', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--pin')
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir()
    sys.path[:0] = [str(args.openpi_source / 'src'), str(args.openpi_source / 'packages/openpi-client/src')]
    if args.product_source is not None:
        sys.path.insert(0, str(args.product_source))
    import av
    import jax
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from openpi import transforms
    from openpi.models.model import Observation
    from openpi.shared.normalize import NormStats
    from openpi.training.config import DataConfig
    from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader, transform_dataset

    from policy_doctor.openpi_training_guard import (
        MARKER,
        OpenpiSourceDataset,
        OpenpiTrainingGuard,
        capture_openpi_reference,
    )
    from policy_doctor.training_guard import TargetContractViolation

    assert np.__version__ == '1.26.4' and torch.version.cuda is None
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    source = args.dataset / 'data/chunk-000/episode_000000.parquet'
    rows = pq.read_table(source, columns=['action', 'observation.state']).slice(0, 64).to_pylist()
    assert len(rows) == 64 and len(rows[0]['action']) == 43
    horizon, dims = 8, 32
    actions = np.asarray([row['action'][:dims] for row in rows], dtype=np.float32)
    states = np.asarray([row['observation.state'][:dims] for row in rows], dtype=np.float32)
    deltas = np.stack([actions[i:i + horizon] - states[i][None] for i in range(len(rows) - horizon + 1)])
    stats = {
        'state': NormStats(mean=states.mean(0).astype(np.float64), std=states.std(0).astype(np.float64) + 1e-5),
        'actions': NormStats(mean=deltas.reshape(-1, dims).mean(0).astype(np.float64),
                             std=deltas.reshape(-1, dims).std(0).astype(np.float64) + 1e-5),
    }
    video = args.dataset / 'videos/chunk-000/observation.images.ego_view/episode_000000.mp4'
    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == 64:
                break
    assert len(frames) == 64

    class Raw:
        def __len__(self):
            return len(rows) - horizon + 1

        def __getitem__(self, index):
            return {'raw_state': states[index].copy(), 'raw_actions': actions[index:index + horizon].copy(),
                    'raw_image': frames[index], 'raw_image_mask': np.bool_(True)}

    relative = args.phase not in ('native-fault', 'guard-fault', 'legal-reference', 'legal-absolute')
    config = DataConfig(repo_id='declared_public_arena_arithmetic_profile', norm_stats=stats,
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
            'state': 'raw_state', 'actions': 'raw_actions', 'image': {'base_0_rgb': 'raw_image'},
            'image_mask': {'base_0_rgb': 'raw_image_mask'}})]),
        data_transforms=transforms.Group(inputs=[transforms.DeltaActions([True] * dims)] if relative else []),
        model_transforms=transforms.Group(inputs=[transforms.ResizeImages(224, 224)]))
    transformed = transform_dataset(Raw(), config)
    wrapped = args.phase not in ('native-fault', 'healthy')
    dataset = OpenpiSourceDataset(transformed, dataset_root=args.dataset) if wrapped else transformed
    inner = TorchDataLoader(dataset, local_batch_size=2, num_batches=3, num_workers=0,
                            framework='pytorch', shuffle=True, seed=43 if args.phase == 'unknown' else 42)
    loader = DataLoaderImpl(config, inner)
    counts, observations, captures = {}, [], []
    seen_keys = None

    def record_tensor(value, *, data=True):
        array = value.detach().cpu().numpy()
        result = {'dtype': array.dtype.str, 'shape': list(array.shape),
                  'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        if data:
            result['data'] = array.tolist()
        return result

    def profile(frame, event, argument):
        nonlocal seen_keys
        module, name = frame.f_globals.get('__name__'), frame.f_code.co_name
        if event == 'call' and (module, name) in (
            ('openpi.transforms', '__call__'), ('openpi.training.data_loader', '_collate_fn'),
            ('openpi.models.model', 'from_dict')):
            owner = type(frame.f_locals.get('self')).__name__ if 'self' in frame.f_locals else ''
            key = module + ':' + owner + '.' + name
            counts[key] = counts.get(key, 0) + 1
        if event == 'call' and module == 'openpi.models.model' and name == 'from_dict':
            assert MARKER not in frame.f_locals['data'], 'Private source metadata reached Observation.from_dict'
        if event == 'return' and module == 'policy_doctor.openpi_training_guard' and name == '_strip':
            if argument is not None and seen_keys is None:
                seen_keys = argument[1]
                captures.append({'boundary': 'transformed', 'source_keys': seen_keys,
                                 'action': record_tensor(argument[0]['actions'])})
        if event == 'return' and module == 'openpi.training.data_loader' and name == '__iter__':
            if isinstance(argument, tuple) and len(argument) == 2 and isinstance(argument[0], Observation):
                observations.append({'action': record_tensor(argument[1]),
                    'state': record_tensor(argument[0].state),
                    'image_sha256': record_tensor(argument[0].images['base_0_rgb'], data=False)['sha256']})

    sys.setprofile(profile)
    delivered, blocked, guard_summary = 0, None, None
    model_inputs = []

    class Consumer(torch.nn.Module):
        """Controlled real Torch hook consumer; explicitly NOT Pi0 or a trained policy."""
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, observation, actions):
            self.calls += 1
            model_inputs.append(record_tensor(actions))
            return actions.square().mean()

    model = Consumer()

    def consume(original):
        observation, actions = original
        # Exact original pinned train_pytorch.py action/device recipe.
        observation = jax.tree.map(lambda x: x.to('cpu'), observation)
        actual = actions.to(torch.float32).to('cpu')
        if args.phase == 'post-delivery-fault':
            actual = actual.clone()
            actual[0, 0, 0] += 1.0
        elif args.phase == 'wrong-model-dtype':
            actual = actions
        elif args.phase == 'reordered-model-batch':
            actual = actual.flip(0)
        return model(observation, actual)

    criterion_pin = args.pin
    if args.phase in ('reference', 'legal-reference'):
        contract = capture_openpi_reference(loader, output=args.output / 'contract.json', batches=2, intent={
            'objective': 'Preserve explicitly declared public-row arithmetic target recipe',
            'use_delta_actions': relative, 'selected_action_and_state_slice': [0, dims], 'horizon': horizon,
            'source_columns': ['action', 'observation.state'], 'source_file_sha256': sha(source),
            'coordinate_names': info['features']['action']['names'][:dims],
            'model_or_physical_semantics_claim': False})
        criterion_pin = contract.criterion_sha256
        (args.output / 'criterion.sha256').write_text(criterion_pin + '\n')
        outcome = 'reference_frozen'
    elif args.phase in ('native-fault', 'healthy'):
        iterator = iter(loader)
        try:
            for observation, actual in iterator:
                assert isinstance(observation, Observation) and actual.shape == (2, horizon, dims)
                assert torch.isfinite(actual).all()
                delivered += 1
                consume((observation, actual))
        finally:
            iterator.close()
        outcome = 'native_accepted'
    else:
        guard = OpenpiTrainingGuard(loader, contract=args.contract, criterion_sha256=args.pin, batches=2, model=model)
        later_hook = None
        try:
            with guard:
                iterator = iter(guard)
                if args.phase == 'unpaired-model-input':
                    model(None, torch.zeros((2, horizon, dims)))
                elif args.phase in ('swapped-model-batch', 'duplicate-model-batch'):
                    first, second = next(iterator), next(iterator)
                    delivered += 2
                    if args.phase == 'duplicate-model-batch':
                        consume(first)
                        consume(first)
                    else:
                        consume(second)
                else:
                    if args.phase == 'later-pre-hook':
                        later_hook = model.register_forward_pre_hook(lambda _model, arguments: arguments)
                    for original in iterator:
                        assert isinstance(original[0], Observation) and original[1].shape == (2, horizon, dims)
                        delivered += 1
                        consume(original)
        except TargetContractViolation as error:
            blocked = error.report
        finally:
            if later_hook is not None:
                later_hook.remove()
        guard_summary = guard.summary()
        outcome = 'blocked' if blocked else 'delivered'
    sys.setprofile(None)
    result = {'phase': args.phase, 'outcome': outcome, 'criterion_sha256': criterion_pin,
        'deliveries': delivered, 'blocked_report': blocked, 'guard': guard_summary,
        'native_calls': counts, 'native_observation_calls': sum(value for name, value in counts.items() if name.endswith('.from_dict')),
        'native_outputs': observations, 'source_captures': captures,
        'model_forwards': model.calls, 'actual_model_inputs': model_inputs,
        'optimizer_steps': 0, 'physical_or_checkpoint_claim': False,
        'model_scope': 'Controlled real torch.nn.Module forward-hook consumer, not Pi0 or trained policy',
        'normalization_statistics': {key: {'mean': value.mean.tolist(), 'std': value.std.tolist()} for key, value in stats.items()},
        'profile': {'action_slice': [0, dims], 'horizon': horizon, 'native_delta_enabled': relative,
                    'statistics': 'Derived from selected actual public values; not a pretrained policy profile'},
        'scope': 'Original native Repack/Delta/Normalize/Resize/TorchDataLoader/Observation; custom explicit public-Parquet input adapter; no model or hardware'}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'outcome': outcome, 'deliveries': delivered, 'criterion_sha256': criterion_pin}))
    return 1 if blocked and blocked['status'] == 'error' else 3 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
