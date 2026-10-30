"""Portable native training example: attach the installed guard to an existing trainer.

The delivered public simulation data and pretrained backbone are fixed inputs.
The native GR00T action head is newly initialized; no trained-policy claim follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def tensor_record(value):
    import torch

    cpu = value.detach().cpu().contiguous()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
            'sha256': hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def run(args):
    import numpy as np
    import torch

    assets = args.assets.resolve(strict=True)
    sys.path.insert(0, str(assets / 'native-source'))
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from safetensors import safe_open
    from transformers import TrainerCallback, TrainingArguments

    from policy_doctor.groot_training_guard import (
        Gr00tSourceObserver,
        Gr00tTrainingGuard,
        capture_gr00t_reference,
    )
    from policy_doctor.training_guard import TargetContractViolation
    from policy_doctor.training_guard_capture import save_capture

    if np.__version__ != '1.26.4' or torch.__version__ != '2.9.0+cu126':
        raise ValueError('Use the delivered pinned preview environment')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('This native model example requires one visible suitable CUDA GPU')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    model_root = assets / 'models/Qwen/Qwen3-VL-2B-Instruct'
    config = Gr00tN1d7Config(model_name=str(model_root), model_dtype='float32', use_flash_attention=False,
        load_bf16=False, tune_llm=False, tune_visual=False, max_state_dim=64, max_action_dim=64,
        action_horizon=50, state_dropout_prob=0.0, use_percentiles=False, use_relative_action=True)
    model = Gr00tN1d7(config, transformers_loading_kwargs={
        'local_files_only': True, 'trust_remote_code': False, 'dtype': torch.float32})
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert config.select_layer == 12 and config.diffusion_model_cfg['num_layers'] == 16
    selected = dict(model.named_parameters())['action_head.action_decoder.layer2.W']
    initial = tensor_record(selected[19])
    before_values = selected[19].detach().cpu().clone().numpy()
    with safe_open(str(model_root / 'model.safetensors'), framework='pt', device='cpu') as weights:
        key = next(key for key in weights.keys() if key.endswith('language_model.embed_tokens.weight'))
        public_row = weights.get_slice(key)[0:1].to(torch.float32)
    state = model.backbone.model.state_dict()
    key = next(key for key in state if key.endswith('language_model.embed_tokens.weight'))
    assert torch.equal(state[key][0:1].cpu(), public_row)
    public_witness = tensor_record(public_row)
    del state, public_row
    checkpoint = args.output / 'checkpoint'
    checkpoint.mkdir()
    processor_config = load(assets / 'checkpoint/processor_config.json')
    relative = args.mode not in ('native-fault', 'guard-fault')
    processor_config['processor_kwargs'].update(model_name=str(model_root), image_target_size=[230, 230],
                                               image_crop_size=[230, 230], use_relative_action=relative)
    write(checkpoint / 'processor_config.json', processor_config)
    for name in ('statistics.json', 'embodiment_id.json'):
        with (checkpoint / name).open('xb') as stream:
            stream.write((assets / 'checkpoint' / name).read_bytes())
    processor = Gr00tN1d7Processor.from_pretrained(checkpoint, local_files_only=True,
        transformers_loading_kwargs={'local_files_only': True, 'trust_remote_code': False})
    tag = EmbodimentTag.UNITREE_G1
    seed = 43 if args.mode == 'unknown' else 42
    dataset = ShardedSingleStepDataset(assets / 'dataset', tag, processor.modality_configs[tag.value],
        shard_size=8, episode_sampling_rate=1 / 256, seed=seed, allow_padding=False)
    dataset.episode_loader.decoder_kwargs = {'device': 'cpu', 'num_ffmpeg_threads': 1}
    assert min(int(value) for value in dataset.shard_lengths) > 1
    mixture = ShardedMixtureDataset([dataset], [1.0], processor, seed=seed, training=True,
                                   num_shards_per_epoch=3, override_pretraining_statistics=False)
    assert processor.state_action_processor.statistics == load(checkpoint / 'statistics.json')
    ledger = {'model_forward_calls': 0, 'optimizer_steps': 0, 'gradient_norms': []}
    gradients, optimizer_groups = [], []

    class Witness(TrainerCallback):
        def on_train_begin(self, _args, _state, _control, **kwargs):
            optimizer = kwargs['optimizer']
            while hasattr(optimizer, 'optimizer'):
                optimizer = optimizer.optimizer

            def before(_optimizer, _args, _kwargs):
                gradient = selected.grad[19]
                assert torch.isfinite(gradient).all() and gradient.norm() > 0
                ledger['gradient_norms'].append(float(gradient.norm().item()))
                gradients.append(gradient.detach().cpu().numpy().copy())
                group = next(group for group in _optimizer.param_groups
                             if any(parameter is selected for parameter in group['params']))
                optimizer_groups.append({key: group[key] for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize')})

            def after(_optimizer, _args, _kwargs):
                ledger['optimizer_steps'] += 1

            optimizer.register_step_pre_hook(before)
            optimizer.register_step_post_hook(after)

    class Tap:
        def __init__(self):
            self.observer, self.first, self.calls = None, None, 0

        def __call__(self, features):
            keys = self.observer.source_keys(features) if self.calls == 0 and self.observer is not None else None
            native = processor.collator(features)
            self.calls += 1
            if self.first is None:
                self.first = native
                if keys is not None:
                    save_capture(native, keys, args.output / 'first-capture.json')
            return native

    tap = Tap()
    arguments = TrainingArguments(output_dir=str(args.output / 'trainer'), max_steps=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=1, learning_rate=1e-4,
        weight_decay=0.01, optim='adamw_torch', dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, bf16=False, fp16=False, tf32=False, report_to=[], save_strategy='no',
        eval_strategy='no', logging_strategy='steps', logging_steps=1, disable_tqdm=True, seed=42,
        data_seed=42, eval_on_start=False, skip_memory_metrics=True)
    # This is the existing native trainer. Its model/data/configuration exist
    # before the optional RoboClue attachment below.
    trainer = Gr00tTrainer(model=model, args=arguments, train_dataset=mixture, data_collator=tap,
                           callbacks=[Witness()], multiprocessing_context='spawn')

    def forward(_module, _args, _kwargs):
        ledger['model_forward_calls'] += 1

    model.register_forward_pre_hook(forward, with_kwargs=True)
    observer = guard = bank = None
    outcome, exit_code, blocked = 'trained', 0, None
    try:
        if args.mode == 'reference':
            bank = capture_gr00t_reference(trainer, output=args.output / 'contract.json',
                intent={'objective': 'Preserve intended native minmax-relative startup targets',
                        'cohort': 'First native batch of two samples from multi-sample shards',
                        'model': 'Public frozen Qwen backbone plus newly initialized native GR00T head'})
            (args.output / 'criterion.sha256').write_text(bank.criterion_sha256 + '\n')
            outcome = 'reference_frozen_before_fault'
        elif args.mode in ('native-fault', 'healthy'):
            # Observation only: no target checker protects these native controls.
            observer = Gr00tSourceObserver(trainer)
            tap.observer = observer
            native_collator = trainer.data_collator

            def passive(features):
                native = native_collator(features)
                if tap.calls == 1:
                    observer.stop_capture()
                return native

            trainer.data_collator = passive
            trainer.train()
        else:
            # Reusable installed adapter: the integration consists of this
            # context around the existing native trainer.train() call.
            with Gr00tTrainingGuard(trainer, contract=args.contract, criterion_sha256=args.criterion_sha256) as guard:
                tap.observer = guard.observer
                trainer.train()
    except TargetContractViolation as error:
        blocked = error.report
        outcome = 'target_blocked'
        exit_code = 1 if blocked['status'] == 'error' else 3
    finally:
        if observer is not None:
            observer.stop_capture()
        future, executor = getattr(mixture, '_cache_job', None), getattr(mixture, '_executor', None)
        try:
            if future is not None and not future.cancel():
                future.result(timeout=60)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if observer is not None:
                observer.close()
        write(args.output / 'cleanup.json', {'executor_shutdown': True,
              'source_hooks_restored': 'get_shard' not in dataset.__dict__, 'background_errors': []})
    expected_steps = 0 if args.mode == 'reference' or outcome == 'target_blocked' else 2
    assert ledger['optimizer_steps'] == ledger['model_forward_calls'] == expected_steps
    assert trainer.state.global_step == expected_steps
    arrays = {'before': before_values, 'after': selected[19].detach().cpu().numpy().copy()}
    arrays.update({f'gradient_{i}': value for i, value in enumerate(gradients)})
    np.savez_compressed(args.output / 'parameter-witness.npz', **arrays)
    result = {'status': outcome, 'mode': args.mode, 'ledger': ledger, 'blocked_report': blocked,
              'criterion_sha256': bank.criterion_sha256 if bank else args.criterion_sha256,
              'initial_active_parameter_slice': initial, 'final_active_parameter_slice': tensor_record(selected[19]),
              'pretrained_embedding_witness': public_witness, 'optimizer_groups': optimizer_groups,
              'guard': guard.summary() if guard else None, 'native_collations': tap.calls,
              'native_shard_min_samples': min(int(v) for v in dataset.shard_lengths),
              'native_shard_max_samples': max(int(v) for v in dataset.shard_lengths),
              'total_model_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'native_source_patched': False, 'old_server_paths_required': False,
              'scope': 'Two-step native startup target study on public simulation data; no trained-policy/physical/human-usefulness claim.'}
    write(args.output / 'result.json', result)
    print(json.dumps({'status': outcome, 'mode': args.mode, 'optimizer_steps': expected_steps,
                      'criterion_sha256': result['criterion_sha256']}))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parent / 'assets')
    parser.add_argument('--mode', required=True, choices=('reference', 'native-fault', 'guard-fault', 'corrected', 'healthy', 'replay', 'unknown'))
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--criterion-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode != 'reference' and (args.contract is None or args.criterion_sha256 is None):
        parser.error('Candidate runs require the contract and pin retained before changing the pipeline')
    if args.output.exists() or args.output.is_symlink():
        parser.error('--output must name a new directory')
    args.output.mkdir(mode=0o700)

    def offline(event, _arguments):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise RuntimeError('The delivered native preview runs offline')

    sys.addaudithook(offline)
    try:
        return run(args)
    except BaseException as error:
        write(args.output / 'runtime-error.json', {'type': type(error).__name__, 'message': str(error)[:8192]})
        raise


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    raise SystemExit(main())
