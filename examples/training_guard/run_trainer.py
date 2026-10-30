"""Pinned native GR00T/Trainer smoke: real public backbone and native action head.

This is a training-control qualification, not a trained-policy benchmark.
Use a new output directory and a pre-fault criterion for every candidate arm.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import Provenance, digest  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_record(value):
    import torch

    cpu = value.detach().cpu().contiguous()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
            'sha256': hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def seed():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def setup_data(binding, out, relative, data_seed):
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    from policy_doctor.training_guard_capture import save_capture

    checkpoint = out / 'checkpoint'
    checkpoint.mkdir()
    config = read(Path(binding['checkpoint']) / 'processor_config.json')
    config['processor_kwargs'].update(model_name=binding['model'], image_target_size=[230, 230],
                                      image_crop_size=[230, 230], use_relative_action=relative)
    write(checkpoint / 'processor_config.json', config)
    for name in ('statistics.json', 'embodiment_id.json'):
        with (checkpoint / name).open('xb') as target:
            target.write((Path(binding['checkpoint']) / name).read_bytes())
    processor = Gr00tN1d7Processor.from_pretrained(
        checkpoint, local_files_only=True,
        transformers_loading_kwargs={'local_files_only': True, 'trust_remote_code': False})
    assert processor.training and processor.state_action_processor.training
    assert not any((processor.use_percentiles, processor.use_mean_std, processor.clip_outliers,
                    processor.apply_sincos_state_encoding, processor.exclude_state, processor.state_dropout_prob))
    assert processor.max_state_dim == processor.max_action_dim == 64 and processor.max_action_horizon == 50
    assert processor.use_relative_action is relative
    tag = EmbodimentTag.UNITREE_G1
    dataset = ShardedSingleStepDataset(binding['dataset'], tag, processor.modality_configs[tag.value],
        shard_size=1, episode_sampling_rate=1 / 256, seed=data_seed, allow_padding=False)
    dataset.episode_loader.decoder_kwargs = {'device': 'cpu', 'num_ffmpeg_threads': 1}
    assert all(int(n) == 1 for n in dataset.shard_lengths)
    mixture = ShardedMixtureDataset([dataset], [1.0], processor, seed=data_seed, training=True,
                                   num_shards_per_epoch=3, override_pretraining_statistics=False)
    assert processor.state_action_processor.statistics == read(checkpoint / 'statistics.json')
    numeric_files = {name.removeprefix(binding['dataset'] + '/'): value for name, value in binding['files'].items()
                     if name.startswith(binding['dataset'] + '/')}
    context = {'files': binding['files'], 'dataset_content_sha256': digest(numeric_files),
               'processor': {'use_relative_action': relative, 'use_percentiles': False, 'use_mean_std': False,
                             'statistics_sha256': sha(checkpoint / 'statistics.json'),
                             'image_target_size': [230, 230], 'image_crop_size': [230, 230]},
               'native_source': {'revision': binding['native_revision'], 'files': binding['native_files']}}
    provenance = Provenance(dataset, context)

    class Tap:
        def __init__(self):
            self.calls = 0
            self.first = self.records = None

        def __call__(self, features):
            records = provenance.records(features) if self.calls == 0 else None
            batch = processor.collator(features)
            self.calls += 1
            if self.calls == 1:
                self.first, self.records = batch, records
                save_capture(batch, [row.source_key for row in records], out / 'first-capture.json')
                provenance.stop_capture()
            return batch

    return dataset, mixture, provenance, Tap()


def cleanup(mixture, provenance, iterator=None):
    if iterator is not None:
        generator = getattr(getattr(iterator, '_dataset_fetcher', None), 'dataset_iter', None)
        if generator is not None:
            generator.close()
    future = getattr(mixture, '_cache_job', None)
    executor = getattr(mixture, '_executor', None)
    errors = []
    if future is not None and not future.cancel():
        try:
            future.result(timeout=60)
        except Exception as error:
            errors.append({'type': type(error).__name__, 'message': str(error)})
    if executor is not None:
        threads = list(executor._threads)
        executor.shutdown(wait=True, cancel_futures=True)
        assert all(not thread.is_alive() for thread in threads)
    provenance.restore()
    result = {'executor_shutdown': True, 'background_errors': errors,
              'captured_source_samples': len(provenance.samples), 'capture_stopped': not provenance.capture_enabled}
    assert not errors, errors
    return result


def run(args, binding, out):
    import numpy as np
    import torch
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from safetensors import safe_open
    from transformers import TrainerCallback, TrainingArguments

    from policy_doctor.trainer_guard import TrainerStartupGuard
    from policy_doctor.training_guard import TargetContract, TargetContractViolation

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.__version__ == '2.9.0+cu126' and np.__version__ == '1.26.4'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    seed()
    cfg = Gr00tN1d7Config(model_name=binding['model'], model_dtype='float32', use_flash_attention=False,
        load_bf16=False, tune_llm=False, tune_visual=False, max_state_dim=64, max_action_dim=64,
        action_horizon=50, state_dropout_prob=0.0, use_percentiles=False, use_relative_action=True)
    assert cfg.select_layer == 12 and cfg.diffusion_model_cfg['num_layers'] == 16
    model = Gr00tN1d7(cfg, transformers_loading_kwargs={
        'local_files_only': True, 'trust_remote_code': False, 'dtype': torch.float32})
    assert type(model).__name__ == 'Gr00tN1d7' and model.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total > 1_000_000_000 and trainable > 1_000_000
    # Verify a real public checkpoint tensor, not merely an architecture/config.
    backbone_state = model.backbone.model.state_dict()
    embed_key = next(key for key in backbone_state if key.endswith('language_model.embed_tokens.weight'))
    with safe_open(str(Path(binding['model']) / 'model.safetensors'), framework='pt', device='cpu') as checkpoint:
        public_key = next(key for key in checkpoint.keys() if key.endswith('language_model.embed_tokens.weight'))
        public_row = checkpoint.get_slice(public_key)[0:1].to(torch.float32)
    assert torch.equal(backbone_state[embed_key][0:1].cpu(), public_row)
    pretrained_witness = tensor_record(public_row)
    del backbone_state, public_row
    selected_name = 'action_head.action_decoder.layer2.W'
    selected = dict(model.named_parameters())[selected_name]
    assert selected.requires_grad and selected.shape[0] == 32
    initial = tensor_record(selected[19])
    initial_values = selected[19].detach().cpu().clone().numpy()
    relative = args.phase not in ('native-fault', 'guard-fault')
    dataset, mixture, provenance, tap = setup_data(binding, out, relative, 43 if args.phase == 'unknown' else 42)
    targs = TrainingArguments(output_dir=str(out / 'trainer'), max_steps=2, per_device_train_batch_size=2,
        gradient_accumulation_steps=1, learning_rate=1e-4, weight_decay=0.01, optim='adamw_torch',
        dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
        bf16=False, fp16=False, tf32=False, report_to=[], save_strategy='no', eval_strategy='no',
        logging_strategy='steps', logging_steps=1, disable_tqdm=True, seed=42, data_seed=42,
        eval_on_start=False, skip_memory_metrics=True)
    ledger = {'optimizer_steps': 0, 'model_forward_calls': 0, 'gradient_norms': [], 'head_input_witness': None}
    gradients, optimizer_groups = [], []

    class OptimizerWitness(TrainerCallback):
        def on_train_begin(self, _args, _state, _control, **kwargs):
            optimizer = kwargs['optimizer']
            while hasattr(optimizer, 'optimizer'):
                optimizer = optimizer.optimizer

            def before(_optimizer, _args, _kwargs):
                gradient = selected.grad
                assert gradient is not None and torch.isfinite(gradient).all()
                norm = float(gradient[19].norm().item())
                assert norm > 0 and np.isfinite(norm)
                ledger['gradient_norms'].append(norm)
                gradients.append(gradient[19].detach().cpu().numpy().copy())
                group = next(group for group in _optimizer.param_groups
                             if any(parameter is selected for parameter in group['params']))
                optimizer_groups.append({key: group[key] for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize')})

            def after(_optimizer, _args, _kwargs):
                ledger['optimizer_steps'] += 1

            optimizer.register_step_pre_hook(before)
            optimizer.register_step_post_hook(after)

    trainer = Gr00tTrainer(model=model, args=targs, train_dataset=mixture, data_collator=tap,
                           callbacks=[OptimizerWitness()], multiprocessing_context='spawn')
    assert inspect.getsourcefile(type(trainer)) == str(Path(binding['native_source']) / 'gr00t/experiment/trainer.py')
    assert model.device.type == 'cuda'

    def forward_started(_module, _args, _kwargs):
        ledger['model_forward_calls'] += 1

    def head_input(_module, positional):
        if ledger['head_input_witness'] is not None:
            return
        action_inputs = positional[1]
        actual_action = action_inputs['action'].detach().cpu().numpy()
        actual_mask = action_inputs['action_mask'].detach().cpu().numpy()
        original_action = tap.first['inputs']['action'].detach().cpu().numpy()
        original_mask = tap.first['inputs']['action_mask'].detach().cpu().numpy()
        assert actual_action.dtype == original_action.dtype and np.array_equal(actual_action, original_action)
        assert np.array_equal(actual_mask, original_mask)
        ledger['head_input_witness'] = {'action': tensor_record(action_inputs['action']),
            'mask': tensor_record(action_inputs['action_mask']), 'source_mask_dtype': str(original_mask.dtype),
            'native_model_mask_dtype': str(actual_mask.dtype), 'numeric_mask_values_preserved': True,
            'primary_guard_tolerance_relaxed': False}

    model.register_forward_pre_hook(forward_started, with_kwargs=True)
    model.action_head.register_forward_pre_hook(head_input)
    guard = None
    iterator = None
    outcome, exit_code, blocked_report = 'trained', 0, None
    try:
        if args.phase == 'reference':
            loader = trainer.get_train_dataloader()
            iterator = iter(loader)
            batch = next(iterator)
            assert batch is tap.first and batch['inputs']['action'].shape == (2, 50, 64)
            records = tap.records
            bank = TargetContract.from_references([(batch, [row.source_key for row in records],
                                                   [row.provenance for row in records])], intent={
                'objective': 'Preserve declared native minmax-relative startup training targets',
                'scope': 'First actual native Trainer batch, two source samples; later batches unassessed',
                'model_profile': 'Public Qwen3-VL-2B backbone plus native newly initialized GR00T N1.7 action head',
                'dtype': 'float32', 'max_steps': 2, 'image_target_and_crop': [230, 230]})
            bank.save(out / 'contract.json')
            with (out / 'criterion.sha256').open('x') as stream:
                stream.write(bank.criterion_sha256 + '\n')
            (out / 'contract.json').chmod(0o444)
            (out / 'criterion.sha256').chmod(0o444)
            outcome = 'reference_frozen_before_fault'
        else:
            bank = TargetContract.load(args.contract, expected_sha256=args.criterion_sha256)
            if args.phase in ('guard-fault', 'guard-correct', 'replay', 'unknown'):
                guard = TrainerStartupGuard(trainer, bank, source_keys=provenance.source_keys, batches=1,
                                            on_scope_end=provenance.stop_capture)
            try:
                trainer.train()
            except TargetContractViolation as error:
                blocked_report = error.report
                outcome = 'target_blocked'
                exit_code = 1 if error.report['status'] == 'error' else 3
            if exit_code == 0:
                assert trainer.state.global_step == ledger['optimizer_steps'] == ledger['model_forward_calls'] == 2
                assert len(ledger['gradient_norms']) == 2 and torch.isfinite(trainer.loss)
                assert tensor_record(selected[19])['sha256'] != initial['sha256']
            else:
                assert trainer.state.global_step == ledger['optimizer_steps'] == ledger['model_forward_calls'] == 0
    finally:
        guard_summary = guard.close() if guard is not None else None
        if guard is not None:
            assert guard.trainer is None and guard.original_loss is None
        cleanup_summary = cleanup(mixture, provenance, iterator)
        write(out / 'cleanup.json', cleanup_summary)
    result = {'outcome': outcome, 'phase': args.phase, 'criterion_sha256': bank.criterion_sha256,
              'native_collations': tap.calls, 'ledger': ledger, 'guard': guard_summary, 'blocked_report': blocked_report,
              'initial_selected_parameter': initial, 'final_selected_parameter': tensor_record(selected[19]),
              'selected_parameter': selected_name, 'active_embodiment': 19, 'pretrained_embedding_witness': pretrained_witness,
              'total_parameters': total, 'trainable_parameters': trainable,
              'model_dtype': str(model.dtype), 'cuda_device': torch.cuda.get_device_name(),
              'cuda_capability': list(torch.cuda.get_device_capability()),
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'trainer_global_step': trainer.state.global_step,
              'native_files_patched': False, 'policy_quality_or_physical_claim': False,
              'later_batches_assessed': False, 'loss': float(trainer.loss.item()) if hasattr(trainer, 'loss') else None}
    witness_arrays = {'before': initial_values, 'after': selected[19].detach().cpu().numpy().copy()}
    witness_arrays.update({f'gradient_{index}': value for index, value in enumerate(gradients)})
    np.savez_compressed(out / 'parameter-witness.npz', **witness_arrays)
    result['parameter_witness_sha256'] = sha(out / 'parameter-witness.npz')
    result['optimizer_groups'] = optimizer_groups
    write(out / 'result.json', result)
    print(json.dumps({'outcome': outcome, 'steps': ledger['optimizer_steps'], 'phase': args.phase,
                      'criterion_sha256': bank.criterion_sha256}))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--phase', required=True, choices=('reference', 'native-fault', 'guard-fault', 'guard-correct', 'healthy', 'replay', 'unknown'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--criterion-sha256')
    args = parser.parse_args()
    if args.phase != 'reference' and (args.contract is None or args.criterion_sha256 is None):
        parser.error('candidate phases require the pre-fault --contract and --criterion-sha256')
    assert sys.flags.isolated and sys.dont_write_bytecode
    assert sha(args.binding) == args.binding_sha256
    binding = read(args.binding)
    assert not args.output.exists()
    args.output.mkdir(mode=0o700)
    assert all(sha(path) == expected for path, expected in binding['files'].items())
    sys.path.insert(0, binding['native_source'])

    def offline(event, _arguments):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise RuntimeError('Native training qualification is offline')

    sys.addaudithook(offline)
    try:
        code = run(args, binding, args.output)
    except BaseException as error:
        write(args.output / 'runtime-error.json', {'outcome': 'runtime_error', 'type': type(error).__name__, 'message': str(error)[:8192]})
        raise
    assert all(sha(path) == expected for path, expected in binding['files'].items())
    return code


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    raise SystemExit(main())
