"""Portable, CPU-only GR00T N1.7 startup cohort; no weights or optimizer."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import random
import stat
import sys
from pathlib import Path

NATIVE_REVISION = '944fe70c4fd54e3fcb80a7d7258a328f9c061176'
PROFILE_SCOPE = 'Two identified startup samples; remaining training data, model and physical behavior unassessed.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files(root, *, limit=512 * 1024**2):
    """Bound this preview's file intake before opening bodies."""
    root = Path(root).resolve(strict=True)
    candidates = sorted(root.rglob('*'))
    require(len(candidates) <= 8192, 'Preview supports at most 8192 dataset/source entries')
    result, total = {}, 0
    for path in candidates:
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f'Unsupported special file or link: {path}')
        if stat.S_ISREG(mode):
            total += path.stat().st_size
            require(total <= limit, 'Preview file-byte budget exceeded')
            result[str(path)] = sha(path)
    return result


def load(path):
    from policy_doctor.conversion_fidelity import _read_small

    raw, _ = _read_small(str(path), limit=4 * 1024**2)
    return json.loads(raw)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def seed():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def prepare_checkpoint(packet, output, *, relative):
    """Create an owned profile; never edit the delivered inputs or criterion."""
    output = Path(output)
    output.mkdir()
    config = load(Path(packet) / 'checkpoint/processor_config.json')
    config['processor_kwargs'].update(
        model_name=str(Path(packet).resolve() / 'processor-assets'),
        image_target_size=[256, 256], image_crop_size=[230, 230],
        use_relative_action=relative,
    )
    write(output / 'processor_config.json', config)
    for name in ('statistics.json', 'embodiment_id.json'):
        with (output / name).open('xb') as stream:
            stream.write((Path(packet) / 'checkpoint' / name).read_bytes())
    return output


def build_probe(packet, checkpoint, *, contract=None, max_samples=2):
    """Construct genuine upstream components, then attach at their collator.

    This factory is the startup-cohort integration point. An existing trainer
    should call the gate before training and retain its own full-data loader.
    Do not reuse the bounded probe after close() or treat its bank as coverage
    of the full training population.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from policy_doctor.training_guard import _batch_arrays  # same declared batch schema

    packet = Path(packet).resolve(strict=True)
    native = packet / 'native-source'
    dataset_root = packet / 'dataset'
    native_files = files(native, limit=4 * 1024**2)
    dataset_files = files(dataset_root)
    require(str(native) not in sys.path, 'Use a fresh process for each startup probe')
    sys.path.insert(0, str(native))
    from adapter import NativeSession, digest
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    for cls in (Gr00tN1d7Processor, ShardedSingleStepDataset, ShardedMixtureDataset):
        require(Path(inspect.getsourcefile(cls)).resolve().is_relative_to(native),
                'A different GR00T source is already imported; start with the declared native source')
    require(np.__version__ == '1.26.4' and torch.__version__ == '2.9.0+cpu',
            'This preview requires the delivered CPU dependency lock')
    require(torch.version.cuda is None, 'This preview is CPU preprocessing only')
    torch.set_num_threads(1)
    seed()
    processor = Gr00tN1d7Processor.from_pretrained(
        checkpoint, transformers_loading_kwargs={'local_files_only': True, 'trust_remote_code': False},
        local_files_only=True,
    )
    require(type(processor).__name__ == 'Gr00tN1d7Processor'
            and type(processor.collator).__name__ == 'Gr00tN1d7DataCollator'
            and type(processor.processor).__name__ == type(processor.collator.processor).__name__ == 'Qwen3VLProcessor',
            'Unsupported native processor/collator')
    require(Path(inspect.getsourcefile(type(processor.collator))).resolve().is_relative_to(native),
            'A different native collator source is already imported')
    require(processor.training and processor.state_action_processor.training, 'Training preprocessing must be active')
    require(processor.use_relative_action == processor.state_action_processor.use_relative_action,
            'Native relative-action declarations disagree')
    require(processor.max_action_dim == processor.max_state_dim == 64 and processor.max_action_horizon == 50,
            'Unsupported native target shape profile')
    require(not any((processor.use_percentiles, processor.use_mean_std, processor.clip_outliers,
                     processor.apply_sincos_state_encoding, processor.exclude_state, processor.state_dropout_prob)),
            'This preview declares min/max normalization without additional target transforms')
    tag = EmbodimentTag.UNITREE_G1
    profile = processor.modality_configs[tag.value]
    require(list(profile['action'].delta_indices) == list(range(50))
            and list(profile['state'].delta_indices) == [0], 'Unsupported native source-row selection')
    seed()
    dataset = ShardedSingleStepDataset(str(dataset_root), tag, profile, shard_size=1,
                                      episode_sampling_rate=1 / 256, seed=42, allow_padding=False)
    dataset.episode_loader.decoder_kwargs = {'device': 'cpu', 'num_ffmpeg_threads': 1}
    lengths = [int(item['length']) for item in dataset.episode_loader.episodes_metadata]
    eligible = [length - 49 for length in lengths if length >= 50]
    require(eligible and max(eligible) <= 256 and all(int(n) == 1 for n in dataset.shard_lengths),
            'This preview requires exactly one selected sample per shard')
    mixture = ShardedMixtureDataset([dataset], [1.0], processor, seed=42, training=True,
                                   num_shards_per_epoch=3, override_pretraining_statistics=False)
    require(processor.state_action_processor.statistics == load(Path(checkpoint) / 'statistics.json'),
            'Native mixture changed declared normalization statistics')
    # Context that enters the criterion is independent of the recipient path.
    # Absolute paths are used only for this process's local source-file lookups.
    context = {
        'files': {**native_files, **dataset_files},
        'dataset_content_sha256': digest({str(Path(name).relative_to(dataset_root)): value
                                          for name, value in dataset_files.items()}),
        'processor': {
            'use_relative_action': processor.use_relative_action,
            'use_percentiles': processor.use_percentiles, 'use_mean_std': processor.use_mean_std,
            'statistics_sha256': sha(Path(checkpoint) / 'statistics.json'),
            'public_processor': 'Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203',
        },
        'native_source': {'revision': NATIVE_REVISION,
                          'files': {str(Path(name).relative_to(native)): value for name, value in native_files.items()}},
    }
    # The loader exists before RoboClue is attached. The adapter replaces only
    # its collate_fn, and returns the very same original collator result on PASS.
    loader = DataLoader(mixture, batch_size=1, num_workers=0, collate_fn=processor.collator,
                        generator=torch.Generator().manual_seed(42))
    session = NativeSession(dataset, mixture, processor.collator, context, contract,
                            max_samples=max_samples, loader=loader)
    require(session.loader is loader and session.existing_loader, 'Guard substituted the native loader')
    session.preview_versions = {name: importlib.metadata.version(name) for name in
                                ('torch', 'torchvision', 'torchcodec', 'numpy', 'transformers')}
    session.preview_schema = _batch_arrays
    return session
