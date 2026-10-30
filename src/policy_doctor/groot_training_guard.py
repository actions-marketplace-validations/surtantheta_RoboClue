"""Opt-in source association for an existing pinned GR00T native trainer.

No factory, server path, or model configuration is injected into the trainer.
Native shard order associates its original returned sample objects with source
rows before the mixture shuffles them. Only the declared startup cohort is gated.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import sys
import threading
from itertools import islice
from pathlib import Path
from types import MethodType

from .artifacts import InputError
from .trainer_guard import TrainerStartupGuard
from .training_guard import TargetContract, TargetContractViolation

REVISION = '944fe70c4fd54e3fcb80a7d7258a328f9c061176'
NATIVE = {
    'gr00t.experiment.trainer': ('Gr00tTrainer', '10701c09cb0aacae75f8d9a764d0e83bdddef5b06c8126152cad8b6d7b4a1616'),
    'gr00t.data.dataset.sharded_mixture_dataset': ('ShardedMixtureDataset', 'f452072c1191249adef45d98bf75239c792e019f4a088bc8dc09f10f7a7441a3'),
    'gr00t.data.dataset.sharded_single_step_dataset': ('ShardedSingleStepDataset', '8d4cfeb4beee92a0d4b4a042b953ebbd1fb5e1e862a38c77f366ce5e37effb80'),
    'gr00t.model.gr00t_n1d7.processing_gr00t_n1d7': ('Gr00tN1d7Processor', '77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f'),
}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    return value


def _digest(value):
    try:
        raw = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise InputError('GR00T source declarations must be bounded finite JSON') from error
    if len(raw) > 4 * 1024**2:
        raise InputError('GR00T source declarations exceed the 4 MiB bound')
    return hashlib.sha256(raw).hexdigest()


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class _Files:
    def __init__(self):
        self.entries = {}
        self.lock = threading.RLock()
        self.bytes_hashed = 0

    def fingerprint(self, path, *, max_bytes=8 * 1024**3):
        path = Path(path)
        with self.lock:
            try:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                    raise InputError(f'Expected a regular GR00T input within the byte bound: {path}')
                previous = self.entries.get(path)
                if previous is not None:
                    if previous[0] != _signature(before):
                        raise InputError(f'GR00T source input changed during observation: {path}')
                    return previous[1]
                flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
                with os.fdopen(os.open(path, flags), 'rb') as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode) or _signature(opened) != _signature(before):
                        raise InputError(f'GR00T input changed before opening: {path}')
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if _signature(os.fstat(stream.fileno())) != _signature(opened):
                        raise InputError(f'GR00T input changed while hashing: {path}')
                if _signature(path.lstat()) != _signature(before):
                    raise InputError(f'GR00T input changed after hashing: {path}')
            except OSError as error:
                raise InputError(f'Cannot read GR00T input {path}: {error.strerror}') from error
            self.entries[path] = (_signature(before), digest)
            self.bytes_hashed += before.st_size
            return digest

    def verify(self):
        with self.lock:
            for path, (signature, _digest_value) in self.entries.items():
                try:
                    same = _signature(path.lstat()) == signature
                except OSError:
                    same = False
                if not same:
                    raise InputError(f'GR00T input changed during source observation: {path}')


def _native(trainer):
    classes = {}
    files = _Files()
    for module_name, (class_name, expected) in NATIVE.items():
        module = sys.modules.get(module_name)
        cls = getattr(module, class_name, None)
        path = inspect.getsourcefile(cls) if inspect.isclass(cls) else None
        if path is None or files.fingerprint(path, max_bytes=4 * 1024**2) != expected:
            raise InputError(f'Load the supported GR00T {REVISION[:7]} source before attaching: {module_name}')
        classes[class_name] = cls
    mixture = getattr(trainer, 'train_dataset', None)
    if type(trainer) is not classes['Gr00tTrainer'] or type(mixture) is not classes['ShardedMixtureDataset']:
        raise InputError('Attach to the supported native Gr00tTrainer and ShardedMixtureDataset before training')
    datasets = getattr(mixture, 'datasets', None)
    if not isinstance(datasets, list) or not 1 <= len(datasets) <= 8 or len({id(d) for d in datasets}) != len(datasets):
        raise InputError('GR00T startup observer supports one to eight distinct native datasets')
    if type(mixture.processor) is not classes['Gr00tN1d7Processor']:
        raise InputError('GR00T startup observer requires the supported native N1.7 processor')
    if any(getattr(mixture, name, None) is not None for name in ('_executor', '_cache_job', 'curr_shard')):
        raise InputError('Attach the GR00T observer before creating or consuming the training iterator')
    for dataset in datasets:
        cls = classes['ShardedSingleStepDataset']
        if type(dataset) is not cls or dataset.allow_padding or dataset.processor is not mixture.processor:
            raise InputError('GR00T observer requires native ShardedSingleStepDataset with allow_padding=False')
        for name in ('get_shard', 'get_datapoint'):
            method = getattr(dataset, name)
            if not inspect.ismethod(method) or method.__self__ is not dataset or method.__func__ is not getattr(cls, name):
                raise InputError(f'An existing replacement of native {name} is unsupported')
    return mixture, files


def _processor(processor):
    flags = {name: _plain(getattr(processor, name)) for name in
             ('use_relative_action', 'use_percentiles', 'use_mean_std', 'clip_outliers',
              'max_action_dim', 'max_action_horizon')}
    flags['statistics_sha256'] = _digest(processor.state_action_processor.statistics)
    return flags


class Gr00tSourceObserver:
    """Discover native source identities while preserving native sample objects."""

    def __init__(self, trainer, *, max_samples=16384):
        if type(max_samples) is not int or not 1 <= max_samples <= 65536:
            raise InputError('max_samples must be an integer from 1 to 65536')
        self.mixture, self.files = _native(trainer)
        self.processor = _processor(self.mixture.processor)
        self.max_samples = max_samples
        self.lock = threading.RLock()
        self.samples, self.hooks = {}, []
        self.enabled, self.closed = True, False
        self.observed_shards = 0
        self.last_provenance = []
        try:
            for dataset in self.mixture.datasets:
                original = dataset.get_shard
                previous = dataset.__dict__.get('get_shard')
                wrapper = MethodType(self._wrap(dataset, original), dataset)
                self.hooks.append((dataset, original, previous, wrapper))
                dataset.get_shard = wrapper
        except BaseException:
            self.close()
            raise

    def _wrap(self, dataset, original):
        def shard(_instance, index):
            with self.lock:
                enabled = self.enabled
            if not enabled:
                return original(index)
            plan = list(islice(((int(pos), int(step)) for pos, steps in dataset.sharded_episodes[index]
                                for step in steps), self.max_samples + 1))
            if not 1 <= len(plan) <= self.max_samples:
                raise InputError('Native shard exceeds max_samples; increase the explicit observation bound or reduce shard_size')
            descriptions = []
            for position, step in plan:
                with self.lock:
                    still_enabled = self.enabled
                if not still_enabled:
                    return original(index)
                descriptions.append(self._describe(dataset, position, step))
            result = original(index)
            if type(result) is not list or len(result) != len(plan):
                raise InputError('Native shard returned an unsupported sample sequence')
            self.files.verify()
            with self.lock:
                if not self.enabled:
                    return result
                if len(self.samples) + len(result) > self.max_samples:
                    raise InputError('Native startup prefetch exceeds max_samples; use a larger explicit observation bound')
                for item, provenance in zip(result, descriptions, strict=True):
                    if id(item) in self.samples:
                        raise InputError('Native shard reused an ambiguous sample object')
                    self.samples[id(item)] = (item, 'sha256:' + _digest(provenance['source']), provenance)
                self.observed_shards += 1
            return result
        return shard

    def _describe(self, dataset, position, step):
        import pyarrow.parquet as pq

        loader = dataset.episode_loader
        root = Path(loader.dataset_path).resolve(strict=True)
        meta = loader.episodes_metadata[position]
        stored, length = int(meta['episode_index']), int(meta['length'])

        def relative_file(relative):
            path = root / relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.resolve().is_relative_to(root):
                raise InputError('GR00T source path escapes its declared dataset root')
            return path

        relative = loader.data_path_pattern.format(episode_chunk=stored // loader.chunk_size, episode_index=stored)
        parquet = relative_file(relative)
        parquet_hash = self.files.fingerprint(parquet)
        if pq.ParquetFile(parquet).metadata.num_rows != length:
            raise InputError('Native episode metadata length differs from the stored Parquet rows')
        fields, rows = [], {}
        for role in ('state', 'action'):
            config = dataset.modality_configs[role]
            rows[role] = [step + int(delta) for delta in config.delta_indices]
            if not rows[role] or not all(0 <= row < length for row in rows[role]):
                raise InputError('GR00T source row selection requires unavailable padding or episode rows')
            offset = 0
            for index, key in enumerate(config.modality_keys):
                declaration = loader.modality_meta[role][key]
                begin, end = int(declaration['start']), int(declaration['end'])
                if not 0 <= begin < end <= 4096:
                    raise InputError('GR00T source slices must be explicit nonnegative bounded intervals')
                field = {'role': role, 'key': key, 'column': declaration.get('original_key') or
                         {'state': 'observation.state', 'action': 'action'}[role],
                         'source_slice': [begin, end], 'packed_slice': [offset, offset + end - begin]}
                if role == 'action':
                    action = config.action_configs[index]
                    field.update(declared_representation=action.rep.value, reference_state_key=action.state_key or key)
                fields.append(field)
                offset += end - begin
        video = []
        for key in dataset.modality_configs['video'].modality_keys:
            meta_key = loader._video_key_mapping.get(key, key)
            original_key = loader.modality_meta['video'][meta_key].get('original_key', 'observation.images.' + meta_key)
            name = loader.video_path_pattern.format(episode_chunk=stored // loader.chunk_size,
                                                    episode_index=stored, video_key=original_key)
            physical_rows = [step + int(delta) for delta in dataset.modality_configs['video'].delta_indices]
            if not physical_rows or not all(0 <= row < length for row in physical_rows):
                raise InputError('GR00T video source rows require unavailable padding or episode rows')
            video.append({'key': key, 'relative_path': name,
                          'sha256': self.files.fingerprint(relative_file(name)),
                          'physical_rows': physical_rows})
        source = {'adapter_schema': 'gr00t-native-shard-v1', 'embodiment': dataset.embodiment_tag.value,
                  'metadata_position': position, 'stored_episode_index': stored, 'physical_step': step,
                  'episode_declaration_sha256': _digest(meta), 'dataset_info_sha256': _digest(loader.info_meta),
                  'rows': rows, 'columns': fields, 'parquet_path': relative, 'parquet_sha256': parquet_hash,
                  'video': video, 'allow_padding': False}
        return {'source': source, 'processor': self.processor,
                'native_source': {'revision': REVISION, 'verified_classes': list(NATIVE)},
                'scope': 'Original native shard/sample association; no historical, camera-meaning or physical authentication'}

    def records(self, features):
        with self.lock:
            result = []
            for original in features:
                record = self.samples.get(id(original))
                if record is None or record[0] is not original:
                    raise InputError('Unobserved native source sample; attach before creating the training iterator')
                result.append(record)
            self.last_provenance = [row[2] for row in result]
            return result

    def source_keys(self, features):
        return [record[1] for record in self.records(features)]

    def stop_capture(self):
        with self.lock:
            self.enabled = False
            self.samples.clear()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.stop_capture()
            if any(dataset.get_shard is not wrapper for dataset, _original, previous, wrapper in self.hooks):
                raise RuntimeError('Native GR00T get_shard hook changed; refusing to overwrite it')
            for dataset, _original, previous, _wrapper in self.hooks:
                if previous is None:
                    del dataset.get_shard
                else:
                    dataset.get_shard = previous
            self.hooks.clear()
            self.mixture = None
            self.closed = True
        self.files.verify()


class Gr00tTrainingGuard:
    """Few-line opt-in attachment to an existing native GR00T trainer."""

    def __init__(self, trainer, *, contract, criterion_sha256, batches=1, max_samples=16384):
        self.contract = TargetContract.load(contract, expected_sha256=criterion_sha256)
        self.observer = Gr00tSourceObserver(trainer, max_samples=max_samples)
        try:
            self.guard = TrainerStartupGuard(trainer, self.contract, source_keys=self.observer.source_keys,
                                             batches=batches, on_scope_end=self.observer.stop_capture)
        except BaseException:
            self.observer.close()
            raise

    def summary(self):
        changes = []
        expected_recipes = {json.dumps(record['provenance'].get('processor'), sort_keys=True)
                            for record in self.contract.document['references'].values()}
        if len(expected_recipes) == 1:
            expected = json.loads(next(iter(expected_recipes)))
            if isinstance(expected, dict):
                changes = [{'option': key, 'expected': value, 'observed': self.observer.processor.get(key)}
                           for key, value in expected.items() if self.observer.processor.get(key) != value]
        return {**self.guard.summary(), 'adapter': 'gr00t-native-shard-v1', 'native_revision': REVISION,
                'configuration_changes': changes, 'source_shards_observed': self.observer.observed_shards,
                'source_bytes_hashed': self.observer.files.bytes_hashed,
                'configuration_change_scope': 'Recipe differences are candidate explanations; a change alone is not a target error.'}

    def close(self):
        try:
            self.guard.close()
        finally:
            self.observer.close()
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        if isinstance(error, TargetContractViolation):
            error.report['configuration_changes'] = self.summary()['configuration_changes']
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f'GR00T observer cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}')
        return False


def capture_gr00t_reference(trainer, *, output, intent, batches=1, max_samples=16384):
    """Freeze intended native startup targets without a forward/optimizer call.

    Run this explicitly on the intended recipe in a separate reference process;
    retain the returned pin independently before changing the pipeline. The
    helper owns the iterator it creates and closes its native prefetch executor.
    It does not infer that the selected reference is physically correct.
    """
    if type(batches) is not int or not 1 <= batches <= 128 or not isinstance(intent, dict) or not intent:
        raise InputError('Declare nonempty reference intent and batches from 1 to 128')
    options = {'dataloader_num_workers': 0, 'remove_unused_columns': False, 'world_size': 1, 'eval_on_start': False}
    for name, expected in options.items():
        if getattr(trainer.args, name, None) != expected:
            raise InputError(f'GR00T reference capture requires {name}={expected}')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise InputError('Reference output must name a new file')
    observer = Gr00tSourceObserver(trainer, max_samples=max_samples)
    original_collator = trainer.data_collator
    references = []
    iterator = None

    def capture(features):
        records = observer.records(features)
        native = original_collator(features)
        references.append((native, [record[1] for record in records], [record[2] for record in records]))
        if len(references) == batches:
            observer.stop_capture()
        return native

    trainer.data_collator = capture
    try:
        iterator = iter(trainer.get_train_dataloader())
        for _ in range(batches):
            next(iterator)
        if len(references) != batches:
            raise InputError('Native reference iterator did not return the declared batch count')
        contract = TargetContract.from_references(references, intent=intent)
    finally:
        trainer.data_collator = original_collator
        observer.stop_capture()
        try:
            generator = getattr(getattr(iterator, '_dataset_fetcher', None), 'dataset_iter', None)
            if generator is not None:
                generator.close()
            future = getattr(observer.mixture, '_cache_job', None)
            executor = getattr(observer.mixture, '_executor', None)
            try:
                if future is not None and not future.cancel():
                    future.result(timeout=60)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
        finally:
            observer.close()
    contract.save(output)
    return contract
