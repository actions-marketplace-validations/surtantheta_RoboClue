"""Bounded native OpenPI PyTorch loader target checks with source-index pairing.

The all-true comparison witness is created by RoboClue. It is NOT a native
OpenPI action/loss mask. The optional Torch pre-hook checks actual post-cast
action inputs before forward. JAX, workers, DDP and Pi0/optimizer qualification
are not inferred from this adapter.
"""
from __future__ import annotations

import hashlib
import inspect
import operator
import sys
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .conversion_fidelity import _read_small
from .training_dataset_identity import fingerprint_training_dataset
from .training_guard import TargetContract, TargetContractViolation

REVISION = '215abfb217dbac7d5f1273282331b9b1866c0479'
MARKER = '__roboclue_openpi_source_index__'
MASK_SCOPE = 'RoboClue all-true comparison witness; NOT a native OpenPI action or loss mask'
INTENT_KEY = 'roboclue_openpi_projection'
TRAIN_SCRIPT_SHA256 = 'baa66f1642e955d5cc2ec3f54d9495035df61d71b028b98ab4ebb1e240548311'
SOURCES = {
    'openpi.training.data_loader': '4d0b055ac28527c1ab1f0ba7496057722c7c748c88564a8de2ff4fac650e97c9',
    'openpi.models.model': '0d74bc1d8f4623ac3d8e543710b8bf75f9b95e588a3ebf6127c659e001385cff',
    'openpi.transforms': 'a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4',
}


def _projection():
    return {'adapter': 'openpi-source-index-v1',
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_revision': REVISION, 'source_pins': SOURCES,
            'target': 'Separate exact native-loader and actual model-input action criteria',
            'model_action_projection': 'Pinned train_pytorch.py: actions.to(torch.float32).to(device)',
            'native_training_script_sha256': TRAIN_SCRIPT_SHA256,
            'comparison_mask_scope': MASK_SCOPE}


class OpenpiSourceDataset:
    """Envelope an existing final native TransformedDataset by its actual index.

    The native transform result is not mutated. A reserved numeric source index
    is appended in a shallow metadata envelope after Repack/Delta/Normalize and
    is removed again before the original Observation.from_dict call.
    """

    def __init__(self, dataset, *, dataset_root, max_files=4096, max_bytes=1024**3):
        if not callable(getattr(dataset, '__getitem__', None)) or not callable(getattr(dataset, '__len__', None)):
            raise InputError('Use an existing random-access transformed dataset')
        self.dataset = dataset
        self.root = Path(dataset_root)
        self.bounds = {'max_files': max_files, 'max_bytes': max_bytes}
        self.identity = fingerprint_training_dataset(self.root, **self.bounds)
        self.dataset_id = self.identity['dataset_id']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if type(index) is bool:
            raise InputError('OpenPI source index must be an original nonnegative dataset index')
        try:
            row = operator.index(index)
        except TypeError as error:
            raise InputError('OpenPI source index must be an original integer dataset index') from error
        if not 0 <= row < len(self) or row >= 2**63:
            raise InputError('OpenPI source index is outside its original dataset')
        original = self.dataset[index]
        if not isinstance(original, Mapping) or MARKER in original:
            raise InputError('OpenPI native sample is unsupported or collides with reserved source metadata')
        return {**original, MARKER: np.int64(row)}

    def key(self, row):
        return f'openpi:{self.dataset_id}:index:{row}'

    def provenance(self, row):
        return {'dataset_id': self.dataset_id, 'original_random_access_index': row,
                'association': 'Same original __getitem__ index through the native final transformed dataset',
                'native_revision': REVISION, 'comparison_mask_scope': MASK_SCOPE,
                'scope': 'Observed materialized content identity and selected targets; no historical or physical claim'}

    def verify(self):
        if fingerprint_training_dataset(self.root, **self.bounds)['dataset_id'] != self.dataset_id:
            raise InputError('Materialized OpenPI source content changed during startup observation')


def _native(loader):
    modules = {}
    for name, pin in SOURCES.items():
        module = sys.modules.get(name)
        path = inspect.getsourcefile(module) if module is not None else None
        if path is None or hashlib.sha256(_read_small(path, limit=4 * 1024**2)[0]).hexdigest() != pin:
            raise InputError(f'Load the supported original OpenPI {REVISION[:7]} modules before attaching: {name}')
        modules[name] = module
    native = modules['openpi.training.data_loader']
    script = Path(inspect.getsourcefile(native)).parents[3] / 'scripts/train_pytorch.py'
    if hashlib.sha256(_read_small(str(script), limit=4 * 1024**2)[0]).hexdigest() != TRAIN_SCRIPT_SHA256:
        raise InputError('The OpenPI training-script action cast differs from the declared float32 projection')
    if type(loader) is not native.DataLoaderImpl or type(loader._data_loader) is not native.TorchDataLoader:
        raise InputError('Use an original native OpenPI DataLoaderImpl wrapping TorchDataLoader')
    inner = loader._data_loader
    if inner._sharding is not None:
        raise InputError('OpenPI startup guard currently requires framework="pytorch"; JAX is unqualified')
    torch_loader = inner.torch_loader
    if torch_loader.num_workers != 0:
        raise InputError('OpenPI startup guard currently requires num_workers=0')
    dataset = torch_loader.dataset
    if type(dataset) is not OpenpiSourceDataset or type(dataset.dataset) is not native.TransformedDataset:
        raise InputError('Wrap the final original TransformedDataset with OpenpiSourceDataset before creating TorchDataLoader')
    import torch

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise InputError('OpenPI startup guard currently requires world_size=1; DDP is unqualified')
    if torch_loader.collate_fn is not native._collate_fn:
        raise InputError('Use the original native OpenPI collator to preserve the source-index envelope')
    return inner, dataset


def _target(actions):
    # The witness selects all actual action values; it never represents a native
    # OpenPI masking decision or changes the action tensor supplied to the model.
    value = actions
    if type(value).__module__.split('.')[0] == 'torch':
        import torch

        if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
            value = value.detach().cpu()
    shape = getattr(value, 'shape', ())
    if len(shape) != 3 or not all(shape) or np.prod(shape) > 1_000_000:
        raise InputError('OpenPI target witness requires bounded actual actions [batch,horizon,channel]')
    return {'inputs': {'action': value, 'action_mask': np.ones(tuple(shape), dtype=np.bool_)}}


def _model_reference(actions):
    if type(actions).__module__.split('.')[0] == 'torch':
        import torch

        return _target(actions.to(torch.float32))
    return _target(np.asarray(actions, dtype=np.float32))


def _boundary_keys(keys, boundary):
    return [key + ':' + boundary for key in keys]


def _strip(batch, dataset):
    if not isinstance(batch, Mapping) or MARKER not in batch or 'actions' not in batch:
        raise InputError('Native OpenPI batch lost its reserved source-index metadata')
    rows = batch[MARKER]
    if type(rows).__module__.split('.')[0] == 'torch':
        rows = rows.detach().cpu().numpy()
    if not isinstance(rows, np.ndarray) or rows.dtype != np.int64 or rows.ndim != 1:
        raise InputError('OpenPI source metadata requires a dense int64 index per original sample')
    if len(rows) != len(batch['actions']) or any(not 0 <= int(row) < len(dataset) for row in rows):
        raise InputError('OpenPI source-index count or original dataset range differs')
    indices = [int(row) for row in rows]
    clean = {key: value for key, value in batch.items() if key != MARKER}
    return clean, [dataset.key(row) for row in indices], [dataset.provenance(row) for row in indices]


class _Inner:
    def __init__(self, original, dataset, callback):
        self.original, self.dataset, self.callback = original, dataset, callback

    def __iter__(self):
        for native in self.original:
            clean, keys, provenance = _strip(native, self.dataset)
            self.callback(clean, keys, provenance)
            yield clean


class OpenpiTrainingGuard:
    """Check the first declared native loader batches before consumer delivery.

    Original native Observation.from_dict executes between the loader checks.
    The same resulting (Observation, actions) tuple is yielded unchanged. With
    model= supplied, metadata-paired source keys also gate actual post-cast
    forward inputs. Consume batches and model calls in native sequential order;
    this does not infer arbitrary asynchronous or reordered source association.
    """

    def __init__(self, loader, *, contract, criterion_sha256, batches=1, model=None):
        if type(batches) is not int or not 1 <= batches <= 128:
            raise InputError('batches must be an integer from 1 to 128')
        self.contract = TargetContract.load(contract, expected_sha256=criterion_sha256)
        if self.contract.document['intent'].get(INTENT_KEY) != _projection():
            raise InputError('The retained criterion was not declared for this OpenPI target projection and adapter')
        self.original, self.dataset = _native(loader)
        self.loader, self.batches = loader, batches
        self.closed = False
        self.iterator_active = False
        self.active_iterator = None
        self.checked_native_batches = self.checked_deliveries = self.unassessed_deliveries = 0
        self.checked_model_calls = self.unassessed_model_calls = 0
        self.blocked_batches = 0
        self.reports = []
        self.pending = deque()
        self.model_pending = deque()
        self.wrapper = _Inner(self.original, self.dataset, self._check_native)
        self.model, self.model_handle = None, None
        self.model_attached = model is not None
        if model is not None:
            import torch

            if not isinstance(model, torch.nn.Module):
                raise InputError('OpenPI model guard requires an actual PyTorch module')
            if getattr(model, '_roboclue_openpi_guard', None) is not None:
                raise InputError('This model already has an OpenPI startup guard attached')
            self.model = model
            self.model_handle = model.register_forward_pre_hook(self._before_model, with_kwargs=True)
            model._roboclue_openpi_guard = self
        loader._data_loader = self.wrapper

    def _check_native(self, batch, keys, _provenance):
        if self.checked_native_batches >= self.batches:
            return
        self.dataset.verify()
        report = self.contract.check(_target(batch['actions']), _boundary_keys(keys, 'native-loader'))
        self.reports.append({'boundary': 'native_transformed_batch', 'report': report})
        if report['status'] != 'pass':
            self.blocked_batches += 1
            raise TargetContractViolation(report)
        self.checked_native_batches += 1
        self.pending.append(keys)

    def __iter__(self):
        if self.closed or self.iterator_active:
            raise InputError('Use one active OpenPI guard iterator inside its open context')
        self.iterator_active = True
        self.active_iterator = self._iterate()
        return self.active_iterator

    def _iterate(self):
        iterator = None
        try:
            iterator = iter(self.loader)
            for original in iterator:
                if self.checked_deliveries < self.batches:
                    if not self.pending:
                        raise InputError('Native OpenPI consumer batch has no paired source-index witness')
                    keys = self.pending.popleft()
                    report = self.contract.check(_target(original[1]), _boundary_keys(keys, 'native-loader'))
                    self.reports.append({'boundary': 'native_observation_and_actions_delivery', 'report': report})
                    if report['status'] != 'pass':
                        self.blocked_batches += 1
                        raise TargetContractViolation(report)
                    self.checked_deliveries += 1
                    if self.model_attached:
                        self.model_pending.append(keys)
                else:
                    self.unassessed_deliveries += 1
                yield original
        finally:
            try:
                if callable(getattr(iterator, 'close', None)):
                    iterator.close()
            finally:
                self.iterator_active = False
                self.active_iterator = None

    def data_config(self):
        if self.closed:
            raise InputError('The OpenPI startup guard is closed')
        return self.loader.data_config()

    def _before_model(self, model, args, kwargs):
        if self.checked_model_calls >= self.batches:
            self.unassessed_model_calls += 1
            return
        if next(reversed(model._forward_pre_hooks), None) != self.model_handle.id:
            report = self.contract.check({}, [])
            report.update(code='unsupported_later_model_pre_hook',
                          message='Attach other model pre-hooks before the startup guard so it checks the final input.')
        elif not self.model_pending:
            report = self.contract.check({}, [])
            report.update(code='unpaired_openpi_model_input',
                          message='No metadata-paired native delivery is available for this model input; no source identity was guessed.')
        else:
            keys = self.model_pending.popleft()
            actions = kwargs.get('actions') if 'actions' in kwargs else args[1] if len(args) > 1 else None
            try:
                report = self.contract.check(_target(actions), _boundary_keys(keys, 'model-float32'))
            except InputError as error:
                report = self.contract.check({}, [])
                report.update(code='unsupported_openpi_model_target', message=str(error))
        self.reports.append({'boundary': 'actual_model_forward_input', 'report': report})
        if report['status'] != 'pass':
            self.blocked_batches += 1
            raise TargetContractViolation(report)
        self.checked_model_calls += 1

    def summary(self):
        return {'kind': 'openpi_training_startup_guard', 'criterion_sha256': self.contract.criterion_sha256,
                'checked_native_batches': self.checked_native_batches, 'checked_deliveries': self.checked_deliveries,
                'unassessed_deliveries': self.unassessed_deliveries, 'blocked_batches': self.blocked_batches,
                'model_hook_attached': self.model_attached, 'checked_model_calls': self.checked_model_calls,
                'unassessed_model_calls': self.unassessed_model_calls,
                'reports': self.reports, 'closed': self.closed, 'comparison_mask_scope': MASK_SCOPE,
                'scope': f'First {self.batches} native PyTorch batches and optionally paired model action inputs; sequential consumption required; later batches unassessed; no Pi0/optimizer claim'}

    def close(self):
        if self.closed:
            return self.summary()
        if self.loader._data_loader is not self.wrapper:
            raise RuntimeError('OpenPI native loader hook changed; refusing to overwrite it')
        dataset = self.dataset
        failure = None
        try:
            if self.active_iterator is not None:
                self.active_iterator.close()
        except BaseException as error:
            failure = error
        finally:
            self.iterator_active = False
            self.active_iterator = None
            self.loader._data_loader = self.original
            if self.model_handle is not None:
                self.model_handle.remove()
                if getattr(self.model, '_roboclue_openpi_guard', None) is self:
                    del self.model._roboclue_openpi_guard
            self.model = self.model_handle = None
            self.model_pending.clear()
            self.closed = True
            self.loader = self.original = self.wrapper = self.dataset = None
        try:
            dataset.verify()
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            try:
                BaseException.add_note(error, f'OpenPI guard cleanup also failed: {type(cleanup_error).__name__}')
            except BaseException:
                pass
        return False


def capture_openpi_reference(loader, *, output, intent, batches=1):
    """Freeze intended targets in a separate reference process, without a model.

    The original native transform, collator and Observation.from_dict all run.
    Publication follows iterator cleanup, original-loader restoration and source
    content verification. The criterion binds this exact adapter/projection.
    """
    if (type(batches) is not int or not 1 <= batches <= 128 or type(intent) is not dict
            or not intent or INTENT_KEY in intent):
        raise InputError('Declare nonempty reference intent, batches from 1 to 128 and no reserved projection key')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise InputError('OpenPI reference output must name a new file')
    original, dataset = _native(loader)
    pending, references = deque(), []

    def capture(_batch, keys, provenance):
        pending.append((keys, provenance))

    wrapper = _Inner(original, dataset, capture)
    loader._data_loader = wrapper
    iterator = None
    try:
        iterator = iter(loader)
        for _ in range(batches):
            actual = next(iterator)
            keys, provenance = pending.popleft()
            references.append((_target(actual[1]), _boundary_keys(keys, 'native-loader'),
                               [{**item, 'boundary': 'native-loader'} for item in provenance]))
            references.append((_model_reference(actual[1]), _boundary_keys(keys, 'model-float32'),
                               [{**item, 'boundary': 'model-float32', 'native_action_cast_source_sha256': TRAIN_SCRIPT_SHA256}
                                for item in provenance]))
        contract = TargetContract.from_references(references, intent={**intent, INTENT_KEY: _projection()})
    finally:
        try:
            if callable(getattr(iterator, 'close', None)):
                iterator.close()
        finally:
            if loader._data_loader is not wrapper:
                raise RuntimeError('OpenPI native loader hook changed during reference capture')
            loader._data_loader = original
            dataset.verify()
    contract.save(output)
    return contract
