"""Preserve a declared target recipe on the first batches of a native trainer.

The collator's original result is gated before delivery. The trainer's actual
post-device input is checked again before compute_loss. Later batches remain
explicitly unassessed; this is a startup gate, not a population certificate.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from types import MethodType

from .artifacts import InputError
from .training_guard import GuardedCollator, TargetContract, TargetContractViolation


def _cpu_target_view(batch):
    """Copy only target witnesses for checking; never modify trainer inputs."""
    if not isinstance(batch, Mapping) or not isinstance(batch.get('inputs'), Mapping):
        return batch
    selected = {}
    for key in ('action', 'action_mask'):
        value = batch['inputs'].get(key)
        if type(value).__module__.split('.')[0] == 'torch':
            import torch

            if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
                value = value.detach().cpu()
        selected[key] = value
    return {'inputs': selected}


class TrainerStartupGuard:
    """Attach to a compatible trainer before it builds or consumes its loader.

    Supports one process and zero DataLoader workers. Source keys must identify
    the original samples before native collation. The native collator and loss
    method are restored by close(); dataset/executor lifetime stays caller-owned.
    """

    def __init__(self, trainer, contract, *, source_keys, batches=1, on_scope_end=None):
        if not isinstance(contract, TargetContract) or not callable(source_keys):
            raise InputError('Supply a loaded TargetContract and a source_keys callback')
        if type(batches) is not int or not 1 <= batches <= 128:
            raise InputError('batches must be an integer from 1 to 128')
        args = getattr(trainer, 'args', None)
        if getattr(args, 'dataloader_num_workers', None) != 0:
            raise InputError('Startup guard currently requires dataloader_num_workers=0')
        if getattr(args, 'remove_unused_columns', None) is not False:
            raise InputError('Startup guard requires remove_unused_columns=False to preserve source pairing')
        if getattr(args, 'world_size', None) != 1:
            raise InputError('Startup guard currently requires world_size=1')
        if getattr(args, 'eval_on_start', False):
            raise InputError('Startup guard requires eval_on_start=False')
        if getattr(trainer, '_roboclue_startup_guard', None) is not None:
            raise InputError('This trainer already has a RoboClue startup guard attached')
        collator, loss = getattr(trainer, 'data_collator', None), getattr(trainer, 'compute_loss', None)
        if not callable(collator) or not callable(loss) or (on_scope_end is not None and not callable(on_scope_end)):
            raise InputError('Trainer needs callable data_collator and compute_loss methods')
        if not hasattr(trainer, '__dict__'):
            raise InputError('Startup guard requires a trainer with ordinary per-instance hook attributes')
        self.trainer, self.contract, self.batches = trainer, contract, batches
        self.original_collator, self.original_loss = collator, loss
        self._original_instance_hooks = {name: trainer.__dict__[name] for name in ('data_collator', 'compute_loss')
                                         if name in trainer.__dict__}
        self.source_keys, self.on_scope_end = source_keys, on_scope_end
        self._pending = deque()
        self._last_keys = None
        self._scope_ended = False
        self.closed = False
        self.native_collations = self.checked_collations = self.checked_loss_inputs = 0
        self.unassessed_collations = self.unassessed_loss_inputs = self.blocked_batches = 0
        self.collation_reports, self.loss_input_reports = [], []
        self.last_native_batch = None
        self._guard = GuardedCollator(self._native_collator, contract, source_keys=self._capture_keys)
        self._collator_wrapper = self._collate
        owner = self

        def checked_loss(_trainer, model, inputs, *positional, **kwargs):
            return owner._compute_loss(model, inputs, *positional, **kwargs)

        self._loss_wrapper = MethodType(checked_loss, trainer)
        trainer.data_collator = self._collator_wrapper
        trainer.compute_loss = self._loss_wrapper
        trainer._roboclue_startup_guard = self

    def _capture_keys(self, features):
        keys = self.source_keys(features)
        self._last_keys = tuple(keys) if isinstance(keys, (list, tuple)) else keys
        return self._last_keys

    def _native_collator(self, features):
        batch = self.original_collator(features)
        self.native_collations += 1
        self.last_native_batch = batch
        return batch

    def _finish_scope(self):
        if not self._scope_ended:
            self._scope_ended = True
            if self.on_scope_end is not None:
                self.on_scope_end()

    def _collate(self, features):
        if self.closed:
            raise RuntimeError('The startup guard is closed')
        if self.checked_collations >= self.batches:
            self.unassessed_collations += 1
            return self._native_collator(features)
        try:
            batch = self._guard(features)
        except TargetContractViolation:
            self.blocked_batches += 1
            self.collation_reports.append(self._guard.last_report)
            self._finish_scope()
            raise
        self.checked_collations += 1
        self.collation_reports.append(self._guard.last_report)
        self._pending.append(self._last_keys)
        if self.checked_collations == self.batches:
            self._finish_scope()
        return batch

    def _compute_loss(self, model, inputs, *args, **kwargs):
        if self.closed:
            raise RuntimeError('The startup guard is closed')
        if self.checked_loss_inputs >= self.batches:
            self.unassessed_loss_inputs += 1
            return self.original_loss(model, inputs, *args, **kwargs)
        if not self._pending:
            report = self.contract.check({}, [])
            report.update(code='unpaired_trainer_batch',
                          message='No checked native collation is paired with this trainer input; training is blocked.')
        else:
            keys = self._pending.popleft()
            report = self.contract.check(_cpu_target_view(inputs), keys)
        self.loss_input_reports.append(report)
        if report['status'] != 'pass':
            self.blocked_batches += 1
            raise TargetContractViolation(report)
        self.checked_loss_inputs += 1
        return self.original_loss(model, inputs, *args, **kwargs)

    def summary(self):
        return {'kind': 'native_trainer_startup_guard', 'criterion_sha256': self.contract.criterion_sha256,
                'scope': f'First {self.batches} collated training batches and their actual compute_loss inputs; later batches unassessed.',
                'native_collations': self.native_collations, 'checked_collations': self.checked_collations,
                'checked_loss_inputs': self.checked_loss_inputs, 'blocked_batches': self.blocked_batches,
                'unassessed_collations': self.unassessed_collations,
                'unassessed_loss_inputs': self.unassessed_loss_inputs,
                'pending_checked_batches': len(self._pending), 'collation_reports': self.collation_reports,
                'loss_input_reports': self.loss_input_reports, 'closed': self.closed}

    def close(self):
        if self.closed:
            return self.summary()
        if (self.trainer.data_collator is not self._collator_wrapper
                or self.trainer.compute_loss is not self._loss_wrapper):
            raise RuntimeError('Trainer hooks changed while the startup guard was attached; refusing to overwrite them')
        try:
            self._finish_scope()
        finally:
            for name in ('data_collator', 'compute_loss'):
                if name in self._original_instance_hooks:
                    setattr(self.trainer, name, self._original_instance_hooks[name])
                else:
                    delattr(self.trainer, name)
            del self.trainer._roboclue_startup_guard
            self.closed = True
            # A retained evidence handle must not retain a trainer, model or its
            # GPU allocations after detaching the hooks.
            self.trainer = self.original_collator = self.original_loss = None
            self.source_keys = self.on_scope_end = self.last_native_batch = None
            self._guard = self._collator_wrapper = self._loss_wrapper = None
            self._original_instance_hooks.clear()
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f'RoboClue guard cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}')
        return False
