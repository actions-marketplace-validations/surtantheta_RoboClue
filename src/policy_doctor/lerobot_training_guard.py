"""Check a LeRobot policy's actual input after native preprocessing, in its parent process.

DataLoader workers transport the dataset's original ``index`` column. This
adapter never installs worker-side hooks or pairs samples by arrival order.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .training_guard import TargetContract, TargetContractViolation


def _cpu(value):
    if isinstance(value, np.ndarray):
        return value
    if type(value).__module__.split('.')[0] == 'torch':
        import torch

        if isinstance(value, torch.Tensor) and value.layout == torch.strided:
            try:
                return value.detach().cpu().numpy()
            except (TypeError, RuntimeError) as error:
                raise InputError('Use dense targets representable as NumPy arrays') from error
    raise InputError('Expected a NumPy array or dense Torch tensor')


def _profile(dataset_id, padding):
    if (type(dataset_id) is not str or len(dataset_id) != 71 or not dataset_id.startswith('sha256:')
            or any(c not in '0123456789abcdef' for c in dataset_id[7:])):
        raise InputError('dataset_id must be sha256: plus a retained dataset content identity')
    if padding not in ('action_is_pad', 'all_valid'):
        raise InputError('padding must be action_is_pad or an explicitly declared all_valid objective')
    return {'framework': 'lerobot', 'dataset_id': dataset_id, 'padding': padding,
            'source_index': 'index', 'action': 'action',
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _declarations(value):
    if value is None:
        return {}
    if type(value) is not dict:
        raise InputError('Model input declarations require an explicit JSON object')
    try:
        raw = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if len(raw.encode()) > 32768:
            raise ValueError('oversized declarations')
        return json.loads(raw)
    except (TypeError, ValueError, RecursionError) as error:
        raise InputError('Use finite JSON model-input declarations up to 32 KiB') from error


def _input_fields(fields):
    if (type(fields) not in (list, tuple) or len(fields) > 16
            or any(type(name) is not str or not name or len(name) > 512 for name in fields)
            or len(set(fields)) != len(fields)):
        raise InputError('Declare at most sixteen distinct bounded model input field names')
    return tuple(fields)


def _validate_model_inputs(expected, source_keys):
    if (type(expected) is not dict or set(expected) != {'fields', 'declarations', 'references', 'scope'}
            or type(expected['fields']) is not list or type(expected['references']) is not dict
            or set(expected['references']) != set(source_keys)
            or type(expected['scope']) is not str or len(expected['scope']) > 2048):
        raise InputError('Retained criterion has unsupported model-input witnesses')
    fields = _input_fields(expected['fields'])
    _declarations(expected['declarations'])
    for record in expected['references'].values():
        if type(record) is not dict or set(record) != set(fields):
            raise InputError('Retained model-input row must bind exactly the declared fields')
        for witness in record.values():
            if (type(witness) is not dict or set(witness) != {'dtype', 'shape', 'sha256'}
                    or type(witness['dtype']) is not str or len(witness['dtype']) > 32
                    or type(witness['shape']) is not list or len(witness['shape']) > 5
                    or any(type(n) is not int or not 0 < n <= 64 * 1024**2 for n in witness['shape'])
                    or type(witness['sha256']) is not str or len(witness['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in witness['sha256'])):
                raise InputError('Retained model-input descriptor has an invalid schema')
            try:
                dtype = np.dtype(witness['dtype'])
            except TypeError as error:
                raise InputError('Retained model-input dtype is unsupported') from error
            if dtype.kind not in 'bifu' or dtype.str != witness['dtype']:
                raise InputError('Retained model-input dtype is unsupported')
            size = dtype.itemsize
            for dimension in witness['shape']:
                size *= dimension
                if size > 64 * 1024**2:
                    raise InputError('Retained model-input shape exceeds its byte bound')


def _input_witnesses(batch, keys, fields):
    fields = _input_fields(fields)
    records = {key: {} for key in keys}
    total_bytes = 0
    for field in fields:
        if field not in batch:
            raise InputError('Declared model input field is missing: ' + field)
        array = _cpu(batch[field])
        if (not 1 <= array.ndim <= 6 or len(array) != len(keys) or not all(array.shape)
                or array.dtype.kind not in 'bifu'):
            raise InputError('Declared model inputs require numeric arrays with one leading row per source sample')
        total_bytes += array.nbytes
        if total_bytes > 64 * 1024**2:
            raise InputError('Model input witness exceeds 64 MiB; select fewer fields or a smaller batch')
        if not np.isfinite(array).all():
            raise InputError('Declared model input has nonfinite values: ' + field)
        for key, row in zip(keys, array, strict=True):
            witness = {'dtype': row.dtype.str, 'shape': list(row.shape),
                       'sha256': hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()}
            if field in records[key] and records[key][field] != witness:
                raise InputError('Duplicate source rows have conflicting model inputs; no row witness was selected or overwritten')
            records[key][field] = witness
    return records


def lerobot_target_view(batch, *, dataset_id, padding='action_is_pad'):
    """Read targets and their retained global row IDs without changing the batch.

    ``all_valid`` is caller intent, never an inferred absence-of-padding claim.
    Native ``action_is_pad`` is Boolean [B,H]; True denotes padding.
    """
    _profile(dataset_id, padding)
    if not isinstance(batch, Mapping) or 'action' not in batch or 'index' not in batch:
        raise InputError('LeRobot policy input needs action and original index; preserve index through preprocessing')
    action, index = _cpu(batch['action']), _cpu(batch['index'])
    if action.ndim != 3 or action.dtype.kind != 'f' or action.size > 1_000_000:
        raise InputError('Supported LeRobot action targets are bounded floating [batch,horizon,dimension] arrays')
    if index.ndim != 1 or len(index) != len(action) or index.dtype.kind not in 'iu' or (index < 0).any():
        raise InputError('LeRobot index must contain one nonnegative integer global row ID per sample')
    if padding == 'action_is_pad':
        if 'action_is_pad' not in batch:
            raise InputError('Missing action_is_pad; supply native padding or explicitly declare padding=all_valid')
        pad = _cpu(batch['action_is_pad'])
        if pad.dtype.kind != 'b' or pad.shape != action.shape[:2]:
            raise InputError('Native action_is_pad must be Boolean [batch,horizon]')
        mask = np.broadcast_to(~pad[..., None], action.shape)
    else:
        if 'action_is_pad' in batch:
            pad = _cpu(batch['action_is_pad'])
            if pad.dtype.kind != 'b' or pad.shape != action.shape[:2] or pad.any():
                raise InputError('Declared all_valid objective contradicts native padded targets or padding schema')
        mask = np.ones(action.shape, dtype=np.bool_)
    keys = [dataset_id + ':row:' + str(int(value)) for value in index]
    return {'inputs': {'action': action, 'action_mask': mask}}, keys


def capture_lerobot_reference(batches, *, dataset_id, intent, padding='action_is_pad',
                              input_fields=(), declarations=None):
    """Freeze explicitly intended, already natively preprocessed batches.

    The caller owns loader lifetime and retains the returned pin independently
    before changing preprocessing. This helper performs no policy forward.
    """
    profile = _profile(dataset_id, padding)
    if (type(intent) is not dict or not intent
            or set(intent) & {'roboclue_batch_schema', 'roboclue_model_inputs'}):
        raise InputError('Supply nonempty intended recipe metadata without reserved RoboClue schema fields')
    records, witnesses = [], {}
    for batch in batches:
        if len(records) >= 128:
            raise InputError('Reference exceeds 128 batches; narrow the declared cohort')
        view, keys = lerobot_target_view(batch, dataset_id=dataset_id, padding=padding)
        for key, value in _input_witnesses(batch, keys, input_fields).items():
            if key in witnesses and witnesses[key] != value:
                raise InputError('One source row has conflicting declared model inputs; do not freeze stochastic augmentation as deterministic intent')
            witnesses[key] = value
        records.append((view, keys, [{'dataset_id': dataset_id, 'global_row': int(k.rsplit(':', 1)[1]),
                                     'boundary': 'actual_preprocessed_policy_input'} for k in keys]))
    return TargetContract.from_references(records, intent={**intent, 'roboclue_batch_schema': profile,
        'roboclue_model_inputs': {'fields': list(input_fields), 'declarations': _declarations(declarations),
                                  'references': witnesses,
                                  'scope': 'Exact declared input preservation on the covered source cohort; declarations are not physical-unit or camera-calibration authentication.'}})


class LeRobotTrainingGuard:
    """Gate ``policy(batch)`` before forward, after native preprocessing/device transfer.

    Attach to the actual module called by the trainer (the outer DDP module when
    applicable). Collective mode coordinates refusal across its process group.
    Every rank must attach with equal startup scope and reach the same calls;
    process failure or uneven iterators remain the trainer's responsibility.
    """

    def __init__(self, policy, *, contract, criterion_sha256, dataset_id,
                 padding='action_is_pad', batches=1, distributed='refuse', process_group=None,
                 declarations=None):
        import torch

        if not isinstance(policy, torch.nn.Module):
            raise InputError('Attach to the Torch policy module actually called by the native training loop')
        if type(batches) is not int or not 1 <= batches <= 128:
            raise InputError('batches must be an integer from 1 to 128')
        if distributed not in ('refuse', 'collective'):
            raise InputError('distributed must be refuse or collective')
        profile = _profile(dataset_id, padding)
        loaded = TargetContract.load(contract, expected_sha256=criterion_sha256)
        if loaded.document['intent'].get('roboclue_batch_schema') != profile:
            raise InputError('Retained criterion uses a different dataset, padding objective or adapter; keep its original runtime')
        expected_inputs = loaded.document['intent'].get('roboclue_model_inputs')
        _validate_model_inputs(expected_inputs, loaded.document['references'])
        actual_declarations = _declarations(declarations)
        dist = torch.distributed
        initialized = dist.is_available() and dist.is_initialized()
        try:
            environment_world = int(os.environ.get('WORLD_SIZE', '1'))
        except ValueError as error:
            raise InputError('WORLD_SIZE must be an integer') from error
        if environment_world < 1:
            raise InputError('WORLD_SIZE must be a positive integer')
        if not initialized and (environment_world > 1 or process_group is not None):
            raise InputError('Initialize the actual process group before attaching a distributed guard')
        world = dist.get_world_size(process_group) if initialized else 1
        if world > 1 and distributed != 'collective':
            raise InputError('Distributed training requires explicit distributed=collective on every rank')
        if getattr(policy, '_roboclue_policy_guard', None) is not None:
            raise InputError('This policy already has a RoboClue input guard')
        self.policy, self.contract = policy, loaded
        self.dataset_id, self.padding, self.batches = dataset_id, padding, batches
        self.declarations = actual_declarations
        self.world_size, self.process_group, self.distributed = world, process_group, distributed
        self.closed = False
        self.checked = self.blocked = self.unassessed = 0
        self.reports = []
        self._handle = policy.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        policy._roboclue_policy_guard = self

    def _before_forward(self, _module, args, kwargs):
        if self.closed:
            raise RuntimeError('The LeRobot policy guard is closed')
        if self.checked >= self.batches:
            self.unassessed += 1
            return None
        try:
            if any(key > self._handle.id for key in self.policy._forward_pre_hooks):
                raise InputError('A later forward pre-hook could alter checked inputs; attach RoboClue last')
            if len(args) == 1 and 'batch' not in kwargs:
                batch = args[0]
            elif not args and 'batch' in kwargs:
                batch = kwargs['batch']
            else:
                raise InputError('Expected native policy(batch) or policy(batch=batch) input')
            view, keys = lerobot_target_view(batch, dataset_id=self.dataset_id, padding=self.padding)
            report = self.contract.check(view, keys)
            if report['status'] == 'pass':
                report = self._check_model_inputs(batch, keys, report)
        except InputError as error:
            report = self.contract.check({}, [])
            report.update(code='unsupported_lerobot_policy_input', message=str(error))
        if self.world_size > 1:
            report = self._collective(report)
        self.reports.append(report)
        if report['status'] != 'pass':
            self.blocked += 1
            raise TargetContractViolation(report)
        self.checked += 1
        return None

    def _check_model_inputs(self, batch, keys, report):
        expected = self.contract.document['intent']['roboclue_model_inputs']
        if expected['declarations'] != self.declarations:
            return {**report, 'status': 'error', 'code': 'declared_model_input_contract_changed',
                    'message': 'Actual caller-declared units/camera map/semantics differ from retained intent; physical truth is not inferred.',
                    'first_difference': {'field': 'declarations', 'expected': expected['declarations'],
                                         'observed': _declarations(self.declarations)}}
        observed = _input_witnesses(batch, keys, expected['fields'])
        checked = 0
        for key in keys:
            for field in expected['fields']:
                checked += 1
                witness = observed[key][field]
                reference = expected['references'][key].get(field)
                if reference != witness:
                    matches = [name for name, other in expected['references'][key].items()
                               if name != field and other == witness]
                    return {**report, 'status': 'error', 'code': 'model_input_witness_changed',
                            'input_fields_checked': checked,
                            'message': 'Actual model input differs for a covered source row; restore declared preprocessing or review a separately intended objective.',
                            'first_difference': {'source_key': key, 'field': field, 'expected': reference,
                                                 'observed': witness,
                                                 'observed_matches_other_reference_fields': matches}}
        return {**report, 'input_fields_checked': checked,
                'input_preservation_scope': expected['scope']}

    def _collective(self, report):
        import torch

        dist = torch.distributed
        backend = str(dist.get_backend(self.process_group))
        device = torch.device('cuda', torch.cuda.current_device()) if backend == 'nccl' else torch.device('cpu')
        local = {'pass': 0, 'unknown': 1, 'error': 2}[report['status']]
        decision = torch.tensor(local, dtype=torch.int64, device=device)
        dist.all_reduce(decision, op=dist.ReduceOp.MAX, group=self.process_group)
        global_decision = int(decision.item())
        if global_decision and local == 0:
            return {**report, 'status': 'error' if global_decision == 2 else 'unknown',
                    'code': 'peer_rank_target_refusal',
                    'message': 'Another rank refused its startup targets; every rank is blocked before policy forward.',
                    'local_status': 'pass', 'world_size': self.world_size}
        return {**report, 'world_size': self.world_size, 'collective_refusal': bool(global_decision)}

    def summary(self):
        inputs = self.contract.document['intent']['roboclue_model_inputs']
        return {'kind': 'lerobot_policy_input_guard', 'criterion_sha256': self.contract.criterion_sha256,
                'scope': f'First {self.batches} actual policy inputs per rank; later inputs unassessed.',
                'declared_input_fields': inputs['fields'],
                'input_preservation_scope': inputs['scope'],
                'world_size': self.world_size, 'checked_forwards': self.checked,
                'blocked_forwards': self.blocked, 'unassessed_forwards': self.unassessed,
                'reports': self.reports, 'closed': self.closed}

    def close(self):
        if self.closed:
            return self.summary()
        if getattr(self.policy, '_roboclue_policy_guard', None) is not self:
            raise RuntimeError('Policy guard ownership changed; refusing to overwrite it')
        self._handle.remove()
        del self.policy._roboclue_policy_guard
        self.policy = self._handle = self.process_group = None
        self.closed = True
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f'RoboClue cleanup also failed: {type(cleanup_error).__name__}')
        return False
