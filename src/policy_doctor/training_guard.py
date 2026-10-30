"""Preserve declared reference targets at the actual native collator boundary.

The bounded reference bank is explicit intent for identified samples. It does
not infer physical semantics or certify a model, dataset population or robot.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .conversion_fidelity import _read_small

MAX_VALUES = 1_000_000
MAX_RECORDS = 128
MAX_CONTRACT_BYTES = 4 * 1024**2
SCOPE = 'Declared native action-target and mask preservation for covered source samples; no model or physical claim.'


class _InvalidTargetBatch(InputError):
    pass


def _encode(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError) as error:
        raise InputError('Target contract metadata must be finite JSON') from error


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _verifier():
    return {'module_sha256': _digest(Path(__file__).read_bytes()), 'numpy': np.__version__}


def _array(value):
    if isinstance(value, np.ndarray):
        return value
    if type(value).__module__.split('.')[0] == 'torch':
        import torch

        if isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided:
            try:
                return value.detach().numpy()
            except (TypeError, RuntimeError) as error:
                raise InputError('Unsupported CPU target tensor representation') from error
    raise InputError('Use numeric NumPy arrays or explicitly supplied dense CPU Torch tensors')


def _batch_arrays(batch):
    if not isinstance(batch, Mapping) or not isinstance(batch.get('inputs'), Mapping):
        raise InputError('Expected the actual native batch inputs mapping')
    inputs = batch['inputs']
    if not {'action', 'action_mask'} <= inputs.keys():
        raise InputError('Native action target and action_mask are required')
    target, mask = _array(inputs['action']), _array(inputs['action_mask'])
    if (target.ndim != 3 or not all(target.shape)
            or target.dtype.kind != 'f' or mask.shape != target.shape or mask.dtype.kind not in 'bifu'):
        raise _InvalidTargetBatch('Expected floating targets and equally shaped numeric masks [batch, horizon, dimension]')
    if target.size > MAX_VALUES:
        raise InputError('Native batch exceeds the supported target-value budget; use a smaller batch')
    if not np.isfinite(target).all() or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise _InvalidTargetBatch('Targets must be finite and masks must contain only zero or one')
    return target, mask


def _keys(values, count):
    if (not isinstance(values, (list, tuple)) or len(values) != count
            or any(type(key) is not str or not key or len(key.encode('utf-8')) > 512 for key in values)):
        raise InputError('Supply one bounded, unambiguous source key for every native batch sample')
    return list(values)


def _descriptor(array):
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'data': array.tolist(),
            'sha256': _digest(np.ascontiguousarray(array).tobytes())}


def _restore(value):
    if type(value) is not dict or set(value) != {'dtype', 'shape', 'data', 'sha256'}:
        raise InputError('Malformed target reference array')
    shape = value['shape']
    if (type(shape) is not list or len(shape) != 2 or any(type(n) is not int or not 0 < n <= 4096 for n in shape)
            or shape[0] * shape[1] > MAX_VALUES):
        raise InputError('Target reference array exceeds its declared bounds')
    try:
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'bifu':
            raise ValueError('nonnumeric reference')
        array = np.asarray(value['data'], dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError('Invalid numeric target reference') from error
    if list(array.shape) != shape or _descriptor(array)['sha256'] != value['sha256']:
        raise InputError('Target reference shape or content identity differs')
    return array


@dataclass(frozen=True)
class TargetContract:
    """An immutable, bounded bank of declared native reference targets."""

    _canonical: bytes

    def __post_init__(self):
        if type(self._canonical) is not bytes or len(self._canonical) > MAX_CONTRACT_BYTES:
            raise InputError('Target contract exceeds its byte bound')
        try:
            value = json.loads(self._canonical)
        except (ValueError, RecursionError, UnicodeError) as error:
            raise InputError('Malformed target contract JSON') from error
        if (type(value) is not dict or set(value) != {'schema_version', 'kind', 'intent', 'verifier', 'references'}
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['kind'] != 'native_target_preservation' or value['verifier'] != _verifier()
                or type(value['intent']) is not dict or not value['intent']
                or type(value['references']) is not dict or not 1 <= len(value['references']) <= MAX_RECORDS
                or _encode(value) != self._canonical):
            raise InputError('Unsupported, changed or noncanonical target criterion')
        for key, record in value['references'].items():
            _keys([key], 1)
            if type(record) is not dict or set(record) != {'action', 'action_mask', 'provenance'}:
                raise InputError('Malformed native target reference record')
            if type(record['provenance']) is not dict or not record['provenance']:
                raise InputError('Reference source provenance is required')
            target, mask = _restore(record['action']), _restore(record['action_mask'])
            _batch_arrays({'inputs': {'action': target[None], 'action_mask': mask[None]}})

    @property
    def criterion_sha256(self):
        return _digest(self._canonical)

    @property
    def document(self):
        return json.loads(self._canonical)

    @classmethod
    def from_references(cls, references, *, intent):
        records = {}
        for batch, source_keys, provenance in references:
            targets, masks = _batch_arrays(batch)
            keys = _keys(source_keys, len(targets))
            if not isinstance(provenance, (list, tuple)) or len(provenance) != len(keys):
                raise InputError('Provide source provenance for each declared reference sample')
            for index, key in enumerate(keys):
                record = {'action': _descriptor(targets[index]), 'action_mask': _descriptor(masks[index]),
                          'provenance': provenance[index]}
                if key in records and _encode(records[key]) != _encode(record):
                    raise InputError('One source key has conflicting declared reference targets')
                records[key] = record
                if len(records) > MAX_RECORDS:
                    raise InputError('Too many reference samples; narrow the declared pilot')
        return cls(_encode({'schema_version': 1, 'kind': 'native_target_preservation', 'intent': intent,
                            'verifier': _verifier(), 'references': records}))

    def save(self, path):
        try:
            with Path(path).open('xb') as stream:
                stream.write(self._canonical)
        except OSError as error:
            raise InputError(f'Target reference requires a new output file: {str(path)!r}') from error

    @classmethod
    def load(cls, path, *, expected_sha256):
        raw, _ = _read_small(str(path), limit=MAX_CONTRACT_BYTES)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or _digest(raw) != expected_sha256:
            raise InputError('Target criterion differs from the independently retained pin')
        return cls(raw)

    def check(self, batch, source_keys):
        report = {'kind': 'native_target_guard', 'status': 'unknown',
                  'criterion_sha256': self.criterion_sha256, 'scope': SCOPE,
                  'samples_checked': 0, 'target_values_checked': 0, 'mask_values_checked': 0,
                  'differing_target_values': 0, 'differing_mask_values': 0, 'first_difference': None,
                  'findings': [], 'finding_count': 0, 'findings_truncated': False}

        def finding(details):
            report['finding_count'] += 1
            if len(report['findings']) < 8:
                report['findings'].append(details)
            else:
                report['findings_truncated'] = True

        try:
            targets, masks = _batch_arrays(batch)
            keys = _keys(source_keys, len(targets))
        except _InvalidTargetBatch as error:
            report.update(status='error', code='invalid_target_batch', message=str(error))
            return report
        except InputError as error:
            report.update(code='unsupported_batch_or_identity', message=str(error))
            return report
        references = self.document['references']
        uncovered = [key for key in keys if key not in references]
        if uncovered:
            report.update(code='uncovered_source_sample', uncovered_source_keys=uncovered,
                          message='The fixed criterion has no reference for these source samples; no pairing was guessed.')
            return report
        report.update(status='pass', code='target_preserved')
        for row, key in enumerate(keys):
            record = references[key]
            for field, observed in (('action', targets[row]), ('action_mask', masks[row])):
                expected = _restore(record[field])
                if expected.shape != observed.shape or expected.dtype != observed.dtype:
                    report.update(status='error', code='target_schema_mismatch')
                    finding({'code': 'target_schema_mismatch', 'field': 'inputs.' + field,
                             'source_key': key, 'expected_shape': list(expected.shape),
                             'observed_shape': list(observed.shape),
                             'expected_dtype': str(expected.dtype), 'observed_dtype': str(observed.dtype)})
                    if report['first_difference'] is None:
                        report['first_difference'] = {'field': 'inputs.' + field, 'source_key': key,
                            'expected_shape': list(expected.shape), 'observed_shape': list(observed.shape),
                            'expected_dtype': str(expected.dtype), 'observed_dtype': str(observed.dtype),
                            'provenance': record['provenance']}
                    continue
                different = np.not_equal(expected, observed)
                count = int(np.count_nonzero(different))
                counter = 'mask' if field == 'action_mask' else 'target'
                report[counter + '_values_checked'] += int(expected.size)
                report['differing_' + counter + '_values'] += count
                if count:
                    report.update(status='error', code='target_values_differ')
                    local = [int(n) for n in np.argwhere(different)[0]]
                    coordinate = tuple(local)
                    finding({'code': 'target_values_differ', 'field': 'inputs.' + field,
                             'source_key': key, 'differing_values': count, 'coordinate': [row, *local],
                             'expected': expected[coordinate].item(), 'observed': observed[coordinate].item()})
                    if report['first_difference'] is None:
                        report['first_difference'] = {'field': 'inputs.' + field, 'source_key': key,
                            'coordinate': [row, *local], 'expected': expected[coordinate].item(),
                            'observed': observed[coordinate].item(), 'provenance': record['provenance']}
            report['samples_checked'] += 1
        return report


class TargetContractViolation(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Native target guard {report['status']}: {report['code']} "
                         f"(criterion {report['criterion_sha256']})")


class GuardedCollator:
    """Run the original collator and gate its actual result before delivery."""

    def __init__(self, native_collator, contract, *, source_keys):
        self.native_collator, self.contract, self.source_keys = native_collator, contract, source_keys
        self.last_report = None
        self.passed_batches = self.blocked_batches = 0

    def __call__(self, features):
        keys = self.source_keys(features)
        if isinstance(keys, (list, tuple)):
            keys = tuple(keys)
        batch = self.native_collator(features)
        self.last_report = self.contract.check(batch, keys)
        if self.last_report['status'] != 'pass':
            self.blocked_batches += 1
            raise TargetContractViolation(self.last_report)
        self.passed_batches += 1
        return batch
