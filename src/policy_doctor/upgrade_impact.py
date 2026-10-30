"""Asset-bound native target impact checks under an independently retained rule.

No package resolution, migration rule inference, source execution or network I/O
occurs here. File identity and supplied capture associations are evidence, not
authentication of a historical producer or attribution to a particular edit.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .artifacts import InputError
from .conversion_fidelity import _read_small, _reference_json
from .training_guard import TargetContract, _descriptor, _encode, _restore
from .training_guard_capture import load_capture

QUALIFIED_CONSUMER = 'gr00t-n1.7-unitree-g1'
QUALIFIED_REVISION = '944fe70c4fd54e3fcb80a7d7258a328f9c061176'
# Original native source used by the retained trainer qualification. These are
# coverage pins, not a list of migration rules or defaults for other revisions.
QUALIFIED_SOURCES = {
    'gr00t/data/collator/collators.py': '2fb50c5a808d023aa759043f4ddaf54a313efc04ff036a3c16b2c126b808c516',
    'gr00t/data/state_action/state_action_processor.py':
        '0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145',
    'gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py':
        '77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f',
}
ISAACLAB_SOURCES = {
    '37ddf626871758333d6ed89cf64ad702aef127d0': {
        'datasets/hdf5_dataset_file_handler.py': '7250793428126dd5a579fb593501233e30321aa8de3b9716a34a8b2b2aa06ff6',
        'datasets/episode_data.py': 'f01fe8f6b27ac5a2a49fc67310fa911fa9285c31022714aa7d12b7acf6443448',
        'datasets/dataset_file_handler_base.py': 'c12d2f55d3125e6c6425e9a3e4d3f3df93fd6cb325941d6122b02fe0bc5ee241',
    },
    'e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4': {
        'datasets/hdf5_dataset_file_handler.py': '92f795f7e6b9aebbefc6f0eb66b179a4010d6a618a868d3c7956236f9185f97d',
        'datasets/episode_data.py': '23cac25ed8cc5fb4827c1b5c82fb4b2fe0a309c1f534a6059e04e26607197fd8',
        'datasets/dataset_file_handler_base.py': '020aa68e4a623322a47787b00415f5dafdf746b66b02ef9bf41521f3fd8d844b',
        'math.py': 'be896b1734900f5ed5751de75a8ad517f131358934f5247968bd5f905ca9cd47',
    },
}
MAX_BYTES = 4 * 1024**2
MAX_ASSETS = 32
MAX_FILES = 32
SCOPE = ('Covered native numeric boundary fields under one retained criterion. '
         'Caller-supplied snapshots may include declared reference adapters. '
         'Configuration associations are supplied by the caller; no causal upgrade attribution, individual-file causality, '
         'historical producer authentication, model-quality or physical benefit is established.')


def _boundary_arrays(fields):
    if type(fields) is not dict or not 1 <= len(fields) <= 128:
        raise InputError('Provide between one and 128 explicit native boundary fields')
    result, total = {}, 0
    for key, value in fields.items():
        _text(key, 'Boundary field/source pointer')
        if (not isinstance(value, np.ndarray) or value.ndim != 2 or not all(value.shape)
                or value.dtype.kind not in 'bifu' or not np.isfinite(value).all()):
            raise InputError('Boundary fields require finite two-dimensional numeric NumPy arrays')
        total += value.size
        result[key] = value
    if total > 1_000_000:
        raise InputError('Boundary fields exceed the supported value budget')
    return result


def save_boundary_capture(fields, path):
    """Save supplied native field snapshots; no simulator call is manufactured."""
    value = {'schema_version': 1, 'kind': 'native_boundary_capture',
             'fields': {key: _descriptor(array) for key, array in _boundary_arrays(fields).items()}}
    _save_new(path, _encode(value))


def _save_new(path, raw):
    if len(raw) > MAX_BYTES:
        raise InputError('Native boundary document exceeds its byte bound')
    try:
        with Path(path).open('xb') as stream:
            stream.write(raw)
    except OSError as error:
        raise InputError('Native boundary output requires a new writable file') from error


class BoundaryContract:
    """An explicit numeric field criterion, never inferred from a candidate."""

    def __init__(self, document):
        _object(document, {'schema_version', 'kind', 'intent', 'fields'}, 'Boundary criterion')
        if (type(document['schema_version']) is not int or document['schema_version'] != 1
                or document['kind'] != 'native_boundary_preservation'
                or type(document['intent']) is not dict or not document['intent']
                or type(document['fields']) is not dict):
            raise InputError('Unsupported native boundary criterion')
        _boundary_arrays({key: _restore(value) for key, value in document['fields'].items()})
        self._canonical = _encode(document)
        if len(self._canonical) > MAX_BYTES:
            raise InputError('Native boundary criterion exceeds its byte bound')
        self.criterion_sha256 = hashlib.sha256(self._canonical).hexdigest()

    @property
    def fields(self):
        return {key: _restore(value) for key, value in json.loads(self._canonical)['fields'].items()}

    @classmethod
    def from_reference(cls, fields, *, intent):
        return cls({'schema_version': 1, 'kind': 'native_boundary_preservation', 'intent': intent,
                    'fields': {key: _descriptor(array) for key, array in _boundary_arrays(fields).items()}})

    def save(self, path):
        _save_new(path, self._canonical)

    def check(self, fields):
        fields = _boundary_arrays(fields)
        expected = self.fields
        report = {'status': 'unknown', 'code': 'uncovered_boundary_fields',
                  'criterion_sha256': self.criterion_sha256, 'fields_checked': 0,
                  'differing_values': 0, 'findings': [], 'finding_count': 0}
        # Partial captures do not turn incomplete coverage into a healthy result.
        if set(fields) != set(expected):
            report.update(missing_fields=sorted(set(expected) - set(fields)),
                          uncovered_fields=sorted(set(fields) - set(expected)))
            return report
        report.update(status='pass', code='boundary_preserved')
        for key in sorted(fields):
            a, b = expected[key], fields[key]
            item = {'field': key, 'expected_shape': list(a.shape), 'observed_shape': list(b.shape),
                    'expected_dtype': str(a.dtype), 'observed_dtype': str(b.dtype)}
            if a.shape != b.shape or a.dtype != b.dtype:
                item['code'] = 'boundary_schema_mismatch'
            else:
                different = np.not_equal(a, b)
                if not different.any():
                    report['fields_checked'] += 1
                    continue
                index = tuple(np.argwhere(different)[0])
                count = int(np.count_nonzero(different))
                report['differing_values'] += count
                item.update(code='boundary_values_differ', differing_values=count,
                            coordinate=[int(n) for n in index], expected=a[index].item(), observed=b[index].item())
            report.update(status='error', code=item['code'])
            report['finding_count'] += 1
            if len(report['findings']) < 8:
                report['findings'].append(item)
            report['fields_checked'] += 1
        return report


def _load_boundary(path):
    raw, _ = _read_small(path, limit=MAX_BYTES)
    value = _reference_json(raw)
    _object(value, {'schema_version', 'kind', 'fields'}, 'Native boundary capture')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['kind'] != 'native_boundary_capture' or type(value['fields']) is not dict):
        raise InputError('Unsupported native boundary capture')
    return _boundary_arrays({key: _restore(item) for key, item in value['fields'].items()})


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise InputError(f'{label} requires exactly: {", ".join(sorted(keys))}')
    return value


def _text(value, label):
    if type(value) is not str or not value or len(value) > 512 or not value.isprintable():
        raise InputError(f'{label} requires a bounded printable string')
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise InputError('Artifact identity requires a lowercase SHA256')
    return value


def _file(value, base, tracked):
    _object(value, {'path', 'sha256'}, 'File identity')
    path = Path(_text(value['path'], 'File path'))
    path = str(path if path.is_absolute() else base / path)
    raw, _ = _read_small(path, limit=MAX_BYTES)
    pin = _sha(value['sha256'])
    if hashlib.sha256(raw).hexdigest() != pin:
        raise InputError(f'Input changed from retained SHA256: {path!r}')
    if path in tracked and tracked[path] != pin:
        raise InputError('One input path cannot name two different retained identities')
    tracked[path] = pin
    return {'path': path, 'sha256': pin}


def _files(value, base, tracked, *, allow_empty=False):
    if type(value) is not dict or not (0 if allow_empty else 1) <= len(value) <= MAX_FILES:
        raise InputError('Provide a bounded named file mapping for each snapshot')
    return {_text(name, 'Logical file name'): _file(item, base, tracked)
            for name, item in sorted(value.items())}


def _snapshot(value, base, tracked):
    _object(value, {'consumer', 'revision', 'source_files', 'configs', 'capture'}, 'Snapshot')
    return {'consumer': _text(value['consumer'], 'Consumer'),
            'revision': _text(value['revision'], 'Native revision'),
            'source_files': _files(value['source_files'], base, tracked, allow_empty=True),
            'configs': _files(value['configs'], base, tracked),
            'capture': _file(value['capture'], base, tracked)}


def _coverage(snapshot, boundary=False):
    if boundary:
        if snapshot['consumer'] != 'isaaclab-hdf5-xyzw' or snapshot['revision'] not in ISAACLAB_SOURCES:
            return 'unsupported_consumer_or_revision'
        required = ISAACLAB_SOURCES[snapshot['revision']]
    elif snapshot['consumer'] != QUALIFIED_CONSUMER or snapshot['revision'] != QUALIFIED_REVISION:
        return 'unsupported_consumer_or_revision'
    else:
        required = QUALIFIED_SOURCES
    if any(snapshot['source_files'].get(name, {}).get('sha256') != pin
           for name, pin in required.items()):
        return 'unqualified_native_source'
    return None


def _delta(before, before_keys, after, after_keys):
    if len(set(before_keys)) != len(before_keys) or len(set(after_keys)) != len(after_keys):
        return {'status': 'unknown', 'code': 'ambiguous_duplicate_source_keys', 'changes': []}
    if set(before_keys) != set(after_keys):
        return {'status': 'unknown', 'code': 'different_source_cohort', 'changes': []}
    changes = []
    count = 0
    for row, key in enumerate(before_keys):
        other = after_keys.index(key)
        for field in ('action', 'action_mask'):
            a, b = before['inputs'][field][row], after['inputs'][field][other]
            item = {'source_key': key, 'field': 'inputs.' + field,
                    'reference_shape': list(a.shape), 'candidate_shape': list(b.shape),
                    'reference_dtype': str(a.dtype), 'candidate_dtype': str(b.dtype)}
            if a.shape != b.shape or a.dtype != b.dtype:
                item['code'] = 'schema_changed'
            else:
                differing = np.not_equal(a, b)
                if not differing.any():
                    continue
                index = tuple(np.argwhere(differing)[0])
                item.update(code='values_changed', differing_values=int(np.count_nonzero(differing)),
                            coordinate=[int(n) for n in index], reference=a[index].item(), candidate=b[index].item())
            count += 1
            if len(changes) < 8:
                changes.append(item)
    return {'status': 'different' if count else 'equal', 'change_count': count,
            'changes': changes, 'changes_truncated': count > len(changes)}


def _config_changes(before, after):
    return [{'name': name, 'reference': before.get(name), 'candidate': after.get(name)}
            for name in sorted(set(before) | set(after))
            if before.get(name, {}).get('sha256') != after.get(name, {}).get('sha256')]


def assess_upgrade_impact(manifest_path):
    """Check named user assets; unsupported native coverage remains UNKNOWN.

    Relative paths resolve beside the manifest. The manifest supplies the
    independently retained contract pin and both actual saved configurations,
    native source pins and captures for each asset. No reference is rebuilt.
    """
    manifest_path = Path(manifest_path).absolute()
    raw, _ = _read_small(str(manifest_path), limit=MAX_BYTES)
    manifest = _reference_json(raw)
    _object(manifest, {'schema_version', 'kind', 'contract', 'assets'}, 'Impact manifest')
    if (type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
            or manifest['kind'] != 'native_upgrade_impact'
            or type(manifest['assets']) is not list or not 1 <= len(manifest['assets']) <= MAX_ASSETS):
        raise InputError('Unsupported impact manifest schema or asset count')
    tracked = {str(manifest_path): hashlib.sha256(raw).hexdigest()}
    base = manifest_path.parent
    contract_file = _file(manifest['contract'], base, tracked)
    contract_raw, _ = _read_small(contract_file['path'], limit=MAX_BYTES)
    contract_document = _reference_json(contract_raw)
    boundary = type(contract_document) is dict and contract_document.get('kind') == 'native_boundary_preservation'
    contract = (BoundaryContract(contract_document) if boundary else
                TargetContract.load(contract_file['path'], expected_sha256=contract_file['sha256']))
    if contract.criterion_sha256 != contract_file['sha256']:
        raise InputError('Boundary criterion must preserve canonical retained bytes')
    criterion_intent = contract_document['intent']
    comparison_basis = {
        'reference_snapshot': 'Caller-supplied snapshot, including any declared reference adapter',
        'candidate_snapshot': 'Caller-supplied snapshot; historical native producer identity is not authenticated',
        'criterion': 'Both snapshots are checked against the same retained exact criterion',
        'reference_adapter': criterion_intent.get('reference_adapter'),
        'causal_attribution': 'Not established for upstream revisions or associated file edits',
    }
    assets, identifiers = [], set()
    for value in manifest['assets']:
        _object(value, {'asset_id', 'reference', 'candidate'}, 'User asset')
        name = _text(value['asset_id'], 'Asset identity')
        if name in identifiers:
            raise InputError('User asset identities must be unique')
        identifiers.add(name)
        before, after = [_snapshot(value[side], base, tracked) for side in ('reference', 'candidate')]
        change_kind = ('upstream_revision_change' if (before['consumer'], before['revision']) !=
                       (after['consumer'], after['revision']) else 'configuration_comparison')
        item = {'asset_id': name, 'status': 'unknown', 'code': 'unassessed',
                'change_kind': change_kind, 'criterion_sha256': contract.criterion_sha256,
                'criterion_intent': criterion_intent, 'comparison_basis': comparison_basis,
                'reference': before, 'candidate': after,
                'associated_config_changes': _config_changes(before['configs'], after['configs'])}
        unsupported = [reason for snap in (before, after) if (reason := _coverage(snap, boundary))]
        if unsupported:
            item.update(code=unsupported[0], coverage={'status': 'unknown', 'reasons': unsupported})
        else:
            if boundary:
                a, b = [_load_boundary(snap['capture']['path']) for snap in (before, after)]
                reference, candidate = contract.check(a), contract.check(b)
                if reference['status'] == 'pass':
                    delta = {'status': {'pass': 'equal', 'error': 'different', 'unknown': 'unknown'}[candidate['status']],
                             'changes': candidate['findings'], 'change_count': candidate['finding_count'],
                             'basis': 'Candidate versus retained intended field values; the passing reference snapshot may be adapted'}
                else:
                    delta = {'status': 'unknown', 'code': 'reference_does_not_pass_retained_criterion', 'changes': []}
            else:
                a, ak = load_capture(before['capture']['path'])
                b, bk = load_capture(after['capture']['path'])
                reference, candidate = contract.check(a, ak), contract.check(b, bk)
                delta = _delta(a, ak, b, bk)
                delta['basis'] = 'Numeric comparison of caller-supplied native-target snapshots'
            delta.update(criterion_intent=criterion_intent, comparison_basis=comparison_basis)
            item.update(reference_check=reference, candidate_check=candidate,
                        native_boundary_delta=delta, coverage={'status': 'covered'})
            if reference['status'] != 'pass':
                item.update(code='reference_does_not_pass_retained_criterion')
            elif candidate['status'] == 'unknown' or delta['status'] == 'unknown':
                item.update(code=candidate['code'] if candidate['status'] == 'unknown' else delta['code'])
            elif candidate['status'] == 'error':
                item.update(status='changed', code='retained_boundary_criterion_violated')
            else:
                item.update(status='healthy', code='retained_criterion_preserved')
        assets.append(item)
    # Both parsing and comparison reread inputs; a result cannot quietly combine
    # different input versions. The final postguard includes the manifest itself.
    for path, pin in tracked.items():
        raw, _ = _read_small(path, limit=MAX_BYTES)
        if hashlib.sha256(raw).hexdigest() != pin:
            raise InputError(f'Input changed during impact assessment: {path!r}')
    counts = {state: sum(item['status'] == state for item in assets) for state in ('changed', 'healthy', 'unknown')}
    status = 'changed' if counts['changed'] else 'unknown' if counts['unknown'] else 'healthy'
    return {'schema_version': 1, 'kind': 'upgrade_impact', 'status': status,
            'criterion_sha256': contract.criterion_sha256, 'manifest_sha256': tracked[str(manifest_path)],
            'criterion_intent': criterion_intent, 'comparison_basis': comparison_basis,
            'counts': counts, 'affected_asset_ids': [a['asset_id'] for a in assets if a['status'] == 'changed'],
            'assets': assets, 'source_input_postguard': 'pass', 'scope': SCOPE}


def write_upgrade_alerts(report, output_dir):
    """Publish deterministic local review files in a new directory; send nothing.

    CI policy:0 healthy,1 retained criterion violation,3 incomplete coverage. Input errors
    raise InputError and are separate from these assessment outcomes.
    """
    if (type(report) is not dict or report.get('kind') != 'upgrade_impact'
            or report.get('status') not in ('healthy', 'changed', 'unknown')):
        raise InputError('Expected a complete upgrade-impact assessment')
    raw = _encode(report)
    lines = ['# Native upgrade impact', '', f"Status: {report['status'].upper()}",
             f"Criterion: `{report['criterion_sha256']}`",
             'Criterion intent: ' + json.dumps(report['criterion_intent'], sort_keys=True, ensure_ascii=True),
             'Comparison basis: ' + json.dumps(report['comparison_basis'], sort_keys=True, ensure_ascii=True), '']
    for asset in report['assets']:
        # JSON quoting keeps arbitrary supplied labels visible as data, including
        # Markdown punctuation; no executable CI workflow commands are emitted.
        lines.append(f"- {json.dumps(asset['asset_id'], ensure_ascii=True)}: "
                     f"{asset['status'].upper()} ({asset['code']}); "
                     f"{len(asset['associated_config_changes'])} associated configuration file changes.")
        for change in asset['associated_config_changes']:
            lines.append('  - Associated file: ' + json.dumps(change['name'], ensure_ascii=True))
        for change in asset.get('native_boundary_delta', {}).get('changes', [])[:2]:
            lines.append('  - Retained intended-boundary comparison: ' + json.dumps(change, sort_keys=True, ensure_ascii=True))
    lines.extend(['', SCOPE, ''])
    output = Path(output_dir)
    try:
        output.mkdir(parents=False, exist_ok=False)
        with (output / 'impact.json').open('xb') as stream:
            stream.write(raw + b'\n')
        with (output / 'alerts.md').open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines))
    except OSError as error:
        raise InputError('Alert output requires a new writable directory with an existing parent') from error
    return {'status': report['status'], 'exit_code': {'healthy': 0, 'changed': 1, 'unknown': 3}[report['status']],
            'report': str(output / 'impact.json'), 'alerts': str(output / 'alerts.md')}
