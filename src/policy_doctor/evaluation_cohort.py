"""Declarative checks over observed grouped native JSON; no benchmark-specific rules."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from .artifacts import InputError

MAX_BYTES = 2 * 1024**2
MAX_FILES = 32
MAX_ENTRIES = 64
MAX_DEPTH = 32
MAX_NODES = 100_000
MAX_ROWS = 100_000
MAX_GROUPS = 2000
MAX_RESPONSE = 128 * 1024
PREVIEW = 16
_MISSING = object()
_TABLE_KEYS = {'groups', 'group_id', 'records', 'record_id', 'outcome'}
_CALLABLE = re.compile(r'functools\.partial\s*\(|<function\b|<bound method\b|<built-in function\b')


def _pointer(value):
    if (not isinstance(value, str) or len(value) > 256 or value and not value.startswith('/')
            or re.search(r'~(?![01])', value) or any(ord(c) < 32 for c in value)):
        raise InputError('Selectors must be bounded RFC6901 JSON pointers')
    return value


def validate_parameters(parameters):
    if (not isinstance(parameters, dict) or not {'table', 'equal_fields'} <= parameters.keys()
            or set(parameters) - {'table', 'equal_fields', 'document_pattern'}):
        raise InputError('Expected table, equal_fields and optional document_pattern only')
    table, fields = parameters['table'], parameters['equal_fields']
    if not isinstance(table, dict) or set(table) != _TABLE_KEYS:
        raise InputError('table requires exactly groups, group_id, records, record_id, outcome')
    table = {key: _pointer(table[key]) for key in sorted(_TABLE_KEYS)}
    if not isinstance(fields, list) or len(fields) > 16:
        raise InputError('equal_fields must contain at most 16 JSON pointers')
    fields = [_pointer(value) for value in fields]
    if len(fields) != len(set(fields)):
        raise InputError('equal_fields pointers must be unique')
    pattern = parameters.get('document_pattern', '*.json')
    if (not isinstance(pattern, str) or not pattern or len(pattern) > 128
            or any(c in pattern for c in '/\\') or '..' in pattern
            or any(ord(c) < 32 for c in pattern)):
        raise InputError('document_pattern must be a bounded root filename pattern')
    return {'table': table, 'equal_fields': fields, 'document_pattern': pattern}


def _get(value, pointer):
    for token in pointer[1:].split('/') if pointer else []:
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif (isinstance(value, list) and re.fullmatch(r'0|[1-9][0-9]*', token)
              and len(token) < 10 and int(token) < len(value)):
            value = value[int(token)]
        else:
            return _MISSING
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def _stamp(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _safe(path):
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    for ancestor in [path, *path.parents]:
        if ancestor.is_symlink():
            raise InputError('Symlinks are unsupported in native JSON intake')
    return path


def _read(path):
    initial = path.lstat()
    if not stat.S_ISREG(initial.st_mode) or initial.st_size > MAX_BYTES:
        raise InputError('Expected a regular JSON file no larger than 2 MiB')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        if _stamp(os.fstat(stream.fileno())) != _stamp(initial):
            raise InputError('Native JSON changed before reading')
        raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES or _stamp(os.fstat(stream.fileno())) != _stamp(initial):
            raise InputError('Native JSON changed or exceeded its byte budget')
    if _stamp(path.lstat()) != _stamp(initial):
        raise InputError('Native JSON changed during reading')
    return raw, _stamp(initial)


def _parse(raw):
    text = raw.decode('utf-8')
    # Preflight nesting before the recursive standard JSON decoder allocates a tree.
    depth, quoted, escape = 0, False, False
    for character in text:
        if quoted:
            if escape:
                escape = False
            elif character == '\\':
                escape = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in '[{':
            depth += 1
            if depth > MAX_DEPTH:
                raise InputError('Native JSON nesting exceeds 32')
        elif character in ']}':
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError('Duplicate native JSON object key')
            result[key] = value
        return result

    def constant(_):
        raise InputError('Native JSON must be finite')

    document = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    pending, nodes = [(document, 0)], 0
    while pending:
        value, nesting = pending.pop()
        nodes += 1
        if nodes > MAX_NODES or nesting > MAX_DEPTH:
            raise InputError('Native JSON tree exceeds entry/depth budget')
        if isinstance(value, float) and not math.isfinite(value):
            raise InputError('Native JSON must be finite')
        if isinstance(value, dict):
            pending.extend((item, nesting + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, nesting + 1) for item in value)
    return document


class _Findings:
    def __init__(self):
        self.items, self.count, self.errors, self.unknowns = [], 0, 0, 0

    def add(self, code, status, message, evidence=None):
        self.count += 1
        self.errors += status == 'error'
        self.unknowns += status == 'unknown'
        if len(self.items) < PREVIEW:
            self.items.append({'code': code, 'status': status, 'message': message,
                               'evidence': evidence or []})


def _intake(directory, parameters, notes, role):
    public = {'status': 'unknown', 'document_selected': False, 'artifact': None,
              'matching_files': 0, 'eligible_documents': 0, 'scope': 'one root native JSON document'}
    result = {'public': public, 'document': None, 'observed': [], 'root': None, 'root_stamp': None}
    if directory is None:
        notes.add('reference_missing', 'unknown', 'Supply a reference; no implicit self comparison.')
        return result
    eligible, failed = [], False
    try:
        root = _safe(directory)
        if not root.is_dir():
            raise InputError('Choose an existing flat native JSON directory')
        result.update(root=root, root_stamp=_stamp(root.stat()))
        names, entries = [], 0
        with os.scandir(root) as children:
            for entry in children:
                entries += 1
                if entries > MAX_ENTRIES:
                    raise InputError('Directory exceeds 64 entries')
                mode = entry.stat(follow_symlinks=False).st_mode
                if not stat.S_ISREG(mode):
                    raise InputError('Choose a flat directory without subdirectories, links or special files')
                if fnmatch.fnmatchcase(entry.name, parameters['document_pattern']):
                    names.append(entry.name)
                    if len(names) > MAX_FILES:
                        raise InputError('Native JSON matching-file budget exceeded')
        for name in sorted(names):
            path = root / name
            public['matching_files'] += 1
            try:
                raw, signature = _read(path)
                artifact = {'path': str(path), 'relative_path': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                            'size_bytes': len(raw)}
                result['observed'].append((path, signature, artifact['sha256']))
                document = _parse(raw)
                if isinstance(_get(document, parameters['table']['groups']), list):
                    eligible.append((document, artifact))
            except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                failed = True
                notes.add('native_json_intake_unknown', 'unknown', f'{role}: {str(exc)[:240]}')
        public['eligible_documents'] = len(eligible)
        if failed:
            raise InputError('Incomplete JSON intake cannot establish unique table selection')
        if len(eligible) != 1:
            raise InputError('Select exactly one native document with the declared groups array')
        result['document'], public['artifact'] = eligible[0]
        public.update(status='pass', document_selected=True)
    except (OSError, ValueError, TypeError) as exc:
        notes.add('document_selection_unknown', 'unknown', f'{role}: {str(exc)[:240]}')
    return result


def _evidence(intake, pointer, present=True):
    artifact = intake['public']['artifact']
    return dict(artifact, pointer=pointer, present=present) if artifact else None


def _identity(value, group=False):
    if isinstance(value, str) and value and len(value) <= 512:
        return 'str', value
    if not group and type(value) is int and value.bit_length() <= 256:
        return 'int', value
    return None


def _cohort(intake, table, notes, role):
    summary = {'status': 'unknown', 'group_count': 0, 'total_records': 0, 'unique_records': 0,
               'true_count': 0, 'false_count': 0, 'unknown_outcomes': 0, 'invalid_identities': 0,
               'duplicate_records': 0, 'duplicate_groups': 0, 'rate': None, 'groups': [],
               'groups_omitted': 0, 'identity_inventory_complete': False,
               'identity_scope': 'observed_rows_only; expected ranges/shards/completion unverified'}
    keys, group_ids, complete, valid = {}, {}, True, True
    if intake['document'] is None or intake['public']['status'] != 'pass':
        return summary, keys
    groups = _get(intake['document'], table['groups'])
    summary['group_count'] = len(groups)
    if not groups or len(groups) > MAX_GROUPS:
        notes.add('cohort_empty_or_budget', 'unknown', f'{role}: groups empty or above 2000.')
        return summary, keys
    for gi, group in enumerate(groups):
        gp = table['groups'] + '/' + str(gi)
        identifier = _get(group, table['group_id']) if isinstance(group, dict) else _MISSING
        group_id = _identity(identifier, group=True)
        records = _get(group, table['records']) if isinstance(group, dict) else _MISSING
        group_summary = {'group_id': identifier if group_id else None, 'total_records': 0,
                         'true_count': 0, 'false_count': 0, 'unknown_outcomes': 0}
        if group_id is None:
            complete, valid = False, False
            summary['invalid_identities'] += 1
            notes.add('group_identity_unknown', 'unknown', f'{role}: invalid/missing native group ID.',
                      [_evidence(intake, gp + table['group_id'], identifier is not _MISSING)])
        elif group_id in group_ids:
            summary['duplicate_groups'] += 1
            complete, valid = False, False
            notes.add('group_identity_duplicate', 'error', f'{role}: duplicate native group ID.',
                      [_evidence(intake, group_ids[group_id]), _evidence(intake, gp + table['group_id'])])
        else:
            group_ids[group_id] = gp + table['group_id']
        if not isinstance(records, list):
            complete, valid = False, False
            notes.add('records_unknown', 'unknown', f'{role}: missing/non-array native records.',
                      [_evidence(intake, gp + table['records'], records is not _MISSING)])
            continue
        summary['total_records'] += len(records)
        group_summary['total_records'] = len(records)
        if summary['total_records'] > MAX_ROWS:
            notes.add('record_budget', 'unknown', f'{role}: observed records exceed 100000.')
            complete, valid = False, False
            break
        for ri, record in enumerate(records):
            rp = gp + table['records'] + '/' + str(ri)
            identity_value = _get(record, table['record_id']) if isinstance(record, dict) else _MISSING
            record_id = _identity(identity_value)
            if group_id is None or record_id is None:
                complete, valid = False, False
                summary['invalid_identities'] += 1
                notes.add('record_identity_unknown', 'unknown', f'{role}: invalid/missing native record ID.',
                          [_evidence(intake, rp + table['record_id'], identity_value is not _MISSING)])
            else:
                key = (group_id[1], *record_id)
                if key in keys:
                    summary['duplicate_records'] += 1
                    complete, valid = False, False
                    notes.add('record_identity_duplicate', 'error', f'{role}: duplicate typed native record key.',
                              [_evidence(intake, keys[key]), _evidence(intake, rp + table['record_id'])])
                else:
                    keys[key] = rp + table['record_id']
            outcome = _get(record, table['outcome']) if isinstance(record, dict) else _MISSING
            if type(outcome) is bool:
                field = 'true_count' if outcome else 'false_count'
                summary[field] += 1
                group_summary[field] += 1
            else:
                valid = False
                summary['unknown_outcomes'] += 1
                group_summary['unknown_outcomes'] += 1
                notes.add('outcome_unknown', 'unknown', f'{role}: native outcome is missing or not Boolean.',
                          [_evidence(intake, rp + table['outcome'], outcome is not _MISSING)])
        if len(summary['groups']) < PREVIEW:
            summary['groups'].append(group_summary)
    summary['groups_omitted'] = max(0, summary['group_count'] - len(summary['groups']))
    summary['unique_records'] = len(keys)
    summary['identity_inventory_complete'] = complete
    if not summary['total_records']:
        valid = False
        notes.add('empty_denominator', 'unknown', f'{role}: no observed outcome records.')
    if summary['duplicate_records'] or summary['duplicate_groups']:
        summary['status'] = 'error'
    elif valid and complete:
        summary['status'] = 'pass'
        summary['rate'] = summary['true_count'] / summary['total_records']
    return summary, keys


def _callable(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str) and _CALLABLE.search(item):
            return True
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _value_evidence(intake, pointer):
    if intake['document'] is None:
        return None, _MISSING
    value = _get(intake['document'], pointer)
    evidence = _evidence(intake, pointer, value is not _MISSING)
    if value is not _MISSING:
        encoded = _canonical(value)
        evidence['canonical_sha256'] = hashlib.sha256(encoded).hexdigest()
        if len(encoded) <= 512:
            evidence['value'] = value
        else:
            evidence.update(preview=encoded[:256].decode('ascii'), value_bytes=len(encoded), preview_truncated=True)
    return evidence, value


def _literal_equal(left, right):
    """Compare finite parsed JSON values; preserve IDs separately as typed keys."""
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_literal_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_literal_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _fields(candidate, reference, pointers, notes):
    result = []
    for pointer in pointers:
        left, lv = _value_evidence(reference, pointer)
        right, rv = _value_evidence(candidate, pointer)
        status, reason = 'pass', 'Selected JSON values match under numeric-format equivalence; original byte/canonical hashes may differ.'
        if lv is _MISSING or rv is _MISSING:
            status, reason = 'unknown', 'Selected native context is missing; no defaults supplied.'
        elif _callable(lv) or _callable(rv):
            status, reason = 'unknown', 'Callable-looking values remain inert text; semantics unknown.'
        elif not _literal_equal(lv, rv):
            status, reason = 'error', 'Selected JSON context values differ under numeric-format equivalence.'
        if status != 'pass':
            notes.add('selected_context_' + status, status, reason, [item for item in (left, right) if item])
        result.append({'pointer': pointer, 'status': status, 'reason': reason,
                       'reference': left, 'candidate': right})
    return result


def _key_preview(key, intake, keys):
    return {'group_id': key[0], 'record_id_type': key[1], 'record_id': key[2],
            'evidence': _evidence(intake, keys[key])}


def _recheck(intake):
    if intake['root'] is None:
        return
    if _stamp(intake['root'].stat()) != intake['root_stamp']:
        raise InputError('Native source directory changed during audit')
    for path, signature, digest in intake['observed']:
        raw, current = _read(path)
        if current != signature or hashlib.sha256(raw).hexdigest() != digest:
            raise InputError('Native source JSON changed during audit')


def audit_evaluation_cohort(directory, *, reference=None, table, equal_fields, document_pattern='*.json'):
    parameters = validate_parameters({'table': table, 'equal_fields': equal_fields,
                                      'document_pattern': document_pattern})
    notes = _Findings()
    candidate = _intake(directory, parameters, notes, 'candidate')
    ref = _intake(reference, parameters, notes, 'reference')
    cand_summary, cand_keys = _cohort(candidate, parameters['table'], notes, 'candidate')
    ref_summary, ref_keys = _cohort(ref, parameters['table'], notes, 'reference')
    comparison = {'status': 'unknown', 'missing_count': 0, 'extra_count': 0, 'missing': [], 'extra': [],
                  'scope': 'typed native key equality over observed rows only',
                  'declared_completeness_status': 'unknown'}
    if (cand_summary['identity_inventory_complete'] and ref_summary['identity_inventory_complete']
            and cand_summary['total_records'] and ref_summary['total_records']):
        missing, extra = set(ref_keys) - set(cand_keys), set(cand_keys) - set(ref_keys)
        def ordered(keys):
            return sorted(keys, key=lambda key: (key[0], key[1], str(key[2])))
        comparison.update(status='error' if missing or extra else 'pass',
                          missing_count=len(missing), extra_count=len(extra),
                          missing=[_key_preview(key, ref, ref_keys) for key in ordered(missing)[:PREVIEW]],
                          extra=[_key_preview(key, candidate, cand_keys) for key in ordered(extra)[:PREVIEW]])
        if missing or extra:
            notes.add('observed_cohort_mismatch', 'error', 'Complete observed native key inventories differ.',
                      [entry['evidence'] for entry in comparison['missing'][:1] + comparison['extra'][:1]])
    fields = _fields(candidate, ref, parameters['equal_fields'], notes)
    try:
        _recheck(candidate)
        _recheck(ref)
    except (OSError, ValueError) as exc:
        raise InputError(f'Stable original evidence required: {exc}') from exc
    status = ('error' if notes.errors else 'unknown' if notes.unknowns or comparison['status'] != 'pass'
              or cand_summary['status'] != 'pass' or ref_summary['status'] != 'pass' else 'pass')
    report = {'kind': 'evaluation_cohort_audit', 'status': status, 'parameters': parameters,
              'source_intake': candidate['public'], 'reference_intake': ref['public'],
              'candidate': cand_summary, 'reference': ref_summary, 'cohort_comparison': comparison,
              'completeness': {'declared_counts': 'unknown', 'shards': 'unknown',
                               'terminal_retry_status': 'unknown', 'native_summaries': 'unknown'},
              'fields': fields, 'findings': notes.items, 'finding_count': notes.count,
              'findings_omitted': max(0, notes.count - len(notes.items)),
              'finding_status_counts': {'error': notes.errors, 'unknown': notes.unknowns},
              'unverified': ['physical state pairing', 'effective runtime and checkpoint/source/asset identity',
                             'causal policy improvement', 'declared cohort completeness',
                             'terminal/retry/shard completeness', 'provided native summaries'],
              'scope': 'Observed native lexical identities and Boolean outcome counts; not qualified policy scores.',
              'next_action': 'Retain these selectors; supply missing native evidence before broader claims.'}
    if len(_canonical(report)) > MAX_RESPONSE:
        report.update(status='unknown', fields=[], findings=[{
            'code': 'response_budget', 'status': 'unknown',
            'message': 'Requested report exceeds 128 KiB; narrow selected context before reuse.', 'evidence': []}],
            finding_count=notes.count + 1, findings_omitted=notes.count,
            finding_status_counts={'error': notes.errors, 'unknown': notes.unknowns + 1},
            response_budget_exceeded=True)
        for summary in (report['candidate'], report['reference']):
            summary['groups_omitted'] = summary['group_count']
            summary['groups'] = []
        comparison['missing'], comparison['extra'] = [], []
        if len(_canonical(report)) > MAX_RESPONSE:
            raise InputError('Native cohort report exceeds bounded response budget')
    return report
