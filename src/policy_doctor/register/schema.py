"""Strict v1 entry schema. JSON syntax only, including files named .yaml."""
from __future__ import annotations

import math
import re
from datetime import date
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from packaging.version import Version

MAX_ENTRIES = 128
MAX_BYTES = 256 * 1024
MAX_PREDICATES = 24
MAX_NODES = 4096
MAX_DEPTH = 16
COMPARATORS = {'eq', 'ne', 'in', 'exists', 'lt', 'lte', 'gt', 'gte'}
VERSION_OPS = {'eq', 'gte', 'gt', 'lte', 'lt'}
ENTRY_FIELDS = {'schema_version', 'id', 'kind', 'title', 'stacks', 'claim_basis', 'applies_when',
                'violation_when', 'severity', 'evidence', 'healthy_controls', 'first_fix',
                'unknown_when', 'last_verified', 'validation'}
OPTIONAL_ENTRY_FIELDS = {'recipe_field', 'launch_target'}
PREDICATES = {'package_version', 'checkpoint_id_in', 'config_key', 'processor_key', 'launch_flag', 'source_files', 'source_contains'}


def bounded_value(value):
    """Iterative node/depth bounds, strict JSON types, finite numbers only."""
    queue = [(value, 0)]
    count = 0
    while queue:
        item, depth = queue.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError('JSON exceeds the register node/depth bound.')
        if type(item) is dict:
            if any(type(k) is not str or len(k) > 256 for k in item):
                raise ValueError('Object keys must be bounded strings.')
            queue.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            queue.extend((v, depth + 1) for v in item)
        elif type(item) is str:
            if len(item) > 16384 or any(ord(c) < 32 and c not in '\n\t\r' for c in item):
                raise ValueError('Text exceeds register bounds or contains controls.')
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Non-finite numbers are unsupported.')
        elif type(item) is int:
            if item.bit_length() > 128:
                raise ValueError('Integer exceeds register bounds.')
        elif item is not None and type(item) is not bool:
            raise ValueError('Only JSON values are supported.')
    return value


def text(value, label, maximum=2048):
    if type(value) is not str or not value.strip() or len(value) > maximum or not value.isprintable():
        raise ValueError(label + ' must be nonempty bounded printable text.')
    return value


def same_json(left, right):
    """Strict equality of already bounded JSON, including nested scalar types."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(same_json(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(same_json(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def version(value):
    """Maintained PEP 440 parsing, bounded before parsing; never Git ordering."""
    if type(value) is not str or not value or len(value) > 200 or not value.isprintable():
        raise ValueError('Version must be bounded PEP 440 text.')
    return Version(value)


def relative_path(value):
    text(value, 'source path', 512)
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith('~') or '..' in path.parts or '\\' in value or not path.parts or str(path) != value:
        raise ValueError('Source paths must be canonical relative POSIX paths.')
    return value


def _strings(value, label, minimum=1, maximum=32):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError(label + ' must be a bounded list.')
    for item in value:
        text(item, label)


def validate_predicate(predicate):
    if type(predicate) is not dict or len(predicate) != 1 or not set(predicate) <= PREDICATES:
        raise ValueError('Each predicate must contain exactly one supported predicate.')
    kind, body = next(iter(predicate.items()))
    if kind == 'source_contains':
        if type(body) is not dict or set(body) - {'path', 'literal', 'occurrences'} or not {'path', 'literal'} <= set(body):
            raise ValueError('source_contains requires a path and literal only, plus optional occurrence bounds.')
        relative_path(body['path'])
        text(body['literal'], 'source literal', 4096)
        bounds = body.get('occurrences', {'gte': 1})
        if type(bounds) is not dict or len(bounds) != 1 or not set(bounds) <= {'gte', 'eq'}:
            raise ValueError('Occurrence bounds support one gte/eq integer only.')
        if any(type(v) is not int or not (0 if op == 'eq' else 1) <= v <= 100000 for op, v in bounds.items()):
            raise ValueError('Occurrence count requires eq:0..100000 or gte:1..100000.')
        return
    if kind == 'checkpoint_id_in':
        _strings(body, kind)
        return
    if kind == 'source_files':
        if type(body) is not dict or set(body) != {'files'} or type(body['files']) is not list or not 1 <= len(body['files']) <= 32:
            raise ValueError('source_files requires 1..32 file hashes.')
        seen = set()
        for row in body['files']:
            if type(row) is not dict or set(row) not in ({'path', 'sha256'}, {'path', 'sha256_in'}):
                raise ValueError('Source file requires path and exactly one of sha256 or sha256_in.')
            relative_path(row['path'])
            hashes = row.get('sha256_in', [row.get('sha256')])
            if type(hashes) is not list or not (2 <= len(hashes) <= 32 if 'sha256_in' in row else len(hashes) == 1):
                raise ValueError('sha256_in requires 2..32 verified hash alternatives.')
            if any(type(h) is not str or not re.fullmatch('[a-f0-9]{64}', h) for h in hashes):
                raise ValueError('Source file hash is invalid.')
            if row['path'] in seen or len(set(hashes)) != len(hashes):
                raise ValueError('Source file hash is invalid or path repeated.')
            seen.add(row['path'])
        return
    if type(body) is not dict:
        raise ValueError('Predicate body must be an object.')
    if kind == 'package_version':
        if set(body) != {'package', 'version'}:
            raise ValueError('package_version requires package and version only.')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}', text(body['package'], 'package', 100)):
            raise ValueError('Invalid package name.')
        bounds = body['version']
        if type(bounds) is not dict or not bounds or not set(bounds) <= VERSION_OPS:
            raise ValueError('Unsupported version bounds.')
        parsed = {op: version(v) for op, v in bounds.items()}
        if ('gt' in parsed and 'gte' in parsed) or ('lt' in parsed and 'lte' in parsed) or ('eq' in parsed and len(parsed) != 1):
            raise ValueError('Ambiguous version bounds.')
        lower, upper = parsed.get('gt', parsed.get('gte')), parsed.get('lt', parsed.get('lte'))
        if lower is not None and upper is not None and (lower > upper or lower == upper and ('gt' in parsed or 'lt' in parsed)):
            raise ValueError('Empty version interval.')
        return
    key = 'flag' if kind == 'launch_flag' else 'path'
    if set(body) - {key, 'op', 'value', 'default_when_absent'} or key not in body or body.get('op') not in COMPARATORS:
        raise ValueError('Invalid value predicate.')
    field = text(body[key], key, 256)
    if kind == 'launch_flag':
        if not re.fullmatch(r'--[a-zA-Z0-9][a-zA-Z0-9_.-]*', field):
            raise ValueError('Launch flag must be an explicit long option.')
    elif not re.fullmatch(r'[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){0,15}', field):
        raise ValueError('Config path must be a bounded dotted object path.')
    if body['op'] == 'exists':
        if set(body) not in ({key, 'op'}, {key, 'op', 'value'}) or type(body.get('value', True)) is not bool:
            raise ValueError('exists accepts only an optional strict boolean value (default true).')
    elif 'value' not in body:
        raise ValueError('Value predicate is missing its comparison value.')
    elif body['op'] == 'in' and (type(body['value']) is not list or not 1 <= len(body['value']) <= 32):
        raise ValueError('in requires 1..32 literal values.')
    elif body['op'] in ('lt', 'lte', 'gt', 'gte') and (kind == 'launch_flag' or type(body['value']) not in (int, float)):
        raise ValueError('Numeric comparisons require config/processor finite numeric values.')
    if 'default_when_absent' in body:
        default = body['default_when_absent']
        if type(default) is not dict or set(default) != {'value', 'source', 'sha256'}:
            raise ValueError('A default requires value, pinned source and exact sha256.')
        validate_source(default['source'])
        if type(default['sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', default['sha256']):
            raise ValueError('Default source hash is invalid.')


def validate_source(source):
    if type(source) is not dict or set(source) != {'repo', 'sha', 'path', 'line'}:
        raise ValueError('Source requires repo, sha, path and line.')
    text(source['repo'], 'repo', 256)
    relative_path(source['path'])
    if type(source['sha']) is not str or not re.fullmatch('[a-f0-9]{40}', source['sha']) or type(source['line']) is not int or not 1 <= source['line'] <= 10000000:
        raise ValueError('Source SHA/line is invalid.')


def _recipe(entry):
    recipe = entry.get('recipe_field')
    if type(recipe) is not dict or not {'field', 'checkpoint', 'status'} <= set(recipe):
        raise ValueError('Recipe requires explicit recipe_field metadata.')
    text(recipe['field'], 'recipe field', 256)
    text(recipe['checkpoint'], 'recipe checkpoint', 512)
    identities = [p['checkpoint_id_in'] for p in entry['applies_when'] if 'checkpoint_id_in' in p]
    if not identities or not all(recipe['checkpoint'] in values for values in identities):
        raise ValueError('Recipe checkpoint must be explicitly bound in every checkpoint_id_in applicability predicate.')
    status = recipe['status']
    if status == 'published':
        if set(recipe) != {'field', 'checkpoint', 'status', 'published_value', 'source_index'}:
            raise ValueError('Published recipe requires value and one unique pinned source_index.')
        index = recipe['source_index']
        if type(index) is not int or not 0 <= index < len(entry['evidence']):
            raise ValueError('Recipe source_index is outside the pinned evidence list.')
        if entry['evidence'][index].get('basis') != 'publisher_recipe':
            raise ValueError('Published recipe source_index must select publisher_recipe evidence.')
        tests = entry['violation_when']
        if len(tests) != 1 or not set(tests[0]) <= {'config_key', 'processor_key', 'launch_flag'}:
            raise ValueError('One recipe field requires exactly one value violation predicate.')
        predicate = next(iter(tests[0].values()))
        if (predicate['op'] != 'ne' or predicate.get('path', predicate.get('flag')) != recipe['field']
                or not same_json(predicate['value'], recipe['published_value'])):
            raise ValueError('Recipe predicate must compare that exact field ne the typed published_value.')
    elif status in ('conflict', 'unknown'):
        allowed = {'field', 'checkpoint', 'status', 'reason'} | ({'source_indices'} if status == 'conflict' else set())
        if set(recipe) != allowed or entry['violation_when']:
            raise ValueError('Unresolved recipe fields require a reason, no violation predicate or invented value.')
        text(recipe['reason'], 'recipe unresolved reason')
        if status == 'conflict':
            indices = recipe['source_indices']
            if (type(indices) is not list or not 2 <= len(indices) <= 8
                    or any(type(i) is not int or not 0 <= i < len(entry['evidence']) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError('Conflict requires 2..8 unique pinned source indices.')
            if any(entry['evidence'][i].get('basis') != 'publisher_recipe' for i in indices):
                raise ValueError('Conflict source_indices must identify publisher_recipe evidence.')
    else:
        raise ValueError('Recipe status must be published, conflict or unknown.')


def validate_entry(entry):
    bounded_value(entry)
    if type(entry) is not dict or not ENTRY_FIELDS <= set(entry) or set(entry) - ENTRY_FIELDS - OPTIONAL_ENTRY_FIELDS or type(entry['schema_version']) is not int or entry['schema_version'] != 1:
        raise ValueError('Entry must contain exactly the v1 schema fields.')
    if not re.fullmatch(r'[A-Z][A-Z0-9-]{2,95}', text(entry['id'], 'id', 96)):
        raise ValueError('Invalid entry ID.')
    if entry['kind'] not in ('version_trap', 'recipe', 'contract') or entry['severity'] not in ('ERROR', 'WARNING'):
        raise ValueError('Unsupported kind or severity.')
    if entry['claim_basis'] not in ('release_metadata', 'source_bytes'):
        raise ValueError('claim_basis must distinguish metadata from source bytes.')
    for field in ('title', 'first_fix', 'unknown_when'):
        text(entry[field], field)
    _strings(entry['stacks'], 'stacks')
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,99}', stack) for stack in entry['stacks']):
        raise ValueError('Stack eligibility identifiers must be canonical lowercase tokens.')
    _strings(entry['healthy_controls'], 'healthy_controls', 2)
    if 'launch_target' in entry:
        _strings(entry['launch_target'], 'launch_target', 1, 8)
        if len(set(entry['launch_target'])) != len(entry['launch_target']):
            raise ValueError('launch_target selectors must be unique.')
        for selector in entry['launch_target']:
            kind, separator, value = selector.partition(':')
            if not separator or kind not in ('module', 'script', 'entrypoint') or not value:
                raise ValueError('launch_target requires module:, script: or entrypoint: selectors.')
            if kind == 'module' and not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', value):
                raise ValueError('Invalid module selector.')
            if kind == 'entrypoint' and not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
                raise ValueError('Invalid entrypoint selector.')
            if kind == 'script':
                relative_path(value)
    for field in ('applies_when', 'violation_when'):
        values = entry[field]
        minimum = 0 if field == 'violation_when' and entry['kind'] == 'recipe' else 1
        if type(values) is not list or not minimum <= len(values) <= MAX_PREDICATES:
            raise ValueError(field + ' requires 1..24 predicates.')
        for predicate in values:
            validate_predicate(predicate)
            if field == 'violation_when' and 'source_contains' in predicate:
                raise ValueError('source_contains is an applicability hint only.')
    for field in ('applies_when', 'violation_when'):
        for predicate in entry[field]:
            body = next(iter(predicate.values()))
            if type(body) is dict and 'default_when_absent' in body:
                default = body['default_when_absent']
                rows = [row for p in entry['applies_when'] if 'source_files' in p for row in p['source_files']['files']]
                if not any(row['path'] == default['source']['path'] and default['sha256'] in row.get('sha256_in', [row.get('sha256')]) for row in rows):
                    raise ValueError('Default requires its exact source hash in a source_files applicability profile.')
                if 'launch_flag' in predicate and 'launch_target' not in entry:
                    raise ValueError('Launch defaults require an explicit applicable launch_target.')
    if entry['claim_basis'] == 'source_bytes' and not any(set(p) & {'source_files', 'source_contains'} for p in entry['applies_when']):
        raise ValueError('Source behavior claims require source_files or a weak source_contains applicability hint.')
    if type(entry['evidence']) is not list or not 1 <= len(entry['evidence']) <= 8:
        raise ValueError('Entry requires bounded pinned public evidence.')
    for row in entry['evidence']:
        if type(row) is not dict or set(row) not in ({'url', 'quote', 'source'}, {'url', 'quote', 'source', 'basis'}):
            raise ValueError('Evidence requires URL, quote and pinned source.')
        text(row['quote'], 'quote', 2048)
        if urlsplit(text(row['url'], 'URL')).scheme != 'https' or not urlsplit(row['url']).netloc:
            raise ValueError('Evidence URL must be HTTPS.')
        if row.get('basis', 'user_report') not in ('user_report', 'maintainer_statement', 'publisher_recipe'):
            raise ValueError('Unsupported evidence basis.')
        validate_source(row['source'])
    if entry['kind'] == 'recipe':
        if any('basis' not in row for row in entry['evidence']):
            raise ValueError('Recipe evidence basis must be explicit.')
        _recipe(entry)
    elif 'recipe_field' in entry or not any(row.get('basis', 'user_report') == 'user_report' for row in entry['evidence']):
        raise ValueError('Version traps/contracts require a user report; publisher-only evidence and recipe_field are recipe-only.')
    if type(entry['last_verified']) is not str:
        raise ValueError('last_verified must be an ISO date.')
    date.fromisoformat(entry['last_verified'])
    validation = entry['validation']
    if type(validation) is not dict or set(validation) != {'status', 'record'} or validation['status'] != 'confirmed':
        raise ValueError('Only independently confirmed entries may be loaded.')
    text(validation['record'], 'validation record', 512)
    return entry
