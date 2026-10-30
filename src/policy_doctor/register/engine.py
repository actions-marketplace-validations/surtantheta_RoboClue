"""Evaluate curated literals against bounded, explicitly selected local evidence.

Nothing imported from an inspected framework; no Git execution, network, shell
execution, callable predicates, YAML constructors or recursive directory scans.
Metadata declares a version; it never attests the installed implementation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import itertools
import json
import math
import re
import shlex
import stat
import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from ..framework_versions import _inspect_distribution, _normal_name, _regular_bytes
from .commands import select_command
from .schema import MAX_BYTES, MAX_ENTRIES, bounded_value, same_json, text, validate_entry, version

INPUT_FIELDS = {'schema_version', 'kind', 'metadata_paths', 'lockfiles', 'collect_installed',
                'config', 'processor', 'launch_command', 'checkpoint_id', 'runtime_source_root', 'stacks', 'command_directory'}
MISSING = object()
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_PACKAGES = 64
MAX_LOCK_BYTES = 2 * 1024 * 1024
MAX_LOCK_NODES = 100000
MAX_LOCK_DEPTH = 24
MAX_VERSION_OBSERVATIONS = 64
SCOPE = ('Curated register predicates on selected local evidence only. Package metadata and lockfiles '
         'declare versions; they do not authenticate installed source or establish task performance.')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are ambiguous.')
        result[key] = value
    return result


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number.')))
    return bounded_value(value)


def _read(path, limit=MAX_BYTES):
    raw = _regular_bytes(path, limit)
    if raw is None:
        raise ValueError('Selected file is missing.')
    return raw


def _expanded(value):
    candidate = Path(value)
    if candidate.parts and candidate.parts[0].startswith('~') and candidate.parts[0] != '~':
        raise ValueError('Other-user home aliases are unsupported; select a direct authorized path.')
    try:
        return candidate.expanduser()
    except RuntimeError as exc:
        raise ValueError('The selected home directory could not be resolved.') from exc


def _path(value, base, roots=None, directory=False):
    text(value, 'path', 4096)
    candidate = _expanded(value)
    candidate = candidate if candidate.is_absolute() else base / candidate
    candidate = candidate.absolute()
    if '..' in candidate.parts:
        raise ValueError('Parent traversal is refused.')
    # Resolve no links: all components, including parents, must be real paths.
    for part in (*reversed(candidate.parents), candidate):
        mode = part.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError('Symlink inputs are refused.')
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError('Only regular files/directories are supported.')
    if roots is not None and not any(candidate.is_relative_to(root) for root in roots):
        raise ValueError('Selected input is outside the authorized read roots.')
    if directory != candidate.is_dir():
        raise ValueError('Selected input has the wrong file/directory type.')
    return candidate


def is_register_input(source):
    """Bounded role detection; never follow nested paths here."""
    try:
        path = Path(source)
        return path.is_file() and path.suffix == '.json' and _json(_read(path)).get('kind') == 'provenance_register_input'
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        return False


def load_entries(directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).with_name('entries')
    entries, errors, ids = [], [], set()
    try:
        # Stop before materializing an arbitrarily large directory.
        paths = list(itertools.islice(directory.iterdir(), MAX_ENTRIES + 1))
        if len(paths) > MAX_ENTRIES:
            raise ValueError('Register directory exceeds 128 entries/files.')
        for path in sorted(paths):
            if path.suffix not in ('.json', '.yaml'):
                continue
            try:
                entry = validate_entry(_json(_read(path)))
                if entry['id'] in ids:
                    raise ValueError('Repeated entry ID.')
                ids.add(entry['id'])
                entries.append(entry)
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                errors.append({'file': path.name, 'reason': str(exc)[:240]})
    except (OSError, ValueError) as exc:
        errors.append({'file': str(directory), 'reason': str(exc)[:240]})
    # Duplicate IDs cannot leave one arbitrarily selected rule active.
    if any(e['reason'] == 'Repeated entry ID.' for e in errors):
        entries = []
    return entries, errors


def _manifest(source, read_roots):
    roots = [_expanded(root).absolute() for root in read_roots] if read_roots is not None else None
    path = _path(str(source), Path.cwd(), roots)
    raw = _read(path)
    doc = _json(raw)
    if type(doc) is not dict or set(doc) - INPUT_FIELDS or type(doc.get('schema_version')) is not int or doc['schema_version'] != 1 or doc.get('kind') != 'provenance_register_input':
        raise ValueError('Expected a v1 provenance_register_input object with supported fields only.')
    if type(doc.get('collect_installed', False)) is not bool:
        raise ValueError('collect_installed must be a strict boolean.')
    if 'stacks' in doc:
        stacks = doc['stacks']
        if type(stacks) is not list or len(stacks) > 16:
            raise ValueError('stacks must be an explicitly complete list of at most 16 selected stacks.')
        for stack in stacks:
            text(stack, 'selected stack', 100)
            if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,99}', stack):
                raise ValueError('Selected stacks require canonical lowercase identifiers, not display names.')
        if len(set(stacks)) != len(stacks):
            raise ValueError('Selected stacks must be unique.')
    for key in ('metadata_paths', 'lockfiles'):
        if type(doc.get(key, [])) is not list or len(doc.get(key, [])) > 16:
            raise ValueError(key + ' must be a list of at most 16 explicit paths.')
        for item in doc.get(key, []):
            text(item, key, 4096)
    for key in ('config', 'processor', 'runtime_source_root', 'launch_command', 'checkpoint_id', 'command_directory'):
        if key in doc:
            text(doc[key], key, 8192 if key == 'launch_command' else 4096)
    return path, doc, roots, hashlib.sha256(raw).hexdigest()


def _document(value, base, roots):
    if value is None:
        return {'reason': 'No document selected.'}
    try:
        path = _path(value, base, roots)
        raw = _read(path)
        document = _json(raw)
        if type(document) is not dict:
            raise ValueError('Config/processor must be a JSON object.')
        return {'document': document, 'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        return {'reason': str(exc)[:240]}


def _lock_document(value):
    """Bound wheel-rich lock metadata separately from small rule/config JSON."""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_LOCK_NODES or depth > MAX_LOCK_DEPTH:
            raise ValueError('Lockfile exceeds its node/depth bounds.')
        if type(item) is dict:
            if any(type(key) is not str or len(key) > 4096 for key in item):
                raise ValueError('Lockfile keys exceed supported bounds.')
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > 16384:
                raise ValueError('Lockfile text exceeds supported bounds.')
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Non-finite lockfile values are unsupported.')
        elif type(item) is int:
            if item.bit_length() > 128:
                raise ValueError('Lockfile integers exceed supported bounds.')
        elif item is not None and type(item) is not bool:
            raise ValueError('Lockfile contains unsupported metadata types.')
    return value


def _requirement_lines(content):
    pending = ''
    for number, line in enumerate(content.splitlines(), 1):
        if number > 4096:
            raise ValueError('Requirements text exceeds 4096 lines.')
        line = line.strip()
        pending += line[:-1] + ' ' if line.endswith('\\') else line
        if len(pending) > 16384:
            raise ValueError('Requirements logical line exceeds 16384 characters.')
        if line.endswith('\\'):
            continue
        yield number, pending
        pending = ''
    if pending:
        raise ValueError('Unfinished requirements continuation is unresolved.')


def _versions_from_lock(path, raw):
    """Read declared exact pins; never resolve an environment, index or URL."""
    if len(raw) > MAX_LOCK_BYTES:
        raise ValueError('Lockfile exceeds the 2 MiB read bound.')
    content = raw.decode('utf-8-sig')
    if path.name in ('uv.lock', 'poetry.lock'):
        data = _lock_document(tomllib.loads(content))
        rows = data.get('package')
        if type(rows) is not list or len(rows) > 2048:
            raise ValueError('Lockfile needs a bounded package list.')
        for row in rows:
            if type(row) is not dict or type(row.get('name')) is not str:
                raise ValueError('Malformed lockfile package.')
            source = row.get('source')
            unresolved = any(key in row for key in ('marker', 'markers', 'resolution-markers'))
            # Local/VCS/direct sources and conditional package alternatives do
            # not establish a selected released distribution. No paths/URLs
            # are followed and no current-process marker is evaluated.
            if source is not None and (type(source) is not dict or set(source) != {'registry'}
                                       or type(source['registry']) is not str):
                unresolved = True
            yield row['name'], None if unresolved else row.get('version'), 'package.version'
    else:
        for number, line in _requirement_lines(content):
            if not line or line.startswith('#'):
                continue
            line = re.sub(r'\s+#.*$', '', line)
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            option, separator, attached = tokens[0].partition('=')
            if option in ('-i', '--index-url', '--extra-index-url', '--find-links', '-f'):
                if not ((separator and attached and len(tokens) == 1) or (not separator and len(tokens) == 2 and tokens[1])):
                    raise ValueError('Malformed requirements index directive.')
                continue  # Resolution location does not change the declared pin.
            if tokens in (['--no-index'], ['--require-hashes']):
                continue
            if tokens[0].startswith('-'):
                raise ValueError('Requirements include/editable/directive cannot be resolved statically.')
            # Hashes attest no downloaded artifact here; retain the whole input
            # identity and read only the explicitly declared package version.
            requirement_text = re.split(r'\s+--hash(?:=|\s+)', line, maxsplit=1)[0]
            suffix = line[len(requirement_text):].strip()
            hashes_valid = not suffix or re.fullmatch(r'(?:--hash(?:=|\s+)sha256:[0-9a-fA-F]{64}\s*)+', suffix)
            try:
                requirement = Requirement(requirement_text.split(' #', 1)[0].strip())
            except InvalidRequirement as exc:
                name = re.match(r'([A-Za-z0-9][A-Za-z0-9_.-]*)', line)
                if name is None:
                    raise ValueError('Requirements line has no unambiguous package name.') from exc
                yield name[1], None, 'line:' + str(number) + ' unresolved requirement'
                continue
            pins = list(requirement.specifier)
            release = (pins[0].version if len(pins) == 1 and pins[0].operator == '=='
                       and '*' not in pins[0].version and requirement.marker is None
                       and requirement.url is None and hashes_valid else None)
            yield requirement.name, release, 'line:' + str(number) + ('' if release is not None else ' unresolved constraint/marker/source')


def _collect(doc, base, roots, packages):
    observations = {p: [] for p in packages}
    errors = []

    def record(package, value, location, **extra):
        name = _normal_name(package)
        if name in observations:
            if len(observations[name]) >= MAX_VERSION_OBSERVATIONS:
                raise ValueError('Too many observations for one selected package; selection remains unresolved.')
            observations[name].append({'version': value, 'location': location, **extra})

    for value in doc.get('metadata_paths', []):
        try:
            path = _path(value, base, roots, directory=True)
            # Read METADATA once to determine the name through the bounded parser.
            result = _inspect_distribution(metadata.PathDistribution(path), tuple(packages))
            if result.get('distribution') is None or result.get('version') is None or result.get('source_identity') == 'unknown':
                raise ValueError(result['reason'])
            record(result['distribution'], result['version'], str(path), metadata_sha256=result.get('metadata_sha256'),
                   declared_commit=result.get('declared_commit'), source_identity='not_attested')
        except (OSError, ValueError, TypeError) as exc:
            errors.append({'input': value, 'reason': str(exc)[:240]})
    if doc.get('collect_installed', False):
        for package in sorted(packages):
            try:
                distributions = list(itertools.islice(metadata.distributions(name=package), 17))
                if len(distributions) > 16:
                    raise ValueError('Too many matching installed distributions.')
                for distribution in distributions:
                    result = _inspect_distribution(distribution, (package,))
                    if result.get('version') is None or result.get('source_identity') == 'unknown':
                        raise ValueError(result['reason'])
                    record(package, result['version'], 'RoboClue process installed metadata',
                           metadata_sha256=result.get('metadata_sha256'), source_identity='not_attested')
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                errors.append({'input': package, 'reason': str(exc)[:240]})
    for value in doc.get('lockfiles', []):
        try:
            path = _path(value, base, roots)
            raw = _read(path, MAX_LOCK_BYTES)
            rows = list(_versions_from_lock(path, raw))
            digest = hashlib.sha256(raw).hexdigest()
            for package, release, selector in rows:
                record(package, release, str(path) + ':' + selector, sha256=digest,
                       source_identity='not_attested')
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            errors.append({'input': value, 'reason': str(exc)[:240]})
    return observations, errors


def _same(a, b):
    # Python's True == 1 must not create an evidenced match.
    return same_json(a, b)


def _observed(value):
    # Selected documents already have byte/node/depth bounds. The detailed
    # witness retains the actual JSON type/value; concise views project it.
    return value


def _source_read(path, context):
    if context['source_reads'] >= 256 or context['source_bytes'] >= 64 * 1024 * 1024:
        raise ValueError('Register source-read budget exhausted (256 files or 64 MiB).')
    context['source_reads'] += 1
    raw = _read(path, min(MAX_SOURCE_BYTES, 64 * 1024 * 1024 - context['source_bytes']))
    context['source_bytes'] += len(raw)
    return raw


def _compare(observed, body, *, absent=False):
    op = body['op']
    if op == 'exists':
        if observed is MISSING and not absent:
            return None
        return (observed is not MISSING) is body.get('value', True)
    if observed is MISSING:
        return None
    expected = body['value']
    if op in ('lt', 'lte', 'gt', 'gte'):
        if type(observed) not in (int, float) or type(expected) not in (int, float):
            return None
        return {'lt': observed < expected, 'lte': observed <= expected,
                'gt': observed > expected, 'gte': observed >= expected}[op]
    if op == 'in':
        scalar = (str, int, float, bool, type(None))
        if not any(type(observed) is type(item) for item in expected) and not (type(observed) in scalar and any(type(item) in scalar for item in expected)):
            return None
        return any(_same(observed, item) for item in expected)
    if type(observed) is not type(expected) and not (type(observed) in (str, int, float, bool, type(None)) and type(expected) in (str, int, float, bool, type(None))):
        return None
    same = _same(observed, expected)
    return same if op == 'eq' else not same


def _default(body, context, witness):
    default = body.get('default_when_absent')
    if default is None or context.get('stack_state') is not True:
        return MISSING
    try:
        root = context['doc'].get('runtime_source_root')
        if root is None:
            return MISSING
        root = _path(root, context['base'], context['roots'], directory=True)
        if 'flag' in body:
            if not context['command'].get('source_binding_safe', False):
                witness['reason'] = 'Launch preparation/wrapper cwd or environment is not source-bound; export the actual resolved config instead.'
                return MISSING
            selector = context['command'].get('selector', '')
            directory = context['doc'].get('command_directory')
            if not selector.startswith('script:') or directory is None:
                witness['reason'] = 'Launch default requires a declared command_directory and a source-bound selected script.'
                return MISSING
            directory = _path(directory, context['base'], context['roots'], directory=True)
            script = _path(selector.removeprefix('script:'), directory, context['roots'])
            if not script.is_relative_to(root):
                witness['reason'] = 'Selected script is outside the declared runtime source root.'
                return MISSING
            relative = script.relative_to(root).as_posix()
            profiles = [row for p in context['entry']['applies_when'] if 'source_files' in p
                        for row in p['source_files']['files'] if row['path'] == relative]
            digest = hashlib.sha256(_source_read(script, context)).hexdigest()
            if not profiles or not any(digest in row.get('sha256_in', [row.get('sha256')]) for row in profiles):
                witness['reason'] = 'Selected script bytes are not part of the independently verified applicability profile.'
                return MISSING
            witness['launch_source'] = {'file': str(script), 'sha256': digest}
        path = _path(default['source']['path'], root, context['roots'])
        raw = _source_read(path, context)
        digest = hashlib.sha256(raw).hexdigest()
        witness['default_source'] = {**default['source'], 'file': str(path), 'sha256': digest}
        if digest != default['sha256']:
            witness['reason'] = 'Default source bytes do not match the independently cited profile.'
            return MISSING
        witness['value_origin'] = 'source_backed_default_for_absent_key'
        return default['value']
    except (OSError, ValueError, TypeError) as exc:
        witness['reason'] = str(exc)[:240]
        return MISSING


def _selected_value(evidence, body, kind, witness):
    document = evidence.get('document', MISSING)
    parts = body['path'].split('.')
    selected = document
    if kind == 'processor_key' and type(document) is dict and type(document.get('steps')) is list:
        steps = document['steps']
        if len(steps) > 256 or any(type(step) is not dict for step in steps):
            witness['reason'] = 'Processor registry table is malformed or exceeds 256 steps.'
            return MISSING, False
        matches = [(i, step) for i, step in enumerate(steps) if step.get('registry_name') == parts[0]]
        if len(matches) > 1 or matches and parts[0] in document:
            witness['reason'] = 'Processor registry selector is ambiguous; duplicate instances are UNKNOWN.'
            return MISSING, False
        if len(matches) == 1:
            index, selected = matches[0]
            witness['processor_registry_index'] = index
            parts = parts[1:]
            if not parts:
                return selected, False
        elif parts[0] not in document:
            witness['reason'] = 'Processor registry instance is missing.'
            return MISSING, False
    for part in parts[:-1]:
        selected = selected.get(part, MISSING) if type(selected) is dict else MISSING
    if type(selected) is not dict:
        return MISSING, False
    return selected.get(parts[-1], MISSING), parts[-1] not in selected


def _predicate(predicate, context):
    kind, body = next(iter(predicate.items()))
    witness = {'predicate': predicate, 'result': 'unknown'}
    result = None
    if kind == 'package_version':
        observations = context['versions'].get(_normal_name(body['package']), [])
        witness['observations'] = observations
        if not context['version_errors'] and observations:
            try:
                values = {version(row['version']) for row in observations}
                if len(values) == 1:
                    observed = next(iter(values))
                    comparisons = {'eq': lambda bound: observed == bound, 'gt': lambda bound: observed > bound,
                                   'gte': lambda bound: observed >= bound, 'lt': lambda bound: observed < bound,
                                   'lte': lambda bound: observed <= bound}
                    result = all(comparisons[op](version(value)) for op, value in body['version'].items())
            except (ValueError, TypeError):
                pass
        witness['source_identity'] = 'not_attested'
    elif kind == 'checkpoint_id_in':
        observed = context['doc'].get('checkpoint_id', MISSING)
        if observed is not MISSING:
            witness['observed'] = _observed(observed)
            witness['location'] = 'input.checkpoint_id (explicit declaration)'
            # A trailing path delimiter does not identify a different artifact.
            # No cache paths, case folding or URI aliases are inferred. An
            # unrecognized identity cannot certify an unaffected checkpoint.
            result = True if observed.rstrip('/') in {value.rstrip('/') for value in body} else None
            if result is None:
                witness['reason'] = 'Checkpoint identity is not one of the explicitly recognized identities.'
    elif kind in ('config_key', 'processor_key'):
        evidence = context['config' if kind == 'config_key' else 'processor']
        witness.update({k: v for k, v in evidence.items() if k != 'document'})
        witness['selector'] = body['path']
        observed, absent = _selected_value(evidence, body, kind, witness)
        if absent and body['op'] != 'exists':
            observed = _default(body, context, witness)
        if observed is not MISSING:
            witness['observed'] = _observed(observed)
        if body['op'] == 'exists' and (absent or observed is not MISSING):
            witness['observed_presence'] = not absent
        result = _compare(observed, body, absent=absent)
    elif kind == 'launch_flag':
        flags = context['flags']
        values = flags.get(body['flag'], []) if flags is not None else []
        witness['location'] = 'input.launch_command'
        if body['op'] == 'exists' and flags is not None:
            witness['observed_presence'] = bool(values)
            result = bool(values) is body.get('value', True)
        elif len(values) == 1:
            witness['observed'] = values[0]
            result = _compare(values[0], body)
        elif not values and flags is not None:
            if context['command']['state'] == 'selected' and context.get('entry_has_launch_target'):
                observed = _default(body, context, witness)
                if observed is not MISSING:
                    witness['observed'] = _observed(observed)
                result = _compare(observed, body)
    elif kind == 'source_files':
        root = context['doc'].get('runtime_source_root')
        checked = []
        witness['files'] = checked
        if root is not None:
            try:
                root = _path(root, context['base'], context['roots'], directory=True)
                total = 0
                for row in body['files']:
                    path = _path(row['path'], root, context['roots'])
                    raw = _source_read(path, context)
                    total += len(raw)
                    if total > 8 * MAX_SOURCE_BYTES:
                        raise ValueError('Source profile exceeds 16 MiB total bound.')
                    digest = hashlib.sha256(raw).hexdigest()
                    expected = row.get('sha256_in', [row.get('sha256')])
                    checked.append({'file': str(path), 'sha256': digest, 'expected_sha256_in': expected})
                    if digest not in expected:
                        raise ValueError('Source bytes differ from the curated profile; applicability is UNKNOWN.')
                result = True
            except (OSError, ValueError, TypeError) as exc:
                witness['reason'] = str(exc)[:240]
    elif kind == 'source_contains':
        witness['evidence_strength'] = 'weak_literal_hint'
        root = context['doc'].get('runtime_source_root')
        if root is not None:
            try:
                root = _path(root, context['base'], context['roots'], directory=True)
                path = _path(body['path'], root, context['roots'])
                raw = _source_read(path, context)
                count = raw.count(body['literal'].encode('utf-8'))
                witness.update(file=str(path), sha256=hashlib.sha256(raw).hexdigest(), observed_occurrences=count)
                op, expected = next(iter(body.get('occurrences', {'gte': 1}).items()))
                result = count == expected if op == 'eq' else count >= expected
            except (OSError, ValueError, TypeError) as exc:
                witness['reason'] = str(exc)[:240]
    witness['result'] = 'true' if result is True else 'false' if result is False else 'unknown'
    return result, witness


def _all(predicates, context):
    pairs = [_predicate(p, context) for p in predicates]
    # The intake contract requires every applicability input to be resolvable;
    # an exclusion does not silently erase a different missing observation.
    result = None if any(v is None for v, _ in pairs) else False if any(v is False for v, _ in pairs) else True
    return result, [w for _, w in pairs]


def _entry_gate(entry, context):
    doc = context['doc']
    stack_state = None if 'stacks' not in doc else bool(set(doc['stacks']) & set(entry['stacks']))
    context['stack_state'] = stack_state
    context['entry'] = entry
    context['entry_has_launch_target'] = 'launch_target' in entry
    command = select_command(doc.get('launch_command'), entry.get('launch_target'))
    context['command'], context['flags'] = command, command['flags']
    if stack_state is False:
        return False
    if stack_state is None:
        return None
    if 'launch_target' in entry:
        if command['state'] == 'absent':
            return False
        if command['state'] != 'selected':
            return None
    return True


def _recipe_output(entry, state, violation, actual_checkpoint=None):
    recipe = entry['recipe_field']
    field, checkpoint = recipe['field'], actual_checkpoint or recipe['checkpoint']
    if recipe['status'] != 'published':
        return recipe['reason'], ''
    published = json.dumps(recipe['published_value'], ensure_ascii=True, sort_keys=True)
    fix = f'set {field} to {published} to reproduce the published setup'
    if state == 'warning':
        source = entry['evidence'][recipe['source_index']]['source']
        location = f"{Path(source['path']).name}:{source['line']}"
        actual = violation[0].get('observed', 'UNKNOWN')
        actual = json.dumps(actual, ensure_ascii=True, sort_keys=True)
        return (f'{field} = {actual} differs from the published evaluation recipe for {checkpoint} '
                f'({published}, {location}). Results are not comparable to the published numbers.'), fix
    if state == 'unknown':
        return (f'The actual {field} or selected stack for {checkpoint} is UNKNOWN; no recipe difference was established.',
                f'Supply the actual {field} and complete selected stacks for {checkpoint}.')
    return f'Published evaluation recipe field {field} for {checkpoint}.', fix


def assess_register(source, *, read_roots=None, entries_directory=None,
                    launch_command=None, runtime_source_root=None):
    """Return a structured report for all malformed, absent and conflicting evidence."""
    report = {'schema_version': 1, 'kind': 'provenance_register', 'rule_id': 'RC012',
              'source': str(source), 'status': 'unknown', 'scope': SCOPE, 'findings': [], 'entries': []}
    try:
        path, doc, roots, input_digest = _manifest(source, read_roots)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        report['findings'].append({'code': 'register.input_unknown', 'status': 'unknown',
                                   'message': str(exc)[:240], 'first_fix': 'Supply a bounded v1 provenance_register_input JSON file.'})
        return report
    entries, errors = load_entries(entries_directory)
    report['catalog_errors'] = errors
    conflicts = []
    for key, supplied in (('launch_command', launch_command), ('runtime_source_root', runtime_source_root)):
        if supplied is None:
            continue
        try:
            text(supplied, key, 8192 if key == 'launch_command' else 4096)
        except (ValueError, TypeError) as exc:
            conflicts.append({'field': key, 'reason': str(exc)})
            doc.pop(key, None)
            continue
        if key in doc and doc[key] != supplied:
            conflicts.append({'field': key, 'reason': 'Captured and explicitly supplied observations disagree.'})
            doc.pop(key)
        else:
            doc[key] = supplied
    if conflicts:
        report['observation_conflicts'] = conflicts
        report['findings'].append({'code': 'register.input_unknown', 'status': 'unknown',
            'message': 'Conflicting launch/source observations remain UNKNOWN; no captured value was overridden.',
            'first_fix': 'Supply one consistent actual launch command and source selection, preserving the captured evidence.'})
    packages = {_normal_name(p['package_version']['package']) for e in entries
                for p in e['applies_when'] + e['violation_when'] if 'package_version' in p}
    if len(packages) > MAX_PACKAGES:
        report['findings'].append({'code': 'register.catalog_unknown', 'status': 'unknown',
            'message': 'Catalogue exceeds the 64-package evaluation bound.',
            'first_fix': 'Select a bounded independently validated register catalogue.'})
        return report
    versions, version_errors = _collect(doc, path.parent, roots, packages)
    context = {'doc': doc, 'base': path.parent, 'roots': roots, 'versions': versions,
               'source_reads': 0, 'source_bytes': 0,
               'version_errors': version_errors, 'config': _document(doc.get('config'), path.parent, roots),
               'processor': _document(doc.get('processor'), path.parent, roots)}
    report['input_sha256'] = input_digest
    report['version_errors'] = version_errors
    for entry in entries:
        gate = _entry_gate(entry, context)
        unresolved_recipe = entry['kind'] == 'recipe' and entry['recipe_field']['status'] != 'published'
        if gate is False or unresolved_recipe:
            applies, applicability, violation, violation_evidence = False, [], False, []
        else:
            applies, applicability = _all(entry['applies_when'], context)
            violation, violation_evidence = _all(entry['violation_when'], context)
            if gate is None:
                applies = None
        state = ('not_applicable' if applies is False else 'unknown' if applies is None or violation is None
                 else entry['severity'].lower() if violation else 'consistent')
        # Text in comments, strings and dead branches cannot prove behavior.
        # Even a confirmed literal-only source rule is at most a WARNING.
        weak_source = (any('source_contains' in p for p in entry['applies_when'])
                       or entry['claim_basis'] == 'source_bytes'
                       and not any('source_files' in p for p in entry['applies_when']))
        if weak_source and state == 'error':
            state = 'warning'
        if entry['kind'] == 'recipe':
            if state == 'error':
                state = 'warning'
            if unresolved_recipe and gate is not False:
                state = entry['recipe_field']['status']
        row = {'id': entry['id'], 'status': state, 'title': entry['title'], 'claim_basis': entry['claim_basis'],
               'applicability': applicability, 'violation': violation_evidence, 'evidence': entry['evidence'],
               'first_fix': entry['first_fix'], 'validation': entry['validation']}
        if weak_source:
            row['source_identity'] = 'weak_literal_hint_only'
        message, fix = entry['title'], entry['first_fix']
        if state == 'unknown':
            fix = 'Supply the missing or conflicting applicability and violation evidence. ' + entry['unknown_when']
            row['first_fix'] = fix
        if entry['kind'] == 'recipe':
            message, fix = _recipe_output(entry, state, violation_evidence, doc.get('checkpoint_id'))
            row.update(title=message, first_fix=fix, recipe_field=entry['recipe_field'])
        row['scope_gate'] = {'stack_state': context['stack_state'], 'command': {k: v for k, v in context['command'].items() if k != 'flags'}}
        report['entries'].append(row)
        if state in ('error', 'warning', 'unknown') and not unresolved_recipe:
            if state == 'unknown' and entry['kind'] != 'recipe':
                message = entry['unknown_when']
            sources = entry['evidence']
            if entry['kind'] == 'recipe':
                index = entry['recipe_field']['source_index']
                sources = [sources[index], *(source for i, source in enumerate(sources) if i != index)]
            decisive = [w for w in violation_evidence if 'observed' in w or 'observed_presence' in w]
            other_violation = [w for w in violation_evidence if 'observed' not in w and 'observed_presence' not in w]
            evidence = [{**w, 'phase': 'violation'} for w in decisive + other_violation]
            evidence += [{**w, 'phase': 'applicability'} for w in applicability]
            report['findings'].append({'code': 'register.' + entry['id'], 'status': state,
                'message': message, 'evidence': evidence,
                'source_evidence': sources, 'first_fix': fix})
    if errors or not entries:
        report['findings'].append({'code': 'register.catalog_unknown', 'status': 'unknown',
            'message': 'No complete validated register catalogue is available.' if not entries else 'Some catalogue entries could not be validated.',
            'first_fix': 'Use independently validated v1 register entries; do not infer missing facts.'})
    states = {row['status'] for row in report['findings']}
    if any(row['status'] in ('unknown', 'conflict') for row in report['entries']):
        states.add('unknown')
    if report['entries'] and all(row['status'] == 'not_applicable' for row in report['entries']):
        states.add('unknown')
        report['assessment_state'] = 'no_applicable_entries'
    report['status'] = next((s for s in ('error', 'warning', 'unknown') if s in states), 'pass')
    report['message'] = ('Register evaluation is scoped to the loaded entries; no compatibility or source-attestation PASS is implied.')
    return report


def register_text(report):
    """Complete bounded recipe wording and fix, with a short displayed citation."""
    findings = sorted(report.get('findings', []), key=lambda row: {'error': 0, 'warning': 1, 'unknown': 2}.get(row['status'], 3))
    lines = ['RoboClue provenance register: ' + report['status'].upper() + '.']
    if findings:
        finding = findings[0]
        lines.append(finding['message'])
        decisive = next((row for row in finding.get('evidence', []) if 'observed' in row), None)
        if decisive:
            location = (Path(decisive['file']).name + '#' + decisive.get('selector', '') if decisive.get('file')
                        else decisive.get('location', 'selected input'))
            lines.append('Evidence: ' + location + '=' + json.dumps(decisive['observed'], ensure_ascii=True) + '.')
        citation = next(iter(finding.get('source_evidence', [])), {}).get('source')
        if citation:
            lines.append('Source: ' + Path(citation['path']).name + ':' + str(citation['line']) + '; full pinned citation in structured evidence.')
        lines.append('Next: ' + finding['first_fix'])
        others = _actionable_others(findings)
        named = _named_others(others)
        for row in named:
            message = row.get('message', '')
            if message.endswith('Results are not comparable to the published numbers.'):
                message = _RECIPE_CAVEAT
            elif len(message) > 300:
                message = message[:240] + ' [...] complete text in the detailed report'
            lines.append('Also ' + row['status'].upper() + ' ' + row['code'] + ': ' + message)
        hidden = len(others) - len(named) + report.get('other_actionable_omitted', 0)
        if hidden:
            lines.append(f'{hidden} more ERROR/WARNING finding(s) are in the detailed report.')
    else:
        lines.append('No contradiction in loaded entries; unresolved publication fields and unchecked inputs remain unassessed.')
    lines.append('Scope: ' + report['scope'])
    return '\n'.join(lines)


def _json_identity(value):
    digest, size = hashlib.sha256(), 0
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
        raw = chunk.encode()
        size += len(raw)
        digest.update(raw)
    return digest.hexdigest(), size


def _brief_value(value, prefix, result):
    """Keep small actual JSON values; a large preview is never an actual value."""
    labels = {str: 'string', bool: 'boolean', int: 'integer', float: 'number',
              type(None): 'null', list: 'array', dict: 'object'}
    result[prefix + '_type'] = labels[type(value)]
    raw = json.dumps(value, sort_keys=True, allow_nan=False)
    if len(raw.encode()) <= 256:
        result[prefix] = value
    else:
        result.update({prefix + '_preview': raw[:128], prefix + '_truncated': True,
                       prefix + '_size_bytes': len(raw.encode()),
                       prefix + '_sha256': hashlib.sha256(raw.encode()).hexdigest()})


def _witness_preview(witness, *, force=False):
    digest, size = _json_identity(witness)
    if size <= 700 and not force:
        return witness
    result = {key: witness[key] for key in ('result', 'phase', 'observed_presence') if key in witness}
    result.update(witness_sha256=digest, witness_size_bytes=size, context_omitted=True)
    predicate = witness.get('predicate', {})
    if len(predicate) == 1:
        kind, body = next(iter(predicate.items()))
        result['predicate_kind'] = kind
        if isinstance(body, dict) and 'op' in body:
            result['op'] = body['op']
            _brief_value(body.get('path', body.get('flag')), 'selector', result)
            if body['op'] == 'exists':
                _brief_value(body.get('value', True), 'expected', result)
            elif 'value' in body:
                _brief_value(body['value'], 'expected', result)
    if 'observed' in witness:
        _brief_value(witness['observed'], 'observed', result)
    for key in ('file', 'selector', 'location', 'sha256', 'reason'):
        if key in witness:
            _brief_value(witness[key], key, result)
    return result


def _summary_text(value, key, result, fallback, *, budget=1024):
    """Never display a clipped repair command as an executable first fix."""
    result[key] = value
    if len(json.dumps(value, allow_nan=False).encode()) > budget:
        raw = value.encode()
        result.update({key: fallback, key + '_truncated': True,
                       key + '_preview': value[:64], key + '_size_bytes': len(raw),
                       key + '_sha256': hashlib.sha256(raw).hexdigest()})


PREVIEW_LIMIT = 8192
_RECIPE_CAVEAT = ('A setting differs from the published evaluation recipe. '
                  'Results are not comparable to the published numbers.')


def _actionable_others(findings):
    return [row for row in findings[1:] if row['status'] in ('error', 'warning')]


def _named_others(others):
    """Up to three other actionable findings; a WARNING always keeps one slot.

    Many independent errors must not crowd out a recipe "not comparable" caveat.
    """
    named = others[:3]
    warning = next((row for row in others if row['status'] == 'warning'), None)
    if warning is not None and warning not in named:
        named = others[:2] + [warning]
    return named


def concise_register_summary(report):
    """Default MCP preview: the first actionable finding with witnesses, plus other actionable codes.

    Room for naming up to three other errors/warnings is reserved before the
    first finding is bounded, so an independent ERROR cannot hide a recipe
    WARNING and the whole preview stays within PREVIEW_LIMIT encoded bytes.
    """
    findings = sorted(report['findings'], key=lambda row: {'error': 0, 'warning': 1, 'unknown': 2}.get(row['status'], 3))
    others = _actionable_others(findings)
    reserve = 0
    if others:
        reserve = len(json.dumps({'findings': [_minimal_row(row) for row in _named_others(others)],
                                  'other_actionable_omitted': len(others), 'findings_omitted': len(findings)},
                                 allow_nan=False).encode())
    result = _primary_preview(report, PREVIEW_LIMIT - reserve)
    _append_other_actionable(result, findings)
    return result


def _primary_preview(report, limit):
    """The first actionable finding with witnesses, bounded to ``limit`` encoded bytes."""
    result = {key: report[key] for key in ('schema_version', 'kind', 'rule_id', 'status', 'scope', 'input_sha256') if key in report}
    result['source'] = str(report.get('source', ''))[:240]
    findings = sorted(report['findings'], key=lambda row: {'error': 0, 'warning': 1, 'unknown': 2}.get(row['status'], 3))
    result['finding_counts'] = {state: sum(row['status'] == state for row in findings) for state in ('error', 'warning', 'unknown')}
    result['entries_checked'] = len(report['entries'])
    result['full_report_sha256'], result['full_report_size_bytes'] = _json_identity(report)
    if 'assessment_state' in report:
        result['assessment_state'] = report['assessment_state']
    result['findings_omitted'] = max(0, len(findings) - 1)
    result['findings'] = []
    if findings:
        row = findings[0]
        summary = {key: row[key] for key in ('code', 'status')}
        message = row.get('message', '')
        fallback = 'Finding text exceeds this preview; inspect the detailed report for the complete finding.'
        if message.endswith('Results are not comparable to the published numbers.'):
            fallback = ('A setting differs from the published evaluation recipe. '
                        'Results are not comparable to the published numbers. '
                        'Read the detailed report for the complete setting and checkpoint.')
        _summary_text(message, 'message', summary, fallback)
        _summary_text(row.get('first_fix', ''), 'first_fix', summary,
                      'Retrieve the detailed report for the complete first fix before changing settings.', budget=4096)
        evidence = row.get('evidence', [])
        summary['evidence'] = []
        for witness in evidence[:4]:
            summary['evidence'].append(_witness_preview(witness))
        summary['evidence_omitted'] = max(0, len(evidence) - 4)
        sources = row.get('source_evidence', [])
        summary['source_evidence'] = [{**source, 'quote': source['quote'][:256]} for source in sources[:1]]
        summary['source_evidence_omitted'] = max(0, len(sources) - 1)
        result['findings'].append(summary)
    result['details'] = 'Repeat preflight_check with response_format=detailed, or roboclue check INPUT.json --json.'
    # Even hostile maximum-length paths and URLs cannot expand this preview.
    if len(json.dumps(result, allow_nan=False).encode()) > limit:
        for source in result['findings'][0].get('source_evidence', []):
            source.pop('quote', None)
            url = source.pop('url', None)
            if url:
                source['url_sha256'] = hashlib.sha256(url.encode()).hexdigest()
        result['findings'][0]['source_context_omitted'] = True
    for _ in range(3):
        if len(json.dumps(result, allow_nan=False).encode()) <= limit:
            break
        if len(result['findings'][0]['evidence']) <= 1:
            break
        result['findings'][0]['evidence'].pop()
        result['findings'][0]['evidence_omitted'] += 1
    if len(json.dumps(result, allow_nan=False).encode()) > limit:
        # Source context and a selector can themselves contain maximum-length
        # Unicode. Keep one decisive typed witness and the original identities,
        # rather than merely annotating an oversized default response.
        summary = result['findings'][0]
        original = findings[0]
        evidence = original.get('evidence', [])
        summary['evidence'] = [_witness_preview(evidence[0], force=True)] if evidence else []
        summary['evidence_omitted'] = max(0, len(evidence) - 1)
        sources = original.get('source_evidence', [])
        summary['source_evidence'] = []
        if sources:
            digest, size = _json_identity(sources[0])
            source = sources[0].get('source', {})
            summary['source_evidence'] = [{'source': {
                'repo': '[full repository in detailed output]',
                'path': '[full path in detailed output]',
                'sha': source.get('sha'), 'line': source.get('line')},
                'source_evidence_sha256': digest, 'source_evidence_size_bytes': size,
                'context_omitted': True}]
        summary['source_evidence_omitted'] = max(0, len(sources) - 1)
        result['source'] = str(report.get('source', ''))[:64]
        result['source_truncated'] = len(str(report.get('source', ''))) > 64
        result['summary_limit_note'] = 'Long context is explicitly omitted; complete values are bound by the full-report hash.'
        if len(json.dumps(result, allow_nan=False).encode()) > limit:
            _summary_text(original.get('first_fix', ''), 'first_fix', summary,
                          'Retrieve the detailed report for the complete first fix before changing settings.')
    return result


def _minimal_row(row):
    return {'code': row['code'], 'status': row['status'], 'detail_omitted': True}


def _append_other_actionable(result, findings):
    """Name other errors/warnings so the first one cannot hide a different action.

    An independent ERROR can co-occur with a recipe WARNING whose caveat changes
    how results may be compared. Every candidate is measured whole, counters
    included; the minimal rows always fit the room reserved for them.
    """
    others = _actionable_others(findings)
    if not others:
        return
    named = _named_others(others)
    rows = []
    for row in named:
        compact = {'code': row['code'], 'status': row['status']}
        message = row.get('message', '')
        if message.endswith('Results are not comparable to the published numbers.'):
            compact['message'] = _RECIPE_CAVEAT
        else:
            _summary_text(message, 'message', compact,
                          'Finding text exceeds this preview; inspect the detailed report.', budget=512)
        compact['detail_omitted'] = True
        rows.append(compact)
    for attempt in (rows, [_minimal_row(row) for row in named], []):
        candidate = dict(result, findings=result['findings'] + attempt,
                         findings_omitted=max(0, len(findings) - len(result['findings']) - len(attempt)))
        if len(others) > len(attempt):
            candidate['other_actionable_omitted'] = len(others) - len(attempt)
        if len(json.dumps(candidate, allow_nan=False).encode()) <= PREVIEW_LIMIT:
            result.clear()
            result.update(candidate)
            return
