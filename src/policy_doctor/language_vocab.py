"""Bounded lexical instruction/catalog checks; no policy capability or token-path claim."""
from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

from . import hub
from .artifacts import InputError, _pairs, finite_tree
from .fs_guard import open_regular

MAX_TASKS = 2048
MAX_CATALOG_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_TASK_BYTES = 2048
MAX_INSTRUCTION_BYTES = 4096
MAX_INSTRUCTIONS = 64
MAX_COMMAND_BYTES = 16384
MAX_SIMILARITY_CHARS = 256
LEXICAL_PROFILE = 'lexical-fold-v1'
SCOPE = ('Lexical membership in a linked, revision-scoped task catalog only. Catalog labels do not prove '
         'historical training coverage, native token-path equivalence, instruction following, physical '
         'capability or success. No model, tokenizer, checkpoint code or launch command was executed; '
         'no instruction was automatically rewritten.')
SUPPORTED_POLICIES = {'smolvla', 'pi0', 'pi05', 'pi0_fast', 'act', 'diffusion', 'vqbet', 'tdmpc'}
REFERENCE = {'repository': 'https://github.com/huggingface/lerobot',
             'revision': '72e1d9e5e76ee719bacfc78bc34d817bee5fb521',
             'locations': ['src/lerobot/rollout/planner.py:283-302',
                           'src/lerobot/scripts/lerobot_eval.py:298-304'],
             'reports': ['https://github.com/huggingface/lerobot/issues/4548',
                         'https://github.com/huggingface/lerobot/issues/4045']}
_WORDS = re.compile(r'[^\W\d_]+', re.UNICODE)
_VERBS = {'pick', 'place', 'put', 'open', 'close', 'turn', 'push', 'pull', 'move', 'stack', 'insert',
          'take', 'lift', 'grasp', 'press', 'pour', 'wipe'}
_STOP = {'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'up', 'and', 'then', 'after', 'that', 'with',
         'it', 'from', 'into', 'please', 'robot'}


class _Unknown(ValueError):
    pass


def _text(value, limit, label, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise _Unknown(f'{label} must be a nonempty string')
    if len(value) > limit or len(value.encode('utf-8')) > limit or '\x00' in value:
        raise _Unknown(f'{label} exceeds its UTF-8 byte bound or contains NUL')
    return value


def _preview(value, limit=240):
    return ''.join(c if c.isprintable() else ' ' for c in value)[:limit]


def lexical_key(text):
    """A documented text-only heuristic, never a native processor equivalence."""
    return ' '.join(text.lower().replace('_', ' ').split()).rstrip('.,!?;:').rstrip()


def _words(text):
    split = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return set(_WORDS.findall(split.lower())) - _STOP


def _compound(text):
    return (bool(re.search(r'\b(?:then|after that|before)\b', text.lower()))
            or len(_words(text) & _VERBS) >= 2)


def _raw_local(path, root, limit, role):
    path, root = Path(path), Path(root).resolve()
    if not path.is_relative_to(root):
        raise _Unknown(f'{role} escapes the selected input directory')
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise _Unknown(f'{role} contains a symlink; supply an explicit regular-file copy')
    with open_regular(path, role) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise _Unknown(f'{role} exceeds the {limit}-byte read bound')
    return raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw), 'role': role}


def _json(raw):
    return hub._json(raw)


def _remote_files(source, repo_type, revision, names, limit):
    repo = hub._repo(source)
    revision = hub._revision(revision)
    plural = 'models' if repo_type == 'model' else 'datasets'
    api_url = f'https://huggingface.co/api/{plural}/{repo}/revision/{quote(revision, safe="")}?blobs=true'
    api_raw = hub._request(api_url, limit=hub.MAX_API_BYTES)
    api = _json(api_raw)
    sha = api.get('sha')
    if (not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{40}', sha)
            or (re.fullmatch('[0-9a-f]{40}', revision) and revision != sha)):
        raise _Unknown('Catalog/model API did not resolve the requested immutable revision')
    if (api.get('id') != repo or api.get('private') is not False or api.get('gated') not in (None, False)):
        raise _Unknown('Only the explicitly selected public ungated repository is supported')
    entries = api.get('siblings')
    if not isinstance(entries, list) or len(entries) > 50000:
        raise _Unknown('Repository listing exceeds its bounded complete-listing scope')
    listing = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('rfilename'), str):
            raise _Unknown('Malformed repository file listing')
        if entry['rfilename'] in listing:
            raise _Unknown('Duplicate repository file path')
        listing[entry['rfilename']] = entry
    prefix = 'datasets/' if repo_type == 'dataset' else ''
    files, artifacts, consumed = {}, [], 0
    for name in names:
        if name not in listing:
            continue
        entry = listing[name]
        size = entry.get('size')
        if size is not None and (type(size) is not int or not 0 <= size <= limit - consumed):
            raise _Unknown('Selected catalog/model metadata exceeds its byte bound')
        url = f'https://huggingface.co/{prefix}{repo}/resolve/{sha}/{name}'
        lfs = entry.get('lfs')
        task_catalog = repo_type == 'dataset' and name in ('meta/tasks.parquet', 'meta/tasks.jsonl')
        if task_catalog and (type(size) is not int or not 0 < size <= limit - consumed
                             or not isinstance(entry.get('blobId'), str)
                             or not re.fullmatch('[0-9a-f]{40}', entry['blobId'])):
            raise _Unknown('Task catalog requires a bounded pinned API size and Git blob identity')
        catalog_lfs = task_catalog and lfs is not None
        if catalog_lfs:
            from .language_catalog_transport import fetch_catalog

            if (not isinstance(lfs, dict) or type(size) is not int or not 0 < size <= limit - consumed
                    or type(lfs.get('size')) is not int or lfs['size'] != size
                    or not isinstance(lfs.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', lfs['sha256'])
                    or not isinstance(entry.get('blobId'), str) or not re.fullmatch('[0-9a-f]{40}', entry['blobId'])):
                raise _Unknown('LFS task catalog requires matching pinned API sizes, SHA256 and Git blob identity')
            raw = fetch_catalog(repo, sha, name, size=size, sha256=lfs['sha256'])
        else:
            raw = hub._request(url, limit=limit - consumed, required_sha=sha)
        if len(raw) > limit - consumed:
            raise _Unknown('Selected metadata exceeds its aggregate byte bound')
        if size is not None and len(raw) != size:
            raise _Unknown('Catalog/model metadata length differs from its pinned listing')
        if isinstance(lfs, dict) and isinstance(lfs.get('sha256'), str):
            if hashlib.sha256(raw).hexdigest() != lfs['sha256']:
                raise _Unknown('Catalog bytes disagree with the pinned LFS hash')
        elif isinstance(entry.get('blobId'), str):
            if hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != entry['blobId']:
                raise _Unknown('Catalog bytes disagree with the pinned Git blob')
        files[name] = raw
        consumed += len(raw)
        artifacts.append({'path': f'{source}@{sha}/{name}', 'sha256': hashlib.sha256(raw).hexdigest(),
                          'size_bytes': len(raw), 'role': 'language_metadata', 'url': url})
    return files, {'repo_id': repo, 'requested_revision': revision, 'resolved_revision': sha,
                   'api_sha256': hashlib.sha256(api_raw).hexdigest(), 'source': 'explicit_public_hub'}, artifacts


def _catalog_tasks(raw, suffix):
    if suffix == '.jsonl':
        lines = raw.decode('utf-8').splitlines()
        if len(lines) > MAX_TASKS:
            raise _Unknown('Task catalog row budget exceeded')
        tasks, indices = [], []
        for line in lines:
            if not line.strip():
                raise _Unknown('Empty row in the task catalog')
            item = _json(line.encode())
            tasks.append(_text(item.get('task'), MAX_TASK_BYTES, 'Catalog task'))
            indices.append(item.get('task_index'))
        layout = 'v2_jsonl_task_column'
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if len(raw) < 12 or raw[:4] != b'PAR1' or raw[-4:] != b'PAR1':
            raise _Unknown('Unsupported task Parquet header/footer')
        if not 0 < int.from_bytes(raw[-8:-4], 'little') <= 1024 * 1024:
            raise _Unknown('Task Parquet footer budget exceeded')
        from .native_parquet_guard import UnsupportedParquetMetadata, require_unencrypted_parquet

        try:
            require_unencrypted_parquet(io.BytesIO(raw), max_footer=1024 * 1024)
        except UnsupportedParquetMetadata as exc:
            raise _Unknown(str(exc)) from exc
        pf = pq.ParquetFile(io.BytesIO(raw), thrift_string_size_limit=MAX_CATALOG_BYTES,
                            thrift_container_size_limit=65536)
        meta = pf.metadata
        if not (0 <= meta.num_rows <= MAX_TASKS and 0 <= meta.num_row_groups <= 64 and 0 <= meta.num_columns <= 8):
            raise _Unknown('Task Parquet row/group/column budget exceeded')
        total = expanded = 0
        for g in range(meta.num_row_groups):
            group = meta.row_group(g)
            if not 0 <= group.num_rows <= MAX_TASKS:
                raise _Unknown('Task Parquet row-group allocation budget exceeded')
            total += group.num_rows
            for c in range(group.num_columns):
                column = group.column(c)
                if not 0 <= column.num_values <= MAX_TASKS or column.total_uncompressed_size < 0:
                    raise _Unknown('Task Parquet value/allocation budget exceeded')
                expanded += column.total_uncompressed_size
        if total != meta.num_rows or total > MAX_TASKS or expanded > MAX_CATALOG_BYTES:
            raise _Unknown('Task Parquet row counts or expanded bytes exceed the catalog budget')
        names = pf.schema_arrow.names
        if 'task' in names and not (pa.types.is_string(pf.schema_arrow.field('task').type)
                                    or pa.types.is_large_string(pf.schema_arrow.field('task').type)):
            raise _Unknown('Present task column is not string typed; no alternative label set is certified')
        text_column = next((name for name in ('task', '__index_level_0__') if name in names
                            and (pa.types.is_string(pf.schema_arrow.field(name).type)
                                 or pa.types.is_large_string(pf.schema_arrow.field(name).type))), None)
        if text_column is None:
            pandas = (pf.schema_arrow.metadata or {}).get(b'pandas')
            if pandas:
                details = _json(pandas)
                index_names = details.get('index_columns', [])
                text_column = next((name for name in index_names if isinstance(name, str) and name in names
                                    and pa.types.is_string(pf.schema_arrow.field(name).type)), None)
        if text_column is None:
            raise _Unknown('Task text is neither a string task column nor a string pandas index')
        wanted = [text_column]
        if 'task_index' in names:
            if not pa.types.is_integer(pf.schema_arrow.field('task_index').type):
                raise _Unknown('Task indices are not integer typed')
            wanted.append('task_index')
        table = pf.read(columns=wanted)
        tasks = [_text(x, MAX_TASK_BYTES, 'Catalog task') for x in table[text_column].to_pylist()]
        indices = table['task_index'].to_pylist() if 'task_index' in wanted else list(range(len(tasks)))
        layout = 'v3_task_column' if text_column == 'task' else 'v3_pandas_string_index'
    if not tasks:
        raise _Unknown('Task catalog is empty')
    if (any(type(i) is not int or not 0 <= i < MAX_TASKS for i in indices)
            or len(set(indices)) != len(indices) or set(indices) != set(range(len(indices)))):
        raise _Unknown('Task catalog identities do not form a unique bounded index set')
    return list(dict.fromkeys(tasks)), layout


def extract_vocabulary(checkpoint, *, catalog_directory=None, fetch=False, revision='main'):
    """Resolve only declared public/local metadata; never follow saved dataset.root hints."""
    artifacts = []
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint):
        raise _Unknown('Supply an explicit LeRobot checkpoint directory or hf:// model ID')
    source = str(checkpoint)
    if source.startswith('hf://'):
        if fetch is not True:
            raise _Unknown('Checkpoint metadata is remote; public fetch was not explicitly enabled')
        files, model, read = _remote_files(source, 'model', revision, ('config.json', 'train_config.json'), 2 * MAX_JSON_BYTES)
        artifacts.extend(read)
    else:
        root = Path(checkpoint).expanduser().resolve()
        if root.is_file() and root.name == 'config.json':
            root = root.parent
        files = {}
        for name in ('config.json', 'train_config.json'):
            raw, artifact = _raw_local(root / name, root, MAX_JSON_BYTES, 'language_checkpoint_metadata')
            files[name] = raw
            artifacts.append(artifact)
        model = {'source': 'explicit_local_checkpoint', 'historical_checkpoint_revision': 'unverified'}
    if 'config.json' not in files or 'train_config.json' not in files:
        raise _Unknown('Checkpoint has no config.json or saved train_config.json')
    config, training = _json(files['config.json']), _json(files['train_config.json'])
    if config.get('type') not in SUPPORTED_POLICIES:
        raise _Unknown('This v1 catalog check covers supported LeRobot checkpoints only; this policy family is unresolved')
    dataset = training.get('dataset')
    if not isinstance(dataset, dict):
        raise _Unknown('Saved training metadata does not identify one dataset')
    repo = _text(dataset.get('repo_id'), 256, 'Saved dataset repo_id')
    hub._repo('hf://' + repo)
    saved_revision = dataset.get('revision')
    if saved_revision is not None:
        hub._revision(_text(saved_revision, 128, 'Saved dataset revision'))
    episodes = dataset.get('episodes')
    if episodes is not None and (not isinstance(episodes, list) or len(episodes) > 100000
                                 or any(type(i) is not int or not 0 <= i < 10000000 for i in episodes)):
        raise _Unknown('Saved episode subset is malformed or exceeds its metadata bound')
    if catalog_directory is not None:
        root = Path(_text(catalog_directory, 4096, 'Catalog directory')).expanduser().resolve()
        files, consumed = {}, 0
        for name in ('meta/tasks.parquet', 'meta/tasks.jsonl'):
            path = root / name
            if path.exists() or path.is_symlink():
                raw, artifact = _raw_local(path, root, MAX_CATALOG_BYTES - consumed, 'language_task_catalog')
                consumed += len(raw)
                files[name] = raw
                artifacts.append(artifact)
        catalog = {'repo_id': repo, 'source': 'explicit_local_catalog', 'resolved_revision': None,
                   'association': 'Caller supplied this catalog for the saved dataset; repository/revision not authenticated'}
    else:
        if fetch is not True:
            raise _Unknown('Linked dataset catalog is unresolved offline; supply an explicit local catalog directory or allow public intake')
        files, catalog, read = _remote_files('hf://' + repo, 'dataset', saved_revision or 'main',
                                              ('meta/tasks.parquet', 'meta/tasks.jsonl'), MAX_CATALOG_BYTES)
        artifacts.extend(read)
    name = next((name for name in ('meta/tasks.parquet', 'meta/tasks.jsonl') if name in files), None)
    if name is None:
        raise _Unknown('Linked dataset has no supported meta/tasks.parquet or meta/tasks.jsonl')
    if len(files) != 1:
        raise _Unknown('Both v2 and v3 task catalogs are present; the selected training catalog is ambiguous')
    tasks, layout = _catalog_tasks(files[name], Path(name).suffix)
    catalog.update(saved_revision=saved_revision, layout=layout, distinct_tasks=len(tasks),
                   historical_training_coverage='unknown', episode_subset_selected=episodes is not None,
                   note=('Saved dataset revision is null; the catalog was read at current revision '
                         + str(catalog['resolved_revision']) + ', which may differ from training.'
                         if saved_revision is None and catalog['resolved_revision'] else
                         'Saved dataset revision is null and the explicit local catalog revision is unverified; '
                         'exact historical training coverage is not authenticated.' if saved_revision is None else
                         'Catalog bytes are scoped to the recorded source; exact historical training coverage is not authenticated.'))
    if episodes is not None:
        catalog['subset_note'] = 'This is the whole linked catalog, not a verified catalog of the saved episode subset.'
    if catalog.get('source') == 'explicit_local_catalog' and saved_revision is not None:
        catalog['note'] = ('The explicit local catalog is not authenticated to saved dataset revision '
                           + saved_revision + '; historical training coverage remains UNKNOWN.')
    return {'tasks': tasks, 'catalog': catalog, 'model': model, 'artifacts': artifacts,
            'native_token_path_equivalence': 'unknown', 'scope': SCOPE}


def instruction_inputs(instruction=None, *, launch_command=None, instruction_file=None):
    supplied = []
    if instruction is not None:
        supplied.append([_text(instruction, MAX_INSTRUCTION_BYTES, 'Instruction')])
    if launch_command is not None:
        command = _text(launch_command, MAX_COMMAND_BYTES, 'Launch command')
        if '\n' in command or '\r' in command:
            raise _Unknown('Multiline shell commands require an explicit resolved instruction')
        lex = shlex.shlex(command, posix=True, punctuation_chars=';&|')
        lex.whitespace_split = True
        lex.commenters = ''
        tokens = list(lex)
        if any(t and set(t) <= set(';&|') or '$' in t or '`' in t for t in tokens):
            raise _Unknown('Instruction from a compound or expanded shell command is unresolved; provide it explicitly')
        found = []
        for i, token in enumerate(tokens):
            key, equal, value = token.partition('=')
            if key in ('--task', '--dataset.single_task', '--env.task_description'):
                if not equal:
                    if i + 1 >= len(tokens) or tokens[i + 1].startswith('--'):
                        raise _Unknown('Instruction flag has no literal value')
                    value = tokens[i + 1]
                found.append(_text(value, MAX_INSTRUCTION_BYTES, 'Command instruction'))
        if not found:
            raise _Unknown('No literal instruction flag; --env.task is a task selector, not the resolved instruction text')
        if len(set(found)) != 1:
            raise _Unknown('Command contains conflicting instruction values')
        supplied.append([found[0]])
    if instruction_file is not None:
        path = Path(_text(instruction_file, 4096, 'Instruction file')).expanduser().resolve()
        raw, _ = _raw_local(path, path.parent, MAX_INSTRUCTIONS * MAX_INSTRUCTION_BYTES, 'language_instruction_file')
        if path.suffix == '.py':
            import ast

            tree = ast.parse(raw.decode('utf-8'))
            if sum(1 for _ in ast.walk(tree)) > 10000:
                raise _Unknown('Static eval instruction file exceeds its syntax-node bound')
            candidates = []
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    continue  # an optional module docstring, never executed
                if not isinstance(node, ast.Assign):
                    raise _Unknown('Static Python input must contain only one literal instruction assignment')
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
                        'TASK_INSTRUCTIONS', 'INSTRUCTIONS', 'task_instructions') for t in node.targets):
                    if (not isinstance(node.value, (ast.List, ast.Tuple))
                            or any(not isinstance(x, ast.Constant) or not isinstance(x.value, str) for x in node.value.elts)):
                        raise _Unknown('Eval instructions are computed; only a literal instruction list is supported')
                    candidates.append([x.value for x in node.value.elts])
                else:
                    raise _Unknown('Static Python input has an unrelated or computed statement')
            if len(candidates) != 1:
                raise _Unknown('Static eval input needs one literal TASK_INSTRUCTIONS or INSTRUCTIONS list')
            values = candidates[0]
        else:
            parsed = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
            finite_tree(parsed)
            values = parsed.get('instructions') if isinstance(parsed, dict) else parsed
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_INSTRUCTIONS:
            raise _Unknown('Explicit eval instructions require a bounded nonempty list')
        supplied.append([_text(x, MAX_INSTRUCTION_BYTES, 'Eval instruction') for x in values])
    if not supplied:
        raise _Unknown('No explicit instruction, literal command instruction or static eval instruction list')
    if any(value != supplied[0] for value in supplied[1:]):
        raise _Unknown('Explicit instruction sources disagree; the effective instruction is unresolved')
    return supplied[0]


def evaluate_vocabulary(tasks, instructions):
    """Frozen text-only predicate, independent of checkpoint extraction or execution."""
    if (not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS
            or not isinstance(instructions, list) or not 1 <= len(instructions) <= MAX_INSTRUCTIONS):
        raise _Unknown('Catalog/instruction count exceeds its bound or is empty')
    tasks = list(dict.fromkeys(_text(t, MAX_TASK_BYTES, 'Catalog task') for t in tasks))
    instructions = [_text(t, MAX_INSTRUCTION_BYTES, 'Instruction') for t in instructions]
    exact, folded = set(tasks), {lexical_key(t) for t in tasks}
    prepared = [(t, _words(t)) for t in tasks]
    catalog_words = set().union(*(words for _, words in prepared))
    single_pattern = all(not _compound(t) for t in tasks)
    result = []
    for value in instructions:
        key = lexical_key(value)
        if value in exact or key in folded:
            result.append({'verdict': 'IN_CATALOG', 'match': 'exact' if value in exact else 'lexical',
                           'instruction': _preview(value), 'instruction_sha256': hashlib.sha256(value.encode()).hexdigest(),
                           'native_token_path_equivalence': 'unknown'})
            continue
        words = _words(value)
        ranked = sorted(prepared, key=lambda row: (-len(words & row[1]) / max(1, len(words | row[1])), row[0]))[:8]
        scored = [(SequenceMatcher(None, key[:MAX_SIMILARITY_CHARS], lexical_key(t)[:MAX_SIMILARITY_CHARS],
                                   autojunk=False).ratio(), t) for t, _ in ranked]
        similarity, nearest = max(scored, key=lambda row: (row[0], row[1]))
        missing = sorted(words - catalog_words)
        compound = _compound(value) and single_pattern
        result.append({'verdict': 'OUT_OF_CATALOG', 'match': 'none', 'instruction': _preview(value),
                       'instruction_sha256': hashlib.sha256(value.encode()).hexdigest(),
                       'nearest_task': _preview(nearest), 'nearest_task_sha256': hashlib.sha256(nearest.encode()).hexdigest(),
                       'similarity': round(similarity, 4), 'similarity_scope': 'Lexical character ratio on first 256 chars after token-overlap shortlist; not semantic similarity',
                       'absent_catalog_words': missing[:16], 'absent_catalog_words_omitted': max(0, len(missing) - 16),
                       'compound_marker': bool(compound), 'native_token_path_equivalence': 'unknown'})
    return result


def check_language_vocabulary(checkpoint, *, instruction=None, launch_command=None, instruction_file=None,
                              catalog_directory=None, fetch=False, revision='main'):
    report = {'schema_version': 1, 'kind': 'language_vocabulary', 'status': 'unknown', 'verdict': 'UNKNOWN',
              'findings': [], 'evaluations': [], 'scope': SCOPE, 'lexical_profile': LEXICAL_PROFILE,
              'native_token_path_equivalence': 'unknown', 'physical_capability': 'unassessed',
              'upstream_reference': REFERENCE}
    try:
        if type(fetch) is not bool:
            raise _Unknown('fetch must be a strict Boolean')
        instructions = instruction_inputs(instruction, launch_command=launch_command, instruction_file=instruction_file)
        extracted = extract_vocabulary(checkpoint, catalog_directory=catalog_directory, fetch=fetch, revision=revision)
        evaluations = evaluate_vocabulary(extracted['tasks'], instructions)
        report.update(catalog=extracted['catalog'], artifacts=extracted['artifacts'], evaluations=evaluations,
                      instructions_checked=len(evaluations))
        bad = [item for item in evaluations if item['verdict'] == 'OUT_OF_CATALOG']
        # Catalog membership never establishes an effective native pipeline.
        report.update(status='warning' if bad else 'unknown', verdict='OUT_OF_CATALOG' if bad else 'IN_CATALOG')
        if len(extracted['tasks']) <= 2:
            report['catalog']['few_task_note'] = ('The linked catalog contains only '
                + str(len(extracted['tasks'])) + ' distinct task string(s). This does not establish whether '
                'changing instructions changes behavior.')
        for item in bad:
            notes = 'No exact or lexical-fold match in the linked catalog.'
            if item['absent_catalog_words']:
                notes += ' Words absent from its strings: ' + ', '.join(item['absent_catalog_words']) + '.'
            if item['compound_marker']:
                notes += ' A sequence/multiple-action-word marker is absent from every catalog string.'
            notes += ' ' + extracted['catalog']['note']
            report['findings'].append({'code': 'language.out_of_catalog', 'status': 'warning',
                'message': f'Instruction {_preview(item["instruction"], 120)!r} is OUT-OF-CATALOG for '
                           + str(extracted['catalog']['repo_id']) + '@'
                           + str(extracted['catalog']['resolved_revision'] or 'unverified-local-snapshot') + '. '
                           f'Nearest lexical task: {item["nearest_task"]!r} ({item["similarity"]:.3f}). ' + notes,
                'remedy': 'Use the exact catalog string for the intended task after confirming it describes that '
                          'task; never substitute the nearest lexical suggestion automatically.',
                'evidence': [{'path': a['path'], 'sha256': a['sha256'], 'pointer': 'linked task catalog'}
                             for a in extracted['artifacts'] if a['role'] == 'language_task_catalog'
                             or 'tasks.' in a['path']]})
        lexical = [item for item in evaluations if item['match'] == 'lexical']
        if lexical:
            report['findings'].append({'code': 'language.lexical_equivalence_unknown', 'status': 'unknown',
                'message': f'{len(lexical)} instruction(s) have a lexical-fold catalog match only. Lowercase, '
                           'underscore and punctuation folding do not establish the native processor token path.',
                'remedy': 'Preserve the intended task. Compare the exact catalog string and effective native processor '
                          'before treating a text variant as equivalent.', 'evidence': []})
        catalog = extracted['catalog']
        if (catalog.get('source') == 'explicit_local_catalog' or catalog.get('episode_subset_selected')
                or not isinstance(catalog.get('saved_revision'), str)
                or not re.fullmatch('[0-9a-f]{40}', catalog['saved_revision'])):
            report['findings'].append({'code': 'language.catalog_provenance_unknown', 'status': 'unknown',
                'message': catalog['note'],
                'remedy': 'Retain the catalog revision and original training provenance; an exact current-catalog '
                          'match does not authenticate historical training coverage.', 'evidence': []})
    except (InputError, _Unknown, OSError, ValueError, TypeError, KeyError, OverflowError, MemoryError,
            UnicodeError, RecursionError, ImportError, SyntaxError, RuntimeError, NotImplementedError) as exc:
        report['findings'] = [{'code': 'language.catalog_unresolved', 'status': 'unknown',
            'message': 'Instruction/catalog comparison is UNKNOWN: ' + _preview(str(exc), 360) + '.',
            'remedy': 'Provide an explicit instruction and supported saved training metadata with an authorized, '
                      'bounded task catalog. This does not certify instruction following.', 'evidence': []}]
    return report


def language_text(language):
    """Add a scoped instruction observation without replacing another finding."""
    lines = ['Instruction catalog: ' + language.get('verdict', 'UNKNOWN') + '; lexical membership only.']
    finding = next((row for row in language.get('findings', []) if row.get('status') == 'warning'), None)
    if finding:
        lines.extend([finding['message'], 'Instruction next: ' + finding['remedy']])
    lines.append('Native token equivalence, historical coverage and physical capability remain unassessed.')
    return '\n'.join(lines)


def augment_report(report, language):
    """Add observations without replacing or demoting existing independent findings."""
    result = dict(report)
    result['findings'] = [*report.get('findings', []), *language.get('findings', [])]
    result['language_vocabulary'] = language
    statuses = {report.get('status', 'unknown'), language['status']}
    result['status'] = next(s for s in ('error', 'unknown', 'warning', 'pass') if s in statuses)
    if type(report.get('total_findings')) is int:
        result['total_findings'] = report['total_findings'] + len(language['findings'])
    if isinstance(report.get('counts'), dict):
        result['counts'] = dict(report['counts'])
        for status in ('error', 'warning', 'unknown', 'pass'):
            old = result['counts'].get(status)
            if type(old) is not int:
                old = sum(f.get('status') == status for f in report.get('findings', []))
            result['counts'][status] = old + sum(f['status'] == status for f in language['findings'])
        if isinstance(report['counts'].get('findings'), dict):
            result['counts']['findings'] = dict(report['counts']['findings'])
            for status in ('error', 'warning', 'unknown', 'pass'):
                result['counts']['findings'][status] = (report['counts']['findings'].get(status, 0)
                    + sum(f['status'] == status for f in language['findings']))
    return result
