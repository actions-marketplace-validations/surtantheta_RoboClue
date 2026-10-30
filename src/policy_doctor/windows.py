"""Bounded native LeRobot v3 episode ranges and integer action-window witnesses."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from .artifacts import LEROBOT_REVISION, MAX_JSON_BYTES, InputError, read_json, sha256
from .fs_guard import DatasetLayoutError, refuse_dataset_link, refuse_special_file

DATA_PATH = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
META_COLUMNS = ('episode_index', 'length', 'dataset_from_index', 'dataset_to_index', 'data/chunk_index', 'data/file_index')
IDENTITY_COLUMNS = ('index', 'episode_index', 'frame_index')
MAX_FILE_BYTES = 512 * 1024 * 1024  # real humanoid shards exceed 128 MiB; reads stay projected and budgeted
MAX_FOOTER_BYTES = 16 * 1024 * 1024  # real humanoid shards carry 3.5 MB footers (719 row groups)
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_VALUES = 1000000
CONSISTENCY_MAX_FILES = 4096  # footers plus one integer column per file; rows stay budgeted
SCOPE = ('Native LeRobot v3 numeric rows, episode membership and whole-dataset video-index/header checks, '
         'with explicit coverage. Temporal queries assess row selection only. Corrections are metadata-only '
         'what-if candidates, never applied. No clock alignment, video decoding, policy, hardware, physical '
         'repair or author-authenticity claim.')
REFERENCE = {'repository': 'https://github.com/huggingface/lerobot',
    'revision': LEROBOT_REVISION,
    'functions': ['DatasetReader._get_query_indices', 'delta_window'],
    'locations': ['src/lerobot/datasets/dataset_reader.py:298-311', 'src/lerobot/datasets/utils.py:146-162'],
    'rule': 'requested=absolute_anchor+delta; is_pad=requested<start or requested>=end; index=clip(requested,start,end-1)'}


class _Unknown(Exception):
    pass


class _FooterRowCountConflict(_Unknown):
    """A bounded parsed footer contradicts itself, before any row allocation."""

    def __init__(self, artifact, relative, file_rows, group_rows):
        total = sum(group_rows)
        message = (f'Parquet footer row-count contradiction in {relative}: file total {file_rows}, '
                   f'row-group sum {total} across {len(group_rows)} group(s)')
        super().__init__(message)
        self.artifact = artifact
        self.finding = {'code': 'episode.parquet_row_counts', 'status': 'error',
            'message': message + '. These parsed counts contradict each other; no row pages were read from this '
                       'file and no allocation based on its row-group sizes was attempted.',
            'remedy': 'Preserve the original file. Restore consistent bytes from the source or investigate its '
                      'footer in a separate copy; this finding does not authorize a rewrite or exclusions.',
            'evidence': [{'path': artifact['path'], 'sha256': artifact['sha256'],
                'pointer': 'Parquet footer /num_rows and /row_groups/*/num_rows',
                'value': {'file_num_rows': file_rows, 'row_group_rows_sum': total,
                          'row_group_rows': group_rows[:32], 'row_groups': len(group_rows)}}]}


class _ParquetEnvelopeConflict(_Unknown):
    """Observed native Parquet bytes contradict the declared container format."""

    def __init__(self, artifact, relative, reason, observed):
        message = f'Native Parquet envelope is incomplete or contradictory in {relative}: {reason}'
        super().__init__(message)
        self.artifact = artifact
        self.finding = {'code': 'episode.parquet_envelope', 'status': 'error',
            'message': message + '. No row pages were read from this file; no episode exclusions are inferred.',
            'remedy': 'Preserve the original. Restore the complete native file from its source or investigate '
                      'a separate copy, then rerun; do not discard intact episode ranges to silence this error.',
            'evidence': [{'path': artifact['path'], 'sha256': artifact['sha256'],
                          'pointer': 'Parquet file header/footer envelope', 'value': observed}]}


def _check_footer_rows(meta, artifact, relative, max_rows):
    if not 0 <= meta.num_row_groups <= 4096 or meta.num_rows < 0:
        raise _Unknown('Parquet row/group budget exceeded or unsupported negative count')
    counts = [meta.row_group(group).num_rows for group in range(meta.num_row_groups)]
    if any(count < 0 for count in counts):
        raise _Unknown('Parquet row-group row budget exceeded or unsupported negative count')
    # Compare the two bounded scalar ledgers before applying the allocation
    # cap. Equal extreme counts establish no contradiction and remain UNKNOWN.
    if sum(counts) != meta.num_rows:
        raise _FooterRowCountConflict(artifact, relative, meta.num_rows, counts)
    if meta.num_rows > max_rows or any(count > max_rows for count in counts):
        raise _Unknown('Parquet row-group row budget exceeded')


def _integer(value, *, minimum=0, maximum=2**53 - 1):
    return type(value) is int and minimum <= value <= maximum


class _Intake:
    def __init__(self, root, max_rows, max_files):
        self.root, self.max_rows, self.max_files = root, max_rows, max_files
        self.artifacts, self.entries, self.expanded, self.values = {}, 0, 0, 0
        self.data_rows = self.meta_rows = 0

    def safe(self, path):
        if not path.is_relative_to(self.root):
            raise InputError(f'Artifact {path} escapes dataset root {self.root}; pass a root-contained native file')
        current = self.root
        for part in path.relative_to(self.root).parts:
            current /= part
            if current.is_symlink():
                refuse_dataset_link(self.root, current)
            if not current.resolve().is_relative_to(self.root):
                raise InputError(f'Artifact {path} escapes dataset root {self.root}; restore the original native path')
        return path

    def redundant_directory_link(self, path, base):
        """Ignore only a directory alias to this listing or one of its in-dataset ancestors.

        A native shard listing is not recursive: these aliases add no native shards,
        and entering data/up would inspect metadata as numeric data instead.
        """
        if not path.is_symlink():
            return False
        if re.fullmatch(r'chunk-[0-9]{3,}', path.name):
            return False  # a declared native shard directory cannot be silently omitted
        try:
            target = path.resolve(strict=True)
            return (target.is_relative_to(self.root) and base.is_relative_to(target)
                    and target.is_dir())
        except (OSError, RuntimeError):
            return False  # safe() supplies the named missing/looping-link refusal

    def discover(self, base):
        self.safe(base)
        if not base.is_dir():
            raise _Unknown(f'Missing native shard directory: {base.relative_to(self.root)}')
        paths = []
        with os.scandir(base) as chunks:
            for chunk in chunks:
                self.entries += 1
                if self.entries > 8192:
                    raise _Unknown('Shard discovery entry budget exceeded')
                directory = Path(chunk.path)
                if self.redundant_directory_link(directory, base):
                    continue
                directory = self.safe(directory)
                if not chunk.is_dir(follow_symlinks=False):
                    continue
                if not re.fullmatch(r'chunk-[0-9]{3,}', chunk.name):
                    raise _Unknown('Nonstandard native shard directory is unsupported')
                with os.scandir(directory) as files:
                    for item in files:
                        self.entries += 1
                        if self.entries > 8192:
                            raise _Unknown('Shard discovery entry budget exceeded')
                        path = self.safe(Path(item.path))
                        if path.suffix != '.parquet':
                            continue
                        if not item.is_file(follow_symlinks=False) or not re.fullmatch(r'file-[0-9]{3,}\.parquet', item.name):
                            raise _Unknown('Nonstandard native Parquet filename is unsupported')
                        paths.append(path)
                        if len(paths) > self.max_files:
                            raise _Unknown('Native shard discovery file budget exceeded')
        return sorted(paths)

    def artifact(self, path, role):
        self.safe(path)
        refuse_special_file(path, 'native artifact')
        relative = path.relative_to(self.root)
        if not path.is_file():
            if relative.as_posix() == 'meta/info.json':
                raise _Unknown(f'Missing required meta/info.json at {path}; this is not a supported native LeRobot '
                               'dataset root. Pass the dataset directory containing meta/info.json, or use '
                               'investigate DIRECTORY to discover supported artifacts in a run folder.')
            raise _Unknown(f'Missing required native artifact {relative}; restore the original recording file or '
                           'check the dataset directory, then rerun')
        limit = MAX_JSON_BYTES if role == 'dataset_info' else MAX_FILE_BYTES
        if path.stat().st_size > limit:
            raise _Unknown(f'Native artifact {relative} is {path.stat().st_size} bytes, exceeding the {limit}-byte '
                           'per-file read cap; its contents were not checked. Check the original file with its '
                           'intended native loader; do not truncate it to make this check pass.')
        if path not in self.artifacts:
            if len(self.artifacts) > self.max_files:
                raise _Unknown('Total native file budget exceeded')
            size = path.stat().st_size
            if sum(a['size_bytes'] for a in self.artifacts.values()) + size > 1024 * 1024 * 1024:
                raise _Unknown('Total native compressed-byte budget exceeded')
            self.artifacts[path] = {'path': str(path), 'sha256': sha256(path), 'size_bytes': size, 'role': role}
        return self.artifacts[path]

    def stable(self):
        for path, artifact in self.artifacts.items():
            self.safe(path)
            if not path.is_file() or sha256(path) != artifact['sha256']:
                raise InputError('Native artifact changed during checking; retry a stable snapshot')

    def parquet(self, path, columns, role, *, action=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        artifact = self.artifact(path, role)
        relative = str(path.relative_to(self.root))
        with path.open('rb') as stream:
            header = stream.read(4)
            if header == b'PARE':
                raise _Unknown('Encrypted Parquet footer is unsupported; supply an authorized readable native snapshot')
            if header != b'PAR1' or artifact['size_bytes'] < 12:
                raise _ParquetEnvelopeConflict(artifact, relative, 'missing native PAR1 header or minimum envelope',
                    {'header_hex': header.hex(), 'file_size_bytes': artifact['size_bytes']})
            stream.seek(-8, 2)
            tail = stream.read(8)
        footer_bytes = int.from_bytes(tail[:4], 'little')
        if tail[4:] != b'PAR1' or not 0 < footer_bytes <= artifact['size_bytes'] - 12:
            raise _ParquetEnvelopeConflict(artifact, relative, 'missing PAR1 trailer or impossible metadata extent',
                {'header_hex': header.hex(), 'trailer_hex': tail[4:].hex(),
                 'footer_size_bytes': footer_bytes, 'file_size_bytes': artifact['size_bytes']})
        if footer_bytes > MAX_FOOTER_BYTES:
            raise _Unknown('Parquet footer budget exceeded; container damage is not established')
        from .native_parquet_guard import UnsupportedParquetMetadata, require_unencrypted_parquet

        try:
            require_unencrypted_parquet(path, max_footer=MAX_FOOTER_BYTES)
        except UnsupportedParquetMetadata as exc:
            raise _Unknown(str(exc)) from exc
        pf = pq.ParquetFile(path, thrift_string_size_limit=MAX_FOOTER_BYTES,
                            thrift_container_size_limit=65536)
        meta = pf.metadata
        column_cap = 4096 if role == 'episode_metadata' else 128  # footer is bounded; only listed columns are read
        if not 0 <= meta.num_columns <= column_cap:
            raise _Unknown('Parquet row/group/column budget exceeded')
        # Arrow allocates from row-group counts, even if FileMetaData.num_rows
        # claims a tiny file. Validate both ledgers before any read/allocation.
        _check_footer_rows(meta, artifact, str(path.relative_to(self.root)), self.max_rows)
        if any(name not in pf.schema_arrow.names for name in columns):
            raise _Unknown('Required native Parquet columns are missing')
        if not action and any(not pa.types.is_integer(pf.schema_arrow.field(name).type) for name in columns):
            raise _Unknown('Native row identity/metadata columns must be integer typed')
        expanded = values = 0
        for group in range(meta.num_row_groups):
            rg = meta.row_group(group)
            for col in range(rg.num_columns):
                item = rg.column(col)
                if item.path_in_schema.split('.')[0] in columns:
                    expanded += item.total_uncompressed_size
                    values += item.num_values
        byte_cap = 8 * 1024 * 1024 if action else MAX_EXPANDED_BYTES
        if expanded > byte_cap or values > MAX_VALUES or self.expanded + expanded > 2 * MAX_EXPANDED_BYTES or self.values + values > 4 * MAX_VALUES:
            raise _Unknown('Parquet expanded-byte/value budget exceeded')
        self.expanded += expanded
        self.values += values
        if not action:
            if role == 'episode_metadata':
                self.meta_rows += meta.num_rows
                if self.meta_rows > min(self.max_rows, 100000):
                    raise _Unknown('Total episode metadata row budget exceeded')
            else:
                self.data_rows += meta.num_rows
                if self.data_rows > self.max_rows:
                    raise _Unknown('Total data row budget exceeded')
        return pf, artifact


def _consistency(root, max_rows, max_files, *, short_only):
    """Declared totals and per-episode lengths against the rows actually present.

    Independent of the strict window engine, which ends unknown on any irregularity: this pass reads
    only integer episode identity columns, so wrong declared totals or spans are reported as the
    contradictions they are. ``short_only`` keeps just the too-short warning when the strict engine
    has already verified identity, totals and spans. Returns (findings, intake).
    """
    import pyarrow as pa
    intake = _Intake(root, max_rows, CONSISTENCY_MAX_FILES)
    info_path = intake.safe(root / 'meta' / 'info.json')
    info_artifact = intake.artifact(info_path, 'dataset_info')
    info = read_json(info_path)
    # Leftover .parquet files under another declared backend are not the current
    # dataset; never diagnose their framing or counts as this dataset's errors.
    if info.get('storage_format', 'parquet') != 'parquet':
        raise _Unknown('Only the absent/default or explicitly declared Parquet backend is supported')
    if not str(info.get('codebase_version', '')).startswith('v3.') or info.get('data_path') != DATA_PATH:
        raise _Unknown('Only native v3 metadata with the standard data_path template is supported')
    if (info.get('total_episodes') == 0 and info.get('total_frames') == 0
            and not (root / 'data').exists() and not (root / 'meta' / 'episodes').exists()):
        return [{'code': 'episode.empty_dataset', 'status': 'warning',
                 'message': 'meta/info.json declares no episodes and the repository holds no episode metadata or data '
                            'files; there is nothing to train or evaluate on.',
                 'evidence': [{'path': info_artifact['path'], 'sha256': info_artifact['sha256'], 'pointer': '/total_episodes',
                               'value': 0}]}], intake
    # Discover files the way the native loader does (every data/*/*.parquet, io_utils.py:76), so stray
    # files that LeRobot would load are counted and reported rather than refused.
    def native_files(base):
        if not base.is_dir():
            raise _Unknown(f'Missing native shard directory: {base.relative_to(root)}')
        intake.safe(base)
        found = []
        with os.scandir(base) as children:
            for child in children:
                directory = Path(child.path)
                if intake.redundant_directory_link(directory, base):
                    continue
                intake.safe(directory)
                if not child.is_dir(follow_symlinks=False):
                    continue
                with os.scandir(directory) as files:
                    for item in files:
                        if not item.name.endswith('.parquet'):
                            continue
                        path = intake.safe(Path(item.path))
                        found.append(path)
                        if len(found) > CONSISTENCY_MAX_FILES:
                            raise _Unknown('Episode consistency file budget exceeded')
        return sorted(found)
    metadata_files, data_files = native_files(root / 'meta' / 'episodes'), native_files(root / 'data')
    if not metadata_files or not data_files:
        raise _Unknown('Missing native episode metadata or data shards')
    # Same template the strict engine accepts (chunk-NNN.../file-NNN..., three or more digits).
    stray = [p for p in data_files if not re.fullmatch(r'chunk-[0-9]{3,}/file-[0-9]{3,}\.parquet', str(p.relative_to(root / 'data')))]

    def read(path, columns, role, optional=()):
        # The fallback must apply the same allocation guards as the strict
        # audit. A refused footer must never reach a second, unbudgeted read.
        pf, artifact = intake.parquet(path, columns, role)
        required = tuple(columns)
        present = required + tuple(name for name in optional if name in pf.schema_arrow.names
                                   and pa.types.is_integer(pf.schema_arrow.field(name).type))
        items = pf.read(columns=list(present)).to_pylist()
        # Only required identity columns must be valid; optional ones are checked per value by their users.
        if any(not _integer(item[name]) for item in items for name in required):
            raise _Unknown('Episode identity columns contain invalid values')
        return artifact, items

    declared, declared_at, repeated, pointers, declared_spans = {}, {}, {}, {}, []
    for path in metadata_files:
        artifact, items = read(path, ('episode_index', 'length'), 'episode_metadata',
                               optional=('data/chunk_index', 'data/file_index',
                                         'dataset_from_index', 'dataset_to_index'))
        for row, item in enumerate(items):
            start, end = item.get('dataset_from_index'), item.get('dataset_to_index')
            if _integer(start, minimum=-(2 ** 53)) and _integer(end, minimum=-(2 ** 53)):
                declared_spans.append((item['episode_index'], start, end, item['length'],
                                       [dict(_locator(artifact, row, name), value=item[name])
                                        for name in ('dataset_from_index', 'dataset_to_index')]))
            if _integer(item.get('data/chunk_index')) and _integer(item.get('data/file_index')):
                pointers.setdefault(item['episode_index'], (
                    f"data/chunk-{item['data/chunk_index']:03d}/file-{item['data/file_index']:03d}.parquet",
                    _locator(artifact, row, 'data/file_index')))
            if item['episode_index'] in declared:
                repeated.setdefault(item['episode_index'], [declared_at[item['episode_index']]]).append(
                    _locator(artifact, row, 'episode_index'))
                continue
            declared[item['episode_index']] = item['length']
            declared_at[item['episode_index']] = _locator(artifact, row, 'length')
    actual, actual_at, seen_index, repeated_index, unindexed = {}, {}, {}, [], 0
    position, index_mismatches, index_examples = 0, 0, []
    for path in data_files:
        # The global index is optional here: its absence must not hide the episode and totals checks.
        artifact, items = read(path, ('episode_index',), 'dataset_parquet', optional=('index',))
        for row, item in enumerate(items):
            if type(item.get('index')) is int and item['index'] != position:
                index_mismatches += 1
                if len(index_examples) < 4:
                    index_examples.append(dict(_locator(artifact, row, 'index'),
                                               value=item['index'], expected_position=position))
            position += 1
            actual[item['episode_index']] = actual.get(item['episode_index'], 0) + 1
            actual_at.setdefault(item['episode_index'], _locator(artifact, row, 'episode_index'))
            if not _integer(item.get('index')):
                unindexed += 1
            elif item['index'] in seen_index:
                repeated_index.append((item['index'], seen_index[item['index']], _locator(artifact, row, 'index')))
            else:
                seen_index[item['index']] = _locator(artifact, row, 'index')
    findings = []
    if stray:
        findings.append({'code': 'episode.nonstandard_data_files', 'status': 'warning',
                         'files': [str(p.relative_to(root)) for p in stray[:16]],
                         'message': f'{len(stray)} data file(s) do not follow the declared data_path template '
                                    f"(e.g. {stray[0].relative_to(root)}). LeRobot {LEROBOT_REVISION[:7]} loads every "
                                    'data/*/*.parquet (src/lerobot/datasets/io_utils.py:76), so their rows are loaded too '
                                    'and are counted here.',
                         'evidence': [{'path': str(p), 'sha256': intake.artifacts[p]['sha256']} for p in stray[:4]]})
    short = sorted(ep for ep, count in actual.items() if count <= 1)
    if short:
        findings.append({'code': 'episode.too_short', 'status': 'warning', 'episodes': short[:16],
                         'message': f'{len(short)} episode(s) hold at most one frame: {short[:16]}. Each provides at most one distinct temporal sample; '
                                    'whether padding or limited statistics are acceptable depends on the consumer.',
                         'evidence': [actual_at[ep] for ep in short[:4]]})
    if short_only:
        return findings, intake
    invalid_spans = [(ep, start, end, length, evidence)
                     for ep, start, end, length, evidence in declared_spans
                     if start < 0 or end <= start or end - start != length or end > position]
    if invalid_spans:
        findings.append({'code': 'episode.invalid_span', 'status': 'error',
                         'episodes': [ep for ep, *_ in invalid_spans[:16]],
                         'message': f'{len(invalid_spans)} episode span(s) are negative, reversed, inconsistent '
                                    'with their declared length, or exceed the readable physical rows. '
                                    'Repair uncertainty does not make these spans valid.',
                         'remedy': 'Restore spans from source-verified episode runs in a separate copy; '
                                   'preserve every original file.',
                         'evidence': [e for *_, evidence in invalid_spans[:2] for e in evidence]})
    if index_mismatches:
        findings.append({'code': 'episode.row_index_mismatch', 'status': 'error',
                         'message': f'{index_mismatches} stored row index value(s) disagree with physical positions. '
                                    'This identity contradiction does not authorize excluding intact episodes.',
                         'remedy': 'Preserve the original rows. Resolve index/position disagreements against the '
                                   'source recording before deriving a rewrite or exclusion plan.',
                         'evidence': index_examples})
    present = {str(p.relative_to(root)) for p in data_files}
    dangling = sorted(ep for ep, (name, _) in pointers.items() if name not in present)
    if dangling:
        findings.append({'code': 'episode.missing_data_file', 'status': 'warning', 'episodes': dangling[:16],
                         'message': f'{len(dangling)} episode metadata row(s) point to a data file that does not exist '
                                    f'(e.g. episode {dangling[0]} -> {pointers[dangling[0]][0]}). Training reads every data file, '
                                    f'but LeRobot {LEROBOT_REVISION[:7]} resumes recording from the last episode\'s pointer '
                                    '(src/lerobot/datasets/dataset_writer.py:454-460) and uses these pointers to merge datasets '
                                    '(src/lerobot/datasets/aggregate.py:221-257).',
                         'evidence': [pointers[ep][1] for ep in dangling[:4]]})
    if unindexed:
        findings.append({'code': 'episode.global_index_unchecked', 'status': 'unknown',
                         'message': f'{unindexed} data row(s) have no integer index column value, so duplicate global '
                                    'indices were not checked for them. Episode spans and declared totals were still checked.'})
    if repeated_index:
        findings.append({'code': 'episode.duplicate_global_index', 'status': 'error',
                         'message': f'{len(repeated_index)} data row(s) repeat a global index already used by another row '
                                    f'(e.g. index {repeated_index[0][0]}; {len(seen_index)} distinct of '
                                    f'{len(seen_index) + len(repeated_index)} rows). LeRobot {LEROBOT_REVISION[:7]} maps absolute '
                                    'indices to rows with a dictionary (src/lerobot/datasets/dataset_reader.py:200-201), so '
                                    'repeated indices resolve to the wrong row when episodes are selected.',
                         'evidence': [item for _, first, again in repeated_index[:2] for item in (first, again)]})
    if repeated:
        episodes = sorted(repeated)
        findings.append({'code': 'episode.metadata_duplicate', 'status': 'error', 'episodes': episodes[:16],
                         'message': f'{len(episodes)} episode index(es) appear in more than one episode metadata row: '
                                    f'{episodes[:16]}. LeRobot looks episode metadata up by row position '
                                    f'({LEROBOT_REVISION[:7]} src/lerobot/datasets/dataset_reader.py:304), so repeated '
                                    'rows misalign the spans of later episodes.',
                         'evidence': [item for ep in episodes[:2] for item in repeated[ep][:2]]})
    observed = {'total_frames': sum(actual.values()), 'total_episodes': len(set(declared) | set(actual))}
    for key, value in observed.items():
        if _integer(info.get(key)) and info[key] != value:
            findings.append({'code': 'episode.declared_totals', 'status': 'error', 'field': key,
                'declared': info[key], 'observed': value,
                'message': f'meta/info.json declares {key}={info[key]} but the dataset holds {value} '
                           f"({'data rows' if key == 'total_frames' else 'distinct episodes'}). LeRobot "
                           f'{LEROBOT_REVISION[:7]} reads these totals directly: ' +
                           ('it skips episode metadata when total_episodes is 0 (src/lerobot/datasets/dataset_metadata.py:222).'
                            if key == 'total_episodes' and info[key] == 0 else
                            'the default episode selection is range(total_episodes) (src/lerobot/datasets/dataset_reader.py:264).'
                            if key == 'total_episodes' else
                            'without an explicit episode selection the frame count falls back to total_frames '
                            '(src/lerobot/datasets/dataset_reader.py:204-208).'),
                'evidence': [{'path': info_artifact['path'], 'sha256': info_artifact['sha256'], 'pointer': '/' + key,
                              'value': info[key]}]})
    for code, episodes, text, locators in (
            ('episode.metadata_without_rows', sorted(set(declared) - set(actual)),
             'episode metadata row(s) have no data rows', declared_at),
            ('episode.rows_without_metadata', sorted(set(actual) - set(declared)),
             'data episode(s) have no episode metadata row', actual_at)):
        if episodes:
            findings.append({'code': code, 'status': 'error', 'episodes': episodes[:16],
                             'message': f'{len(episodes)} {text}: {episodes[:16]}.',
                             'evidence': [locators[ep] for ep in episodes[:4]]})
    mismatch = [(ep, declared[ep], actual[ep]) for ep in sorted(set(declared) & set(actual))
                if ep not in repeated and declared[ep] != actual[ep]]
    if mismatch:
        examples = ', '.join(f'episode {ep}: {d} declared, {a} rows' for ep, d, a in mismatch[:3])
        findings.append({'code': 'episode.length_mismatch', 'status': 'error',
                         'episodes': [{'episode_index': ep, 'declared_length': d, 'actual_rows': a} for ep, d, a in mismatch[:16]],
                         'message': f'{len(mismatch)} episode(s) declare a length that differs from their actual rows '
                                    f'({examples}). A loader that trusts the declared spans reads windows across episode '
                                    'boundaries.',
                         'evidence': [item for ep, _, _ in mismatch[:2] for item in (declared_at[ep], actual_at[ep])]})
    return findings, intake


def _locator(artifact, row, column):
    return {'path': artifact['path'], 'sha256': artifact['sha256'], 'row': row, 'column': column,
            'row_identity': 'zero-based physical Parquet row within this file'}


def audit_episode_windows(dataset, *, episode_index=None, anchor_index=None, delta_indices=None,
                          max_rows=200000, max_files=1024, video_context=None, video_headers=True):
    if not _integer(max_rows, minimum=1, maximum=200000) or not _integer(max_files, minimum=1, maximum=1024):
        raise InputError('Require strict integer max_rows in [1,200000], max_files in [1,1024]')
    query_requested = any(v is not None for v in (episode_index, anchor_index, delta_indices))
    if query_requested and (not _integer(episode_index) or not _integer(anchor_index) or
            not isinstance(delta_indices, list) or not 1 <= len(delta_indices) <= 64 or
            any(not _integer(v, minimum=-1000000, maximum=1000000) for v in delta_indices)):
        raise InputError('Query requires episode_index, absolute anchor_index and 1–64 strict integer delta_indices')
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise InputError('Install policy-deployment-doctor[dataset] for native episode windows') from exc
    root = Path(dataset).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f'Expected a native dataset directory at {str(root)!r}; check the path and rerun')
    intake = _Intake(root, max_rows, max_files)
    findings, patches, summaries = [], [], []
    finding_count = patch_count = episode_count = 0
    coverage = {'identity_verified': False, 'complete': False, 'data_files_read': 0,
                'metadata_files_read': 0, 'data_rows_read': 0, 'metadata_rows_read': 0,
                'max_rows': max_rows, 'max_files': max_files}
    result = {'schema_version': 1, 'kind': 'episode_window_audit', 'status': 'unknown', 'findings': findings,
              'metadata_patches': patches, 'episodes': summaries, 'coverage': coverage, 'artifacts': [],
              'query': None, 'scope': SCOPE, 'upstream_reference': REFERENCE.copy()}
    try:
        info_path = intake.safe(root / 'meta' / 'info.json')
        intake.artifact(info_path, 'dataset_info')
        info = read_json(info_path)
        features = info.get('features')
        if isinstance(features, dict) and any(isinstance(feature, dict) and feature.get('dtype') == 'video'
                                            for feature in features.values()):
            intake.safe(root / 'videos')
        if info.get('storage_format', 'parquet') != 'parquet':
            raise _Unknown('Only the absent/default or explicitly declared Parquet backend is supported')
        if not str(info.get('codebase_version', '')).startswith('v3.') or info.get('data_path') != DATA_PATH:
            raise _Unknown('Only native v3 metadata with the standard data_path template is supported')
        total, total_episodes = info.get('total_frames'), info.get('total_episodes')
        if not _integer(total, minimum=1) or not _integer(total_episodes, minimum=1):
            raise _Unknown('Missing or invalid native total_frames/total_episodes; completeness is unknown')
        metadata_files = intake.discover(root / 'meta' / 'episodes')
        data_files = intake.discover(root / 'data')
        if not metadata_files or not data_files or len(metadata_files) + len(data_files) > max_files:
            raise _Unknown('Missing shards or combined native shard budget exceeded')
        metadata = {}
        for path in metadata_files:
            pf, artifact = intake.parquet(path, META_COLUMNS, 'episode_metadata')
            row = 0
            for batch in pf.iter_batches(batch_size=512, columns=list(META_COLUMNS)):
                for item in batch.to_pylist():
                    if any(not _integer(item[k]) for k in META_COLUMNS):
                        raise _Unknown('Incomplete/invalid native episode metadata values')
                    ep = item['episode_index']
                    if ep in metadata:
                        raise _Unknown('Duplicate episode metadata rows are ambiguous')
                    if ep != len(metadata):
                        raise _Unknown('Physical metadata row order does not match native positional episode lookup')
                    metadata[ep] = (item, artifact, row)
                    start = root / 'data' / f"chunk-{item['data/chunk_index']:03d}" / f"file-{item['data/file_index']:03d}.parquet"
                    if intake.safe(start) not in data_files:
                        raise _Unknown('Referenced native start shard is missing')
                    row += 1
            coverage['metadata_files_read'] += 1
            coverage['metadata_rows_read'] += row
        if len(metadata) != total_episodes or set(metadata) != set(range(total_episodes)):
            raise _Unknown('Incomplete native episode metadata coverage')
        actual, rows = {}, []
        for path in data_files:
            pf, artifact = intake.parquet(path, IDENTITY_COLUMNS, 'dataset_parquet')
            row = 0
            for batch in pf.iter_batches(batch_size=1024, columns=list(IDENTITY_COLUMNS)):
                for item in batch.to_pylist():
                    if any(not _integer(item[k]) for k in IDENTITY_COLUMNS):
                        raise _Unknown('Native global/episode/frame identity contains invalid values')
                    index, ep, frame = (item[k] for k in IDENTITY_COLUMNS)
                    # Native loader uses physical lexicographic file/row order, never a sort by index.
                    if index != len(rows):
                        raise _Unknown('Duplicate/gapped/reordered native global index does not match physical row identity')
                    if ep not in metadata:
                        raise _Unknown('Data row episode has no native metadata row')
                    records = actual.setdefault(ep, [])
                    if frame != len(records) or (records and records[-1] + 1 != index):
                        raise _Unknown('Episode rows/frame indices are noncontiguous or incomplete')
                    records.append(index)
                    rows.append((ep, frame, path, row, artifact))
                    row += 1
            # Counters cover files read to the end; a file the engine stopped inside is not counted.
            coverage['data_files_read'] += 1
            coverage['data_rows_read'] += row
        if len(rows) != total or set(actual) != set(metadata):
            raise _Unknown('Actual rows do not establish complete total-frame/episode coverage')
        coverage.update(identity_verified=True, complete=True)
        episode_count = len(metadata)
        for ep, (item, artifact, row) in metadata.items():
            observed_span = {'dataset_from_index': actual[ep][0], 'dataset_to_index': actual[ep][-1] + 1,
                             'length': len(actual[ep])}
            if len(summaries) < 16:
                summaries.append({'episode_index': ep, 'declared': {k: item[k] for k in observed_span}, 'actual': observed_span})
            for column, expected in observed_span.items():
                if item[column] != expected:
                    evidence = [_locator(artifact, row, column),
                                _locator(rows[actual[ep][0]][4], rows[actual[ep][0]][3], 'index'),
                                _locator(rows[actual[ep][-1]][4], rows[actual[ep][-1]][3], 'index')]
                    finding = {'code': 'episode.range_contradiction', 'status': 'error', 'episode_index': ep,
                               'column': column, 'observed': item[column], 'expected': expected, 'evidence': evidence}
                    finding_count += 1
                    patch_count += 1
                    if len(findings) < 16:
                        findings.append(finding)
                    if len(patches) < 16:
                        patches.append(dict(evidence[0], episode_index=ep, observed=item[column], expected=expected,
                            scope='Metadata-only what-if; no write or physical repair claim'))
        if query_requested:
            if anchor_index >= len(rows) or rows[anchor_index][0] != episode_index:
                raise InputError('Absolute anchor index must identify an actual row of the selected episode')
            item = metadata[episode_index][0]
            corrected_span = (actual[episode_index][0], actual[episode_index][-1] + 1)
            declared_span = (item['dataset_from_index'], item['dataset_to_index'])
            required_indices = set()
            for start, end in (declared_span, corrected_span):
                if end > start:
                    required_indices.update(min(max(anchor_index + d, start), end - 1) for d in delta_indices)
            actions, value_reason = {}, None
            wanted = {i for i in required_indices if 0 <= i < len(rows)}
            features = info.get('features')
            action_feature = features.get('action') if isinstance(features, dict) else None
            action_shape = action_feature.get('shape') if isinstance(action_feature, dict) else None
            action_width = (action_shape[0] if isinstance(action_shape, list) and len(action_shape) == 1 and
                            _integer(action_shape[0], minimum=1, maximum=64) else None)
            if (action_width is None or action_feature.get('dtype') not in ('float16', 'float32', 'float64')):
                value_reason = 'Native features.action requires a declared floating dtype and one-dimensional width 1–64'
                wanted = set()
            for path in {rows[i][2] for i in wanted}:
                try:
                    pf, _ = intake.parquet(path, ('action',), 'dataset_parquet', action=True)
                    physical_wanted = {rows[i][3]: i for i in wanted if rows[i][2] == path}
                    row = 0
                    for batch in pf.iter_batches(batch_size=256, columns=['action']):
                        for value in batch.column(0).to_pylist():
                            if row in physical_wanted:
                                if (not isinstance(value, list) or len(value) != action_width or
                                        any(type(v) is not float or not math.isfinite(v) for v in value)):
                                    value_reason = 'Native action must be a finite floating vector matching features.action.shape; no field alias is guessed'
                                else:
                                    actions[physical_wanted[row]] = value
                            row += 1
                except _Unknown as exc:
                    value_reason = str(exc)
            requested_cells = sum(len(actions.get(min(max(anchor_index + d, start), end - 1), []))
                                  for start, end in (declared_span, corrected_span) if end > start
                                  for d in delta_indices)
            if requested_cells > 4096:
                raise InputError('Declared plus corrected action values exceed 4096 cells; reduce delta_indices')
            def window(span):
                start, end = span
                if end <= start:
                    return {'status': 'unknown', 'reason': 'Declared native episode range is empty or reversed'}
                indices = [min(max(anchor_index + d, start), end - 1) for d in delta_indices]
                pad = [anchor_index + d < start or anchor_index + d >= end for d in delta_indices]
                selected, locators, witnesses = [], [], []
                for offset, (index, is_pad) in enumerate(zip(indices, pad, strict=True)):
                    selected.append(rows[index][0] if 0 <= index < len(rows) else None)
                    locators.append(_locator(rows[index][4], rows[index][3], 'action') if 0 <= index < len(rows) else None)
                    if not is_pad and selected[-1] is not None and selected[-1] != episode_index:
                        witnesses.append({'delta_position': offset, 'delta_index': delta_indices[offset], 'index': index,
                            'actual_episode_index': selected[-1], 'action_is_pad': False,
                            'evidence': _locator(rows[index][4], rows[index][3], 'episode_index')})
                have_values = all(i in actions for i in indices)
                reason = None
                if witnesses:
                    others = sorted({w['actual_episode_index'] for w in witnesses})
                    reason = (f'{len(witnesses)} of {len(indices)} window row(s) belong to episode(s) {others} and are not '
                              f'marked as padding. LeRobot {LEROBOT_REVISION[:7]} clamps the window to the declared span '
                              '(src/lerobot/datasets/dataset_reader.py:298-311, src/lerobot/datasets/utils.py:146-162) '
                              f'without checking row episode_index, so a training sample at this anchor reads another '
                              f"episode's actions as episode {episode_index}'s.")
                return {'status': 'error' if witnesses else 'pass' if have_values else 'unknown', 'reason': reason,
                    'dataset_from_index': start, 'dataset_to_index': end,
                    'indices': indices, 'action_is_pad': pad, 'selected_episode_indices': selected,
                    'values': [actions[i] for i in indices] if have_values else None,
                    'value_status': 'pass' if have_values else 'unknown', 'value_reason': None if have_values else value_reason,
                    'rows': locators, 'cross_episode_witnesses': witnesses}
            result['query'] = {'episode_index': episode_index, 'anchor_index': anchor_index, 'delta_indices': delta_indices,
                               'declared': window(declared_span), 'corrected_what_if': dict(window(corrected_span),
                               basis='Actual verified contiguous row span; metadata-only what-if, not physical repair')}
        query_states = ([result['query'][role]['status'] for role in ('declared', 'corrected_what_if')]
                        if result['query'] is not None else [])
        result['status'] = ('error' if finding_count or 'error' in query_states else
                            'unknown' if 'unknown' in query_states else 'pass')
    except _Unknown as exc:
        footer_conflict = isinstance(exc, (_FooterRowCountConflict, _ParquetEnvelopeConflict))
        retained = [f for f in findings if f.get('status') == 'error'] if footer_conflict else []
        findings[:] = retained
        if footer_conflict:
            findings.append(exc.finding)
        result['status'] = 'error' if footer_conflict else 'unknown'
        patches.clear()
        summaries.clear()
        findings.append({'code': 'episode.coverage_unknown', 'status': 'unknown', 'reason': str(exc)})
        finding_count, patch_count, episode_count = len(findings), 0, 0
    except DatasetLayoutError:
        raise
    except (OSError, ValueError, OverflowError, MemoryError) as exc:
        raise InputError(f'Cannot inspect bounded native episode evidence: {exc}') from exc
    if result['query'] is None:
        # Whole-dataset audits: report definite count/span contradictions the strict engine cannot reach.
        try:
            extra, extra_intake = _consistency(root, max_rows, max_files, short_only=result['status'] != 'unknown')
            coverage['episode_consistency'] = 'checked'
            roles = [a['role'] for a in extra_intake.artifacts.values()]
            coverage['episode_consistency_read'] = {
                'data_files_read': roles.count('dataset_parquet'), 'data_rows_read': extra_intake.data_rows,
                'metadata_files_read': roles.count('episode_metadata'), 'metadata_rows_read': extra_intake.meta_rows}
        except _Unknown as exc:
            extra, extra_intake = [], None
            if isinstance(exc, (_FooterRowCountConflict, _ParquetEnvelopeConflict)):
                observed_path = Path(exc.artifact['path'])
                previous = intake.artifacts.get(observed_path)
                if previous is not None and (previous['sha256'], previous['size_bytes']) != (
                        exc.artifact['sha256'], exc.artifact['size_bytes']):
                    raise InputError('Native artifact changed between bounded consistency reads; '
                                     'retry a stable snapshot') from exc
                intake.artifacts.setdefault(observed_path, exc.artifact)
                if not any(f.get('code') == exc.finding['code'] and f.get('evidence') == exc.finding['evidence']
                           for f in findings):
                    extra = [exc.finding]
            coverage['episode_consistency'] = f'not checked: {exc}'
        except DatasetLayoutError:
            raise
        except (OSError, ValueError, OverflowError, MemoryError) as exc:
            raise InputError(f'Cannot inspect bounded native episode evidence: {exc}') from exc
        if extra_intake is not None:
            extra_intake.stable()
            # Carry only artifacts that findings cite; a stray-file dataset can hold thousands of shards.
            cited = {e.get('path') for item in extra for e in item.get('evidence', [])}
            intake.artifacts.update({k: v for k, v in extra_intake.artifacts.items()
                                     if k not in intake.artifacts and v['path'] in cited})
        if any(item['status'] == 'error' for item in extra):
            result['status'] = 'error'
        elif result['status'] == 'pass' and any(item['status'] == 'warning' for item in extra):
            result['status'] = 'warning'
        findings.extend(extra)
        finding_count += len(extra)
        # Declared video features against episode columns, files and real frame counts (MP4 headers only).
        from .video_index import check_video_index

        context = video_context or {}
        try:
            video, video_coverage = check_video_index(root, listing=context.get('listing'),
                headers=context.get('headers'), max_rows=max_rows, max_files=max_files, video_headers=video_headers)
        except (OSError, ValueError, OverflowError, MemoryError) as exc:
            # A separately refused header must not discard verified row/episode contradictions.
            video = []
            video_coverage = {'video_index': 'not checked: video evidence refused',
                'video_index_details': {'status': 'unknown', 'reason': str(exc),
                    'finding_counts': {'error': 0, 'warning': 0, 'unknown': 0, 'pass': 0}}}
        coverage.update(video_coverage)
        if any(item['status'] == 'error' for item in video):
            result['status'] = 'error'
        details = video_coverage.get('video_index_details', {})
        if details.get('status') in ('partial', 'unknown'):
            video.append({'code': 'video.coverage_unknown', 'status': 'unknown',
                'message': 'Some declared video files or metadata were not checked.',
                'reason': details.get('reason', video_coverage.get('video_index')),
                'evidence': []})
            if result['status'] != 'error':
                result['status'] = 'unknown'
        findings.extend(video)
        finding_count += len(video)
    intake.stable()
    coverage.update(expanded_bytes_checked=intake.expanded, values_checked=intake.values)
    result['artifacts'] = list(intake.artifacts.values())
    result['counts'] = {'findings': finding_count, 'metadata_patches': patch_count, 'episodes': episode_count}
    result['findings_truncated'], result['metadata_patches_truncated'] = finding_count > 16, patch_count > 16
    result['episodes_truncated'] = episode_count > 16
    result['findings'], result['metadata_patches'] = findings[:16], patches[:16]
    bound_artifacts(result, str(root), 192 * 1024)
    if len(json.dumps(result, allow_nan=False).encode()) > 192 * 1024:
        raise InputError('Window response exceeds bounded transport budget')
    return result


def _cited_paths(value, found=None):
    """Every artifact path a report cites outside its artifact list (findings, query row locators, ...)."""
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'path' and isinstance(item, str):
                found.add(item)
            elif key != 'artifacts':
                _cited_paths(item, found)
    elif isinstance(value, list):
        for item in value:
            _cited_paths(item, found)
    return found


def bound_artifacts(result, root_prefix, limit):
    """Keep a report under `limit` bytes for datasets with many healthy shards.

    Cited artifacts (findings and query row locators) are always kept, plus a bounded prefix. The full list is
    pinned by a digest over root-relative paths, so it is reproducible from the same dataset at any location
    (or from the same pinned Hub commit).
    """
    if result.get('artifacts_truncated') or len(json.dumps(result, allow_nan=False).encode()) <= limit:
        return result
    every = result['artifacts']
    cited = _cited_paths({k: v for k, v in result.items() if k != 'artifacts'})
    kept = [a for a in every if a['path'] in cited][:256]
    kept += [a for a in every if a['path'] not in cited][:max(0, 96 - len(kept))]
    prefix = root_prefix.rstrip('/') + '/'
    manifest = [dict(a, path=a['path'][len(prefix):] if a['path'].startswith(prefix) else a['path']) for a in every]
    result.update(artifacts=kept, artifacts_truncated=True, artifact_count=len(every),
                  artifacts_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  artifacts_manifest_basis='sha256 of the JSON list of every artifact with root-relative paths, sorted keys')
    return result
